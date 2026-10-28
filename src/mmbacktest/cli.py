"""Command-line entry point.

Two modes:

    run       one strategy, one fee assumption
    compare   all three strategies on identical data, plus fee sensitivity

`compare` is the one that matters for the research question. It replays the
same event stream through every strategy, so a difference in results is
attributable to the strategy and nothing else: same book, same trades, same
calibration, same fill model, same fees.

Data source is either the collector's ClickHouse instance or the synthetic
generator. Synthetic is for development only and the CLI says so on every
run, so a synthetic number never quietly ends up in a report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .calibration.intensity import (
    IntensityEstimate,
    IntensitySchedule,
    attach_mid_to_trades,
    estimate_by_session,
    session_exposure_seconds,
)
from .config import Config
from .data.book import OrderBook, _is_bid
from .data.clickhouse_client import MBODataStore
from .data.synthetic import generate_mbo_events, generate_trades
from .metrics.performance import annualisation_caveat  # noqa: E402
from .metrics.performance import compare as compare_metrics, compute_metrics
from .sim.engine import BacktestEngine
from .sim.fills import compute_markouts
from .strategy.base import make_quoter

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Data acquisition
# --------------------------------------------------------------------------

def load_data(
    cfg: Config, synthetic: bool, n_events: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Return (book events, trades, provenance label)."""
    if synthetic:
        logger.warning(
            "SYNTHETIC DATA. For development only; not valid for reported results."
        )
        events = generate_mbo_events(n_events=n_events, seed=cfg.seed)
        trades = generate_trades(events, trade_rate=0.02, seed=cfg.seed)
        # No funding on synthetic data: there is no venue to charge it.
        return events, trades, pd.DataFrame(), "synthetic"

    store = MBODataStore(cfg.clickhouse)
    if not store.ping():
        raise SystemExit(
            "cannot reach ClickHouse at "
            f"{cfg.clickhouse.host}:{cfg.clickhouse.port}. "
            "Is Tailscale up? Use --synthetic to develop offline."
        )

    epoch = store.resolve_epoch(cfg.data)
    logger.info("using %s", epoch)

    ok, detail = store.integrity_ok(cfg.data.symbol, epoch.epoch)
    if not ok:
        logger.warning("data integrity check failed for epoch %d: %s",
                       epoch.epoch, detail)
        logger.warning("proceeding, but the replay may be built on a drifted book")
    else:
        logger.info("data integrity (epoch %d): %s", epoch.epoch, detail)

    # Trades must span exactly the same window as the book events. Falling
    # back to the epoch bounds when the config narrows the window loads trades
    # with no book behind them: intensity calibration then matches them
    # against a mid series that does not cover their timestamps, and the
    # replay sees trades arriving after the book has stopped updating. It
    # shows up as sessions reporting n=0 or n=1 samples for hours that are
    # not in the requested window at all.
    window_start = cfg.data.start if cfg.data.start is not None else epoch.start
    window_end = cfg.data.end if cfg.data.end is not None else epoch.end

    events = store.load_book_events(
        cfg.data.symbol, epoch.epoch, window_start, window_end
    )
    trades = store.load_trades(cfg.data.symbol, window_start, window_end)

    # Funding settlements bracketing the window. Widened by one interval so a
    # settlement sitting on a window edge is present rather than silently
    # dropped; the engine only charges those the replay clock actually passes.
    funding = store.load_funding(
        cfg.data.symbol,
        window_start - timedelta(hours=8),
        window_end + timedelta(hours=8),
    )
    if funding.empty:
        logger.warning(
            "no funding data for this window -- run scripts/backfill_deriv.py. "
            "Results will omit funding, a leg whose size scales with inventory "
            "and whose sign depends on the regime."
        )
    store.close()

    label = f"{cfg.data.symbol}@epoch{epoch.epoch}"
    return events, trades, funding, label


def _mid_series(
    events: pd.DataFrame,
    tick_size: float | None,
    sample_from: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Reconstruct the book and sample its mid on a 1 s grid.

    One second is enough resolution to locate a trade and far cheaper than
    storing every event. `sample_from` discards samples before a point while
    still applying every event, which is how a segment that does not begin at
    a snapshot is warmed up: its book starts empty and fills in from the
    incremental stream, and its touch is meaningless until it has.
    """
    book = OrderBook(tick_size=tick_size)
    rows = []
    prev_action = ""
    ts_col = events["ts_exch"].to_numpy()
    oid = events["order_id"].to_numpy()
    side = events["side"].to_numpy()
    action = events["action"].to_numpy()
    price = events["price"].to_numpy(dtype=float)
    amount = events["amount"].to_numpy(dtype=float)
    seq = events["seq"].to_numpy()

    next_sample = None
    step = pd.Timedelta(seconds=1)
    for i in range(len(events)):
        act = str(action[i]).lower()
        if act == "snapshot" and prev_action != "snapshot":
            book.clear()
        prev_action = act
        book.apply(int(oid[i]), side[i], act, float(price[i]),
                   float(amount[i]), int(seq[i]))

        mid = book.mid
        if mid is None:
            continue
        ts = pd.Timestamp(ts_col[i])
        if sample_from is not None and ts < sample_from:
            continue
        if next_sample is None:
            next_sample = ts
        if ts >= next_sample:
            rows.append({"ts": ts, "mid": mid})
            while next_sample <= ts:
                next_sample += step
    return pd.DataFrame(rows)


def _fit(
    cfg: Config,
    with_mid: pd.DataFrame,
    coverage: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[IntensityEstimate | IntensitySchedule, dict]:
    """Fit pooled and per-session intensity; return what the engine should use.

    Returns the pooled estimate, or a session schedule when
    `calibration.session_intensity` is set, plus metadata for the output JSON.
    """
    estimates = estimate_by_session(with_mid, cfg.calibration, coverage)
    exposure = session_exposure_seconds(coverage, cfg.calibration.session_boundaries_utc)
    for _label, est in sorted(estimates.items()):
        logger.info("  %s", est)
    pooled = estimates["all"]
    if not pooled.is_usable:
        logger.warning(
            "pooled intensity fit is not usable (kappa=%.6f, R2=%.3f); "
            "quotes will fall back to the volatility term only",
            pooled.kappa, pooled.r_squared,
        )
    meta = {"kappa": pooled.kappa, "A": pooled.A, "r_squared": pooled.r_squared,
            "n_trades": pooled.n_trades, "usable": pooled.is_usable,
            "session_intensity": cfg.calibration.session_intensity,
            "sessions": {
                lb: {"kappa": e.kappa, "A": e.A, "r_squared": e.r_squared,
                     "n_trades": e.n_trades, "usable": e.is_usable,
                     "exposure_hours": round(exposure.get(lb, 0.0) / 3600, 3)}
                for lb, e in sorted(estimates.items()) if lb != "all"
            }}
    if not cfg.calibration.session_intensity:
        return pooled, meta
    if not cfg.calibration.stratify_by_session:
        raise SystemExit("session_intensity needs stratify_by_session")
    schedule = IntensitySchedule(estimates, cfg.calibration.session_boundaries_utc)
    logger.info("intensity: %s", schedule)
    meta["fallback_sessions"] = schedule.fallback_sessions
    return schedule, meta


def calibrate_intensity(
    cfg: Config, events: pd.DataFrame, trades: pd.DataFrame
) -> tuple[IntensityEstimate | IntensitySchedule, dict]:
    """Fit lambda(delta) on the replay window itself (in-sample).

    The mid series needed to measure trade distance is built by a first pass
    over the book. That pass is not wasted: it is the same reconstruction the
    replay will do, so its cost is one extra traversal rather than a separate
    data product.

    In-sample means kappa and A are fitted on the same trades the replay then
    quotes against. That is identical across strategies, so orderings between
    them stay fair, but it is a lookahead; see calibrate_walk_forward.
    """
    logger.info("building mid series for intensity calibration (in-sample)")
    mid_series = _mid_series(events, cfg.strategy.tick_size)
    if mid_series.empty or trades.empty:
        logger.warning("insufficient data to calibrate intensity; using fallback")
        est = IntensityEstimate(A=0.01, kappa=0.01, n_trades=0,
                                n_buckets_used=0, r_squared=0.0)
        return est, {"kappa": est.kappa, "A": est.A, "r_squared": 0.0,
                     "n_trades": 0, "usable": False}
    ts = pd.to_datetime(events["ts_exch"])
    return _fit(cfg, attach_mid_to_trades(trades, mid_series),
                [(ts.min(), ts.max())])


def calibrate_walk_forward(
    cfg: Config,
    replay_start: pd.Timestamp,
    hours: float,
    min_coverage: float = 0.5,
) -> tuple[IntensityEstimate, dict]:
    """Fit lambda(delta) on the `hours` before the replay starts, and nothing after.

    The window [replay_start - hours, replay_start) usually spans several
    collector epochs. Each is reconstructed on its own -- a book cannot be
    replayed across an epoch boundary without ghost orders -- and trades are
    matched to the mid of their own segment only, so a trade near a gap is
    never measured against a stale mid from the previous epoch. Compromised
    epochs are skipped. A segment that begins mid-epoch has no snapshot, so
    its first `warmup_seconds` are applied to the book but not sampled.

    Raises SystemExit if the window has less than `min_coverage` of its hours
    covered. It deliberately does not fall back to in-sample: a comparison
    that silently mixed the two would be worse than one with fewer epochs.
    """
    replay_start = pd.Timestamp(replay_start)
    w1 = replay_start
    w0 = replay_start - pd.Timedelta(hours=hours)
    warm = pd.Timedelta(seconds=cfg.simulation.warmup_seconds)

    store = MBODataStore(cfg.clickhouse)
    segments = []
    for e in store.list_epochs(cfg.data.symbol, 0.0):
        s0, s1 = max(pd.Timestamp(e.start), w0), min(pd.Timestamp(e.end), w1)
        if s1 > s0:
            segments.append((e, s0, s1))
    segments.sort(key=lambda x: x[1])

    parts, coverage, covered = [], [], 0.0
    for e, s0, s1 in segments:
        at_snapshot = s0 <= pd.Timestamp(e.start)
        sample_from = s0 if at_snapshot else s0 + warm
        if sample_from >= s1:
            continue
        ev = store.load_book_events(cfg.data.symbol, e.epoch, s0, s1)
        tr = store.load_trades(cfg.data.symbol, sample_from, s1)
        # Strictly before the replay: load_trades is inclusive at the end.
        tr = tr[pd.to_datetime(tr["ts_exch"]) < w1]
        if ev.empty or tr.empty:
            continue
        mid = _mid_series(ev, cfg.strategy.tick_size, sample_from=sample_from)
        if mid.empty:
            continue
        parts.append(attach_mid_to_trades(tr, mid))
        coverage.append((sample_from, s1))
        covered += (s1 - sample_from).total_seconds() / 3600.0
    store.close()

    meta = {
        "mode": "walk_forward",
        "hours": hours,
        "window_start": str(w0),
        "window_end": str(w1),
        "covered_hours": round(covered, 3),
        "n_segments": len(parts),
    }
    if not parts or covered < min_coverage * hours:
        raise SystemExit(
            f"walk-forward calibration: only {covered:.1f} h of the {hours:g} h "
            f"before {w1} are covered ({len(parts)} usable segments); need at "
            f"least {min_coverage:.0%}. Not falling back to in-sample."
        )

    logger.info(
        "walk-forward calibration: %s -> %s, %.1f h covered in %d segment(s)",
        w0, w1, covered, len(parts),
    )
    est, fit_meta = _fit(cfg, pd.concat(parts, ignore_index=True), coverage)
    meta.update(fit_meta)
    return est, meta


def calibrate(
    cfg: Config, args, events: pd.DataFrame, trades: pd.DataFrame
) -> tuple[IntensityEstimate, dict]:
    """Dispatch on --calibration and return the estimate with its provenance."""
    mode = getattr(args, "calibration", "in_sample")
    if mode == "walk_forward":
        if args.synthetic:
            raise SystemExit("walk-forward calibration needs real data")
        start = pd.to_datetime(events["ts_exch"]).min()
        return calibrate_walk_forward(cfg, start, args.calib_hours)
    est, meta = calibrate_intensity(cfg, events, trades)
    return est, {"mode": "in_sample", **meta}


# --------------------------------------------------------------------------
# Run modes
# --------------------------------------------------------------------------

def run_one(
    cfg: Config,
    events: pd.DataFrame,
    trades: pd.DataFrame,
    intensity: IntensityEstimate,
    strategy_name: str,
    funding: pd.DataFrame | None = None,
):
    """Replay one strategy and return its metrics."""
    run_cfg = Config.from_dict(cfg.to_dict())
    run_cfg.strategy.name = strategy_name

    quoter = make_quoter(run_cfg.strategy)
    engine = BacktestEngine(run_cfg, quoter, intensity)
    result = engine.run(events, trades, funding=funding)

    fills_df = compute_markouts(
        result.fills,
        result.timeline[["ts", "mid"]],
        run_cfg.simulation.markout_horizons_ms,
    )
    metrics = compute_metrics(
        result.timeline,
        fills_df,
        result.fill_stats,
        q_max=run_cfg.strategy.q_max,
        markout_horizons_ms=run_cfg.simulation.markout_horizons_ms,
        strategy=strategy_name,
        funding_stats=result.funding_stats,
        quote_stats=result.quote_stats,
    )
    return metrics, result


def cmd_run(args, cfg: Config) -> int:
    events, trades, funding, provenance = load_data(
        cfg, args.synthetic, args.n_events
    )
    if args.no_funding:
        logger.info("--no-funding: funding excluded from this run")
        funding = pd.DataFrame()
    intensity, calib_meta = calibrate(cfg, args, events, trades)

    metrics, result = run_one(
        cfg, events, trades, intensity, cfg.strategy.name, funding=funding
    )

    print()
    print(f"data: {provenance}")
    print()
    print(metrics.summary())

    if args.out:
        _write_outputs(Path(args.out), {cfg.strategy.name: metrics}, result,
                       provenance, calibration=calib_meta)
    return 0


def cmd_compare(args, cfg: Config) -> int:
    events, trades, funding, provenance = load_data(
        cfg, args.synthetic, args.n_events
    )
    if args.no_funding:
        logger.info("--no-funding: funding excluded from this run")
        funding = pd.DataFrame()
    intensity, calib_meta = calibrate(cfg, args, events, trades)

    strategies = ["symmetric", "avellaneda_stoikov", "glft"]
    fee_levels = [float(f) for f in args.fees.split(",")]

    all_metrics: dict[str, object] = {}
    last_result = None

    for fee in fee_levels:
        for name in strategies:
            run_cfg = Config.from_dict(cfg.to_dict())
            run_cfg.simulation.maker_fee_bps = fee
            label = f"{name} @ {fee:+.1f}bp"
            logger.info("running %s", label)

            metrics, result = run_one(
                run_cfg, events, trades, intensity, name, funding=funding
            )
            metrics.strategy = label
            all_metrics[label] = metrics
            last_result = result

    table = compare_metrics(all_metrics)

    print()
    print(f"data: {provenance}")
    pooled = getattr(intensity, "pooled", intensity)
    print(f"intensity ({calib_meta.get('mode')}): A={pooled.A:.6f} "
          f"kappa={pooled.kappa:.6f} R2={pooled.r_squared:.3f}"
          + (f"  [{intensity}]" if pooled is not intensity else ""))
    print()
    print(table.to_string())

    caveat = annualisation_caveat(all_metrics)
    if caveat:
        print()
        print(caveat)

    weak_inv = [
        (m.strategy, m.pnl_inventory, m.pnl_inventory_tstat)
        for m in all_metrics.values()
        if abs(m.pnl_inventory) > abs(m.pnl_spread_capture)
        and abs(m.pnl_inventory_tstat) < 2.0
    ]
    if weak_inv:
        print()
        print("[!] Inventory PnL dominates total PnL but is not statistically")
        print("    distinguishable from a random walk (|t| < 2). This is not a")
        print("    result -- it is the residual of a position held through noise:")
        for label, pnl, t in weak_inv:
            print(f"      {label:32s} inventory {pnl:>+10,.2f}  t = {t:+.2f}")

    worst_new_level = max(
        (m.frac_fills_at_new_level for m in all_metrics.values()), default=0.0
    )
    if worst_new_level > 0.5:
        print()
        print(
            f"[!] Up to {worst_new_level:.0%} of fills landed on price levels the "
            "strategy created rather than joined. Those fills never appear\n"
            "    on the historical tape, so that PnL assumes our quote would not "
            "have changed the flow that traded against it."
        )

    if args.out:
        _write_outputs(Path(args.out), all_metrics, last_result, provenance, table,
                       calibration=calib_meta)
    return 0


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def _write_outputs(out_dir: Path, metrics: dict, result, provenance: str,
                   table: pd.DataFrame | None = None,
                   calibration: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_provenance": provenance,
        "calibration": calibration or {},
        "metrics": {k: v.to_dict() for k, v in metrics.items()},
    }
    (out_dir / f"metrics_{stamp}.json").write_text(json.dumps(payload, indent=2))

    if table is not None:
        table.to_csv(out_dir / f"comparison_{stamp}.csv")

    if result is not None and not result.timeline.empty:
        result.timeline.to_parquet(out_dir / f"timeline_{stamp}.parquet")

    logger.info("wrote outputs to %s", out_dir)


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mmbacktest",
        description="Market-making backtest on Bitfinex MBO data",
    )
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--symbol", help="override the configured symbol")
    p.add_argument("--epoch", type=int, help="override the configured epoch")
    p.add_argument("--synthetic", action="store_true",
                   help="use generated data instead of ClickHouse (development only)")
    p.add_argument("--n-events", type=int, default=200_000,
                   help="synthetic event count")
    p.add_argument("--gamma", type=float, help="override risk aversion")
    p.add_argument("--q-max", type=float, help="override inventory bound")
    p.add_argument("--maker-fee-bps", type=float,
                   help="override maker fee, negative for a rebate")
    p.add_argument("--no-queue-model", action="store_true",
                   help="disable queue-position fills (for the queue sensitivity study)")
    p.add_argument("--calibration", choices=["in_sample", "walk_forward"],
                   default="in_sample",
                   help="fit kappa/A on the replay window itself (in_sample) or "
                        "only on the hours before it (walk_forward)")
    p.add_argument("--calib-hours", type=float, default=24.0,
                   help="walk-forward look-back; 24 h covers each session once")
    p.add_argument("--session-intensity", action="store_true",
                   help="switch kappa/A at the UTC session boundaries instead "
                        "of quoting on the pooled fit all day")
    p.add_argument("--no-funding", action="store_true",
                   help="leave funding out even where it was backfilled, to "
                        "reproduce results from before the deriv backfill")
    p.add_argument("--out", help="directory for metrics and timeline output")
    p.add_argument("-v", "--verbose", action="store_true")

    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="single strategy")
    r.add_argument("--strategy", default=None,
                   choices=["symmetric", "avellaneda_stoikov", "glft"])

    c = sub.add_parser("compare", help="all strategies, optionally across fee levels")
    c.add_argument("--fees", default="0.0",
                   help="comma-separated maker fees in bps, e.g. '0,8,-1'")

    return p


def apply_overrides(cfg: Config, args) -> Config:
    if args.symbol:
        cfg.data.symbol = args.symbol
    if args.epoch is not None:
        cfg.data.epoch = args.epoch
    if args.gamma is not None:
        cfg.strategy.gamma = args.gamma
    if args.q_max is not None:
        cfg.strategy.q_max = args.q_max
    if args.maker_fee_bps is not None:
        cfg.simulation.maker_fee_bps = args.maker_fee_bps
    if args.no_queue_model:
        cfg.simulation.use_queue_model = False
    if args.session_intensity:
        cfg.calibration.session_intensity = True
    if getattr(args, "strategy", None):
        cfg.strategy.name = args.strategy
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg_path = Path(args.config)
    cfg = Config.load(str(cfg_path)) if cfg_path.exists() else Config()
    cfg = apply_overrides(cfg, args)

    if args.command == "run":
        return cmd_run(args, cfg)
    if args.command == "compare":
        return cmd_compare(args, cfg)
    return 1


if __name__ == "__main__":
    sys.exit(main())
