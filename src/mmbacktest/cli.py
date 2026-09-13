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
from datetime import datetime
from pathlib import Path

import pandas as pd

from .calibration.intensity import (
    IntensityEstimate,
    attach_mid_to_trades,
    estimate_by_session,
)
from .config import Config
from .data.book import OrderBook, _is_bid
from .data.clickhouse_client import MBODataStore
from .data.synthetic import generate_mbo_events, generate_trades
from .metrics.performance import compare as compare_metrics, compute_metrics
from .sim.engine import BacktestEngine
from .sim.fills import compute_markouts
from .strategy.base import make_quoter

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Data acquisition
# --------------------------------------------------------------------------

def load_data(cfg: Config, synthetic: bool, n_events: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (book events, trades, provenance label)."""
    if synthetic:
        logger.warning(
            "SYNTHETIC DATA. For development only; not valid for reported results."
        )
        events = generate_mbo_events(n_events=n_events, seed=cfg.seed)
        trades = generate_trades(events, trade_rate=0.02, seed=cfg.seed)
        return events, trades, "synthetic"

    store = MBODataStore(cfg.clickhouse)
    if not store.ping():
        raise SystemExit(
            "cannot reach ClickHouse at "
            f"{cfg.clickhouse.host}:{cfg.clickhouse.port}. "
            "Is Tailscale up? Use --synthetic to develop offline."
        )

    ok, detail = store.integrity_ok(cfg.data.symbol)
    if not ok:
        logger.warning("data integrity check failed: %s", detail)
        logger.warning("proceeding, but the replay may be built on a drifted book")
    else:
        logger.info("data integrity: %s", detail)

    epoch = store.resolve_epoch(cfg.data)
    logger.info("using %s", epoch)

    events = store.load_book_events(
        cfg.data.symbol, epoch.epoch, cfg.data.start, cfg.data.end
    )
    trades = store.load_trades(cfg.data.symbol, epoch.start, epoch.end)
    store.close()

    label = f"{cfg.data.symbol}@epoch{epoch.epoch}"
    return events, trades, label


def calibrate_intensity(
    cfg: Config, events: pd.DataFrame, trades: pd.DataFrame
) -> IntensityEstimate:
    """Fit lambda(delta) on the replay window.

    The mid series needed to measure trade distance is built by a first pass
    over the book. That pass is not wasted: it is the same reconstruction the
    replay will do, so its cost is one extra traversal rather than a separate
    data product.
    """
    logger.info("building mid series for intensity calibration")
    book = OrderBook(tick_size=cfg.strategy.tick_size)

    rows = []
    prev_action = ""
    ts_col = events["ts_exch"].to_numpy()
    oid = events["order_id"].to_numpy()
    side = events["side"].to_numpy()
    action = events["action"].to_numpy()
    price = events["price"].to_numpy(dtype=float)
    amount = events["amount"].to_numpy(dtype=float)
    seq = events["seq"].to_numpy()

    # Sample the mid on a 1s grid: enough resolution to locate a trade, far
    # cheaper than storing every event.
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
        if next_sample is None:
            next_sample = ts
        if ts >= next_sample:
            rows.append({"ts": ts, "mid": mid})
            while next_sample <= ts:
                next_sample += step

    mid_series = pd.DataFrame(rows)
    if mid_series.empty or trades.empty:
        logger.warning("insufficient data to calibrate intensity; using fallback")
        return IntensityEstimate(A=0.01, kappa=0.01, n_trades=0,
                                 n_buckets_used=0, r_squared=0.0)

    with_mid = attach_mid_to_trades(trades, mid_series)
    estimates = estimate_by_session(with_mid, cfg.calibration)

    for label, est in sorted(estimates.items()):
        logger.info("  %s", est)

    pooled = estimates["all"]
    if not pooled.is_usable:
        logger.warning(
            "pooled intensity fit is not usable (kappa=%.6f, R2=%.3f); "
            "quotes will fall back to the volatility term only",
            pooled.kappa, pooled.r_squared,
        )
    return pooled


# --------------------------------------------------------------------------
# Run modes
# --------------------------------------------------------------------------

def run_one(
    cfg: Config,
    events: pd.DataFrame,
    trades: pd.DataFrame,
    intensity: IntensityEstimate,
    strategy_name: str,
):
    """Replay one strategy and return its metrics."""
    run_cfg = Config.from_dict(cfg.to_dict())
    run_cfg.strategy.name = strategy_name

    quoter = make_quoter(run_cfg.strategy)
    engine = BacktestEngine(run_cfg, quoter, intensity)
    result = engine.run(events, trades)

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
    )
    return metrics, result


def cmd_run(args, cfg: Config) -> int:
    events, trades, provenance = load_data(cfg, args.synthetic, args.n_events)
    intensity = calibrate_intensity(cfg, events, trades)

    metrics, result = run_one(cfg, events, trades, intensity, cfg.strategy.name)

    print()
    print(f"data: {provenance}")
    print()
    print(metrics.summary())

    if args.out:
        _write_outputs(Path(args.out), {cfg.strategy.name: metrics}, result, provenance)
    return 0


def cmd_compare(args, cfg: Config) -> int:
    events, trades, provenance = load_data(cfg, args.synthetic, args.n_events)
    intensity = calibrate_intensity(cfg, events, trades)

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

            metrics, result = run_one(run_cfg, events, trades, intensity, name)
            metrics.strategy = label
            all_metrics[label] = metrics
            last_result = result

    table = compare_metrics(all_metrics)

    print()
    print(f"data: {provenance}")
    print(f"intensity: A={intensity.A:.6f} kappa={intensity.kappa:.6f} "
          f"R2={intensity.r_squared:.3f}")
    print()
    print(table.to_string())

    if args.out:
        _write_outputs(Path(args.out), all_metrics, last_result, provenance, table)
    return 0


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def _write_outputs(out_dir: Path, metrics: dict, result, provenance: str,
                   table: pd.DataFrame | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_provenance": provenance,
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
