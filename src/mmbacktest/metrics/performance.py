"""Performance and market-making metrics.

Three groups, reported together because any one of them alone is misleading
for a market-making strategy.

Absolute performance
--------------------
Sharpe, drawdown, return. These are the metrics a non-specialist reader
expects, but they are not sufficient here: a market maker with a runaway
inventory can post a fine Sharpe during a trending period purely because the
position happened to be on the right side. They describe the outcome, not the
behaviour that produced it.

Market-making specific
----------------------
Spread capture, fill rate, inventory control, adverse selection. These
describe whether the strategy is actually market making. A maker that never
fills has perfect inventory control and zero adverse selection and is doing
nothing; a maker that fills constantly but loses on every markout is being
picked off. Neither shows up clearly in Sharpe.

PnL attribution
---------------
Total PnL split into spread capture, inventory mark-to-market, and fees.
This is the group that answers "where did the money come from", and it is
what distinguishes a market maker earning the spread from one that is
accidentally running a directional position. A strategy whose PnL is mostly
inventory drift is not doing what it claims to be doing, regardless of how
good the headline number looks.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


SECONDS_PER_YEAR = 365.25 * 24 * 3600


@dataclass
class PerformanceMetrics:
    """Full metric set for one run."""

    # Absolute
    total_pnl: float
    annualised_return_on_capital: float
    annualised_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    calmar_ratio: float

    # Market making
    n_fills: int
    fills_per_hour: float
    volume_traded: float
    spread_capture_bps: float
    fill_rate: float
    median_queue_wait_seconds: float

    # Inventory
    mean_abs_inventory: float
    inventory_variance: float
    max_abs_inventory: float
    inventory_turnover: float
    pct_time_at_bound: float

    # Adverse selection
    adverse_selection_bps: float
    markout_bps: dict

    # Attribution
    pnl_spread_capture: float
    pnl_inventory: float
    pnl_fees: float

    # Context
    duration_hours: float
    strategy: str = ""
    annualisation_is_reliable: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """One-screen summary, the format used in progress reports."""
        lines = [
            f"strategy                  {self.strategy}",
            f"duration                  {self.duration_hours:.1f} h",
        ]
        if not self.annualisation_is_reliable:
            lines.append(
                "  NOTE: sample shorter than 24h. Annualised return, Sharpe,"
            )
            lines.append(
                "  Sortino and Calmar are scaled from a short window and are"
            )
            lines.append(
                "  not statistically meaningful. Use the level metrics below."
            )
        lines += [
            "",
            "-- absolute --------------------------------------------",
            f"total PnL                 {self.total_pnl:>14,.2f}",
            f"annualised return         {self.annualised_return_on_capital:>13.2%}",
            f"annualised volatility     {self.annualised_volatility:>13.2%}",
            f"Sharpe                    {self.sharpe_ratio:>14.3f}",
            f"Sortino                   {self.sortino_ratio:>14.3f}",
            f"max drawdown              {self.max_drawdown:>14,.2f}"
            f"  ({self.max_drawdown_pct:.2%})",
            f"Calmar                    {self.calmar_ratio:>14.3f}",
            "",
            "-- market making ---------------------------------------",
            f"fills                     {self.n_fills:>14,}",
            f"fills / hour              {self.fills_per_hour:>14.1f}",
            f"volume traded             {self.volume_traded:>14,.2f}",
            f"spread capture            {self.spread_capture_bps:>14.3f} bps",
            f"quote fill rate           {self.fill_rate:>14.2%}",
            f"median queue wait         {self.median_queue_wait_seconds:>14.3f} s",
            "",
            "-- inventory -------------------------------------------",
            f"mean |inventory|          {self.mean_abs_inventory:>14.3f}",
            f"inventory variance        {self.inventory_variance:>14.3f}",
            f"max |inventory|           {self.max_abs_inventory:>14.3f}",
            f"turnover                  {self.inventory_turnover:>14.2f}",
            f"time at bound             {self.pct_time_at_bound:>13.2%}",
            "",
            "-- adverse selection -----------------------------------",
            f"adverse selection         {self.adverse_selection_bps:>14.3f} bps",
        ]
        for h, v in sorted(self.markout_bps.items(), key=_horizon_sort_key):
            lines.append(f"  markout {h:>7}        {v:>14.3f} bps")
        lines += [
            "",
            "-- attribution -----------------------------------------",
            f"spread capture            {self.pnl_spread_capture:>14,.2f}",
            f"inventory MTM             {self.pnl_inventory:>14,.2f}",
            f"fees                      {self.pnl_fees:>14,.2f}",
        ]
        return "\n".join(lines)


def _horizon_sort_key(item: tuple[str, float]) -> int:
    """Sort markout horizons numerically, not lexically."""
    label = item[0]
    try:
        return int(label.replace("ms", ""))
    except ValueError:
        return 0


def compute_metrics(
    timeline: pd.DataFrame,
    fills: pd.DataFrame,
    fill_stats: dict,
    q_max: float,
    markout_horizons_ms: tuple[int, ...] = (100, 1000, 5000, 30000),
    strategy: str = "",
) -> PerformanceMetrics:
    """Compute the full metric set from a run's timeline and fills."""
    if timeline.empty:
        return _empty_metrics(strategy)

    tl = timeline.dropna(subset=["equity", "mid"]).reset_index(drop=True)
    if len(tl) < 2:
        return _empty_metrics(strategy)

    duration_s = (tl["ts"].iloc[-1] - tl["ts"].iloc[0]).total_seconds()
    duration_h = duration_s / 3600.0

    # ---------------------------------------------------------------
    # Absolute performance
    # ---------------------------------------------------------------

    equity = tl["equity"].to_numpy(dtype=float)
    total_pnl = float(equity[-1] - equity[0])

    # Capital base. A market maker does not deploy notional the way a
    # directional strategy does, so return is expressed against the capital
    # required to carry the maximum position: q_max contracts at the average
    # price. This is the honest denominator -- it is the capital that must be
    # committed for the strategy to run at its configured limit.
    avg_price = float(tl["mid"].mean())
    capital_base = max(q_max * avg_price, 1.0)

    pnl_increments = np.diff(equity)
    dt_s = duration_s / max(len(equity) - 1, 1)
    periods_per_year = SECONDS_PER_YEAR / max(dt_s, 1e-9)

    ann_return = (total_pnl / capital_base) * (SECONDS_PER_YEAR / max(duration_s, 1e-9))

    step_vol = float(np.std(pnl_increments, ddof=1)) if len(pnl_increments) > 1 else 0.0
    ann_vol = step_vol * np.sqrt(periods_per_year) / capital_base

    sharpe = float(ann_return / ann_vol) if ann_vol > 1e-12 else 0.0

    downside = pnl_increments[pnl_increments < 0]
    down_vol = (
        float(np.std(downside, ddof=1)) * np.sqrt(periods_per_year) / capital_base
        if len(downside) > 1 else 0.0
    )
    sortino = float(ann_return / down_vol) if down_vol > 1e-12 else 0.0

    running_max = np.maximum.accumulate(equity)
    drawdowns = equity - running_max
    max_dd = float(-drawdowns.min()) if len(drawdowns) else 0.0
    max_dd_pct = max_dd / capital_base
    calmar = float(ann_return / max_dd_pct) if max_dd_pct > 1e-12 else 0.0

    # ---------------------------------------------------------------
    # Inventory
    # ---------------------------------------------------------------

    inv = tl["inventory"].to_numpy(dtype=float)
    mean_abs_inv = float(np.mean(np.abs(inv)))
    inv_var = float(np.var(inv, ddof=1)) if len(inv) > 1 else 0.0
    max_abs_inv = float(np.max(np.abs(inv))) if len(inv) else 0.0
    at_bound = float(np.mean(np.abs(inv) >= q_max * 0.999)) if q_max > 0 else 0.0

    # ---------------------------------------------------------------
    # Fills and market-making behaviour
    # ---------------------------------------------------------------

    n_fills = len(fills)
    volume = float(fills["size"].sum()) if n_fills else 0.0
    fills_per_hour = n_fills / max(duration_h, 1e-9)
    turnover = volume / max(q_max, 1e-9)

    median_wait = (
        float(fills["queue_wait_seconds"].median()) if n_fills else 0.0
    )

    # Fraction of placed quotes that received at least one fill.
    #
    # The obvious definition -- fill events divided by placements -- is wrong
    # and can exceed 100%, because a single resting order can be partially
    # filled several times while quote persistence keeps it alive across many
    # decision ticks. Counting distinct filled orders instead keeps the
    # quantity bounded and answers the question actually being asked: of the
    # quotes we put in the book, how many ever traded?
    n_placed = float(fill_stats.get("n_placed", 0.0))
    if n_fills and "order_id" in fills.columns and n_placed > 0:
        quote_fill_rate = float(fills["order_id"].nunique() / n_placed)
    else:
        quote_fill_rate = 0.0

    # Spread capture: how far inside the mid each fill was, signed so that a
    # buy below the mid is positive. This is the gross edge per unit traded,
    # before any adverse price move.
    if n_fills:
        direction = np.where(fills["is_buy"].to_numpy(), 1.0, -1.0)
        edge = direction * (fills["mid_at_fill"].to_numpy(dtype=float)
                            - fills["price"].to_numpy(dtype=float))
        notional = (fills["price"].to_numpy(dtype=float)
                    * fills["size"].to_numpy(dtype=float))
        spread_capture_bps = float(
            np.sum(edge * fills["size"].to_numpy(dtype=float))
            / max(np.sum(notional), 1e-9) * 10_000.0
        )
        pnl_spread_capture = float(np.sum(edge * fills["size"].to_numpy(dtype=float)))
    else:
        spread_capture_bps = 0.0
        pnl_spread_capture = 0.0

    # ---------------------------------------------------------------
    # Adverse selection
    # ---------------------------------------------------------------

    markout_bps: dict[str, float] = {}
    adverse_bps = 0.0
    for h in markout_horizons_ms:
        col = f"markout_{h}ms"
        if n_fills and col in fills.columns:
            m = fills[col].to_numpy(dtype=float)
            sizes = fills["size"].to_numpy(dtype=float)
            prices = fills["price"].to_numpy(dtype=float)
            valid = np.isfinite(m)
            if valid.any():
                bps = float(
                    np.sum(m[valid] * sizes[valid])
                    / max(np.sum(prices[valid] * sizes[valid]), 1e-9) * 10_000.0
                )
                markout_bps[f"{h}ms"] = bps
                # The longest horizon is taken as the adverse-selection
                # measure: short horizons are dominated by bid-ask bounce,
                # and it is the persistent drift that represents being
                # picked off by informed flow.
                if h == max(markout_horizons_ms):
                    adverse_bps = -bps

    # ---------------------------------------------------------------
    # Attribution
    # ---------------------------------------------------------------

    fees = float(tl["fees_paid"].iloc[-1]) if "fees_paid" in tl else 0.0
    # Whatever is not spread capture or fees is inventory mark-to-market.
    pnl_inventory = total_pnl - pnl_spread_capture + fees

    return PerformanceMetrics(
        total_pnl=total_pnl,
        annualised_return_on_capital=float(ann_return),
        annualised_volatility=float(ann_vol),
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        calmar_ratio=calmar,
        n_fills=n_fills,
        fills_per_hour=float(fills_per_hour),
        volume_traded=volume,
        spread_capture_bps=spread_capture_bps,
        fill_rate=float(quote_fill_rate),
        median_queue_wait_seconds=median_wait,
        mean_abs_inventory=mean_abs_inv,
        inventory_variance=inv_var,
        max_abs_inventory=max_abs_inv,
        inventory_turnover=float(turnover),
        pct_time_at_bound=at_bound,
        adverse_selection_bps=adverse_bps,
        markout_bps=markout_bps,
        pnl_spread_capture=pnl_spread_capture,
        pnl_inventory=float(pnl_inventory),
        pnl_fees=-fees,
        duration_hours=float(duration_h),
        strategy=strategy,
        # Annualising from a short window is arithmetically fine and
        # statistically meaningless. A 12-minute replay scaled to a year
        # produces Sharpe ratios in the thousands, which say nothing about
        # the strategy and everything about the scaling factor. Flag it so
        # the number is never read as if it were an estimate.
        annualisation_is_reliable=duration_h >= 24.0,
    )


def compare(results: dict[str, PerformanceMetrics]) -> pd.DataFrame:
    """Side-by-side comparison table across strategies.

    This is the artefact the three-way comparison produces: one row per
    metric, one column per strategy, so the reader can see immediately where
    a strategy wins and where it pays for that win.
    """
    rows = [
        ("total PnL", "total_pnl", "{:,.2f}"),
        ("ann. return", "annualised_return_on_capital", "{:.2%}"),
        ("ann. vol", "annualised_volatility", "{:.2%}"),
        ("Sharpe", "sharpe_ratio", "{:.3f}"),
        ("Sortino", "sortino_ratio", "{:.3f}"),
        ("max drawdown", "max_drawdown", "{:,.2f}"),
        ("Calmar", "calmar_ratio", "{:.3f}"),
        ("fills", "n_fills", "{:,}"),
        ("fills/hour", "fills_per_hour", "{:.1f}"),
        ("spread capture (bps)", "spread_capture_bps", "{:.3f}"),
        ("adverse selection (bps)", "adverse_selection_bps", "{:.3f}"),
        ("mean |inventory|", "mean_abs_inventory", "{:.3f}"),
        ("inventory variance", "inventory_variance", "{:.3f}"),
        ("max |inventory|", "max_abs_inventory", "{:.3f}"),
        ("time at bound", "pct_time_at_bound", "{:.2%}"),
        ("median queue wait (s)", "median_queue_wait_seconds", "{:.3f}"),
        ("PnL: spread", "pnl_spread_capture", "{:,.2f}"),
        ("PnL: inventory", "pnl_inventory", "{:,.2f}"),
        ("PnL: fees", "pnl_fees", "{:,.2f}"),
    ]

    data: dict[str, list[str]] = {}
    for name, m in results.items():
        col = []
        for _, attr, fmt in rows:
            val = getattr(m, attr)
            col.append(fmt.format(val))
        data[name] = col

    return pd.DataFrame(data, index=[label for label, _, _ in rows])


def _empty_metrics(strategy: str) -> PerformanceMetrics:
    return PerformanceMetrics(
        total_pnl=0.0, annualised_return_on_capital=0.0, annualised_volatility=0.0,
        sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0, max_drawdown_pct=0.0,
        calmar_ratio=0.0, n_fills=0, fills_per_hour=0.0, volume_traded=0.0,
        spread_capture_bps=0.0, fill_rate=0.0, median_queue_wait_seconds=0.0,
        mean_abs_inventory=0.0, inventory_variance=0.0, max_abs_inventory=0.0,
        inventory_turnover=0.0, pct_time_at_bound=0.0, adverse_selection_bps=0.0,
        markout_bps={}, pnl_spread_capture=0.0, pnl_inventory=0.0, pnl_fees=0.0,
        duration_hours=0.0, strategy=strategy, annualisation_is_reliable=False,
    )
