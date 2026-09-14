"""Typed configuration for the market-making backtest.

Everything that is a tunable knob lives here rather than being scattered as
magic numbers through the code. A run is fully described by a Config instance,
which makes results reproducible: serialise the config alongside the results
and the run can be repeated exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

import yaml


# --------------------------------------------------------------------------
# Venue price grid
# --------------------------------------------------------------------------

def bitfinex_tick_size(
    price: float,
    sig_digits: int = 5,
    max_decimals: int = 8,
) -> float:
    """Minimum price increment on Bitfinex at a given price level.

    Bitfinex does not use a fixed tick. Prices carry at most 5 significant
    digits (and at most 8 decimals, whichever binds first); anything finer
    submitted over the API is truncated. The tick is therefore the place
    value of the 5th significant digit, which means it *changes* as price
    crosses a power of ten:

        BTC-PERP   76,647     -> 1
        ETH-PERP    2,470.8   -> 0.1
        XAUT-PERP   4,348.9   -> 0.1
        EUR-PERP        1.1610-> 0.0001
        IOT-PERP        0.040598 -> 0.000001

    This matters for BTC specifically: at 76k the tick is 1, but below
    10,000 it becomes 0.1. A hard-coded tick is wrong at some price, and
    wrong in a way that does not announce itself -- quoting off-grid puts
    orders at prices the venue cannot represent, which in a queue-position
    simulator reads as an empty queue and therefore as an instant fill.

    The 8-decimal cap binds only for very low-priced instruments, where the
    5th significant digit would fall past the 8th decimal.
    """
    if not math.isfinite(price) or price <= 0.0:
        raise ValueError(f"tick size undefined for price {price!r}")

    # +1e-12 guards the case where log10 of an exact power of ten comes back
    # a hair under the integer (log10(1000) -> 2.9999999999999996), which
    # would drop the tick by a full decade.
    exponent = math.floor(math.log10(price) + 1e-12)
    tick = 10.0 ** (exponent - (sig_digits - 1))
    return max(tick, 10.0 ** (-max_decimals))


def round_to_tick(price: float, tick: float, mode: str = "nearest") -> float:
    """Snap a price onto the venue's grid.

    `mode` is "down", "up" or "nearest". Quote hygiene rounds outward (bids
    down, asks up) so that rounding never tightens a quote beyond what the
    strategy asked for.

    Naive `floor(price / tick) * tick` is not safe here. Ticks like 0.1 and
    1e-6 are not exactly representable in binary, so the division lands a
    hair below an integer and floor() drops a whole tick: 2470.7 / 0.1 is
    24706.999999999996, which floors to 2470.6. The epsilon absorbs that,
    and the final round() removes the residue from multiplying back up
    (24707 * 0.1 = 2470.7000000000003), so the result is exactly on grid.
    """
    if tick <= 0.0:
        raise ValueError(f"tick must be positive, got {tick!r}")

    scaled = price / tick
    if mode == "down":
        n = math.floor(scaled + 1e-9)
    elif mode == "up":
        n = math.ceil(scaled - 1e-9)
    elif mode == "nearest":
        n = math.floor(scaled + 0.5)
    else:
        raise ValueError(f"unknown rounding mode {mode!r}")

    decimals = max(0, -int(round(math.log10(tick))))
    return round(n * tick, decimals)


def is_on_tick_grid(price: float, tick: float, rel_tol: float = 1e-9) -> bool:
    """Whether a price is representable on the venue's grid.

    Used as an assertion at quote time. An off-grid price is not a rounding
    nuisance: the reconstructed book can have no depth at a level the venue
    cannot quote, so a queue-position fill model reads it as an empty queue
    and fills instantly. That failure is silent, which is why it is checked
    rather than trusted.
    """
    if tick <= 0.0:
        return False
    scaled = price / tick
    return abs(scaled - round(scaled)) <= max(rel_tol * abs(scaled), 1e-6)


# --------------------------------------------------------------------------
# Data source
# --------------------------------------------------------------------------

@dataclass
class ClickHouseConfig:
    """Connection settings for the MBO data store.

    Defaults point at the collector host on the Tailscale network. Override
    via config file or CLI when running from a different machine.
    """

    host: str = "100.76.49.84"
    port: int = 8123
    database: str = "bfx"
    user: str = "default"
    password: str = ""
    connect_timeout: int = 30
    send_receive_timeout: int = 600


@dataclass
class DataConfig:
    """Which slice of the dataset a run consumes.

    `symbol` uses the exchange's own contract code. The three instruments with
    real trade flow on Bitfinex are BTC, ETH and XAUT perpetuals; HYPE-PERP was
    excluded at instrument-selection time for insufficient liquidity (see
    docs/dataset.md).

    `epoch` matters: the collector increments it on every reconnect, and a
    replay that crosses an epoch boundary will contain ghost orders (orders
    whose delete message fell into the gap). All replays are therefore
    constrained to a single epoch.
    """

    symbol: str = "tBTCF0:USTF0"
    epoch: int | None = None          # None -> pick the longest available
    start: datetime | None = None     # None -> epoch start
    end: datetime | None = None       # None -> epoch end
    min_epoch_minutes: float = 120.0  # ignore short fragments
    book_levels: int = 25             # levels to retain when aggregating MBO


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    """Parameters for estimating the inputs the quoters need.

    Intensity: the AS/GLFT family assumes fill intensity decays exponentially
    with quote distance, lambda(delta) = A * exp(-kappa * delta). We estimate A
    and kappa by measuring, for each trade, how far the trade price sat from
    the prevailing mid, then regressing log-counts on distance.

    Volatility: short-horizon realised volatility of the mid, computed on a
    rolling window at the decision frequency.
    """

    # Intensity estimation.
    #
    # max_delta_bps has to cover the distance real trades actually print from
    # the mid. Measured on tBTCF0:USTF0 epoch 27 (226 trades matched against a
    # 100 ms mid series, 2026-09-06 02:00-03:00 UTC), aggressor-signed:
    #
    #     p50   0.94 bps      p90   3.41 bps
    #     p95   4.50 bps      p99   5.32 bps      p99.9  5.54 bps
    #
    # The p50 of 0.94 bps coincides with the median touch half-spread of
    # 0.9375 bps, which is the expected result and a useful check that the
    # measurement is sound: most aggressive flow lifts or hits the touch.
    #
    # This window has now been wrong in both directions. At 20 bps (equity
    # scale, 222 USD) every trade fell in the first bucket and the fit had no
    # slope. The correction to 1.0 bps overshot: it was derived from the
    # synthetic generator's spread, not from the book, and discards 46.9% of
    # real trades -- truncating exactly the tail that carries the decay
    # information kappa is supposed to measure. 6.0 bps covers the observed
    # support with room to spare.
    intensity_n_buckets: int = 12
    intensity_max_delta_bps: float = 6.0

    # Sample-size caveat: BTC-PERP prints ~230 trades/hour in a quiet session,
    # so 12 buckets x 30 samples = 360 cannot be filled from one hour. Session
    # stratification cuts this further. Calibration windows need to be several
    # hours long, and the fit reports usable-bucket counts precisely so that a
    # thin window fails loudly rather than returning a plausible number.
    intensity_min_samples_per_bucket: int = 30

    # Volatility estimation
    vol_window_seconds: float = 300.0
    vol_min_observations: int = 30
    vol_floor: float = 1e-9      # guard against degenerate zero-vol windows

    # Session stratification. The dataset shows a 3.7x intraday swing in event
    # rate, so calibrating a single unconditional (A, kappa, sigma) blends
    # regimes that behave differently. Splitting by UTC hour bucket keeps the
    # estimates comparable within a session.
    stratify_by_session: bool = True
    session_boundaries_utc: tuple[int, ...] = (0, 6, 9, 13, 17, 21)


# --------------------------------------------------------------------------
# Strategy
# --------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """Quoter parameters.

    gamma is the inventory risk aversion. In the classical formulations it is
    a constant calibrated once; the adaptive variants re-decide it per cycle.

    q_max caps inventory in contract units. GLFT uses it directly in the
    boundary terms; AS and the symmetric quoter use it as a hard stop.
    """

    name: str = "glft"                # symmetric | avellaneda_stoikov | glft
    gamma: float = 0.1                # inventory risk aversion
    q_max: float = 10.0               # inventory bound, in contracts
    order_size: float = 1.0           # size per quote, in contracts
    # T - t for the finite-horizon formulations (AS only; GLFT's asymptotic
    # form has no horizon term).
    #
    # This parameter does more work than it looks like it does. AS's risk term
    # is gamma * sigma^2 * (T - t), so with sigma in price units per sqrt(s),
    # a one-hour horizon on BTC-PERP gives 0.1 * 0.1225 * 3600 = 44 USD of
    # half-spread against a market spread near 1 USD: AS quotes 25x outside
    # the touch and never trades. Setting it to the trading session length is
    # the intuitive choice and it is wrong.
    #
    # The defensible reading of T - t for a continuously running maker is the
    # horizon over which inventory is expected to be worked off, not the
    # length of the session. At observed fill rates that is on the order of a
    # minute, so that is the default. The sensitivity of results to this
    # choice belongs in the report rather than being buried here.
    horizon_seconds: float = 60.0

    # Symmetric baseline only.
    #
    # Sized from the real book, not from the synthetic generator. Measured on
    # tBTCF0:USTF0 epoch 27 (2026-09-06 02:00-03:00 UTC, 7,300 observations on
    # a 100 ms grid):
    #
    #     mid            79,996
    #     touch spread   median 15.0 USD  = 1.875 bps  (p10 13, p90 25.1)
    #     half-spread    median 0.9375 bps
    #
    # A half-spread of 1.0 bps therefore quotes *at* the touch, which is the
    # honest meaning for a "no inventory logic" baseline: it is present in the
    # book on both sides at the prevailing spread and does nothing cleverer.
    #
    # The previous default of 0.05 bps was 0.4 USD against a real touch of
    # 7.5 USD, i.e. 7 USD inside the best bid. Combined with an off-grid tick
    # that put the order on a price level the venue cannot represent, it
    # produced quotes that were both better than the whole market and had an
    # empty queue in front of them -- an instant-fill machine.
    fixed_half_spread_bps: float = 1.0

    # Quote hygiene, applies to every quoter
    #
    # The floor is one tick (1 USD at BTC's 76-80k, = 0.125 bps). Quoting
    # tighter than one tick is not a quote the venue can accept. The ceiling
    # of 5 bps is 40 USD, comfortably outside the p90 spread of 25 USD.
    min_half_spread_bps: float = 0.125
    max_half_spread_bps: float = 5.0

    # Price grid. None means "derive from price using the venue's rule"
    # (see bitfinex_tick_size). Set a float only to pin the grid for
    # synthetic data or tests, where there is no venue to ask.
    #
    # This was 0.5 -- a value that is not on Bitfinex's grid at any BTC price
    # level. Every quote that rounded to a .5 landed on a price the exchange
    # cannot quote, where the reconstructed book necessarily has zero depth,
    # so the queue model saw an empty queue and filled immediately.
    tick_size: float | None = None


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Execution model and fee assumptions.

    decision_interval_ms is the strategy clock. 100ms is the industry
    convention for a quoting loop: fast enough to react to book changes,
    slow enough that the run completes in reasonable time over a multi-day
    replay.

    Fees are expressed in basis points of notional. A negative maker fee is a
    rebate, which is the operative assumption for a designated market maker
    tier. All three of {0, standard, rebate} are run in the fee sensitivity
    study rather than picking one and hoping.
    """

    decision_interval_ms: int = 100
    maker_fee_bps: float = 0.0        # negative = rebate
    taker_fee_bps: float = 6.5        # only charged if we cross, which we avoid

    # Queue model. The key lesson from the earlier iteration of this project
    # was that treating quotes as unconditionally filled at the quoted distance
    # inflates fill counts badly on tight-spread instruments. Fills here are
    # conditional on queue position: we only fill once the volume ahead of us
    # at our price level has been consumed.
    use_queue_model: bool = True
    queue_ahead_inflation: float = 1.0   # >1 = conservative, assume more ahead

    # Latency. Quotes placed at time t only become live at t + latency.
    quote_latency_ms: float = 80.0       # p50 collector latency, SG -> Bitfinex

    # Book warm-up. A replay window that does not begin at a snapshot starts
    # with an empty book and fills in from the incremental stream, so the
    # reconstructed touch opens far too wide and narrows as orders arrive.
    # Measured on tBTCF0:USTF0 epoch 27 starting at 02:00 UTC:
    #
    #     t = 0     spread 168 USD
    #     t = 60s   spread  13 USD
    #     t = 300s  spread  14 USD      steady state (>30 min): 14 USD
    #
    # During that opening stretch a quoter is alone inside a spread that does
    # not exist, and takes fills no real maker could have had. The effect is
    # worst for strategies that trade rarely: Avellaneda-Stoikov took 30.6% of
    # its 62 fills in the first 60 seconds, carrying 89% of its PnL.
    #
    # The engine still consumes every event during warm-up -- the book and the
    # volatility estimator need it -- it simply does not quote, and records
    # nothing, so the measured window contains only quoting on a settled book.
    # 300s is five times the observed settling time.
    #
    # Set to 0.0 when the window genuinely starts at a snapshot (epoch start),
    # where there is nothing to warm up.
    warmup_seconds: float = 300.0

    # Adverse selection measurement horizon
    markout_horizons_ms: tuple[int, ...] = (100, 1000, 5000, 30000)


@dataclass
class Config:
    """Top-level run configuration."""

    clickhouse: ClickHouseConfig = field(default_factory=ClickHouseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    seed: int = 20260913

    # ---------------------------------------------------------------- I/O

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        def _sub(key: str, klass):
            return klass(**(raw.get(key) or {}))

        return cls(
            clickhouse=_sub("clickhouse", ClickHouseConfig),
            data=_sub("data", DataConfig),
            calibration=_sub("calibration", CalibrationConfig),
            strategy=_sub("strategy", StrategyConfig),
            simulation=_sub("simulation", SimulationConfig),
            seed=raw.get("seed", 20260913),
        )
