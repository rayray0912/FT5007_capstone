"""Typed configuration for the market-making backtest.

Everything that is a tunable knob lives here rather than being scattered as
magic numbers through the code. A run is fully described by a Config instance,
which makes results reproducible: serialise the config alongside the results
and the run can be repeated exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

import yaml


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

    # Intensity estimation
    intensity_n_buckets: int = 12
    intensity_max_delta_bps: float = 20.0
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
    horizon_seconds: float = 3600.0   # T - t for finite-horizon formulations

    # Symmetric baseline only
    fixed_half_spread_bps: float = 2.0

    # Quote hygiene, applies to every quoter
    min_half_spread_bps: float = 0.1
    max_half_spread_bps: float = 100.0
    tick_size: float = 0.5            # BTC-PERP on Bitfinex quotes in 0.5 USD


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
