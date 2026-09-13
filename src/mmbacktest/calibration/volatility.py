"""Short-horizon volatility estimation.

sigma enters the AS/GLFT quotes twice: it scales the inventory skew (how far
the reservation price moves per unit of inventory) and it scales the risk term
in the spread. Getting it wrong is directly a quoting error, so it is worth
being careful about two things that are easy to get wrong on high-frequency
data.

First, the theory's sigma is the diffusion coefficient of the mid in price
units per sqrt(second), not a returns volatility and not an annualised
number. Everything here stays in price units per sqrt(second) so it can be
substituted into the formulas without a conversion step.

Second, MBO observations arrive on an irregular clock. A naive rolling
standard deviation over the last N observations mixes a burst of 200 events in
one second with 200 events spread over a minute, and the resulting number
means neither. Sampling on a fixed time grid before differencing fixes this.

Microstructure noise is a known problem for realised variance at very high
frequency: bid-ask bounce inflates the estimate as the sampling interval
shrinks. Two mitigations are available here: sampling the mid (rather than
trade prices) already removes most of the bounce, and the sampling interval is
configurable so the sensitivity can be checked rather than assumed away.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import CalibrationConfig

logger = logging.getLogger(__name__)


@dataclass
class VolatilityEstimate:
    """Volatility over one window, in price units per sqrt(second)."""

    sigma: float
    n_observations: int
    window_seconds: float
    sampling_interval_seconds: float

    @property
    def is_usable(self) -> bool:
        return self.sigma > 0 and self.n_observations >= 10

    def scaled(self, seconds: float) -> float:
        """Standard deviation of the mid move over a horizon, in price units."""
        return self.sigma * np.sqrt(max(seconds, 0.0))

    def __str__(self) -> str:
        return (
            f"sigma={self.sigma:.6f} price/sqrt(s)  "
            f"n={self.n_observations}  window={self.window_seconds:.0f}s"
        )


def realised_volatility(
    ts: pd.Series | np.ndarray,
    mid: pd.Series | np.ndarray,
    sampling_interval_seconds: float = 1.0,
    min_observations: int = 30,
    floor: float = 1e-9,
) -> VolatilityEstimate:
    """Realised volatility of the mid on a fixed time grid.

    The series is resampled onto a regular grid (last observation carried
    forward) before differencing, so each squared increment covers the same
    amount of wall-clock time and the estimator is a proper realised variance.
    """
    ts = pd.to_datetime(pd.Series(ts).reset_index(drop=True))
    mid = pd.Series(mid).reset_index(drop=True).astype(float)

    frame = pd.DataFrame({"ts": ts, "mid": mid}).dropna()
    if len(frame) < 2:
        return VolatilityEstimate(floor, 0, 0.0, sampling_interval_seconds)

    frame = frame.set_index("ts").sort_index()
    span = (frame.index[-1] - frame.index[0]).total_seconds()

    rule = pd.Timedelta(seconds=sampling_interval_seconds)
    grid = frame["mid"].resample(rule).last().ffill().dropna()

    if len(grid) < min_observations:
        # Too few grid points for a stable estimate. Fall back to the raw
        # observations rather than returning nothing, but the caller can see
        # from n_observations that this is a thin estimate.
        diffs = frame["mid"].diff().dropna().to_numpy()
        if diffs.size == 0 or span <= 0:
            return VolatilityEstimate(floor, 0, span, sampling_interval_seconds)
        per_step = float(np.sqrt(np.mean(diffs**2)))
        mean_dt = span / max(len(frame) - 1, 1)
        sigma = per_step / np.sqrt(max(mean_dt, 1e-9))
        return VolatilityEstimate(
            max(sigma, floor), len(frame), span, sampling_interval_seconds
        )

    increments = grid.diff().dropna().to_numpy(dtype=float)
    # Realised variance per interval, converted to per-second.
    rv_per_interval = float(np.mean(increments**2))
    sigma = float(np.sqrt(rv_per_interval / sampling_interval_seconds))

    return VolatilityEstimate(
        sigma=max(sigma, floor),
        n_observations=int(len(grid)),
        window_seconds=span,
        sampling_interval_seconds=sampling_interval_seconds,
    )


class RollingVolatility:
    """Online volatility tracker for use inside the backtest loop.

    Keeps a deque of recent (timestamp, mid) samples on the decision clock and
    recomputes over the trailing window. The strategy needs a sigma at every
    decision point, and recomputing from the full history each time would make
    the replay quadratic.

    Implementation detail: the running sums are maintained incrementally, so
    each update is O(1) amortised rather than O(window).
    """

    def __init__(self, cfg: CalibrationConfig, sampling_interval_seconds: float = 1.0):
        self.window_seconds = cfg.vol_window_seconds
        self.min_obs = cfg.vol_min_observations
        self.floor = cfg.vol_floor
        self.dt = sampling_interval_seconds

        self._ts: list[float] = []      # epoch seconds
        self._mid: list[float] = []
        self._sum_sq_diff: float = 0.0
        self._n_diff: int = 0

    def update(self, ts: pd.Timestamp, mid: float) -> None:
        """Record a new observation and drop anything outside the window."""
        if mid is None or not np.isfinite(mid):
            return

        t = ts.timestamp() if hasattr(ts, "timestamp") else float(ts)

        if self._mid:
            diff = mid - self._mid[-1]
            self._sum_sq_diff += diff * diff
            self._n_diff += 1

        self._ts.append(t)
        self._mid.append(mid)

        cutoff = t - self.window_seconds
        while len(self._ts) > 2 and self._ts[0] < cutoff:
            # Removing the oldest sample also removes the increment that led
            # into the second sample, keeping the sums consistent.
            old_diff = self._mid[1] - self._mid[0]
            self._sum_sq_diff -= old_diff * old_diff
            self._n_diff -= 1
            self._ts.pop(0)
            self._mid.pop(0)

    @property
    def sigma(self) -> float:
        """Current sigma in price units per sqrt(second)."""
        if self._n_diff < self.min_obs:
            return self.floor
        mean_sq = self._sum_sq_diff / self._n_diff
        # Increments are spaced by the decision clock, so convert to per-second.
        return max(float(np.sqrt(mean_sq / self.dt)), self.floor)

    @property
    def n_observations(self) -> int:
        return len(self._mid)

    @property
    def is_warm(self) -> bool:
        """Whether enough history has accumulated to quote on this estimate."""
        return self._n_diff >= self.min_obs

    def estimate(self) -> VolatilityEstimate:
        return VolatilityEstimate(
            sigma=self.sigma,
            n_observations=self.n_observations,
            window_seconds=self.window_seconds,
            sampling_interval_seconds=self.dt,
        )
