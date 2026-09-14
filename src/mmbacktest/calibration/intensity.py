"""Order-flow intensity calibration.

The AS/GLFT family assumes the rate at which a quote posted at distance delta
from the mid gets filled decays exponentially in that distance:

    lambda(delta) = A * exp(-kappa * delta)

A sets the overall trading rate, kappa sets how quickly fill probability falls
away as the quote is posted further out. Both feed directly into the optimal
spread, so a bad estimate here propagates into every quoting decision.

Estimation procedure:

  1. For each trade, find the prevailing mid just before it. This needs the
     reconstructed book, not the trade tape alone, because the distance is
     measured relative to the mid at that instant.
  2. Compute delta = |trade_price - mid|, the distance the liquidity-taking
     order reached into the book.
  3. Bucket by delta, count trades per bucket, divide by elapsed time to get
     an empirical arrival rate per bucket.
  4. Regress log(rate) on delta. The slope is -kappa, the intercept is log(A).

Step 4 is a weighted least squares rather than an ordinary one: buckets near
the touch carry far more observations than the tail, and an unweighted fit
lets a handful of noisy far-out buckets dominate the slope.

Session stratification: the dataset shows a 3.7x intraday swing in event rate
between the quiet 09:00-10:00 UTC hour and the 23:00-02:00 peak. An
unconditional fit blends those regimes into an average that describes neither.
Estimates are therefore produced per session bucket by default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import CalibrationConfig

logger = logging.getLogger(__name__)


# Minimum number of populated distance buckets before a decay fit means
# anything. Two parameters are being estimated (slope and intercept), so
# three points leave one degree of freedom and R^2 is then near 1 almost
# regardless of the data -- a fit on three buckets reported R^2 = 0.993 on a
# window whose kappa was not identified at all. Four is the smallest number
# at which R^2 carries information.
#
# This threshold is used in two places that previously disagreed: the fit
# fell back at fewer than 3 buckets while is_usable required 4, so a 3-bucket
# window produced a healthy-looking fit that was then silently rejected.
MIN_BUCKETS_FOR_FIT = 4


@dataclass
class IntensityEstimate:
    """Fitted intensity parameters for one stratum."""

    A: float                 # arrival rate at zero distance, per second
    kappa: float             # decay in price units^-1
    n_trades: int
    n_buckets_used: int
    r_squared: float
    session: str = "all"

    @property
    def is_usable(self) -> bool:
        """Whether the fit is good enough to quote on.

        A negative or near-zero kappa means the fit found no decay, which
        makes the optimal spread formula degenerate. An R^2 below 0.5 means
        the exponential form is not describing this data well and any quote
        built on it is guesswork.
        """
        return (
            self.kappa > 1e-9
            and self.r_squared >= 0.5
            and self.n_buckets_used >= MIN_BUCKETS_FOR_FIT
        )

    @property
    def arrival_intensity(self) -> float:
        """GLFT's `A`: fills per second at a quote sitting on the mid, one side.

        This is NOT the fitted `A`, and the difference is a factor of
        1 / (2 * kappa) -- about 4.8x on BTC-PERP.

        The fit estimates a *density*. It buckets trades by distance from the
        mid and regresses the per-bucket rate, so `self.A` is arrivals per
        second per unit of distance, pooled over both aggressor directions.

        Gueant-Lehalle-Fernandez-Tapia define the intensity of the point
        process that executes our quote: lambda(d) = A exp(-k d) is the rate
        at which an order resting at distance d gets filled. An order at
        distance d is executed by any trade printing at distance >= d,
        because a marketable order consumes every level it crosses on the way
        out. So the intensity at d is the density integrated outwards:

            lambda(d) = int_d^inf A_density e^{-kappa u} du
                      = (A_density / kappa) e^{-kappa d}

        giving lambda(0) = A_density / kappa. The extra factor of two is the
        aggressor direction: the density pools buy- and sell-initiated
        trades, while a resting bid is only ever hit by sellers.

        Feeding the raw density into GLFT understates A by ~4.8x. A sits
        under a square root in the inventory coefficient, so the quote comes
        out ~2.2x too wide -- enough on this instrument to hit the
        max_half_spread clamp and stop the strategy trading at all.
        """
        return self.A / max(self.kappa, 1e-12) / 2.0

    def fill_rate(self, delta: float | np.ndarray) -> float | np.ndarray:
        """lambda(delta), the expected fills per second at distance delta.

        Uses the GLFT-convention intensity (see `arrival_intensity`), so this
        is the rate at which a quote resting at `delta` is executed.
        """
        return self.arrival_intensity * np.exp(
            -self.kappa * np.asarray(delta, dtype=float)
        )

    def __str__(self) -> str:
        flag = "" if self.is_usable else "  [UNUSABLE]"
        return (
            f"session={self.session:>10s}  A={self.A:9.4f}/s  "
            f"kappa={self.kappa:10.6f}  R2={self.r_squared:.3f}  "
            f"n={self.n_trades:,}{flag}"
        )


def session_of(ts: pd.Series, boundaries: tuple[int, ...]) -> pd.Series:
    """Map timestamps to a session label from UTC-hour boundaries."""
    hours = pd.to_datetime(ts).dt.hour
    edges = sorted(boundaries)
    labels = []
    for h in hours:
        lo = max((b for b in edges if b <= h), default=edges[-1])
        idx = edges.index(lo)
        hi = edges[idx + 1] if idx + 1 < len(edges) else edges[0] + 24
        labels.append(f"{lo:02d}-{hi % 24:02d}")
    return pd.Series(labels, index=ts.index if hasattr(ts, "index") else None)


def attach_mid_to_trades(
    trades: pd.DataFrame,
    mid_series: pd.DataFrame,
) -> pd.DataFrame:
    """Join each trade to the mid prevailing just before it.

    `mid_series` must have columns ts and mid, sorted by ts. The join is
    backward-looking (asof), so a trade is matched to the most recent book
    observation at or before the trade timestamp. Matching forward would leak
    the trade's own price impact into the reference mid.
    """
    if trades.empty or mid_series.empty:
        return trades.assign(mid=np.nan, delta=np.nan)

    left = trades.sort_values("ts_exch").reset_index(drop=True)
    right = mid_series.sort_values("ts").reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right[["ts", "mid"]],
        left_on="ts_exch",
        right_on="ts",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["delta"] = (merged["price"] - merged["mid"]).abs()
    return merged.drop(columns=["ts"])


def estimate_intensity(
    trades_with_mid: pd.DataFrame,
    cfg: CalibrationConfig,
    duration_seconds: float,
    session: str = "all",
) -> IntensityEstimate:
    """Fit lambda(delta) = A exp(-kappa delta) for one stratum."""
    df = trades_with_mid.dropna(subset=["delta", "mid"])
    if df.empty or duration_seconds <= 0:
        return IntensityEstimate(0.0, 0.0, 0, 0, 0.0, session)

    # Work in price units, but set the bucket range from a bps budget so the
    # same config transfers across instruments at very different price levels.
    ref_mid = float(df["mid"].median())
    max_delta = ref_mid * cfg.intensity_max_delta_bps / 10_000.0

    edges = np.linspace(0.0, max_delta, cfg.intensity_n_buckets + 1)
    counts, _ = np.histogram(df["delta"].to_numpy(dtype=float), bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Empirical arrival rate per bucket, per second.
    rates = counts / duration_seconds

    keep = counts >= cfg.intensity_min_samples_per_bucket
    if keep.sum() < MIN_BUCKETS_FOR_FIT:
        # Not enough resolved buckets to fit a slope. Fall back to a single
        # aggregate rate with no decay information, and mark it unusable via
        # kappa <= 0 so callers do not quote on it.
        logger.warning(
            "intensity fit for session=%s has only %d usable buckets",
            session, int(keep.sum()),
        )
        return IntensityEstimate(
            A=float(counts.sum() / duration_seconds),
            kappa=0.0,
            n_trades=int(counts.sum()),
            n_buckets_used=int(keep.sum()),
            r_squared=0.0,
            session=session,
        )

    x = centres[keep]
    y = np.log(rates[keep])
    # Weight by observation count: the touch buckets are estimated far more
    # precisely than the tail, and should dominate the slope accordingly.
    w = counts[keep].astype(float)

    slope, intercept, r2 = _weighted_linfit(x, y, w)

    # Normalise out the bucket width. The histogram gives counts per bucket,
    # so rate_i = A * exp(-kappa * centre_i) * width, and the fitted intercept
    # is log(A * width) rather than log(A). Without this correction A would
    # scale with the bucket count, which would silently change every quote
    # when intensity_n_buckets is retuned. After it, A is a rate density in
    # arrivals per second per unit of price distance, and kappa is unaffected
    # since a constant factor only shifts the intercept.
    bucket_width = float(edges[1] - edges[0])
    A = float(np.exp(intercept) / bucket_width) if bucket_width > 0 else 0.0

    return IntensityEstimate(
        A=A,
        kappa=float(-slope),
        n_trades=int(counts.sum()),
        n_buckets_used=int(keep.sum()),
        r_squared=float(r2),
        session=session,
    )


def estimate_by_session(
    trades_with_mid: pd.DataFrame,
    cfg: CalibrationConfig,
) -> dict[str, IntensityEstimate]:
    """Fit one intensity model per session bucket.

    Returns a dict keyed by session label, always including an "all" key with
    the unconditional fit so the stratified and pooled estimates can be
    compared directly.
    """
    df = trades_with_mid.dropna(subset=["delta", "mid"]).copy()
    out: dict[str, IntensityEstimate] = {}

    if df.empty:
        return {"all": IntensityEstimate(0.0, 0.0, 0, 0, 0.0, "all")}

    total_seconds = _span_seconds(df["ts_exch"])
    out["all"] = estimate_intensity(df, cfg, total_seconds, "all")

    if not cfg.stratify_by_session:
        return out

    df["session"] = session_of(df["ts_exch"], cfg.session_boundaries_utc)
    for label, grp in df.groupby("session"):
        # Duration for a session is the share of wall-clock time that session
        # covers, not the span of its trades: a session with few trades still
        # occupied its full share of the window, and dividing by the trade
        # span would inflate its rate.
        hours_in_session = _session_hours(label)
        seconds = total_seconds * hours_in_session / 24.0
        out[label] = estimate_intensity(grp, cfg, max(seconds, 1.0), label)

    return out


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _weighted_linfit(
    x: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[float, float, float]:
    """Weighted least squares of y on x. Returns (slope, intercept, R^2)."""
    w = w / w.sum()
    xm = float(np.sum(w * x))
    ym = float(np.sum(w * y))
    cov = float(np.sum(w * (x - xm) * (y - ym)))
    var = float(np.sum(w * (x - xm) ** 2))
    if var <= 0:
        return 0.0, ym, 0.0

    slope = cov / var
    intercept = ym - slope * xm

    resid = y - (slope * x + intercept)
    ss_res = float(np.sum(w * resid**2))
    ss_tot = float(np.sum(w * (y - ym) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r2


def _span_seconds(ts: pd.Series) -> float:
    if len(ts) < 2:
        return 1.0
    span = pd.to_datetime(ts).max() - pd.to_datetime(ts).min()
    return max(span.total_seconds(), 1.0)


def _session_hours(label: str) -> float:
    """Hours covered by a session label of the form 'HH-HH'."""
    try:
        lo_s, hi_s = label.split("-")
        lo, hi = int(lo_s), int(hi_s)
    except (ValueError, AttributeError):
        return 24.0
    span = (hi - lo) % 24
    return float(span if span > 0 else 24)
