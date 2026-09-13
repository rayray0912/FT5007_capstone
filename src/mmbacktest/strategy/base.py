"""Quoter interface.

Every strategy is a pure function from market state to a pair of quotes. That
purity is deliberate: it means a strategy can be unit-tested without a
simulator, swapped without touching the execution path, and compared against
another strategy on exactly the same state sequence.

The three implementations in this package form a deliberate progression:

  symmetric            fixed half-spread around the mid, no inventory logic
  avellaneda_stoikov   inventory-skewed reservation price, finite horizon
  glft                 GLFT, with an explicit inventory bound

Each adds one piece of machinery to the one before it, so a difference in
results can be attributed to that piece rather than to a wholesale change of
approach.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..config import StrategyConfig


@dataclass(frozen=True)
class MarketState:
    """Everything a quoter is allowed to see at a decision point.

    Deliberately narrow. A quoter cannot reach into the book or the future;
    if a strategy needs a new input it has to be added here explicitly, which
    keeps lookahead bugs from creeping in unnoticed.
    """

    ts_seconds: float          # epoch seconds, for horizon arithmetic
    mid: float
    best_bid: float
    best_ask: float
    microprice: float
    imbalance: float           # [-1, 1], positive means bid-heavy
    sigma: float               # price units per sqrt(second)
    kappa: float               # intensity decay, per price unit
    A: float                   # intensity level, arrivals/s per price unit
    inventory: float           # signed, in contracts
    time_remaining: float      # seconds to horizon end

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid


@dataclass(frozen=True)
class Quote:
    """A two-sided quote.

    Either side may be None, which means "do not quote this side". That is a
    real decision, not an error case: at the inventory bound a quoter should
    stop adding to the position rather than widen indefinitely.
    """

    bid_price: float | None
    ask_price: float | None
    bid_size: float
    ask_size: float
    reservation_price: float
    half_spread: float

    @property
    def is_two_sided(self) -> bool:
        return self.bid_price is not None and self.ask_price is not None

    @property
    def quoted_spread(self) -> float | None:
        if not self.is_two_sided:
            return None
        return self.ask_price - self.bid_price


class Quoter(ABC):
    """Base class for quoting strategies."""

    name: str = "base"

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    @abstractmethod
    def compute(self, state: MarketState) -> Quote:
        """Return the quote for this decision point."""

    # ------------------------------------------------------------------
    # Shared post-processing
    # ------------------------------------------------------------------

    def _finalise(
        self,
        state: MarketState,
        reservation: float,
        half_spread: float,
    ) -> Quote:
        """Turn a (reservation price, half-spread) pair into a live quote.

        Three things happen here, applied identically to every strategy so
        that comparisons are not contaminated by differences in quote hygiene:

        1. The half-spread is clamped to a configured band. An unclamped
           formula can return a half-spread of zero when sigma collapses, or
           an enormous one when inventory is extreme; neither is a quote a
           real system would send.
        2. Prices are rounded to the tick grid, away from the mid on both
           sides so rounding never tightens the quote beyond what the
           strategy asked for.
        3. Sides are suppressed at the inventory bound.
        """
        mid = state.mid
        lo = self.cfg.min_half_spread_bps * mid / 10_000.0
        hi = self.cfg.max_half_spread_bps * mid / 10_000.0
        half_spread = float(np.clip(half_spread, lo, hi))

        tick = self.cfg.tick_size
        raw_bid = reservation - half_spread
        raw_ask = reservation + half_spread

        # Round outward: bids down, asks up. Rounding inward would post a
        # tighter quote than the model chose, which biases fills upward.
        bid_price = float(np.floor(raw_bid / tick) * tick)
        ask_price = float(np.ceil(raw_ask / tick) * tick)

        # Never quote through the opposite side of the book. Crossing would
        # make us a taker, which is a different strategy entirely.
        if bid_price >= state.best_ask:
            bid_price = state.best_ask - tick
        if ask_price <= state.best_bid:
            ask_price = state.best_bid + tick

        q = self.cfg.order_size
        q_max = self.cfg.q_max
        inv = state.inventory

        # Inventory bound. At the cap we stop adding on that side but keep
        # quoting the other, so the position can still be worked down.
        bid_size = q if inv + q <= q_max else 0.0
        ask_size = q if inv - q >= -q_max else 0.0

        return Quote(
            bid_price=bid_price if bid_size > 0 else None,
            ask_price=ask_price if ask_size > 0 else None,
            bid_size=bid_size,
            ask_size=ask_size,
            reservation_price=reservation,
            half_spread=half_spread,
        )


def make_quoter(cfg: StrategyConfig) -> Quoter:
    """Factory. Keeps the CLI from importing every strategy module."""
    from .symmetric import SymmetricQuoter
    from .avellaneda_stoikov import AvellanedaStoikovQuoter
    from .glft import GLFTQuoter

    registry: dict[str, type[Quoter]] = {
        "symmetric": SymmetricQuoter,
        "avellaneda_stoikov": AvellanedaStoikovQuoter,
        "as": AvellanedaStoikovQuoter,
        "glft": GLFTQuoter,
    }
    key = cfg.name.lower()
    if key not in registry:
        raise ValueError(
            f"unknown strategy {cfg.name!r}; available: {sorted(set(registry))}"
        )
    return registry[key](cfg)
