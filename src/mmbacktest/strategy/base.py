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

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..config import (
    StrategyConfig,
    bitfinex_tick_size,
    is_on_tick_grid,
    round_to_tick,
)


logger = logging.getLogger(__name__)

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
        # Sides not quoted because the raw price was non-positive or
        # non-finite. Zero in ordinary operation; see _finalise.
        self.n_sides_suppressed = 0

    @abstractmethod
    def compute(self, state: MarketState) -> Quote:
        """Return the quote for this decision point."""

    # ------------------------------------------------------------------
    # Price grid
    # ------------------------------------------------------------------

    def tick_for(self, price: float) -> float:
        """Tick size applicable at `price`.

        `cfg.tick_size` of None means "ask the venue rule", which is the
        correct behaviour against real data because Bitfinex's tick is a
        function of price. A float pins the grid, which synthetic data and
        tests need since there is no venue to ask.
        """
        if self.cfg.tick_size is not None:
            return self.cfg.tick_size
        return bitfinex_tick_size(price)

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

        raw_bid = reservation - half_spread
        raw_ask = reservation + half_spread

        # Order of operations matters here, and used to be wrong.
        #
        # The no-cross repair runs on the *raw* prices, before any tick
        # lookup. A reservation price pushed far from the mid -- which happens
        # for a few minutes after a flash move, when realised sigma is
        # legitimately enormous -- can make the raw bid negative. Bitfinex's
        # tick is defined only for positive prices, so rounding first raised
        # on it (epoch 68, 2026-09-20 16:31, AS at gamma=1e-4: the book had
        # spiked from 81,200 to 153,960 and back three minutes earlier, sigma
        # was ~12,300 USD/sqrt(s), and an inventory of 0.154 put the raw bid at
        # -58,595). Repairing first pins the near side to the touch, which is
        # what the strategy is asking for when it skews that hard.
        #
        # For every finite positive raw price this is identical to the old
        # round-then-repair order: rounding is outward, and the touch is
        # already on the grid, so the repair triggers on exactly the same
        # inputs and lands on exactly the same price.
        #
        # The tick is a function of price on this venue, not a constant, so
        # each side is snapped on its own grid. The two agree except in the
        # rare case where a quote straddles a power of ten. Stepping back from
        # the touch takes the tick at the touch; the result is then rounded on
        # the raw price's grid, as the original code did. When the raw price
        # sits a decade above the touch (a skew of ~19,000 USD at BTC 81k,
        # reachable only after a flash move) that rounding is coarser than the
        # touch's grid and the repaired quote lands a few ticks further out:
        # still on the venue grid, conservative, and kept so that results stay
        # comparable with sweeps run before this reordering. Rounding is
        # outward (bids down, asks up): rounding inward would post a tighter
        # quote than the model chose.
        if np.isfinite(raw_ask) and raw_ask <= state.best_bid:
            ask_price: float | None = state.best_bid + self.tick_for(state.best_bid)
            ask_tick = self.tick_for(ask_price)
            ask_price = round_to_tick(ask_price, ask_tick, "up")
        elif np.isfinite(raw_ask) and raw_ask > 0.0:
            ask_tick = self.tick_for(raw_ask)
            ask_price = round_to_tick(raw_ask, ask_tick, "up")
        else:
            ask_price, ask_tick = None, None

        if np.isfinite(raw_bid) and raw_bid >= state.best_ask:
            bid_tick = self.tick_for(raw_bid)
            bid_price: float | None = round_to_tick(
                state.best_ask - self.tick_for(state.best_ask), bid_tick, "down"
            )
        elif np.isfinite(raw_bid) and raw_bid > 0.0:
            bid_tick = self.tick_for(raw_bid)
            bid_price = round_to_tick(raw_bid, bid_tick, "down")
        else:
            bid_price, bid_tick = None, None

        # Rounding can still land a tiny positive raw price on zero: the venue
        # caps prices at 8 decimals, so anything under 1e-8 rounds to 0.0.
        # Positivity has to be checked after rounding, not only before.
        if bid_price is not None and bid_price <= 0.0:
            bid_price, bid_tick = None, None
        if ask_price is not None and ask_price <= 0.0:
            ask_price, ask_tick = None, None

        # A side with no valid price is not quoted. A non-positive bid is not a
        # quote at any level the model meant; the alternative, clamping the
        # reservation to a maximum skew, would also bind in ordinary
        # conditions at the higher gammas and silently change their results.
        # Counted, because how often this fires is itself worth reporting.
        for side, px in (("bid", bid_price), ("ask", ask_price)):
            if px is None:
                self.n_sides_suppressed += 1
                if self.n_sides_suppressed == 1:
                    logger.warning(
                        "%s: %s side suppressed, raw price non-positive or "
                        "non-finite (reservation %.1f, half-spread %.1f, "
                        "sigma %.2f, inventory %.4f)",
                        self.name, side, reservation, half_spread,
                        state.sigma, state.inventory,
                    )

        # An off-grid quote is a silent instant-fill bug in the queue model
        # (no depth can exist at a price the venue cannot represent), so it
        # is asserted rather than assumed.
        for px, tk in ((bid_price, bid_tick), (ask_price, ask_tick)):
            if px is not None and not is_on_tick_grid(px, tk):
                raise AssertionError(
                    f"{self.name}: quote {px!r} is off the {tk!r} tick grid"
                )

        q = self.cfg.order_size
        q_max = self.cfg.q_max
        inv = state.inventory

        # Inventory bound. At the cap we stop adding on that side but keep
        # quoting the other, so the position can still be worked down.
        bid_size = q if (inv + q <= q_max and bid_price is not None) else 0.0
        ask_size = q if (inv - q >= -q_max and ask_price is not None) else 0.0

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
