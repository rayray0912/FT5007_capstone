"""Order-level book reconstruction from MBO events.

Why order level rather than aggregated levels: a market-making backtest lives
or dies on the realism of its fill model. With aggregated L2 depth you know
how much size sits at a price, but not where a simulated order would sit in
that queue, so the only available fill rule is "the level traded through",
which systematically overstates fills. With order-level data the book keeps
individual orders in arrival order, so a simulated order can be inserted at
the back of a specific queue and only fills once the volume ahead of it is
actually consumed.

The reconstruction is a straightforward state machine over the event stream:

  snapshot  reset the book, then insert
  add       insert order at the back of its price level
  update    modify an existing order in place
  delete    remove the order, crediting its residual size to queue depletion

All of this is epoch-scoped by the caller. Within an epoch the stream is
complete (zero seq gaps, zero checksum failures in this dataset), so the book
is exact rather than approximate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd


# Bitfinex encodes side in the sign of `amount` on the raw feed, but the
# collector normalises it into an explicit column. Keep both spellings working.
_BID = {"bid", "BID", "b", 1, "1"}
_ASK = {"ask", "ASK", "a", -1, "-1"}


def _is_bid(side) -> bool:
    return side in _BID


@dataclass
class Order:
    """A single resting order."""

    order_id: int
    price: float
    size: float
    is_bid: bool
    seq: int  # arrival sequence, defines queue priority within a price level


@dataclass
class PriceLevel:
    """Orders resting at one price, in arrival order.

    `orders` is kept sorted by seq so index 0 is the front of the queue. For
    the book sizes involved (250 orders per side, typically a handful per
    level) a list with linear scan is faster in practice than anything
    cleverer, and it keeps the queue semantics obvious.
    """

    price: float
    orders: list[Order] = field(default_factory=list)

    @property
    def total_size(self) -> float:
        return sum(abs(o.size) for o in self.orders)

    @property
    def n_orders(self) -> int:
        return len(self.orders)

    def volume_ahead_of(self, seq: int) -> float:
        """Size resting in front of an order that arrived at `seq`.

        This is the quantity a simulated order at this level would have to
        wait through before it can trade.
        """
        return sum(abs(o.size) for o in self.orders if o.seq < seq)


class OrderBook:
    """Order-level limit order book for one instrument."""

    def __init__(self, tick_size: float | None = None):
        # Recorded for reference only: reconstruction keys levels by the exact
        # prices the exchange sent, so the book never needs to know the grid.
        # It is deliberately NOT used to round or bucket anything -- doing so
        # would impose a grid on data that already has one, and an incorrect
        # value here should not be able to corrupt the book. The quoter, which
        # does need the grid, derives it per price (see config.bitfinex_tick_size).
        self.tick_size = tick_size
        self._orders: dict[int, Order] = {}
        self._bids: dict[float, PriceLevel] = {}
        self._asks: dict[float, PriceLevel] = {}
        self._last_seq: int = 0
        self.n_events: int = 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._orders.clear()
        self._bids.clear()
        self._asks.clear()

    def apply(self, order_id: int, side, action: str, price: float,
              amount: float, seq: int) -> float:
        """Apply one event. Returns the size removed from the book, if any.

        The return value is what drives queue depletion in the fill simulator:
        when an order in front of ours disappears, either it was cancelled or
        it traded, and either way our queue position improves.
        """
        self.n_events += 1
        self._last_seq = seq
        action = str(action).lower()

        if action == "snapshot":
            # A snapshot arrives as a burst of rows after (re)connect. The
            # caller resets the book once at the start of the burst; here we
            # simply insert.
            self._insert(order_id, side, price, amount, seq)
            return 0.0

        if action == "add":
            self._insert(order_id, side, price, amount, seq)
            return 0.0

        if action == "update":
            existing = self._orders.get(order_id)
            if existing is None:
                # Update for an order we never saw (can happen at the very
                # start of an epoch before the snapshot completes). Treat as
                # an insert so the book stays consistent.
                self._insert(order_id, side, price, amount, seq)
                return 0.0
            removed = self._remove(order_id)
            # A price change loses queue priority, which the new seq encodes.
            self._insert(order_id, side, price, amount, seq)
            return removed

        if action == "delete":
            return self._remove(order_id)

        raise ValueError(f"unknown book action: {action!r}")

    def _insert(self, order_id: int, side, price: float,
                amount: float, seq: int) -> None:
        if price <= 0 or amount == 0:
            return
        is_bid = _is_bid(side)
        order = Order(order_id, float(price), abs(float(amount)), is_bid, seq)
        self._orders[order_id] = order

        book = self._bids if is_bid else self._asks
        level = book.get(price)
        if level is None:
            level = PriceLevel(price)
            book[price] = level
        level.orders.append(order)
        # Orders normally arrive in seq order, so the append is already
        # sorted; sort defensively only when it is not.
        if len(level.orders) > 1 and level.orders[-2].seq > seq:
            level.orders.sort(key=lambda o: o.seq)

    def _remove(self, order_id: int) -> float:
        order = self._orders.pop(order_id, None)
        if order is None:
            return 0.0
        book = self._bids if order.is_bid else self._asks
        level = book.get(order.price)
        if level is None:
            return 0.0
        level.orders = [o for o in level.orders if o.order_id != order_id]
        if not level.orders:
            del book[order.price]
        return order.size

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def best_bid(self) -> float | None:
        return max(self._bids) if self._bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self._asks) if self._asks else None

    @property
    def mid(self) -> float | None:
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        return 0.5 * (b + a)

    @property
    def spread(self) -> float | None:
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        return a - b

    @property
    def microprice(self) -> float | None:
        """Size-weighted mid.

        Weighting by the opposite side's size is the standard convention: a
        thick bid and thin ask implies upward pressure, so the fair value sits
        nearer the ask.
        """
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        bid_sz = self._bids[b].total_size
        ask_sz = self._asks[a].total_size
        total = bid_sz + ask_sz
        if total <= 0:
            return 0.5 * (b + a)
        return (b * ask_sz + a * bid_sz) / total

    def depth(self, is_bid: bool, price: float) -> float:
        book = self._bids if is_bid else self._asks
        level = book.get(price)
        return level.total_size if level else 0.0

    def volume_ahead(self, is_bid: bool, price: float, seq: int) -> float:
        """Resting size in front of a hypothetical order at (price, seq)."""
        book = self._bids if is_bid else self._asks
        level = book.get(price)
        return level.volume_ahead_of(seq) if level else 0.0

    def levels(self, is_bid: bool, n: int = 10) -> list[tuple[float, float, int]]:
        """Top n levels as (price, size, n_orders), best first."""
        book = self._bids if is_bid else self._asks
        prices = sorted(book, reverse=is_bid)[:n]
        return [(p, book[p].total_size, book[p].n_orders) for p in prices]

    def imbalance(self, n_levels: int = 5) -> float:
        """Order-book imbalance in [-1, 1]. Positive means bid-heavy."""
        bid_sz = sum(s for _, s, _ in self.levels(True, n_levels))
        ask_sz = sum(s for _, s, _ in self.levels(False, n_levels))
        total = bid_sz + ask_sz
        if total <= 0:
            return 0.0
        return (bid_sz - ask_sz) / total

    def snapshot(self) -> dict:
        """Flat summary for logging or feature extraction."""
        return {
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid": self.mid,
            "microprice": self.microprice,
            "spread": self.spread,
            "imbalance": self.imbalance(),
            "n_orders": len(self._orders),
        }

    def is_crossed(self) -> bool:
        """Sanity check. A crossed book means the reconstruction is wrong."""
        b, a = self.best_bid, self.best_ask
        return b is not None and a is not None and b >= a


def replay(
    events: pd.DataFrame,
    tick_size: float = 0.5,
    emit_every_ms: int | None = None,
) -> Iterator[tuple[pd.Timestamp, OrderBook, float]]:
    """Replay an event stream, yielding book state.

    Yields (timestamp, book, removed_size). `removed_size` is the size that
    left the book on this event, which the fill simulator consumes to advance
    queue positions.

    When `emit_every_ms` is set, the book is only yielded on that clock rather
    than on every event. The book is still updated on every event; only the
    observation is downsampled. That is the right way round: downsampling the
    updates would corrupt the book, downsampling the observations just means
    the strategy looks less often, which is what a real quoting loop does.
    """
    book = OrderBook(tick_size=tick_size)

    ts_col = events["ts_exch"].to_numpy()
    seq_col = events["seq"].to_numpy()
    oid_col = events["order_id"].to_numpy()
    side_col = events["side"].to_numpy()
    action_col = events["action"].to_numpy()
    price_col = events["price"].to_numpy(dtype=float)
    amount_col = events["amount"].to_numpy(dtype=float)

    next_emit: np.datetime64 | None = None
    interval = np.timedelta64(emit_every_ms, "ms") if emit_every_ms else None

    # A snapshot burst means "the book is now this". Detect the start of a
    # burst (first snapshot row after a non-snapshot row) and reset there.
    prev_action = ""

    for i in range(len(events)):
        action = str(action_col[i]).lower()
        if action == "snapshot" and prev_action != "snapshot":
            book.clear()
        prev_action = action

        removed = book.apply(
            order_id=int(oid_col[i]),
            side=side_col[i],
            action=action,
            price=float(price_col[i]),
            amount=float(amount_col[i]),
            seq=int(seq_col[i]),
        )

        ts = ts_col[i]
        if interval is None:
            yield pd.Timestamp(ts), book, removed
        else:
            if next_emit is None:
                next_emit = ts + interval
            elif ts >= next_emit:
                yield pd.Timestamp(ts), book, removed
                # Skip forward rather than emitting a burst if there was a gap
                while next_emit <= ts:
                    next_emit = next_emit + interval
