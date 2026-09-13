"""Queue-position-aware fill simulation.

This module is the reason the project uses order-level data at all.

The naive fill rule
-------------------

The rule almost every backtest starts with is: if a trade prints at or through
my quoted price, I filled. On a liquid instrument with a tight spread this is
badly wrong, and wrong in a direction that flatters the strategy. A quote
posted at the touch sits behind whatever was already resting there. When a
market order arrives and consumes 2 contracts, the orders at the front of the
queue trade; an order that joined the queue a moment ago does not. Assuming it
did inflates the fill count, and every fill it invents is a fill the strategy
did not have to wait or compete for.

That is the concrete error that produced an over-optimistic result earlier in
this project, and it is worth being precise about why it flatters: the invented
fills are disproportionately the *good* ones. Real queue priority means a
market-maker's fills cluster in exactly the moments when flow is heaviest, and
heavy flow is correlated with informed flow. Filling unconditionally removes
that correlation and therefore removes most of the adverse selection.

The rule used here
------------------

An order is placed at a price level with a recorded queue position: the volume
resting ahead of it at that level when it arrives. That queue position is then
decremented by every subsequent event at that level that removes volume, i.e.
cancels and trades. Only once the volume ahead reaches zero can the order
trade, and even then only against volume that actually arrives.

This is still an approximation. Real exchanges do not publish which resting
order a trade consumed, so cancels ahead of us are treated as improving our
position, which is correct, and trades ahead of us likewise, which is also
correct, but we cannot distinguish a cancel-and-repost from a genuine
departure. The error is second order compared to ignoring the queue entirely.

Latency
-------

A quote decided at time t is not live until t + latency. The collector
measured a p50 of about 80ms from Singapore to the Bitfinex matching engine,
which is the figure used by default. Ignoring latency lets the strategy react
to a book state it could not have acted on, which is a subtle form of
lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import SimulationConfig


@dataclass
class RestingOrder:
    """A simulated order resting in the book."""

    order_id: int
    price: float
    size: float
    is_bid: bool
    placed_at: float          # epoch seconds, decision time
    live_at: float            # epoch seconds, decision time + latency
    queue_ahead: float        # volume still in front of us at this level
    filled: float = 0.0

    @property
    def remaining(self) -> float:
        return max(self.size - self.filled, 0.0)

    @property
    def is_done(self) -> bool:
        return self.remaining <= 1e-12

    def is_live(self, now: float) -> bool:
        return now >= self.live_at


@dataclass
class Fill:
    """A simulated execution."""

    ts: float                 # epoch seconds
    price: float
    size: float               # always positive
    is_buy: bool              # True if we bought (our bid filled)
    order_id: int
    mid_at_fill: float
    queue_wait_seconds: float

    @property
    def signed_size(self) -> float:
        return self.size if self.is_buy else -self.size


@dataclass
class FillSimulator:
    """Tracks our resting orders and decides when they trade.

    Usage inside the replay loop:

        sim.place(...)                      # on a quote decision
        fills = sim.on_market_event(...)    # on every book/trade event
        sim.cancel_all()                    # before re-quoting

    The simulator never looks ahead: every method is driven by events at or
    before the current timestamp.
    """

    cfg: SimulationConfig
    _orders: dict[int, RestingOrder] = field(default_factory=dict)
    _next_id: int = 1
    n_placed: int = 0
    n_filled: int = 0
    n_cancelled: int = 0

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place(
        self,
        now: float,
        price: float,
        size: float,
        is_bid: bool,
        queue_ahead: float,
    ) -> int:
        """Place a simulated order.

        `queue_ahead` comes from the reconstructed book: the volume already
        resting at this price level. The inflation factor lets the assumption
        be stressed: setting it above 1 assumes we are further back than the
        book says, which is the conservative direction and a useful robustness
        check on whether results depend on optimistic queue placement.
        """
        if size <= 0:
            return -1

        oid = self._next_id
        self._next_id += 1
        self.n_placed += 1

        self._orders[oid] = RestingOrder(
            order_id=oid,
            price=price,
            size=size,
            is_bid=is_bid,
            placed_at=now,
            live_at=now + self.cfg.quote_latency_ms / 1000.0,
            queue_ahead=max(queue_ahead, 0.0) * self.cfg.queue_ahead_inflation,
        )
        return oid

    def cancel(self, order_id: int) -> None:
        if self._orders.pop(order_id, None) is not None:
            self.n_cancelled += 1

    def cancel_all(self) -> None:
        self.n_cancelled += len(self._orders)
        self._orders.clear()

    @property
    def open_orders(self) -> list[RestingOrder]:
        return list(self._orders.values())

    @property
    def n_open(self) -> int:
        return len(self._orders)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def on_book_removal(
        self,
        now: float,
        price: float,
        is_bid: bool,
        size_removed: float,
    ) -> None:
        """A resting order left the book at `price`.

        If it was in front of one of ours at the same level, our queue
        position improves. Cancels and trades are treated identically here,
        which is correct: either way the volume ahead of us is gone.
        """
        if size_removed <= 0:
            return

        for order in self._orders.values():
            if order.is_bid != is_bid or not np.isclose(order.price, price):
                continue
            if not order.is_live(now):
                continue
            order.queue_ahead = max(order.queue_ahead - size_removed, 0.0)

    def on_trade(
        self,
        now: float,
        trade_price: float,
        trade_size: float,
        aggressor_is_buy: bool,
        mid: float,
    ) -> list[Fill]:
        """A trade printed. Decide whether any of our orders participated.

        A buy aggressor lifts offers, so it can only fill our asks; a sell
        aggressor hits bids, so it can only fill our bids. The trade volume is
        consumed first by the queue ahead of us, and only the remainder
        reaches our order.
        """
        if trade_size <= 0:
            return []

        fills: list[Fill] = []
        # Our asks are filled by buy aggressors, our bids by sell aggressors.
        want_bid_side = not aggressor_is_buy

        # Price condition: a trade fills our order only if it reached our
        # price. For a bid, the trade must print at or below our price; for an
        # ask, at or above.
        candidates = [
            o for o in self._orders.values()
            if o.is_bid == want_bid_side
            and o.is_live(now)
            and not o.is_done
            and (
                (o.is_bid and trade_price <= o.price + 1e-9)
                or ((not o.is_bid) and trade_price >= o.price - 1e-9)
            )
        ]
        if not candidates:
            return []

        # Better prices trade first. Among our own orders that is the natural
        # priority; within a level, earlier placement.
        candidates.sort(
            key=lambda o: (-o.price if o.is_bid else o.price, o.placed_at)
        )

        remaining_volume = trade_size
        for order in candidates:
            if remaining_volume <= 1e-12:
                break

            if self.cfg.use_queue_model:
                # The queue in front of us absorbs volume first.
                absorbed = min(order.queue_ahead, remaining_volume)
                order.queue_ahead -= absorbed
                remaining_volume -= absorbed
                if order.queue_ahead > 1e-12 or remaining_volume <= 1e-12:
                    continue

            fill_size = min(order.remaining, remaining_volume)
            if fill_size <= 1e-12:
                continue

            order.filled += fill_size
            remaining_volume -= fill_size
            self.n_filled += 1

            fills.append(
                Fill(
                    ts=now,
                    price=order.price,
                    size=fill_size,
                    is_buy=order.is_bid,
                    order_id=order.order_id,
                    mid_at_fill=mid,
                    queue_wait_seconds=now - order.placed_at,
                )
            )

        # Drop anything fully executed.
        for oid in [o.order_id for o in self._orders.values() if o.is_done]:
            self._orders.pop(oid, None)

        return fills

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, float]:
        return {
            "n_placed": self.n_placed,
            "n_filled": self.n_filled,
            "n_cancelled": self.n_cancelled,
            "n_open": self.n_open,
            "fill_ratio": self.n_filled / self.n_placed if self.n_placed else 0.0,
        }


def compute_markouts(
    fills: list[Fill],
    mid_series: pd.DataFrame,
    horizons_ms: tuple[int, ...],
) -> pd.DataFrame:
    """Adverse-selection markouts for a set of fills.

    For each fill, look at where the mid went over the following horizon. A
    buy that is followed by the mid falling was a bad buy: someone with better
    information sold to us. Summed over all fills this is the adverse-
    selection cost, and it is the metric that separates a market maker who is
    earning the spread from one who is being run over.

    Sign convention: a positive markout is favourable to us. For a buy that
    means the mid rose after we bought.
    """
    if not fills or mid_series.empty:
        return pd.DataFrame()

    fill_df = pd.DataFrame([
        {
            "ts": pd.Timestamp(f.ts, unit="s"),
            "price": f.price,
            "size": f.size,
            "is_buy": f.is_buy,
            "mid_at_fill": f.mid_at_fill,
        }
        for f in fills
    ]).sort_values("ts").reset_index(drop=True)

    mids = mid_series.sort_values("ts").reset_index(drop=True)

    for h in horizons_ms:
        target = fill_df[["ts"]].copy()
        target["ts"] = target["ts"] + pd.Timedelta(milliseconds=h)
        joined = pd.merge_asof(
            target, mids[["ts", "mid"]], on="ts", direction="forward"
        )
        future_mid = joined["mid"].to_numpy(dtype=float)

        direction = np.where(fill_df["is_buy"].to_numpy(), 1.0, -1.0)
        # Markout relative to our fill price, in price units, signed so that
        # positive is favourable.
        fill_df[f"markout_{h}ms"] = direction * (
            future_mid - fill_df["price"].to_numpy(dtype=float)
        )

    return fill_df
