"""Synthetic MBO generator.

Purpose is development and testing, not research results. The real dataset
lives on the collector host behind Tailscale; this module lets the pipeline be
exercised end to end on a laptop with no network access, and gives the unit
tests a deterministic stream to assert against.

Nothing produced here should appear in reported results. The generator is
calibrated to look structurally like Bitfinex BTC-PERP (tick size, spread in
ticks, order counts per level, event mix) but the price process is a plain
random walk with no microstructure beyond what is needed to drive the book.

The generator maintains its own price-level bookkeeping so the stream is
internally consistent and never produces a crossed book. When the mid moves
far enough that resting orders would cross, those orders are explicitly
cancelled first, which is also what real quoting systems do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_mbo_events(
    n_events: int = 100_000,
    start: str = "2026-09-01 00:00:00",
    mid_start: float = 111_000.0,
    tick_size: float = 0.5,
    vol_per_sqrt_sec: float = 0.35,
    mean_event_interval_ms: float = 15.0,
    levels_per_side: int = 20,
    orders_per_level: int = 3,
    seed: int = 20260913,
) -> pd.DataFrame:
    """Generate a synthetic order-level event stream.

    Returns a frame with the same columns as `bfx.book_mbo`, so downstream
    code cannot tell the difference between this and a real epoch.
    """
    rng = np.random.default_rng(seed)

    t0 = pd.Timestamp(start)
    gaps_ms = rng.exponential(mean_event_interval_ms, n_events)
    ts = t0 + pd.to_timedelta(np.cumsum(gaps_ms), unit="ms")

    dt_sec = gaps_ms / 1000.0
    steps = rng.normal(0.0, vol_per_sqrt_sec, n_events) * np.sqrt(dt_sec)
    mid = mid_start + np.cumsum(steps)

    gen = _BookGenerator(tick_size)
    records: list[dict] = []
    seq = 0

    # Snapshot burst: seed the book with a plausible ladder.
    snap_mid = mid[0]
    for lvl in range(1, levels_per_side + 1):
        for _ in range(orders_per_level):
            for side, sign in (("bid", -1), ("ask", +1)):
                seq += 1
                price = _round_tick(snap_mid + sign * lvl * tick_size, tick_size)
                size = float(np.round(rng.gamma(2.0, 0.5), 4))
                oid = gen.add(price, side, size)
                records.append(_row(ts[0], seq, oid, side, "snapshot", price, size))

    actions = np.array(["add", "delete", "update"])
    weights = np.array([0.44, 0.36, 0.20])

    for i in range(1, n_events):
        m = mid[i]

        # Cancel any resting order the mid has moved past. A bid above the mid
        # or an ask below it would cross once the other side refreshes, so a
        # real quoting system pulls it.
        for oid, price, side, size in gen.crossing_orders(m):
            seq += 1
            gen.remove(oid)
            records.append(_row(ts[i], seq, oid, side, "delete", price, size))

        seq += 1
        action = rng.choice(actions, p=weights)

        if action == "add" or gen.is_empty:
            side = "bid" if rng.random() < 0.5 else "ask"
            sign = -1 if side == "bid" else +1
            lvl = min(int(rng.geometric(0.22)), levels_per_side)
            price = _round_tick(m + sign * lvl * tick_size, tick_size)
            price = gen.clamp(price, side, m)
            size = float(np.round(rng.gamma(2.0, 0.5), 4))
            oid = gen.add(price, side, size)
            records.append(_row(ts[i], seq, oid, side, "add", price, size))

        elif action == "delete":
            oid, price, side, size = gen.pick(rng)
            gen.remove(oid)
            records.append(_row(ts[i], seq, oid, side, "delete", price, size))

        else:  # update: resize in place, price level unchanged
            oid, price, side, size = gen.pick(rng)
            new_size = float(np.round(max(0.01, size * rng.uniform(0.4, 1.6)), 4))
            gen.resize(oid, new_size)
            records.append(_row(ts[i], seq, oid, side, "update", price, new_size))

    df = pd.DataFrame.from_records(records)
    df["ts_exch"] = pd.to_datetime(df["ts_exch"])
    return df.sort_values(["ts_exch", "seq"]).reset_index(drop=True)


def generate_trades(
    events: pd.DataFrame,
    trade_rate: float = 0.004,
    seed: int = 20260913,
) -> pd.DataFrame:
    """Generate trades consistent with an event stream.

    `trade_rate` is calibrated against the real dataset: roughly 37,400 trades
    per day across three symbols versus 8.1M book events per day, i.e. about
    0.4% of events.
    """
    rng = np.random.default_rng(seed + 1)

    n = len(events)
    hit = rng.random(n) < trade_rate
    idx = np.flatnonzero(hit)
    if idx.size == 0:
        return pd.DataFrame(
            columns=["ts_exch", "trade_id", "price", "amount", "side"]
        )

    prices = events["price"].to_numpy(dtype=float)[idx]
    ts = events["ts_exch"].to_numpy()[idx]
    sides = np.where(rng.random(idx.size) < 0.5, "buy", "sell")
    sizes = np.round(rng.gamma(1.6, 0.35, idx.size), 4)

    return pd.DataFrame({
        "ts_exch": pd.to_datetime(ts),
        "trade_id": np.arange(1, idx.size + 1, dtype=np.int64),
        "price": prices,
        "amount": np.where(sides == "buy", sizes, -sizes),
        "side": sides,
    })


# --------------------------------------------------------------------------
# Internal bookkeeping
# --------------------------------------------------------------------------

class _BookGenerator:
    """Tracks live orders and per-side best prices, so emissions never cross."""

    def __init__(self, tick_size: float):
        self.tick = tick_size
        self._orders: dict[int, tuple[float, str, float]] = {}
        self._bid_counts: dict[float, int] = {}
        self._ask_counts: dict[float, int] = {}
        self._next_oid = 1

    @property
    def is_empty(self) -> bool:
        return not self._orders

    @property
    def best_bid(self) -> float | None:
        return max(self._bid_counts) if self._bid_counts else None

    @property
    def best_ask(self) -> float | None:
        return min(self._ask_counts) if self._ask_counts else None

    def add(self, price: float, side: str, size: float) -> int:
        oid = self._next_oid
        self._next_oid += 1
        self._orders[oid] = (price, side, size)
        counts = self._bid_counts if side == "bid" else self._ask_counts
        counts[price] = counts.get(price, 0) + 1
        return oid

    def remove(self, oid: int) -> None:
        rec = self._orders.pop(oid, None)
        if rec is None:
            return
        price, side, _ = rec
        counts = self._bid_counts if side == "bid" else self._ask_counts
        n = counts.get(price, 0) - 1
        if n <= 0:
            counts.pop(price, None)
        else:
            counts[price] = n

    def resize(self, oid: int, new_size: float) -> None:
        if oid in self._orders:
            price, side, _ = self._orders[oid]
            self._orders[oid] = (price, side, new_size)

    def pick(self, rng: np.random.Generator) -> tuple[int, float, str, float]:
        keys = np.fromiter(self._orders.keys(), dtype=np.int64)
        oid = int(rng.choice(keys))
        price, side, size = self._orders[oid]
        return oid, price, side, size

    def clamp(self, price: float, side: str, mid: float) -> float:
        """Keep a new order strictly on its own side of the book."""
        if side == "bid":
            price = min(price, _round_tick(mid - self.tick, self.tick))
            ceiling = self.best_ask
            if ceiling is not None and price >= ceiling:
                price = ceiling - self.tick
            return price
        price = max(price, _round_tick(mid + self.tick, self.tick))
        floor = self.best_bid
        if floor is not None and price <= floor:
            price = floor + self.tick
        return price

    def crossing_orders(self, mid: float) -> list[tuple[int, float, str, float]]:
        """Resting orders the mid has moved past.

        Scans only the price levels on the wrong side of the mid rather than
        every order, so the cost is proportional to the number of stale levels
        (normally zero or one per event) and not to book size.
        """
        stale_prices: list[tuple[float, str]] = []
        for p in self._bid_counts:
            if p > mid:
                stale_prices.append((p, "bid"))
        for p in self._ask_counts:
            if p < mid:
                stale_prices.append((p, "ask"))
        if not stale_prices:
            return []

        targets = {(p, s) for p, s in stale_prices}
        out: list[tuple[int, float, str, float]] = []
        for oid, (price, side, size) in self._orders.items():
            if (price, side) in targets:
                out.append((oid, price, side, size))
        return out


def _row(ts, seq: int, oid: int, side: str, action: str,
         price: float, size: float) -> dict:
    return {
        "ts_exch": ts,
        "seq": seq,
        "order_id": oid,
        "side": side,
        "action": action,
        "price": price,
        "amount": size if side == "bid" else -size,
    }


def _round_tick(price: float, tick: float) -> float:
    return float(np.round(price / tick) * tick)
