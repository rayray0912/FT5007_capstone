"""Backtest engine.

Drives the replay: reconstruct the book event by event, wake the strategy on a
fixed decision clock, place and cancel quotes, and match incoming trades
against them.

Ordering within a timestamp matters and is deliberate:

  1. apply the book event, so the book reflects the market before we act
  2. update the queue position of our resting orders
  3. match trades against our orders
  4. only then, if the decision clock has ticked, re-quote

Re-quoting before matching would let the strategy cancel an order that was
about to be filled by a trade at the same timestamp, which is a lookahead
bug. Doing it in this order means our quote at time t is exposed to
everything that happens at time t.

Position and cash are tracked in contracts and quote currency. Mark-to-market
uses the mid, not our own quoted prices, so unrealised PnL is not a function
of how we happen to be quoting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..calibration.intensity import IntensityEstimate
from ..calibration.volatility import RollingVolatility
from ..config import Config
from ..data.book import OrderBook, _is_bid
from ..strategy.base import MarketState, Quoter
from .fills import Fill, FillSimulator

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Everything a run produces."""

    timeline: pd.DataFrame          # per-decision state
    fills: list[Fill]
    strategy_name: str
    config: dict
    fill_stats: dict

    @property
    def n_fills(self) -> int:
        return len(self.fills)

    def fills_frame(self) -> pd.DataFrame:
        if not self.fills:
            return pd.DataFrame(
                columns=["ts", "price", "size", "is_buy", "mid_at_fill",
                         "queue_wait_seconds"]
            )
        return pd.DataFrame([
            {
                "ts": pd.Timestamp(f.ts, unit="s"),
                "price": f.price,
                "size": f.size,
                "is_buy": f.is_buy,
                "signed_size": f.signed_size,
                "mid_at_fill": f.mid_at_fill,
                "queue_wait_seconds": f.queue_wait_seconds,
            }
            for f in self.fills
        ])


@dataclass
class _Position:
    """Running position and cash."""

    inventory: float = 0.0          # contracts, signed
    cash: float = 0.0               # quote currency
    fees_paid: float = 0.0
    realised_trades: int = 0

    def apply_fill(self, fill: Fill, maker_fee_bps: float) -> None:
        notional = fill.price * fill.size
        if fill.is_buy:
            self.inventory += fill.size
            self.cash -= notional
        else:
            self.inventory -= fill.size
            self.cash += notional

        # Maker fee, negative bps meaning a rebate. Charged on notional either
        # way, so a rebate credits cash.
        fee = notional * maker_fee_bps / 10_000.0
        self.cash -= fee
        self.fees_paid += fee
        self.realised_trades += 1

    def equity(self, mid: float) -> float:
        """Mark-to-market on the mid."""
        return self.cash + self.inventory * mid


def _aggressor_is_buy(trades: pd.DataFrame) -> pd.Series:
    """Which side initiated each trade.

    This decides everything: a buy aggressor can only lift our ask, a sell
    aggressor can only hit our bid. Get it wrong in one direction and the
    strategy can only ever sell.

    The `side` column is authoritative. The raw Bitfinex feed signs `amount`
    by aggressor direction, but the collector stores the magnitude and keeps
    the direction in `side`, so reading the sign of `amount` off the database
    returns True for every row -- every trade looks like a buy, our bids never
    fill, and the book runs one-way short. On a falling market that produces a
    healthy-looking profit which is purely the short position.

    The sign of `amount` is used only as a fallback, for the synthetic
    generator, which does sign it.
    """
    if "side" in trades.columns:
        side = trades["side"].astype(str).str.strip().str.lower()
        known = side.isin(("buy", "sell"))
        if known.all():
            return side.eq("buy")
        logger.warning(
            "trades.side has %d unrecognised values; falling back to the sign "
            "of amount for those rows", int((~known).sum()),
        )
        return side.eq("buy") | (~known & (trades["amount"].astype(float) > 0))

    logger.warning("trades frame has no `side` column; inferring aggressor "
                   "from the sign of amount")
    return trades["amount"].astype(float) > 0


class BacktestEngine:
    """Replays an event stream against a quoting strategy."""

    def __init__(
        self,
        cfg: Config,
        quoter: Quoter,
        intensity: IntensityEstimate,
    ):
        self.cfg = cfg
        self.quoter = quoter
        self.intensity = intensity

        self.book = OrderBook(tick_size=cfg.strategy.tick_size)
        self.fills_sim = FillSimulator(cfg.simulation)
        self.position = _Position()
        self.vol = RollingVolatility(
            cfg.calibration,
            sampling_interval_seconds=cfg.simulation.decision_interval_ms / 1000.0,
        )

        self._fills: list[Fill] = []
        self._timeline: list[dict] = []
        self._replay_start: float | None = None
        self._open_order_ids: list[int] = []

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        events: pd.DataFrame,
        trades: pd.DataFrame,
        progress_every: int | None = 1_000_000,
    ) -> BacktestResult:
        """Replay `events`, matching against `trades`.

        Both frames must be sorted by timestamp. Trades are merged into the
        event stream by timestamp so the two are processed in true order
        rather than in two passes.
        """
        merged = self._merge_streams(events, trades)
        logger.info("replaying %s combined events", f"{len(merged):,}")

        decision_dt = self.cfg.simulation.decision_interval_ms / 1000.0
        next_decision: float | None = None
        horizon = self.cfg.strategy.horizon_seconds

        kinds = merged["_kind"].to_numpy()
        ts_s = merged["_ts_seconds"].to_numpy(dtype=float)

        # Book columns (NaN on trade rows)
        oid = merged["order_id"].to_numpy()
        side = merged["side"].to_numpy()
        action = merged["action"].to_numpy()
        b_price = merged["price"].to_numpy(dtype=float)
        b_amount = merged["amount"].to_numpy(dtype=float)
        seq = merged["seq"].to_numpy()

        # Trade columns
        t_price = merged["trade_price"].to_numpy(dtype=float)
        t_size = merged["trade_size"].to_numpy(dtype=float)
        t_buy = merged["aggressor_is_buy"].to_numpy()

        prev_action = ""

        for i in range(len(merged)):
            now = ts_s[i]

            if kinds[i] == "book":
                act = str(action[i]).lower()
                # Snapshot bursts reset the book.
                if act == "snapshot" and prev_action != "snapshot":
                    self.book.clear()
                prev_action = act

                removed = self.book.apply(
                    order_id=int(oid[i]),
                    side=side[i],
                    action=act,
                    price=float(b_price[i]),
                    amount=float(b_amount[i]),
                    seq=int(seq[i]),
                )
                # Step 2: volume leaving the book improves our queue position.
                if removed > 0:
                    self.fills_sim.on_book_removal(
                        now=now,
                        price=float(b_price[i]),
                        is_bid=_is_bid(side[i]),
                        size_removed=removed,
                    )
            else:
                prev_action = ""
                mid = self.book.mid
                if mid is None:
                    continue
                # Step 3: match trades against our resting orders.
                new_fills = self.fills_sim.on_trade(
                    now=now,
                    trade_price=float(t_price[i]),
                    trade_size=float(t_size[i]),
                    aggressor_is_buy=bool(t_buy[i]),
                    mid=mid,
                )
                for f in new_fills:
                    self.position.apply_fill(f, self.cfg.simulation.maker_fee_bps)
                    self._fills.append(f)

            # Step 4: decision clock.
            mid = self.book.mid
            if mid is None:
                continue

            if next_decision is None:
                next_decision = now + decision_dt
                self.vol.update(pd.Timestamp(now, unit="s"), mid)
                continue

            if now >= next_decision:
                self.vol.update(pd.Timestamp(now, unit="s"), mid)
                self._decide(now, horizon)
                while next_decision <= now:
                    next_decision += decision_dt

            if progress_every and i and i % progress_every == 0:
                logger.info(
                    "  %s/%s events, inventory=%.2f, fills=%d",
                    f"{i:,}", f"{len(merged):,}",
                    self.position.inventory, len(self._fills),
                )

        return BacktestResult(
            timeline=pd.DataFrame(self._timeline),
            fills=self._fills,
            strategy_name=self.quoter.name,
            config=self.cfg.to_dict(),
            fill_stats=self.fills_sim.stats(),
        )

    # ------------------------------------------------------------------
    # Decision step
    # ------------------------------------------------------------------

    def _decide(self, now: float, horizon: float) -> None:
        """Cancel, re-quote, and record state."""
        book = self.book
        mid = book.mid
        bid, ask = book.best_bid, book.best_ask
        if mid is None or bid is None or ask is None:
            return

        # Book warm-up. Unless the window starts at a snapshot, the book is
        # rebuilt from the incremental stream and its touch is far too wide
        # until enough orders have arrived. Quoting into that fictitious
        # spread produces fills no real maker could have had, so we stay out
        # of the market and record nothing until it has settled. Events are
        # still consumed: the book fills and sigma warms up as normal.
        if self._replay_start is None:
            self._replay_start = now
        if now - self._replay_start < self.cfg.simulation.warmup_seconds:
            return

        # Do not quote until the volatility estimate has warmed up. Quoting on
        # a floor sigma would produce a spread that reflects nothing.
        if not self.vol.is_warm:
            self._record(now, mid, None, None)
            return

        state = MarketState(
            ts_seconds=now,
            mid=mid,
            best_bid=bid,
            best_ask=ask,
            microprice=book.microprice or mid,
            imbalance=book.imbalance(),
            sigma=self.vol.sigma,
            kappa=self.intensity.kappa,
            # GLFT's A is an arrival intensity, not the fitted density.
            # See IntensityEstimate.arrival_intensity.
            A=self.intensity.arrival_intensity,
            inventory=self.position.inventory,
            time_remaining=horizon,
        )

        quote = self.quoter.compute(state)

        # Quote persistence. Cancelling and reposting at the same price would
        # send us to the back of the queue, which is exactly the priority the
        # queue model says is valuable. A real quoting system amends only when
        # the desired price actually moves, so that is what happens here: an
        # order already resting at the wanted price is left alone, and only a
        # genuine price change triggers a cancel-replace.
        #
        # Without this the strategy destroys its own queue position every
        # decision tick, and with a decision interval near the quote latency
        # it would almost never be live long enough to trade.
        self._reconcile_side(now, book, True, quote.bid_price, quote.bid_size)
        self._reconcile_side(now, book, False, quote.ask_price, quote.ask_size)

        self._record(now, mid, quote.bid_price, quote.ask_price, state, quote)

    def _reconcile_side(
        self,
        now: float,
        book: OrderBook,
        is_bid: bool,
        wanted_price: float | None,
        wanted_size: float,
    ) -> None:
        """Bring one side of our quote in line with what the strategy wants."""
        existing = [o for o in self.fills_sim.open_orders if o.is_bid == is_bid]

        if wanted_price is None or wanted_size <= 0:
            for o in existing:
                self.fills_sim.cancel(o.order_id)
            return

        # Already resting at the wanted price: keep it, and keep its queue
        # position with it.
        for o in existing:
            if np.isclose(o.price, wanted_price):
                return

        for o in existing:
            self.fills_sim.cancel(o.order_id)

        self.fills_sim.place(
            now=now,
            price=wanted_price,
            size=wanted_size,
            is_bid=is_bid,
            queue_ahead=book.depth(is_bid, wanted_price),
        )

    def _record(
        self,
        now: float,
        mid: float,
        bid_px: float | None,
        ask_px: float | None,
        state: MarketState | None = None,
        quote=None,
    ) -> None:
        row = {
            "ts": pd.Timestamp(now, unit="s"),
            "mid": mid,
            "best_bid": self.book.best_bid,
            "best_ask": self.book.best_ask,
            "market_spread": self.book.spread,
            "inventory": self.position.inventory,
            "cash": self.position.cash,
            "equity": self.position.equity(mid),
            "quote_bid": bid_px,
            "quote_ask": ask_px,
            "n_fills_cum": len(self._fills),
            "fees_paid": self.position.fees_paid,
        }
        if state is not None:
            row["sigma"] = state.sigma
            row["imbalance"] = state.imbalance
        if quote is not None:
            row["reservation_price"] = quote.reservation_price
            row["half_spread"] = quote.half_spread
            row["quoted_spread"] = quote.quoted_spread
        self._timeline.append(row)

    # ------------------------------------------------------------------
    # Stream merging
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_streams(events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        """Interleave book events and trades into one time-ordered stream."""
        ev = events.copy()
        ev["_kind"] = "book"
        ev["_ts"] = pd.to_datetime(ev["ts_exch"])
        ev["trade_price"] = np.nan
        ev["trade_size"] = np.nan
        ev["aggressor_is_buy"] = False

        if trades is not None and not trades.empty:
            tr = pd.DataFrame({
                "_ts": pd.to_datetime(trades["ts_exch"]),
                "_kind": "trade",
                "trade_price": trades["price"].astype(float),
                "trade_size": trades["amount"].abs().astype(float),
                "aggressor_is_buy": _aggressor_is_buy(trades),
                "order_id": 0,
                "side": "bid",
                "action": "trade",
                "price": np.nan,
                "amount": np.nan,
                "seq": 0,
            })
            merged = pd.concat([ev, tr], ignore_index=True)
        else:
            merged = ev

        merged = merged.sort_values(["_ts", "_kind"], kind="stable")

        # Normalise to nanoseconds before converting. `astype("int64")` on a
        # datetime column returns whatever the column's *resolution* is, not
        # nanoseconds, and pandas >= 2 preserves the source resolution instead
        # of coercing everything to ns.
        #
        # ClickHouse hands back DateTime64(3), i.e. datetime64[ms], so the old
        # `astype("int64") / 1e9` divided milliseconds by a nanosecond scale
        # and produced timestamps 1e6 too small: a two-hour replay spanned
        # 0.0072 "seconds", the 100 ms decision clock never came due, and the
        # engine placed zero orders over the entire window.
        #
        # It only ever bit on real data. The synthetic generator builds its
        # index with pd.to_timedelta, which is datetime64[ns], so the whole
        # test suite and every offline run took the correct branch by accident.
        merged["_ts_seconds"] = (
            merged["_ts"].dt.as_unit("ns").astype("int64") / 1e9
        )
        return merged.reset_index(drop=True)
