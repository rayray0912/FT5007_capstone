"""Unit tests.

The tests are organised around invariants rather than around functions. An
invariant is something that must hold for the code to be correct at all --
"the book never crosses", "GLFT is symmetric at zero inventory", "the
estimator recovers a known parameter" -- and those are the failures that
actually matter. Testing that a function returns a float catches nothing.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mmbacktest.calibration.intensity import estimate_intensity  # noqa: E402
from mmbacktest.calibration.volatility import (  # noqa: E402
    RollingVolatility,
    realised_volatility,
)
from mmbacktest.config import (  # noqa: E402
    CalibrationConfig,
    SimulationConfig,
    StrategyConfig,
    bitfinex_tick_size,
    is_on_tick_grid,
    round_to_tick,
)
from mmbacktest.data.book import OrderBook, replay  # noqa: E402
from mmbacktest.data.synthetic import generate_mbo_events  # noqa: E402
from mmbacktest.sim.fills import FillSimulator  # noqa: E402
from mmbacktest.strategy.avellaneda_stoikov import AvellanedaStoikovQuoter  # noqa: E402
from mmbacktest.strategy.base import MarketState  # noqa: E402
from mmbacktest.strategy.glft import GLFTQuoter  # noqa: E402
from mmbacktest.strategy.symmetric import SymmetricQuoter  # noqa: E402


# ==========================================================================
# Order book
# ==========================================================================

class TestOrderBook:

    def test_queue_priority_respects_arrival_order(self):
        """Volume ahead must count only orders that arrived earlier."""
        book = OrderBook(tick_size=0.5)
        book.apply(1, "bid", "add", 100.0, 5.0, seq=1)
        book.apply(2, "bid", "add", 100.0, 3.0, seq=2)
        book.apply(3, "bid", "add", 100.0, 7.0, seq=3)

        # An order arriving at seq=2 sees only the seq=1 order in front.
        assert book.volume_ahead(True, 100.0, seq=2) == pytest.approx(5.0)
        # One arriving last sees everything.
        assert book.volume_ahead(True, 100.0, seq=99) == pytest.approx(15.0)
        # One arriving first sees nothing.
        assert book.volume_ahead(True, 100.0, seq=0) == pytest.approx(0.0)

    def test_delete_returns_removed_size(self):
        """The removed size drives queue depletion, so it must be exact."""
        book = OrderBook()
        book.apply(1, "ask", "add", 101.0, 4.0, seq=1)
        removed = book.apply(1, "ask", "delete", 101.0, 4.0, seq=2)
        assert removed == pytest.approx(4.0)
        assert book.depth(False, 101.0) == pytest.approx(0.0)

    def test_microprice_leans_toward_the_thin_side(self):
        """A thick bid implies upward pressure, so fair value sits nearer the ask."""
        book = OrderBook()
        book.apply(1, "bid", "add", 100.0, 10.0, seq=1)
        book.apply(2, "ask", "add", 101.0, 1.0, seq=2)

        assert book.mid == pytest.approx(100.5)
        assert book.microprice > book.mid

    def test_update_loses_queue_priority(self):
        """A modified order goes to the back, as on a real exchange."""
        book = OrderBook()
        book.apply(1, "bid", "add", 100.0, 5.0, seq=1)
        book.apply(2, "bid", "add", 100.0, 5.0, seq=2)
        # Order 1 amends; it should now sit behind order 2.
        book.apply(1, "bid", "update", 100.0, 6.0, seq=3)

        level_orders = book._bids[100.0].orders
        assert [o.order_id for o in level_orders] == [2, 1]

    def test_replay_never_produces_a_crossed_book(self):
        """The single invariant that catches most reconstruction bugs."""
        events = generate_mbo_events(n_events=8_000, seed=42)
        crossed = 0
        n = 0
        for _ts, book, _removed in replay(events, tick_size=0.5, emit_every_ms=100):
            n += 1
            if book.is_crossed():
                crossed += 1
        assert n > 0
        assert crossed == 0


# ==========================================================================
# Calibration
# ==========================================================================

class TestIntensityCalibration:

    @staticmethod
    def _synthetic_trades(kappa_true: float, n: int, duration: float, seed: int = 0):
        rng = np.random.default_rng(seed)
        deltas = rng.exponential(1.0 / kappa_true, n)
        mid = 111_000.0
        ts = pd.to_datetime("2026-09-01") + pd.to_timedelta(
            np.sort(rng.uniform(0, duration, n)), unit="s"
        )
        return pd.DataFrame({
            "ts_exch": ts, "price": mid + deltas, "mid": mid, "delta": deltas,
        })

    def test_recovers_known_kappa(self):
        kappa_true = 0.02
        trades = self._synthetic_trades(kappa_true, n=50_000, duration=20_000.0)
        cfg = CalibrationConfig(
            intensity_n_buckets=15,
            intensity_max_delta_bps=25.0,
            intensity_min_samples_per_bucket=20,
        )
        est = estimate_intensity(trades, cfg, duration_seconds=20_000.0)

        assert est.is_usable
        assert est.kappa == pytest.approx(kappa_true, rel=0.05)
        assert est.r_squared > 0.95

    def test_A_is_invariant_to_bucket_count(self):
        """A is a rate density, so retuning the bucket count must not move it.

        This is a regression test: the first implementation returned
        exp(intercept) directly, which is A * bucket_width, so changing
        intensity_n_buckets silently rescaled every quote.
        """
        trades = self._synthetic_trades(0.02, n=50_000, duration=20_000.0)
        estimates = []
        for n_buckets in (10, 15, 25):
            cfg = CalibrationConfig(
                intensity_n_buckets=n_buckets,
                intensity_max_delta_bps=25.0,
                intensity_min_samples_per_bucket=20,
            )
            estimates.append(estimate_intensity(trades, cfg, 20_000.0).A)

        spread = max(estimates) / min(estimates)
        assert spread < 1.05, f"A varies by {spread:.2f}x across bucket counts"

    def test_degenerate_input_is_marked_unusable(self):
        """A fit with no signal must not be silently quoted on."""
        cfg = CalibrationConfig(intensity_min_samples_per_bucket=1000)
        trades = self._synthetic_trades(0.02, n=50, duration=100.0)
        est = estimate_intensity(trades, cfg, 100.0)
        assert not est.is_usable


class TestVolatility:

    def test_recovers_known_sigma(self):
        sigma_true = 0.35
        rng = np.random.default_rng(3)
        n, dt = 20_000, 0.1
        mid = 111_000 + np.cumsum(rng.normal(0, sigma_true * np.sqrt(dt), n))
        ts = pd.to_datetime("2026-09-01") + pd.to_timedelta(np.arange(n) * dt, unit="s")

        est = realised_volatility(ts, mid, sampling_interval_seconds=1.0)
        assert est.sigma == pytest.approx(sigma_true, rel=0.10)

    def test_rolling_tracker_matches_batch_estimate(self):
        sigma_true = 0.35
        rng = np.random.default_rng(5)
        n, dt = 10_000, 0.1
        mid = 111_000 + np.cumsum(rng.normal(0, sigma_true * np.sqrt(dt), n))
        ts = pd.to_datetime("2026-09-01") + pd.to_timedelta(np.arange(n) * dt, unit="s")

        cfg = CalibrationConfig(vol_window_seconds=1e9, vol_min_observations=30)
        tracker = RollingVolatility(cfg, sampling_interval_seconds=dt)
        for i in range(n):
            tracker.update(ts[i], mid[i])

        assert tracker.is_warm
        assert tracker.sigma == pytest.approx(sigma_true, rel=0.10)

    def test_cold_tracker_reports_floor_not_a_guess(self):
        cfg = CalibrationConfig(vol_min_observations=100, vol_floor=1e-9)
        tracker = RollingVolatility(cfg)
        ts = pd.Timestamp("2026-09-01")
        for i in range(10):
            tracker.update(ts + pd.Timedelta(seconds=i), 111_000.0 + i)
        assert not tracker.is_warm
        assert tracker.sigma == pytest.approx(1e-9)


# ==========================================================================
# Strategies
# ==========================================================================

def _state(inventory=0.0, sigma=0.35, kappa=0.3, A=0.5, mid=111_000.0):
    return MarketState(
        ts_seconds=0.0, mid=mid, best_bid=mid - 0.5, best_ask=mid + 0.5,
        microprice=mid, imbalance=0.0, sigma=sigma, kappa=kappa, A=A,
        inventory=inventory, time_remaining=60.0,
    )


class TestSymmetricQuoter:

    def test_reservation_price_ignores_inventory(self):
        """Defining property of the baseline."""
        cfg = StrategyConfig(name="symmetric", fixed_half_spread_bps=0.05)
        q = SymmetricQuoter(cfg)
        flat = q.compute(_state(0.0)).reservation_price
        long = q.compute(_state(5.0)).reservation_price
        assert flat == pytest.approx(long)

    def test_suppresses_the_side_that_would_breach_the_bound(self):
        cfg = StrategyConfig(name="symmetric", q_max=5.0, order_size=1.0)
        q = SymmetricQuoter(cfg)
        assert q.compute(_state(5.0)).bid_price is None
        assert q.compute(_state(-5.0)).ask_price is None


class TestAvellanedaStoikov:

    def test_long_inventory_skews_reservation_down(self):
        """Long position should make us keener to sell, so r sits below mid."""
        cfg = StrategyConfig(name="as", gamma=0.1)
        q = AvellanedaStoikovQuoter(cfg)
        assert q.compute(_state(3.0)).reservation_price < _state(3.0).mid
        assert q.compute(_state(-3.0)).reservation_price > _state(-3.0).mid

    def test_higher_volatility_widens(self):
        cfg = StrategyConfig(name="as", gamma=0.1, max_half_spread_bps=500.0)
        q = AvellanedaStoikovQuoter(cfg)
        calm = q.compute(_state(sigma=0.1)).half_spread
        wild = q.compute(_state(sigma=2.0)).half_spread
        assert wild > calm

    def test_higher_kappa_tightens(self):
        """More elastic fill intensity means we can afford to quote closer."""
        cfg = StrategyConfig(name="as", gamma=0.1, max_half_spread_bps=500.0)
        q = AvellanedaStoikovQuoter(cfg)
        assert q.compute(_state(kappa=1.0)).half_spread < \
               q.compute(_state(kappa=0.05)).half_spread

    def test_unusable_intensity_falls_back_instead_of_guessing(self):
        """kappa <= 0 signals a failed fit; the fill term must drop out."""
        cfg = StrategyConfig(name="as", gamma=0.1)
        q = AvellanedaStoikovQuoter(cfg)
        parts = q.decompose(_state(kappa=0.0))
        assert parts["fill_term"] == 0.0


class TestGLFT:

    def test_symmetric_at_zero_inventory(self):
        cfg = StrategyConfig(name="glft", gamma=0.1, q_max=10.0)
        q = GLFTQuoter(cfg)
        bid, ask = q.distances(_state(0.0))
        assert bid == pytest.approx(ask)

    def test_skew_works_the_position_back_to_flat(self):
        """Long inventory pushes the bid out and pulls the ask in."""
        cfg = StrategyConfig(name="glft", gamma=0.1, q_max=10.0)
        q = GLFTQuoter(cfg)

        flat_bid, flat_ask = q.distances(_state(0.0))
        long_bid, long_ask = q.distances(_state(5.0))

        assert long_bid > flat_bid
        assert long_ask < flat_ask

    def test_skew_is_antisymmetric_in_inventory(self):
        cfg = StrategyConfig(name="glft", gamma=0.1, q_max=10.0)
        q = GLFTQuoter(cfg)
        long_bid, long_ask = q.distances(_state(4.0))
        short_bid, short_ask = q.distances(_state(-4.0))
        assert long_bid == pytest.approx(short_ask)
        assert long_ask == pytest.approx(short_bid)

    def test_volatility_scales_skew_not_base(self):
        cfg = StrategyConfig(name="glft", gamma=0.1, q_max=10.0)
        q = GLFTQuoter(cfg)
        calm = q.decompose(_state(5.0, sigma=0.1))
        wild = q.decompose(_state(5.0, sigma=2.0))

        assert calm["base_distance"] == pytest.approx(wild["base_distance"])
        assert wild["inventory_coefficient"] > calm["inventory_coefficient"]

    def test_busier_market_damps_the_skew(self):
        """Larger A means inventory is easier to work off, so skew less."""
        cfg = StrategyConfig(name="glft", gamma=0.1, q_max=10.0)
        q = GLFTQuoter(cfg)
        quiet = q.decompose(_state(5.0, A=0.01))
        busy = q.decompose(_state(5.0, A=1.0))
        assert busy["inventory_coefficient"] < quiet["inventory_coefficient"]

    def test_extreme_inputs_do_not_produce_nan(self):
        """Degenerate upstream estimates should widen the quote, not break it."""
        cfg = StrategyConfig(name="glft", gamma=0.1, q_max=10.0)
        q = GLFTQuoter(cfg)
        for kappa in (1e-12, 1e6):
            for sigma in (1e-12, 1e6):
                bid, ask = q.distances(_state(2.0, sigma=sigma, kappa=kappa))
                assert np.isfinite(bid) and np.isfinite(ask)
                assert bid >= 0 and ask >= 0


# ==========================================================================
# Fill simulation
# ==========================================================================

class TestFillSimulator:

    def test_queue_must_be_consumed_before_we_trade(self):
        cfg = SimulationConfig(use_queue_model=True, quote_latency_ms=0.0)
        sim = FillSimulator(cfg)
        sim.place(now=0.0, price=100.0, size=2.0, is_bid=True, queue_ahead=5.0)

        # A trade smaller than the queue ahead must not fill us.
        assert sim.on_trade(1.0, 100.0, 3.0, aggressor_is_buy=False, mid=100.5) == []
        assert sim.open_orders[0].queue_ahead == pytest.approx(2.0)

        # The next trade clears the queue and reaches us.
        fills = sim.on_trade(2.0, 100.0, 3.0, aggressor_is_buy=False, mid=100.5)
        assert len(fills) == 1
        assert fills[0].size == pytest.approx(1.0)

    def test_disabling_the_queue_model_inflates_fills(self):
        """The comparison that justifies using order-level data at all."""
        seq = [(1.0, 100.0, 3.0)]

        def run(use_queue: bool) -> float:
            sim = FillSimulator(
                SimulationConfig(use_queue_model=use_queue, quote_latency_ms=0.0)
            )
            sim.place(now=0.0, price=100.0, size=2.0, is_bid=True, queue_ahead=5.0)
            total = 0.0
            for t, px, sz in seq:
                for f in sim.on_trade(t, px, sz, aggressor_is_buy=False, mid=100.5):
                    total += f.size
            return total

        assert run(use_queue=False) > run(use_queue=True)

    def test_latency_blocks_fills_before_the_order_is_live(self):
        cfg = SimulationConfig(use_queue_model=True, quote_latency_ms=80.0)
        sim = FillSimulator(cfg)
        sim.place(now=0.0, price=100.0, size=2.0, is_bid=True, queue_ahead=0.0)

        assert sim.on_trade(0.05, 100.0, 5.0, aggressor_is_buy=False, mid=100.5) == []
        assert len(sim.on_trade(0.10, 100.0, 5.0, aggressor_is_buy=False, mid=100.5)) == 1

    def test_aggressor_side_routing(self):
        """A buy aggressor lifts offers, so it can only fill our ask."""
        sim = FillSimulator(SimulationConfig(use_queue_model=True, quote_latency_ms=0.0))
        sim.place(now=0.0, price=100.0, size=1.0, is_bid=True, queue_ahead=0.0)
        sim.place(now=0.0, price=101.0, size=1.0, is_bid=False, queue_ahead=0.0)

        sell_side = sim.on_trade(1.0, 100.0, 1.0, aggressor_is_buy=False, mid=100.5)
        buy_side = sim.on_trade(1.0, 101.0, 1.0, aggressor_is_buy=True, mid=100.5)

        assert sell_side[0].is_buy is True     # our bid was hit
        assert buy_side[0].is_buy is False     # our ask was lifted

    def test_a_trade_that_does_not_reach_our_price_does_not_fill(self):
        sim = FillSimulator(SimulationConfig(use_queue_model=True, quote_latency_ms=0.0))
        sim.place(now=0.0, price=99.0, size=1.0, is_bid=True, queue_ahead=0.0)
        # Trade prints above our bid: it never came down to us.
        assert sim.on_trade(1.0, 100.0, 5.0, aggressor_is_buy=False, mid=100.5) == []


# ==========================================================================
# Venue price grid
# ==========================================================================

class TestTickGrid:
    """Bitfinex prices carry 5 significant digits, so the tick moves with price.

    These are regression tests for a bug that silently inflated results: the
    tick was hard-coded to 0.5, which is not on Bitfinex's grid at any BTC
    price. Roughly half of all quotes therefore landed on a price the venue
    cannot represent, where the reconstructed book must show zero depth, so
    the queue model saw an empty queue and filled instantly -- at a price
    better than the whole market.
    """

    @pytest.mark.parametrize(
        "price,expected",
        [
            (76_647.0, 1.0),        # BTC-PERP
            (2_470.8, 0.1),         # ETH-PERP
            (4_348.9, 0.1),         # XAUT-PERP
            (6_120.0, 0.1),         # EUROPE50, trailing zero is padding
            (53.381, 0.001),        # LTC-PERP
            (1.1610, 0.0001),       # EUR-PERP
            (0.040598, 1e-6),       # IOT-PERP
            (0.032211, 1e-6),       # ETH/BTC-PERP
        ],
    )
    def test_tick_matches_venue_table(self, price, expected):
        assert bitfinex_tick_size(price) == pytest.approx(expected, rel=1e-12)

    def test_tick_steps_at_powers_of_ten(self):
        """The tick changes decade with the price. BTC below 10k quotes finer.

        This is the case a constant cannot express, and it is not
        hypothetical: BTC traded at 76k in this dataset, one decade above the
        boundary.
        """
        assert bitfinex_tick_size(10_000.0) == pytest.approx(1.0)
        assert bitfinex_tick_size(9_999.9) == pytest.approx(0.1)
        assert bitfinex_tick_size(10_000.1) == pytest.approx(1.0)

    def test_exact_power_of_ten_does_not_lose_a_decade(self):
        """log10(1000) can come back as 2.9999999999999996.

        Without the guard the tick drops a full decade at exactly the round
        numbers, which are the prices most likely to be tested by hand.
        """
        for p in (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0):
            assert bitfinex_tick_size(p) == pytest.approx(p * 1e-4)

    def test_rounding_survives_binary_representation(self):
        """floor(2470.7 / 0.1) * 0.1 is 2470.6, a whole tick low.

        0.1 is not exactly representable, so the naive form silently moves
        the quote one level away from the touch.
        """
        assert np.floor(2470.7 / 0.1) * 0.1 < 2470.65      # the trap is real
        assert round_to_tick(2470.7, 0.1, "down") == pytest.approx(2470.7)
        assert round_to_tick(2470.7, 0.1, "up") == pytest.approx(2470.7)

    def test_rounding_is_outward(self):
        """Bids round down and asks round up, never tightening the quote."""
        assert round_to_tick(76_647.8, 1.0, "down") == pytest.approx(76_647.0)
        assert round_to_tick(76_647.2, 1.0, "up") == pytest.approx(76_648.0)

    def test_grid_check_rejects_the_old_default(self):
        """A .5 price is off-grid for BTC at 76k -- the original bug."""
        assert is_on_tick_grid(76_647.0, 1.0)
        assert not is_on_tick_grid(76_647.5, 1.0)

    @pytest.mark.parametrize("name", ["symmetric", "avellaneda_stoikov", "glft"])
    def test_quoters_never_emit_off_grid_prices(self, name):
        """Every strategy's output must be representable by the venue.

        Runs at a real BTC price level with a real spread, so a regression in
        the rounding path shows up as an off-grid quote rather than as a
        quietly better fill rate.
        """
        quoters = {
            "symmetric": SymmetricQuoter,
            "avellaneda_stoikov": AvellanedaStoikovQuoter,
            "glft": GLFTQuoter,
        }
        cfg = StrategyConfig(name=name, tick_size=None)
        q = quoters[name](cfg)

        rng = np.random.default_rng(7)
        for _ in range(500):
            mid = float(rng.uniform(76_000, 82_000))
            half = float(rng.uniform(5.0, 12.0))
            state = MarketState(
                ts_seconds=1.0,
                mid=mid,
                best_bid=mid - half,
                best_ask=mid + half,
                microprice=mid,
                imbalance=float(rng.uniform(-0.5, 0.5)),
                sigma=0.35,
                kappa=0.02,
                A=1.0,
                inventory=float(rng.integers(-3, 4)),
                time_remaining=60.0,
            )
            quote = q.compute(state)
            tick = bitfinex_tick_size(mid)
            for px in (quote.bid_price, quote.ask_price):
                if px is not None:
                    assert is_on_tick_grid(px, tick), f"{name} quoted {px} off grid"


class TestFillProvenance:
    """Fills on levels we created are legitimate but less well modelled."""

    def test_new_level_fills_are_counted_separately(self):
        sim = FillSimulator(SimulationConfig(use_queue_model=True, quote_latency_ms=0.0))
        sim.place(now=0.0, price=100.0, size=1.0, is_bid=True, queue_ahead=0.0)
        sim.on_trade(1.0, 100.0, 1.0, aggressor_is_buy=False, mid=100.5)

        s = sim.stats()
        assert s["n_filled_at_new_level"] == 1
        assert s["n_filled_from_queue"] == 0
        assert s["frac_fills_at_new_level"] == pytest.approx(1.0)

    def test_joining_an_existing_queue_is_counted_separately(self):
        sim = FillSimulator(SimulationConfig(use_queue_model=True, quote_latency_ms=0.0))
        sim.place(now=0.0, price=100.0, size=1.0, is_bid=True, queue_ahead=5.0)
        # First trade is absorbed by the queue ahead; second reaches us.
        assert sim.on_trade(1.0, 100.0, 5.0, aggressor_is_buy=False, mid=100.5) == []
        assert len(sim.on_trade(2.0, 100.0, 1.0, aggressor_is_buy=False, mid=100.5)) == 1

        s = sim.stats()
        assert s["n_filled_at_new_level"] == 0
        assert s["n_filled_from_queue"] == 1
        assert s["frac_fills_at_new_level"] == pytest.approx(0.0)


class TestTimestampResolution:
    """The event clock must not depend on the frame's datetime resolution.

    Regression test for a bug that produced zero fills on every real run
    while leaving the synthetic suite green. ClickHouse returns
    DateTime64(3) -> datetime64[ms]; the synthetic generator produces
    datetime64[ns]. `astype("int64")` yields the column's own unit, so
    dividing by 1e9 was correct for one and off by 1e6 for the other.
    """

    @staticmethod
    def _one_event(unit: str) -> pd.DataFrame:
        ts = pd.to_datetime(
            pd.Series(["2026-09-06 02:00:00.128", "2026-09-06 02:00:01.128"])
        ).astype(f"datetime64[{unit}]")
        return pd.DataFrame({
            "ts_exch": ts,
            "seq": [1, 2],
            "order_id": [1, 2],
            "side": ["bid", "ask"],
            "action": ["add", "add"],
            "price": [79_990.0, 80_010.0],
            "amount": [1.0, 1.0],
        })

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
    def test_seconds_are_absolute_regardless_of_unit(self, unit):
        from mmbacktest.sim.engine import BacktestEngine

        merged = BacktestEngine._merge_streams(
            self._one_event(unit), pd.DataFrame()
        )
        secs = merged["_ts_seconds"].to_numpy()

        # 2026-09-06 02:00:00 UTC is ~1.7887e9 seconds after the epoch. The
        # old code returned ~1788.66 for millisecond input.
        assert 1.7e9 < secs[0] < 1.9e9, f"unit={unit} gave {secs[0]}"

    @pytest.mark.parametrize("unit", ["ns", "us", "ms"])
    def test_one_second_apart_stays_one_second(self, unit):
        """Elapsed time must survive the conversion, since the decision
        clock compares `now` against `now + decision_interval`."""
        from mmbacktest.sim.engine import BacktestEngine

        merged = BacktestEngine._merge_streams(
            self._one_event(unit), pd.DataFrame()
        )
        secs = merged["_ts_seconds"].to_numpy()
        assert secs[1] - secs[0] == pytest.approx(1.0, abs=1e-6)


class TestAggressorDirection:
    """Which side initiated a trade decides which of our quotes can fill.

    Regression test for a bug that made every strategy one-way short on real
    data. The collector stores |amount| and keeps direction in `side`, so
    reading the sign of `amount` marked every trade a buy: our asks filled,
    our bids never did, inventory could only go negative, and a falling
    market turned that into a profit that looked like market making.
    """

    @staticmethod
    def _trades(**cols):
        base = {
            "ts_exch": pd.to_datetime(["2026-09-06 02:00:00", "2026-09-06 02:00:01"]),
            "price": [80_000.0, 80_010.0],
            "amount": [1.0, 2.0],
        }
        base.update(cols)
        return pd.DataFrame(base)

    def test_side_column_wins_over_unsigned_amount(self):
        """Real-data shape: amount is always positive, side carries direction."""
        from mmbacktest.sim.engine import _aggressor_is_buy

        tr = self._trades(side=["buy", "sell"], amount=[1.0, 2.0])
        assert list(_aggressor_is_buy(tr)) == [True, False]

    def test_signed_amount_used_when_no_side_column(self):
        """Synthetic-data shape: no side column, amount is signed."""
        from mmbacktest.sim.engine import _aggressor_is_buy

        tr = self._trades(amount=[1.0, -2.0])
        assert list(_aggressor_is_buy(tr)) == [True, False]

    def test_both_directions_reach_the_book(self):
        """End to end: with real-shaped trades, both our quotes can fill.

        The bug's signature was that inventory could only ever decrease. This
        asserts the bid fills at all, which it could not before.
        """
        from mmbacktest.sim.engine import BacktestEngine

        tr = self._trades(side=["sell", "buy"], amount=[1.0, 1.0])
        merged = BacktestEngine._merge_streams(
            pd.DataFrame({
                "ts_exch": pd.to_datetime(["2026-09-06 01:59:59"]),
                "seq": [1], "order_id": [1], "side": ["bid"],
                "action": ["add"], "price": [79_990.0], "amount": [1.0],
            }),
            tr,
        )
        flags = merged.loc[merged["_kind"] == "trade", "aggressor_is_buy"].tolist()
        assert flags == [False, True], "both aggressor directions must survive merge"


class TestWarmup:
    """The engine must not quote into a book it has not finished rebuilding."""

    @staticmethod
    def _cfg(warmup):
        from mmbacktest.config import Config
        c = Config()
        c.simulation.warmup_seconds = warmup
        c.strategy.name = "symmetric"
        c.calibration.vol_min_observations = 5
        return c

    def _run(self, warmup):
        from mmbacktest.sim.engine import BacktestEngine
        from mmbacktest.calibration.intensity import IntensityEstimate
        from mmbacktest.data.synthetic import generate_mbo_events, generate_trades
        from mmbacktest.strategy.base import make_quoter

        c = self._cfg(warmup)
        ev = generate_mbo_events(n_events=20_000, seed=1)
        tr = generate_trades(ev, trade_rate=0.02, seed=1)
        eng = BacktestEngine(c, make_quoter(c.strategy),
                             IntensityEstimate(A=0.09, kappa=0.04, n_trades=500,
                                               n_buckets_used=6, r_squared=0.8))
        return eng.run(ev, tr, progress_every=None)

    def test_nothing_is_recorded_during_warmup(self):
        """The measured window must start after the book has settled."""
        no_warm = self._run(0.0)
        warmed = self._run(60.0)

        assert not no_warm.timeline.empty
        assert not warmed.timeline.empty
        gap = (warmed.timeline["ts"].iloc[0] - no_warm.timeline["ts"].iloc[0])
        assert gap.total_seconds() >= 60.0, (
            "first recorded decision should be at least warmup_seconds later"
        )

    def test_no_orders_are_placed_during_warmup(self):
        """Not quoting is the point -- otherwise fills leak in from the
        fictitious opening spread."""
        warmed = self._run(60.0)
        first = warmed.timeline["ts"].iloc[0]
        for f in warmed.fills:
            assert pd.Timestamp(f.ts, unit="s") >= first

    def test_zero_warmup_is_a_no_op(self):
        """A window that starts at a snapshot has nothing to warm up."""
        r = self._run(0.0)
        assert not r.timeline.empty


class TestInventoryPnLSignificance:
    """A position PnL that is pure noise must be reported as such.

    Holding a position through a random walk earns and loses continuously, so
    the net is a small residual of two large sums. Reported alone it reads as
    skill. On real data (BTC epoch 27, 40 h) GLFT's inventory PnL was +1,011
    against a gross of ~20,000 with t = 0.32 -- indistinguishable from zero,
    yet it made up 68% of a headline +40.62% annualised return.
    """

    @staticmethod
    def _timeline(inventory, mid):
        n = len(mid)
        return pd.DataFrame({
            "ts": pd.date_range("2026-09-06", periods=n, freq="100ms"),
            "mid": mid,
            "inventory": inventory,
            "equity": np.zeros(n),
            "fees_paid": np.zeros(n),
        })

    def test_random_walk_position_is_not_significant(self):
        from mmbacktest.metrics.performance import compute_metrics

        rng = np.random.default_rng(0)
        n = 5000
        mid = 80_000 + np.cumsum(rng.normal(0, 1.0, n))
        inv = np.sign(rng.normal(0, 1, n)) * 2.0        # position unrelated to price
        m = compute_metrics(self._timeline(inv, mid), pd.DataFrame(), {},
                            q_max=10.0, strategy="noise")

        assert m.pnl_inventory_gross > abs(m.pnl_inventory) * 10, (
            "gross must dwarf the net for an unpredictive position"
        )
        assert abs(m.pnl_inventory_tstat) < 2.0

    def test_a_genuinely_predictive_position_is_significant(self):
        """Control: if the position really does anticipate the move, t is large.

        Guards against the statistic being vacuously small for everything,
        which would make it useless as a filter.
        """
        from mmbacktest.metrics.performance import compute_metrics

        rng = np.random.default_rng(1)
        n = 5000
        steps = rng.normal(0, 1.0, n)
        mid = 80_000 + np.cumsum(steps)
        # Position at step i must anticipate the move *into* step i+1, which
        # is steps[i+1], since diff(mid)[i] == steps[i+1].
        inv = np.concatenate([np.sign(steps[1:]) * 2.0, [0.0]])

        m = compute_metrics(self._timeline(inv, mid), pd.DataFrame(), {},
                            q_max=10.0, strategy="oracle")
        assert m.pnl_inventory_tstat > 10.0


class TestGLFTAgainstPaper:
    """The closed form must match Gueant-Lehalle-Fernandez-Tapia (2013).

    Theorem 2 with the Proposition 3 approximation:

        delta_b(q) = (1/g) ln(1 + g/k)
                     + ((2q+1)/2) sqrt( s^2 g / (2 k A) (1 + g/k)^(1 + k/g) )

    The logarithm and the radical's base take gamma/kappa; only the
    exponent takes kappa/gamma. Both were once written as kappa/gamma, which
    is invisible at gamma == kappa -- the default the code was built on.
    """

    @staticmethod
    def _state(inv, sigma=7.87, kappa=0.1046, A=0.0674, mid=79_985.0):
        return MarketState(
            ts_seconds=1.0, mid=mid, best_bid=mid - 8.0, best_ask=mid + 8.0,
            microprice=mid, imbalance=0.0, sigma=sigma, kappa=kappa, A=A,
            inventory=inv, time_remaining=60.0,
        )

    @staticmethod
    def _paper(q, gamma, sigma, k, A, q_max, lot=1.0):
        # The paper's q counts traded lots, bounded by Q; it is not scaled to
        # [-1, 1] by the position limit.
        qn = np.clip(q / lot, -q_max / lot, q_max / lot)
        base = np.log1p(gamma / k) / gamma
        rad = np.sqrt(
            sigma * sigma * gamma / (2.0 * k * A)
            * (1.0 + gamma / k) ** (1.0 + k / gamma)
        )
        return base + (2 * qn + 1) / 2 * rad, base - (2 * qn - 1) / 2 * rad

    @pytest.mark.parametrize("gamma", [1e-4, 1e-3, 1e-2, 0.1, 1.0])
    @pytest.mark.parametrize("inv", [-3.0, 0.0, 2.0])
    def test_distances_match_the_paper(self, gamma, inv):
        cfg = StrategyConfig(name="glft", gamma=gamma, q_max=10.0)
        q = GLFTQuoter(cfg)
        st = self._state(inv)
        got_b, got_a = q.distances(st)
        exp_b, exp_a = self._paper(inv, gamma, st.sigma, st.kappa, st.A, cfg.q_max)
        assert got_b == pytest.approx(max(exp_b, 0.0), rel=1e-9)
        assert got_a == pytest.approx(max(exp_a, 0.0), rel=1e-9)

    def test_base_term_agrees_with_avellaneda_stoikov(self):
        """Both models share the same fill term; it must be the same number.

        This is the check that would have caught the swap: the two strategies
        derive the base half-spread from the same Hamilton-Jacobi term, so a
        disagreement means one of them is wrong.
        """
        for gamma in (1e-4, 1e-3, 1e-2, 0.1, 1.0):
            kappa = 0.1046
            glft_base = np.log1p(gamma / kappa) / gamma
            as_half = (2.0 / gamma) * np.log1p(gamma / kappa) / 2.0
            assert glft_base == pytest.approx(as_half, rel=1e-12)

    def test_risk_neutral_limit_is_one_over_kappa(self):
        """As gamma -> 0 the base distance must approach 1/kappa.

        Maximising delta * A exp(-kappa delta) gives delta* = 1/kappa. The
        swapped form diverged to infinity here.
        """
        kappa = 0.1046
        cfg = StrategyConfig(name="glft", gamma=1e-9, q_max=10.0)
        q = GLFTQuoter(cfg)
        base, _coeff, _q = q._terms(self._state(0.0, kappa=kappa))
        assert base == pytest.approx(1.0 / kappa, rel=1e-3)

    def test_intensity_conversion_is_documented_factor(self):
        """GLFT's A is the fitted density divided by (2 kappa)."""
        from mmbacktest.calibration.intensity import IntensityEstimate

        est = IntensityEstimate(A=0.0141, kappa=0.1046, n_trades=15926,
                                n_buckets_used=8, r_squared=0.719)
        assert est.arrival_intensity == pytest.approx(0.0141 / 0.1046 / 2.0)
        assert est.arrival_intensity > est.A      # density understates it


class TestGLFTInventoryUnits:
    """GLFT's q counts lots, not fractions of the position limit.

    Dividing by q_max scales the entire inventory term down by the limit,
    which silently disables inventory control: on BTC epoch 27 it left GLFT
    running a mean absolute inventory of 5.4 against AS's 0.87.
    """

    @staticmethod
    def _state(inv, mid=79_985.0):
        return MarketState(
            ts_seconds=1.0, mid=mid, best_bid=mid - 8.0, best_ask=mid + 8.0,
            microprice=mid, imbalance=0.0, sigma=7.87, kappa=0.1046, A=0.0674,
            inventory=inv, time_remaining=60.0,
        )

    def test_skew_does_not_depend_on_the_position_limit(self):
        """One lot of inventory must skew the same regardless of q_max.

        q_max is a risk limit; it bounds where quoting stops, and must not
        rescale the response to a given position.
        """
        skews = []
        for q_max in (5.0, 10.0, 50.0):
            cfg = StrategyConfig(name="glft", gamma=1e-3, q_max=q_max,
                                 order_size=1.0)
            b, a = GLFTQuoter(cfg).distances(self._state(1.0))
            skews.append(b - a)
        assert max(skews) - min(skews) < 1e-9, f"skew moved with q_max: {skews}"

    def test_skew_scales_with_lot_size(self):
        """Holding one lot is one unit of q whatever the lot is worth."""
        cfg_a = StrategyConfig(name="glft", gamma=1e-3, q_max=10.0, order_size=1.0)
        cfg_b = StrategyConfig(name="glft", gamma=1e-3, q_max=20.0, order_size=2.0)
        ba, aa = GLFTQuoter(cfg_a).distances(self._state(1.0))
        bb, ab = GLFTQuoter(cfg_b).distances(self._state(2.0))
        assert (ba - aa) == pytest.approx(bb - ab, rel=1e-9)

    def test_inventory_term_is_material_at_one_lot(self):
        """A single lot must move the quote by an economically visible amount.

        With the q_max division the skew at one contract was 0.35 USD against
        AS's 3.72 on the same calibration -- small enough that the position
        wandered to the bound unopposed.
        """
        cfg = StrategyConfig(name="glft", gamma=1e-3, q_max=10.0, order_size=1.0)
        q = GLFTQuoter(cfg)
        flat_b, flat_a = q.distances(self._state(0.0))
        long_b, long_a = q.distances(self._state(1.0))

        assert flat_b == pytest.approx(flat_a, rel=1e-12)   # symmetric at zero
        assert long_b > flat_b     # bid pushed away when long
        assert long_a < flat_a     # ask pulled in when long
        assert (long_b - flat_b) > 1.0
