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


class TestDerivBackfillParsing:
    """Field positions in the /status/deriv/hist payload.

    The `hist` rows omit the leading KEY field that the snapshot endpoint
    carries, so every index is one lower than Bitfinex's published field
    table. An off-by-one here is silent and plausible-looking: mark price
    becomes spot price, or the funding rate becomes the funding step. The
    fixture is a real payload slice spanning an 8h settlement boundary.
    """

    @staticmethod
    def _load():
        import importlib.util
        import json

        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "backfill_deriv", root / "scripts" / "backfill_deriv.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        raw = json.loads(
            (root / "tests" / "fixtures" / "deriv_hist_btc.json").read_text()
        )
        return mod, raw

    def test_every_row_parses(self):
        mod, raw = self._load()
        df = mod.to_frame("tBTCF0:USTF0", raw)
        assert len(df) == len(raw)

    def test_fields_match_the_raw_row(self):
        """Positional check against the payload, field by field."""
        mod, raw = self._load()
        df = mod.to_frame("tBTCF0:USTF0", raw).sort_values("ts").reset_index(drop=True)
        r = sorted(raw, key=lambda x: x[0])[0]
        got = df.iloc[0]

        assert got["ts"].value // 10**6 == r[0]
        assert got["deriv_price"] == pytest.approx(r[2])
        assert got["spot_price"] == pytest.approx(r[3])
        assert got["insurance_fund"] == pytest.approx(r[5])
        assert got["next_funding_ts"].value // 10**6 == r[7]
        assert got["next_funding_accrued"] == pytest.approx(r[8])
        assert got["next_funding_step"] == int(r[9])
        assert got["mark_price"] == pytest.approx(r[14])
        assert got["open_interest"] == pytest.approx(r[17])

    def test_settlements_land_on_the_eight_hour_grid(self):
        """The sharpest guard against an index shift.

        Bitfinex settles funding at 00:00, 08:00 and 16:00 UTC. If the
        timestamp index were off by one we would be reading the funding step
        (an integer in the thousands) or the insurance fund, and the hours
        would be arbitrary.
        """
        mod, raw = self._load()
        df = mod.to_frame("tBTCF0:USTF0", raw)
        assert set(df["next_funding_ts"].dt.hour.unique()) <= {0, 8, 16}
        assert (df["next_funding_ts"].dt.minute == 0).all()

    def test_values_are_in_plausible_ranges(self):
        """Catches a swap between two fields that are both well-formed."""
        mod, raw = self._load()
        df = mod.to_frame("tBTCF0:USTF0", raw)
        assert df["deriv_price"].between(10_000, 500_000).all()
        assert df["spot_price"].between(10_000, 500_000).all()
        assert df["mark_price"].between(10_000, 500_000).all()
        # A funding rate is a small fraction; the step is a whole-number
        # counter that climbs through the 8h window and resets at settlement
        # (so it is legitimately small just after one -- the fixture spans a
        # boundary and contains 17 and 37). Confusing the two is the likeliest
        # single-index error, so discriminate on integrality and on the
        # within-window maximum rather than on every row's magnitude.
        assert df["next_funding_accrued"].abs().lt(0.01).all()
        assert (df["next_funding_step"] % 1 == 0).all()
        assert df["next_funding_step"].max() > 1_000
        assert not (df["next_funding_accrued"] % 1 == 0).all()

    def test_timestamps_carry_utc_and_survive_resolution(self):
        """Same class of bug as the engine's ms/ns defect: assert the unit."""
        mod, raw = self._load()
        df = mod.to_frame("tBTCF0:USTF0", raw)
        for col in ("ts", "next_funding_ts"):
            assert str(df[col].dt.tz) == "UTC"
            # Seconds since epoch must be a real 2026 timestamp, not 1.79e9/1e6.
            secs = df[col].iloc[0].timestamp()
            assert 1.7e9 < secs < 1.9e9, f"{col} came back as {secs}"

    def test_rows_without_a_timestamp_are_dropped_not_zeroed(self):
        mod, raw = self._load()
        broken = [list(raw[0]), list(raw[1])]
        broken[0][0] = None
        df = mod.to_frame("tBTCF0:USTF0", broken)
        assert len(df) == 1


class TestFunding:
    """Funding is charged on the position held at each 8h settlement."""

    def test_long_pays_when_rate_is_positive(self):
        from mmbacktest.sim.engine import _Position
        p = _Position(inventory=2.0)
        charge = p.apply_funding(rate=0.0005, mark_price=80_000.0)
        assert charge == pytest.approx(2.0 * 80_000.0 * 0.0005)   # 80 USD
        assert p.cash == pytest.approx(-80.0)
        assert p.funding_paid == pytest.approx(80.0)

    def test_short_is_paid_when_rate_is_positive(self):
        """Sign falls out of inventory * rate: no special case needed."""
        from mmbacktest.sim.engine import _Position
        p = _Position(inventory=-2.0)
        p.apply_funding(rate=0.0005, mark_price=80_000.0)
        assert p.cash == pytest.approx(+80.0)
        assert p.funding_paid == pytest.approx(-80.0)

    def test_flat_book_pays_nothing(self):
        from mmbacktest.sim.engine import _Position
        p = _Position(inventory=0.0)
        p.apply_funding(rate=0.0005, mark_price=80_000.0)
        assert p.cash == 0.0
        assert p.n_funding_events == 1

    @staticmethod
    def _run_with(funding):
        from mmbacktest.calibration.intensity import IntensityEstimate
        from mmbacktest.config import Config
        from mmbacktest.data.synthetic import generate_mbo_events, generate_trades
        from mmbacktest.sim.engine import BacktestEngine
        from mmbacktest.strategy.base import make_quoter

        c = Config()
        c.simulation.warmup_seconds = 0.0
        c.strategy.name = "symmetric"
        c.calibration.vol_min_observations = 5
        ev = generate_mbo_events(n_events=20_000, seed=3)
        tr = generate_trades(ev, trade_rate=0.02, seed=3)
        eng = BacktestEngine(
            c, make_quoter(c.strategy),
            IntensityEstimate(A=0.09, kappa=0.04, n_trades=500,
                              n_buckets_used=6, r_squared=0.8),
        )
        return ev, eng.run(ev, tr, progress_every=None, funding=funding)

    def test_settlements_outside_the_window_are_not_counted(self):
        """The loader widens the window by one interval each side; only
        settlements the replay clock actually passes may be charged."""
        from mmbacktest.data.synthetic import generate_mbo_events
        ev = generate_mbo_events(n_events=20_000, seed=3)
        t0, t1 = ev["ts_exch"].min(), ev["ts_exch"].max()
        mid_t = t0 + (t1 - t0) / 2
        funding = pd.DataFrame({
            "settles_at": [t0 - pd.Timedelta(hours=1), mid_t,
                           t1 + pd.Timedelta(hours=1)],
            "rate": [0.0005, 0.0005, 0.0005],
            "mark_at_settle": [80_000.0, 80_000.0, 80_000.0],
        })
        _, res = self._run_with(funding)
        assert res.funding_stats["n_funding_events"] == 1

    def test_attribution_sums_with_funding(self):
        """total = spread + inventory + fees + funding, to the cent.

        Funding is charged to cash during the replay, so it is inside total
        PnL; if it were not removed from the inventory residual the cost of
        carrying a position would be booked as mark-to-market.
        """
        from mmbacktest.metrics.performance import compute_metrics
        from mmbacktest.sim.fills import compute_markouts
        from mmbacktest.data.synthetic import generate_mbo_events

        ev = generate_mbo_events(n_events=20_000, seed=3)
        t0, t1 = ev["ts_exch"].min(), ev["ts_exch"].max()
        funding = pd.DataFrame({
            "settles_at": [t0 + (t1 - t0) * f for f in (0.3, 0.6, 0.9)],
            "rate": [0.0005, -0.0003, 0.0007],
            "mark_at_settle": [80_000.0] * 3,
        })
        _, res = self._run_with(funding)
        fills = compute_markouts(res.fills, res.timeline[["ts", "mid"]],
                                 (100, 1000))
        m = compute_metrics(res.timeline, fills, res.fill_stats, q_max=10.0,
                            markout_horizons_ms=(100, 1000),
                            funding_stats=res.funding_stats)

        assert m.n_funding_events == 3
        parts = m.pnl_spread_capture + m.pnl_inventory + m.pnl_fees + m.pnl_funding
        assert parts == pytest.approx(m.total_pnl, abs=1e-6)

    def test_no_schedule_means_not_modelled(self):
        """Absence must be distinguishable from funding that netted to zero."""
        _, res = self._run_with(None)
        assert res.funding_stats["n_funding_events"] == 0
        assert res.funding_stats["funding_paid"] == 0.0


class TestFlashMoveQuoteHygiene:
    """Quotes after a flash move inflates sigma.

    Regression for the epoch-68 crash. BTC-PERP spiked from 81,200 to 153,960
    and back in ~1.5 s on 2026-09-20 16:28:12 (real trades, not a book
    artefact). Realised sigma over the 300 s window rose to ~12,300 USD/sqrt(s)
    against a normal ~8, so AS's skew q*gamma*sigma^2*tau put the raw bid at
    -58,595 with only 0.154 contracts held, and the tick lookup raised on a
    negative price. The state below is the one captured at the crash.
    """

    CRASH = dict(mid=81_299.0, best_bid=81_290.0, best_ask=81_308.0,
                 sigma=12_301.59, inventory=0.1540278600000016,
                 kappa=0.0674, A=0.0696)

    def _state(self, **over):
        p = {**self.CRASH, **over}
        return MarketState(
            ts_seconds=1.0, mid=p["mid"], best_bid=p["best_bid"],
            best_ask=p["best_ask"], microprice=p["mid"], imbalance=0.0,
            sigma=p["sigma"], kappa=p["kappa"], A=p["A"],
            inventory=p["inventory"], time_remaining=60.0,
        )

    def test_crash_state_no_longer_raises(self):
        q = AvellanedaStoikovQuoter(StrategyConfig(name="avellaneda_stoikov",
                                                   gamma=1e-4, tick_size=None))
        quote = q.compute(self._state())
        # Long and skewing hard: the near (ask) side is pinned to the touch,
        # the bid -- non-positive -- is not quoted at all.
        assert quote.bid_price is None and quote.bid_size == 0.0
        assert quote.ask_price == pytest.approx(81_291.0)
        assert q.n_sides_suppressed == 1

    def test_short_side_mirror_needs_no_suppression(self):
        """Short inventory pushes the reservation *up*; the raw prices stay
        positive and the no-cross repair handles it, so nothing is dropped."""
        q = AvellanedaStoikovQuoter(StrategyConfig(name="avellaneda_stoikov",
                                                   gamma=1e-4, tick_size=None))
        quote = q.compute(self._state(inventory=-0.154))
        # The repaired bid steps back from the touch (81,307) and is then
        # rounded on the raw bid's grid. The raw bid is ~139,800, a decade
        # above the touch, where the tick is 10, so it lands on 81,300: on the
        # venue grid, below the touch, a few ticks conservative. Kept as-is so
        # the reorder stays identical to the code the earlier sweep ran on.
        assert quote.bid_price == pytest.approx(81_300.0)
        assert is_on_tick_grid(quote.bid_price, 1.0)
        assert quote.bid_price < 81_308.0
        assert quote.ask_price is not None and quote.ask_price > 81_308.0
        assert q.n_sides_suppressed == 0

    @pytest.mark.parametrize("name", ["symmetric", "avellaneda_stoikov", "glft"])
    def test_extreme_sigma_never_emits_a_bad_price(self, name):
        quoters = {"symmetric": SymmetricQuoter,
                   "avellaneda_stoikov": AvellanedaStoikovQuoter,
                   "glft": GLFTQuoter}
        for gamma in (1e-4, 1e-3, 1e-2):
            q = quoters[name](StrategyConfig(name=name, gamma=gamma,
                                             q_max=10.0, tick_size=None))
            for sigma in (8.0, 500.0, 12_301.59, 1e6):
                for inv in (-10.0, -3.0, -0.154, 0.0, 0.154, 3.0, 10.0):
                    quote = q.compute(self._state(sigma=sigma, inventory=inv))
                    for px in (quote.bid_price, quote.ask_price):
                        if px is None:
                            continue
                        assert np.isfinite(px) and px > 0.0
                        assert is_on_tick_grid(px, bitfinex_tick_size(px))
                    if quote.bid_price is not None:
                        assert quote.bid_price < 81_308.0     # never crosses
                    if quote.ask_price is not None:
                        assert quote.ask_price > 81_290.0

    def test_reorder_is_identical_on_ordinary_inputs(self):
        """The fix moves the no-cross repair ahead of rounding. For every
        finite positive raw price that must give exactly what the old
        round-then-repair order gave, or the earlier sweep results would no
        longer be comparable. The old algorithm is reproduced here verbatim
        as the reference."""
        def old(raw_bid, raw_ask, bb, ba):
            bt, at = bitfinex_tick_size(raw_bid), bitfinex_tick_size(raw_ask)
            b = round_to_tick(raw_bid, bt, "down")
            a = round_to_tick(raw_ask, at, "up")
            if b >= ba:
                b = round_to_tick(ba - bitfinex_tick_size(ba), bt, "down")
            if a <= bb:
                a = round_to_tick(bb + bitfinex_tick_size(bb), at, "up")
            return b, a

        q = SymmetricQuoter(StrategyConfig(name="symmetric", tick_size=None,
                                           min_half_spread_bps=0.0,
                                           max_half_spread_bps=1e6))
        rng = np.random.default_rng(11)
        n = 0
        for _ in range(4000):
            mid = float(rng.choice([rng.uniform(9_000, 11_000),     # decade edge
                                    rng.uniform(76_000, 84_000)]))
            half_mkt = float(rng.uniform(0.5, 30.0))
            bb = round_to_tick(mid - half_mkt, bitfinex_tick_size(mid - half_mkt), "down")
            ba = round_to_tick(mid + half_mkt, bitfinex_tick_size(mid + half_mkt), "up")
            if bb >= ba:
                continue
            # Skews from none to far through the touch on either side.
            r = mid + float(rng.uniform(-300.0, 300.0))
            h = float(rng.uniform(0.0, 60.0))
            if r - h <= 0:
                continue
            st = MarketState(ts_seconds=1.0, mid=mid, best_bid=bb, best_ask=ba,
                             microprice=mid, imbalance=0.0, sigma=8.0,
                             kappa=0.1, A=0.01, inventory=0.0,
                             time_remaining=60.0)
            new = q._finalise(st, r, h)
            ob, oa = old(r - h, r + h, bb, ba)
            assert new.bid_price == pytest.approx(ob, abs=1e-9)
            assert new.ask_price == pytest.approx(oa, abs=1e-9)
            n += 1
        assert n > 3000
        assert q.n_sides_suppressed == 0
