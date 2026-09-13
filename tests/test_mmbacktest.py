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
