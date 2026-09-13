"""Avellaneda-Stoikov quoter.

Reference: Avellaneda & Stoikov (2008), "High-frequency trading in a limit
order book", Quantitative Finance 8(3), 217-224.

The model has a market maker maximising expected exponential utility of
terminal wealth over a finite horizon, with mid-price following arithmetic
Brownian motion and fills arriving at an intensity that decays exponentially
in quote distance. Two results come out of it.

Reservation price, the indifference price given current inventory:

    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

Read it as: holding inventory q makes the maker want to trade out of it, so
the price at which they are indifferent shifts against the position. Long
inventory pushes r below the mid, which tightens the ask and widens the bid,
so the position tends to get sold down. The size of that shift scales with
risk aversion, with variance, and with how long the position must be held.

Optimal total spread:

    delta_a + delta_b = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / kappa)

Two additive terms. The first is a risk premium that widens with volatility
and horizon. The second depends only on gamma and kappa: it is the part that
trades off earning more per fill against getting fewer fills, and it does not
vanish as the horizon shrinks.

Implementation notes:

The quotes are placed symmetrically around r, not around the mid. That is
what produces inventory skew: the spread width is symmetric, but its centre
moves, so the effective distances to the mid differ on the two sides.

The horizon term (T - t) is handled by treating the session as a rolling
horizon rather than a hard terminal time. A literal finite-horizon
implementation collapses the risk term to zero as t approaches T, which makes
the maker quote recklessly tight in the last seconds of every session. Since
the interest here is steady-state quoting behaviour rather than liquidating
into a deadline, time_remaining is floored.
"""

from __future__ import annotations

import numpy as np

from .base import MarketState, Quote, Quoter


class AvellanedaStoikovQuoter(Quoter):
    """Classical AS quoter with inventory-skewed reservation price."""

    name = "avellaneda_stoikov"

    # Floor on (T - t) in seconds. Prevents the risk term collapsing to zero
    # at the end of the horizon and producing degenerate tight quotes.
    MIN_TIME_REMAINING = 1.0

    def compute(self, state: MarketState) -> Quote:
        gamma = self.cfg.gamma
        sigma = max(state.sigma, 1e-12)
        kappa = state.kappa
        q = state.inventory

        tau = max(state.time_remaining, self.MIN_TIME_REMAINING)

        # Reservation price: r = s - q * gamma * sigma^2 * tau
        variance_term = gamma * sigma * sigma * tau
        reservation = state.mid - q * variance_term

        # Optimal total spread, halved to get the distance each side.
        #
        # The second term needs kappa > 0. When the intensity fit failed
        # (kappa <= 0 signals an unusable estimate) there is no basis for the
        # fill-probability trade-off, so the quoter falls back to the risk
        # term alone. That is conservative: it quotes on volatility only,
        # rather than inventing a kappa.
        if kappa > 1e-12:
            fill_term = (2.0 / gamma) * np.log1p(gamma / kappa)
        else:
            fill_term = 0.0

        total_spread = variance_term + fill_term
        half_spread = 0.5 * total_spread

        return self._finalise(state, reservation, half_spread)

    # ------------------------------------------------------------------
    # Introspection, used by the diagnostics in the report
    # ------------------------------------------------------------------

    def decompose(self, state: MarketState) -> dict[str, float]:
        """Break the quote into its components.

        Useful when explaining why a quote moved: was it inventory skew, a
        volatility change, or an intensity change? Reporting the pieces
        separately is what makes the strategy auditable rather than a
        black box that emits numbers.
        """
        gamma = self.cfg.gamma
        sigma = max(state.sigma, 1e-12)
        tau = max(state.time_remaining, self.MIN_TIME_REMAINING)
        kappa = state.kappa

        variance_term = gamma * sigma * sigma * tau
        fill_term = (2.0 / gamma) * np.log1p(gamma / kappa) if kappa > 1e-12 else 0.0

        return {
            "inventory_skew": -state.inventory * variance_term,
            "risk_term": variance_term,
            "fill_term": fill_term,
            "half_spread": 0.5 * (variance_term + fill_term),
            "sigma": sigma,
            "kappa": kappa,
            "tau": tau,
        }
