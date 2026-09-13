"""GLFT quoter.

Reference: Guéant, Lehalle & Fernandez-Tapia (2013), "Dealing with the
inventory risk: a solution to the market making problem", Mathematics and
Financial Economics 7(4), 477-507.

What GLFT adds over Avellaneda-Stoikov
--------------------------------------

AS solves an unconstrained problem: inventory can in principle drift
arbitrarily far, and the exponential utility is what keeps it in check. That
is not how a real book works. A market maker has a position limit, imposed by
risk management or by margin, and the interesting question is how to quote
optimally *given* that limit.

GLFT solves the constrained problem. Inventory lives in [-Q, Q], and the
optimal quotes come out as explicit per-side distances that widen as the
position approaches its bound on that side. Where AS produces one symmetric
half-spread around a shifted centre, GLFT produces two genuinely different
distances, bid and ask, each a function of the current inventory.

The asymptotic (long-horizon) form used here:

    delta_b(q) = (1/kappa) * ln(1 + kappa/gamma)
                 + ((2q + 1) / 2) * sqrt( (sigma^2 * gamma) / (2 * k_A * kappa)
                                            * (1 + kappa/gamma)^(1 + kappa/gamma) )

    delta_a(q) = (1/kappa) * ln(1 + kappa/gamma)
                 - ((2q - 1) / 2) * sqrt( same radical )

Both have the same structure: a constant base distance that depends only on
gamma and kappa, plus an inventory-proportional term with opposite sign on
the two sides. At q = 0 the two distances are equal and the quote is
symmetric. As q rises the bid pushes out and the ask pulls in, so the book
works the position back toward flat. The coefficient on the inventory term is
where sigma enters: more volatile markets make inventory more expensive to
hold, so the skew per unit of inventory is larger.

k_A is the intensity level parameter (called A in the calibration module). It
scales how quickly fills arrive in absolute terms, and it sits under the
square root, so it damps the skew: in a busy market inventory can be worked
off quickly and there is less need to skew hard.

Why the asymptotic form rather than the finite-horizon one
----------------------------------------------------------

The finite-horizon GLFT solution requires solving a system of ODEs backward
from the terminal condition at each step, which is both slower and dominated
by terminal-liquidation behaviour that is not what is being studied here. The
asymptotic form is the steady-state limit, and steady-state quoting is
precisely the regime a continuously-running market maker operates in. This is
also the form practitioners generally use.
"""

from __future__ import annotations

import numpy as np

from .base import MarketState, Quote, Quoter


class GLFTQuoter(Quoter):
    """GLFT quoter with explicit inventory bound and per-side distances."""

    name = "glft"

    # Numerical guards. These are not tuning knobs: they stop the closed form
    # producing inf or nan when an upstream estimate degenerates.
    MIN_KAPPA = 1e-9
    MIN_GAMMA = 1e-9
    MIN_A = 1e-12

    def compute(self, state: MarketState) -> Quote:
        delta_bid, delta_ask = self.distances(state)

        # GLFT gives distances from the mid, not from a shifted reservation
        # price. To reuse the shared finalisation path, convert the asymmetric
        # pair into an equivalent (centre, half-width) representation. The two
        # parameterisations are equivalent: centre is the midpoint of the
        # quoted prices, half-width is half the quoted spread.
        bid_price = state.mid - delta_bid
        ask_price = state.mid + delta_ask

        centre = 0.5 * (bid_price + ask_price)
        half_spread = 0.5 * (ask_price - bid_price)

        return self._finalise(state, centre, half_spread)

    # ------------------------------------------------------------------
    # Core formula
    # ------------------------------------------------------------------

    def distances(self, state: MarketState) -> tuple[float, float]:
        """Optimal (bid, ask) distances from the mid, in price units."""
        gamma = max(self.cfg.gamma, self.MIN_GAMMA)
        kappa = max(state.kappa, self.MIN_KAPPA)
        sigma = max(state.sigma, 1e-12)
        k_A = max(state.A, self.MIN_A)

        # Inventory normalised by the bound. GLFT's q is in units of the
        # position limit, so a maker with q_max = 10 holding 5 contracts is at
        # q = 0.5, not q = 5. Skipping this normalisation makes the skew scale
        # with the arbitrary choice of contract size.
        q = state.inventory / max(self.cfg.q_max, 1e-9)
        q = float(np.clip(q, -1.0, 1.0))

        ratio = kappa / gamma

        # Base distance: the part that does not depend on inventory.
        base = np.log1p(ratio) / kappa

        # Inventory coefficient. The exponent (1 + ratio) can overflow for
        # large kappa/gamma, so the radical is computed in log space.
        log_radical = (
            np.log(sigma * sigma * gamma)
            - np.log(2.0 * k_A * kappa)
            + (1.0 + ratio) * np.log1p(ratio)
        )
        # Clip before exponentiating: an extreme upstream estimate should
        # produce a wide quote, not an overflow warning and an inf.
        coeff = float(np.exp(0.5 * np.clip(log_radical, -700.0, 700.0)))

        delta_bid = base + 0.5 * (2.0 * q + 1.0) * coeff
        delta_ask = base - 0.5 * (2.0 * q - 1.0) * coeff

        # Distances must stay positive: a negative distance would mean quoting
        # through the mid, which is a taker order. At extreme inventory the
        # formula can push one side negative, and the correct response is to
        # sit at the mid on that side and let the bound logic suppress it.
        delta_bid = max(delta_bid, 0.0)
        delta_ask = max(delta_ask, 0.0)

        return delta_bid, delta_ask

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def decompose(self, state: MarketState) -> dict[str, float]:
        """Break the quote into base and inventory components."""
        gamma = max(self.cfg.gamma, self.MIN_GAMMA)
        kappa = max(state.kappa, self.MIN_KAPPA)
        sigma = max(state.sigma, 1e-12)
        k_A = max(state.A, self.MIN_A)

        q = float(np.clip(state.inventory / max(self.cfg.q_max, 1e-9), -1.0, 1.0))
        ratio = kappa / gamma
        base = np.log1p(ratio) / kappa

        log_radical = (
            np.log(sigma * sigma * gamma)
            - np.log(2.0 * k_A * kappa)
            + (1.0 + ratio) * np.log1p(ratio)
        )
        coeff = float(np.exp(0.5 * np.clip(log_radical, -700.0, 700.0)))

        delta_bid, delta_ask = self.distances(state)

        return {
            "base_distance": float(base),
            "inventory_coefficient": coeff,
            "q_normalised": q,
            "delta_bid": delta_bid,
            "delta_ask": delta_ask,
            "skew": delta_bid - delta_ask,
            "sigma": sigma,
            "kappa": kappa,
            "A": k_A,
        }
