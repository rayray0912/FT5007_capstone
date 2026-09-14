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

The asymptotic (long-horizon) form used here, from Theorem 2 combined with
the closed-form approximation of the eigenvector in Proposition 3:

    delta_b(q) = (1/gamma) * ln(1 + gamma/k)
                 + ((2q + 1) / 2) * sqrt( (sigma^2 * gamma) / (2 * k * A)
                                            * (1 + gamma/k)^(1 + k/gamma) )

    delta_a(q) = (1/gamma) * ln(1 + gamma/k)
                 - ((2q - 1) / 2) * sqrt( same radical )

Note which ratio goes where: the logarithm and the radical's base take
gamma/k, while the radical's exponent takes k/gamma. The paper's Theorem 2
states the base term exactly as AS's, (1/gamma) ln(1 + gamma/k), and gives
the resulting spread as (2/gamma) ln(1 + gamma/k) + <radical>.

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

        # Cap the distances *before* converting to (centre, half-width).
        #
        # The conversion below is only algebraically equivalent while the
        # distances stay comparable to the mid. Let a distance reach 1e151 --
        # which the closed form does for small gamma, since kappa/gamma then
        # drives the radical's exponent -- and `mid - delta` and `mid + delta`
        # sum to exactly 0.0 in float64: the mid is rounded away entirely and
        # the quote centres on zero instead of on the market. `_finalise`
        # clamps the half-spread afterwards but cannot recover the centre, so
        # the result is a quote at +/- max_half_spread around price zero.
        #
        # Capping here costs nothing real: `_finalise` clamps the half-spread
        # to the same band anyway, so any distance beyond it was going to be
        # truncated. The cap is generous (twice the band) so it binds only in
        # the degenerate regime and never shapes a quote the model meant.
        max_delta = 2.0 * state.mid * self.cfg.max_half_spread_bps / 10_000.0
        delta_bid = min(delta_bid, max_delta)
        delta_ask = min(delta_ask, max_delta)

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

    def _terms(self, state: MarketState) -> tuple[float, float, float]:
        """(base distance, inventory coefficient, normalised inventory).

        Single source of truth for the closed form. `distances` and
        `decompose` both read it: they previously each carried their own copy
        of the algebra, which is how the swapped gamma/kappa survived a fix
        applied to only one of them.
        """
        gamma = max(self.cfg.gamma, self.MIN_GAMMA)
        kappa = max(state.kappa, self.MIN_KAPPA)
        sigma = max(state.sigma, 1e-12)
        k_A = max(state.A, self.MIN_A)

        # Inventory in units of the traded lot. The paper's q is "the (signed)
        # quantity of shares he holds", with transactions "of constant size,
        # scaled to 1", and the bound Q is a count of those units: the
        # eigenvector f^0 lives in R^(2Q+1), indexed by integer q.
        #
        # This was previously divided by q_max, which puts q in [-1, 1] and
        # therefore divides the whole inventory term by the position limit --
        # a factor of 10 at the default q_max. The skew per contract came out
        # 10x too small, so GLFT ran a mean absolute inventory of 5.4 against
        # AS's 0.87 on the same data, and the inventory control the model
        # exists to provide was effectively switched off. The bound itself is
        # enforced separately, by suppressing a side at the limit.
        lot = max(self.cfg.order_size, 1e-9)
        bound = self.cfg.q_max / lot
        q = float(np.clip(state.inventory / lot, -bound, bound))

        # Two ratios, and they are not interchangeable. The paper's asymptotic
        # approximation (Theorem 2 plus Proposition 3) is
        #
        #   delta_b(q) = (1/gamma) ln(1 + gamma/k)
        #                + ((2q+1)/2) sqrt( sigma^2 gamma / (2 k A)
        #                                   * (1 + gamma/k)^(1 + k/gamma) )
        #
        # so the logarithm and the radical's base take gamma/k, while only
        # the radical's *exponent* takes k/gamma.
        #
        # Both were previously written as k/gamma. That is wrong in a way
        # nothing here could catch: at gamma = 0.1 against a calibrated
        # kappa = 0.1001 the two forms are numerically identical (6.93 USD),
        # which is exactly the default this was developed on. They diverge as
        # soon as gamma moves -- at gamma = 1e-3 the correct base is 9.94 USD
        # and the swapped one is 46.12 -- and the swapped form diverges to
        # infinity as gamma -> 0, where the risk-neutral optimum is the finite
        # 1/kappa (maximise delta * A exp(-kappa delta)). Every sign and
        # symmetry test passes either way.
        gk = gamma / kappa      # appears in the logarithm and the radical base
        kg = kappa / gamma      # appears only in the radical exponent

        # Base distance: the part that does not depend on inventory. Matches
        # AS's (1/gamma) ln(1 + gamma/kappa), as it must -- both come from the
        # same Hamilton-Jacobi term.
        base = np.log1p(gk) / gamma

        # Inventory coefficient. The exponent (1 + k/gamma) can overflow for
        # small gamma, so the radical is computed in log space.
        log_radical = (
            np.log(sigma * sigma * gamma)
            - np.log(2.0 * k_A * kappa)
            + (1.0 + kg) * np.log1p(gk)
        )
        # Clip before exponentiating: an extreme upstream estimate should
        # produce a wide quote, not an overflow warning and an inf.
        coeff = float(np.exp(0.5 * np.clip(log_radical, -700.0, 700.0)))
        return float(base), coeff, q

    def distances(self, state: MarketState) -> tuple[float, float]:
        """Optimal (bid, ask) distances from the mid, in price units."""
        base, coeff, q = self._terms(state)

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
        base, coeff, q = self._terms(state)
        delta_bid, delta_ask = self.distances(state)

        return {
            "base_distance": float(base),
            "inventory_coefficient": coeff,
            "q_normalised": q,
            "delta_bid": delta_bid,
            "delta_ask": delta_ask,
            "skew": delta_bid - delta_ask,
            "sigma": state.sigma,
            "kappa": state.kappa,
            "A": state.A,
        }
