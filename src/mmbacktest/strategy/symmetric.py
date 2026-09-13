"""Symmetric quoter: the naive baseline.

Posts a fixed half-spread either side of the mid and does nothing about
inventory beyond respecting the hard cap. No volatility input, no intensity
input, no skew.

This exists to answer a question the other two strategies cannot answer on
their own: how much of any observed performance comes from the modelling, and
how much comes simply from being present in the book and capturing spread?
A market maker that quotes symmetrically in a market with no drift will still
earn the spread; the question is what happens to inventory while it does.

Concretely, this is the baseline that isolates the value of inventory
management. If AS and GLFT do not beat it on inventory variance and
adverse-selection cost, their extra machinery is not paying for itself.
"""

from __future__ import annotations

from .base import MarketState, Quote, Quoter


class SymmetricQuoter(Quoter):
    """Fixed half-spread around the mid, no inventory skew."""

    name = "symmetric"

    def compute(self, state: MarketState) -> Quote:
        # Reservation price is just the mid: no inventory skew at all. This is
        # the defining property of the baseline, so it is stated explicitly
        # rather than falling out of a formula with gamma set to zero.
        reservation = state.mid

        half_spread = self.cfg.fixed_half_spread_bps * state.mid / 10_000.0

        return self._finalise(state, reservation, half_spread)
