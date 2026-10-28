#!/usr/bin/env python3
"""What funding does to the sweep results, and a regression check on the replay.

    python scripts/funding_report.py
    python scripts/funding_report.py --new results/sweep_funding --new-fee -1 \
        --old results/sweep --old-fee 0 --gamma 0.001

Three sections:

1. Regression check. Funding does not feed back into quoting, and fees do not
   change fills, so on any epoch present in both sweeps the fill count, spread
   capture, inventory PnL and inventory variance must be identical. A mismatch
   means something other than funding changed between the two runs -- code or
   data -- and the rest of the report should not be trusted until it is found.

2. Funding per strategy: whether it paid or received, and how large it is
   against spread capture. No direction is assumed. A quoter without
   inventory control accumulates against the trend (long into a falling
   market, short into a rising one), and BTC funding is mostly positive, so
   the naive quoter pays in falls and is paid in rallies.

3. Total PnL built up leg by leg: spread + inventory, then + rebate, then +
   funding, with paired sign tests at each step. Inventory PnL is a random-walk
   residual on every single epoch (|t| < 2), so these tests are reported, not
   leaned on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_sweep import PAIRS, SHORT, STRATS, load  # noqa: E402

CHECK = ["n_fills", "spread_capture_bps", "pnl_spread_capture",
         "pnl_inventory", "inventory_variance", "mean_abs_inventory"]


def regression_check(new: pd.DataFrame, old: pd.DataFrame) -> bool:
    keys = ["epoch", "strategy"]
    both = new.merge(old, on=keys, suffixes=("_new", "_old"))
    n_ep = both["epoch"].nunique()
    print(f"\n=== 1. regression check: {n_ep} epochs present in both sweeps ===")
    if both.empty:
        print("  no overlap -- cannot check")
        return True
    ok = True
    for col in CHECK:
        a, b = f"{col}_new", f"{col}_old"
        if a not in both or b not in both:
            print(f"  {col:<22} missing in one sweep")
            continue
        diff = (both[a] - both[b]).abs()
        tol = 1e-6 * np.maximum(1.0, both[b].abs())
        bad = both.loc[diff > tol, keys + [a, b]]
        if bad.empty:
            print(f"  {col:<22} identical on all {len(both)} strategy-epochs")
        else:
            ok = False
            print(f"  {col:<22} DIFFERS on {len(bad)} of {len(both)}:")
            print(bad.head(8).to_string(index=False))
    return ok


def funding_table(df: pd.DataFrame) -> None:
    print("\n=== 2. funding per strategy ===")
    if "pnl_funding" not in df or not (df.get("n_funding_events", 0) > 0).any():
        print("  no funding in these results (was the deriv backfill run?)")
        return
    modelled = df["n_funding_events"] > 0
    if not modelled.all():
        miss = sorted(df.loc[~modelled, "epoch"].unique())
        print(f"  WARNING: funding not modelled on epochs {miss}")
    for s in STRATS:
        g = df[(df["strategy"] == s) & modelled]
        if g.empty:
            continue
        f = g["pnl_funding"]
        ratio = (-f / g["pnl_spread_capture"]).replace([np.inf, -np.inf], np.nan)
        print(f"  {SHORT[s]:>4}: paid on {int((f < 0).sum())}/{len(g)} epochs, "
              f"received on {int((f > 0).sum())}; "
              f"total {f.sum():+,.0f}  median {f.median():+,.1f}  "
              f"[{f.min():+,.0f}, {f.max():+,.0f}]")
        print(f"        cost / spread capture: median {ratio.median():.0%}  "
              f"[{ratio.min():.0%}, {ratio.max():.0%}]   "
              f"events/epoch {g['n_funding_events'].median():.0f}")

    wide = df[modelled].pivot_table(index="epoch", columns="strategy",
                                    values="pnl_funding")
    cols = [s for s in STRATS if s in wide]
    print("\n  per epoch (funding PnL, negative = paid):")
    with pd.option_context("display.float_format", "{:+,.1f}".format,
                           "display.width", 160):
        print(wide[cols].rename(columns=SHORT).to_string())


def sign(wide: pd.DataFrame, label: str) -> None:
    pos = ", ".join(f"{SHORT[s]} {int((wide[s] > 0).sum())}/{wide[s].notna().sum()}"
                    for s in STRATS if s in wide)
    print(f"  {label:<34} >0 on: {pos}")
    for a, b in PAIRS:
        if a not in wide or b not in wide:
            continue
        p = wide[[a, b]].dropna()
        d = p[a] - p[b]
        wins, n = int((d > 0).sum()), int((d != 0).sum())
        if n:
            pv = binomtest(wins, n, 0.5).pvalue
            print(f"  {'':<34}   {SHORT[a]:>4} > {SHORT[b]:<4} {wins:>2}/{n:<2}  p={pv:.2g}")


def legs(df: pd.DataFrame) -> None:
    print("\n=== 3. total PnL built up leg by leg ===")
    d = df.copy()
    for c in ("pnl_fees", "pnl_funding"):
        if c not in d:
            d[c] = 0.0
    d["L1_spread_inv"] = d["pnl_spread_capture"] + d["pnl_inventory"]
    d["L2_plus_rebate"] = d["L1_spread_inv"] + d["pnl_fees"]
    d["L3_plus_funding"] = d["L2_plus_rebate"] + d["pnl_funding"]
    d["X_carry_edge"] = d["pnl_spread_capture"] + d["pnl_fees"] + d["pnl_funding"]

    resid = (d["L3_plus_funding"] - d["total_pnl"]).abs().max()
    print(f"  (legs reconstruct total_pnl to within {resid:.2e})")

    for col, label in [("L1_spread_inv", "spread + inventory"),
                       ("L2_plus_rebate", "  + rebate (-1 bp)"),
                       ("L3_plus_funding", "  + funding  (= total)"),
                       ("X_carry_edge", "spread + rebate + funding (no MTM)")]:
        wide = d.pivot_table(index="epoch", columns="strategy", values=col)
        sign(wide, label)

    cols = ["pnl_spread_capture", "pnl_inventory", "pnl_fees", "pnl_funding",
            "total_pnl"]
    names = ["spread", "inventory", "rebate", "funding", "total"]
    order = [s for s in STRATS if s in set(d["strategy"])]
    for how, agg in (("pooled sum", "sum"), ("median per epoch", "median")):
        t = d.groupby("strategy")[cols].agg(agg).reindex(order)
        t.index = [SHORT[s] for s in t.index]
        t.columns = names
        print(f"\n  {how} over {d['epoch'].nunique()} epochs:")
        with pd.option_context("display.float_format", "{:+,.0f}".format):
            print(t.to_string())
    print("\n  Sums are fragile here: a single flash move can dominate them. On"
          "\n  BTC epoch 68 (2026-09-20, 81,200 -> 153,960 -> 81,300 in ~1.5 s) one"
          "\n  epoch is 89-98% of the pooled AS/GLFT loss in the earlier sweep."
          "\n  Compare against --exclude-epochs 68, and prefer the medians.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="results/sweep_funding")
    ap.add_argument("--new-fee", type=float, default=-1.0)
    ap.add_argument("--old", default="results/sweep")
    ap.add_argument("--old-fee", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.001)
    ap.add_argument("--symbol", default="tBTCF0_USTF0")
    ap.add_argument("--exclude-epochs", type=int, nargs="*", default=[],
                    help="drop these epochs (e.g. 68, the only flash move in "
                         "the dataset) from every section")
    args = ap.parse_args()

    def pick(root: str, fee: float) -> pd.DataFrame:
        df = load(Path(root), fee)
        if df.empty:
            return df
        return df[(df["symbol"] == args.symbol)
                  & np.isclose(df["gamma"], args.gamma)
                  & ~df["epoch"].isin(args.exclude_epochs)].reset_index(drop=True)

    new = pick(args.new, args.new_fee)
    if new.empty:
        print(f"no completed epochs under {args.new} at fee {args.new_fee}")
        return 1
    print(f"{args.symbol}  gamma={args.gamma:g}  "
          f"{new['epoch'].nunique()} epochs with funding, fee {args.new_fee:+g} bp"
          + (f"  (excluding {args.exclude_epochs})" if args.exclude_epochs else ""))

    old = pick(args.old, args.old_fee)
    clean = regression_check(new, old) if not old.empty else True
    if not clean:
        print("\n  !! replay is not reproducible across the two sweeps; "
              "sections 2-3 are unreliable until this is explained")
    funding_table(new)
    legs(new)
    return 0 if clean else 3


if __name__ == "__main__":
    sys.exit(main())
