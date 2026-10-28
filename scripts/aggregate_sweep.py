#!/usr/bin/env python3
"""Aggregate a sweep into one table and test the strategy orderings across epochs.

    python scripts/aggregate_sweep.py
    python scripts/aggregate_sweep.py --root results/sweep --fee 0

Writes  <root>/summary_epochs.csv  (one row per symbol x gamma x epoch x strategy)
and prints paired sign tests on the behavioural metrics. PnL is included in the
CSV for completeness but deliberately not tested: on this data it is dominated
by inventory noise.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

STRATS = ["symmetric", "avellaneda_stoikov", "glft"]
SHORT = {"symmetric": "SYM", "avellaneda_stoikov": "AS", "glft": "GLFT"}

# (metric, direction): +1 means higher is better, -1 means lower is better
TESTS = [
    ("spread_capture_bps", +1),
    ("inventory_variance", -1),
    ("mean_abs_inventory", -1),
]
PAIRS = [("avellaneda_stoikov", "symmetric"),
         ("glft", "symmetric"),
         ("glft", "avellaneda_stoikov")]


def load(root: Path, fee: float) -> pd.DataFrame:
    rows = []
    for mfile in root.glob("*/gamma_*/epoch_*/metrics_*.json"):
        edir = mfile.parent
        if not (edir / "DONE").exists():
            continue
        symbol = edir.parent.parent.name
        gamma = float(edir.parent.name.split("_", 1)[1])
        epoch = int(edir.name.split("_", 1)[1])
        payload = json.loads(mfile.read_text())
        for label, m in payload.get("metrics", {}).items():
            match = re.match(r"(\w+) @ ([+-]?[\d.]+)bp", label)
            if not match:
                continue
            strat, lbl_fee = match.group(1), float(match.group(2))
            if abs(lbl_fee - fee) > 1e-9:
                continue
            row = {"symbol": symbol, "gamma": gamma, "epoch": epoch, "strategy": strat}
            row.update({k: v for k, v in m.items() if isinstance(v, (int, float))})
            rows.append(row)
    return pd.DataFrame(rows)


def sign_tests(df: pd.DataFrame) -> None:
    for (symbol, gamma), grp in df.groupby(["symbol", "gamma"]):
        n_ep = grp["epoch"].nunique()
        print(f"\n=== {symbol}  gamma={gamma:g}  ({n_ep} epochs) ===")
        for metric, direction in TESTS:
            if metric not in grp.columns:
                continue
            wide = grp.pivot_table(index="epoch", columns="strategy", values=metric)
            print(f"  {metric}   (median: " + ", ".join(
                f"{SHORT[s]} {wide[s].median():.4g}" for s in STRATS if s in wide) + ")")
            for a, b in PAIRS:
                if a not in wide or b not in wide:
                    continue
                pair = wide[[a, b]].dropna()
                diff = (pair[a] - pair[b]) * direction
                wins = int((diff > 0).sum())
                n = int((diff != 0).sum())
                if n == 0:
                    continue
                p = binomtest(wins, n, 0.5, alternative="two-sided").pvalue
                print(f"    {SHORT[a]:>4} better than {SHORT[b]:<4} on {wins:>3}/{n:<3} epochs"
                      f"   sign-test p = {p:.2g}")


def gamma_effect(df: pd.DataFrame) -> None:
    """Does the gamma tier change behaviour? Median across epochs, per strategy."""
    if df["gamma"].nunique() < 2:
        return
    print("\n=== gamma effect (median across epochs) ===")
    cols = [m for m, _ in TESTS if m in df.columns] + (
        ["n_fills"] if "n_fills" in df.columns else [])
    tbl = df.groupby(["symbol", "strategy", "gamma"])[cols].median()
    with pd.option_context("display.float_format", "{:.4g}".format,
                           "display.width", 200, "display.max_columns", 20):
        print(tbl)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/sweep")
    ap.add_argument("--fee", type=float, default=0.0)
    args = ap.parse_args()

    root = Path(args.root)
    df = load(root, args.fee)
    if df.empty:
        print(f"no completed epochs under {root}")
        return

    out = root / "summary_epochs.csv"
    df.sort_values(["symbol", "gamma", "epoch", "strategy"]).to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows)")

    sign_tests(df)
    gamma_effect(df)


if __name__ == "__main__":
    main()
