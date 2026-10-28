#!/usr/bin/env python3
"""In-sample vs walk-forward calibration, on the same epochs.

    python scripts/compare_calibration.py
    python scripts/compare_calibration.py --exclude-epochs 68

The September sweep fitted kappa and A on the epoch it then replayed. The
walk-forward sweep fits them only on the 24 h before each epoch starts. Same
epochs, same gamma, zero fees, no funding: calibration is the only change.

Sections:
1. Symmetric sanity check. Symmetric ignores kappa and A, so its results must
   be identical under both modes; anything else means a second variable moved.
2. The behavioural ordering under each mode, side by side (paired sign tests).
   The acceptance question from the handoff: does it survive walk-forward?
3. What walk-forward changes, strategy by strategy (paired, median of the
   per-epoch difference).
4. How far the walk-forward kappa and A sit from the in-sample fit.
5. Sections 2 and 4 split by epoch length. For an epoch shorter than the 24 h
   window, walk-forward fits on more trades than in-sample did, so the two
   modes differ in sample size as well as in lookahead. For a longer epoch the
   in-sample fit has more data. If the ordering holds in both groups, neither
   lookahead nor sample size is what produces it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_sweep import PAIRS, SHORT, STRATS, TESTS, load  # noqa: E402

SYM_CHECK = ["n_fills", "spread_capture_bps", "inventory_variance",
             "mean_abs_inventory", "pnl_inventory"]


def pick(root: str, gamma: float, symbol: str, excl: list[int]) -> pd.DataFrame:
    df = load(Path(root), 0.0)
    if df.empty:
        return df
    return df[(df["symbol"] == symbol) & np.isclose(df["gamma"], gamma)
              & ~df["epoch"].isin(excl)].reset_index(drop=True)


def calib_params(root: str, gamma: float, symbol: str) -> pd.DataFrame:
    """kappa, A per epoch: from the JSON when recorded, else from run.log."""
    rows = []
    base = Path(root) / symbol / f"gamma_{gamma:g}"
    for edir in base.glob("epoch_*"):
        if not (edir / "DONE").exists():
            continue
        epoch = int(edir.name.split("_", 1)[1])
        cal = {}
        for mf in edir.glob("metrics_*.json"):
            cal = json.loads(mf.read_text()).get("calibration") or {}
        if "kappa" in cal:
            rows.append({"epoch": epoch, "kappa": cal["kappa"], "A": cal["A"],
                         "r2": cal.get("r_squared"), "usable": cal.get("usable")})
            continue
        log = edir / "run.log"
        if log.exists():
            m = re.search(r"intensity(?: \([^)]*\))?: A=([\d.eE+-]+) "
                          r"kappa=([\d.eE+-]+) R2=([\d.eE+-]+)", log.read_text())
            if m:
                rows.append({"epoch": epoch, "A": float(m.group(1)),
                             "kappa": float(m.group(2)), "r2": float(m.group(3)),
                             "usable": None})
    return pd.DataFrame(rows)


def symmetric_check(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    sa = a[a["strategy"] == "symmetric"].set_index("epoch")
    sb = b[b["strategy"] == "symmetric"].set_index("epoch")
    common = sa.index.intersection(sb.index)
    print(f"\n=== 1. symmetric sanity check ({len(common)} shared epochs) ===")
    ok = True
    for col in SYM_CHECK:
        d = (sa.loc[common, col] - sb.loc[common, col]).abs()
        tol = 1e-6 * np.maximum(1.0, sa.loc[common, col].abs())
        bad = int((d > tol).sum())
        ok &= bad == 0
        print(f"  {col:<22} {'identical' if bad == 0 else f'DIFFERS on {bad}'}")
    return ok


def ordering(df: pd.DataFrame) -> dict:
    out = {}
    for metric, direction in TESTS:
        wide = df.pivot_table(index="epoch", columns="strategy", values=metric)
        for a, b in PAIRS:
            p = wide[[a, b]].dropna()
            d = (p[a] - p[b]) * direction
            wins, n = int((d > 0).sum()), int((d != 0).sum())
            out[(metric, a, b)] = (wins, n, binomtest(wins, n, 0.5).pvalue if n else np.nan)
    return out


def side_by_side(ins: pd.DataFrame, wf: pd.DataFrame) -> None:
    common = sorted(set(ins["epoch"]) & set(wf["epoch"]))
    ins, wf = ins[ins["epoch"].isin(common)], wf[wf["epoch"].isin(common)]
    oi, ow = ordering(ins), ordering(wf)
    print(f"\n=== 2. behavioural ordering, same {len(common)} epochs ===")
    print(f"  {'metric':<20} {'pair':<14} {'in-sample':>18} {'walk-forward':>18}")
    for metric, _ in TESTS:
        for a, b in PAIRS:
            wi, ni, pi = oi[(metric, a, b)]
            ww, nw, pw = ow[(metric, a, b)]
            flag = "" if (wi > ni / 2) == (ww > nw / 2) else "   <-- FLIPPED"
            print(f"  {metric:<20} {SHORT[a]+' > '+SHORT[b]:<14} "
                  f"{wi:>3}/{ni:<3} p={pi:<8.2g} {ww:>3}/{nw:<3} p={pw:<8.2g}{flag}")


def what_changes(ins: pd.DataFrame, wf: pd.DataFrame) -> None:
    print("\n=== 3. effect of walk-forward, per strategy (paired over epochs) ===")
    cols = ["spread_capture_bps", "inventory_variance", "mean_abs_inventory", "n_fills"]
    for s in STRATS[1:]:
        a = ins[ins["strategy"] == s].set_index("epoch")
        b = wf[wf["strategy"] == s].set_index("epoch")
        common = a.index.intersection(b.index)
        print(f"  {SHORT[s]}  ({len(common)} epochs)")
        for c in cols:
            d = (b.loc[common, c] - a.loc[common, c]).dropna()
            rel = (d / a.loc[common, c].abs().replace(0, np.nan)).dropna()
            nz = d[d != 0]
            p = wilcoxon(nz).pvalue if len(nz) >= 10 else np.nan
            print(f"    {c:<20} median change {d.median():+.4g} "
                  f"({rel.median():+.1%})   up on {int((d > 0).sum())}/{len(d)}"
                  f"   wilcoxon p={p:.2g}")


def params(ip: pd.DataFrame, wp: pd.DataFrame) -> None:
    m = ip.merge(wp, on="epoch", suffixes=("_in", "_wf"))
    print(f"\n=== 4. calibrated parameters, {len(m)} epochs ===")
    if m.empty:
        print("  none matched")
        return
    for k in ("kappa", "A"):
        r = m[f"{k}_wf"] / m[f"{k}_in"] - 1
        print(f"  {k:<5} in-sample median {m[f'{k}_in'].median():.4g}  "
              f"walk-forward median {m[f'{k}_wf'].median():.4g}   "
              f"wf/in - 1: median {r.median():+.1%}, IQR [{r.quantile(.25):+.1%}, "
              f"{r.quantile(.75):+.1%}]")
    if "usable_wf" in m:
        u = m["usable_wf"].dropna()
        if len(u):
            print(f"  walk-forward fits usable: {int(u.sum())}/{len(u)}")


def by_length(ins: pd.DataFrame, wf: pd.DataFrame, ip: pd.DataFrame,
              wp: pd.DataFrame, hours: float) -> None:
    dur = ins.groupby("epoch")["duration_hours"].first()
    common = sorted(set(ins["epoch"]) & set(wf["epoch"]))
    groups = [(f"shorter than {hours:g} h (walk-forward fits on more data)",
               [e for e in common if dur[e] < hours]),
              (f"{hours:g} h or longer (in-sample fits on more data)",
               [e for e in common if dur[e] >= hours])]
    m = ip.merge(wp, on="epoch", suffixes=("_in", "_wf"))
    print(f"\n=== 5. split by epoch length ===")
    for label, eps in groups:
        print(f"  {label}: {len(eps)} epochs")
        if not eps:
            continue
        oi = ordering(ins[ins["epoch"].isin(eps)])
        ow = ordering(wf[wf["epoch"].isin(eps)])
        for metric, _ in TESTS:
            for a, b in PAIRS:
                wi, ni, pi = oi[(metric, a, b)]
                ww, nw, pw = ow[(metric, a, b)]
                flag = "" if (wi > ni / 2) == (ww > nw / 2) else "   <-- FLIPPED"
                print(f"    {metric:<20} {SHORT[a]+' > '+SHORT[b]:<14} "
                      f"{wi:>3}/{ni:<3} p={pi:<8.2g} {ww:>3}/{nw:<3} p={pw:<8.2g}{flag}")
        g = m[m["epoch"].isin(eps)]
        for k in ("kappa", "A"):
            r = g[f"{k}_wf"] / g[f"{k}_in"] - 1
            if len(r):
                print(f"    {k:<5} wf/in - 1: median {r.median():+.1%}, "
                      f"median |.| {r.abs().median():.1%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-sample", default="results/sweep")
    ap.add_argument("--walk-forward", default="results/sweep_wf")
    ap.add_argument("--gamma", type=float, default=0.001)
    ap.add_argument("--symbol", default="tBTCF0_USTF0")
    ap.add_argument("--exclude-epochs", type=int, nargs="*", default=[])
    ap.add_argument("--calib-hours", type=float, default=24.0,
                    help="walk-forward window used by the sweep (splits section 5)")
    args = ap.parse_args()

    ins = pick(args.in_sample, args.gamma, args.symbol, args.exclude_epochs)
    wf = pick(args.walk_forward, args.gamma, args.symbol, args.exclude_epochs)
    if ins.empty or wf.empty:
        print("missing results for one of the two modes")
        return 1
    print(f"{args.symbol} gamma={args.gamma:g}: in-sample {ins['epoch'].nunique()} "
          f"epochs, walk-forward {wf['epoch'].nunique()}"
          + (f"  (excluding {args.exclude_epochs})" if args.exclude_epochs else ""))
    missing = sorted(set(ins["epoch"]) - set(wf["epoch"]))
    if missing:
        print(f"  not in walk-forward (insufficient history or not run): {missing}")

    ok = symmetric_check(ins, wf)
    if not ok:
        print("\n  !! symmetric differs: something besides calibration changed")
    side_by_side(ins, wf)
    what_changes(ins, wf)
    ip = calib_params(args.in_sample, args.gamma, args.symbol)
    wp = calib_params(args.walk_forward, args.gamma, args.symbol)
    params(ip, wp)
    by_length(ins, wf, ip, wp, args.calib_hours)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
