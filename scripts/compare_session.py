#!/usr/bin/env python3
"""Static vs session-switching kappa/A, paired by epoch (HANDOFF item 3).

    python scripts/compare_session.py
    python scripts/compare_session.py --exclude-epochs 68

Both arms calibrate walk-forward on the 24 h before each epoch, at the same
gamma, zero fees and no funding. The static arm (results/sweep_wf) quotes on
the pooled fit all day; the session arm (results/session) switches kappa/A at
the UTC session boundaries. Nothing else differs.

Sections:
1. Symmetric sanity check: it ignores kappa and A, so must be identical.
2. Session vs static per strategy: median paired change, sign test, Wilcoxon.
   "Better" is direction-aware (lower inventory variance is better). Total PnL
   is shown but is inventory noise on this data; spread capture and inventory
   variance are the primary metrics.
3. How often the session arm actually ran on a session fit, rather than
   falling back to the pooled one.
4. Is there intraday structure to exploit at all? For each epoch and session,
   does the walk-forward *session* kappa predict that session's in-sample
   kappa (fitted on the replay itself) better than the walk-forward *pooled*
   kappa does? If not, switching cannot help however it is wired.

Writes <session root>/paired_<gamma>.csv and summary_<gamma>.txt.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_sweep import SHORT, load  # noqa: E402

SYM_CHECK = ["n_fills", "spread_capture_bps", "inventory_variance",
             "mean_abs_inventory", "pnl_inventory"]
# (metric, +1 if higher is better, -1 if lower is better, 0 if neither)
METRICS = [
    ("spread_capture_bps", +1),
    ("pnl_spread_capture", +1),
    ("inventory_variance", -1),
    ("mean_abs_inventory", -1),
    ("adverse_selection_bps", -1),
    ("n_fills", 0),
    ("total_pnl", +1),
]


def pick(root: str, gamma: float, symbol: str, excl: list[int]) -> pd.DataFrame:
    df = load(Path(root), 0.0)
    if df.empty:
        return df
    return df[(df["symbol"] == symbol) & np.isclose(df["gamma"], gamma)
              & ~df["epoch"].isin(excl)].reset_index(drop=True)


def symmetric_check(st: pd.DataFrame, se: pd.DataFrame) -> bool:
    a = st[st["strategy"] == "symmetric"].set_index("epoch")
    b = se[se["strategy"] == "symmetric"].set_index("epoch")
    common = a.index.intersection(b.index)
    print(f"\n=== 1. symmetric sanity check ({len(common)} shared epochs) ===")
    ok = True
    for col in SYM_CHECK:
        d = (a.loc[common, col] - b.loc[common, col]).abs()
        bad = int((d > 1e-6 * np.maximum(1.0, a.loc[common, col].abs())).sum())
        ok &= bad == 0
        print(f"  {col:<22} {'identical' if bad == 0 else f'DIFFERS on {bad}'}")
    return ok


def paired(st: pd.DataFrame, se: pd.DataFrame) -> pd.DataFrame:
    print("\n=== 2. session vs static, paired over epochs ===")
    print(f"  {'':<6}{'metric':<24}{'median static':>14}{'median Δ':>12}"
          f"{'Δ %':>8}{'better':>10}{'sign p':>9}{'wilcoxon p':>12}")
    rows = []
    for s in ("avellaneda_stoikov", "glft"):
        a = st[st["strategy"] == s].set_index("epoch")
        b = se[se["strategy"] == s].set_index("epoch")
        common = a.index.intersection(b.index)
        for metric, direction in METRICS:
            x, y = a.loc[common, metric], b.loc[common, metric]
            d = (y - x).dropna()
            rel = (d / x.abs().replace(0, np.nan)).dropna()
            nz = d[d != 0]
            if direction:
                better = int(((d * direction) > 0).sum())
                n = int((d != 0).sum())
                p_sign = binomtest(better, n, 0.5).pvalue if n else np.nan
                better_s = f"{better}/{n}"
            else:
                better, n, p_sign, better_s = np.nan, len(nz), np.nan, "-"
            p_w = wilcoxon(nz).pvalue if len(nz) >= 10 else np.nan
            rows.append({"strategy": s, "metric": metric, "n_epochs": len(common),
                         "median_static": x.median(), "median_session": y.median(),
                         "median_change": d.median(), "median_rel_change": rel.median(),
                         "n_better": better, "n_nonzero": n,
                         "sign_p": p_sign, "wilcoxon_p": p_w})
            print(f"  {SHORT[s]:<6}{metric:<24}{x.median():>14.4g}{d.median():>+12.4g}"
                  f"{rel.median():>+8.1%}{better_s:>10}{p_sign:>9.2g}{p_w:>12.2g}")
    return pd.DataFrame(rows)


def calib_meta(root: str, gamma: float, symbol: str) -> dict[int, dict]:
    out = {}
    for mf in (Path(root) / symbol / f"gamma_{gamma:g}").glob("epoch_*/metrics_*.json"):
        if (mf.parent / "DONE").exists():
            out[int(mf.parent.name.split("_", 1)[1])] = json.loads(
                mf.read_text()).get("calibration") or {}
    return out


def fallback_report(se: pd.DataFrame, meta: dict[int, dict]) -> None:
    print("\n=== 3. how much of the session arm ran on session fits ===")
    col = "frac_decisions_intensity_fallback"
    if col in se:
        f = se[se["strategy"] != "symmetric"].groupby("epoch")[col].first()
        print(f"  decisions on the pooled fallback: median {f.median():.1%}, "
              f"mean {f.mean():.1%}, max {f.max():.1%}; "
              f"{int((f == 0).sum())}/{len(f)} epochs never fell back")
    counts: dict[str, int] = {}
    for m in meta.values():
        for lb in m.get("fallback_sessions", []):
            counts[lb] = counts.get(lb, 0) + 1
    if counts:
        print("  sessions with an unusable walk-forward fit (epochs): "
              + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


_SESSION_LINE = re.compile(r"session=\s*(\S+)\s+A=\s*[\d.eE+-]+/s\s+kappa=\s*([\d.eE+-]+)"
                           r"\s+R2=([\d.eE+-]+)\s+n=([\d,]+)(\s+\[UNUSABLE\])?")


def in_sample_sessions(root: str, gamma: float, symbol: str) -> dict[int, dict]:
    """Per-session kappa fitted on the replay itself, from the in-sample run.log.

    Only kappa is used: the in-sample A was divided by an hours/24 exposure,
    which is wrong for epochs that are not whole days. kappa is a slope and
    does not depend on exposure.
    """
    out = {}
    for log in (Path(root) / symbol / f"gamma_{gamma:g}").glob("epoch_*/run.log"):
        ep = int(log.parent.name.split("_", 1)[1])
        sess = {}
        for m in _SESSION_LINE.finditer(log.read_text()):
            sess[m.group(1)] = {"kappa": float(m.group(2)),
                                "usable": m.group(5) is None}
        out[ep] = sess
    return out


def structure(meta: dict[int, dict], ins: dict[int, dict], dur: pd.Series) -> None:
    print("\n=== 4. does the session kappa from the day before predict the replay's? ===")
    rows = []
    for ep, m in meta.items():
        pooled_wf = m.get("kappa")
        for lb, wf in (m.get("sessions") or {}).items():
            target = ins.get(ep, {}).get(lb)
            if not (target and target["usable"] and wf.get("usable")):
                continue
            rows.append({"epoch": ep, "session": lb, "k_in": target["kappa"],
                         "k_wf_sess": wf["kappa"], "k_wf_pooled": pooled_wf})
    df = pd.DataFrame(rows)
    if df.empty:
        print("  no usable session pairs")
        return
    df["err_sess"] = (np.log(df["k_wf_sess"] / df["k_in"])).abs()
    df["err_pooled"] = (np.log(df["k_wf_pooled"] / df["k_in"])).abs()
    wins = int((df["err_sess"] < df["err_pooled"]).sum())
    n = int((df["err_sess"] != df["err_pooled"]).sum())
    print(f"  {len(df)} (epoch, session) pairs where both fits are usable")
    print(f"  median |log error| vs the replay's session kappa: "
          f"session fit {df['err_sess'].median():.3f}, pooled fit "
          f"{df['err_pooled'].median():.3f}")
    print(f"  session fit closer on {wins}/{n}, sign p={binomtest(wins, n, .5).pvalue:.2g}")
    long_eps = set(dur[dur >= 6].index)
    g = df[df["epoch"].isin(long_eps)]
    if len(g):
        w = int((g["err_sess"] < g["err_pooled"]).sum())
        print(f"  epochs >= 6 h only ({g['epoch'].nunique()} epochs, {len(g)} pairs): "
              f"session closer on {w}/{len(g)}, sign p={binomtest(w, len(g), .5).pvalue:.2g}")
    # Is there a stable intraday pattern at all? Session kappa relative to the
    # epoch's own pooled level, averaged across epochs.
    rel = (df.assign(rel=np.log(df["k_wf_sess"] / df["k_wf_pooled"]))
             .groupby("session")["rel"].agg(["median", "count"]))
    print("  walk-forward session kappa / pooled kappa, median across epochs:")
    for lb, r in rel.iterrows():
        print(f"    {lb}: {np.exp(r['median']) - 1:+.1%}  (n={int(r['count'])})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", default="results/sweep_wf")
    ap.add_argument("--session", default="results/session")
    ap.add_argument("--in-sample", default="results/sweep",
                    help="in-sample sweep whose run.log holds each replay's own session kappa")
    ap.add_argument("--gamma", type=float, default=0.001)
    ap.add_argument("--symbol", default="tBTCF0_USTF0")
    ap.add_argument("--exclude-epochs", type=int, nargs="*", default=[])
    args = ap.parse_args()

    st = pick(args.static, args.gamma, args.symbol, args.exclude_epochs)
    se = pick(args.session, args.gamma, args.symbol, args.exclude_epochs)
    if st.empty or se.empty:
        print("missing results for one of the two arms")
        return 1

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"{args.symbol} gamma={args.gamma:g}: static {st['epoch'].nunique()} "
              f"epochs, session {se['epoch'].nunique()}"
              + (f"  (excluding {args.exclude_epochs})" if args.exclude_epochs else ""))
        missing = sorted(set(st["epoch"]) - set(se["epoch"]))
        if missing:
            print(f"  not in session arm: {missing}")
        ok = symmetric_check(st, se)
        if not ok:
            print("\n  !! symmetric differs: something besides kappa/A changed")
        table = paired(st, se)
        meta = calib_meta(args.session, args.gamma, args.symbol)
        fallback_report(se, meta)
        dur = st.groupby("epoch")["duration_hours"].first()
        structure(meta, in_sample_sessions(args.in_sample, args.gamma, args.symbol), dur)
    print(buf.getvalue(), end="")

    out = Path(args.session)
    tag = f"{args.gamma:g}" + ("_excl" + "_".join(map(str, args.exclude_epochs))
                               if args.exclude_epochs else "")
    table.to_csv(out / f"paired_{tag}.csv", index=False)
    (out / f"summary_{tag}.txt").write_text(buf.getvalue())
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
