#!/usr/bin/env python3
"""Resumable sweep: run `compare` on every usable epoch of a symbol, for one gamma.

Each epoch runs in its own subprocess, so a crash or a memory spike on one
epoch cannot take down the sweep, and memory is released between epochs.
Results go to  <out>/<symbol>/gamma_<g>/epoch_<n>/ . A DONE marker is written
only after a run exits cleanly, so re-running the same command skips finished
epochs and retries failed ones.

    python scripts/sweep_epochs.py --gamma 0.001
    python scripts/sweep_epochs.py --gamma 0.001 --symbol tETHF0:USTF0
    python scripts/sweep_epochs.py --gamma 0.001 --dry-run     # list only

Note: the CLI calibrates kappa and A on the same epoch it replays (in-sample).
That is identical across strategies, so orderings between them are fair, but
this sweep is not a walk-forward evaluation.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mmbacktest.config import Config  # noqa: E402
from mmbacktest.data.clickhouse_client import MBODataStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="tBTCF0:USTF0")
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--fees", default="0", help="passed to compare --fees")
    ap.add_argument("--min-minutes", type=float, default=120.0)
    ap.add_argument("--include-epoch0", action="store_true",
                    help="epoch 0 may be internally discontinuous; excluded by default")
    ap.add_argument("--out", default="results/sweep")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--epochs", default=None,
                    help="comma-separated epoch list; overrides --min-minutes. "
                         "Use it to re-run exactly the epochs of an earlier sweep.")
    ap.add_argument("--cli-args", default="",
                    help="extra global CLI arguments, e.g. "
                         "'--calibration walk_forward --no-funding'")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg_path = ROOT / args.config
    cfg = Config.load(str(cfg_path)) if cfg_path.exists() else Config()

    store = MBODataStore(cfg.clickhouse)
    epochs = store.list_epochs(args.symbol, args.min_minutes)
    store.close()
    if not epochs:
        print("no usable epochs found (is Tailscale up?)", file=sys.stderr)
        return 1

    if args.epochs:
        wanted = {int(x) for x in args.epochs.split(",") if x.strip()}
        store = MBODataStore(cfg.clickhouse)
        pool = store.list_epochs(args.symbol, 0.0, exclude_compromised=False)
        store.close()
        epochs = [e for e in pool if e.epoch in wanted]
        missing = wanted - {e.epoch for e in epochs}
        if missing:
            print(f"requested epochs not found: {sorted(missing)}", file=sys.stderr)
            return 1
    elif not args.include_epoch0:
        epochs = [e for e in epochs if e.epoch != 0]
    # Shortest first: early failures surface quickly, and partial results
    # cover many epochs rather than a few long ones.
    epochs.sort(key=lambda e: e.minutes)

    sym_tag = args.symbol.replace(":", "_")
    base = ROOT / args.out / sym_tag / f"gamma_{args.gamma:g}"
    base.mkdir(parents=True, exist_ok=True)

    todo = [e for e in epochs if not (base / f"epoch_{e.epoch}" / "DONE").exists()]
    total_h = sum(e.minutes for e in todo) / 60
    print(f"{args.symbol} gamma={args.gamma:g}: {len(epochs)} usable epochs, "
          f"{len(epochs) - len(todo)} done, {len(todo)} to run ({total_h:.0f} h of data)")

    if args.dry_run:
        for e in todo:
            print(f"  epoch {e.epoch:>4}  {e.minutes / 60:6.1f} h  {e.n_events:>12,} events")
        return 0

    failed: list[int] = []
    t_start = time.time()
    for i, e in enumerate(todo, 1):
        edir = base / f"epoch_{e.epoch}"
        edir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(ROOT / "scripts" / "run_backtest.py"),
            "--config", str(cfg_path),
            "--symbol", args.symbol,
            "--epoch", str(e.epoch),
            "--gamma", str(args.gamma),
            "--out", str(edir),
            *shlex.split(args.cli_args),
            "compare", "--fees", args.fees,
        ]
        t0 = time.time()
        with open(edir / "run.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        dt = time.time() - t0

        if rc == 0 and any(edir.glob("metrics_*.json")):
            (edir / "DONE").write_text(f"{dt:.0f}s\n")
            status = "ok"
        else:
            failed.append(e.epoch)
            status = f"FAILED rc={rc} (see {edir / 'run.log'})"

        elapsed = time.time() - t_start
        eta = elapsed / i * (len(todo) - i)
        print(f"[{i}/{len(todo)}] epoch {e.epoch} ({e.minutes / 60:.1f} h) "
              f"{dt / 60:.1f} min  {status}  | eta {eta / 60:.0f} min", flush=True)

    print(f"\nfinished. failed epochs: {failed or 'none'}")
    if failed:
        print("re-run the same command to retry them")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
