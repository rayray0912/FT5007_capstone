#!/usr/bin/env python3
"""Audit the collector's dataset before trusting any replay built on it.

Run this first, on any machine that can reach the collector. It answers the
questions that decide whether a backtest is worth running at all: is the data
intact, which continuous segments are long enough to replay, and does the
trade flow support calibrating fill intensity.

    python scripts/audit_data.py
    python scripts/audit_data.py --symbol tETHF0:USTF0 --min-minutes 240
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mmbacktest.config import Config  # noqa: E402
from mmbacktest.data.clickhouse_client import MBODataStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--min-minutes", type=float, default=120.0)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = Config.load(str(cfg_path)) if cfg_path.exists() else Config()

    store = MBODataStore(cfg.clickhouse)
    if not store.ping():
        print(
            f"cannot reach ClickHouse at {cfg.clickhouse.host}:{cfg.clickhouse.port}\n"
            "Is Tailscale up on this machine?",
            file=sys.stderr,
        )
        return 1

    print("=" * 72)
    print("COVERAGE")
    print("=" * 72)
    print(store.list_symbols().to_string(index=False))

    print()
    print("=" * 72)
    print("INTEGRITY")
    print("=" * 72)
    print(store.audit().to_string(index=False))

    ok, detail = store.integrity_ok()
    print()
    if ok:
        print(f"  PASS  {detail}")
        print("  No silent book drift: every interruption is explicit and locatable.")
    else:
        print(f"  FAIL  {detail}")
        print("  Replays over the affected windows cannot be trusted.")

    symbols = (
        [args.symbol] if args.symbol
        else store.list_symbols()["symbol"].tolist()
    )

    print()
    print("=" * 72)
    print(f"CONTINUOUS SEGMENTS  (>= {args.min_minutes:.0f} min)")
    print("=" * 72)
    for sym in symbols:
        epochs = store.list_epochs(sym, args.min_minutes)
        total_h = sum(e.minutes for e in epochs) / 60.0
        print(f"\n{sym}: {len(epochs)} usable segments, {total_h:.1f} h total")
        for e in epochs[:args.top]:
            print(f"    {e}")
        if len(epochs) > args.top:
            print(f"    ... and {len(epochs) - args.top} more")

    store.close()

    print()
    print("=" * 72)
    print("Replays are constrained to a single epoch. Crossing an epoch boundary")
    print("produces ghost orders: orders whose delete message fell into the gap")
    print("and which therefore never leave the reconstructed book.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
