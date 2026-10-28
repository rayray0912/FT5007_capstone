#!/usr/bin/env python3
"""Backfill mark price and funding rate from Bitfinex REST into ClickHouse.

Why this exists
---------------
The collector does not subscribe to `status:deriv`, so the dataset has no
funding data. For a perpetual swap, funding is a real component of the cost of
holding inventory. Its size scales with position, so it matters most for a
quoter without inventory control; its sign does not favour anyone in
particular. On BTC-PERP epoch 27, a falling market with positive funding
throughout, it cost the naive quoter 368% of its spread capture against 20%
and 28% for AS and GLFT -- but across 14 epochs the naive quoter received
funding in 8 and paid in 6, and it netted to -1,066 against +6,597 of spread
capture. Epoch 27 is not representative of the sign.

It turns out this does not need the collector changed. The public endpoint

    /v2/status/deriv/<SYMBOL>/hist?start=<ms>&end=<ms>

serves history at ~60s resolution for the whole period already collected, so
the gap can be closed offline. (Without explicit start/end it returns only the
last few hours, which is what made this look un-backfillable at first.)

What is and is not recoverable
------------------------------
Recoverable: the settled funding rate at every 8h event, and mark price at
~60s. That is everything the backtest needs -- funding settles 3x a day, and
PnL is marked on the book mid rather than on mark price.

Not recoverable: sub-minute mark price. Nothing here uses it.

Usage
-----
    python scripts/backfill_deriv.py --apply-schema          # create the table
    python scripts/backfill_deriv.py --epochs-min-hours 20   # cover backtest windows
    python scripts/backfill_deriv.py --full                  # whole collected span
    python scripts/backfill_deriv.py --symbol tBTCF0:USTF0 \
        --start '2026-09-05 22:00' --end '2026-09-07 14:00'
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mmbacktest.config import Config  # noqa: E402
from mmbacktest.data.clickhouse_client import MBODataStore  # noqa: E402

log = logging.getLogger("backfill_deriv")

REST = "https://api-pub.bitfinex.com/v2/status/deriv/{symbol}/hist"

# The endpoint caps a response at 5000 rows; at ~60s resolution a 24h chunk is
# ~1440, which leaves headroom for denser stretches without risking a silent
# truncation that would look like a data gap.
CHUNK = timedelta(hours=24)

# Public REST allows ~90 requests/minute. One per second is well inside that
# and makes a full 59-day, 3-symbol backfill take about 3 minutes.
SLEEP_S = 1.0

# Field positions in the /hist payload. NOTE: the `hist` rows omit the leading
# KEY field that the non-hist snapshot carries, so every index here is one
# lower than the figures in Bitfinex's own field table. Getting this wrong
# silently swaps mark price for spot, or funding for the funding step.
IX = {
    "mts": 0,
    "deriv_price": 2,
    "spot_price": 3,
    "insurance_fund": 5,
    "next_funding_ts": 7,
    "next_funding_accrued": 8,
    "next_funding_step": 9,
    "current_funding": 11,
    "mark_price": 14,
    "open_interest": 17,
}

COLS = [
    "symbol", "ts", "deriv_price", "spot_price", "mark_price",
    "insurance_fund", "next_funding_ts", "next_funding_accrued",
    "next_funding_step", "current_funding", "open_interest",
]


def _ms(t: datetime) -> int:
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return int(t.timestamp() * 1000)


def fetch_window(symbol: str, start: datetime, end: datetime,
                 session: requests.Session) -> list[list]:
    """One REST call. Returns raw rows, ascending by timestamp."""
    r = session.get(
        REST.format(symbol=symbol),
        params={"start": _ms(start), "end": _ms(end), "limit": 5000, "sort": 1},
        timeout=30,
    )
    r.raise_for_status()
    rows = r.json()
    if not isinstance(rows, list):
        raise RuntimeError(f"unexpected payload for {symbol}: {rows!r:.200}")
    return rows


def to_frame(symbol: str, rows: list[list]) -> pd.DataFrame:
    """Raw rows -> typed frame. Rows missing a timestamp are dropped."""
    recs = []
    for r in rows:
        if not isinstance(r, list) or len(r) <= IX["open_interest"]:
            continue
        mts = r[IX["mts"]]
        if mts is None:
            continue

        def g(key: str, default: float = 0.0) -> float:
            v = r[IX[key]]
            return default if v is None else float(v)

        nft = r[IX["next_funding_ts"]]
        recs.append({
            "symbol": symbol,
            "ts": pd.Timestamp(int(mts), unit="ms", tz="UTC"),
            "deriv_price": g("deriv_price"),
            "spot_price": g("spot_price"),
            "mark_price": g("mark_price"),
            "insurance_fund": g("insurance_fund"),
            "next_funding_ts": (
                pd.Timestamp(int(nft), unit="ms", tz="UTC") if nft
                else pd.Timestamp(0, unit="ms", tz="UTC")
            ),
            "next_funding_accrued": g("next_funding_accrued"),
            "next_funding_step": int(g("next_funding_step")),
            "current_funding": g("current_funding"),
            "open_interest": g("open_interest"),
        })
    return pd.DataFrame(recs, columns=COLS)


def backfill(store: MBODataStore, symbol: str, start: datetime, end: datetime,
             dry_run: bool = False) -> int:
    """Walk [start, end] in chunks, inserting as we go."""
    session = requests.Session()
    total = 0
    cursor = start
    while cursor < end:
        stop = min(cursor + CHUNK, end)
        try:
            rows = fetch_window(symbol, cursor, stop, session)
        except Exception as exc:  # noqa: BLE001 - surfaced, then retried once
            log.warning("  %s %s: %s -- retrying once", symbol, cursor, exc)
            time.sleep(5.0)
            rows = fetch_window(symbol, cursor, stop, session)

        df = to_frame(symbol, rows)
        if not df.empty:
            # A chunk that comes back exactly at the cap was probably truncated;
            # say so rather than leaving an invisible hole.
            if len(rows) >= 5000:
                log.warning("  %s %s: hit the 5000-row cap, window may be "
                            "truncated", symbol, cursor)
            if dry_run:
                log.info("  [dry-run] %s %s -> %s  %d rows",
                         symbol, cursor.strftime("%m-%d %H:%M"),
                         stop.strftime("%H:%M"), len(df))
            else:
                store.client.insert_df("deriv_status", df)
            total += len(df)
        else:
            log.info("  %s %s -> %s  no rows", symbol,
                     cursor.strftime("%m-%d %H:%M"), stop.strftime("%H:%M"))

        cursor = stop
        time.sleep(SLEEP_S)
    return total


def collected_span(store: MBODataStore, symbol: str) -> tuple[datetime, datetime]:
    df = store.client.query_df(
        "SELECT min(ts_exch) AS a, max(ts_exch) AS b FROM book_mbo "
        "WHERE symbol = {s:String}",
        parameters={"s": symbol},
    )
    return df["a"].iloc[0].to_pydatetime(), df["b"].iloc[0].to_pydatetime()


def epoch_windows(store: MBODataStore, symbol: str,
                  min_hours: float) -> list[tuple[int, datetime, datetime]]:
    """Epochs long enough to be worth replaying, as (epoch, start, end)."""
    df = store.client.query_df(
        """
        SELECT epoch, min(ts_exch) AS t0, max(ts_exch) AS t1,
               dateDiff('minute', min(ts_exch), max(ts_exch)) AS mins
        FROM book_mbo
        WHERE symbol = {s:String}
        GROUP BY epoch
        HAVING mins >= {m:Float64}
        ORDER BY epoch
        """,
        parameters={"s": symbol, "m": min_hours * 60.0},
    )
    return [
        (int(r.epoch), r.t0.to_pydatetime(), r.t1.to_pydatetime())
        for r in df.itertuples(index=False)
    ]


def coverage_report(store: MBODataStore, symbol: str) -> None:
    df = store.client.query_df(
        """
        SELECT count() AS rows, min(ts) AS first, max(ts) AS last,
               uniqExact(next_funding_ts) AS settlements,
               round(avg(next_funding_accrued) * 100, 5) AS avg_rate_pct,
               round(avg(mark_price - spot_price), 2) AS avg_basis
        FROM deriv_status WHERE symbol = {s:String}
        """,
        parameters={"s": symbol},
    )
    if df.empty or int(df["rows"].iloc[0]) == 0:
        print(f"  {symbol}: nothing stored")
        return
    r = df.iloc[0]
    print(f"  {symbol}: {int(r['rows']):,} rows  {r['first']} -> {r['last']}")
    print(f"      {int(r['settlements']):,} funding settlements, "
          f"mean accrued {r['avg_rate_pct']}%/8h, "
          f"mean mark-spot basis {r['avg_basis']} USD")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--symbol", action="append", dest="symbols",
                    help="repeatable; default is all three perpetuals")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--full", action="store_true",
                    help="cover the whole collected span for each symbol")
    ap.add_argument("--epochs-min-hours", type=float,
                    help="cover only epochs at least this long (with margin)")
    ap.add_argument("--apply-schema", action="store_true",
                    help="create deriv_status and v_funding, then exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true", help="coverage only")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    cfg_path = Path(args.config)
    cfg = Config.load(str(cfg_path)) if cfg_path.exists() else Config()
    store = MBODataStore(cfg.clickhouse)
    if not store.ping():
        print(f"cannot reach ClickHouse at {cfg.clickhouse.host}:"
              f"{cfg.clickhouse.port}", file=sys.stderr)
        return 1

    if args.apply_schema:
        ddl = (Path(__file__).resolve().parent.parent / "sql" / "deriv_status.sql")
        # Strip comment lines before splitting on ';'. Splitting first breaks
        # on a semicolon inside a comment -- "status:deriv; this table ..." --
        # which hands ClickHouse a fragment of prose as a statement.
        body = "\n".join(
            line for line in ddl.read_text().splitlines()
            if not line.lstrip().startswith("--")
        )
        stmts = [s.strip() for s in body.split(";") if s.strip()]
        for s in stmts:
            store.client.command(s)
        print(f"applied {len(stmts)} statements from {ddl.name}")
        store.close()
        return 0

    symbols = args.symbols or ["tBTCF0:USTF0", "tETHF0:USTF0", "tXAUTF0:USTF0"]

    if args.report:
        print("deriv_status coverage")
        for s in symbols:
            coverage_report(store, s)
        store.close()
        return 0

    for sym in symbols:
        if args.start and args.end:
            windows = [(None, pd.Timestamp(args.start).to_pydatetime(),
                        pd.Timestamp(args.end).to_pydatetime())]
        elif args.epochs_min_hours:
            windows = epoch_windows(store, sym, args.epochs_min_hours)
            # One settlement either side, so the 8h event bracketing a window
            # edge is present and the rate can be attributed to it.
            windows = [(e, t0 - timedelta(hours=9), t1 + timedelta(hours=9))
                       for e, t0, t1 in windows]
        elif args.full:
            a, b = collected_span(store, sym)
            windows = [(None, a, b + timedelta(hours=1))]
        else:
            print("pick one of --full, --epochs-min-hours, or --start/--end",
                  file=sys.stderr)
            return 2

        log.info("%s: %d window(s)", sym, len(windows))
        got = 0
        for epoch, t0, t1 in windows:
            tag = f"epoch {epoch}" if epoch is not None else "range"
            log.info("  %s %s  %s -> %s (%.1f h)", sym, tag,
                     t0.strftime("%m-%d %H:%M"), t1.strftime("%m-%d %H:%M"),
                     (t1 - t0).total_seconds() / 3600)
            got += backfill(store, sym, t0, t1, dry_run=args.dry_run)
        log.info("%s: %s rows", sym, f"{got:,}")

    if not args.dry_run:
        print()
        print("coverage after backfill")
        for s in symbols:
            coverage_report(store, s)

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
