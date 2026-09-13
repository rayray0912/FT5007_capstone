"""Access layer for the Bitfinex MBO dataset stored in ClickHouse.

The collector writes three tables:

  book_mbo          order-level book events (snapshot/add/update/delete)
  trades            executed trades
  collector_events  connection lifecycle, used to audit data validity

The important operational detail is `epoch`: it increments on every reconnect.
A replay that crosses an epoch boundary sees ghost orders (an order whose
delete arrived during a gap can never be removed from the reconstructed book),
so every query in this module is epoch-scoped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ..config import ClickHouseConfig, DataConfig

logger = logging.getLogger(__name__)


@dataclass
class EpochInfo:
    """A contiguous collection segment for one symbol."""

    symbol: str
    epoch: int
    start: datetime
    end: datetime
    minutes: float
    n_events: int

    def __str__(self) -> str:
        return (
            f"{self.symbol} epoch={self.epoch} "
            f"{self.start:%Y-%m-%d %H:%M} -> {self.end:%H:%M} "
            f"({self.minutes:.0f} min, {self.n_events:,} events)"
        )


class MBODataStore:
    """Read-only access to the collector's ClickHouse tables."""

    def __init__(self, cfg: ClickHouseConfig):
        self.cfg = cfg
        self._client = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            import clickhouse_connect

            logger.info("connecting to clickhouse at %s:%s", self.cfg.host, self.cfg.port)
            self._client = clickhouse_connect.get_client(
                host=self.cfg.host,
                port=self.cfg.port,
                database=self.cfg.database,
                username=self.cfg.user,
                password=self.cfg.password,
                connect_timeout=self.cfg.connect_timeout,
                send_receive_timeout=self.cfg.send_receive_timeout,
            )
        return self._client

    def ping(self) -> bool:
        try:
            self.client.query("SELECT 1")
            return True
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            logger.error("clickhouse ping failed: %s", exc)
            return False

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_symbols(self) -> pd.DataFrame:
        """Per-symbol row counts and coverage window."""
        return self.client.query_df(
            """
            SELECT symbol,
                   count()        AS n_events,
                   min(ts_exch)   AS first_event,
                   max(ts_exch)   AS last_event,
                   uniqExact(epoch) AS n_epochs
            FROM book_mbo
            GROUP BY symbol
            ORDER BY n_events DESC
            """
        )

    def list_epochs(self, symbol: str, min_minutes: float = 0.0) -> list[EpochInfo]:
        """Contiguous segments for a symbol, longest first.

        A segment is one `epoch` value. Because the collector persists its
        epoch counter across restarts, epoch numbers are globally increasing
        and the boundaries reported here can be taken at face value.
        """
        df = self.client.query_df(
            """
            SELECT epoch,
                   min(ts_exch) AS seg_start,
                   max(ts_exch) AS seg_end,
                   count()      AS n_events
            FROM book_mbo
            WHERE symbol = {symbol:String}
            GROUP BY epoch
            ORDER BY epoch
            """,
            parameters={"symbol": symbol},
        )
        if df.empty:
            return []

        out: list[EpochInfo] = []
        for row in df.itertuples(index=False):
            minutes = (row.seg_end - row.seg_start).total_seconds() / 60.0
            if minutes < min_minutes:
                continue
            out.append(
                EpochInfo(
                    symbol=symbol,
                    epoch=int(row.epoch),
                    start=row.seg_start,
                    end=row.seg_end,
                    minutes=minutes,
                    n_events=int(row.n_events),
                )
            )
        out.sort(key=lambda e: e.minutes, reverse=True)
        return out

    def resolve_epoch(self, data_cfg: DataConfig) -> EpochInfo:
        """Pick the epoch a run should use.

        If the config names an epoch, validate it. Otherwise take the longest
        segment above the minimum-length threshold, which gives the run the
        most continuous replay available.
        """
        epochs = self.list_epochs(data_cfg.symbol, data_cfg.min_epoch_minutes)
        if not epochs:
            raise ValueError(
                f"no epoch for {data_cfg.symbol} longer than "
                f"{data_cfg.min_epoch_minutes} minutes"
            )

        if data_cfg.epoch is None:
            chosen = epochs[0]
            logger.info("auto-selected longest epoch: %s", chosen)
            return chosen

        for e in epochs:
            if e.epoch == data_cfg.epoch:
                return e
        raise ValueError(
            f"epoch {data_cfg.epoch} not found for {data_cfg.symbol} "
            f"(available: {[e.epoch for e in epochs[:10]]})"
        )

    # ------------------------------------------------------------------
    # Event loading
    # ------------------------------------------------------------------

    def load_book_events(
        self,
        symbol: str,
        epoch: int,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Order-level book events for one epoch, in replay order.

        Ordering by (ts_exch, seq) is what makes deterministic replay possible:
        seq is the exchange's own global sequence number, so ties on the
        millisecond timestamp still resolve to the true order of events.

        Delete events carry the last known price and amount (the collector
        back-fills them, since the raw protocol sends price=0 on cancel). That
        is what allows queue depletion to be computed without maintaining a
        separate state machine here.
        """
        clauses = ["symbol = {symbol:String}", "epoch = {epoch:UInt32}"]
        params: dict[str, object] = {"symbol": symbol, "epoch": epoch}
        if start is not None:
            clauses.append("ts_exch >= {start:DateTime64(3)}")
            params["start"] = start
        if end is not None:
            clauses.append("ts_exch <= {end:DateTime64(3)}")
            params["end"] = end

        query = f"""
            SELECT ts_exch, seq, order_id, side, action, price, amount
            FROM book_mbo
            WHERE {' AND '.join(clauses)}
            ORDER BY ts_exch, seq
        """
        logger.info("loading book events: %s epoch=%s", symbol, epoch)
        df = self.client.query_df(query, parameters=params)
        logger.info("loaded %s book events", f"{len(df):,}")
        return df

    def load_trades(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Executed trades in a window.

        Trades are not epoch-scoped: a trade is a point event and carries no
        book state, so a gap in book collection does not invalidate it. They
        are still filtered to the replay window so that intensity calibration
        uses the same period as the replay.
        """
        clauses = ["symbol = {symbol:String}"]
        params: dict[str, object] = {"symbol": symbol}
        if start is not None:
            clauses.append("ts_exch >= {start:DateTime64(3)}")
            params["start"] = start
        if end is not None:
            clauses.append("ts_exch <= {end:DateTime64(3)}")
            params["end"] = end

        query = f"""
            SELECT ts_exch, trade_id, price, amount, side
            FROM trades
            WHERE {' AND '.join(clauses)}
            ORDER BY ts_exch, trade_id
        """
        df = self.client.query_df(query, parameters=params)
        logger.info("loaded %s trades", f"{len(df):,}")
        return df

    # ------------------------------------------------------------------
    # Data quality
    # ------------------------------------------------------------------

    def audit(self, symbol: str | None = None) -> pd.DataFrame:
        """Collector event counts by kind.

        This is the authoritative check on whether the data is usable.
        seq_gap and checksum_fail are the two that matter: a non-zero count of
        either means the reconstructed book may have silently drifted, and any
        replay over the affected period is suspect.
        """
        where = "WHERE symbol = {symbol:String}" if symbol else ""
        params = {"symbol": symbol} if symbol else {}
        return self.client.query_df(
            f"""
            SELECT kind, count() AS n
            FROM collector_events
            {where}
            GROUP BY kind
            ORDER BY n DESC
            """,
            parameters=params,
        )

    def integrity_ok(self, symbol: str | None = None) -> tuple[bool, str]:
        """Boolean gate on the two failure modes that invalidate a replay."""
        df = self.audit(symbol)
        if df.empty:
            return False, "no collector events recorded"

        counts = dict(zip(df["kind"], df["n"]))
        gaps = int(counts.get("seq_gap", 0))
        checksum = int(counts.get("checksum_fail", 0))

        if gaps or checksum:
            return False, f"seq_gap={gaps}, checksum_fail={checksum}"
        return True, "no seq gaps, no checksum failures"
