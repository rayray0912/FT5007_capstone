-- Derivative status (mark price, funding) for Bitfinex perpetuals.
--
-- Backfilled from the public REST endpoint
--   /v2/status/deriv/<SYMBOL>/hist?start=&end=
-- by scripts/backfill_deriv.py, NOT written by the collector. The collector
-- does not subscribe to status:deriv; this table exists so that the funding
-- component of inventory carrying cost can be attributed on data that was
-- already collected without it.
--
-- Resolution is ~60s, which is the endpoint's own granularity. That is coarse
-- against the 100ms decision clock and deliberately not used for anything that
-- needs finer: funding settles every 8 hours, and PnL is marked on the book
-- mid, not on mark price. What this table is for is (a) the settled funding
-- rate at each 8h event, and (b) the mark-to-spot basis as a sanity check.
--
-- ReplacingMergeTree on (symbol, ts) so the backfill is idempotent: re-running
-- over an already-covered window collapses rather than duplicating.
CREATE TABLE IF NOT EXISTS bfx.deriv_status
(
    symbol              LowCardinality(String),
    ts                  DateTime64(3, 'UTC')  CODEC(DoubleDelta, ZSTD(3)),
    deriv_price         Float64               CODEC(Gorilla, ZSTD(3)),
    spot_price          Float64               CODEC(Gorilla, ZSTD(3)),
    mark_price          Float64               CODEC(Gorilla, ZSTD(3)),
    insurance_fund      Float64               CODEC(Gorilla, ZSTD(3)),
    next_funding_ts     DateTime64(3, 'UTC')  CODEC(DoubleDelta, ZSTD(3)),
    next_funding_accrued Float64              CODEC(Gorilla, ZSTD(3)),
    next_funding_step   Int64                 CODEC(DoubleDelta, ZSTD(1)),
    current_funding     Float64               CODEC(Gorilla, ZSTD(3)),
    open_interest       Float64               CODEC(Gorilla, ZSTD(3))
)
ENGINE = ReplacingMergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (symbol, ts)
SETTINGS index_granularity = 8192;

-- Settled funding rate per 8h event: the last accrued value observed before
-- each settlement timestamp. This is the number actually charged, and the one
-- the backtest consumes.
CREATE VIEW IF NOT EXISTS bfx.v_funding AS
SELECT
    symbol,
    next_funding_ts                            AS settles_at,
    argMax(next_funding_accrued, ts)           AS rate,
    argMax(mark_price, ts)                     AS mark_at_settle,
    count()                                    AS n_samples,
    max(ts)                                    AS last_sample
FROM bfx.deriv_status
WHERE next_funding_ts > toDateTime64(0, 3)
GROUP BY symbol, settles_at
ORDER BY symbol, settles_at;
