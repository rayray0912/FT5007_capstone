# Handoff

Paste the **Opening prompt** below into a fresh Claude Code session in the repo
root. Everything after it is reference material for that session.

Last updated: 2026-10-28. Supersedes the 2026-09-28 version: the 62-epoch γ
sweep has been run since then, and its results change several statements the
old file made (see "What the sweep established").

---

## Opening prompt

> I'm working on my NUS FT5007 capstone (supervisor: Asst. Prof. Lu Yao). It's
> a market-making backtest on order-level (MBO) Bitfinex perpetual data that
> compares GLFT against symmetric and Avellaneda–Stoikov baselines. On top of
> that I'm designing an adaptive parameter-control layer.
>
> Read `HANDOFF.md` in full first. Then read `docs/progress_report.md`
> (section 3 matters most) and `src/mmbacktest/config.py`, which documents the
> non-obvious modelling decisions inline.
>
> State of play: the pipeline runs on real data, and a 3-γ × 62-epoch BTC sweep
> is complete. The results are in `results/sweep/summary_epochs.csv`;
> `python scripts/aggregate_sweep.py` reproduces the sign tests. The next work
> is listed in priority order under "Work queue" in HANDOFF.md.
>
> You have the shell on my Mac, which reaches ClickHouse on my NUC over
> Tailscale. Work through the queue yourself: run things, read the outputs,
> fix what's broken, and commit in small, well-described commits on a branch
> (never on main). Stop and ask me before anything irreversible, such as
> dropping or altering ClickHouse tables, changing the collector on the NUC,
> force-pushing, or deleting results.
>
> Things I care about: don't invent numbers, don't claim something ran if it
> didn't, and tell me when a result looks too good. This project has already
> had to retract a +2% annualised result, and later caught +475% and +40%
> results that were inventory noise. A small run that's understood beats a
> large one that isn't.

---

## 1. Environment

### Paths and machines

| What | Where |
|---|---|
| Repo (Mac) | `/Users/rayhe/Desktop/NUS_Capstone/code/FT5007_capstone` |
| Working branch | `fix/real-data-first-run` (last commit `0c1d907`) |
| Remote | `github.com/rayray0912/FT5007_capstone` |
| Python env | conda env `FT5007` (`scripts/run_sweep.sh` activates it); a `.venv` also works, see below |
| Data | ClickHouse on the NUC, `100.76.49.84:8123` over Tailscale, database `bfx` |
| Tables | `bfx.book_mbo` (order-level book events), plus trades; see `docs/dataset.md` |
| Collector | `bfx-mbo-collector` on the NUC (Singapore), running since 2026-08-30 06:29 UTC |

ClickHouse is reachable **only** from the Mac, or anything else on the tailnet.
A cloud sandbox cannot reach it, which is why this work moves to a Claude Code
session on the Mac.

### Setup (has cost time twice)

```bash
conda activate FT5007            # or:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pyarrow scipy        # neither is in requirements.txt; cli writes parquet, aggregate uses scipy
```

**Local HTTP proxies break the ClickHouse connection.** `clickhouse_connect`
uses urllib3, which honours `HTTP_PROXY`/`HTTPS_PROXY`. A proxy has no route to
a tailnet address. Check reachability with the proxy bypassed first, then
exempt the host:

```bash
curl --noproxy '*' -sS "http://100.76.49.84:8123/?query=SELECT%20version()"
export NO_PROXY="100.76.49.84,localhost,127.0.0.1"; export no_proxy="$NO_PROXY"
```

`MBODataStore.ping()` swallows every exception and reports "cannot reach
ClickHouse", so a missing import and a real network failure look the same.
Check the import before blaming the network.

Add both of these to `requirements.txt` as the first commit:
`pyarrow`, `scipy`.

### Smoke test before anything else

```bash
python scripts/run_backtest.py --epoch 27 --gamma 0.001 --out /tmp/smoke compare --fees 0
```

Expected on BTC epoch 27: κ ≈ 0.104627, 16,168 trades in the window. A
different κ means the data or calibration changed. Find out which before going
further.

### Tests

```bash
pytest -q          # 54 test functions; all must pass before and after every change
```

---

## 2. Repo map

```
src/mmbacktest/
  config.py                 all parameters; bitfinex_tick_size(); session boundaries (0,6,9,13,17,21 UTC)
  cli.py                    `run` / `compare`; single --epoch; calibrates κ, A in-sample on that epoch
  data/clickhouse_client.py MBODataStore, list_epochs, resolve_epoch (epoch-scoped queries)
  data/book.py              MBO order book reconstruction (OrderBook.apply is the hot loop)
  data/synthetic.py         synthetic generator, rescaled to the measured market
  calibration/intensity.py  λ(δ)=A·exp(−κδ) fit; estimate_by_session() exists but is NOT wired into the engine
  calibration/volatility.py σ estimation
  strategy/{symmetric,avellaneda_stoikov,glft}.py, base.py (_finalise: clamp, tick-round, bound suppression)
  sim/engine.py             100 ms decision clock, 80 ms quote latency, quote persistence
  sim/fills.py              queue-position-aware fills
  metrics/performance.py    spread capture, inventory variance, PnL attribution, inventory t-stat, fill provenance
scripts/
  audit_data.py             dataset integrity audit
  sweep_epochs.py           resumable per-epoch sweep (one subprocess per epoch, DONE markers)
  aggregate_sweep.py        → results/sweep/summary_epochs.csv + paired sign tests + γ-effect table
  run_sweep.sh              launches γ ∈ {1e-4, 1e-3, 1e-2} in parallel (nohup + caffeinate)
docs/progress_report.md     current written results; section 3 says what survives scrutiny
docs/dataset.md             schema, collection, integrity
config/default.yaml         default config (note: gamma 0.1, q_max 10, horizon_seconds 60)
```

**Uncommitted right now:** `scripts/sweep_epochs.py`, `scripts/aggregate_sweep.py`,
`scripts/run_sweep.sh`, and this file. First commit:

```bash
git add scripts/sweep_epochs.py scripts/aggregate_sweep.py scripts/run_sweep.sh HANDOFF.md
git commit -m "feat(scripts): resumable epoch sweep and cross-epoch sign tests; refresh handoff"
```

`results/` holds 558 runs in `results/sweep` plus the funding sweep in
`results/sweep_funding`. Check `.gitignore` before adding any of it.
Commit `summary_epochs.csv` only, not the per-epoch timelines.

---

## 3. Dataset

Collected since 2026-08-30. As of 2026-09-28: 261.6M book events over 29.1
days, 0 sequence gaps, 0 checksum failures, 275 clean resyncs. Usable segments
of at least 2 h: 184 in total (BTC 62, ETH 63, XAUT 59). 1,095,785 trades. The
collector is still running, so there is more data now. Re-run
`scripts/audit_data.py` for current numbers; don't reuse these.

| | book events | trades/day | quote:trade |
|---|---:|---:|---:|
| tBTCF0:USTF0 | 124.9M | 19,250 | 222:1 |
| tETHF0:USTF0 | 80.9M | 15,766 | 176:1 |
| tXAUTF0:USTF0 | 55.5M | 2,512 | 757:1 |

- ClickHouse TTL is 6 months, so the earliest data expires around 2027-02-28.
- **Funding and mark price are backfilled, not collected live.** The collector
  does not subscribe to `status:deriv`. `scripts/backfill_deriv.py` pulls
  `/v2/status/deriv/<SYMBOL>/hist` (~60s resolution; it needs explicit
  `start`/`end`, otherwise it returns only the last few hours) into
  `bfx.deriv_status`, with settled rates in the `v_funding` view. Sub-minute
  mark price is unrecoverable and nothing uses it. Mean settled rate per 8h:
  BTC +0.029%, ETH +0.025%, XAUT -0.003% (XAUT's runs the other way).

---

## 4. What the sweep established (real numbers)

Sweep settings: BTC, all usable epochs (epoch 0 excluded), γ ∈ {1e-4, 1e-3,
1e-2}, zero fees, κ and A calibrated **in-sample** on each epoch. 62 epochs at
every γ (epoch 68 at γ=1e-4 crashed originally and was re-run after the fix in
§6 item 1; the γ=1e-4 figures below are the 62-epoch ones). Paired binomial sign
tests across epochs. Reproduce with `python scripts/aggregate_sweep.py`.

### 4.1 Behavioural ordering vs symmetric: robust

| | AS > SYM | GLFT > SYM |
|---|---|---|
| spread capture, γ=1e-3 | 60/62, p=8.5e-16 | 61/62, p=2.7e-17 |
| inventory variance (lower), γ=1e-3 | 61/62, p=2.7e-17 | 61/62, p=2.7e-17 |
| at γ=1e-2 | 60–62/62 on every metric | 61–62/62 on every metric |
| at γ=1e-4 | 58–59/62 on every metric | 59–61/62 |

What used to be a four-epoch anecdote is now a 62-epoch result.
It also survives walk-forward calibration (§5 item 2): at γ=1e-3 every
comparison above keeps its direction when κ/A are fitted only on the 24 h
before each epoch.

### 4.2 γ is a monotone risk dial (median across epochs)

| strategy | γ | spread capture (bps) | inventory variance | mean abs inventory | fills |
|---|---|---:|---:|---:|---:|
| AS | 1e-4 | 1.028 | 0.753 | 1.198 | 799 |
| AS | 1e-3 | 1.308 | 0.165 | 0.401 | 517 |
| AS | 1e-2 | 2.119 | 0.026 | 0.099 | 163 |
| GLFT | 1e-4 | 1.029 | 0.724 | 0.975 | 806 |
| GLFT | 1e-3 | 1.195 | 0.317 | 0.575 | 706 |
| GLFT | 1e-2 | 1.456 | 0.121 | 0.297 | 484 |
| SYM | any | 0.719 | 3.83 | 2.53 | ~1,358 |

Raising γ buys capture per unit and lower inventory, paid for in fill count
(AS loses 80% of its fills from 1e-4 to 1e-2). GLFT reacts much less to γ than
AS does.

### 4.3 AS vs GLFT: the ordering flips with γ

- γ=1e-4: GLFT beats AS on capture in 43 of 62 epochs (p=0.0032) and on
  inventory variance in 45 of 62 (p=5e-4).
- γ=1e-3 and γ=1e-2: AS beats GLFT on both, in 48–60 of 62 epochs.

So "AS > GLFT" is **not** a property of the models. The same γ means different
things in the two closed forms, so comparing them at equal γ is not a fair
comparison. The fair comparison is at **matched inventory variance**: interpolate
each model's (γ → inventory variance) curve and compare spread capture at
equal risk. The old handoff's "AS > GLFT > symmetric" should not be repeated
without this caveat.

### 4.4 PnL: still noise in magnitude, but a possible sign effect

- `pnl_inventory_tstat`: |t| ≥ 1.96 in 3.0% (17) of the 558 runs, which is below
  the 5% expected under the null. Total PnL remains uninterpretable, and the
  "PnL is not the metric" argument stands.
- **New, needs checking:** for AS and GLFT at γ ≥ 1e-3, inventory PnL is
  negative in 41–45 of 62 epochs (p between 5e-4 and 0.015). Symmetric sits at
  33/62 (p=0.7), and γ=1e-4 is about 50/50. The skewed quoters seem to lose on
  inventory *systematically*, even though each single epoch is
  insignificant. Adverse selection is the obvious candidate. Test it before
  saying it anywhere: is the effect concentrated in high-σ epochs, does it
  survive splitting into first and second halves, and does it appear on ETH?
  The 9 tests share epochs, so they are not independent. Bonferroni at 9 still
  leaves the γ=1e-2 result (p=5e-4) standing.
- Total PnL > 0 in only 24–28 of 62 epochs for AS/GLFT at γ ≥ 1e-3.
- 91–92% of fills land on price levels the strategy created, for every
  strategy. See trap §6.4.

### 4.5 From earlier single-epoch work (still valid, still narrow)

- κ across 4 BTC epochs spanning up, flat and down markets: 0.0915 / 0.0918 /
  0.1046 / 0.0940, R² 0.72–0.86.
- Session-stratified κ on epoch 27: 0.0955–0.1218, a 28% spread across the day.
  This is **one epoch**, R² falls when the sample is split, and the 17–21 UTC
  bucket is flagged as too thin. Treat it as a lead.
- The epoch-27 apparent results (GLFT +40.62%, Sharpe 8.66) and epoch-22
  (symmetric +475.58%, Sharpe 11.87) are inventory noise.

---

## 5. Work queue (priority order)

Each item says what "done" means. Commit after each one.

**0. Housekeeping.** Commit the scripts (see §2), add `pyarrow` and `scipy`
to requirements, and confirm `pytest -q` passes.

**1. Fix the epoch-68 crash.** Done (2026-10-28). The diagnosis here was
wrong: it was not weak inventory control. At the crash AS held 0.154
contracts. BTC-PERP flash-spiked from 81,200 to 153,960 and back in ~1.5 s at
2026-09-20 16:28:12 -- real trades, not a book artefact -- and realised sigma
over the 300 s window rose to ~12,300 USD/sqrt(s) against a normal ~8. With
that sigma, q*gamma*sigma^2*tau put the raw bid at -58,595, and `_finalise`
looked up a tick for a negative price. The volatility estimator was checked
and is correct (incremental and exact recompute agree); it is faithfully
measuring a jump.

Fix in `strategy/base.py::_finalise`: run the no-cross repair on the raw
prices *before* rounding, so the near side is pinned to the touch, and do not
quote a side whose price is non-positive (checked after rounding too, since
the 8-decimal cap rounds anything under 1e-8 to zero). A max-skew clamp was
rejected because it would bind in ordinary conditions at gamma=1e-2 and
change those results. For every finite positive input the reorder is
identical to the old order; a test checks this against the old algorithm on
>3,000 random states. Count: 113 suppressed sides on epoch 68 (AS, 1e-4);
reported per run as `n_sides_suppressed`.

That was the only flash move in the dataset: across 59 days and all three
symbols it is the single one-minute window with a trade more than 2% from that
minute's median. But it dominates every pooled PnL sum. On the earlier sweep
epoch 68 is 95% of AS's pooled loss at gamma=1e-3 and 98% at 1e-2, 89% of
GLFT's, 56% of symmetric's -- in all three strategies, including symmetric,
which ignores sigma, because the quoters re-centre on a mid of ~148,000 and
are filled there before it snaps back. Sign tests and medians are robust to
it; sums and means are not. **Decided (2026-10-28):** epoch 68 is excluded
from headline PnL magnitudes and every such figure is also reported with it
included (`funding_report.py --exclude-epochs 68`, and without the flag).
`results/sweep/.../gamma_0.0001/epoch_68` now holds the re-run (no funding,
zero fees, to match the other 61); the failed attempt is kept as
`run.failed_20260928.log`. The whole sweep is now 62/62 at all three gammas.

**2. Walk-forward calibration.** The CLI calibrates κ and A on the same epoch
it replays, which is in-sample. Add a mode that calibrates on the preceding
N hours (or the preceding epoch), using only data before the replay window.
Re-run γ=1e-3 under it and compare with §4. The behavioural ordering should
survive; if it doesn't, stop and tell me.
*Done =* `--calibration {in_sample,walk_forward}` flag, default unchanged,
and a comparison table.

Done (2026-10-28). `--calibration walk_forward --calib-hours 24` fits κ/A on
the 24 h before the replay starts, per earlier epoch (each with its own book
rebuild and warmup), trades strictly before the replay; it raises rather than
falling back to in-sample when coverage is under 50%. Results in
`results/sweep_wf` (γ=1e-3, zero fees, `--no-funding`, all 62 epochs).
Comparison: `python scripts/compare_calibration.py [--exclude-epochs 68]`.
**The ordering survives**: no comparison flips (table in progress report §3,
"Calibration without lookahead"); symmetric is identical across the two sweeps
on all 62 epochs, confirming calibration is the only variable. Per-strategy
medians move ≤4.4%, no Wilcoxon p < 0.15. κ moves a median −1.6% (IQR −19%
to +12%), A −9% (IQR −22% to +37%). Sample-size confound: 55 epochs are
<24 h (walk-forward has *more* data), 7 are ≥24 h; no flips in either group,
but the long group has little power. 12 epochs failed on the first pass on
Tailscale/stream errors, not on history, and were re-run (`log_retry*.txt`);
use ≤2 parallel processes for walk-forward sweeps.

**3. Wire session-stratified κ/A into the engine.**
`estimate_by_session()` already returns per-session estimates; the engine uses
only `"all"`. Make `MarketState.kappa` and `A` switch at the UTC session
boundaries, behind a config flag, using walk-forward estimates from item 2.
Run static vs session on all BTC epochs for AS and GLFT at γ=1e-3, then sign
test. This is ladder step 2 and fills the right-hand table of slide 12. Either
result can be reported: a gain supports the intraday-structure claim, and no
gain motivates online estimation.
*Done =* the paired table and sign tests in `results/session/`.

**4. Matched-risk AS vs GLFT.** Add γ points (for example 3e-4, 3e-3, 3e-2) and
compare spread capture at equal inventory variance (see §4.3).
*Done =* one figure of capture vs inventory variance per strategy, with the
epoch-level spread shown.

**5. Investigate the inventory-PnL sign effect (§4.4).** Look at it
conditioned on σ, the trend over the epoch, and session; check markouts at
100 ms, 1 s, 5 s and 30 s. Don't report it as a finding until it holds on a
split sample.

**6. Cross-instrument.** Repeat the sweep on ETH (63 epochs) and XAUT (59).
`python scripts/sweep_epochs.py --symbol tETHF0:USTF0 --gamma 0.001` (check the symbol string against `list_epochs`). XAUT's 757:1
quote-to-trade ratio makes it the external-validity test.

**7. Funding data.** Done (2026-10-28, table creation authorised by the
user). The engine charges each 8h settlement on the signed position held at
that instant; attribution carries `PnL: funding` as a fourth leg and the four
legs sum to total PnL exactly. Funding does not feed back into quoting, so it
never changes fills, spread capture or inventory paths -- only cash. Runs
without a funding schedule print "not modelled" rather than zero.
Result over 15 epochs ≥20 h at γ=1e-3, rebate −1 bp (`results/sweep_funding`,
progress report §3): funding is large and two-signed for symmetric (received
in 8 of 14 epochs excl. 68, per-epoch −1,560…+1,222) and small for AS/GLFT
(median 2–4% of spread capture). The epoch-27 "368% of spread capture" figure
is not representative of the sign. No pairwise total-PnL difference is
significant. Backfill
coverage is per window: `--full` for a whole symbol, `--epochs-min-hours N` for
long epochs only. The collector was not touched; a restart would be safe if
live collection is ever wanted (epoch counter resumes from `max(epoch)`).

**8. Adaptive ladder steps 3–5**, only after 2 and 3 exist:
- online estimators (Kalman / rolling) for σ, κ and A;
- a contextual bandit over parameter tiers (γ, q_max, size, spread floor), with
  reward = spread capture − λ·inventory variance, not PnL;
- a γ PI controller that holds inventory variance at a target;
- a Jev event layer for text only, evaluated against a rule tree on the same
  inputs.

Every step is compared against the previous step, paired by epoch.

---

## 6. Traps

1. **Epoch boundaries.** Replays must stay inside one epoch. Crossing one
   creates ghost orders whose delete message fell into a gap. `resolve_epoch()`
   handles this, but every new query needs the same constraint.
2. **Epoch 0 may be internally discontinuous.** It comes from collector
   restarts on 2026-08-30. The sweep excludes it by default. To verify the
   others:
   ```sql
   SELECT epoch, max(gap_sec) AS worst_gap_sec, count() AS n_gaps FROM (
     SELECT epoch, dateDiff('second',
       lagInFrame(ts_exch) OVER (PARTITION BY epoch ORDER BY ts_exch, seq), ts_exch) AS gap_sec
     FROM bfx.book_mbo WHERE symbol = 'tBTCF0:USTF0')
   WHERE gap_sec > 5 GROUP BY epoch ORDER BY worst_gap_sec DESC LIMIT 10
   ```
3. **Total PnL is noise.** Never headline PnL or Sharpe. Always show them next to
   `pnl_inventory_tstat`. The CLI warns when an inventory-dominated result has
   |t| < 2.
4. **Fills on self-created levels (~91%).** These are legitimate in a replay,
   but they assume the quote did not change the flow that hit it, and a
   replay can't test that. Say so wherever fill-based numbers appear.
5. **Markouts are non-monotonic in horizon** (GLFT ep27: +1.574 bps at 100 ms,
   −1.202 at 1 s, +0.108 at 5 s, +4.384 at 30 s). Always name the horizon.
6. **Equal γ is not equal risk across models** (§4.3).
7. **Scale.** Long epochs take 6–10 min per γ. The bottleneck is
   `OrderBook.apply()`. Profile before optimising, and distrust any speedup
   that changes results. Check the results are bit-identical before and after.
8. **`config/default.yaml` has γ=0.1**, outside the swept range. Always pass
   `--gamma` explicitly.
9. **Zero fees throughout.** Maker fees enter as a constant per fill. Run
   `--fees` as a sensitivity check only after the behavioural results are settled.
10. **Mac sleep.** `run_sweep.sh` uses `caffeinate`. Long runs launched any
    other way need it too.

---

## 7. Presentation and report status

- The Second Interim deck exists, and so does a bilingual script (EN/ZH).
- **Slide 12's γ table currently holds illustrative placeholder values, which
  are wrong.** Replace them with §4.2. The real inventory variances are much
  lower at high γ (AS 0.026, not 0.39), which shows why placeholders must
  never be presented. Slide 12's right-hand table stays as placeholders until
  work item 3 is done; then remove the "ILLUSTRATIVE" labels.
- `docs/progress_report.md` predates the sweep. Update section 3 with §4.1–4.4
  once items 1–2 are done.
- Questions for the supervisor:
  - reframe RQ1 around intraday structure rather than market regimes;
  - adopt spread capture and inventory variance as the primary metrics.

---

## 8. Standing instructions

- Don't invent numbers. If something didn't run, say so.
- Flag results that look too good. The failure mode here is a plausible number
  that nobody questions.
- Separate "the code is wrong" from "the code was never tested here".
- Separate behavioural findings, which can be defended, from performance
  findings, which currently can't be.
- Small, understood runs before large ones.
- Small commits on a branch, with messages that say what changed and why.
  Never commit to main, never force-push.

## Context

- **Instruments:** BTC, ETH and XAUT perpetuals on Bitfinex. HYPE was dropped
  for liquidity; older documents that mention it predate that change.
- **The retracted result:** an earlier iteration reported +2% annualised at a
  zero-fee optimum. There were two causes: inventory carrying cost was not
  netted, and fills were treated as unconditional. Both are why the metrics
  layer now attributes PnL and the fill model is queue-aware.
