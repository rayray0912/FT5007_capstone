# Handoff to Claude Code

Paste the **Opening prompt** below into a fresh Claude Code session in the repo
root. Everything else here is reference for you, not for pasting.

---

## Opening prompt

> I'm working on my NUS FT5007 capstone: a market-making backtest on
> order-level Bitfinex perpetual data, implementing GLFT against classical
> baselines. The repo is built but has never been run against the real
> dataset — only synthetic data.
>
> Read `docs/progress_report.md` and `README.md` first. They cover the current
> state, what's implemented, what's broken, and what was learned building it.
> Then read `src/mmbacktest/config.py`, which documents most of the non-obvious
> modelling decisions inline.
>
> The immediate task is to get a first real run working. The data lives in
> ClickHouse on a NUC reachable over Tailscale at 100.76.49.84:8123, database
> `bfx`. Start with `python scripts/audit_data.py` and we'll work from
> whatever it reports.
>
> Things I care about: don't invent numbers, don't claim something ran if it
> didn't, and tell me when a result looks too good rather than reporting it.
> An earlier version of this project reported a +2% annualised return that had
> to be retracted because the fill model was too permissive — I'd rather catch
> that kind of thing early.

---

## Where the project actually stands

**Built and tested (28 unit tests passing):** order-level book
reconstruction, intensity and volatility calibration, three quoting strategies
(symmetric / Avellaneda-Stoikov / GLFT), queue-position-aware fill simulation,
backtest engine on a 100ms clock, metrics layer, CLI.

**Never run on real data.** Every number produced so far is from the synthetic
generator, which has no informed flow. Behavioural metrics from it are
informative; PnL is not.

**Not started:** the LLM-adaptive layer. Prompt and memory schemas are
designed, no skill code written. This is deliberate — an adaptive layer on top
of an unvalidated baseline produces a comparison nobody can interpret.

---

## First session plan

```bash
# 1. Confirm the collector is reachable and the data is intact
python scripts/audit_data.py

# 2. Short pilot on one epoch before committing to a long replay
python scripts/run_backtest.py --symbol tBTCF0:USTF0 --epoch <N> \
    run --strategy glft

# 3. The run that matters
python scripts/run_backtest.py --symbol tBTCF0:USTF0 \
    compare --fees 0,8,-1 --out results/

# 4. Queue-model sensitivity — see below for why this is the priority
python scripts/run_backtest.py --symbol tBTCF0:USTF0 \
    --no-queue-model compare --fees 0 --out results/no_queue/
```

### Why step 4 matters most

The difference between runs 3 and 4 quantifies, on real data, how much a
conventional backtest overstates this strategy. That is a methodological
finding, not a performance claim, which makes it the most defensible number
the project will produce for a while. It also directly substantiates the
earlier retraction of the +2% result.

---

## Traps that will bite

**Scale.** 115M book events. The sandbox testing was on 40k. Start with a
2–4 hour epoch. If it's unusably slow the bottleneck is almost certainly the
Python loop in `OrderBook.apply()`; profile before optimising, and be
suspicious of any speedup that changes results.

**Epoch boundaries.** Replays must stay inside one epoch. Crossing one produces
ghost orders — orders whose delete message fell into a collection gap and which
therefore never leave the reconstructed book. `resolve_epoch()` handles this,
but any new query needs the same constraint.

**Schema drift.** The ClickHouse client was written against the documented
schema and has never touched the live database. Column names, dtypes, and the
`side` / `action` enum encodings are the likely first failures. They'll be
obvious and quick.

**Config defaults are crypto-scale now.** Three were originally equity-scale
and wrong by two orders of magnitude. If you ever run this on an equity book,
`intensity_max_delta_bps`, the half-spread bounds, and `horizon_seconds` all
need revisiting. Reasoning is inline in `config.py`.

**Trade side convention.** `_merge_streams` assumes Bitfinex signs trade
`amount` by the aggressor's direction (positive = buy aggressor). Worth
verifying against real data — if it's inverted, every fill routes to the wrong
side and adverse selection flips sign.

---

## Open items, roughly in priority order

1. **Funding rate collection.** The `status:deriv` channel (funding rate, mark
   price, index price) isn't being collected. For a perpetual, funding is a
   real component of inventory carrying cost, so PnL attribution currently
   understates it. Historical funding backfills over REST; high-frequency mark
   price does not, so the loss grows daily. This is the most consequential gap.
2. First real run + queue-model sensitivity (above).
3. Session-stratified calibration on real data — the dataset has a 3.7x
   intraday swing in event rate, and the code supports stratification but it's
   never been exercised on anything but synthetic.
4. LLM-adaptive layer, only after the baseline is validated.

---

## Useful follow-up prompts

**After audit_data.py runs:**
> Here's the audit output: [paste]. Pick an epoch for a pilot run and tell me
> why that one.

**When the first real run fails:**
> [paste traceback]. Fix it, but first tell me whether this is a schema
> mismatch or a logic error — I want to know if the code was wrong or just
> untested against reality.

**After the first successful comparison:**
> Walk me through these results as if I'll have to defend them. Which numbers
> would you challenge if you were my supervisor, and what's the honest answer?

**For the queue-model sensitivity:**
> Compare results/ and results/no_queue/ and write the finding up as three or
> four sentences I could put in a progress report. Just the difference and what
> it means — no speculation about what it implies for the strategy.

**Before adding the LLM layer:**
> Read the prompt and memory schema notes in docs/progress_report.md and
> `src/mmbacktest/strategy/base.py`. Propose how the two LLM skills should
> slot into the existing Quoter interface without touching the execution path,
> and tell me what could go wrong.

**Performance work:**
> Profile a replay over a 2-hour epoch and show me where the time goes before
> changing anything. I want to see the profile, not a guess.

---

## Standing instructions worth repeating

These are the things that matter for research integrity on this project:

- **Don't invent numbers.** If something wasn't run, say it wasn't run.
- **Flag results that look too good.** The project already had to retract a
  +2% annualised figure because the fill model was too permissive. The failure
  mode is a plausible-looking number that nobody questions.
- **Distinguish "the code is wrong" from "the code was never tested here".**
  Different fixes, different implications.
- **Prefer a small run that's understood to a large one that isn't.**

---

## Context you may need to explain

**Instrument selection.** The target is BTC / ETH / XAUT perpetuals on
Bitfinex. HYPE-PERP was dropped at selection for liquidity (~13,675 USD/24h,
too few trades to calibrate fill intensity). Some earlier project documents
still reference HYPE — the code and data are correct, those documents predate
the change.

**The retracted result.** An earlier iteration reported +2% annualised at a
zero-fee sweep optimum. Two causes: the sweep summary didn't net inventory
carrying cost against trading revenue, and the fill model treated quotes as
unconditionally filled rather than conditional on queue position. The current
codebase fixes the second properly; the first is why PnL attribution is now
reported as a separate metric group.
