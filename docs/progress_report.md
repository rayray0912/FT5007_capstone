# Progress Report

**Project:** Adaptive market making on crypto perpetuals, GLFT on order-level Bitfinex data
**Student:** Ziheng He (e1538759@u.nus.edu)
**Supervisor:** Asst. Prof. Lu Yao
**Repository:** `FT5007_capstone`

---

## 1. What has been tried

### Implemented and working

**Order-level book reconstruction.** A state machine over the MBO event stream
that maintains per-price queues in arrival order. The point of doing this at
order level rather than from aggregated depth is `volume_ahead()`: it makes it
possible to know where a simulated order would sit in a specific queue, which
aggregated L2 data cannot tell you. Verified against an 8,000-event replay with
zero crossed books.

**Calibration of the two inputs the quoters need.**
λ(δ) = A·exp(−κδ) is fitted by bucketing trade distance from the prevailing mid
and regressing log-rate on distance, weighted by bucket count. Against a
generating process with a known κ = 0.02 the estimator recovers it to 0.2% with
R² > 0.999. Realised volatility is computed on a fixed time grid and recovers a
known σ = 0.35 to within 0.3–1.3% across sampling intervals from 100 ms to 5 s.

**Three quoting strategies** sharing one interface: symmetric, Avellaneda–
Stoikov, and GLFT. Each was checked against the sign conditions theory
predicts rather than just run to see if it produced numbers. GLFT is exactly
symmetric at zero inventory, antisymmetric in inventory, scales its skew with
σ while leaving the base distance untouched, and damps the skew as A rises.

**Queue-position-aware fill simulation** with quote latency, and markout
computation at 100 ms / 1 s / 5 s / 30 s horizons for adverse selection.

**Backtest engine** on a 100 ms decision clock, with strict event ordering
within a timestamp so that a quote at time *t* is exposed to everything that
happens at *t*.

**Metric layer** covering absolute performance, market-making behaviour, and
PnL attribution.

**28 unit tests**, organised around invariants.

### Tried and did not work

**Unconditional cancel-replace every decision tick.** The first version of the
engine cancelled all quotes and reposted on every 100 ms tick. With 80 ms quote
latency each order was live for 20 ms, and the result was 11,926 orders placed
and **zero fills**. The deeper problem is that reposting throws away queue
priority every tick, which defeats the entire purpose of modelling the queue.
Replaced with quote persistence: amend only when the desired price actually
moves. Placements dropped to 211 over the same replay and fills behaved
sensibly.

**Equity-scale configuration defaults.** Three defaults were set at scales that
make sense for an equity book and are wrong by roughly two orders of magnitude
for BTC-PERP. Details in section 4.

### Implemented but not yet exercised on real data

Everything above runs end to end on synthetic data. **None of it has been run
against the ClickHouse dataset yet** — that requires the Tailscale connection
from a machine with the collector reachable, and is the immediate next step.

### Not started

The LLM-adaptive layer. The prompt schema and memory schema are designed, but
no skill code is written. That is deliberate sequencing: an adaptive layer on
top of an unreliable baseline produces a comparison that cannot be interpreted.

---

## 2. Baselines, datasets, metrics

### Baselines

Three, forming a deliberate progression where each adds exactly one piece of
machinery to the one before it.

| | what it does | what a difference against it means |
|---|---|---|
| **symmetric** | fixed half-spread around the mid, no inventory logic | how much comes from simply being in the book |
| **avellaneda_stoikov** | inventory-skewed reservation price, finite horizon | value of inventory management |
| **glft** | explicit inventory bound, asymmetric per-side distances | value of solving the *constrained* problem |

The intended fourth comparison — the LLM-adaptive quoter — slots into the same
interface and will be scored on the same metric layer.

### Dataset

Bitfinex perpetual swaps, order-level (market-by-order), collected continuously
since 2026-08-30 to a ClickHouse instance on a NUC in Singapore.

| | |
|---|---|
| instruments | BTC-PERP, ETH-PERP, XAUT-PERP |
| granularity | market-by-order, `prec=R0`, `len=250` orders per side |
| span | 14.2 days as of 2026-09-13, collection ongoing |
| book events | 114,992,905 |
| trades | 531,045 (~37,400/day) |
| integrity | **0 sequence gaps, 0 checksum failures** over ~1.34M checksum validations |
| continuity | 41–49 segments per symbol, longest 40–54 h, 29–33 segments over 2 h |

HYPE-PERP was excluded at selection: ~13,675 USD per 24h on this venue, too few
trades to calibrate λ(δ) or to backtest fills against.

The zero sequence gaps and zero checksum failures are the figures that matter
most. They mean there is no silent book drift: every interruption is explicit
and locatable, so the affected windows can be excluded rather than quietly
corrupting a replay.

### Metrics

Three groups, reported together because any one alone misleads for a market
maker.

**Absolute:** total PnL, annualised return on committed capital, Sharpe,
Sortino, max drawdown, Calmar.

**Market-making behaviour:** fills and fill rate, spread capture (bps per unit
traded), median queue wait, mean and max absolute inventory, inventory
variance, time at the inventory bound, adverse selection from signed markouts.

**PnL attribution:** total PnL split into spread capture, inventory
mark-to-market, and fees. This is the group that answers *where the money came
from*, and it is what distinguishes a maker earning the spread from one
accidentally running a directional book.

Annualised statistics are flagged as unreliable when the replay is under 24 h,
because scaling a short window to a year produces Sharpe ratios in the
thousands that say nothing about the strategy.

---

## 3. Current results

Synthetic data only. PnL from synthetic data is meaningless — the generator has
no informed flow — but the behavioural metrics show the strategies ordering as
theory predicts, which is the sanity check the synthetic mode exists for.

60,000 events, γ = 0.1, q_max = 5, zero fees:

| | symmetric | avellaneda_stoikov | glft |
|---|---|---|---|
| fills | 436 | 156 | 40 |
| spread capture (bps) | 0.074 | 0.282 | **0.544** |
| mean abs inventory | 2.694 | **0.792** | 1.200 |
| inventory variance | 8.262 | **0.973** | 1.813 |
| median queue wait (s) | 5.08 | 7.07 | 9.27 |

The naive quoter trades most and controls inventory worst. AS cuts inventory
variance by a factor of eight at the cost of a third of the fills. GLFT quotes
widest, trades least, and captures the most per fill.

**These are not results.** They are evidence that the pipeline behaves
correctly. The real run is the next step.

---

## 4. What has been learned

**Configuration defaults carry modelling assumptions.** Three defaults were set
at equity scale and were wrong for a crypto perpetual by about two orders of
magnitude:

- `intensity_max_delta_bps = 20` is 222 USD on BTC at 111k, against a market
  spread near 1 USD. Every trade fell in the first bucket and the fit had no
  slope to find. It was caught because the fit reported "only 1 usable bucket"
  rather than silently returning a plausible-looking number.
- Half-spread defaults quoted ~22 USD outside a ~1 USD spread. Nothing filled.
- `horizon_seconds = 3600` is the interesting one. AS's risk term is
  γσ²(T−t), so an hour-long horizon gives 44 USD of half-spread and **AS never
  traded at all** while the other two strategies did. Setting T−t to the
  session length is the intuitive reading and it is wrong; the defensible
  reading for a continuously running maker is the horizon over which inventory
  is worked off, which at observed fill rates is about a minute.

The third point is a genuine model difference worth reporting: **GLFT is immune
to it**, because its asymptotic form has no horizon term at all. A result
showing AS underperforming could easily be an artefact of a badly chosen T−t
rather than a property of the model.

**Queue priority is the whole game in a tight book, and the engine has to
respect it in two places.** Modelling the queue in the fill simulator is
necessary but not sufficient: if the engine cancels and reposts every tick, the
strategy destroys its own queue position faster than the model can reward it.
The fix — amend only on a genuine price change — is also what real quoting
systems do.

**An estimator that is right up to a constant is still wrong.** The intensity
fit returned `exp(intercept)`, which is A·(bucket width), not A. κ was
unaffected, so every sign check passed and the fit looked healthy. But A feeds
GLFT's inventory coefficient directly, so retuning `intensity_n_buckets` would
have silently rescaled every quote. Caught by asking whether the estimate was
invariant to a knob it should not depend on; now a regression test.

**Calibrate by session, not pooled.** The dataset has a 3.7× intraday swing in
event rate between the quiet 09:00–10:00 UTC hour and the 23:00–02:00 peak. A
pooled fit describes neither regime.

---

## 5. Next steps

1. **Run against the real dataset.** Everything is built; it needs the
   Tailscale connection and a first pass over a long BTC-PERP epoch.
2. **Collect the `status:deriv` channel.** Funding rate, mark price and index
   price are not being captured. For a perpetual, funding is a real component
   of inventory carrying cost, and its absence means PnL attribution
   systematically understates the cost of holding inventory. Historical funding
   can be backfilled over REST but high-frequency mark price cannot, so the
   loss grows with every day of delay.
3. **Fee sensitivity across the three tiers** (zero, standard maker, MM
   rebate), which the CLI already supports in a single pass.
4. **Queue-model sensitivity**, quantifying how much `--no-queue-model` inflates
   results, so the magnitude is reported rather than asserted.
5. **Then** the LLM-adaptive layer, on top of a baseline that has been shown to
   behave.

---

## 6. Open questions for supervision

1. **Funding-rate collection** — worth interrupting the current collector to
   add the `status:deriv` subscription now, or run a second collector process
   alongside it to avoid any risk to the existing stream?
2. **Cross-instrument scope** — is the BTC/ETH/XAUT comparison (with XAUT as a
   non-crypto control) worth the extra work, or is a single instrument done
   thoroughly the better use of the remaining time?
3. **AS horizon parameter** — is treating T−t as the inventory-holding horizon
   rather than the session length the right call, and should the sensitivity to
   it be reported as a result in its own right?

---

## Appendix: fee sensitivity, synthetic data

Run with `--fees 0,8,-1` on 40,000 synthetic events. PnL levels are not
meaningful here, but the *relative* effect of the fee tier is, because it is
arithmetic on the traded notional rather than a property of the generator.

| fee tier | symmetric | avellaneda_stoikov | glft |
|---|---|---|---|
| zero | +95.64 | +144.40 | +41.84 |
| standard maker, 8 bp | −11,287 | −3,854 | −629 |
| MM rebate, −1 bp | +1,518 | +644 | +126 |

The magnitude of the 8 bp column is the point. BTC-PERP quotes a spread near
1 USD on a 111,000 mid, which is roughly **0.09 bps**. A standard maker fee of
8 bps is therefore about ninety times the entire spread being competed for. No
amount of quoting skill recovers that: at the standard tier, passive market
making on this instrument is arithmetically unprofitable before any question of
adverse selection arises.

Two consequences for the research design:

1. **The MM rebate tier is not a convenience assumption, it is the only regime
   in which the strategy exists at all.** That needs stating explicitly in any
   write-up rather than appearing as a footnote, because a reader who assumes
   standard fees will correctly conclude the whole thing is unprofitable.
2. **Strategies that trade less are less exposed to the fee term.** GLFT loses
   least at 8 bp purely because it fills least. Any comparison across fee
   regimes has to account for this, or it will read fee exposure as strategy
   quality.
