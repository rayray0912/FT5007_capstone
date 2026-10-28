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
and regressing log-rate on distance, weighted by bucket count. By default the fit
uses the epoch being replayed; `--calibration walk_forward` uses only the 24 h
before it (section 3). Against a
generating process with a known κ = 0.02 the estimator recovers it to 0.2% with
R² > 0.999. Realised volatility is computed on a fixed time grid and recovers a
known σ = 0.35 to within 0.3–1.3% across sampling intervals from 100 ms to 5 s.

**Three quoting strategies** sharing one interface: symmetric, Avellaneda–
Stoikov, and GLFT. Each was checked against the sign conditions theory
predicts rather than just run to see if it produced numbers. GLFT is exactly
symmetric at zero inventory, antisymmetric in inventory, scales its skew with
σ while leaving the base distance untouched, and damps the skew as A rises.

Those checks all passed while the GLFT closed form had γ and κ transposed and
its inventory argument scaled by the position limit. Sign and symmetry
conditions constrain the *shape* of a quote function and say nothing about
whether the magnitudes are right, which is the failure mode a closed form
actually has. The tests that would have caught it — a limit as γ → 0, and
comparing the shared base term against AS's — are now in the suite. See
section 4.

**Queue-position-aware fill simulation** with quote latency, and markout
computation at 100 ms / 1 s / 5 s / 30 s horizons for adverse selection.

**Backtest engine** on a 100 ms decision clock, with strict event ordering
within a timestamp so that a quote at time *t* is exposed to everything that
happens at *t*.

**Metric layer** covering absolute performance, market-making behaviour, and
PnL attribution.

**103 unit tests**, organised around invariants.

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

### Now exercised on real data

Everything above now runs end to end against the ClickHouse dataset. Doing so
surfaced five defects that synthetic data could not, because in each case the
generator's convention happened to differ from the collector's:

| defect | consequence |
|---|---|
| `astype("int64")/1e9` on a `datetime64[ms]` column | the 100 ms decision clock never came due; **zero orders placed** over a two-hour window |
| trades loaded over the epoch while book events were loaded over the configured window | intensity calibration matched trades against a mid series that did not cover them |
| aggressor direction read from the sign of `amount` | the collector stores `abs(amount)` and keeps direction in `side`, so every trade looked like a buy: **our bids could never fill** and the book ran one-way short |
| GLFT quote centre computed as `0.5*((mid-δ)+(mid+δ))` | at large δ the mid is rounded away in float64 and the quote centres on **zero** |
| GLFT closed form with γ and κ transposed | see section 4 |

The synthetic suite passed throughout. The generator produces `datetime64[ns]`
timestamps and signed trade amounts, so it took the correct branch in the first
three cases by accident, and its default γ coincided with the calibrated κ in
the fifth, which makes the transposed formula numerically identical.

The lesson is not that the tests were bad but that they were all on one side of
an interface. Every one of these lived at the boundary between the collector's
conventions and the code's assumptions about them.

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
most. The mechanisms behind them are real: the collector enables `SEQ_ALL` and
`OB_CHECKSUM` via `conf` flag 229376, and the counts come from its own logs
(761,766 checksum validations recorded to 09-07, ~1.34M extrapolated).

One gap in the *reporting* of this is worth noting. `integrity_ok()` gates a
replay by counting `collector_events` rows of kind `seq_gap` and
`checksum_fail`, and neither kind has ever appeared in that table. Since the
mechanisms have never fired, that is consistent with clean data — but read from
the database alone, a PASS cannot be distinguished from a predicate that is
never written. Confirming that the collector persists those kinds when they do
fire requires its source, which lives on the NUC outside this repository.

The claim was therefore also checked directly against the data. Over
tBTCF0:USTF0 epoch 27 (4.32M events) the exchange sequence numbers show 104,428
unused values across 71,313 discontinuities, but **the largest single
discontinuity is 37**, and gaps of a single missing value account for half of
them. That shape is what a sequence shared with other message types on the same
connection looks like — trades, heartbeats, checksums — and it rules out the
failure that would actually matter, a contiguous block of book events going
missing. The residual count is not yet fully attributed.

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

**No performance figure is being reported.** The pipeline now runs against the
real dataset and produces numbers; this section is about which of them survive
scrutiny, and most do not.

### What the runs produced

BTC-PERP, four epochs chosen to span market regimes, gamma = 1e-3, zero fees.
Each is a single continuous replay from the epoch's snapshot, so no warm-up is
needed.

| epoch | hours | price move | trades | κ | R² |
|---|---|---|---|---|---|
| 1 | 17.2 | +0.90% | 11,741 | 0.0915 | 0.784 |
| 22 | 18.1 | +0.11% | 29,949 | 0.0918 | 0.856 |
| 27 | 39.9 | −0.95% | 16,168 | 0.1046 | 0.719 |
| 35 | 15.9 | −1.75% | 16,290 | 0.0940 | 0.818 |

The fitted κ sits in 0.092–0.105 across all four, which is a useful check in
its own right: the intensity decay is a property of the book, and an estimate
that moved with the regime would suggest the fit was picking up something
else.

### The headline numbers are noise, and can be shown to be

Two runs produced results that would have been very tempting to report:
GLFT at +40.62% annualised (Sharpe 8.66) on epoch 27 before the aggressor fix,
and symmetric at +475.58% annualised (Sharpe 11.87) on epoch 22. Both are
artefacts of the same thing.

Total PnL splits into spread capture and inventory mark-to-market. The
inventory term is the sum of `position * d_mid` over every step, and holding a
position through a random walk earns and loses continuously, so the net is a
small residual of two very large sums:

| epoch 22 | net | gross | net/gross | t |
|---|---|---|---|---|
| symmetric | +7,394.80 | 3,825,376 | 0.19% | **+0.51** |
| avellaneda_stoikov | −616.70 | 269,243 | −0.23% | −0.40 |
| glft | −973.66 | 675,310 | −0.14% | −0.28 |

Every inventory PnL measured so far, on every strategy and every epoch, has
|t| < 1. None is distinguishable from a random walk. It also changes sign with
the regime — symmetric's inventory PnL is −7,866 on epoch 27 and +425 on epoch
1 — which is what a residual does and what a skill does not.

This is now enforced rather than remembered: the comparison table carries
`inventory gross` and `inventory t-stat`, and the CLI prints a warning when an
inventory-dominated result has |t| < 2.

### What does survive

Spread capture per unit traded is stable across regimes and orders the
strategies the same way every time:

| spread capture (bps) | ep 1 (+0.90%) | ep 22 (+0.11%) | ep 27 (−0.95%) | ep 35 (−1.75%) |
|---|---|---|---|---|
| symmetric | 0.565 | 0.719 | 0.615 | 0.562 |
| **avellaneda_stoikov** | **1.662** | **1.328** | **0.942** | **1.278** |
| glft | 1.304 | 1.056 | 0.933 | 0.913 |

So does inventory control:

| mean abs inventory | ep 1 | ep 22 | ep 27 | ep 35 |
|---|---|---|---|---|
| symmetric | 3.437 | 6.154 | 7.035 | 2.327 |
| **avellaneda_stoikov** | **0.259** | **0.428** | **0.868** | **0.283** |
| glft | 0.463 | 0.997 | 0.940 | 0.411 |

Both orderings hold on all four epochs without exception, across a rising, a
flat, a falling and a sharply falling market. That consistency is the reason
these two are reported and the totals are not: over the same four epochs the
total PnL for symmetric runs from +7,984 to −7,432, and every one of the twelve
strategy-epoch inventory PnLs has |t| < 1.2.

The inventory variances behind those means are the sharper statement: on epoch
22, symmetric runs 34.707 against AS's 0.285, a factor of 122. The naive quoter
is not earning less than the others so much as taking an entirely different
kind of risk — its PnL is dominated by an unmanaged directional position that
happens to pay on some epochs and not others.

AS and GLFT separate on style rather than on quality. AS captures more per unit
traded (its spread capture in bps is highest on every epoch); GLFT trades more
and captures more in total (on epoch 27, 621.50 against AS's 585.66 from 3,166
fills against 2,908). GLFT also holds inventory at least as tightly as AS —
variance 0.701 against 0.737 on epoch 27 — which is what solving the
*constrained* problem is supposed to buy, and which only became visible once
the inventory argument was corrected: before that fix GLFT ran a mean absolute
inventory of 5.42 on the same epoch, and a total PnL of −5,213 rather than
−732.

**These are behavioural findings, not performance findings.** They say the
inventory machinery does what the theory says it does. They do not say the
strategy is profitable, and on this instrument at these fee tiers the totals
are negative more often than not.

### Funding and rebate across 15 epochs

BTC-PERP, every epoch of at least 20 h except epoch 0 (possibly discontinuous),
γ = 1e-3, maker rebate −1 bp, funding charged at each 8h settlement on the
signed position held at that instant. Settled rates were backfilled from
Bitfinex REST (`scripts/backfill_deriv.py`). Reproduce with
`python scripts/funding_report.py --exclude-epochs 68` and without the flag.

**Epoch 68 is excluded from the headline figures and reported alongside.** It
contains the only flash move in the dataset -- BTC-PERP traded from 81,200 to
153,960 and back in about 1.5 s on 2026-09-20 16:28, the single one-minute
window across 59 days and three symbols with a trade more than 2% from that
minute's median. The quoters re-centre on the spiked mid and are filled there
before it snaps back, which produces totals near −100k to −170k for all three
strategies, symmetric included. The exclusion criterion is mechanical, not a
judgement about which results look wrong.

**Replay is reproducible.** Eight of these epochs were also in the September
sweep. Fill count, spread capture, inventory PnL, inventory variance and mean
absolute inventory are identical on all 24 strategy-epochs -- including four
epochs replayed after the quote-hygiene reorder that fixed the epoch-68 crash,
which is the real-data check that the reorder changed nothing it should not.

**Funding: large and two-signed for symmetric, small for AS and GLFT.**

| 14 epochs, excl. 68 | paid on | received on | net | median / epoch | range | median cost / spread capture |
|---|---|---|---|---|---|---|
| symmetric | 6 | 8 | −1,066 | +117 | −1,560 … +1,222 | −42% |
| avellaneda_stoikov | 7 | 7 | −33 | −12 | −148 … +131 | 2% |
| glft | 7 | 7 | −206 | −19 | −197 … +155 | 4% |

The epoch-27 figures quoted earlier (funding at 368% of symmetric's spread
capture) were true of epoch 27 and are not representative of the sign. Epoch 27
fell 0.95% with funding positive at every settlement; symmetric, accumulating
against the trend, was long throughout and paid every time. Across 14 epochs it
receives funding more often than it pays. What does generalise is the
*magnitude*: funding scales with inventory, so it is a variance term for the
quoter that does not control inventory and close to negligible for the two
that do. It does not systematically flatter or penalise anyone.

**Totals, built up leg by leg.** Count of epochs with PnL > 0:

| 14 epochs, excl. 68 | spread + inventory | + rebate | + funding (= total) | spread + rebate + funding (no MTM) |
|---|---|---|---|---|
| symmetric | 8 | 8 | 9 | 13 |
| avellaneda_stoikov | 6 | 11 | 11 | 14 |
| glft | 5 | 9 | 10 | 14 |

No pairwise difference in total PnL is significant (paired sign tests, all
p ≥ 0.42). The rebate is what moves AS and GLFT from mostly negative to mostly
positive; for symmetric the median rebate per epoch (+845) exceeds the median
spread capture (+533), so at this tier the subsidy is the larger part of its
revenue.

The last column removes the inventory mark-to-market and is positive almost
everywhere. GLFT beats AS on it in 12 of 14 epochs (p = 0.013). Read that
carefully: it is in USD, so it largely tracks turnover (GLFT fills more than AS
at γ = 1e-3, and the rebate is proportional to notional), and section 4.3 of
the handoff shows AS-versus-GLFT comparisons at equal γ are not fair
comparisons at all. It is a lead for the matched-inventory-variance analysis,
not a result.

**Pooled sums and medians disagree, which is the point.**

| total PnL | pooled, 14 epochs | median / epoch, 14 | pooled, 15 (incl. 68) | median / epoch, 15 |
|---|---|---|---|---|
| symmetric | −7,097 | +903 | −102,012 | +718 |
| avellaneda_stoikov | +5,548 | +240 | −166,647 | +234 |
| glft | +5,950 | +335 | −116,618 | +261 |

Excluding epoch 68, the pooled sum ranks symmetric last and the median ranks it
first: a few large inventory losses dominate its sum. Including epoch 68 flips
every pooled total to a six-figure loss, while the medians barely move. Epoch 68
also inflates pooled spread capture by about 55,000 for symmetric and GLFT --
fills against a mid near 148,000 score as enormous "captured spread" -- which is
another reason not to read pooled spread capture over a window containing it.
Totals on this instrument remain a statement about inventory noise, with or
without funding.

### Calibration without lookahead (walk-forward)

Every run above fits κ and A on the trades of the epoch it then replays, so the
quoters know the coming fill intensity in advance. `--calibration walk_forward`
fits them only on the 24 h before the replay starts, rebuilding the book
separately in each earlier epoch that window touches, and refuses to run (it
does not fall back to in-sample) when less than half of the window is covered.
The 62-epoch sweep at γ=1e-3 was re-run this way, with zero fees and no funding
to match the in-sample one, so calibration is the only change. Symmetric, which
ignores κ and A, is identical between the two sweeps on all 62 epochs on every
metric checked -- the control that confirms nothing else moved.

| γ=1e-3, 62 epochs | in-sample | walk-forward |
|---|---|---|
| spread capture: AS > SYM | 60/62, p=8.5e-16 | 61/62, p=2.7e-17 |
| spread capture: GLFT > SYM | 61/62, p=2.7e-17 | 62/62, p=4.3e-19 |
| spread capture: GLFT > AS | 10/62, p=5.7e-08 | 10/62, p=5.7e-08 |
| inventory variance (lower): AS > SYM | 61/62, p=2.7e-17 | 62/62, p=4.3e-19 |
| inventory variance (lower): GLFT > SYM | 61/62, p=2.7e-17 | 61/62, p=2.7e-17 |
| inventory variance (lower): GLFT > AS | 12/62, p=1.2e-06 | 7/62, p=2.4e-10 |
| mean abs inventory (lower): AS > SYM | 61/62, p=2.7e-17 | 62/62, p=4.3e-19 |
| mean abs inventory (lower): GLFT > SYM | 61/62, p=2.7e-17 | 60/62, p=8.5e-16 |
| mean abs inventory (lower): GLFT > AS | 8/62, p=1.7e-09 | 9/62, p=1.1e-08 |

No comparison changes direction. Excluding epoch 68 changes no count by more
than one. Per strategy, walk-forward moves the median of each behavioural metric
by at most 4.4% and none of the paired changes is significant (Wilcoxon
p ≥ 0.2 for AS and GLFT on spread capture, inventory variance, mean abs
inventory and fill count; p ≥ 0.15 excluding epoch 68). The parameters themselves do move: walk-forward κ sits a median
1.6% below the in-sample fit with an IQR of −19% to +12%, and A a median 9%
below, IQR −22% to +37%. So the result is not two modes producing the same
number; the ordering is set by γ and the inventory terms, and a ±20% error in κ
shifts the base half-spread without reordering the strategies.

The two modes differ in sample size as well as lookahead. For the 55 epochs
shorter than 24 h, walk-forward fits on more trades than in-sample did; for the
7 epochs of 24 h or more, fewer. The ordering holds in both groups with no
flips, but the long group is seven epochs and only its unanimous results
reach p < 0.05, so it confirms direction rather than adding power.

This removes one objection to the behavioural findings. It does not touch the
PnL ones, which were noise before and are noise now. Reproduce with
`python scripts/compare_calibration.py` (add `--exclude-epochs 68` for the
other basis); results are in `results/sweep_wf`.

### Session-switching κ/A: no gain

The dataset has a 3.7× intraday swing in event rate, which suggested quoting
on per-session intensity fits rather than one pooled fit. `--session-intensity`
switches κ and A at the UTC session boundaries (00, 06, 09, 13, 17, 21), using
walk-forward fits from the 24 h before each epoch; a session whose fit is
unusable falls back to the pooled one, and the share of decisions that did so
is reported. Static arm: the walk-forward sweep above. Session arm: identical
but for the flag. Symmetric is identical across the two on all 62 epochs.

| γ=1e-3, 61 epochs* | median static | median change | session better | sign p | Wilcoxon p |
|---|---|---|---|---|---|
| AS spread capture (bps) | 1.388 | −0.5% | 28/61 | 0.61 | 0.45 |
| AS inventory variance | 0.1735 | +0.2% | 29/61 | 0.80 | 0.74 |
| AS mean abs inventory | 0.4097 | −2.9% | 36/61 | 0.20 | 0.0098 |
| GLFT spread capture (bps) | 1.191 | −0.1% | 30/61 | 1.00 | 0.65 |
| GLFT inventory variance | 0.2646 | +1.4% | 27/61 | 0.44 | 0.77 |
| GLFT mean abs inventory | 0.5528 | −2.7% | 34/61 | 0.44 | 0.14 |

\*One epoch ran entirely on the pooled fallback and is identical in both arms.
Excluding epoch 68 moves no count by more than one and no conclusion.

Neither primary metric moves for either strategy. AS's mean absolute
inventory falls by a median 2.9% with Wilcoxon p = 0.0098, but its sign test is
36/61 (p = 0.2), its inventory variance does not fall with it, and it is one of
14 metric-strategy comparisons in the table (Bonferroni threshold 0.0036): it
is not reported as a finding. AS total PnL improves on 38/61 (Wilcoxon p = 0.03)
and GLFT's worsens on 34/61; both are inventory noise. Restricting to the 38
epochs that never fell back to the pooled fit changes nothing: spread capture
and inventory variance are better under session switching on 17/38 for both
strategies.

The reason is visible before any replay. For 115 (epoch, session) pairs where
both fits are usable, the previous day's *session* κ is no closer to the κ
fitted on that session of the replay itself than the previous day's *pooled*
κ is: median absolute log error 0.167 for both, session closer on 61/115
(p = 0.58; 43/83 on epochs of 6 h or more). There is a systematic pattern --
median across epochs, 13–17 UTC κ sits 13% below the pooled level and 21–00
UTC 12% above -- but a single day's session fit misses its target by about 17%,
so switching on it adds as much noise as signal. The thin sessions make this
worse: 06–09 and 21–00 were unusable in 17 and 18 of the 62 windows, and 14.6%
of decisions overall ran on the fallback.

So per-session fits from one day do not help, and this is a statement about the
estimator, not about whether intraday structure exists. Two things would
separate them: a longer look-back per session (72 h triples each session's
sample), or a pooled κ scaled by a session multiplier estimated over many days,
which uses the stable part of the pattern without refitting each session on a
few hundred trades. Either is a step towards the online estimation in the
adaptive ladder. Reproduce with `python scripts/compare_session.py
[--exclude-epochs 68]`; tables are in `results/session/`.

### What still limits the results

- **84–92% of fills land on price levels the strategy created rather than
  joined.** On a 15 USD spread this is unavoidable and legitimate, but those
  fills never appear on the historical tape, so their PnL assumes our quote
  would not have changed the flow that traded against it. A replay cannot test
  that. See open question 4.
- **Four epochs on one instrument over one fortnight.** Spread capture is
  consistent across an up, a flat and a down epoch, but that is three
  observations of a ratio, not a distribution.
- **One flash move dominates any PnL sum that includes it** (epoch 68, above).
  Sign tests and medians are robust to it; sums and means are not. Every pooled
  PnL figure in this report states whether epoch 68 is in it.
- **Adverse selection is horizon-dependent in a way that undermines a single
  number.** GLFT's markouts on epoch 27 ran +1.574 bps at 100 ms, −1.202 at
  1 s, +0.108 at 5 s and +4.384 at 30 s. The reported figure takes the longest
  horizon, so the sign of the conclusion depends on that choice. Genuine
  adverse selection should worsen monotonically with horizon; this does not,
  which suggests the 30 s number is measuring the same price noise as the
  inventory term.

## 4. What has been learned

**Configuration defaults carry modelling assumptions, and the assumptions have
to come from the data.** Three defaults were set at equity scale and were wrong
for a crypto perpetual by about two orders of magnitude:

- `intensity_max_delta_bps = 20` is 222 USD on BTC, against a market spread of
  15 USD. Every trade fell in the first bucket and the fit had no slope to
  find. It was caught because the fit reported "only 1 usable bucket" rather
  than silently returning a plausible-looking number.
- Half-spread defaults quoted ~22 USD outside the touch. Nothing filled.

**The correction was then made against the wrong reference, and that was worse.**
The rescaled defaults were derived from the synthetic generator rather than
from the book. `synthetic.py` had `mid_start = 111_000` and placed adjacent
levels one 0.5 tick apart, which produces a ~1 USD spread; those two numbers
were read back out of the generator and written into the config comments, this
report and the fee appendix as though they described Bitfinex.

Measured on tBTCF0:USTF0 epoch 27 (7,300 observations on a 100 ms grid):

| | assumed | measured |
|---|---|---|
| mid | 111,000 | **79,996** |
| tick | 0.5 | **1.0**, and price-dependent |
| touch spread | ~1 USD (0.09 bps) | **15 USD (1.875 bps)** |
| trade distance from mid | under 1 bps | p50 0.94, **p99 5.32 bps** |

The tick is the sharpest of these. Bitfinex quotes to five significant digits,
so the increment is a function of price -- 1.0 for BTC at 76-80k, 0.1 for ETH
at 2,470, and it steps by a decade when price crosses a power of ten. No
constant is correct. Worse, 0.5 is not on the grid at any BTC price, so roughly
half of all quotes landed on a level the venue cannot represent, where the
reconstructed book necessarily shows zero depth -- the queue model read that as
an empty queue and filled instantly, at a price better than the whole market.

The generator has since been rescaled to the measured values, and the quoter
derives the tick from price and asserts that every quote lands on the grid.
- `horizon_seconds = 3600` is the interesting one. AS's risk term is
  γσ²(T−t), so an hour-long horizon gives 44 USD of half-spread and **AS never
  traded at all** while the other two strategies did. Setting T−t to the
  session length is the intuitive reading and it is wrong; the defensible
  reading for a continuously running maker is the horizon over which inventory
  is worked off, which at observed fill rates is about a minute.

**A closed form can be wrong in a way every invariant test accepts.** The GLFT
base distance was implemented as (1/κ)·ln(1 + κ/γ). Theorem 2 of the paper
gives (1/γ)·ln(1 + γ/k) — γ and κ transposed. The radical was wrong the same
way: its base takes γ/k while only its exponent takes k/γ, and both had been
written with k/γ.

Nothing in the test suite could see it. GLFT stayed exactly symmetric at zero
inventory, antisymmetric in inventory, monotone in σ and damped by A; every
sign condition held under either form. Two things hid it:

- **The default γ = 0.1 happened to sit at the calibrated κ = 0.1001.** At
  γ = κ the two expressions are algebraically equal (both 6.93 USD). The
  parameter the code was developed at is the one point where the bug is
  invisible.
- **The error is smooth.** At γ = 10⁻³ the correct base is 9.94 USD and the
  transposed one 46.12 — a plausible-looking number, just wrong.

Two checks would have caught it, and both are now tests. The first is a limit:
as γ → 0 a risk-neutral maker maximises δ·A·e^(−κδ), giving δ* = 1/κ. The
correct form converges there; the transposed one diverges. The second is
cheaper still — **AS and GLFT derive the base half-spread from the same
Hamilton-Jacobi term, so the two implementations must agree numerically.** They
did not, and nobody had compared them.

The same review found the inventory argument scaled by `q_max`, putting q in
[−1, 1]. The paper's q is "the (signed) quantity of shares he holds" with
transactions "scaled to 1", so dividing by the position limit shrinks the whole
inventory term tenfold at the default. GLFT ran a mean absolute inventory of
5.4 against AS's 0.87 on the same data: the inventory control the model exists
to provide was, in effect, switched off.

**An intensity is not a density.** λ(δ) = A·e^(−kδ) in the paper is the arrival
rate of executions against a quote resting at distance δ. The calibration
estimates something different: a rate per unit of distance, pooled across
aggressor directions. A resting order is executed by any trade printing at
distance ≥ δ, since a marketable order consumes every level it crosses, so the
two are related by A_paper = A_fitted/(2κ) — a factor of 4.8 here. A sits under
a square root in the inventory coefficient, so feeding in the density made
every GLFT quote ~2.2× too wide, enough to pin it against the maximum
half-spread clamp and stop it trading.

The horizon point is a genuine model difference worth reporting: **GLFT is immune
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

1. **Repeat across epochs and instruments.** Every figure below comes from
   BTC-PERP epochs on a handful of days. Inventory PnL has already been shown
   to change sign with the regime, so nothing that depends on it can be
   concluded from one window; spread capture looks stable across an up and a
   down epoch but that is two observations. ETH and XAUT are untouched.
2. **Attribute the residual sequence-number gaps** (section 2), which needs the
   collector source, and confirm that it persists `seq_gap` / `checksum_fail`
   rows when those mechanisms fire rather than only logging them.
   **Resolved.** The collector source enforces `seq == prev + 1` on the
   connection-level SEQ_ALL stream and writes `seq_gap` before forcing a
   resync, so zero `seq_gap` rows over 59 days and 561M rows means the stream
   never lost a message. The ~104k numbers absent from `book_mbo.seq` on epoch
   27 are the trades (16,168), heartbeats and per-iteration checksum messages
   that share the same connection. `checksum_fail` is likewise persisted: it
   fired for real on 2026-10-08 (9 events across all three symbols in 11 s),
   and the collector resynced each into a new epoch, so the incident is
   confined to BTC epoch 127, ETH 132 and XAUT 123.
3. **Funding: done by backfill, not by changing the collector.** The REST
   endpoint `/v2/status/deriv/<SYMBOL>/hist` serves ~60s history for the
   whole collected period when given explicit `start`/`end` (without them it
   returns only the last few hours, which is what made it look
   unrecoverable). `scripts/backfill_deriv.py` loads it into
   `bfx.deriv_status`; the engine charges each 8h settlement on the signed
   position held at that instant, and attribution carries funding as a
   fourth leg. On epoch 27 it costs symmetric 1,595 against 433 of spread
   capture (368%), AS 116 against 586 (20%), GLFT 173 against 622 (28%) -- but
   epoch 27 is not representative of the sign: across 14 epochs symmetric
   receives funding in 8 (section 3). The attribution sums exactly: inventory PnL is unchanged by adding it, and
   total PnL falls by precisely the funding term. Mean settled rates over the
   backfill: BTC +0.029%, ETH +0.025%, XAUT -0.003% per 8h -- XAUT's funding
   runs the other way. Adding `status:deriv` to the collector would only buy
   sub-minute mark price, which nothing here uses.
4. **Fee sensitivity on the remaining epochs and instruments.** Done for
   epoch 27 (appendix), which corrected the claimed cost of the standard tier
   a second time: measured against *captured* spread rather than quoted
   spread, an 8 bp fee is 8.5-13x the edge, not 4.3x. The other three epochs
   and ETH/XAUT are untouched, and the rebate tier is the only one worth
   carrying forward into the adaptive study.
5. **Queue-model sensitivity**, quantifying how much `--no-queue-model` inflates
   results, so the magnitude is reported rather than asserted. This now has a
   companion measurement: the share of fills landing on levels the strategy
   created rather than joined runs 84-92%, and those are precisely the fills
   the replay models least well.
6. **Sensitivity to gamma**, which turned out to be a scale parameter rather
   than a free one. Against a calibrated kappa the workable range on BTC-PERP
   is roughly 1e-4 to 3e-3; the default 0.1 pins both AS and GLFT against the
   maximum half-spread clamp and stops them trading. Worth reporting as a
   result in its own right, alongside the horizon sensitivity.
7. **Then** the LLM-adaptive layer, on top of a baseline that has been shown to
   behave.

---

## 6. Open questions for supervision

1. ~~**Funding-rate collection**~~ — no longer needs the collector touched:
   history is backfilled over REST (next steps, item 3). If live collection is
   wanted later, a restart is safe -- the collector resumes its epoch counter
   from `max(epoch)` in ClickHouse, so it costs one epoch boundary, and BTC
   already has 222 of those over 59 days.
2. **Cross-instrument scope** — is the BTC/ETH/XAUT comparison (with XAUT as a
   non-crypto control) worth the extra work, or is a single instrument done
   thoroughly the better use of the remaining time?
3. **AS horizon parameter** — is treating T−t as the inventory-holding horizon
   rather than the session length the right call, and should the sensitivity to
   it be reported as a result in its own right?
4. **How much of the counterfactual to accept.** Between 84% and 92% of fills
   land on price levels the strategy created rather than joined. On a book
   quoting a 15 USD spread this is unavoidable — stepping inside the touch is
   what a maker does, and being alone at a new level genuinely means being
   first in the queue — but those fills never happened on the tape, so their
   PnL rests on our quote not having changed the flow that traded against it.
   A replay cannot test that assumption. Options are to report the share and
   argue the orders are small enough not to matter, to add an explicit market
   impact model, or to restrict attention to fills where we joined an existing
   queue and accept a much smaller sample. This seems the most consequential
   open methodological choice.
5. **Whether inventory PnL should be reported at all.** It dominates total PnL
   in every run so far, changes sign with the market regime, and has |t| < 1
   throughout — on BTC epoch 27 the net was −7,866 against a gross of 5.1M.
   Reporting total PnL without this decomposition invites reading a random
   residual as skill, which is how the earlier version of this project
   produced a +2% annualised figure that had to be retracted. The current
   answer is to report net, gross and t together and to warn when the net is
   not significant, but a cleaner convention may exist.

---

## Appendix: fee sensitivity, real data

Run with `--fees 0,8,-1` on BTC-PERP epoch 27 (39.9 h, 4.32M book events,
16,168 trades). Fees are arithmetic on traded notional, so this table is one
replay per strategy scored three ways: the fills, the spread capture and the
inventory path are identical down each column, and only the fee term moves.

| epoch 27 | symmetric | avellaneda_stoikov | glft |
|---|---|---|---|
| spread capture | +433.38 | +585.66 | +621.50 |
| inventory (t) | −7,865.58 (−0.43) | −1,999.21 (−0.85) | −1,353.60 (−0.52) |
| fees @ 0 bp | 0.00 | 0.00 | 0.00 |
| fees @ 8 bp | −5,635.68 | −4,973.31 | −5,330.43 |
| fees @ −1 bp | +704.46 | +621.66 | +666.30 |

Read the fee rows against the spread capture row, not against total PnL: the
inventory term is a random-walk residual with |t| < 1 (section 3) and would
swamp the comparison for reasons that have nothing to do with fees.

### The size of the fee claim, corrected twice

This appendix has now stated three different multiples for how much a standard
maker fee costs relative to the spread, and the arithmetic behind the change is
worth keeping visible.

| version | claimed spread | claimed multiple | where it came from |
|---|---|---|---|
| first | 0.09 bps | ~90× | synthetic generator's 1 USD spread on a 111k mid |
| second | 1.875 bps | ~4.3× | measured *quoted* touch spread, real book |
| **current** | **0.615–0.942 bps** | **8.5–13×** | **spread actually captured per fill** |

The second correction fixed the instrument but compared against the wrong
quantity. A maker quoting at the touch does not capture the touch spread: it
captures the distance from the mid to its own fill, and adverse selection eats
into that before the position is unwound. Measured, the strategies capture
0.615 (symmetric), 0.942 (AS) and 0.933 (GLFT) bps per unit traded — roughly
half the 1.875 bps quoted. So an 8 bp fee is 13.0× what symmetric actually
earns per unit and 8.5× what AS earns, and the absolute numbers agree: 5,635.68
of fees against 433.38 of spread capture is the same 13.0.

The direction has survived all three versions — at the standard tier, passive
market making on this instrument is unprofitable on the fee term alone, before
inventory risk or adverse selection enter. The magnitude has not, and the ratio
that matters is the one against captured spread, because that is the revenue
the fee is actually charged against.

### Consequences for the research design

1. **The standard tier is unviable by roughly an order of magnitude.** Any
   write-up has to state the fee regime up front; a reader assuming standard
   fees will correctly conclude the whole exercise is unprofitable.
2. **The rebate tier is where the strategy exists, but it is no longer the
   whole story.** At −1 bp the rebate is +704 against +433 of spread capture
   for symmetric: the subsidy exceeds the earned edge, and for the naive
   quoter the rebate *is* the business. For AS and GLFT the two are comparable
   (+622 vs +586, +666 vs +622), so quoting skill and the subsidy contribute on
   the same order. That is the regime in which "does adaptive control improve
   quoting" is a question with content.
3. **Fee exposure tracks turnover, not skill.** Ranked by fees paid at 8 bp the
   order is symmetric > GLFT > AS, which is just the ranking by notional
   traded. A cross-tier comparison that does not normalise for turnover will
   read low fill rates as strategy quality.
4. **Fees do not change the strategy ranking here, because nothing does yet.**
   At every tier all three totals are dominated by the inventory residual, so
   the fee study cannot be used to rank the strategies until the inventory term
   is either controlled for or averaged out over enough epochs.
