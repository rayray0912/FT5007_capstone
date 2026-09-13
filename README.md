# FT5007 Capstone — Adaptive Market Making on Crypto Perpetuals

Market-making backtest framework built on order-level (MBO) data from Bitfinex
perpetual swaps. Implements a Guéant–Lehalle–Fernandez-Tapia (GLFT) quoter and
evaluates it against classical baselines on a shared simulator.

**Status:** baselines implemented and running; LLM-adaptive layer not yet built.
See [`docs/progress_report.md`](docs/progress_report.md).

---

## What this is

The research question is whether adaptive parameter control improves
market-making performance over statically calibrated quoting on a thin-book
crypto perpetual. To answer it credibly the backtest has to be realistic about
the two things that usually break market-making backtests: queue position and
fees. Both are handled explicitly here.

Three strategies share one simulator, one data stream, one calibration and one
metric layer, so any difference between them is attributable to the strategy:

| strategy | what it adds | what it isolates |
|---|---|---|
| `symmetric` | fixed half-spread, no inventory logic | value of simply being present in the book |
| `avellaneda_stoikov` | inventory-skewed reservation price, finite horizon | value of inventory management |
| `glft` | explicit inventory bound, asymmetric per-side distances | value of solving the constrained problem |

## Instrument selection

**The target is BTC / ETH / XAUT perpetuals on Bitfinex, not HYPE.**

HYPE-PERP was excluded during instrument selection on liquidity grounds: it
turns over roughly 13,675 USD per 24h on this venue, a few dozen trades a day.
The AS/GLFT family calibrates fill intensity λ(δ) from the trade flow, and that
sample size cannot support the estimate, let alone a meaningful fill backtest.

The three instruments retained are the only ones on the venue with real trade
flow. XAUT (tokenised gold) is kept as a cross-asset control: its quoting
structure differs markedly from BTC/ETH, which makes it useful for testing
whether conclusions are specific to crypto microstructure.

Full reasoning and the venue comparison table: [`docs/dataset.md`](docs/dataset.md).

## Quick start

```bash
pip install -r requirements.txt

# Develop offline, no ClickHouse needed (synthetic data, not for results)
python scripts/run_backtest.py --synthetic --n-events 60000 compare --fees 0

# Against the real dataset (requires Tailscale to the collector host)
python scripts/run_backtest.py --symbol tBTCF0:USTF0 compare --fees 0,8,-1

# Single strategy, with output
python scripts/run_backtest.py run --strategy glft --out results/

pytest tests/ -q
```

## Layout

```
src/mmbacktest/
  config.py              typed run configuration; a run is fully described by it
  data/
    clickhouse_client.py epoch-scoped queries + integrity gate
    book.py              order-level book reconstruction, queue tracking
    synthetic.py         generated stream for offline development
  calibration/
    intensity.py         λ(δ) = A·exp(−κδ), session-stratified
    volatility.py        realised σ on a fixed time grid
  strategy/
    base.py              Quoter interface, shared quote hygiene
    symmetric.py         naive baseline
    avellaneda_stoikov.py
    glft.py              main strategy
  sim/
    fills.py             queue-position-aware fill simulation, markouts
    engine.py            replay loop on a 100ms decision clock
  metrics/
    performance.py       absolute + market-making + attribution metrics
  cli.py
```

## Design decisions worth knowing about

**Replays are epoch-scoped.** The collector increments `epoch` on every
reconnect. A replay crossing an epoch boundary contains ghost orders — orders
whose delete message fell into the gap and which therefore never leave the
reconstructed book. Every query is constrained to a single epoch.

**Fills are conditional on queue position.** The naive rule (fill whenever a
trade prints at or through my price) inflates fill counts on a tight book, and
it inflates them in the direction that flatters the strategy: real queue
priority means fills cluster where flow is heaviest, and heavy flow correlates
with informed flow, so unconditional filling strips out most of the adverse
selection along with the queue wait. `--no-queue-model` runs the naive rule so
the size of that difference can be reported rather than asserted.

**Quotes persist across decision ticks.** Cancelling and reposting at the same
price sends the order to the back of the queue. A real quoting system amends
only when the desired price actually moves, and so does this one. This turned
out to be load-bearing rather than cosmetic — see the progress report.

**Fees are a parameter, not an assumption.** `--fees 0,8,-1` runs the zero-fee
case, the standard maker rate, and the market-maker rebate tier in one pass.
Reporting a single fee level is how a strategy ends up looking profitable at
one tier and not at another.

## Known limitations

- **No funding-rate data yet.** The `status:deriv` channel (mark price, index
  price, funding rate) is not being collected. For a perpetual, funding is a
  real component of inventory carrying cost, so the current PnL attribution
  understates the cost of holding inventory. This is the most consequential
  open gap.
- **Venue is mid-liquidity.** Bitfinex BTC-PERP turns over ~13M USD/24h,
  two to three orders of magnitude below the major venues. Parameters
  calibrated here do not transfer directly to a thick book. This is a scope
  statement, not a defect — a thin book makes individual maker behaviour
  easier to identify.
- **Synthetic mode is for development only.** It has no informed flow, so PnL
  from it is meaningless. Behavioural metrics (inventory control, fill rate,
  spread capture) are still informative for checking that a strategy does what
  it claims.
