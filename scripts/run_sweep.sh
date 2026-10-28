#!/usr/bin/env bash
# One-shot launcher for the epoch sweep. Run from the repo root on the Mac:
#     bash scripts/run_sweep.sh                 # BTC, gamma 1e-4 / 1e-3 / 1e-2
#     SYMBOL=tETHF0:USTF0 bash scripts/run_sweep.sh
#
# Runs the three gamma tiers in parallel in the background, keeps the Mac awake
# until they finish, and is safe to re-run: finished epochs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

SYMBOL="${SYMBOL:-tBTCF0:USTF0}"
GAMMAS="${GAMMAS:-0.0001 0.001 0.01}"
export NO_PROXY="100.76.49.84,localhost,127.0.0.1"
export no_proxy="$NO_PROXY"

# Activate the project env if we are not already in it.
if [[ "${CONDA_DEFAULT_ENV:-}" != "FT5007" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate FT5007
fi

python -c "import clickhouse_connect, pyarrow, scipy" \
  || { echo "missing deps: pip install clickhouse-connect pyarrow scipy"; exit 1; }

# Fail fast if the collector is unreachable (Tailscale down / proxy in TUN mode).
curl -sS -m 10 --noproxy '*' "http://100.76.49.84:8123/?query=SELECT%201" >/dev/null \
  || { echo "cannot reach ClickHouse at 100.76.49.84:8123 - check Tailscale / proxy TUN mode"; exit 1; }

mkdir -p results/sweep
pids=()
for g in $GAMMAS; do
  log="results/sweep/log_${SYMBOL//:/_}_gamma_${g}.txt"
  nohup python scripts/sweep_epochs.py --symbol "$SYMBOL" --gamma "$g" >> "$log" 2>&1 &
  pids+=($!)
  echo "started gamma=$g  pid=$!  log=$log"
done

# Keep the machine awake until every sweep process exits (lid-close still sleeps
# a MacBook unless it is on power with an external display).
caffeinate -i -w "${pids[0]}" &
for p in "${pids[@]:1}"; do caffeinate -i -w "$p" & done

echo
echo "running in background. progress:  tail -f results/sweep/log_*.txt"
echo "partial summary any time:        python scripts/aggregate_sweep.py"
