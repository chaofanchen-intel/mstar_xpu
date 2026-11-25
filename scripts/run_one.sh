#!/usr/bin/env bash
# One experiment: guard -> start server -> wait ready -> benchmark -> stop (always).
# Usage: scripts/run_one.sh <results_dir> <label> <mstar|omni> <config> <n_gpus> <cases> [bench_bagel.py args...]
#   results_dir: e.g. results/20261122_rerun        label: e.g. mstar-tp2
# Refuses to start (exit 1) if fewer than n_gpus XPUs are idle.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
res="${1:?}"; label="${2:?}"; fw="${3:?}"; config="${4:?}"; ngpu="${5:?}"; cases="${6:?}"; shift 6
config="$(realpath "$config")"
run_dir="$WS/$res/$label"; mkdir -p "$run_dir"
[[ "${NO_COMPILE:-0}" == 1 ]] && export BENCH_DISABLE_COMPILE=1
[[ "${MATH_SDPA:-0}" == 1 ]] && export BENCH_FORCE_MATH_SDPA=1
if [[ "${NO_COMPILE:-0}" == 1 || "${MATH_SDPA:-0}" == 1 ]]; then export PYTHONPATH="$WS/scripts/diagnostics/hooks${PYTHONPATH:+:$PYTHONPATH}"; fi
port=$([[ $fw == mstar ]] && echo 18765 || echo 18767)

gpus=$("$WS/scripts/gpu_guard.sh" "$ngpu") || exit 1
echo "[$label] using XPU $gpus" | tee "$run_dir/gpus.txt"

sampler() {  # per-card memory/util timeline for all cards, to spot foreign load on our cards
  echo "ts,dev,mem_mib,util_pct" > "$run_dir/gpu_timeline.csv"
  while true; do
    for d in 0 1 2; do
      s=$(xpu-smi stats -d $d 2>/dev/null)
      m=$(awk -F'|' '/GPU Memory Used/ {gsub(/ /,"",$3); print $3}' <<<"$s")
      u=$(awk -F'|' '/Average % utilization of/ {gsub(/ /,"",$3); print $3; exit}' <<<"$s")
      echo "$(date +%s),$d,$m,$u" >> "$run_dir/gpu_timeline.csv"
    done
    sleep 5
  done
}
cleanup() { kill "${sampler_pid:-0}" 2>/dev/null; "$WS/scripts/stop_server.sh" "$run_dir"; }
trap cleanup EXIT

if [[ $fw == mstar ]]; then "$WS/scripts/serve_mstar.sh" "$gpus" "$port" "$run_dir" "$config"
else "$WS/scripts/serve_omni.sh" "$gpus" "$port" "$run_dir" "$config"; fi
sampler & sampler_pid=$!
"$WS/scripts/wait_ready.sh" "$run_dir" "$port" /health 1500 || exit 2
echo "[$label] server ready: $(date -u +%FT%TZ)" | tee -a "$run_dir/gpus.txt"

"$MSTAR_VENV/bin/python" -I "$WS/scripts/bench_bagel.py" --framework "$fw" --mode "$label" \
  --url "http://127.0.0.1:$port" --cases "$cases" --out "$run_dir" "$@" 2>&1 | tee "$run_dir/bench.log"
rc=${PIPESTATUS[0]}
# steady-state memory right after the benchmark, before the server is stopped
for d in 0 1 2; do echo "xpu$d $(xpu-smi stats -d $d | awk -F'|' '/GPU Memory Used/ {gsub(/ /,"",$3); print $3}') MiB"; done > "$run_dir/gpu_mem_after.txt"
exit $rc
