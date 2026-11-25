#!/usr/bin/env bash
# Kernel-level profile of the denoise loop. Usage: run_profile.sh <results_dir> <label> <mstar|omni> <config> <n_gpus> [driver args...]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
res="$1"; label="$2"; fw="$3"; config="$(realpath "$4")"; ngpu="$5"; shift 5
run_dir="$WS/$res/$label"; mkdir -p "$run_dir/prof"
export PYTHONPATH="$WS/scripts/diagnostics/hooks${PYTHONPATH:+:$PYTHONPATH}"
export BENCH_TORCH_PROFILE_DIR="$run_dir/prof"
port=$([[ $fw == mstar ]] && echo 18765 || echo 18767)
gpus=$("$WS/scripts/gpu_guard.sh" "$ngpu") || exit 1
echo "[$label] using XPU $gpus" | tee "$run_dir/gpus.txt"
trap '"$WS/scripts/stop_server.sh" "$run_dir"' EXIT
if [[ $fw == mstar ]]; then "$WS/scripts/serve_mstar.sh" "$gpus" "$port" "$run_dir" "$config"
else "$WS/scripts/serve_omni.sh" "$gpus" "$port" "$run_dir" "$config"; fi
"$WS/scripts/wait_ready.sh" "$run_dir" "$port" /health 1500 || exit 2
"$MSTAR_VENV/bin/python" -I "$WS/scripts/profile_denoise.py" "$fw" "http://127.0.0.1:$port" "$run_dir/prof" "$@" 2>&1 | tee "$run_dir/profile.log"
