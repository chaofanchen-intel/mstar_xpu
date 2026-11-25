#!/usr/bin/env bash
# Start an M* BAGEL server in the background.
# Usage: scripts/serve_mstar.sh <physical-gpus e.g. 1,2> <port> <run_dir> [config.yaml]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
gpus="${1:?gpus}"; port="${2:?port}"; run_dir="${3:?run_dir}"
config="${4:-$WS/configs/mstar/bagel_tp2_2gpu.yaml}"
mkdir -p "$run_dir"
cp "$config" "$run_dir/effective_config.yaml"
export LD_LIBRARY_PATH="$FFMPEG_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ZE_AFFINITY_MASK="$gpus"
cd "$WS/external/mstar"
setsid nohup "$MSTAR_VENV/bin/mstar" serve bagel --config "$config" --cache-dir "$HF_HUB_CACHE" \
  --host 127.0.0.1 --port "$port" -O"${MSTAR_OPT:-0}" --log-level INFO > "$run_dir/server.log" 2>&1 &
echo $! > "$run_dir/server.pid"
echo "mstar pid $(cat "$run_dir/server.pid") on XPU $gpus, port $port, log $run_dir/server.log"
