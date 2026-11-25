#!/usr/bin/env bash
# Start a vLLM-Omni BAGEL server in the background.
# Usage: scripts/serve_omni.sh <physical-gpus e.g. 1,2> <port> <run_dir> <deploy-config template>
# The template's @GPUS@ / @GPU_A@ / @GPU_B@ placeholders are replaced with the given ids; the
# resolved file is kept in <run_dir>/effective_config.yaml.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
gpus="${1:?gpus}"; port="${2:?port}"; run_dir="${3:?run_dir}"; template="${4:?deploy config template}"
IFS=, read -r gpu_a gpu_b gpu_c _ <<< "$gpus"
mkdir -p "$run_dir"
sed -e "s/@GPUS@/$gpus/g" -e "s/@GPU_A@/$gpu_a/g" -e "s/@GPU_B@/${gpu_b:-$gpu_a}/g" -e "s/@GPU_C@/${gpu_c:-$gpu_a}/g" "$template" > "$run_dir/effective_config.yaml"
export LD_LIBRARY_PATH="$OMNI_VENV/lib:$FFMPEG_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ZE_AFFINITY_MASK   # vLLM-Omni sets it per stage from `devices`
cd "$WS/external/vllm-omni"
setsid nohup "$OMNI_VENV/bin/vllm" serve "$MODEL_ID" --omni --deploy-config "$run_dir/effective_config.yaml" \
  --host 127.0.0.1 --port "$port" --dtype bfloat16 ${OMNI_EXTRA_ARGS:-} > "$run_dir/server.log" 2>&1 &
echo $! > "$run_dir/server.pid"
echo "omni pid $(cat "$run_dir/server.pid") on XPU $gpus, port $port, log $run_dir/server.log"
