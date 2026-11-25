#!/usr/bin/env bash
# Launch M* TP2 graph on 3 cards with extra env, wait until the first LLM graph capture result, then stop.
# Usage: try_tp2_graph.sh <label> [VAR=VALUE ...]
cd "$(dirname "$0")/.."
label=$1; shift
R=$PWD/results/20261124_tp2graph; run_dir=$R/$label; mkdir -p $run_dir
gpus=$(scripts/gpu_guard.sh 3) || exit 1
for kv in "$@"; do export "$kv"; done
scripts/serve_mstar.sh "$gpus" 18765 "$run_dir" $PWD/${CFG:-configs/mstar/bagel_tp2_3gpu_graph.yaml} > /dev/null
for i in $(seq 1 90); do
  sleep 10
  kill -0 "$(cat $run_dir/server.pid 2>/dev/null)" 2>/dev/null || { echo "$label: server died"; break; }
  if grep -q "Failed to capture accelerator graph for LLM" $run_dir/server.log 2>/dev/null; then echo "$label: FAIL (first LLM capture failed)"; break; fi
  if grep -q "AcceleratorGraphRunner\[LLM\]: captured" $run_dir/server.log 2>/dev/null; then echo "$label: LLM summary: $(sed 's/\x1b\[[0-9;]*m//g' $run_dir/server.log | grep -m1 'AcceleratorGraphRunner\[LLM\]: captured' | cut -c1-120)"; break; fi
  if grep -q "warmup_and_capture done" $run_dir/server.log 2>/dev/null && grep -c "AcceleratorGraphRunner\[LLM\]: warmup_and_capture done" $run_dir/server.log | grep -q 2; then echo "$label: OK (all LLM captured)"; break; fi
done
sleep 5
scripts/stop_server.sh $run_dir
