#!/usr/bin/env bash
# Omni two-stage candidates (eager vs Thinker graph) after fixing the --max-model-len flag issue.
cd "$(dirname "$0")/.."
R=results/20261122_best_cfg
run() { local label=$1 fw=$2 cfg=$3 cases=$4; shift 4
  scripts/run_one.sh $R "$label" "$fw" "$cfg" 2 "$cases" --repeats 3 "$@" > $R/$label.driver.log 2>&1; }
run omni-two-stage       omni configs/vllm_omni/bagel_two_stage_2gpu.yaml       t2i,i2t_64tok --omni-ignore-eos
run omni-two-stage-graph omni configs/vllm_omni/bagel_two_stage_2gpu_graph.yaml t2i,i2t_64tok --omni-ignore-eos
echo done > $R/queue4.done
