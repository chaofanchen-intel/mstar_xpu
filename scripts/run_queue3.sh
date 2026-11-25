#!/usr/bin/env bash
# Screening queue: candidate "best" configurations of each framework on 2 idle XPUs.
# t2i is the most sensitive case; i2t_64tok exposes decode (TP vs single-card). Each run is independent.
cd "$(dirname "$0")/.."
R=results/20261122_best_cfg; mkdir -p $R
run() { # label fw config cases [extra...]
  local label=$1 fw=$2 cfg=$3 cases=$4; shift 4
  scripts/run_one.sh $R "$label" "$fw" "$cfg" 2 "$cases" --repeats 3 "$@" > $R/$label.driver.log 2>&1
}
O=configs/vllm_omni; M=configs/mstar
run omni-single-tp2        omni $O/bagel_single_stage_tp2.yaml     t2i
run omni-single-cfgpar2    omni $O/bagel_single_stage_cfgpar2.yaml t2i
run omni-two-stage         omni $O/bagel_two_stage_2gpu.yaml       t2i,i2t_64tok --omni-ignore-eos
run omni-two-stage-graph   omni $O/bagel_two_stage_2gpu_graph.yaml t2i,i2t_64tok --omni-ignore-eos
run mstar-tp2              mstar $M/bagel_tp2_2gpu.yaml            t2i,i2t_64tok
run mstar-cfgpar           mstar $M/bagel_cfgpar_2gpu.yaml         t2i,i2t_64tok
run mstar-tp2-graph        mstar $M/bagel_tp2_2gpu_graph.yaml      t2i,i2t_64tok
run mstar-cfgpar-graph     mstar $M/bagel_cfgpar_2gpu_graph.yaml   t2i,i2t_64tok
echo done > $R/queue3.done
