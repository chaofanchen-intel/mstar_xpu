#!/usr/bin/env bash
# Final matrix for the best-configuration comparison (2 idle XPUs, 2 warmups + 5 repeats).
cd "$(dirname "$0")/.."
R=results/20261122_best_cfg/final; mkdir -p $R
O=configs/vllm_omni; M=configs/mstar
run() { local label=$1 fw=$2 cfg=$3 cases=$4; shift 4
  scripts/run_one.sh $R "$label" "$fw" "$cfg" 2 "$cases" "$@" > $R/$label.driver.log 2>&1; }
ALL=t2i,i2t_1tok,i2t_64tok,i2i
run mstar-cfgpar                    mstar $M/bagel_cfgpar_2gpu.yaml $ALL
NO_COMPILE=1 run mstar-cfgpar-nocompile mstar $M/bagel_cfgpar_2gpu.yaml $ALL
run mstar-tp2                       mstar $M/bagel_tp2_2gpu.yaml    $ALL
run omni-single-cfgpar2             omni  $O/bagel_single_stage_cfgpar2.yaml t2i
run omni-single-tp2                 omni  $O/bagel_single_stage_tp2.yaml     t2i,i2i
run omni-two-stage                  omni  $O/bagel_two_stage_2gpu.yaml       $ALL --omni-ignore-eos
run omni-two-stage-graph            omni  $O/bagel_two_stage_2gpu_graph.yaml i2t_1tok,i2t_64tok --omni-ignore-eos
echo done > $R/queue5.done
