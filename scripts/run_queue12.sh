#!/usr/bin/env bash
# 3 idle XPUs: graph on/off in the 3-card layouts, M* and vLLM-Omni. I2I uses the equal-workload instruction (see run_queue11.sh).
cd "$(dirname "$0")/.."
R=results/20261123_graph; mkdir -p $R; P="Replace the bagel with a croissant."
M=configs/mstar; O=configs/vllm_omni
CASES=t2i,i2t_1tok,i2t_64tok,i2i
m() { local label=$1 cfg=$2; shift 2
  scripts/run_one.sh $R "$label" mstar "$cfg" 3 $CASES --mstar-i2i-preprocess default --i2i-prompt "$P" "$@" > $R/$label.driver.log 2>&1; }
o() { local label=$1 cfg=$2; shift 2
  scripts/run_one.sh $R "$label" omni "$cfg" 3 $CASES --i2i-prompt "$P" "$@" > $R/$label.driver.log 2>&1; }
m mstar-cfgpar3-eager        $M/bagel_cfgpar_3gpu.yaml
m mstar-cfgpar3-graph        $M/bagel_cfgpar_3gpu_graph.yaml
m mstar-cfgpar3vit-eager     $M/bagel_cfgpar_3gpu_vitrank.yaml
m mstar-cfgpar3vit-graph     $M/bagel_cfgpar_3gpu_vitrank_graph.yaml
m mstar-tp2-3gpu-eager       $M/bagel_tp2_3gpu.yaml
m mstar-tp2-3gpu-graph       $M/bagel_tp2_3gpu_graph.yaml
o omni-single-cfgpar3        $O/bagel_single_stage_cfgpar3.yaml
o omni-two-stage-3gpu-tp2-eager $O/bagel_two_stage_3gpu_tp2.yaml
o omni-two-stage-3gpu-tp2-graph $O/bagel_two_stage_3gpu_tp2_graph.yaml
echo done > $R/queue12.done
