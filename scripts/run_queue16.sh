#!/usr/bin/env bash
# M* I2T with prefix cache off (equal to Omni default), and per-phase timing of the decode loop (eager vs graph).
cd "$(dirname "$0")/.."
R=results/20261123_graph; M=configs/mstar
s() { local label=$1 cfg=$2; shift 2; scripts/run_one.sh $R "$label" mstar $M/$cfg 3 i2t_1tok,i2t_64tok "$@" > $R/$label.driver.log 2>&1; }
s mstar-cfgpar3-eager-nocache bagel_cfgpar_3gpu_nocache.yaml
s mstar-cfgpar3-graph-nocache bagel_cfgpar_3gpu_graph_nocache.yaml
MSTAR_PHASE_TIMING=50 s mstar-phase-eager bagel_cfgpar_3gpu.yaml
MSTAR_PHASE_TIMING=50 s mstar-phase-graph bagel_cfgpar_3gpu_graph.yaml
echo done > $R/queue16.done
