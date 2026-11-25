#!/usr/bin/env bash
# I2I with one instruction both frameworks follow at 1369 ViT patches ("Replace the bagel with a croissant."):
# M* default preprocessing vs vLLM-Omni native 512^2 input. mstar-cfgpar and omni-single-tp2 were run by
# run_diag_i2i.sh; this queue adds the other M* configurations so every I2I row in the report is the same workload.
cd "$(dirname "$0")/.."
R=results/20261122_i2i_diag; P="Replace the bagel with a croissant."
M=configs/mstar
run() { local label=$1 cfg=$2; shift 2
  scripts/run_one.sh $R "$label" mstar "$cfg" 2 i2i --mstar-i2i-preprocess default --i2i-prompt "$P" "$@" > $R/$label.driver.log 2>&1; }
run mstar-tp2-default-croissant        $M/bagel_tp2_2gpu.yaml
NO_COMPILE=1 run mstar-cfgpar-nocompile-default-croissant $M/bagel_cfgpar_2gpu.yaml
run mstar-cfgpar-nocache-default-croissant $M/bagel_cfgpar_2gpu_nocache.yaml
echo done > $R/queue11.done
