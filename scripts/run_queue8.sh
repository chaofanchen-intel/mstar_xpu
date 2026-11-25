#!/usr/bin/env bash
# I2T workload parity: M* with default ViT preprocessing (1369 patches = Omni Thinker's 1386 prompt tokens),
# instead of the 4900-patch "vllm" mode used by mistake in queues 1-7. I2T cases only; T2I/I2I are unaffected.
cd "$(dirname "$0")/.."
R=results/20261122_i2t_parity; mkdir -p $R
M=configs/mstar
run() { local label=$1 cfg=$2; shift 2
  scripts/run_one.sh $R "$label" mstar "$cfg" 2 i2t_1tok,i2t_64tok "$@" > $R/$label.driver.log 2>&1; }
run mstar-cfgpar            $M/bagel_cfgpar_2gpu.yaml
NO_COMPILE=1 run mstar-cfgpar-nocompile  $M/bagel_cfgpar_2gpu.yaml
run mstar-tp2               $M/bagel_tp2_2gpu.yaml
run mstar-cfgpar-nocache    $M/bagel_cfgpar_2gpu_nocache.yaml
echo done > $R/queue8.done
