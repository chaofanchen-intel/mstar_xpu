#!/usr/bin/env bash
# Diagnostic: does M*'s default I2I preprocessing (1369 ViT patches) apply an edit at all?
# Same stronger instruction for both frameworks, M* default vs vLLM-Omni native (512^2 input).
cd "$(dirname "$0")/.."
R=results/20261122_i2i_diag; mkdir -p $R
P="Replace the bagel with a croissant."
scripts/run_one.sh $R mstar-cfgpar-default-croissant mstar configs/mstar/bagel_cfgpar_2gpu.yaml 2 i2i \
  --mstar-i2i-preprocess default --i2i-prompt "$P" > $R/mstar-cfgpar-default-croissant.driver.log 2>&1
echo done > $R/diag1.done
# Equal-workload pair with the instruction M* follows at 1369 patches: vLLM-Omni native 512^2 input (1369 patches)
scripts/run_one.sh $R omni-single-tp2-croissant omni configs/vllm_omni/bagel_single_stage_tp2.yaml 2 i2i \
  --i2i-prompt "$P" > $R/omni-single-tp2-croissant.driver.log 2>&1
echo done > $R/diag2.done
