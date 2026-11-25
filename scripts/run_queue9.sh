#!/usr/bin/env bash
# I2I workload parity (main): M* with default ViT preprocessing (1369 patches, 1024 VAE latents at 512^2)
# = vLLM-Omni main's I2I input at 512^2. Earlier I2I runs used image_preprocess:"vllm" (4900 patches), which
# matches vLLM-Omni v0.32.0 only. The Omni run also records usage.prompt_tokens for I2I to confirm the count.
cd "$(dirname "$0")/.."
R=results/20261122_i2i_parity; mkdir -p $R
M=configs/mstar
run() { local label=$1 fw=$2 cfg=$3; shift 3
  scripts/run_one.sh $R "$label" "$fw" "$cfg" 2 i2i "$@" > $R/$label.driver.log 2>&1; }
run omni-single-tp2      omni configs/vllm_omni/bagel_single_stage_tp2.yaml
run mstar-cfgpar         mstar $M/bagel_cfgpar_2gpu.yaml --mstar-i2i-preprocess default
run mstar-tp2            mstar $M/bagel_tp2_2gpu.yaml --mstar-i2i-preprocess default
NO_COMPILE=1 run mstar-cfgpar-nocompile mstar $M/bagel_cfgpar_2gpu.yaml --mstar-i2i-preprocess default
run mstar-cfgpar-nocache mstar $M/bagel_cfgpar_2gpu_nocache.yaml --mstar-i2i-preprocess default
echo done > $R/queue9.done
