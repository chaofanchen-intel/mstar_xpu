#!/usr/bin/env bash
# I2I workload parity, second attempt. M* I2I with image_preprocess:"default" (1369 ViT patches) drops the
# requested edit (no cat in the output), so it cannot be compared. Keep M*'s valid 4900-patch ("vllm") I2I and
# give vLLM-Omni the same token count: the 512^2 input upscaled to 980^2 -> ViT 4900 patches, VAE 1024 latents.
cd "$(dirname "$0")/.."
R=results/20261122_i2i_equal; mkdir -p $R
scripts/run_one.sh $R omni-single-tp2-i2i980 omni configs/vllm_omni/bagel_single_stage_tp2.yaml 2 i2i \
  --omni-i2i-size 980 > $R/omni-single-tp2-i2i980.driver.log 2>&1
echo done > $R/queue10.done
