#!/usr/bin/env bash
# vLLM-Omni per-stage durations (diffusion pipeline profiler: vae.encode/decode, diffuse, per denoise_step; syncs around each call).
cd "$(dirname "$0")/.."
R=results/20261123_graph; P="Replace the bagel with a croissant."; O=configs/vllm_omni
export OMNI_EXTRA_ARGS="--enable-diffusion-pipeline-profiler"
scripts/run_one.sh $R omni-prof-t2i-cfgpar2 omni $O/bagel_single_stage_cfgpar2.yaml 2 t2i > $R/omni-prof-t2i-cfgpar2.driver.log 2>&1
scripts/run_one.sh $R omni-prof-i2i-cfgpar3 omni $O/bagel_single_stage_cfgpar3.yaml 3 i2i --i2i-prompt "$P" > $R/omni-prof-i2i-cfgpar3.driver.log 2>&1
echo done > $R/queue18.done
