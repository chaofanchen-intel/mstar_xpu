#!/usr/bin/env bash
# Omni two-stage I2T re-run with the server-side chat template for the image placeholder
# (the hand-written <|image_pad|> prompt is mis-handled by vLLM-Omni main / vLLM 0.34).
cd "$(dirname "$0")/.."
R=results/20261122_best_cfg/final
mkdir -p $R/_invalid_bagel_raw_prompt && mv $R/omni-two-stage $R/omni-two-stage.driver.log $R/omni-two-stage-graph $R/omni-two-stage-graph.driver.log $R/_invalid_bagel_raw_prompt/ 2>/dev/null
for v in "omni-two-stage bagel_two_stage_2gpu.yaml" "omni-two-stage-graph bagel_two_stage_2gpu_graph.yaml"; do
  set -- $v
  scripts/run_one.sh $R $1 omni configs/vllm_omni/$2 2 i2t_1tok,i2t_64tok --omni-ignore-eos --omni-prompt-style chat > $R/$1.driver.log 2>&1
done
echo done > $R/queue6.done
