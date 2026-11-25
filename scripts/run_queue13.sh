#!/usr/bin/env bash
# Follow-up to run_queue12.sh: vLLM-Omni cfg_parallel_size=3 only supports 3-branch CFG (needs cfg_img_scale>1),
# so the T2I case errors; run I2I alone once queue12 has finished.
cd "$(dirname "$0")/.."
R=results/20261123_graph
while [ ! -f $R/queue12.done ]; do sleep 30; done
scripts/run_one.sh $R omni-single-cfgpar3-i2i omni configs/vllm_omni/bagel_single_stage_cfgpar3.yaml 3 i2i \
  --i2i-prompt "Replace the bagel with a croissant." > $R/omni-single-cfgpar3-i2i.driver.log 2>&1
echo done > $R/queue13.done
