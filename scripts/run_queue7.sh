#!/usr/bin/env bash
# Factors that change BAGEL performance: stage connector (SharedMemory / Mooncake-TCP / NIXL) and prefix caches.
cd "$(dirname "$0")/.."
R=results/20261122_factors; mkdir -p $R
O=configs/vllm_omni; M=configs/mstar
run() { local label=$1 fw=$2 cfg=$3 cases=$4; shift 4
  scripts/run_one.sh $R "$label" "$fw" "$cfg" 2 "$cases" "$@" > $R/$label.driver.log 2>&1; }
run omni-two-stage-shm            omni  $O/bagel_two_stage_2gpu.yaml              t2i,i2i
run omni-two-stage-mooncake-tcp   omni  $O/bagel_two_stage_2gpu_mooncake_tcp.yaml t2i,i2i
run omni-two-stage-nixl           omni  $O/bagel_two_stage_2gpu_nixl.yaml         t2i,i2i
run omni-two-stage-prefixcache    omni  $O/bagel_two_stage_2gpu_prefixcache.yaml  i2t_1tok,i2t_64tok --omni-ignore-eos --omni-prompt-style chat
run mstar-cfgpar-nocache          mstar $M/bagel_cfgpar_2gpu_nocache.yaml         i2t_1tok,i2t_64tok,i2i
echo done > $R/queue7.done
