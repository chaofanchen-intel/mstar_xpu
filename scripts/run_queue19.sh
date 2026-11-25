#!/usr/bin/env bash
cd "$(dirname "$0")/.."
R=results/20261124_kprof
scripts/run_profile.sh $R omni-t2i-cfgpar2 omni configs/vllm_omni/bagel_single_stage_cfgpar2.yaml 2 --case t2i > $R.omni.log 2>&1
scripts/run_profile.sh $R mstar-t2i-cfgpar mstar configs/mstar/bagel_cfgpar_3gpu.yaml 3 --case t2i > $R.mstar.log 2>&1
echo done > $R.done
