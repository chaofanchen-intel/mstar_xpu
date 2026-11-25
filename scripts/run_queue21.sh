#!/usr/bin/env bash
# Kernel profile of the denoise loop, short window (one-two steps) so the XPU profiler buffer does not overflow.
cd "$(dirname "$0")/.."
R=results/20261124_kprof2
scripts/run_profile.sh $R omni-t2i-cfgpar2 omni configs/vllm_omni/bagel_single_stage_cfgpar2.yaml 2 --case t2i --at 2.5 --window 0.35 > $R.omni.log 2>&1
scripts/run_profile.sh $R mstar-t2i-cfgpar mstar configs/mstar/bagel_cfgpar_3gpu.yaml 3 --case t2i --at 2.5 --window 0.25 > $R.mstar.log 2>&1
echo done > $R.done
