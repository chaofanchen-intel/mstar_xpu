#!/usr/bin/env bash
# Queue: M* default, M* with torch.compile off (A/B), Omni single-stage TP2. Sequential; each step
# re-checks that its XPUs are idle and refuses to start otherwise.
cd "$(dirname "$0")/.."
R=results/20261122_rerun
CASES=t2i,i2t_1tok,i2t_64tok,i2i
scripts/run_one.sh $R mstar-tp2 mstar configs/mstar/bagel_tp2_2gpu.yaml 2 $CASES > $R/mstar-tp2.driver.log 2>&1
NO_COMPILE=1 scripts/run_one.sh $R mstar-tp2-nocompile mstar configs/mstar/bagel_tp2_2gpu.yaml 2 $CASES > $R/mstar-tp2-nocompile.driver.log 2>&1
scripts/run_one.sh $R omni-single-tp2 omni configs/vllm_omni/bagel_single_stage_tp2.yaml 2 $CASES > $R/omni-single-tp2.driver.log 2>&1
echo QUEUE1_DONE > $R/queue1.done
