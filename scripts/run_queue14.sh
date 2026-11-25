#!/usr/bin/env bash
# Omni main returns wrong I2T captions (pizza) with Thinker prefix caching off; rerun the 3-card two-stage TP2 pair with it on
# (M* default is cache on, so this is also the equal-cache comparison).
cd "$(dirname "$0")/.."
R=results/20261123_graph; P="Replace the bagel with a croissant."
O=configs/vllm_omni
scripts/run_one.sh $R omni-tp2-eager-cache omni $O/bagel_two_stage_3gpu_tp2_prefixcache.yaml 3 t2i,i2t_1tok,i2t_64tok,i2i --i2i-prompt "$P" > $R/omni-tp2-eager-cache.driver.log 2>&1
scripts/run_one.sh $R omni-tp2-graph-cache omni $O/bagel_two_stage_3gpu_tp2_graph_prefixcache.yaml 3 t2i,i2t_1tok,i2t_64tok,i2i --i2i-prompt "$P" > $R/omni-tp2-graph-cache.driver.log 2>&1
echo done > $R/queue14.done
