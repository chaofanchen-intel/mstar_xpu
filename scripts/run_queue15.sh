#!/usr/bin/env bash
# Omni I2T must use the chat prompt style (+ ignore_eos) for the Thinker to see the image; queue12/14 omitted it (pizza captions).
# Rerun I2T for the 3-card two-stage TP2 pair, cache off and on.
cd "$(dirname "$0")/.."
R=results/20261123_graph; O=configs/vllm_omni
F="--omni-ignore-eos --omni-prompt-style chat"
r() { local label=$1 cfg=$2; scripts/run_one.sh $R "$label" omni $O/$cfg 3 i2t_1tok,i2t_64tok $F > $R/$label.driver.log 2>&1; }
r omni-tp2-eager-chat        bagel_two_stage_3gpu_tp2.yaml
r omni-tp2-graph-chat        bagel_two_stage_3gpu_tp2_graph.yaml
r omni-tp2-eager-chat-cache  bagel_two_stage_3gpu_tp2_prefixcache.yaml
r omni-tp2-graph-chat-cache  bagel_two_stage_3gpu_tp2_graph_prefixcache.yaml
echo done > $R/queue15.done
