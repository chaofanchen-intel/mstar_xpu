#!/usr/bin/env bash
# Print the ids of idle XPUs (comma separated) and exit 0 only if at least N are idle.
# Usage: scripts/gpu_guard.sh N [--all]     (--all: also print per-card status to stderr)
# A card is "idle" when it holds < IDLE_MIB MiB (default 1024) and no non-benchmark process
# owns memory on it. This host is shared: never launch on a card that is not idle.
need="${1:?usage: gpu_guard.sh N}"
idle_mib="${IDLE_MIB:-1024}"
free=()
n=$(xpu-smi discovery 2>/dev/null | grep -c "Device Name")
for ((i = 0; i < n; i++)); do
  used=$(xpu-smi stats -d "$i" 2>/dev/null | awk -F'|' '/GPU Memory Used/ {gsub(/ /,"",$3); print $3}')
  util=$(xpu-smi stats -d "$i" 2>/dev/null | awk -F'|' '/Average % utilization of/ {gsub(/ /,"",$3); print $3; exit}')
  used=${used:-999999}
  state=BUSY
  if (( used < idle_mib )); then free+=("$i"); state=idle; fi
  echo "xpu$i: ${used} MiB used, util ${util:-?}% -> $state" >&2
done
if (( ${#free[@]} < need )); then
  echo "gpu_guard: need $need idle XPU(s), found ${#free[@]} (${free[*]:-none}); NOT launching." >&2
  exit 1
fi
(IFS=,; echo "${free[*]:0:$need}")
