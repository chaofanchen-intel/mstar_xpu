#!/usr/bin/env bash
# Stop a server started by serve_*.sh and wait until its XPU memory is released.
# Usage: scripts/stop_server.sh <run_dir>
run_dir="${1:?run_dir}"
pid=$(cat "$run_dir/server.pid" 2>/dev/null) || { echo "no pid file"; exit 0; }
pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
if [[ -n "$pgid" ]]; then
  kill -TERM -- "-$pgid" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$pgid" 2>/dev/null || true
fi
sleep 3
echo "stopped $pid"
