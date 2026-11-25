#!/usr/bin/env bash
# Wait until GET <url><path> returns 200, or the server process dies / timeout.
# Usage: scripts/wait_ready.sh <run_dir> <port> <path> [timeout_s]
run_dir="${1:?}"; port="${2:?}"; path="${3:?}"; timeout="${4:-1500}"
pid=$(cat "$run_dir/server.pid")
for ((t = 0; t < timeout; t += 5)); do
  if ! kill -0 "$pid" 2>/dev/null; then echo "server exited early; see $run_dir/server.log" >&2; exit 2; fi
  if [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port$path")" == 200 ]]; then
    echo "ready after ~${t}s"; exit 0
  fi
  sleep 5
done
echo "timeout waiting for server" >&2; exit 3
