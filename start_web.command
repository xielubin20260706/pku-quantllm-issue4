#!/bin/zsh
cd "$(dirname "$0")"
data_source="feathers"
if [[ ! -f "$data_source/close.feather" ]]; then
  data_source="data"
fi
python3 webapp.py --data "$data_source" &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null' EXIT INT TERM
for attempt in {1..20}; do
  if curl -fsS http://127.0.0.1:8765/ >/dev/null; then
    open http://127.0.0.1:8765
    break
  fi
  sleep 0.5
done
wait "$server_pid"
