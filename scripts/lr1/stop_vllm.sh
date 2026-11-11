#!/usr/bin/env bash
# Останавливает сервер, запущенный start_vllm.sh, и ждёт освобождения GPU.
set -euo pipefail

VLLM_DIR="${VLLM_DIR:-/content/vllm}"
PID_FILE="$VLLM_DIR/vllm.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "[vLLM] нет $PID_FILE: сервер не запускался этим скриптом"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2> /dev/null; then
  kill "$PID"
  for _ in $(seq 1 30); do
    kill -0 "$PID" 2> /dev/null || break
    sleep 2
  done
  # vLLM запускает дочерние процессы: добиваем, что осталось
  pkill -f "vllm serve" 2> /dev/null || true
fi
rm -f "$PID_FILE"
echo "[vLLM] остановлен"
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2> /dev/null || true
