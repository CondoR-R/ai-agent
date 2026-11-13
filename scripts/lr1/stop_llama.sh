#!/usr/bin/env bash
# Останавливает llama-server, запущенный start_llama.sh.
set -uo pipefail

LR1_DIR="${LR1_DIR:-/tmp/lr1-llama}"
PORT="${PORT:-8000}"
PID_FILE="$LR1_DIR/llama.pid"

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  kill "$PID" 2> /dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$PID" 2> /dev/null || break
    sleep 1
  done
  kill -9 "$PID" 2> /dev/null || true
  rm -f "$PID_FILE"
fi

# на случай, если pid-файл потерян
pkill -TERM -f "llama-server" 2> /dev/null || true
sleep 1
pkill -KILL -f "llama-server" 2> /dev/null || true

if curl -sf "127.0.0.1:$PORT/v1/models" > /dev/null; then
  echo "[llama.cpp] ОШИБКА: порт $PORT всё ещё отвечает" >&2
  exit 1
fi

echo "[llama.cpp] остановлен, порт $PORT свободен"
