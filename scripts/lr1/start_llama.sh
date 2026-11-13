#!/usr/bin/env bash
# Запуск llama.cpp для ЛР1 (профиль CPU: Qwen2.5-1.5B-Instruct, Q4_K_M).
# Дополнительные флаги передаются аргументами, например для опыта 1:
#   bash scripts/lr1/start_llama.sh --no-cache-prompt
# Для опыта 3 нужен больший общий контекст и число слотов:
#   CTX=65536 PARALLEL=32 bash scripts/lr1/start_llama.sh
# Журнал и команда лежат вне репозитория ($LR1_DIR), чтобы не пачкать
# рабочее дерево git: итоговые запуски требуют чистого дерева.
set -euo pipefail

LR1_DIR="${LR1_DIR:-/tmp/lr1-llama}"
PORT="${PORT:-8000}"
CTX="${CTX:-32768}"
PARALLEL="${PARALLEL:-1}"
THREADS="${THREADS:-6}"
LOG_LEVEL="${LOG_LEVEL:-5}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M}"
ALIAS="${ALIAS:-Qwen/Qwen2.5-1.5B-Instruct}"
LOG="$LR1_DIR/llama.log"
CMD_FILE="$LR1_DIR/llama.cmd"
PID_FILE="$LR1_DIR/llama.pid"
mkdir -p "$LR1_DIR"

if ! command -v llama-server > /dev/null; then
  echo "[llama.cpp] команда llama-server не найдена." \
    "Установите: brew install llama.cpp" >&2
  exit 1
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2> /dev/null; then
  echo "[llama.cpp] уже запущен pid=$(cat "$PID_FILE"):" \
    "сначала bash scripts/lr1/stop_llama.sh" >&2
  exit 1
fi

CMD=(
  llama-server
  -hf "$MODEL"
  --alias "$ALIAS"
  -c "$CTX"
  -np "$PARALLEL"
  -t "$THREADS"
  -tb "$THREADS"
  --host 127.0.0.1
  --port "$PORT"
  --metrics
  --no-webui
  -lv "$LOG_LEVEL"
  --log-file "$LOG"
  "$@"
)

# команда со всеми флагами идёт в паспорт сервера
printf '%q ' "${CMD[@]}" > "$CMD_FILE"
# журнал прошлого запуска не затираем: в нём может быть причина сбоя
[[ -s "$LOG" ]] && mv "$LOG" "$LR1_DIR/llama.prev.log"
nohup "${CMD[@]}" > "$LR1_DIR/llama.stdout" 2>&1 &
echo $! > "$PID_FILE"
echo "[llama.cpp] pid=$(cat "$PID_FILE"), журнал: $LOG"

fail() {
  echo "[llama.cpp] $1" >&2
  echo "--- строки с ошибками (причина обычно выше трейсбека):" >&2
  grep -n -m 15 -E "error|ERROR|Traceback|failed|out of memory|assert" \
    "$LOG" >&2 || true
  echo "--- конец журнала:" >&2
  tail -n 30 "$LOG" >&2
  exit 1
}

# первая загрузка модели из Hugging Face занимает несколько минут
for _ in $(seq 1 360); do
  if curl -sf "127.0.0.1:$PORT/v1/models" > /dev/null; then
    echo "[llama.cpp] сервер отвечает"
    grep -E "llama_kv_cache: size|n_ctx_slot|prompt cache" "$LOG" || true
    exit 0
  fi
  if ! kill -0 "$(cat "$PID_FILE")" 2> /dev/null; then
    fail "процесс завершился"
  fi
  sleep 5
done

fail "не дождались ответа за 30 минут"
