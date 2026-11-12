#!/usr/bin/env bash
# Запуск vLLM для ЛР1 (профиль GPU: Qwen3-4B-Instruct-2507, FP16, 32 768).
# Дополнительные флаги передаются аргументами, например для опыта 1:
#   bash scripts/lr1/start_vllm.sh --no-enable-prefix-caching
# Журнал и команда лежат вне репозитория ($VLLM_DIR), чтобы не пачкать
# рабочее дерево git: итоговые запуски требуют чистого дерева.
set -euo pipefail

VLLM_DIR="${VLLM_DIR:-/content/vllm}"
PORT="${PORT:-8000}"
LOG="$VLLM_DIR/vllm.log"
LIMIT_MIB=1000
mkdir -p "$VLLM_DIR"

# в новой сессии Colab пакеты стираются: без vllm сервер не запустится
if ! command -v vllm > /dev/null; then
  echo "[vLLM] команда vllm не найдена: в этой сессии vLLM не установлен." \
    "Выполните: pip install vllm" >&2
  exit 1
fi

# vLLM берёт под себя ~90% памяти GPU: на занятом он упадёт с невнятной
# ошибкой инициализации, поэтому проверяем заранее
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
  2> /dev/null | head -n 1 | tr -d ' ' || true)"
if [[ -n "$used" && "$used" -ge "$LIMIT_MIB" ]]; then
  echo "[vLLM] GPU уже занят (${used} MiB): сначала" \
    "bash scripts/lr1/stop_vllm.sh" >&2
  exit 1
fi

CMD=(
  vllm serve Qwen/Qwen3-4B-Instruct-2507
  --dtype half
  --max-model-len 32768
  --enable-auto-tool-choice --tool-call-parser hermes
  --port "$PORT"
  "$@"
)

# команда со всеми флагами идёт в паспорт сервера
printf '%q ' "${CMD[@]}" > "$VLLM_DIR/vllm.cmd"
# журнал прошлого запуска не затираем: в нём может быть причина сбоя
[[ -s "$LOG" ]] && mv "$LOG" "$VLLM_DIR/vllm.prev.log"
nohup "${CMD[@]}" > "$LOG" 2>&1 &
echo $! > "$VLLM_DIR/vllm.pid"
echo "[vLLM] pid=$(cat "$VLLM_DIR/vllm.pid"), журнал: $LOG"

fail() {
  echo "[vLLM] $1" >&2
  echo "--- строки с ошибками (причина обычно выше трейсбека):" >&2
  grep -n -m 15 -E "ERROR|Traceback|OutOfMemory|out of memory|ValueError|AssertionError|RuntimeError" \
    "$LOG" >&2 || true
  echo "--- конец журнала:" >&2
  tail -n 30 "$LOG" >&2
  exit 1
}

# ждём готовности: первая загрузка весов занимает несколько минут
for _ in $(seq 1 180); do
  if curl -sf "localhost:$PORT/v1/models" > /dev/null; then
    echo "[vLLM] сервер отвечает"
    grep -E "GPU KV cache size|Maximum concurrency|Available KV cache" \
      "$LOG" || true
    exit 0
  fi
  if ! kill -0 "$(cat "$VLLM_DIR/vllm.pid")" 2> /dev/null; then
    fail "процесс завершился"
  fi
  sleep 5
done

fail "не дождались ответа за 15 минут"
