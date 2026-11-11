#!/usr/bin/env bash
# Запуск vLLM для ЛР1 (профиль GPU: Qwen3-4B-Instruct-2507, FP16, 32 768).
# Дополнительные флаги передаются аргументами, например для опыта 1:
#   bash scripts/lr1/start_vllm.sh --no-enable-prefix-caching
# Журнал и команда лежат вне репозитория ($VLLM_DIR), чтобы не пачкать
# рабочее дерево git: итоговые запуски требуют чистого дерева.
set -euo pipefail

VLLM_DIR="${VLLM_DIR:-/content/vllm}"
PORT="${PORT:-8000}"
mkdir -p "$VLLM_DIR"

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
: > "$VLLM_DIR/vllm.log"
nohup "${CMD[@]}" > "$VLLM_DIR/vllm.log" 2>&1 &
echo $! > "$VLLM_DIR/vllm.pid"
echo "[vLLM] pid=$(cat "$VLLM_DIR/vllm.pid"), журнал: $VLLM_DIR/vllm.log"

# ждём готовности: первая загрузка весов занимает несколько минут
for _ in $(seq 1 180); do
  if curl -sf "localhost:$PORT/v1/models" > /dev/null; then
    echo "[vLLM] сервер отвечает"
    grep -E "GPU KV cache size|Maximum concurrency|Available KV cache" \
      "$VLLM_DIR/vllm.log" || true
    exit 0
  fi
  if ! kill -0 "$(cat "$VLLM_DIR/vllm.pid")" 2> /dev/null; then
    echo "[vLLM] процесс завершился, конец журнала:" >&2
    tail -n 40 "$VLLM_DIR/vllm.log" >&2
    exit 1
  fi
  sleep 5
done

echo "[vLLM] не дождались ответа за 15 минут" >&2
tail -n 40 "$VLLM_DIR/vllm.log" >&2
exit 1
