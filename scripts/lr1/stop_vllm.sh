#!/usr/bin/env bash
# Останавливает vLLM, запущенный start_vllm.sh, и ждёт освобождения GPU.
# Процесс EngineCore отдельный: он держит память GPU и может пережить
# родителя, поэтому завершаем и его. Если память не освободилась —
# выходим с ошибкой, чтобы следующий запуск не упал в неё же.
set -uo pipefail

VLLM_DIR="${VLLM_DIR:-/content/vllm}"
PID_FILE="$VLLM_DIR/vllm.pid"
LIMIT_MIB=1000   # столько GPU занимает и «пустой» Colab

gpu_used() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    2> /dev/null | head -n 1 | tr -d ' '
}

kill_vllm() {
  # $1 — сигнал; шаблоны не совпадают с именем этого скрипта
  pkill "$1" -f "vllm serve" 2> /dev/null || true
  pkill "$1" -f "VLLM::" 2> /dev/null || true
}

if [[ -f "$PID_FILE" ]]; then
  kill "$(cat "$PID_FILE")" 2> /dev/null || true
  rm -f "$PID_FILE"
fi
kill_vllm -TERM

used=""
for _ in $(seq 1 30); do
  used="$(gpu_used)"
  [[ -z "$used" || "$used" -lt "$LIMIT_MIB" ]] && break
  sleep 2
done

if [[ -n "$used" && "$used" -ge "$LIMIT_MIB" ]]; then
  echo "[vLLM] мягкая остановка не освободила GPU (${used} MiB), добиваю" >&2
  kill_vllm -KILL
  sleep 5
  used="$(gpu_used)"
fi

if [[ -n "$used" && "$used" -ge "$LIMIT_MIB" ]]; then
  echo "[vLLM] ОШИБКА: GPU занят (${used} MiB). Процессы:" >&2
  ps aux | grep -i -E "vllm|EngineCore|python" | grep -v grep >&2 || true
  echo "Если процесс не находится: Runtime -> Restart session" \
    "(файлы в /content при этом сохраняются)." >&2
  exit 1
fi

echo "[vLLM] остановлен, память GPU: ${used:-н/д} MiB"
