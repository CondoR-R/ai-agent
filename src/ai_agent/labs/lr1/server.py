"""Сведения о сервере модели: паспорт, KV-кеш, /metrics (ЛР1)"""

import json
import os
import platform
import re
import subprocess
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .endpoint import Endpoint

GIB = 2**30

# метрики сервера, которые снимаются до и после уровня нагрузки (опыт 3).
# vLLM и llama.cpp называют их по-разному, поэтому набор общий.
DEFAULT_METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "llamacpp:requests_processing",
    "llamacpp:requests_deferred",
    "llamacpp:prompt_tokens_total",
    "llamacpp:prompt_tokens_cached_total",
    "llamacpp:tokens_predicted_total",
    "llamacpp:prompt_seconds_total",
    "llamacpp:tokens_predicted_seconds_total",
)

# счётчики кеша префикса (токены); имена зависят от версии сервера
PREFIX_CACHE_METRICS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:gpu_prefix_cache_queries_total",
    "vllm:gpu_prefix_cache_hits_total",
    "llamacpp:prompt_tokens_cached_total",
)
EXP1_METRICS = DEFAULT_METRICS + PREFIX_CACHE_METRICS

_KV_TOKENS = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
_CONCURRENCY = re.compile(
    r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x"
)
_AVAILABLE = re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB")
_DTYPE = re.compile(r"dtype=torch\.(\w+)")
_PREFIX_CACHING = re.compile(r"enable_prefix_caching=(True|False)")
_LLAMA_KV = re.compile(
    r"llama_kv_cache: size =\s*(?P<size>[\d.]+)\s*MiB\s*\("
    r"\s*(?P<cells>[\d,]+)\s*cells,\s*\d+\s*layers,"
    r"\s*\d+/(?P<seqs>\d+)\s*seqs\).*?"
    r"K \((?P<kdtype>[\w]+)\):\s*(?P<k>[\d.]+)\s*MiB,"
    r"\s*V \((?P<vdtype>[\w]+)\):\s*(?P<v>[\d.]+)\s*MiB"
)
_LLAMA_CTX = re.compile(r"llama_context: n_ctx\s*=\s*([\d,]+)")
_LLAMA_FTYPE = re.compile(r"print_info: file type\s*=\s*(.+)")
_METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>\S+)",
    re.MULTILINE,
)


# ---------- KV-кеш ----------


def kv_bytes_per_token(
    n_layers: int, n_kv_heads: int, head_dim: int, dtype_bytes: int
) -> int:
    """Байт KV-кеша на один токен: 2 * слои * головы KV * размер головы * b.

    Двойка — ключи и значения. Формула верна для моделей, у которых кеш
    есть в каждом слое; у гибридных моделей проверьте конфигурацию.
    """
    return 2 * n_layers * n_kv_heads * head_dim * dtype_bytes


def kv_config_from_hf(repo: str, revision: str | None = None) -> dict:
    """Параметры KV-кеша из config.json модели.

    raise:
        ValueError для гибридных моделей (не только полное внимание)
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(repo, revision=revision)
    layer_types = set(getattr(cfg, "layer_types", None) or [])
    if layer_types - {"full_attention"}:
        raise ValueError(
            f"У {repo} слои разных типов {sorted(layer_types)}: "
            "размер KV-кеша посчитайте вручную"
        )
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads
    )
    return {
        "n_layers": cfg.num_hidden_layers,
        "n_kv_heads": getattr(
            cfg, "num_key_value_heads", cfg.num_attention_heads
        ),
        "head_dim": head_dim,
    }


def check_kv_capacity(
    kv_cache_tokens: int,
    bytes_per_token: int,
    available_gib: float,
    rel_tol: float = 0.05,
) -> dict:
    """Ёмкость кеша из журнала * байт на токен ≈ Available KV cache memory.

    Расхождение больше rel_tol значит другую модель или другой тип
    данных кеша (--kv-cache-dtype): разберитесь, почему.
    """
    computed = kv_cache_tokens * bytes_per_token / GIB
    ratio = computed / available_gib
    return {
        "computed_gib": computed,
        "available_gib": available_gib,
        "ratio": ratio,
        "ok": abs(ratio - 1.0) <= rel_tol,
    }


# ---------- журнал сервера ----------


def _last_match(pattern: re.Pattern, text: str):
    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


def _to_int(value: str) -> int:
    return int(value.replace(",", ""))


def parse_vllm_log(text: str) -> dict:
    """Достаёт из журнала запуска vLLM ёмкость кеша и dtype весов.

    Берётся последнее вхождение: в журнале могут быть прошлые запуски.
    Что не нашлось, становится None (формат строк зависит от версии).
    """
    tokens = _last_match(_KV_TOKENS, text)
    concurrency = _last_match(_CONCURRENCY, text)
    available = _last_match(_AVAILABLE, text)
    dtype = _last_match(_DTYPE, text)
    prefix = _last_match(_PREFIX_CACHING, text)
    return {
        "kv_cache_tokens": _to_int(tokens.group(1)) if tokens else None,
        "max_concurrency_tokens_per_request": (
            _to_int(concurrency.group(1)) if concurrency else None
        ),
        "max_concurrency_x": (
            float(concurrency.group(2)) if concurrency else None
        ),
        "available_kv_gib": float(available.group(1)) if available else None,
        "weights_dtype": dtype.group(1) if dtype else None,
        "kv_cache_dtype": dtype.group(1) if dtype else None,
        "prefix_caching": (prefix.group(1) == "True" if prefix else None),
    }


def parse_llama_log(text: str) -> dict:
    """Достаёт из журнала llama.cpp ёмкость KV-кеша и типы данных.

    Строка загрузки выглядит так (llama.cpp, -lv 4+):
    ``llama_kv_cache: size = 896.00 MiB (32768 cells, 28 layers, 1/1 seqs),
    K (f16): 448.00 MiB, V (f16): 448.00 MiB``.
    """
    kv = _last_match(_LLAMA_KV, text)
    ctx = _last_match(_LLAMA_CTX, text)
    ftype = _last_match(_LLAMA_FTYPE, text)
    # cells — это ячейки на одну последовательность; при -np N общая
    # ёмкость равна cells * N, что и видно в конце строки (N/N seqs).
    tokens = (
        _to_int(kv.group("cells")) * _to_int(kv.group("seqs"))
        if kv
        else (_to_int(ctx.group(1)) if ctx else None)
    )
    return {
        "kv_cache_tokens": tokens,
        "available_kv_gib": float(kv.group("size")) / 1024 if kv else None,
        "weights_dtype": ftype.group(1).strip() if ftype else None,
        "kv_cache_dtype": kv.group("kdtype") if kv else None,
        "kv_cache_k_gib": float(kv.group("k")) / 1024 if kv else None,
        "kv_cache_v_gib": float(kv.group("v")) / 1024 if kv else None,
    }


def parse_server_log(text: str) -> dict:
    """Разбирает журнал vLLM или llama.cpp: что нашлось, то и вернёт."""
    info = parse_llama_log(text)
    vllm = parse_vllm_log(text)
    info.update(
        {key: value for key, value in vllm.items() if value is not None}
    )
    return info


def prefix_caching_state(
    log_info: dict, launch_command: str | None
) -> tuple[bool, str]:
    """Включено ли кеширование префикса и откуда это известно.

    Порядок источников: журнал сервера, флаги команды запуска, умолчание
    сервера (у vLLM V1 и llama.cpp кеш промпта по умолчанию включён).

    return: (включено, источник)
    """
    value = log_info.get("prefix_caching")
    if value is not None:
        return value, "журнал сервера"
    command = launch_command or ""
    if (
        "--no-enable-prefix-caching" in command
        or "--no-cache-prompt" in command
    ):
        return False, "команда запуска"
    if "--enable-prefix-caching" in command or "--cache-prompt" in command:
        return True, "команда запуска"
    return True, "умолчание сервера (флага нет в команде запуска)"


# ---------- HTTP к серверу ----------


def _get(url: str, api_key: str | None = None, timeout: float = 30.0) -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def fetch_props(endpoint: Endpoint) -> dict:
    """Сведения сервера из /props (llama.cpp); пустой словарь, если нет."""
    try:
        payload = json.loads(_get(f"{endpoint.origin}/props", endpoint.api_key))
    except (urllib.error.URLError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def fetch_models(endpoint: Endpoint) -> dict:
    """Карточка модели из /v1/models.

    У vLLM там есть max_model_len, у llama.cpp — meta.n_ctx и meta.ftype.

    raise:
        ValueError, если сервер не отдаёт модель из endpoint.yaml
    """
    try:
        payload = json.loads(
            _get(f"{endpoint.base_url.rstrip('/')}/models", endpoint.api_key)
        )
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Сервер {endpoint.base_url} не отвечает ({exc.reason}). "
            "Он запущен? Смотрите scripts/lr1/start_llama.sh и журнал "
            "/tmp/lr1-llama/llama.log"
        ) from exc
    models = payload.get("data", [])
    for model in models:
        if model.get("id") == endpoint.model:
            return model
    raise ValueError(
        f"Сервер не отдаёт модель {endpoint.model}; "
        f"доступны: {[m.get('id') for m in models]}"
    )


def parse_metrics(text: str, names: tuple[str, ...] | None = None) -> dict:
    """Разбирает текст /metrics (формат Prometheus).

    Значения одной метрики с разными метками складываются.
    """
    result: dict[str, float] = {}
    for match in _METRIC_LINE.finditer(text):
        name = match.group("name")
        if names is not None and name not in names:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        result[name] = result.get(name, 0.0) + value
    return result


def fetch_metrics(
    endpoint: Endpoint, names: tuple[str, ...] = DEFAULT_METRICS
) -> dict:
    """Снимок метрик сервера с адреса /metrics (очередь, заполнение KV)."""
    return parse_metrics(_get(f"{endpoint.origin}/metrics"), names)


# ---------- окружение ----------


def installed_version(package: str) -> str | None:
    """Версия пакета в этом окружении (None, если не установлен)."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _command_version(command: str) -> str | None:
    """Первая строка ``<command> --version`` (для внешнего сервера)."""
    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    out = (result.stdout + "\n" + result.stderr).strip()
    line = out.splitlines()[0].strip() if out else ""
    if line.lower().startswith("version:"):
        line = line.split(":", 1)[1].strip()
    return line or None


def server_version(name: str) -> str | None:
    """Версия сервера: пакет Python или внешняя команда llama-server."""
    package_version = installed_version(name)
    if package_version is not None:
        return package_version
    if "llama" in name.lower():
        return _command_version("llama-server")
    return None


def hf_cached_revision(repo: str) -> str | None:
    """Коммит модели в локальном кеше Hugging Face (что реально скачано)."""
    try:
        from huggingface_hub import constants
    except ImportError:
        return None
    ref = (
        Path(constants.HF_HUB_CACHE)
        / f"models--{repo.replace('/', '--')}"
        / "refs"
        / "main"
    )
    return ref.read_text().strip() if ref.exists() else None


def gpu_info() -> dict | None:
    """Название, память и драйвер GPU (None, если nvidia-smi недоступен)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    name, memory_mib, driver = [
        p.strip() for p in out.splitlines()[0].split(",")
    ]
    return {
        "name": name,
        "memory_mib": int(float(memory_mib)),
        "driver": driver,
    }


def _snapshot_revision(model_path: str | None) -> str | None:
    """Коммит GGUF-файла из пути кеша Hugging Face (.../snapshots/<hash>/)."""
    if not model_path:
        return None
    parts = Path(model_path).parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def _cpu_model() -> str | None:
    """Название процессора (на macOS — через sysctl)."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return out.strip() or None
    return platform.processor() or None


def _total_ram_gib() -> float | None:
    """Объём оперативной памяти, ГиБ (None, если платформа не позволяет)."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None
    return pages * page_size / GIB


def _read_or_none(path: str | None) -> str | None:
    if not path or not Path(path).exists():
        return None
    return Path(path).read_text(encoding="utf-8", errors="replace")


def collect_server_info(endpoint: Endpoint, server_cfg: dict) -> dict:
    """Паспорт сервера для metadata.json (Правила, п. 3).

    params:
        endpoint - параметры сервера
        server_cfg - секция server конфигурации: name, max_model_len,
                     log_path, cmd_path

    return: {"server", "model", "device", "kv_cache"}

    raise:
        ValueError, если предел контекста сервера не совпал с ожидаемым
    """
    card = fetch_models(endpoint)
    props = fetch_props(endpoint)
    meta = card.get("meta") or {}
    served_len = card.get("max_model_len") or meta.get("n_ctx")
    settings = props.get("default_generation_settings") or {}
    served_len = served_len or settings.get("n_ctx")
    if served_len is not None:
        served_len = int(served_len)
    context_per_slot = served_len
    slots = int(props.get("total_slots") or 1)
    if "llama" in str(server_cfg["name"]).lower() and slots > 1:
        served_len = context_per_slot * slots

    expected_len = server_cfg.get("max_model_len")
    if (
        expected_len is not None
        and served_len is not None
        and served_len != expected_len
    ):
        raise ValueError(
            f"Предел контекста сервера {served_len} не равен ожидаемому "
            f"{expected_len}"
        )

    log_text = _read_or_none(server_cfg.get("log_path"))
    cmd_text = _read_or_none(server_cfg.get("cmd_path"))
    log_info = parse_server_log(log_text) if log_text else {}
    weights_dtype = (
        log_info.get("weights_dtype")
        or meta.get("ftype")
        or props.get("model_ftype")
    )
    model_path = props.get("model_path")
    if log_info.get("kv_cache_tokens") is None and served_len is not None:
        log_info["kv_cache_tokens"] = served_len

    return {
        "server": {
            "name": server_cfg["name"],
            "version": server_version(server_cfg["name"]),
            "build_info": props.get("build_info"),
            "launch_command": cmd_text.strip() if cmd_text else None,
            "max_model_len": served_len,
            "context_per_slot": context_per_slot,
            "total_slots": slots,
            "log_path": server_cfg.get("log_path"),
        },
        "model": {
            "id": card["id"],
            "repo": endpoint.model,
            "revision": hf_cached_revision(endpoint.model),
            "weights_dtype": weights_dtype,
            "weights_file": Path(model_path).name if model_path else None,
            "weights_revision": _snapshot_revision(model_path),
            "n_params": meta.get("n_params"),
            "weights_size_bytes": meta.get("size"),
        },
        "device": {
            "gpu": gpu_info(),
            "cpu_count": os.cpu_count(),
            "cpu_model": _cpu_model(),
            "system": platform.system(),
            "machine": platform.machine(),
            "total_ram_gib": _total_ram_gib(),
        },
        "kv_cache": log_info,
    }
