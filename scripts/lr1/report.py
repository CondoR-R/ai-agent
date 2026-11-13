"""Сборка Markdown-отчёта ЛР1 из сохранённых артефактов.

Числа берутся только из JSON, которые построены скриптами отчётов
экспериментов из results.jsonl (Правила, п. 4.2, 8):

    python scripts/lr1/report.py
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

from ai_agent import files, utils

REPORTS_DIR = files.PROJECT_ROOT / "reports" / "lr1"
FIGURES_DIR = REPORTS_DIR / "figures"
DEFAULT_TEMPLATE = REPORTS_DIR / "report_template.md"
DEFAULT_OUTPUT = REPORTS_DIR / "report.md"
GIB = 2**30

MANUAL_RE = re.compile(
    r"<!-- manual:([\w-]+) -->(.*?)<!-- /manual:\1 -->", re.S
)
PLACEHOLDER_RE = re.compile(r"\{\{\s*([\w-]+)\s*\}\}")


def _load_json(name: str) -> dict:
    """Читает JSON из reports/lr1 или возвращает пустой словарь."""
    try:
        return files.read_json(REPORTS_DIR / name)
    except FileNotFoundError:
        return {}


def _table(headers: list[str], rows: list[list]) -> str:
    """Собирает Markdown-таблицу."""
    if not rows:
        return "_нет данных_"
    lines = ["| " + " | ".join(map(str, headers)) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value * 1000:.0f}"


def _num(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _report_meta() -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    commit = utils.get_commit_hash()[:7]
    return (
        f"_Отчёт сформирован автоматически {stamp}, коммит `{commit}`. "
        "Числа берутся из `reports/lr1/*.json`; блоки `manual` предназначены "
        "для ручного текста и сохраняются при повторной сборке._"
    )


def _figures(items: tuple[tuple[str, str], ...]) -> str:
    """Ссылки на готовые рисунки; отсутствующие пропускаются."""
    parts = []
    for name, label in items:
        if (FIGURES_DIR / name).exists():
            parts.append(f"![{label}](figures/{name})\n_{label}._")
    return (
        "\n\n".join(parts)
        or "_Графики не построены: запустите отчёт соответствующего эксперимента._"
    )


def _data_table() -> str:
    try:
        entry = files.read_json(files.MANIFEST_PATH)["gutenberg_1342"]
    except (FileNotFoundError, KeyError):
        return "_Нет данных манифеста._"

    info = entry["files"]["data/lr1/pride_and_prejudice.txt"]
    selection = entry["selection"]
    rows = [
        ["Источник", entry["source"]],
        ["URL", entry["url"]],
        ["Правило отбора", selection["rule"]],
        [
            "Итог отбора",
            f"символов: было {selection['chars_before']}, "
            f"осталось {selection['chars_after']}; исключено: "
            f"{selection['excluded']}",
        ],
        ["Записей в файле", info["n_records"]],
        ["sha256", f"`{info['sha256'][:16]}…`"],
    ]
    return _table(["Поле", "Значение"], rows)


def _passport_table() -> str:
    passport = _load_json("passport.json")
    if not passport:
        return (
            "_Паспорт не собран: запустите `scripts/lr1/passport.py` "
            "и скопируйте `passport.json` в `reports/lr1/`._"
        )

    server = passport.get("server", {})
    model = passport.get("model", {})
    device = passport.get("device", {})
    kv = passport.get("kv_cache", {})
    checks = passport.get("checks", {})
    gpu = device.get("gpu") or {}
    if gpu:
        device_text = (
            f"{gpu.get('name')}, {gpu.get('memory_mib', '—')} MiB; "
            f"CPU потоков: {device.get('cpu_count')}"
        )
    else:
        device_text = (
            f"{device.get('cpu_model') or 'CPU'}, "
            f"{_num(device.get('total_ram_gib'), 1)} ГиБ ОЗУ; "
            f"потоков: {device.get('cpu_count')}"
        )

    def check(name: str) -> str:
        value = checks.get(name)
        return "—" if value is None else ("да" if value else "НЕТ")

    context = server.get("max_model_len")
    slots = server.get("total_slots")
    per_slot = server.get("context_per_slot")
    context_text = (
        f"{context} ({per_slot} на слот × {slots})"
        if slots and slots > 1 and per_slot
        else context
    )

    rows = [
        ["Модель", f"{model.get('repo')} @ {str(model.get('revision'))[:12]}…"],
        ["Сервер", f"{server.get('name')} {server.get('version')}"],
        ["Команда запуска", f"`{server.get('launch_command')}`"],
        ["Предел контекста", context_text],
        ["Формат весов", model.get("weights_dtype")],
        [
            "KV-кеш",
            f"{kv.get('bytes_per_token')} байт/токен; "
            f"{kv.get('kv_cache_tokens')} токенов; "
            f"{_num(kv.get('available_kv_gib'), 2)} GiB; "
            f"тип {kv.get('kv_cache_dtype') or '—'}",
        ],
        ["Устройство", device_text],
        [
            "Проверки",
            f"KV-кеш: {check('kv_capacity')}; ровно токенов: "
            f"{check('exact_tokens')}; длина запроса: "
            f"{check('prompt_length')}; все запросы: "
            f"{check('requests_ok')}",
        ],
    ]
    return _table(["Поле", "Значение"], rows)


def _memory_table() -> str:
    exp2 = _load_json("experiment_2.json").get("config", {})
    decode = exp2.get("decode_memory") or {}
    weights_gib = decode.get("weights_gib")
    bandwidth = decode.get("memory_bandwidth_gb_s")
    description = decode.get("description")

    passport = _load_json("passport.json")
    kv = passport.get("kv_cache", {})
    kv_bytes = kv.get("bytes_per_token")
    context_gib = kv.get("context_gib")

    rows = []
    if weights_gib is not None:
        suffix = f" ({description})" if description else ""
        rows.append(["Память весов", f"{weights_gib:.2f} GiB{suffix}"])
    if weights_gib and bandwidth:
        lower = weights_gib * GIB / (bandwidth * 1e9)
        rows.append(
            [
                "Нижняя граница шага decode",
                f"{lower * 1000:.0f} мс при {bandwidth:.0f} ГБ/с "
                f"({1 / lower:.0f} ток/с)",
            ]
        )
    if kv_bytes is not None:
        rows.append(["KV-кеш на токен", f"{kv_bytes} байт"])
        rows.append(
            [
                "KV-кеш на 8 пользователей по 4000 токенов",
                f"{8 * 4000 * kv_bytes / GIB:.2f} GiB",
            ]
        )
    if context_gib is not None:
        rows.append(["KV-кеш на весь контекст", f"{context_gib:.2f} GiB"])
    return _table(["Величина", "Значение"], rows)


# ---------- эксперимент 1 ----------


def _exp1_length(exp1: dict) -> tuple[str, str]:
    item = exp1.get("ttft_vs_length") or {}
    metrics = item.get("metrics") or {}
    if not metrics.get("by_length"):
        return (
            "_Нет данных: запустите `report_exp1.py --length-run ...`._",
            "",
        )

    ecfg = item.get("config", {})
    rows = []
    for length, e in metrics["by_length"].items():
        if "ttft_median_s" not in e:
            rows.append(
                [length, "—", f"{e['n_ok']}/{e['n_planned']}", "—", "—", "—"]
            )
            continue
        rows.append(
            [
                length,
                f"{e['prompt_tokens_median']:.0f}",
                f"{e['n_ok']}/{e['n_planned']}",
                _ms(e["ttft_median_s"]),
                f"{_ms(e['ttft_min_s'])}–{_ms(e['ttft_max_s'])}",
                f"{e['prefill_tok_s_median']:.0f}",
            ]
        )
    table = _table(
        [
            "Длина (цель)",
            "Запрос по usage",
            "Успешно",
            "TTFT медиана, мс",
            "TTFT мин–макс, мс",
            "Prefill, ток/с",
        ],
        rows,
    )
    table = (
        f"**Таблица 1.** TTFT и скорость prefill по длине запроса. "
        f"Серия `{item.get('series', '—')}`, run_id `{item.get('run_id', '—')}`; "
        f"запросов на длину: {ecfg.get('repeats', '—')}, ответ "
        f"{ecfg.get('output_tokens', '—')} токенов.\n\n{table}"
    )

    parts = []
    fit = metrics.get("ttft_fit")
    if fit:
        parts.append(
            f"Прямая по медианам: TTFT = {round(fit['intercept_s'] * 1000)} мс"
            f" + {fit['slope_s_per_token'] * 1000:.4f} мс × токены; "
            f"скорость prefill ≈ {fit['prefill_tok_s']:.0f} ток/с."
        )
    scaling = metrics.get("scaling", {})
    if scaling:
        parts.append(
            "Рост TTFT между соседними длинами: "
            + "; ".join(
                f"{pair}: ×{s['ttft_ratio']:.2f} при росте длины "
                f"×{s['length_ratio']:.2f}"
                for pair, s in scaling.items()
            )
            + "."
        )
    if metrics.get("cache_hit_tokens_max") is not None:
        parts.append(
            "Максимум токенов из кеша префикса за запрос: "
            f"{metrics['cache_hit_tokens_max']:.0f}."
        )
    if metrics.get("n_wrong_output_len"):
        parts.append(
            "Запросов с длиной ответа не "
            f"{ecfg.get('output_tokens', '—')}: "
            f"{metrics['n_wrong_output_len']}."
        )
    return table, "\n\n".join(parts)


def _exp1_cache(exp1: dict) -> tuple[str, str]:
    item = exp1.get("prefix_cache") or {}
    metrics = item.get("metrics") or {}
    if not metrics.get("prompts"):
        return (
            "_Нет данных: запустите `report_exp1.py --cache-run ...`._",
            "",
        )

    rows = []
    for number, e in metrics["prompts"].items():
        ttft = e["ttft_s"] + [None] * (3 - len(e["ttft_s"]))
        rows.append(
            [
                number,
                f"{e['n_ok']}/{e['n_planned']}",
                _ms(ttft[0]),
                _ms(ttft[1]),
                _ms(ttft[2]),
                _num(e.get("speedup_2"), 2),
                _num(e.get("speedup_3"), 2),
            ]
        )
    table = _table(
        [
            "Запрос",
            "Успешно",
            "TTFT 1-й, мс",
            "TTFT 2-й, мс",
            "TTFT 3-й, мс",
            "Ускорение 1/2",
            "Ускорение 1/3",
        ],
        rows,
    )
    table = (
        f"**Таблица 2.** Один и тот же запрос повторён трижды при включённом "
        f"кешировании префикса. Серия `{item.get('series', '—')}`, "
        f"run_id `{item.get('run_id', '—')}`; "
        f"запрос {item.get('config', {}).get('prompt_tokens', '—')} токенов."
        f"\n\n{table}"
    )
    summary = (
        f"Медиана ускорения: второй запрос ×{_num(metrics.get('speedup_2_median'), 2)}, "
        f"третий ×{_num(metrics.get('speedup_3_median'), 2)}. "
        "Первый запрос проверен по счётчику попаданий: он должен быть холодным."
    )
    return table, summary


def _exp1_sections(exp1: dict) -> dict:
    length_table, length_summary = _exp1_length(exp1)
    cache_table, cache_summary = _exp1_cache(exp1)
    return {
        "exp1_length_table": length_table,
        "exp1_length_summary": length_summary,
        "exp1_length_plots": _figures(
            (
                ("exp1_ttft_vs_length.png", "TTFT от длины запроса"),
                ("exp1_prefill_speed.png", "Скорость prefill по длине"),
            )
        ),
        "exp1_cache_table": cache_table,
        "exp1_cache_summary": cache_summary,
        "exp1_cache_plots": _figures(
            (("exp1_prefix_cache.png", "TTFT повторных запросов"),)
        ),
    }


# ---------- эксперимент 2 ----------


def _exp2_sections(exp2: dict) -> dict:
    metrics = exp2.get("metrics") or {}
    if not metrics.get("by_output"):
        empty = "_Нет данных: запустите `report_exp2.py --run ...`._"
        return {
            "exp2_table": empty,
            "exp2_summary": empty,
            "exp2_plots": empty,
        }

    ecfg = exp2.get("config", {}).get("exp2", {})
    rows = []
    for length, e in metrics["by_output"].items():
        if "e2e_median_s" not in e:
            rows.append(
                [
                    length,
                    "—",
                    f"{e['n_ok']}/{e['n_planned']}",
                    "—",
                    "—",
                    "—",
                    "—",
                ]
            )
            continue
        rows.append(
            [
                length,
                f"{e['completion_tokens_median']:.0f}",
                f"{e['n_ok']}/{e['n_planned']}",
                _ms(e["e2e_median_s"]),
                _ms(e["ttft_median_s"]),
                _ms(e["tpot_median_s"]),
                f"{e['decode_tok_s_median']:.0f}"
                if e.get("decode_tok_s_median") is not None
                else "—",
            ]
        )
    table = _table(
        [
            "Длина ответа",
            "Ответ по usage",
            "Успешно",
            "e2e медиана, мс",
            "TTFT медиана, мс",
            "TPOT медиана, мс",
            "Decode, ток/с",
        ],
        rows,
    )
    table = (
        f"**Таблица 3.** Полное время ответа и скорость генерации по длине "
        f"ответа. Серия `{exp2.get('series', '—')}`, "
        f"run_id `{exp2.get('run_id', '—')}`; запросов на длину: "
        f"{ecfg.get('repeats', '—')}, запрос {ecfg.get('prompt_tokens', '—')} "
        f"токенов, ignore_eos.\n\n{table}"
    )

    parts = []
    fit = metrics.get("e2e_fit")
    if fit:
        parts.append(
            f"Прямая e2e = {_ms(fit['intercept_s'])} мс + "
            f"{_ms(fit['slope_s_per_token'])} мс × N; "
            f"свободный член отличается от медианы TTFT на "
            f"{_ms(fit['intercept_minus_ttft_s'])} мс, наклон — от медианы "
            f"TPOT на {_ms(fit['slope_minus_tpot_s'])} мс."
        )
    lower = metrics.get("decode_lower_bound_s")
    if lower is not None:
        parts.append(
            f"Нижняя граница TPOT по памяти весов: {_ms(lower)} мс; "
            f"медиана TPOT {_ms(metrics.get('tpot_median_s'))} мс, "
            f"превышение ×{_num(metrics.get('tpot_over_lower_bound'), 2)}."
        )
    if metrics.get("n_wrong_output_len"):
        parts.append(
            "Запросов, где usage.completion_tokens не совпал с целью: "
            f"{metrics['n_wrong_output_len']}."
        )
    return {
        "exp2_table": table,
        "exp2_summary": "\n\n".join(parts),
        "exp2_plots": _figures(
            (
                ("exp2_e2e_vs_output.png", "Полное время от длины ответа"),
                ("exp2_tpot_vs_output.png", "TPOT от длины ответа"),
            )
        ),
    }


# ---------- эксперимент 3 ----------


def _exp3_sections(exp3: dict) -> dict:
    metrics = exp3.get("metrics") or {}
    if not metrics.get("by_clients"):
        empty = "_Нет данных: запустите `report_exp3.py --run ...`._"
        return {
            "exp3_table": empty,
            "exp3_summary": empty,
            "exp3_plots": empty,
        }

    ecfg = (exp3.get("config") or {}).get("exp3") or {}
    prompt_tokens = ecfg.get("prompt_tokens", "—")
    output_tokens = ecfg.get("output_tokens", "—")
    rows = []
    for clients, e in metrics["by_clients"].items():
        rows.append(
            [
                clients,
                f"{e['n_ok']}/{e['n_planned']}",
                _num(e.get("throughput_tok_s"), 0),
                _num(e.get("requests_per_s"), 2),
                _ms(e.get("ttft_median_s")),
                _ms(e.get("ttft_p90_s")),
                _ms(e.get("tpot_median_s")),
                _num(e.get("goodput_rps"), 2),
                "да" if e["limits_ok"] else "НЕТ",
                _num(e.get("queue_waiting_after"), 1),
                _num(e.get("kv_cache_usage_after"), 2),
            ]
        )
    table = _table(
        [
            "Клиентов",
            "Успешно",
            "Пропускная, ток/с",
            "Запросов/с",
            "TTFT медиана, мс",
            "TTFT p90, мс",
            "TPOT медиана, мс",
            "Goodput, зап/с",
            "Пределы",
            "Очередь после",
            "KV после",
        ],
        rows,
    )
    table = (
        f"**Таблица 4.** Уровни нагрузки. Серия `{exp3.get('series', '—')}`, "
        f"run_id `{exp3.get('run_id', '—')}`; запросов на уровень "
        f"max({ecfg.get('min_requests', '—')}, "
        f"{ecfg.get('requests_per_client', '—')}c); пределы TTFT p90 и "
        f"медианы TPOT — в конфигурации.\n\n{table}"
    )

    max_load = metrics.get("max_clients_within_limits")
    capacity = metrics.get("kv_requests_capacity")
    summary = (
        "Наибольшая нагрузка в пределах: "
        + ("—" if max_load is None else f"{max_load} клиентов")
        + ". Ёмкость KV-кеша: "
        + (
            "—"
            if metrics.get("kv_cache_tokens") is None
            else f"{metrics['kv_cache_tokens']} токенов"
        )
        + (
            ""
            if capacity is None
            else f"; запросов по {prompt_tokens} + {output_tokens} токенов "
            f"помещается ≈ {capacity}"
        )
        + f". Ошибок вызова: {metrics.get('n_error')} из "
        f"{metrics.get('n_planned')}."
    )
    return {
        "exp3_table": table,
        "exp3_summary": summary,
        "exp3_plots": _figures(
            (
                (
                    "exp3_throughput_latency.png",
                    "Пропускная способность — задержка",
                ),
                (
                    "exp3_throughput_clients.png",
                    "Пропускная способность по клиентам",
                ),
            )
        ),
    }


# ---------- сборка ----------


def _merge_manual(template_text: str, previous_text: str) -> str:
    """Переносит ручные блоки из предыдущего отчёта в новый шаблон."""
    if not previous_text:
        return template_text

    old_blocks = {
        match.group(1): match.group(2)
        for match in MANUAL_RE.finditer(previous_text)
    }

    def replace(match: re.Match) -> str:
        key = match.group(1)
        content = old_blocks.get(key, match.group(2))
        return f"<!-- manual:{key} -->{content}<!-- /manual:{key} -->"

    return MANUAL_RE.sub(replace, template_text)


def _render(template_text: str, values: dict) -> str:
    """Подставляет значения вместо {{ placeholder }}."""

    def replace(match: re.Match) -> str:
        return str(values.get(match.group(1), match.group(0)))

    return PLACEHOLDER_RE.sub(replace, template_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    template_path = Path(args.template)
    output_path = Path(args.output)
    if not template_path.exists():
        raise SystemExit(f"Шаблон не найден: {template_path}")

    exp1 = _load_json("experiment_1.json")
    exp2 = _load_json("experiment_2.json")
    exp3 = _load_json("experiment_3.json")

    values = {
        "report_meta": _report_meta(),
        "passport_table": _passport_table(),
        "data_table": _data_table(),
        "memory_table": _memory_table(),
    }
    values.update(_exp1_sections(exp1))
    values.update(_exp2_sections(exp2))
    values.update(_exp3_sections(exp3))

    template_text = template_path.read_text(encoding="utf-8")
    if output_path.exists():
        previous = output_path.read_text(encoding="utf-8")
        template_text = _merge_manual(template_text, previous)

    report_text = _render(template_text, values)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")

    leftovers = sorted(set(PLACEHOLDER_RE.findall(report_text)))
    print(f"[REPORT] отчёт: {output_path}", flush=True)
    if leftovers:
        print(f"[REPORT] не заполнено: {leftovers}", flush=True)


if __name__ == "__main__":
    main()
