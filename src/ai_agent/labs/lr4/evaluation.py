"""Метрики эпизодов ЛР4."""

from collections import Counter, defaultdict
from statistics import median

MODES = ("react", "plan")


def _records_by_task(records: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in records:
        grouped[record["task_id"]][record["mode"]] = record
    return grouped


def group_of(record: dict) -> str:
    meta = record["task_meta"]
    calls = meta["n_ground_truth_calls"]
    if calls == 1:
        size = "single"
    elif calls == 2:
        size = "two"
    else:
        size = "three_plus"
    kind = "stateful" if meta["changes_state"] else "read_only"
    services = "multi_service" if meta["n_services"] >= 2 else "single_service"
    return f"{kind}/{size}/{services}"


def success_table(records: list[dict], group: str = "all") -> dict:
    result = {}
    for mode in MODES:
        selected = [
            record
            for record in records
            if record["mode"] == mode
            and (group == "all" or group_of(record) == group)
        ]
        n = len(selected)
        passed = sum(bool(record["utility"]) for record in selected)
        result[mode] = {
            "n": n,
            "passed": passed,
            "pct": round(100 * passed / n, 1) if n else None,
        }
    return result


def all_success_tables(records: list[dict]) -> dict:
    groups = [
        "all",
        "single",
        "two",
        "three_plus",
        "read_only",
        "stateful",
        "multi_service",
    ]
    result = {}
    for group in groups:
        if group == "all":
            result[group] = success_table(records)
            continue
        selected = []
        for record in records:
            meta = record["task_meta"]
            keep = {
                "single": meta["n_ground_truth_calls"] == 1,
                "two": meta["n_ground_truth_calls"] == 2,
                "three_plus": meta["n_ground_truth_calls"] >= 3,
                "read_only": not meta["changes_state"],
                "stateful": meta["changes_state"],
                "multi_service": meta["n_services"] >= 2,
            }[group]
            if keep:
                selected.append(record)
        result[group] = success_table(selected)
    return result


def pairwise(records: list[dict]) -> dict:
    counts = Counter()
    for task_records in _records_by_task(records).values():
        if set(task_records) != set(MODES):
            continue
        a = bool(task_records["react"]["utility"])
        b = bool(task_records["plan"]["utility"])
        if a and b:
            counts["both_correct"] += 1
        elif not a and not b:
            counts["both_wrong"] += 1
        elif b:
            counts["plan_fixed"] += 1
        else:
            counts["plan_broken"] += 1
    n = sum(counts.values())
    delta = counts["plan_fixed"] - counts["plan_broken"]
    return {
        **dict(counts),
        "n": n,
        "delta": delta,
        "delta_pct": round(100 * delta / n, 1) if n else None,
    }


def _summary(values: list[float | int]) -> dict | None:
    if not values:
        return None
    return {
        "median": round(median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def costs(records: list[dict]) -> dict:
    fields = {
        "model_calls": "n_model_calls",
        "tool_calls": "n_tool_calls",
        "prompt_tokens": "prompt_tokens",
        "completion_tokens": "completion_tokens",
        "episode_seconds": "episode_s",
    }
    result = {}
    for mode in MODES:
        for status in ("all", "success", "failure"):
            selected = [
                record
                for record in records
                if record["mode"] == mode
                and (
                    status == "all"
                    or bool(record["utility"]) == (status == "success")
                )
            ]
            key = f"{mode}/{status}"
            result[key] = {
                "n": len(selected),
                **{
                    name: _summary([record[field] for record in selected])
                    for name, field in fields.items()
                },
            }
    return result


def failure_reasons(records: list[dict]) -> dict:
    result = {}
    for mode in MODES:
        selected = [
            record
            for record in records
            if record["mode"] == mode and not record["utility"]
        ]
        result[mode] = {
            "n": len(selected),
            "reasons": dict(
                Counter(
                    record["stop_reason"] for record in selected
                ).most_common()
            ),
            "tool_errors": sum(record["n_tool_errors"] for record in selected),
        }
    return result


def build_metrics(records: list[dict]) -> dict:
    grouped = _records_by_task(records)
    return {
        "n_episodes": len(records),
        "n_tasks": len(grouped),
        "success": all_success_tables(records),
        "pairwise": pairwise(records),
        "costs": costs(records),
        "failures": failure_reasons(records),
    }
