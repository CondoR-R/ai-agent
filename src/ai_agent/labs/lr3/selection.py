"""Правило выбора варианта по dev и запись selection.json."""

from pathlib import Path

from ai_agent import files, utils

CANDIDATES = ("bm25", "rrf", "rrf_ce")
COMPLEXITY = {"bm25": 0, "rrf": 1, "rrf_ce": 2}
SELECTION_RULE = (
    "Из BM25, RRF и RRF+CE выбрать максимум MRR@3 на dev; "
    "при равенстве — максимум Recall@3; затем более простой вариант."
)


def choose_variant(values: dict[str, dict]) -> tuple[str, dict]:
    """Применяет правило MRR@3 -> Recall@3 -> простота."""

    def key(variant: str) -> tuple[float, float, int]:
        metrics = values[variant]
        return (
            float(metrics["mrr@3"]),
            float(metrics["recall@3"]),
            -COMPLEXITY[variant],
        )

    selected = max(CANDIDATES, key=key)
    return selected, {
        "selected": selected,
        "rule": SELECTION_RULE,
        "comparison_key": [
            "MRR@3 (max)",
            "Recall@3 (max)",
            "complexity (min)",
        ],
        "dev_metrics": {variant: values[variant] for variant in CANDIDATES},
        "complexity_order": list(CANDIDATES),
    }


def write_selection(
    path: Path,
    values: dict[str, dict],
    *,
    config_path: Path,
    dev_runs: dict[str, str],
) -> dict:
    """Сохраняет выбор, метрики dev, коммит и хеш конфигурации."""
    selected, record = choose_variant(values)
    record.update(
        {
            "selected": selected,
            "config": {
                "path": str(config_path.relative_to(files.PROJECT_ROOT)),
                "sha256": files.get_sha256(config_path),
            },
            "dev_runs": dev_runs,
            "commit": utils.get_commit_hash(),
            "generated_at": utils.time_now(),
        }
    )
    files.write_json(path, record)
    return record
