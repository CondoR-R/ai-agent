"""Применение правила выбора ЛР3 к результатам dev."""

import argparse
from pathlib import Path

from ai_agent import files
from ai_agent.labs.lr2 import data, evaluation
from ai_agent.labs.lr3 import metrics, selection
from ai_agent.labs.lr3.ranking import ids


def _records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return files.read_jsonl(path)


def _rankings(records: list[dict], variant: str) -> dict:
    return {record["claim_id"]: record[variant] for record in records}


def _check_same_rrf(first: list[dict], second: list[dict]) -> None:
    left = {record["claim_id"]: record["rrf"] for record in first}
    right = {record["claim_id"]: record["rrf"] for record in second}
    if set(left) != set(right):
        raise ValueError("Наборы claim_id в RRF двух запусков не совпали")
    for claim_id in left:
        if ids(left[claim_id]) != ids(right[claim_id]):
            raise ValueError(f"RRF двух запусков не совпал: {claim_id}")


def main(
    exp1_dir: Path,
    exp2_dir: Path,
    config_path: Path,
    out_path: Path,
    *,
    force: bool,
) -> None:
    if out_path.exists() and not force:
        raise SystemExit(
            f"{out_path} уже существует; для перезаписи нужен --force"
        )
    first = _records(exp1_dir / files.RESULTS_FILENAME)
    second = _records(exp2_dir / "retrieval.jsonl")
    _check_same_rrf(first, second)

    rankings = {
        "bm25": _rankings(first, "bm25"),
        "rrf": _rankings(first, "rrf"),
        "rrf_ce": _rankings(second, "rrf_ce"),
    }
    claims = data.dev_claims()
    gold = evaluation.gold_by_id(claims)
    search = metrics.search_metrics(rankings, gold)
    values = {
        variant: metrics.selection_values(search["variants"][variant])
        for variant in selection.CANDIDATES
    }
    record = selection.write_selection(
        out_path,
        values,
        config_path=config_path,
        dev_runs={
            "exp1": exp1_dir.name,
            "exp2": exp2_dir.name,
        },
    )
    print(
        f"selected={record['selected']}; dev={record['dev_metrics']}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp1", required=True)
    parser.add_argument("--exp2", required=True)
    parser.add_argument("--config", default="configs/lr3/holdout.yaml")
    parser.add_argument("--out", default="reports/lr3/selection.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = files.PROJECT_ROOT
    main(
        (root / args.exp1).resolve(),
        (root / args.exp2).resolve(),
        (root / args.config).resolve(),
        (root / args.out).resolve(),
        force=args.force,
    )
