"""Сравнение предсказаний запусков и запись отчётов ЛР0."""

from pathlib import Path

from ai_agent import files

REPORTS_DIR = files.PROJECT_ROOT / "reports" / "lr0"


def load_predictions(run_dir: Path, split: str) -> dict[int, dict]:
    """Читает predictions.csv и индексирует строки по index.

    params:
        run_dir - каталог запуска
        split - часть разбиения (val или test)

    return: словарь {index: строка predictions.csv}

    raise:
        ValueError, если для указанного split нет ни одной строки
    """
    rows = files.read_csv(
        run_dir / "predictions.csv",
        cast={"index": int, "y_true": int, "y_pred": int},
    )
    predictions = {row["index"]: row for row in rows if row["split"] == split}
    if not predictions:
        raise ValueError(
            f"В {run_dir}/predictions.csv нет строк split={split!r}"
        )
    return predictions


def compare_predictions(
    first_dir: Path,
    second_dir: Path,
    split: str,
    with_examples: bool = True,
) -> dict:
    """Сравнивает два запуска по одним и тем же записям split.

    first - базовый вариант, second - сравниваемый. Счётчики:
        both_correct - оба верно;
        only_first_correct - верно только first;
        only_second_correct - верно только second;
        both_wrong - оба ошиблись.

    params:
        first_dir - каталог базового запуска
        second_dir - каталог сравниваемого запуска
        split - часть разбиения (val или test)
        with_examples - добавлять ли примеры расхождений

    return: словарь с n, counts, first_acc, second_acc, delta_acc, examples

    raise:
        ValueError, если наборы index или метки y_true различаются
    """
    first = load_predictions(first_dir, split)
    second = load_predictions(second_dir, split)

    if first.keys() != second.keys():
        raise ValueError(
            f"Разные наборы index в split={split!r}: "
            f"{first_dir.name} и {second_dir.name}"
        )

    counts = {
        "both_correct": 0,
        "only_first_correct": 0,
        "only_second_correct": 0,
        "both_wrong": 0,
    }
    examples = []

    for index in sorted(first):
        row_1, row_2 = first[index], second[index]
        if row_1["y_true"] != row_2["y_true"]:
            raise ValueError(f"Разные y_true для index={index}")

        ok_1 = row_1["y_true"] == row_1["y_pred"]
        ok_2 = row_2["y_true"] == row_2["y_pred"]

        if ok_1 and ok_2:
            counts["both_correct"] += 1
        elif ok_1:
            counts["only_first_correct"] += 1
        elif ok_2:
            counts["only_second_correct"] += 1
        else:
            counts["both_wrong"] += 1

        if with_examples and ok_1 != ok_2:
            examples.append(
                {
                    "index": index,
                    "y_true": row_1["y_true"],
                    "first_pred": row_1["y_pred"],
                    "second_pred": row_2["y_pred"],
                }
            )

    n = len(first)
    return {
        "n": n,
        "counts": counts,
        "first_acc": (
            counts["both_correct"] + counts["only_first_correct"]
        ) / n,
        "second_acc": (
            counts["both_correct"] + counts["only_second_correct"]
        ) / n,
        "delta_acc": (
            counts["only_second_correct"] - counts["only_first_correct"]
        ) / n,
        "examples": examples,
    }


def update_report(filename: str, section: str, value: dict) -> Path:
    """Дописывает секцию в reports/lr0/<filename>, сохраняя остальные.

    params:
        filename - имя файла отчёта (например, experiment_1.json)
        section - имя секции (например, task_3)
        value - данные секции

    return: путь к файлу отчёта
    """
    path = REPORTS_DIR / filename
    try:
        report = files.read_json(path)
    except FileNotFoundError:
        report = {}
    report[section] = value
    files.write_json(path, report)
    return path
