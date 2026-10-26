import csv
from pathlib import Path

def read_csv(path: Path, cast: dict[str, type] | None = None) -> list[dict]:
    """Читает CSV-файл в список словарей.

    params:
        path - путь к файлу
        cast - необязательный словарь {имя_колонки: тип} для приведения
               значений. Колонки, которых нет в cast, остаются строками.

    return: список словарей {имя_колонки: значение}

    Пример:
        rows = read(Path("history.csv"), cast={"epoch": int, "loss": float})
        # [{"epoch": 1, "loss": 0.53, "val_acc": "0.81"}, ...]
    """
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            return []

        if cast:
            missing = set(cast) - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"В файле нет колонок для приведения: {sorted(missing)}. "
                    f"Есть: {list(reader.fieldnames)}"
                )

        rows: list[dict] = []
        for row in reader:
            if cast:
                for col, typ in cast.items():
                    raw = row[col]
                    # пустая строка → None, чтобы не падать на int("")
                    row[col] = typ(raw) if raw != "" else None
            rows.append(row)

    return rows
