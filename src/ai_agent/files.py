"""Работа с файлами, чтение/запись/обновление"""

import csv
import hashlib
import json
import traceback
from pathlib import Path
import uuid

import yaml

from . import utils

MARKER_FILENAME = "pyproject.toml"

METADATA_FILENAME = "metadata.json"
RESULTS_FILENAME = "results.jsonl"
ERROR_FILENAME = "error.txt"
CONFIG_FILENAME = "config.yaml"
METRICS_FILENAME = "metrics.json"


def _find_project_root(marker: str = MARKER_FILENAME) -> Path:
    """Ищет корень проекта, поднимаясь от текущего файла вверх
    до каталога, в котором лежит файл-маркер (по умолчанию pyproject.toml).

    params:
        marker - имя файла-маркера

    return: путь к корню проекта

    raise:
        RuntimeError, если маркер не найден
    """
    current = Path(__file__).resolve().parent
    while True:
        if (current / marker).exists():
            return current
        if current.parent == current:
            raise RuntimeError(
                f"Не удалось найти {marker}: корень проекта не определён"
            )
        current = current.parent


PROJECT_ROOT = _find_project_root()

RUNS_DIR = PROJECT_ROOT / "runs"
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"
MLFLOW_ARTIFACTS_DIR = PROJECT_ROOT / "mlartifacts"


def get_run_id(work: str, experiment: int) -> str:
    """Генерация названия <run_id>/

    params:
        work - название работы
        experiment - номер эксперимента

    return: название директории
    """
    suffix = uuid.uuid4().hex[:6]
    return f"{work}-experiment{experiment}-{utils.time_now()}-{suffix}"


def create_run_dir(dir_name: Path | str) -> Path:
    """Создание директории запуска в runs/

    params:
        dir_name - название директории <run_id>/

    return: путь к директории запуска
    """
    dir_name = Path(dir_name)
    dir_path = RUNS_DIR / dir_name

    dir_path.parent.mkdir(parents=True, exist_ok=True)
    dir_path.mkdir(parents=True, exist_ok=False)

    return dir_path


def get_sha256(path: str | Path) -> str:
    """Вычисляет SHA-256 файла

    params:
        path - путь к файлу

    return: hash-строку для файла
    """
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path: Path, record: dict) -> None:
    """Запись/перезапись json файла

    params:
        path - путь к файлу
        record - записываемый словарь, сериализуемый в JSON)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_json(path: Path) -> dict:
    """Чтение JSON файла

    params:
        path - путь к файлу

    return: словарь из файла
    """
    if not path.exists():
        raise FileNotFoundError("Файл не найден по указанному пути")
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    return record


def write_yaml(path: Path, record: dict) -> None:
    """Запись/перезапись yaml файла

    params:
        path - путь к файлу
        record - словрь записи
    """
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(record, f, allow_unicode=True, sort_keys=False)


def read_yaml(path: Path) -> dict:
    """Чтение YAML файла

    params:
        path - путь к файлу

    return: словарь с данными из файла
    """
    if not path.exists():
        raise FileNotFoundError("Файл не найден по указанному пути")
    with open(path, encoding="utf-8") as f:
        record = yaml.safe_load(f)
    return record


def write_csv(path: Path, records: list[dict]) -> None:
    """Запись/перезапись в csv файл

    params:
        path - путь к csv файлу
        record - записываемые данные
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not records:
            return

        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def update_csv(path: Path, record: dict) -> None:
    """Добавление одной записи в csv файл.

    Если первый вызов - создаст файл

    params:
        path - путь к файлу
        record - записываемый словарь
    """
    is_new = not path.exists() or path.stat().st_size == 0

    if is_new:
        header = list(record.keys())
    else:
        with path.open("r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), list(record.keys()))

    mode = "w" if is_new else "a"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(record)


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
    if not path.exists():
        raise FileNotFoundError("Файл не найден по указанному пути")
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


def write_error(path: Path, exc: BaseException) -> None:
    """Записывает трейсбек исключения в текстовый файл.

    params:
        path - путь к файлу (например, runs/<run_id>/error.txt)
        exc  - пойманное исключение
    """

    text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )

    with path.open("w", encoding="utf-8") as f:
        f.write(text)
