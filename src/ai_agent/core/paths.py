"""Пути, имена файлов и создание каталога runs/<run_id>/"""

from datetime import datetime
from pathlib import Path

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


def get_run_id(work: str, series: str, seed: int) -> str:
    """Генерация названия run_id

    params:
        work - номер работы
        series - имя серии (base/final/search)
        seed - сид генерации

    return: уникальное название для run_id
    """
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{work}_{series}_seed{seed}_{date}"


def create_run_id_dir(run_id: str, runs_dir: Path = RUNS_DIR) -> Path:
    """Создает директорию с названием run_id в директории
    runs_dir, если ее нет. Если нет директории runs_dir, создает ее.

    params:
        run_id - название директории
        runs_dir — базовый каталог запусков, по умолчанию RUNS_DIR

    return: путь к созданной директории.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id_dir = runs_dir / run_id
    run_id_dir.mkdir(parents=True, exist_ok=True)
    return run_id_dir
