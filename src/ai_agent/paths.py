"""Пути, имена файлов"""

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
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
