"""Пути, имена файлов и создание каталога runs/<run_id>/"""
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path('runs')
METADATA_FILENAME = 'metadata.json'
RESULTS_FILENAME = 'results.jsonl'
ERROR_FILENAME = 'error.txt'
CONFIG_FILENAME = 'config.yaml'

def _get_current_time() -> str:
    """Получение текущих даты и времени в формате
    YYYY-MM-DD_HH-MM-SS
    
    return: строку в формате YYYY-MM-DD_HH-MM-SS
    """
    date = datetime.now()

    year = date.year
    month = date.month
    day = date.day

    hour = date.hour
    minute = date.minute
    second = date.second

    return f'{year}-{month}-{day}_{hour}-{minute}-{second}'


def get_run_id(work: str, series: str, seed: int) -> str:
    """Генерация названия run_id
    
    params:
        work - номер работы
        series - имя серии (base/final/search)
        seed - сид генерации

    return: уникальное название для run_id
    """
    date = _get_current_time()
    return f'{work}_{series}_seed{seed}_{date}'


def create_run_id_dir(run_id: str) -> Path:
    """Создает директорию с названием run_id в директории
    runs, если ее нет. Если нет директории runs, создает ее.
    
    params:
        run_id - название директории
        
    return: путь к созданной директории.
    """
    RUNS_DIR.mkdir(exist_ok=True)
    run_id_dir = RUNS_DIR / run_id
    run_id_dir.mkdir(exist_ok=True)
    return run_id_dir
