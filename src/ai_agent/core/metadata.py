"""Работа с файлом metadata.json в runs/<run_id>/"""
from pathlib import Path
import json

from .paths import METADATA_FILENAME


def write_metadata(run_dir: Path, metadata_dict: dict) -> None:
    """Записывает словарь в metadata.json. Ожидается, что в
    нём есть поле status со значением running.
    Создает или перезаписывает файл metadata.json

    params:
        run_dir - путь к директории с текущим run_id
        metadata_dict - словарь, с полями для записи в metadata.json
    """
    validate_metadata(metadata_dict, "running")
    metadata_path = run_dir / METADATA_FILENAME
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)


def update_status(run_dir: Path, status: str) -> None:
    """Обновляет статус в metadata.json

    params:
        run_dir - путь к директории с текущим run_id
        status - новый статус (completed или failed)

    raise:
        ValueError при некорректном статусе
    """
    if status not in ('completed', 'failed'):
        raise ValueError(
            f'Статус {status} некорректен, возможные значения: completed или failed'
        )
    metadata_path = run_dir / METADATA_FILENAME
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_dict = json.load(f)
    metadata_dict['status'] = status
    validate_metadata(metadata_dict, status)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)


def validate_metadata(data: dict, expected_status: str | None = None) -> None:
    """Валидация данных для metadata.json.
    Проверяет обязательные поля, типы, допустимые значения status.

    params:
        data - валидируемый словарь
        expected_status - статус, который будет присоен после валидации
    """


def write_error(path: Path, err) -> None:

