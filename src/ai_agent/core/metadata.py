"""Работа с файлом metadata.json в runs/<run_id>/"""

import json
import traceback
from datetime import datetime
from pathlib import Path

from .paths import ERROR_FILENAME, METADATA_FILENAME

REQUIRED_FIELDS = (
    "run_id",
    "work",
    "series",
    "status",
    "started_at",
    "commit",
    "device",
    "seed",
    "split_seed",
    "num_threads",
    "data_sha256",
    "versions",
)
VALID_STATUSES = ("running", "completed", "failed")
FINAL_STATUSES = ("completed", "failed")


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
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)


def update_status(run_dir: Path, status: str) -> None:
    """Обновляет статус в metadata.json

    params:
        run_dir - путь к директории с текущим run_id
        status - новый статус (completed или failed)

    raise:
        ValueError при некорректном статусе
    """
    if status not in FINAL_STATUSES:
        raise ValueError(f"Статус {status} некорректен, возможные значения: completed или failed")
    metadata_path = run_dir / METADATA_FILENAME
    with open(metadata_path, encoding="utf-8") as f:
        metadata_dict = json.load(f)
    metadata_dict["status"] = status
    metadata_dict["finished_at"] = datetime.now().isoformat(timespec="seconds")
    validate_metadata(metadata_dict, status)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)


def validate_metadata(data: dict, expected_status: str | None = None) -> None:
    """Валидация данных для metadata.json.
    Проверяет обязательные поля, типы, допустимые значения status.

    params:
        data - валидируемый словарь
        expected_status - статус, который будет присвоен после валидации

    raise:
        TypeError, при неправильном типе данных
        ValueError, при отсутствии обязательных полей, некорректных значениях
    """
    # data - словарь
    if not isinstance(data, dict):
        raise TypeError("Данные для metadata должны быть словарем (dict)")

    # наличие обязательных полей
    for field in REQUIRED_FIELDS:
        if field not in data:
            raise ValueError(f"Отсутствует обязательное поле {field}")

    status = data["status"]
    # значение status допустимо
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Поле status может принимать значения: {VALID_STATUSES}, получено {status}"
        )

    # совпадение expected_status с фактическим
    if expected_status is not None and expected_status != status:
        raise ValueError("Значение expected_status не совпадает с фактическим в поле status")

    # наличие finished_at при status = "completed" или "failed"
    if (status == "completed" or status == "failed") and data.get("finished_at") is None:
        raise ValueError(
            f"При значении поля status = {status} должно быть непустое поле finished_at"
        )

    # отсутствие finished_at при status = "running"
    if status == "running" and data.get("finished_at") is not None:
        raise ValueError(
            f"При значении поля status = {status} поле finished_at должно отсутствовать или быть пустым"
        )


def write_error(run_dir: Path, err: BaseException) -> None:
    """пишет traceback в error.txt и добавляет короткое error_message в metadata.json.

    params:
        run_dir - путь к директории текущего запуска
        err - ошибка
    """
    err_msg_list = traceback.format_exception(type(err), err, err.__traceback__)
    err_msg = "".join(err_msg_list)

    with open(run_dir / ERROR_FILENAME, "w", encoding="utf-8") as f:
        f.write(err_msg)

    metadata_path = run_dir / METADATA_FILENAME
    with open(metadata_path, encoding="utf-8") as f:
        metadata_dict = json.load(f)
    metadata_dict["error_message"] = f"{type(err).__name__}: {err}"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
