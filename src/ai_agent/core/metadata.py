"""Работа с файлом metadata.json в runs/<run_id>/"""

import json
import platform
import traceback
from datetime import datetime, timezone
from importlib.metadata import version as lib_version
from pathlib import Path

from .paths import ERROR_FILENAME, METADATA_FILENAME


VALID_STATUSES = ("running", "completed", "failed")
FINAL_STATUSES = ("completed", "failed")

REQUIRED_FIELDS = (
    "run_id",
    "work",
    "series",
    "status",
    "started_at",
    "commit",
    "device",
    "seed",
    "data_sha256",
    "versions",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(
    run_dir: Path,
    metadata_dict: dict,
    required_fields: tuple[str, ...] = REQUIRED_FIELDS,
) -> None:
    """Записывает словарь в metadata.json. Ожидается, что в
    нём есть поле status со значением running.
    Создает или перезаписывает файл metadata.json

    params:
        run_dir - путь к директории с текущим run_id
        metadata_dict - словарь, с полями для записи в metadata.json
        required_fields - кортеж обязательных полей, default = REQUIRED_FIELDS
    """
    validate(metadata_dict, "running", required_fields)
    metadata_path = run_dir / METADATA_FILENAME
    metadata_path.write_text(
        json.dumps(metadata_dict, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def update_status(run_dir: Path, status: str) -> None:
    """Обновляет статус в metadata.json

    params:
        run_dir - путь к директории с текущим run_id
        status - новый статус (completed или failed)

    raise:
        ValueError при некорректном статусе
    """
    if status not in FINAL_STATUSES:
        raise ValueError(
            f"Статус {status} некорректен, возможные значения: completed или failed"
        )
    metadata_path = run_dir / METADATA_FILENAME
    metadata_dict = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_dict["status"] = status
    metadata_dict["finished_at"] = _now()
    validate(metadata_dict, status)
    metadata_path.write_text(
        json.dumps(metadata_dict, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate(
    data: dict,
    expected_status: str | None = None,
    required_fields: tuple[str, ...] = REQUIRED_FIELDS,
) -> None:
    """Валидация данных для metadata.json.
    Проверяет обязательные поля, типы, допустимые значения status.

    params:
        data - валидируемый словарь
        expected_status - статус, который будет присвоен после валидации
        required_fields - кортеж обязательных полей, default = REQUIRED_FIELDS

    raise:
        TypeError, при неправильном типе данных
        ValueError, при отсутствии обязательных полей, некорректных значениях
    """
    # data - словарь
    if not isinstance(data, dict):
        raise TypeError("Данные для metadata должны быть словарем (dict)")

    # наличие обязательных полей
    for field in required_fields:
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
        raise ValueError(
            "Значение expected_status не совпадает с фактическим в поле status"
        )

    # наличие finished_at при status = "completed" или "failed"
    if (status == "completed" or status == "failed") and data.get(
        "finished_at"
    ) is None:
        raise ValueError(
            f"При значении поля status = {status} должно быть непустое поле finished_at"
        )

    # отсутствие finished_at при status = "running"
    if status == "running" and data.get("finished_at") is not None:
        raise ValueError(
            f"При значении поля status = {status} поле finished_at должно отсутствовать или быть пустым"
        )


def write_error(run_dir: Path, err: BaseException) -> None:
    """Пишет traceback в error.txt и добавляет короткое error_message в metadata.json.

    params:
        run_dir - путь к директории текущего запуска
        err - ошибка
    """
    err_msg_list = traceback.format_exception(type(err), err, err.__traceback__)
    err_msg = "".join(err_msg_list)

    (run_dir / ERROR_FILENAME).write_text(err_msg, encoding="utf-8")

    metadata_path = run_dir / METADATA_FILENAME
    metadata_dict = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_dict["error_message"] = f"{type(err).__name__}: {err}"
    metadata_path.write_text(
        json.dumps(metadata_dict, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def collect_versions(libs: list[str]) -> dict[str, str]:
    """Собирает версии python и библиотек

    params:
        libs - список библиотек

    return: словарь с версиями
    """
    versions: dict[str, str] = {}
    versions["python"] = platform.python_version()

    for name in libs:
        versions[name] = lib_version(name)

    return versions
