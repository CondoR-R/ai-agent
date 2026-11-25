"""Данные и группы задач AgentDojo для ЛР4."""

import hashlib
import json
from importlib.metadata import version

from ai_agent import files

BENCHMARK_VERSION = "v1.2.2"
SUITE = "workspace"
DEV_STEP = 4
TASKS_REL = "data/lr4/tasks.json"
TASKS_PATH = files.DATA_DIR / "lr4" / "tasks.json"


def package_version() -> str:
    return version("agentdojo")


def read_tasks() -> dict:
    return files.read_json(TASKS_PATH)


def task_ids(mode: str) -> list[str]:
    tasks = read_tasks()
    return list(tasks[mode])


def task_meta(task_id: str) -> dict:
    tasks = read_tasks()
    return tasks["tasks"][task_id]


def split_mode(task_id: str) -> str:
    number = int(task_id.rsplit("_", 1)[1])
    return "dev" if number % DEV_STEP == 0 else "holdout"


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def write_tasks(tasks: dict) -> None:
    TASKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TASKS_PATH.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
