"""Подготовка manifest задач AgentDojo для ЛР4."""

from agentdojo.task_suite.load_suites import get_suites

from ai_agent import files
from ai_agent.labs.lr4 import data

WRITE_TOOLS = {
    "add_calendar_event_participants",
    "append_to_file",
    "cancel_calendar_event",
    "create_calendar_event",
    "create_file",
    "delete_email",
    "delete_file",
    "reschedule_calendar_event",
    "send_email",
    "share_file",
}


def service(tool: str) -> str | None:
    if "calendar" in tool:
        return "calendar"
    if "email" in tool or "draft" in tool or "sent" in tool or "unread" in tool:
        return "mail"
    if "file" in tool:
        return "files"
    return None


def main() -> None:
    suite = get_suites(data.BENCHMARK_VERSION)[data.SUITE]
    package = data.package_version()
    tasks = {}
    for task_id, task in suite.user_tasks.items():
        number = int(task_id.rsplit("_", 1)[1])
        env = suite.load_and_inject_default_environment({})
        grounded = task.ground_truth(task.init_environment(env))
        tools = [call.function for call in grounded]
        services = sorted({service(tool) for tool in tools} - {None})
        tasks[task_id] = {
            "id": task_id,
            "number": number,
            "split": data.split_mode(task_id),
            "prompt_sha256": data.prompt_sha256(task.PROMPT),
            "difficulty": task.DIFFICULTY.name,
            "n_ground_truth_calls": len(grounded),
            "changes_state": any(tool in WRITE_TOOLS for tool in tools),
            "n_services": len(services),
        }
    ordered = dict(sorted(tasks.items(), key=lambda item: item[1]["number"]))
    payload = {
        "benchmark_version": data.BENCHMARK_VERSION,
        "suite": data.SUITE,
        "package_version": package,
        "n_tasks": len(ordered),
        "dev": [
            task_id
            for task_id, meta in ordered.items()
            if meta["split"] == "dev"
        ],
        "holdout": [
            task_id
            for task_id, meta in ordered.items()
            if meta["split"] == "holdout"
        ],
        "tasks": ordered,
    }
    data.write_tasks(payload)

    manifest = files.read_json(files.MANIFEST_PATH)
    manifest["agentdojo"] = {
        "source": "AgentDojo PyPI package",
        "version": package,
        "benchmark_version": data.BENCHMARK_VERSION,
        "suite": data.SUITE,
        "rule": {
            "dev": f"user_task_n где n кратно {data.DEV_STEP}",
            "holdout": "остальные 30 задач",
        },
        "sizes": {
            "dev": len(payload["dev"]),
            "holdout": len(payload["holdout"]),
            "total": len(ordered),
        },
        "files": {
            data.TASKS_REL: {
                "sha256": files.get_sha256(data.TASKS_PATH),
                "n_records": len(ordered),
            }
        },
    }
    files.write_json(files.MANIFEST_PATH, manifest)
    print(
        f"ЛР4: {len(ordered)} задач, dev={len(payload['dev'])}, "
        f"holdout={len(payload['holdout'])}"
    )
    print(f"Манифест: {data.TASKS_PATH}, {files.MANIFEST_PATH}")


if __name__ == "__main__":
    main()
