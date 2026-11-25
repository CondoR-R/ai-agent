"""SHA-256 входных файлов ЛР4."""

from ai_agent import files


def input_sha256(cfg: dict) -> dict[str, str]:
    result = {
        files.ENDPOINT_PATH.name: files.get_sha256(files.ENDPOINT_PATH),
        files.MANIFEST_PATH.name: files.get_sha256(files.MANIFEST_PATH),
        "lr4_tasks": files.get_sha256(
            files.PROJECT_ROOT / cfg["agentdojo"]["tasks_file"]
        ),
    }
    for name, path in cfg["prompts"].items():
        result[f"prompt_{name}"] = files.get_sha256(files.PROJECT_ROOT / path)
    forecast = files.PROJECT_ROOT / "reports/lr4/forecast.md"
    if forecast.exists():
        result["lr4_forecast"] = files.get_sha256(forecast)
    return result
