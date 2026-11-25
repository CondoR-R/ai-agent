"""Проверка tool calling сервера одной задачей."""

from ai_agent import files
from ai_agent.labs.lr4.client import ModelClient

TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_day",
        "description": "Return the current day.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}


def main() -> None:
    client = ModelClient()
    result = client.chat(
        [
            {
                "role": "user",
                "content": "What day is it today? Use the available tool.",
            }
        ],
        [TOOL],
        {"type": "function", "function": {"name": "get_current_day"}},
    )
    report = {
        "base_url": client.base_url,
        "model": client.model,
        "n_tool_calls": len(result.tool_calls),
        "tool_calls": result.tool_calls,
        "content": result.content,
        "pass": any(
            call["name"] == "get_current_day" for call in result.tool_calls
        ),
    }
    path = files.PROJECT_ROOT / "reports/lr4/model_tool_check.json"
    files.write_json(path, report)
    print(f"[tool-check] pass={report['pass']}, файл={path}")


if __name__ == "__main__":
    main()
