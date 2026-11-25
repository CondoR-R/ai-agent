"""Шаблоны промптов ЛР4."""

from pathlib import Path

from ai_agent import files


def load(path: str | Path) -> str:
    full = path if Path(path).is_absolute() else files.PROJECT_ROOT / path
    return full.read_text().strip() + "\n"


def render(template: str, **values: object) -> str:
    text = template
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def tools_description(tools: list[dict]) -> str:
    lines = []
    for tool in tools:
        fn = tool["function"]
        params = fn.get("parameters", {}).get("properties", {})
        required = set(fn.get("parameters", {}).get("required", []))
        args = []
        for name, spec in params.items():
            typ = spec.get("type")
            if typ is None and "anyOf" in spec:
                typ = "/".join(
                    t.get("type", "object")
                    for t in spec["anyOf"]
                    if "type" in t
                )
            req = "required" if name in required else "optional"
            args.append(f"{name}:{typ}:{req}")
        arg_text = ", ".join(args) if args else "no args"
        lines.append(f"- {fn['name']}({arg_text}): {fn['description']}")
    return "\n".join(lines)


def results_description(results: list[dict]) -> str:
    return "\n".join(f"- {item['tool']}: {item['result']}" for item in results)
