"""Преобразование инструментов AgentDojo в формат OpenAI tools."""

import ast
import json

from agentdojo.agent_pipeline.tool_execution import (
    tool_result_to_str as ad_tool_result_to_str,
)
from agentdojo.functions_runtime import Function


def to_openai_tool(function: Function) -> dict:
    return {
        "type": "function",
        "function": {
            "name": function.name,
            "description": function.description,
            "parameters": function.parameters.model_json_schema(),
        },
    }


def to_openai_tools(functions: dict[str, Function]) -> list[dict]:
    return [to_openai_tool(function) for function in functions.values()]


def plan_tool(names: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": "Submit the ordered plan of tool calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool": {"type": "string", "enum": names},
                                "purpose": {"type": "string"},
                            },
                            "required": ["tool", "purpose"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["steps"],
                "additionalProperties": False,
            },
        },
    }


def forced_choice(name: str) -> dict:
    return {"type": "function", "function": {"name": name}}


def step_call_valid(call: dict, expected_tool: str) -> bool:
    return call.get("name") == expected_tool


def parse_plan(args: dict, names: list[str]) -> list[dict] | None:
    steps = args.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 8:
        return None
    result = []
    for step in steps:
        if not isinstance(step, dict):
            return None
        tool = step.get("tool")
        purpose = step.get("purpose")
        if tool not in names or not isinstance(purpose, str) or not purpose:
            return None
        result.append({"tool": tool, "purpose": purpose})
    return result


def coerce(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    if isinstance(parsed, (list, dict, int, float, bool)):
        return parsed
    return value


def coerce_args(args: dict) -> dict:
    return {key: coerce(value) for key, value in args.items()}


def result_to_str(result: object, max_chars: int = 0) -> str:
    text = ad_tool_result_to_str(result)
    if max_chars > 0 and len(text) > max_chars:
        return (
            text[:max_chars] + f"\n... truncated {len(text) - max_chars} chars"
        )
    return text


def estimate_tokens(messages: list[dict]) -> int:
    return len(json.dumps(messages, ensure_ascii=False)) // 4 + 4 * len(
        messages
    )
