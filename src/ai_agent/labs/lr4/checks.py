"""Проверки обвязки ЛР4 без модели."""

import json
from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import (
    EmptyEnv,
    Env,
    FunctionCall,
    FunctionsRuntime,
)
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatToolResultMessage,
)
from agentdojo.types import (
    text_content_block_from_string as text,
)

from ai_agent import files, utils
from ai_agent.labs.lr4 import data, tools
from ai_agent.labs.lr4.agent import ToolAgent
from ai_agent.labs.lr4.client import ChatResult


def call(name: str, args: dict, call_id: str) -> dict:
    return {
        "id": call_id,
        "name": name,
        "args": args,
        "raw_args": json.dumps(args),
    }


class ScriptedClient:
    def __init__(self, script: list[ChatResult]):
        self.script = script
        self.index = 0

    def chat(self, messages: list[dict], **kwargs: object) -> ChatResult:
        if "tools" not in kwargs or kwargs.get("tools") is None:
            return ChatResult("Scripted final answer.", [], 10, 5, 0.001)
        result = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        return result


def result_model(
    content: str = "", calls: list[dict] | None = None
) -> ChatResult:
    return ChatResult(content, calls or [], 10, 5, 0.001)


class GroundTruthEchoAgent(BasePipelineElement):
    def __init__(self, task: object):
        self.task = task
        self.name = "local-ground-truth-echo"

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env | None = None,
        messages: Sequence[ChatMessage] = (),
        extra_args: dict | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        env = EmptyEnv() if env is None else env
        log: list[ChatMessage] = []
        outputs = []
        for index, ground_call in enumerate(self.task.ground_truth(env)):
            function_call = FunctionCall(
                function=ground_call.function,
                args=ground_call.args,
                id=f"gt_{index}",
            )
            result, error = runtime.run_function(
                env, ground_call.function, ground_call.args
            )
            output = tools.result_to_str(result)
            if error:
                output = f"{error}\n{output}"
            outputs.append(output)
            log.append(
                ChatAssistantMessage(
                    role="assistant", content=None, tool_calls=[function_call]
                )
            )
            log.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text(output)],
                    tool_call_id=function_call.id,
                    tool_call=function_call,
                    error=error,
                )
            )
        answer = "\n".join([*outputs, self.task.GROUND_TRUTH_OUTPUT])
        log.append(
            ChatAssistantMessage(
                role="assistant", content=[text(answer)], tool_calls=None
            )
        )
        return query, runtime, env, log, dict(extra_args or {})


def base_cfg() -> dict:
    cfg = files.read_yaml(files.PROJECT_ROOT / "configs/lr4/exp.yaml")
    cfg["agent"]["mode"] = "react"
    return cfg


def tool_agent(cfg: dict, client: ScriptedClient) -> ToolAgent:
    return ToolAgent(client, cfg)


def invalid_calls_check(suite: object, task: object, cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["agent"] = dict(cfg["agent"], max_model_calls=5, max_tool_calls=5)
    client = ScriptedClient(
        [
            result_model(calls=[call("does_not_exist", {}, "bad_1")]),
            result_model(calls=[call("search_calendar_events", {}, "bad_2")]),
        ]
    )
    agent = tool_agent(cfg, client)
    suite.run_task_with_pipeline(agent, task, None, {})
    errors = [
        step["error"] or "" for step in agent.trace if step["kind"] == "tool"
    ]
    passed = (
        any("ToolNotFoundError" in error for error in errors)
        and any("ValidationError" in error for error in errors)
        and bool(agent.answer)
    )
    return {
        "name": "invalid_tool_and_missing_argument",
        "passed": passed,
        "errors": errors,
        "answer": agent.answer,
        "stop_reason": agent.stop_reason,
    }


def tool_limit_check(suite: object, task: object, cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["agent"] = dict(cfg["agent"], max_model_calls=30, max_tool_calls=15)
    client = ScriptedClient(
        [result_model(calls=[call("does_not_exist", {}, "loop")])]
    )
    agent = tool_agent(cfg, client)
    suite.run_task_with_pipeline(agent, task, None, {})
    tool_steps = [step for step in agent.trace if step["kind"] == "tool"]
    passed = (
        agent.stop_reason == "max_tools"
        and len(tool_steps) == 15
        and bool(agent.answer)
    )
    return {
        "name": "max_tool_calls",
        "passed": passed,
        "n_tool_steps": len(tool_steps),
        "stop_reason": agent.stop_reason,
        "answer": agent.answer,
    }


def ground_truth_echo_check(suite: object) -> dict:
    passed = []
    failed = []
    for task_id, task in sorted(suite.user_tasks.items()):
        utility, _ = suite.run_task_with_pipeline(
            GroundTruthEchoAgent(task), task, None, {}
        )
        (passed if utility else failed).append(task_id)
    return {
        "name": "ground_truth_echo",
        "passed": len(passed) >= 36,
        "n_passed": len(passed),
        "n_failed": len(failed),
        "expected": "36 из 40",
        "failed_ids": failed,
    }


def plan_step_check(suite: object, task: object, cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["agent"] = dict(cfg["agent"], mode="plan")
    plan = call(
        "submit_plan",
        {
            "steps": [
                {
                    "tool": "search_calendar_events",
                    "purpose": "find the calendar event",
                }
            ]
        },
        "plan_1",
    )
    wrong = call(
        "send_email",
        {"recipient": "x@example.com", "subject": "x", "body": "x"},
        "wrong_1",
    )
    client = ScriptedClient(
        [result_model(calls=[plan]), result_model(calls=[wrong])]
    )
    agent = tool_agent(cfg, client)
    before = suite.user_tasks[task.ID].init_environment(
        suite.load_and_inject_default_environment({})
    )
    env = before.model_copy(deep=True)
    runtime = FunctionsRuntime(suite.tools)
    agent.query(task.PROMPT, runtime, env)
    executed = [step for step in agent.trace if step["kind"] == "tool"]
    passed = (
        not executed
        and env.model_dump_json() == before.model_dump_json()
        and bool(agent.answer)
    )
    return {
        "name": "plan_rejects_wrong_tool",
        "passed": passed,
        "n_executed_tools": len(executed),
        "env_unchanged": env.model_dump_json() == before.model_dump_json(),
        "answer": agent.answer,
    }


def main() -> None:
    cfg = base_cfg()
    suite = get_suites(cfg["agentdojo"]["benchmark_version"])[
        cfg["agentdojo"]["suite"]
    ]
    task = suite.get_user_task_by_id("user_task_0")
    checks = [
        invalid_calls_check(suite, task, cfg),
        tool_limit_check(suite, task, cfg),
        ground_truth_echo_check(suite),
        plan_step_check(suite, task, cfg),
    ]
    report = {
        "benchmark_version": cfg["agentdojo"]["benchmark_version"],
        "suite": cfg["agentdojo"]["suite"],
        "agentdojo_version": data.package_version(),
        "checks": checks,
        "pass": all(check["passed"] for check in checks),
        "generated_at": utils.time_now(),
    }
    path = files.PROJECT_ROOT / "reports/lr4/checks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    files.write_json(path, report)
    for check in checks:
        print(
            f"[checks] {check['name']}: {'ok' if check['passed'] else 'FAIL'}"
        )
    print(f"[checks] pass={report['pass']}, файл={path}")


if __name__ == "__main__":
    main()
