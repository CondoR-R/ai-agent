"""Собственная обвязка агента: ReAct и plan-then-execute."""

import time
from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import (
    EmptyEnv,
    Env,
    FunctionCall,
    FunctionsRuntime,
)
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    ChatUserMessage,
)
from agentdojo.types import (
    text_content_block_from_string as text,
)

from ai_agent.labs.lr4 import prompts, tools
from ai_agent.labs.lr4.client import ChatResult, ModelClient


class ToolAgent(BasePipelineElement):
    def __init__(self, client: ModelClient, cfg: dict):
        agent_cfg = cfg["agent"]
        self.client = client
        self.cfg = cfg
        self.mode = agent_cfg["mode"]
        self.limits = agent_cfg
        self.name = "local-" + cfg["model"]["name"]
        self.trace: list[dict] = []
        self.totals: dict[str, int | float] = {
            "model_calls": 0,
            "tool_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_input_tokens": 0,
            "latency_s": 0.0,
        }
        self.stop_reason = "text"
        self.answer = ""
        self.plan: list[dict] = []
        prompt_paths = cfg.get("prompts", {})
        self.system = prompts.load(
            prompt_paths["system_react"]
            if self.mode == "react"
            else prompt_paths["system_plan"]
        )

    @property
    def max_calls(self) -> int:
        return int(self.limits["max_model_calls"])

    @property
    def max_tools(self) -> int:
        return int(self.limits["max_tool_calls"])

    def _record_model(
        self,
        result: ChatResult,
        messages: list[dict],
        offered: list[str],
        label: str,
        forced: str | None,
    ) -> None:
        self.totals["prompt_tokens"] += result.prompt_tokens
        self.totals["completion_tokens"] += result.completion_tokens
        self.totals["latency_s"] += result.latency_s
        self.trace.append(
            {
                "kind": "model",
                "step": len(self.trace) + 1,
                "label": label,
                "messages": messages,
                "n_messages": len(messages),
                "estimated_input_tokens": tools.estimate_tokens(messages),
                "tools_offered": offered,
                "tool_choice": forced,
                "content": result.content,
                "tool_calls": result.tool_calls,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_s": result.latency_s,
                "model_calls": self.totals["model_calls"],
            }
        )

    def _record_tool(
        self,
        name: str,
        args: dict,
        output: str,
        error: str | None,
        latency_s: float,
    ) -> None:
        self.trace.append(
            {
                "kind": "tool",
                "step": len(self.trace) + 1,
                "tool": name,
                "args": args,
                "result": output,
                "error": error,
                "latency_s": latency_s,
                "tool_calls": self.totals["tool_calls"],
            }
        )

    def _model(
        self,
        messages: list[dict],
        label: str,
        openai_tools: list[dict] | None = None,
        forced: str | None = None,
    ) -> ChatResult | None:
        estimated = tools.estimate_tokens(messages)
        self.totals["estimated_input_tokens"] = max(
            self.totals["estimated_input_tokens"], estimated
        )
        if estimated > int(self.limits["max_input_tokens"]):
            self.stop_reason = "context_limit"
            return None
        if self.totals["model_calls"] >= self.max_calls:
            self.stop_reason = "max_model_calls"
            return None
        self.totals["model_calls"] += 1
        choice = tools.forced_choice(forced) if forced else None
        result = self.client.chat(
            messages,
            tools=openai_tools,
            tool_choice=choice,
            max_tokens=int(self.limits["max_output_tokens"]),
        )
        self._record_model(
            result,
            messages,
            [tool["function"]["name"] for tool in openai_tools or []],
            label,
            forced,
        )
        return result

    def _execute(
        self,
        call: dict,
        runtime: FunctionsRuntime,
        env: Env,
        chat: list[dict],
        log: list[ChatMessage],
    ) -> str:
        started = time.perf_counter()
        self.totals["tool_calls"] += 1
        name = call["name"]
        args = tools.coerce_args(call["args"])
        function_call = FunctionCall(
            function=name,
            args=args,
            id=call.get("id") or f"call_{self.totals['tool_calls']}",
        )
        result, error = runtime.run_function(env, name, args)
        output = tools.result_to_str(
            result, int(self.limits.get("tool_result_chars") or 0)
        )
        if error:
            output = f"{error}\n{output}"
        self._record_tool(
            name, args, output, error, time.perf_counter() - started
        )
        chat.append(
            {
                "role": "tool",
                "tool_call_id": function_call.id,
                "content": output,
            }
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
        return output

    def _assistant_calls(
        self, calls: list[dict], chat: list[dict], log: list[ChatMessage]
    ) -> None:
        function_calls = []
        for call in calls:
            function_calls.append(
                FunctionCall(
                    function=call["name"],
                    args=tools.coerce_args(call["args"]),
                    id=call.get("id")
                    or f"assistant_call_{len(function_calls)}",
                )
            )
        chat.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function,
                            "arguments": __import__("json").dumps(
                                call.args, ensure_ascii=False
                            ),
                        },
                    }
                    for call in function_calls
                ],
            }
        )
        log.append(
            ChatAssistantMessage(
                role="assistant", content=None, tool_calls=function_calls
            )
        )

    def _text_message(
        self, role: str, content: str, chat: list[dict], log: list[ChatMessage]
    ) -> None:
        chat.append({"role": role, "content": content})
        message = (
            ChatUserMessage(role="user", content=[text(content)])
            if role == "user"
            else ChatAssistantMessage(
                role="assistant", content=[text(content)], tool_calls=None
            )
        )
        log.append(message)

    def _guess_tools(self, query: str, openai_tools: list[dict]) -> list[str]:
        q = query.lower()
        names = {tool["function"]["name"] for tool in openai_tools}
        hints: list[str] = []

        def add(*items: str) -> None:
            for item in items:
                if item in names and item not in hints:
                    hints.append(item)

        if "unread" in q:
            add("get_unread_emails")
        if (
            "email" in q
            or "mail" in q
            or "security code" in q
            or "password" in q
        ):
            add("search_emails", "get_received_emails")
        if (
            "calendar" in q
            or "appointment" in q
            or "meeting" in q
            or "lunch" in q
            or "event" in q
        ):
            add("get_day_calendar_events", "search_calendar_events")
        if (
            "file" in q
            or "drive" in q
            or "doc" in q
            or "feedback" in q
            or "packing" in q
            or "recipe" in q
        ):
            add("search_files_by_filename", "search_files", "list_files")
        if "contact" in q or "who" in q:
            add("search_contacts_by_name", "search_contacts_by_email")
        if "delete" in q:
            add("delete_file", "delete_email")
        return hints[:3]

    def _finish(
        self, chat: list[dict], log: list[ChatMessage], reason: str
    ) -> None:
        self.stop_reason = reason
        if not self.answer:
            if reason == "max_tools":
                self.answer = "I could not complete the request because the tool-call limit was reached."
            elif reason == "max_model_calls":
                self.answer = "I could not complete the request because the model-call limit was reached."
            elif reason == "plan_invalid":
                self.answer = (
                    "I could not create a valid plan for this request."
                )
            else:
                self.answer = "I could not complete the request."
        self._text_message("assistant", self.answer, chat, log)

    def _react(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env,
        chat: list[dict],
        log: list[ChatMessage],
        openai_tools: list[dict],
    ) -> None:
        while True:
            if self.totals["model_calls"] >= self.max_calls:
                self._finish(chat, log, "max_model_calls")
                return
            tool_limit_hit = self.totals["tool_calls"] >= self.max_tools
            model_limit_hit = self.totals["model_calls"] == self.max_calls - 1
            final_call = tool_limit_hit or model_limit_hit
            result = self._model(
                chat,
                "final" if final_call else "react",
                None if final_call else openai_tools,
            )
            if result is None:
                self._finish(chat, log, self.stop_reason)
                return
            if final_call:
                self.answer = result.content.strip()
                reason = (
                    "max_tools"
                    if tool_limit_hit
                    else ("max_model_calls" if model_limit_hit else "text")
                )
                self._finish(chat, log, reason)
                return
            if result.tool_calls and self.totals["tool_calls"] < self.max_tools:
                self._assistant_calls(result.tool_calls, chat, log)
                for call in result.tool_calls:
                    if self.totals["tool_calls"] >= self.max_tools:
                        break
                    self._execute(call, runtime, env, chat, log)
                continue
            if result.content.strip():
                refusal = any(
                    phrase in result.content.lower()
                    for phrase in (
                        "i don't have the capability",
                        "i cannot access",
                        "i can't access",
                        "i do not have access",
                    )
                )
                if (
                    refusal
                    and self.totals["tool_calls"] == 0
                    and self.totals["model_calls"] < self.max_calls
                ):
                    self._text_message(
                        "assistant", result.content.strip(), chat, log
                    )
                    self._text_message(
                        "user",
                        "Use the available tools in this environment to answer the request. Do not refuse because of missing access.",
                        chat,
                        log,
                    )
                    forced = self._model(chat, "react_forced", openai_tools)
                    if forced is None:
                        self._finish(chat, log, self.stop_reason)
                        return
                    if (
                        forced.tool_calls
                        and self.totals["tool_calls"] < self.max_tools
                    ):
                        self._assistant_calls(forced.tool_calls, chat, log)
                        for call in forced.tool_calls:
                            if self.totals["tool_calls"] >= self.max_tools:
                                break
                            self._execute(call, runtime, env, chat, log)
                        continue
                    hint_tools = self._guess_tools(query, openai_tools)
                    executed_hint = False
                    for hint in hint_tools:
                        if self.totals["model_calls"] >= self.max_calls:
                            break
                        self._text_message(
                            "user",
                            f"Call the tool {hint} with appropriate arguments to solve the request.",
                            chat,
                            log,
                        )
                        hinted = self._model(
                            chat,
                            f"react_hint_{hint}",
                            [t for t in openai_tools if t["function"]["name"] == hint],
                            hint,
                        )
                        if (
                            hinted
                            and hinted.tool_calls
                            and self.totals["tool_calls"] < self.max_tools
                        ):
                            self._assistant_calls(hinted.tool_calls, chat, log)
                            for call in hinted.tool_calls:
                                if self.totals["tool_calls"] >= self.max_tools:
                                    break
                                self._execute(call, runtime, env, chat, log)
                            executed_hint = True
                            break
                    if executed_hint:
                        continue
                    if forced.content.strip():
                        self.answer = forced.content.strip()
                        self._text_message("assistant", self.answer, chat, log)
                        return
                    continue
                self.answer = result.content.strip()
                self._text_message("assistant", self.answer, chat, log)
                return
            self._text_message(
                "assistant", "I need more information.", chat, log
            )

    def _plan(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env,
        chat: list[dict],
        log: list[ChatMessage],
        openai_tools: list[dict],
    ) -> None:
        names = [tool["function"]["name"] for tool in openai_tools]
        plan_prompt = prompts.render(
            prompts.load(self.cfg["prompts"]["plan"]),
            query=query,
            tools=prompts.tools_description(openai_tools),
        )
        self._text_message("user", plan_prompt, chat, log)
        result = self._model(
            chat,
            "plan",
            [tools.plan_tool(names)],
            "submit_plan",
        )
        plan_args = (
            result.tool_calls[0]["args"] if result and result.tool_calls else {}
        )
        self.plan = tools.parse_plan(plan_args, names) or []
        if not self.plan:
            self.answer = "I could not create a valid plan."
            self._finish(chat, log, "plan_invalid")
            return

        plan_text = "\n".join(
            f"{i + 1}. {step['tool']}: {step['purpose']}"
            for i, step in enumerate(self.plan)
        )
        step_prompt_template = prompts.load(self.cfg["prompts"]["step"])
        completed: list[dict] = []
        for index, step in enumerate(self.plan):
            if self.totals["model_calls"] >= self.max_calls:
                self._finish(chat, log, "max_model_calls")
                return
            rendered = prompts.render(
                step_prompt_template,
                query=query,
                plan=plan_text,
                results=prompts.results_description(completed)
                or "No results yet.",
                step_number=index + 1,
                n_steps=len(self.plan),
                purpose=step["purpose"],
                tool=step["tool"],
            )
            self._text_message("user", rendered, chat, log)
            allowed = [
                tool
                for tool in openai_tools
                if tool["function"]["name"] == step["tool"]
            ]
            result = self._model(
                chat,
                f"step_{index + 1}_{step['tool']}",
                allowed,
                step["tool"],
            )
            if result is None:
                self._finish(chat, log, self.stop_reason)
                return
            if result.tool_calls and tools.step_call_valid(
                result.tool_calls[0], step["tool"]
            ):
                call = result.tool_calls[0]
                self._assistant_calls([call], chat, log)
                output = self._execute(call, runtime, env, chat, log)
                completed.append({"tool": step["tool"], "result": output})
            else:
                completed.append(
                    {
                        "tool": step["tool"],
                        "result": f"Step failed: {result.content or 'no tool call'}",
                    }
                )

        final_prompt = prompts.render(
            prompts.load(self.cfg["prompts"]["final"]),
            query=query,
            results=prompts.results_description(completed) or "No results.",
        )
        self._text_message("user", final_prompt, chat, log)
        result = self._model(chat, "final", None)
        self.answer = result.content.strip() if result else ""
        self._finish(chat, log, "text" if self.answer else self.stop_reason)

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env | None = None,
        messages: Sequence[ChatMessage] = (),
        extra_args: dict | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        env = EmptyEnv() if env is None else env
        args = dict(extra_args or {})
        chat = [{"role": "system", "content": self.system}]
        log: list[ChatMessage] = [
            ChatSystemMessage(role="system", content=[text(self.system)])
        ]
        self._text_message("user", query, chat, log)
        openai_tools = tools.to_openai_tools(runtime.functions)
        if self.mode == "react":
            self._react(query, runtime, env, chat, log, openai_tools)
        elif self.mode == "plan":
            self._plan(query, runtime, env, chat, log, openai_tools)
        else:
            raise ValueError(f"Unknown agent mode: {self.mode}")
        return query, runtime, env, log, args
