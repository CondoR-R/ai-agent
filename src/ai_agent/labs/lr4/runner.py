"""Запуск эпизодов агента на задачах AgentDojo."""

import json
import time
from importlib.metadata import version
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suites

from ai_agent import files, metadata, tracking
from ai_agent.labs.lr4 import data, evaluation
from ai_agent.labs.lr4.agent import ToolAgent
from ai_agent.labs.lr4.client import ModelClient


def relative(path: Path) -> str:
    try:
        return (
            path.resolve().relative_to(files.PROJECT_ROOT.resolve()).as_posix()
        )
    except ValueError:
        return path.as_posix()


def prompt_hashes(cfg: dict) -> dict[str, str]:
    return {
        name: files.get_sha256(files.PROJECT_ROOT / path)
        for name, path in cfg["prompts"].items()
    }


def setup_metadata(cfg: dict, run_dir: Path, client: ModelClient) -> None:
    agentdojo_cfg = cfg["agentdojo"]
    metadata.add(
        run_dir,
        {
            "agentdojo": {
                "package_version": version("agentdojo"),
                "benchmark_version": agentdojo_cfg["benchmark_version"],
                "suite": agentdojo_cfg["suite"],
                "server_flags": agentdojo_cfg["server_flags"],
            },
            "model": {
                "name": client.model,
                "base_url": client.base_url,
                "generation": client.generation,
                "context_tokens": cfg["model"]["context_tokens"],
            },
            "prompts_sha256": prompt_hashes(cfg),
            "tasks_sha256": files.get_sha256(
                files.PROJECT_ROOT / agentdojo_cfg["tasks_file"]
            ),
        },
    )
    tracking.log_metrics(
        {
            "max_model_calls": cfg["agent"]["max_model_calls"],
            "max_tool_calls": cfg["agent"]["max_tool_calls"],
            "max_output_tokens": cfg["agent"]["max_output_tokens"],
        }
    )


def episode_record(
    task_id: str,
    mode: str,
    agent: ToolAgent,
    utility: bool,
    prompt: str,
    trace_path: Path,
    episode_s: float,
) -> dict:
    meta = data.task_meta(task_id)
    return {
        "task_id": task_id,
        "mode": mode,
        "user_prompt": prompt,
        "utility": bool(utility),
        "answer": agent.answer,
        "stop_reason": agent.stop_reason,
        "plan": agent.plan,
        "n_model_calls": agent.totals["model_calls"],
        "n_tool_calls": agent.totals["tool_calls"],
        "n_tool_errors": sum(
            bool(step.get("error"))
            for step in agent.trace
            if step["kind"] == "tool"
        ),
        "prompt_tokens": agent.totals["prompt_tokens"],
        "completion_tokens": agent.totals["completion_tokens"],
        "estimated_input_tokens": agent.totals["estimated_input_tokens"],
        "model_latency_s": round(agent.totals["latency_s"], 4),
        "episode_s": round(episode_s, 4),
        "tool_names": [
            step["tool"] for step in agent.trace if step["kind"] == "tool"
        ],
        "trace_path": relative(trace_path),
        "task_meta": meta,
    }


def experiment_lab(cfg: dict, run_dir: Path) -> None:
    client = ModelClient()
    setup_metadata(cfg, run_dir, client)
    agentdojo_cfg = cfg["agentdojo"]
    suite = get_suites(agentdojo_cfg["benchmark_version"])[
        agentdojo_cfg["suite"]
    ]
    split = cfg["run"]["split"]
    task_ids = data.task_ids(split)
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, task_id in enumerate(task_ids):
        task = suite.get_user_task_by_id(task_id)
        modes = list(cfg["run"]["modes"])
        if cfg["run"].get("alternate") and index % 2:
            modes.reverse()
        for mode in modes:
            episode_cfg = dict(cfg)
            episode_cfg["agent"] = dict(cfg["agent"], mode=mode)
            agent = ToolAgent(client, episode_cfg)
            trace_path = traces_dir / f"{task_id}_{mode}.json"
            started = time.perf_counter()
            utility, _ = suite.run_task_with_pipeline(agent, task, None, {})
            episode_s = time.perf_counter() - started
            trace_path.write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "mode": mode,
                        "utility": bool(utility),
                        "answer": agent.answer,
                        "stop_reason": agent.stop_reason,
                        "plan": agent.plan,
                        "totals": agent.totals,
                        "steps": agent.trace,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
            record = episode_record(
                task_id,
                mode,
                agent,
                utility,
                task.PROMPT,
                trace_path,
                episode_s,
            )
            files.append_jsonl(run_dir / files.RESULTS_FILENAME, record)
            records.append(record)
            tracking.log_metrics(
                {
                    f"episode_utility.{task_id}.{mode}": int(bool(utility)),
                    f"episode_model_calls.{task_id}.{mode}": record[
                        "n_model_calls"
                    ],
                }
            )
            print(
                f"[{index + 1}/{len(task_ids)}] {task_id} {mode}: "
                f"{'ok' if utility else 'fail'} ({agent.stop_reason})",
                flush=True,
            )
    files.write_json(
        run_dir / files.METRICS_FILENAME, evaluation.build_metrics(records)
    )
