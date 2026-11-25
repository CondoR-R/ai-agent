"""Дописать недостающие эпизоды в каталог запуска ЛР4."""

import argparse
import json
import time
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suites

from ai_agent import files, metadata
from ai_agent.labs.lr4 import data, evaluation
from ai_agent.labs.lr4.agent import ToolAgent
from ai_agent.labs.lr4.client import ModelClient
from ai_agent.labs.lr4.runner import episode_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    cfg = files.read_yaml(run_dir / files.CONFIG_FILENAME)
    results_path = run_dir / files.RESULTS_FILENAME
    records = []
    if results_path.exists():
        records = [
            json.loads(line)
            for line in results_path.read_text().splitlines()
            if line.strip()
        ]
    seen = {(r["task_id"], r["mode"]) for r in records}
    client = ModelClient()
    suite = get_suites(cfg["agentdojo"]["benchmark_version"])[
        cfg["agentdojo"]["suite"]
    ]
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    task_ids = data.task_ids(cfg["run"]["split"])
    for index, task_id in enumerate(task_ids):
        task = suite.get_user_task_by_id(task_id)
        modes = list(cfg["run"]["modes"])
        if cfg["run"].get("alternate") and index % 2:
            modes.reverse()
        for mode in modes:
            if (task_id, mode) in seen:
                continue
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
            files.append_jsonl(results_path, record)
            records.append(record)
            print(
                f"[resume] {task_id} {mode}: {'ok' if utility else 'fail'} "
                f"({agent.stop_reason})",
                flush=True,
            )

    files.write_json(
        run_dir / files.METRICS_FILENAME, evaluation.build_metrics(records)
    )
    meta = files.read_json(run_dir / files.METADATA_FILENAME)
    if meta.get("status") != "completed":
        metadata.update(run_dir, "completed")
    print(f"[resume] done, records={len(records)}")


if __name__ == "__main__":
    main()
