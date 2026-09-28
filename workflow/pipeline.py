"""Per-task orchestration: solve each test task with the learned skills and, optionally, as a baseline."""

import asyncio
import json
from pathlib import Path

from modules.helpers import load_json_if_exists, save_json
from workflow.output_layout import build_task_paths
from workflow.phases import solve


async def run_single_item(
    item: dict,
    model_id: str,
    output_root: Path,
    run_baseline: bool = False,
    baseline_variant: str = "baseline",
    forced_min_steps: int | None = None,
) -> dict:
    """Run with_skill and (optionally) baseline solves concurrently; write a per-task record.

    item["_skill_dir"] holds the curriculum's final skill directory (set by run_sequential).
    baseline_variant="forced" runs the forced-exploration baseline into baseline_forced/.
    """
    task_id = str(item["id"])
    paths = build_task_paths(output_root, task_id)

    cached = load_json_if_exists(paths["record_path"])
    if cached is not None:
        baseline_ok = cached["solve_results"].get("baseline") is not None and (
            cached.get("baseline_variant", "baseline") == baseline_variant
        )
        if not run_baseline or baseline_ok:
            print(f"[task:cached] task={task_id}", flush=True)
            return cached

    skill_dir = item.get("_skill_dir")
    if skill_dir is None and item.get("learning_commits"):
        raise RuntimeError(f"no learned skills available for task {task_id}")

    skill_solve = asyncio.create_task(solve(item, model_id, paths["solve_with_skill_dir"], skill_dir=skill_dir))
    baseline_solve = None
    baseline_dir = paths["solve_baseline_forced_dir"] if baseline_variant == "forced" else paths["solve_baseline_dir"]
    if run_baseline:
        baseline_solve = asyncio.create_task(
            solve(item, model_id, baseline_dir, skill_dir=None, variant=baseline_variant, min_steps=forced_min_steps)
        )
    skill_traj = await skill_solve
    baseline_traj = await baseline_solve if baseline_solve else None

    def _brief(traj):
        return {"status": traj.get("status"), "cost_usd": traj.get("cost_usd")} if traj else None

    record = {
        "task_id": task_id,
        "status": "completed",
        "solve_results": {"with_skill": _brief(skill_traj), "baseline": _brief(baseline_traj)},
        "baseline_variant": baseline_variant,
        "artifacts": {
            "with_skill_dir": str(paths["solve_with_skill_dir"]),
            "baseline_dir": str(baseline_dir) if run_baseline else None,
        },
    }
    save_json(paths["record_path"], record)
    return record


async def run_single_item_with_semaphore(sem: asyncio.Semaphore, item: dict, *args, **kwargs) -> dict:
    async with sem:
        return await run_single_item(item, *args, **kwargs)


def _resolve(path: str, base_dir: Path) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else (base_dir / p).resolve())


def load_items(input_file: str | Path, max_items: int = 0) -> list[dict]:
    """Load benchmark_tasks.jsonl; relative paths are resolved against the file's directory."""
    input_file = Path(input_file).resolve()
    base_dir = input_file.parent
    items = []
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            if max_items and len(items) >= max_items:
                break
            item = json.loads(line)
            item["id"] = str(item.get("id") or f"item_{idx}")
            for key in ("snapshot_path", "oracle_diff"):
                if item.get(key):
                    item[key] = _resolve(item[key], base_dir)
            for commit in item.get("learning_commits", []):
                for key in ("snapshot_path", "diff_path"):
                    commit[key] = _resolve(commit[key], base_dir)
            items.append(item)
    return items
