#!/usr/bin/env python3
"""Run the Learning to Commit pipeline on a prepared benchmark_tasks.jsonl.

Step 1 (onboarding): for every curriculum of learning commits, learn sequentially
    (blind attempt -> contrastive reflection against the oracle diff -> skill update).
Step 2 (resolution): solve every test task with the learned skills (with_skill) and,
    with --run-baseline, without them (baseline).
Evaluation is offline: see evaluation/programmatic_metrics.py and evaluation/pairwise_judge.py.

Example:
    python workflow/run_sequential.py \\
        --input-file examples/toy/build/benchmark_tasks.jsonl \\
        --signature toy --run-baseline

Outputs go to output/<signature>/<model>/. Rerunning the same command resumes.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm  # noqa: E402

from modules.config import CONCURRENCY, DEFAULT_MODEL  # noqa: E402
from modules.helpers import load_json_if_exists, upsert_jsonl_by_key  # noqa: E402
from workflow.config import DEFAULT_FORCED_MIN_STEPS, DEFAULT_OUTPUT_ROOT  # noqa: E402
from workflow.learning import get_curriculum_key, group_learning_curricula, learn_or_load  # noqa: E402
from workflow.output_layout import (  # noqa: E402
    initialize_experiment_layout,
    step1_dir,
    step2_dir,
    step2_task_records_index_path,
)
from workflow.pipeline import load_items, run_single_item_with_semaphore  # noqa: E402


def total_cost(output_root: Path) -> float:
    """Sum of the per-phase costs reported by the SDK (learning iterations + solves)."""
    cost = 0.0
    for summary in list(step1_dir(output_root).rglob("summary.json")) + list(step2_dir(output_root).rglob("summary.json")):
        data = load_json_if_exists(summary) or {}
        cost += float(data.get("cost_usd") or 0.0)
    return cost


def parse_args():
    parser = argparse.ArgumentParser(description="Learning to Commit: sequential learning + solving")
    parser.add_argument("--input-file", required=True, help="benchmark_tasks.jsonl (from scripts/s1_prepare_benchmark.py or the toy example)")
    parser.add_argument("--signature", default="run", help="experiment name; outputs go to <output-dir>/<signature>/<model>/")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model for all phases (default: {DEFAULT_MODEL}, env LTC_MODEL)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT), help="root output directory")
    parser.add_argument("--max-items", type=int, default=0, help="only run the first N tasks (0 = all)")
    parser.add_argument("--max-learning-commits", type=int, default=0, help="only learn from the first N learning commits (0 = all)")
    parser.add_argument("--skill-dir", default="", help="skip learning and solve with this existing skill directory")
    parser.add_argument("--run-baseline", action="store_true", help="also solve every task without skills")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY, help="test tasks solved in parallel (env LTC_CONCURRENCY)")
    parser.add_argument(
        "--learning-mode", choices=["attempt_reflect", "observe_only"], default="attempt_reflect",
        help="attempt_reflect: the method; observe_only: ablation that skips the blind attempt and "
             "studies the oracle diff directly",
    )
    parser.add_argument(
        "--baseline-variant", choices=["baseline", "forced"], default="baseline",
        help="forced: forced-exploration baseline (exploration prompt + minimum tool steps), "
             "written to 02_resolution/baseline_forced/",
    )
    parser.add_argument("--forced-min-steps", type=int, default=DEFAULT_FORCED_MIN_STEPS,
                        help="minimum tool steps for --baseline-variant forced")
    return parser.parse_args()


async def main():
    args = parse_args()
    output_root = Path(args.output_dir) / args.signature / args.model.replace("/", "_")
    initialize_experiment_layout(output_root)

    items = load_items(args.input_file, args.max_items)
    for item in items:
        item["max_learning_commits"] = args.max_learning_commits
    print(f"Loaded {len(items)} tasks, model={args.model}, output -> {output_root}", flush=True)

    # Step 1: learning, one run per curriculum (curricula are learned concurrently).
    if args.skill_dir:
        skill_dir = Path(args.skill_dir).resolve()
        if not (skill_dir / "SKILL.md").exists():
            raise FileNotFoundError(f"--skill-dir has no SKILL.md: {skill_dir}")
        for item in items:
            item["_skill_dir"] = skill_dir
    else:
        curricula = group_learning_curricula(items)
        print(f"Found {len(curricula)} learning curriculum(s)", flush=True)

        async def _learn(key, group):
            litem = group["learning_item"]
            print(f"Learning curriculum {key[:8]}: {len(litem['learning_commits'])} commits, "
                  f"{len(group['task_ids'])} test tasks", flush=True)
            return key, await learn_or_load(litem, args.model, output_root, key, mode=args.learning_mode)

        results = await asyncio.gather(*(_learn(k, g) for k, g in curricula.items()))
        skill_dirs = dict(results)
        for item in items:
            item["_skill_dir"] = skill_dirs.get(get_curriculum_key(item))

    # Step 2: solving.
    sem = asyncio.Semaphore(max(1, args.concurrency))
    tasks = [
        asyncio.create_task(run_single_item_with_semaphore(
            sem, item, args.model, output_root, args.run_baseline,
            baseline_variant=args.baseline_variant, forced_min_steps=args.forced_min_steps,
        ))
        for item in items
    ]
    index_path = step2_task_records_index_path(output_root)
    for finished in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="solving", file=sys.stdout):
        upsert_jsonl_by_key(index_path, await finished, "task_id")

    print(f"Done. Output: {output_root.resolve()}")
    print(f"Total reported cost (all phases so far): ${total_cost(output_root):.2f}")


if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    print(f"Time: {time.time() - start:.1f}s")
