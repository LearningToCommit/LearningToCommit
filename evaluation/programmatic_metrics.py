#!/usr/bin/env python3
"""LLM-free metrics comparing two solve variants against the oracle diff.

Per task and on average:
    File IoU          Jaccard overlap between files changed by the agent and by the oracle commit.
    Trajectory steps  number of LLM responses that issued tool calls while solving.
    Line deviation    (agent_lines - oracle_lines) / max(1, oracle_lines), where lines are
                      added + removed lines; reported as signed mean and mean absolute value.

Reads <exp-dir>/02_resolution/<variant>/<task_id>/{changes.diff, trajectory.json} and the
oracle diffs referenced by the tasks file.

Usage:
    python evaluation/programmatic_metrics.py \\
        --exp-dir output/toy/claude-sonnet-4-6 \\
        --tasks-file examples/toy/build/benchmark_tasks.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.agent import count_tool_steps  # noqa: E402
from workflow.output_layout import step2_task_records_dir, step2_variant_dir, step3_dir  # noqa: E402
from workflow.pipeline import load_items  # noqa: E402


def extract_files_from_diff(diff_text: str) -> set[str]:
    return {m.group(2) for m in re.finditer(r"^diff --git a/(.+?) b/(.+?)$", diff_text, re.MULTILINE)}


def count_diff_lines(diff_text: str) -> int:
    changed = 0
    for line in diff_text.splitlines():
        if (line.startswith("+") and not line.startswith("+++")) or (line.startswith("-") and not line.startswith("---")):
            changed += 1
    return changed


def line_deviation_ratio(agent_lines: int, oracle_lines: int) -> float:
    return (agent_lines - oracle_lines) / max(1, oracle_lines)


def compute_iou(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def count_steps(trajectory_path: Path) -> int | None:
    if not trajectory_path.exists():
        return None
    return count_tool_steps(json.loads(trajectory_path.read_text(encoding="utf-8")).get("messages", []))


def oracle_diff_for(task: dict) -> str:
    if task.get("oracle_diff"):
        return read_text(Path(task["oracle_diff"]))
    return task.get("oracle_diff_text", "")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="LLM-free metrics: File IoU, trajectory steps, line deviation")
    parser.add_argument("--exp-dir", required=True, help="output/<signature>/<model>")
    parser.add_argument("--tasks-file", required=True, help="the benchmark_tasks.jsonl used for the run")
    parser.add_argument("--variants", default="with_skill,baseline",
                        help="two Step 2 variants to compare, e.g. with_skill,baseline_forced")
    parser.add_argument("--output", default=None, help="JSON output (default: <exp-dir>/03_evaluation/programmatic.json)")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    variant_a, variant_b = [v.strip() for v in args.variants.split(",")]
    tasks = {t["id"]: t for t in load_items(args.tasks_file)}
    task_ids = sorted(p.stem for p in step2_task_records_dir(exp_dir).glob("*.json"))
    if not task_ids:
        sys.exit(f"no task records under {step2_task_records_dir(exp_dir)}")

    print(f"A = {variant_a}, B = {variant_b}")
    print(f"{'Task':<25} {'A-IoU':>6} {'B-IoU':>6} {'A-Step':>7} {'B-Step':>7} {'A-Dev':>6} {'B-Dev':>6} {'OLines':>7}")
    print("-" * 80)
    per_task = []
    for tid in task_ids:
        if tid not in tasks:
            print(f"  {tid:<23} (not in tasks file, skipped)")
            continue
        oracle = oracle_diff_for(tasks[tid])
        oracle_files, oracle_lines = extract_files_from_diff(oracle), count_diff_lines(oracle)
        row = {"task_id": tid, "oracle_lines": oracle_lines}
        for tag, variant in (("a", variant_a), ("b", variant_b)):
            stage = step2_variant_dir(exp_dir, variant, tid)
            diff = read_text(stage / "changes.diff")
            row[f"{tag}_iou"] = compute_iou(extract_files_from_diff(diff), oracle_files)
            row[f"{tag}_steps"] = count_steps(stage / "trajectory.json")
            row[f"{tag}_line_deviation"] = line_deviation_ratio(count_diff_lines(diff), oracle_lines)
        per_task.append(row)
        fmt = lambda s: "-" if s is None else str(s)  # noqa: E731
        print(f"  {tid:<23} {row['a_iou']:>5.0%} {row['b_iou']:>5.0%} {fmt(row['a_steps']):>7} {fmt(row['b_steps']):>7}"
              f" {row['a_line_deviation']:>5.1f} {row['b_line_deviation']:>5.1f} {oracle_lines:>7}")

    summary = {}
    for tag in ("a", "b"):
        summary[f"{tag}_iou"] = _mean([r[f"{tag}_iou"] for r in per_task])
        summary[f"{tag}_steps"] = _mean([r[f"{tag}_steps"] for r in per_task if r[f"{tag}_steps"] is not None])
        devs = [r[f"{tag}_line_deviation"] for r in per_task]
        summary[f"{tag}_line_deviation"] = _mean(devs)
        summary[f"{tag}_abs_line_deviation"] = _mean([abs(d) for d in devs])
    print("-" * 80)
    print(f"  {'AVERAGE':<23} {summary['a_iou']:>5.0%} {summary['b_iou']:>5.0%} {summary['a_steps']:>7.1f} "
          f"{summary['b_steps']:>7.1f} {summary['a_line_deviation']:>5.1f} {summary['b_line_deviation']:>5.1f}")
    print(f"\n  File IoU delta (A-B):     {summary['a_iou'] - summary['b_iou']:+.1%}")
    print(f"  Steps delta (A-B):        {summary['a_steps'] - summary['b_steps']:+.1f}")
    print(f"  Line deviation (signed):  A {summary['a_line_deviation']:+.2f}  B {summary['b_line_deviation']:+.2f}"
          "  (>0 wrote more than the oracle, <0 less)")
    print(f"  Mean |line deviation|:    A {summary['a_abs_line_deviation']:.2f}   B {summary['b_abs_line_deviation']:.2f}"
          "   (closer to 0 is better)")

    output = Path(args.output) if args.output else step3_dir(exp_dir) / "programmatic.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "exp_dir": str(exp_dir), "variants": [variant_a, variant_b],
        "task_count": len(per_task), "per_task": per_task, "summary": summary,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Saved: {output}")


if __name__ == "__main__":
    main()
