#!/usr/bin/env python3
"""Pairwise A/B blind judging of the with_skill and baseline patches.

For each task, the two patches are shown as "Developer A" and "Developer B" together with
the task description and the oracle diff; the judge compares them on four dimensions
(system_prompts/judge_pairwise.txt): scope alignment, logic similarity, redundancy /
hallucination, and code style. Each task is judged --rounds times in an ABBA order to
control for position bias.

Each judgement is two plain Claude calls (Anthropic Messages API):
    stage 1 (LTC_JUDGE_MODEL): free-form analysis ending in per-dimension verdicts;
    stage 2 (LTC_JUDGE_EXTRACTOR_MODEL): converts the analysis into JSON.

Usage:
    python evaluation/pairwise_judge.py \\
        --exp-dir output/toy/claude-sonnet-4-6 \\
        --tasks-file examples/toy/build/benchmark_tasks.jsonl
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import JUDGE_EXTRACTOR_MODEL, JUDGE_MODEL, REPO_ROOT  # noqa: E402
from modules.helpers import find_first_json, load_jsonl, save_jsonl  # noqa: E402
from modules.llm import complete_text  # noqa: E402
from workflow.output_layout import step2_task_records_dir, step2_variant_dir, step3_dir  # noqa: E402
from workflow.pipeline import load_items  # noqa: E402

DEFAULT_PROMPT = REPO_ROOT / "system_prompts" / "judge_pairwise.txt"
EMPTY_DIFF = "(empty diff - no changes produced)"
ABBA = [True, False, False, True]  # True = with_skill is shown as A

JUDGE_JSON_SCHEMA = """{
  "Q1_scope_alignment": {"analysis": "...", "winner": "A_WIN | B_WIN | TIE"},
  "Q2_logic_similarity": {"analysis": "...", "winner": "A_WIN | B_WIN | TIE"},
  "Q3_redundancy_hallucination": {"analysis": "...", "winner": "A_WIN | B_WIN | TIE"},
  "Q4_code_style": {"analysis": "...", "winner": "A_WIN | B_WIN | TIE"},
  "overall_summary": "...",
  "final_winner": "A_WIN | B_WIN | TIE"
}"""

STRUCTIFY_PROMPT = """You are a JSON extractor. Given the analysis below, output a JSON object
matching the requested schema. Copy the verdicts exactly as stated in the analysis.

Analysis:
{analysis}

Required JSON schema:
{schema}

Output JSON only, no extra prose."""

DIMENSIONS = [
    ("Q1_scope_alignment", "Q1 Scope"),
    ("Q2_logic_similarity", "Q2 Logic"),
    ("Q3_redundancy_hallucination", "Q3 Redund"),
    ("Q4_code_style", "Q4 Style"),
]


def read_diff(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    return text if text.strip() else EMPTY_DIFF


def resolve_winner(raw: str | None, label_a: str, label_b: str) -> str:
    return {"A_WIN": label_a, "B_WIN": label_b, "TIE": "tie"}.get((raw or "").strip(), "error")


async def judge_once(sem, prompt_template, task_id, query, oracle, skill_diff, base_diff, skill_is_a, round_idx) -> dict:
    diff_a, diff_b = (skill_diff, base_diff) if skill_is_a else (base_diff, skill_diff)
    label_a, label_b = ("skill", "baseline") if skill_is_a else ("baseline", "skill")
    user_content = (
        f"## Task description\n\n{query}\n\n"
        f"## Oracle Diff (reference)\n\n```diff\n{oracle}\n```\n\n"
        f"## Developer A diff\n\n```diff\n{diff_a}\n```\n\n"
        f"## Developer B diff\n\n```diff\n{diff_b}\n```"
    )
    record = {"task_id": task_id, "round": round_idx, "skill_is_a": skill_is_a}
    async with sem:
        analysis = await complete_text(f"{prompt_template}\n\n{user_content}", model=JUDGE_MODEL, temperature=0.7)
        if not analysis:
            return {**record, "actual_winner": "error", "reason": "analysis call failed"}
        extracted = await complete_text(
            STRUCTIFY_PROMPT.format(analysis=analysis, schema=JUDGE_JSON_SCHEMA),
            model=JUDGE_EXTRACTOR_MODEL, temperature=0.0,
        )
    result = find_first_json(extracted or "")
    if not result:
        return {**record, "actual_winner": "error", "reason": "could not parse judge JSON", "analysis": analysis}
    actual = resolve_winner(result.get("final_winner"), label_a, label_b)
    dims = {}
    for key, _ in DIMENSIONS:
        section = result.get(key)
        dims[key] = resolve_winner(section.get("winner") if isinstance(section, dict) else None, label_a, label_b)
    return {**record, "raw_winner": result.get("final_winner"), "actual_winner": actual,
            "dim_winners": dims, "judge_result": result}


def summarize(judgments: list[dict]) -> dict:
    valid = [j for j in judgments if j["actual_winner"] != "error"]
    n = len(valid)

    def rates(winners: list[str]) -> dict:
        m = len([w for w in winners if w != "error"])
        out = {k: winners.count(k) for k in ("skill", "baseline", "tie")}
        out.update({f"{k}_rate": (winners.count(k) / m if m else 0.0) for k in ("skill", "baseline", "tie")})
        out["valid"] = m
        return out

    summary = rates([j["actual_winner"] for j in valid])
    summary["errors"] = len(judgments) - n
    summary["per_dimension"] = {key: rates([j["dim_winners"][key] for j in valid]) for key, _ in DIMENSIONS}
    summary["per_task"] = {}
    for tid in sorted({j["task_id"] for j in judgments}):
        summary["per_task"][tid] = rates([j["actual_winner"] for j in judgments if j["task_id"] == tid])
    return summary


def print_summary(summary: dict):
    print("=" * 56)
    print(f"  Pairwise results ({summary['valid']} valid judgements, {summary['errors']} errors)")
    print("=" * 56)
    for k in ("skill", "baseline", "tie"):
        print(f"    {k:<9} {summary[k]:>3}  ({summary[f'{k}_rate']:.1%})")
    print(f"\n  {'dimension':<12} {'skill':>12} {'baseline':>12} {'tie':>12}")
    for key, label in DIMENSIONS:
        d = summary["per_dimension"][key]
        print(f"  {label:<12} {d['skill']:>4} ({d['skill_rate']:>4.0%}) {d['baseline']:>4} ({d['baseline_rate']:>4.0%})"
              f" {d['tie']:>4} ({d['tie_rate']:>4.0%})")
    print("\n  per task (skill / baseline / tie):")
    for tid, d in summary["per_task"].items():
        print(f"    {tid}: {d['skill']} / {d['baseline']} / {d['tie']}")


async def main():
    parser = argparse.ArgumentParser(description="Pairwise A/B blind judge (with_skill vs. baseline)")
    parser.add_argument("--exp-dir", required=True, help="output/<signature>/<model>")
    parser.add_argument("--tasks-file", required=True, help="the benchmark_tasks.jsonl used for the run")
    parser.add_argument("--baseline-variant", default="baseline", help="baseline or baseline_forced")
    parser.add_argument("--rounds", type=int, default=4, help="judgements per task (ABBA order)")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT), help="judge prompt file")
    parser.add_argument("--output-dir", default=None, help="default: <exp-dir>/03_evaluation/pairwise")
    parser.add_argument("--rerun", action="store_true", help="ignore cached judgements")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    tasks = {t["id"]: t for t in load_items(args.tasks_file)}
    task_ids = sorted(p.stem for p in step2_task_records_dir(exp_dir).glob("*.json") if p.stem in tasks)
    if not task_ids:
        sys.exit(f"no task records under {step2_task_records_dir(exp_dir)} that match {args.tasks_file}")
    prompt_template = Path(args.prompt).read_text(encoding="utf-8")

    out_dir = Path(args.output_dir) if args.output_dir else step3_dir(exp_dir) / "pairwise"
    judgments_path, summary_path = out_dir / "judgments.jsonl", out_dir / "summary.json"
    settings = {"judge_model": JUDGE_MODEL, "extractor_model": JUDGE_EXTRACTOR_MODEL,
                "prompt": str(args.prompt), "baseline_variant": args.baseline_variant}
    cached = []
    if not args.rerun and summary_path.exists():
        if json.loads(summary_path.read_text()).get("settings") == settings:
            cached = [j for j in load_jsonl(judgments_path) if j["actual_winner"] != "error"]
    done = {(j["task_id"], j["round"]) for j in cached}

    sem = asyncio.Semaphore(args.concurrency)
    jobs = []
    for tid in task_ids:
        task = tasks[tid]
        oracle = Path(task["oracle_diff"]).read_text(encoding="utf-8", errors="replace") if task.get("oracle_diff") \
            else task.get("oracle_diff_text", "")
        skill_diff = read_diff(step2_variant_dir(exp_dir, "with_skill", tid) / "changes.diff")
        base_diff = read_diff(step2_variant_dir(exp_dir, args.baseline_variant, tid) / "changes.diff")
        for r in range(args.rounds):
            if (tid, r) not in done:
                jobs.append(judge_once(sem, prompt_template, tid, task["synthetic_query"], oracle,
                                       skill_diff, base_diff, ABBA[r % len(ABBA)], r))
    print(f"Judge: {JUDGE_MODEL} (analysis) -> {JUDGE_EXTRACTOR_MODEL} (JSON); "
          f"{len(jobs)} new judgements, {len(cached)} cached", flush=True)
    new = await asyncio.gather(*jobs)
    judgments = sorted(cached + list(new), key=lambda j: (j["task_id"], j["round"]))

    summary = summarize(judgments)
    print_summary(summary)
    save_jsonl(judgments_path, judgments)
    summary_path.write_text(json.dumps({"settings": settings, "rounds": args.rounds, "summary": summary},
                                       indent=2) + "\n", encoding="utf-8")
    print(f"\n  Saved: {judgments_path}\n         {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
