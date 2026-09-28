#!/usr/bin/env python3
"""S2: cluster the suitability rationales of S1 into 4-8 development categories.

Usage:
    python data_construction/s2_cluster.py --input data_construction/runs/<repo>/s1_scanned.jsonl
Output: s2_categories.json (+ s2_analysis.md with the model's free-form analysis) next to the input.
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

from utils import analyze_then_extract, load_jsonl, load_prompt


async def main():
    parser = argparse.ArgumentParser(description="S2: derive development categories")
    parser.add_argument("--input", required=True, help="s1_scanned.jsonl")
    parser.add_argument("--output", default=None, help="default: <input dir>/s2_categories.json")
    parser.add_argument("--sample", type=int, default=100, help="number of rationales shown to the model")
    parser.add_argument("--model", default=None, help="default: LTC_MODEL")
    args = parser.parse_args()

    input_path = Path(args.input)
    output = Path(args.output or input_path.parent / "s2_categories.json")
    reasons = [
        c["reason"] for c in load_jsonl(input_path)
        if c.get("suitable") and c.get("reason") and c["reason"] not in ("diff_too_large_or_unavailable", "llm_failed")
    ]
    if not reasons:
        raise SystemExit("no suitable commits with a rationale; run S1 first")
    random.seed(42)
    sample = random.sample(reasons, min(args.sample, len(reasons)))

    prompt = (
        f"{load_prompt('wash_cluster.txt')}\n\n"
        f"========== {len(sample)} rationales sampled from {len(reasons)} suitable commits ==========\n\n"
        + "\n".join(f"- {r}" for r in sample)
        + "\n\n========== end of sample ==========\n\n"
        "Think through all of the rationales above, then propose 4-8 mutually distinct categories. "
        "For each category give a short snake_case label and a one-paragraph description."
    )
    result, analysis = await analyze_then_extract(
        prompt, '{"categories": [{"name": "snake_case_label", "description": "what the category covers"}]}',
        model=args.model,
    )
    (output.parent / "s2_analysis.md").write_text(analysis, encoding="utf-8")
    if not result or "categories" not in result:
        raise SystemExit("could not extract categories; see s2_analysis.md")
    output.write_text(json.dumps({"categories": result["categories"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    for cat in result["categories"]:
        print(f"  {cat['name']}: {cat['description']}")
    print(f"Output: {output}\nNext: python data_construction/s3_tag.py --input {input_path} --categories {output} --repo-dir <repo>")


if __name__ == "__main__":
    asyncio.run(main())
