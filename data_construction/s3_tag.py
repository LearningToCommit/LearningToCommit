#!/usr/bin/env python3
"""S3: assign one S2 category to every suitable commit.

Usage:
    python data_construction/s3_tag.py --input .../s1_scanned.jsonl --categories .../s2_categories.json --repo-dir /path/to/repo
Output: s3_tagged.jsonl next to the input.
"""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

from utils import call_json, get_commit_diff, load_jsonl, load_prompt, save_jsonl


async def tag_one(sem, commit, repo_dir, template, categories_json, model):
    async with sem:
        diff = get_commit_diff(commit["sha"], repo_dir)
        if diff is None:
            commit["category"] = "uncategorized"
            return commit
        prompt = (
            template.replace("{subject}", commit.get("subject", ""))
            .replace("{reason}", commit.get("reason", ""))
            .replace("{patch}", diff)
            .replace("{categories_json}", categories_json)
        )
        result = await call_json(prompt, model=model)
        commit["category"] = result.get("category", "uncategorized") if result else "uncategorized"
        return commit


async def main():
    parser = argparse.ArgumentParser(description="S3: tag commits with categories")
    parser.add_argument("--input", required=True, help="s1_scanned.jsonl")
    parser.add_argument("--categories", required=True, help="s2_categories.json")
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--output", default=None, help="default: <input dir>/s3_tagged.jsonl")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--model", default=None, help="default: LTC_MODEL")
    args = parser.parse_args()

    input_path = Path(args.input)
    output = Path(args.output or input_path.parent / "s3_tagged.jsonl")
    commits = load_jsonl(input_path)
    categories = json.loads(Path(args.categories).read_text(encoding="utf-8"))["categories"]
    suitable = [c for c in commits if c.get("suitable")]
    repo_dir = str(Path(args.repo_dir).resolve())

    sem = asyncio.Semaphore(args.concurrency)
    template = load_prompt("classify.txt")
    categories_json = json.dumps(categories, ensure_ascii=False, indent=2)
    await tqdm_asyncio.gather(*(tag_one(sem, c, repo_dir, template, categories_json, args.model) for c in suitable))
    save_jsonl(commits, output)

    counts = Counter((c["category"], c["split"]) for c in suitable)
    print(f"{'category':<40} {'learning':>9} {'test':>6}")
    for cat in sorted({c for c, _ in counts}):
        print(f"{cat:<40} {counts[(cat, 'learning')]:>9} {counts[(cat, 'test')]:>6}")
    print(f"Output: {output}\nNext: python data_construction/s4_sample.py --input {output} --n-learn 10 --n-test 10")


if __name__ == "__main__":
    asyncio.run(main())
