#!/usr/bin/env python3
"""S5: build a runnable benchmark from the S4 sample.

For every sampled commit: oracle diff (`git diff <sha>~1 <sha>`), parent snapshot
(`git archive <sha>~1`) and an LLM-written issue-style task description (synthetic_query).
Then assemble benchmark_tasks.jsonl in the format consumed by workflow/run_sequential.py.

Usage:
    python data_construction/s5_build.py --input .../benchmark_train10_test10.jsonl --repo-dir /path/to/repo
Output: <input dir>/<input stem>/{benchmark_tasks.jsonl, diffs/, snapshots/}
"""

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

from utils import load_jsonl, load_prompt, populate_synthetic_query, write_commit_diff, write_git_archive


def learning_record(c: dict, repo: str) -> dict:
    sha10 = c["sha"][:10]
    return {
        "sha": c["sha"], "repo": repo, "title": c.get("subject", ""), "category": c.get("category", ""),
        "synthetic_query": c.get("synthetic_query", ""),
        "diff_path": f"diffs/commit_{sha10}.diff", "snapshot_path": f"snapshots/snapshot_{sha10}.tar.gz",
    }


def build_tasks(test: list[dict], learn: list[dict], repo: str, mode: str) -> list[dict]:
    """mode="all": every test task shares all learning commits; "by_category": same category only."""
    by_cat = defaultdict(list)
    for c in learn:
        by_cat[c.get("category", "")].append(c)
    tasks = []
    for t in test:
        sha10 = t["sha"][:10]
        commits = learn if mode == "all" else by_cat.get(t.get("category", ""), [])
        tasks.append({
            "id": f"commit_{sha10}", "repo": repo, "category": t.get("category", ""),
            "synthetic_query": t.get("synthetic_query", ""),
            "oracle_diff": f"diffs/commit_{sha10}.diff", "snapshot_path": f"snapshots/snapshot_{sha10}.tar.gz",
            "learning_commits": [learning_record(c, repo) for c in commits],
        })
    return tasks


async def main():
    parser = argparse.ArgumentParser(description="S5: diffs, snapshots, task descriptions, benchmark_tasks.jsonl")
    parser.add_argument("--input", required=True, help="S4 output jsonl")
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--output-dir", default=None, help="default: <input dir>/<input stem>")
    parser.add_argument("--mode", choices=["all", "by_category"], default="all",
                        help="which learning commits each test task gets (the released benchmark uses 'all')")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--model", default=None, help="default: LTC_MODEL")
    args = parser.parse_args()

    input_path = Path(args.input)
    repo_dir = str(Path(args.repo_dir).resolve())
    repo = Path(repo_dir).name
    out = Path(args.output_dir or input_path.parent / input_path.stem)
    commits = load_jsonl(input_path)

    for c in commits:
        sha10 = c["sha"][:10]
        ok_diff = write_commit_diff(c["sha"], repo_dir, out / "diffs" / f"commit_{sha10}.diff")
        ok_snap = write_git_archive(f"{c['sha']}~1", repo_dir, out / "snapshots" / f"snapshot_{sha10}.tar.gz")
        if not (ok_diff and ok_snap):
            print(f"  warning: could not build diff/snapshot for {sha10}")

    template = load_prompt("synthetic_query.txt")
    sem = asyncio.Semaphore(args.concurrency)

    async def _query(c):
        async with sem:
            await populate_synthetic_query(c, repo_dir, template, model=args.model)

    await tqdm_asyncio.gather(*(_query(c) for c in commits))

    learn = [c for c in commits if c.get("split") == "learning"]
    test = [c for c in commits if c.get("split") == "test"]
    tasks = build_tasks(test, learn, repo, args.mode)
    with open(out / "benchmark_tasks.jsonl", "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    missing = sum(1 for c in commits if not c.get("synthetic_query"))
    print(f"{len(tasks)} test task(s), {len(learn)} learning commit(s); {missing} missing task description(s)")
    print(f"Output: {out / 'benchmark_tasks.jsonl'}")


if __name__ == "__main__":
    asyncio.run(main())
