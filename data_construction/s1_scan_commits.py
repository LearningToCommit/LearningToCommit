#!/usr/bin/env python3
"""S1: git log -> heuristic prefilter -> keep the most recent N -> temporal split -> LLM suitability scan.

Usage:
    python data_construction/s1_scan_commits.py --repo-dir /path/to/repo --max-commits 500
Output: data_construction/runs/<repo>/s1_scanned.jsonl
"""

import argparse
import asyncio
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

from utils import REPO_ROOT, call_json, get_commit_diff, load_prompt, parse_commits, prefilter, save_jsonl


async def scan_one(sem, commit, repo_dir, scan_prompt, model):
    async with sem:
        diff = get_commit_diff(commit["sha"], repo_dir)
        if diff is None:
            commit.update(suitable=False, reason="diff_too_large_or_unavailable")
            return commit
        result = await call_json(f"{scan_prompt}\n\n{diff}", model=model)
        if result is None:
            commit.update(suitable=False, reason="llm_failed")
        else:
            commit.update(suitable=bool(result.get("suitable", False)), reason=result.get("reason", ""))
        return commit


async def main():
    parser = argparse.ArgumentParser(description="S1: scan commits for learning value")
    parser.add_argument("--repo-dir", required=True, help="local clone of the repository")
    parser.add_argument("--output", default=None, help="default: data_construction/runs/<repo>/s1_scanned.jsonl")
    parser.add_argument("--max-commits", type=int, default=500, help="keep the most recent N commits after prefiltering")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="oldest fraction goes to the learning pool")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--model", default=None, help="default: LTC_MODEL")
    args = parser.parse_args()

    repo_dir = str(Path(args.repo_dir).resolve())
    repo_name = Path(repo_dir).name
    output = Path(args.output or REPO_ROOT / "data_construction" / "runs" / repo_name / "s1_scanned.jsonl")

    commits = parse_commits(repo_dir, max_raw=args.max_commits * 4)
    kept = prefilter(commits)
    kept.sort(key=lambda c: c["date"])
    kept = kept[-args.max_commits:]
    split = int(len(kept) * args.split_ratio)
    for i, c in enumerate(kept):
        c["split"] = "learning" if i < split else "test"
    print(f"git log: {len(commits)} commits, after prefilter: {len(kept)} (learning={split}, test={len(kept) - split})")

    sem = asyncio.Semaphore(args.concurrency)
    scan_prompt = load_prompt("wash_suitable.txt")
    await tqdm_asyncio.gather(*(scan_one(sem, c, repo_dir, scan_prompt, args.model) for c in kept))
    save_jsonl(kept, output)

    for split_name in ("learning", "test"):
        pool = [c for c in kept if c["split"] == split_name]
        print(f"  {split_name:<9} suitable {sum(c['suitable'] for c in pool)}/{len(pool)}")
    print(f"Output: {output}\nNext: python data_construction/s2_cluster.py --input {output}")


if __name__ == "__main__":
    asyncio.run(main())
