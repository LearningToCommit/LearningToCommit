#!/usr/bin/env python3
"""Turn a released benchmark JSONL into a runnable benchmark_tasks.jsonl.

The released files (data/learning_to_commit_benchmark.jsonl, data/swebench_pro_tasks.jsonl)
inline all diffs but not the repository snapshots. This script:

  1. clones each needed upstream repository (blobless clone) into <out-dir>/repos/<repo>;
  2. builds snapshot tarballs with `git archive`:
       - learning commits and Learning-to-Commit test tasks: the parent revision `<sha>~1`;
       - SWE-bench Pro test tasks: the instance's base commit;
  3. writes every oracle / learning diff to <out-dir>/diffs/ (for Learning-to-Commit test tasks
     the released diff is checked against `git diff <sha>~1 <sha>`);
  4. emits <out-dir>/benchmark_tasks.jsonl with relative paths, ready for
     workflow/run_sequential.py.

Examples:
    # one repository, first test task, first learning commit only (a quick check)
    python scripts/s1_prepare_benchmark.py --repos DeepSpeed --limit 1 --max-learning-commits 1

    # the full Learning to Commit benchmark (~5 large clones; needs several GB)
    python scripts/s1_prepare_benchmark.py
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Repository short name (the "repo" field) -> upstream URL.
UPSTREAM_URLS = {
    # Learning to Commit benchmark
    "vllm": "https://github.com/vllm-project/vllm.git",
    "sglang": "https://github.com/sgl-project/sglang.git",
    "DeepSpeed": "https://github.com/deepspeedai/DeepSpeed.git",
    "milvus": "https://github.com/milvus-io/milvus.git",
    "llama.cpp": "https://github.com/ggml-org/llama.cpp.git",
    # SWE-bench Pro subset
    "NodeBB": "https://github.com/NodeBB/NodeBB.git",
    "ansible": "https://github.com/ansible/ansible.git",
    "element-web": "https://github.com/element-hq/element-web.git",
    "flipt": "https://github.com/flipt-io/flipt.git",
    "vuls": "https://github.com/future-architect/vuls.git",
    "teleport": "https://github.com/gravitational/teleport.git",
    "openlibrary": "https://github.com/internetarchive/openlibrary.git",
    "navidrome": "https://github.com/navidrome/navidrome.git",
    "qutebrowser": "https://github.com/qutebrowser/qutebrowser.git",
    "tutanota": "https://github.com/tutao/tutanota.git",
}

# SWE-bench Pro base commits (from the public SWE-bench Pro dataset), keyed by instance id.
SWEBENCH_PRO_BASE_COMMITS = REPO_ROOT / "data" / "swebench_pro_base_commits.json"


def git(args: list[str], cwd: Path, timeout: int = 3600, binary: bool = False):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])} failed in {cwd}: {result.stderr.decode(errors='replace')[:500]}")
    return result.stdout if binary else result.stdout.decode("utf-8", errors="replace")


def ensure_clone(repo: str, repos_dir: Path) -> Path:
    path = repos_dir / repo
    if (path / ".git").exists():
        return path
    url = UPSTREAM_URLS.get(repo)
    if url is None:
        sys.exit(f"no upstream URL known for repo {repo!r}; add it to UPSTREAM_URLS")
    repos_dir.mkdir(parents=True, exist_ok=True)
    print(f"  cloning {url} -> {path} (blobless; this can take a few minutes)", flush=True)
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(path)], check=True)
    return path


def resolve_commit(rev: str, repo_dir: Path) -> str:
    """Full SHA for a (possibly abbreviated) revision; fetches it if the clone lacks it."""
    try:
        return git(["rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"], repo_dir).strip()
    except RuntimeError:
        if re.fullmatch(r"[0-9a-f]{40}", rev):
            git(["fetch", "--quiet", "--filter=blob:none", "origin", rev], repo_dir)
            return git(["rev-parse", "--verify", f"{rev}^{{commit}}"], repo_dir).strip()
        raise


def write_snapshot(rev: str, repo_dir: Path, out_path: Path):
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = git(["archive", "--format=tar.gz", rev], repo_dir, binary=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(out_path)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_diff(text: str) -> str:
    """Drop `index` lines (abbreviation length varies) before comparing diffs."""
    return "\n".join(line for line in text.splitlines() if not line.startswith("index ")).strip()


def main():
    parser = argparse.ArgumentParser(description="Build snapshots and oracle diffs for a released benchmark JSONL")
    parser.add_argument("--benchmark", default=str(REPO_ROOT / "data" / "learning_to_commit_benchmark.jsonl"),
                        help="released benchmark JSONL")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: data/prepared/<benchmark name>)")
    parser.add_argument("--repos-dir", default=None, help="where to clone repositories (default: <out-dir>/repos)")
    parser.add_argument("--repos", default="", help="comma-separated repo filter, e.g. DeepSpeed,vllm")
    parser.add_argument("--limit", type=int, default=0, help="max test tasks per repository (0 = all)")
    parser.add_argument("--max-learning-commits", type=int, default=0,
                        help="keep only the first N learning commits per task (0 = all)")
    args = parser.parse_args()

    bench_path = Path(args.benchmark).resolve()
    out_dir = Path(args.out_dir or REPO_ROOT / "data" / "prepared" / bench_path.stem).resolve()
    repos_dir = Path(args.repos_dir).resolve() if args.repos_dir else out_dir / "repos"
    wanted = {r.strip() for r in args.repos.split(",") if r.strip()}

    rows = [json.loads(line) for line in bench_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected, per_repo = [], {}
    for row in rows:
        if wanted and row["repo"] not in wanted:
            continue
        if args.limit and per_repo.get(row["repo"], 0) >= args.limit:
            continue
        per_repo[row["repo"]] = per_repo.get(row["repo"], 0) + 1
        selected.append(row)
    if not selected:
        sys.exit(f"no tasks selected (repos in file: {sorted({r['repo'] for r in rows})})")
    print(f"{len(selected)} test task(s) from {sorted(per_repo)} -> {out_dir}", flush=True)

    swe_base = json.loads(SWEBENCH_PRO_BASE_COMMITS.read_text()) if SWEBENCH_PRO_BASE_COMMITS.exists() else {}
    out_tasks, mismatches = [], 0
    for row in selected:
        repo_dir = ensure_clone(row["repo"], repos_dir)
        task_id = row["id"]

        # Test task snapshot revision.
        if task_id in swe_base:
            test_rev = resolve_commit(swe_base[task_id]["base_commit"], repo_dir)
        else:
            m = re.fullmatch(r"commit_([0-9a-f]{7,40})", task_id)
            if not m:
                sys.exit(f"cannot derive a commit from task id {task_id!r}")
            sha = resolve_commit(m.group(1), repo_dir)
            test_rev = f"{sha}~1"
            actual = git(["diff", f"{sha}~1", sha, "--no-color"], repo_dir)
            if normalize_diff(actual) != normalize_diff(row["oracle_diff_text"]):
                mismatches += 1
                print(f"  warning: released oracle diff of {task_id} differs from git diff {sha[:10]}~1..{sha[:10]}")

        snapshot_rel = f"snapshots/{row['snapshot_filename']}"
        oracle_rel = f"diffs/{task_id}.diff"
        shown = f"{test_rev[:10]}~1" if test_rev.endswith("~1") else test_rev[:10]
        print(f"  [{row['repo']}] {task_id}: snapshot of {shown}", flush=True)
        write_snapshot(test_rev, repo_dir, out_dir / snapshot_rel)
        write_text(out_dir / oracle_rel, row["oracle_diff_text"])

        learning = []
        commits = row.get("learning_commits", [])
        if args.max_learning_commits:
            commits = commits[: args.max_learning_commits]
        for lc in commits:
            lsha = resolve_commit(lc["sha"], repo_dir)
            l_snapshot_rel = f"snapshots/{lc['snapshot_filename']}"
            l_diff_rel = f"diffs/commit_{lsha[:10]}.diff"
            write_snapshot(f"{lsha}~1", repo_dir, out_dir / l_snapshot_rel)
            write_text(out_dir / l_diff_rel, lc["diff_text"])
            learning.append({
                "sha": lsha,
                "repo": lc.get("repo", row["repo"]),
                "title": lc.get("title", ""),
                "category": lc.get("category", ""),
                "synthetic_query": lc["synthetic_query"],
                "diff_path": l_diff_rel,
                "snapshot_path": l_snapshot_rel,
            })
        print(f"      + {len(learning)} learning commit snapshot(s)", flush=True)

        out_tasks.append({
            "id": task_id,
            "repo": row["repo"],
            "category": row.get("category", ""),
            "synthetic_query": row["synthetic_query"],
            "oracle_diff": oracle_rel,
            "snapshot_path": snapshot_rel,
            "learning_commits": learning,
        })

    tasks_path = out_dir / "benchmark_tasks.jsonl"
    with open(tasks_path, "w", encoding="utf-8") as f:
        for task in out_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(out_tasks)} task(s) to {tasks_path}")
    if mismatches:
        print(f"{mismatches} oracle diff(s) differ from upstream git history (see warnings above)")


if __name__ == "__main__":
    main()
