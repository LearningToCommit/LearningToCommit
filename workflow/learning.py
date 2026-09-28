"""Sequential learning over a curriculum of historical commits.

Test tasks that share the same learning commits form one curriculum and share one learning
run. Learning is resumable: completed iterations (iter_XX_<sha>/summary.json) are skipped and
the run continues from the first unfinished commit with the latest skill snapshot.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

from modules.helpers import load_json_if_exists, save_json
from workflow.output_layout import step1_curriculum_dir
from workflow.phases import run_learning_iteration


def iteration_dir_name(index: int, commit_info: dict) -> str:
    return f"iter_{index:02d}_{commit_info['sha'][:10]}"


def learning_iteration_is_complete(iter_dir: Path, commit_info: dict) -> bool:
    summary = load_json_if_exists(iter_dir / "summary.json")
    return bool(
        summary
        and summary.get("commit_sha") == commit_info["sha"][:10]
        and summary.get("attempt_status") == "success"
        and summary.get("reflect_status") == "success"
    )


def get_learning_commits(item: dict) -> list[dict]:
    """Learning commits of a task, optionally truncated by item['max_learning_commits'] (0 = all)."""
    commits = item.get("learning_commits", [])
    limit = int(item.get("max_learning_commits") or 0)
    return commits[:limit] if limit > 0 else commits


def get_curriculum_key(item: dict) -> str:
    """Hash of the (sha, diff) sequence; tasks with equal keys share one learning run."""
    commits = get_learning_commits(item)
    if not commits:
        return "no_learning"
    identity = [(c["sha"], Path(c["diff_path"]).name) for c in commits]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def group_learning_curricula(items: list[dict]) -> dict[str, dict]:
    """{curriculum_key: {"learning_item": <virtual item with the commits>, "task_ids": [...]}}"""
    groups: dict[str, dict] = {}
    for item in items:
        commits = get_learning_commits(item)
        if not commits:
            continue
        key = get_curriculum_key(item)
        if key not in groups:
            virtual = {"id": f"curriculum_{key[:8]}", "learning_commits": commits}
            groups[key] = {"learning_item": virtual, "task_ids": []}
        groups[key]["task_ids"].append(str(item["id"]))
    return groups


def discover_learning_resume_state(learning_dir: Path, commits: list[dict]) -> tuple[int, Path | None]:
    """Index of the first unfinished iteration, and the latest skill snapshot before it."""
    resume_index, skill_dir = 0, None
    for i, commit_info in enumerate(commits):
        iter_dir = learning_dir / iteration_dir_name(i, commit_info)
        if not learning_iteration_is_complete(iter_dir, commit_info):
            break
        resume_index = i + 1
        if (iter_dir / "skills" / "SKILL.md").exists():
            skill_dir = iter_dir / "skills"
    return resume_index, skill_dir


def skill_dir_size_kb(skill_dir: Path) -> float:
    return sum(f.stat().st_size for f in skill_dir.rglob("*") if f.is_file()) / 1024


def _finalize(skill_dir: Path, final_dir: Path) -> Path:
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(skill_dir, final_dir)
    return final_dir


async def learn_or_load(
    item: dict,
    model_id: str,
    output_root: Path,
    curriculum_key: str,
    mode: str = "attempt_reflect",
) -> Path | None:
    """Run (or resume) sequential learning; return the final skill directory.

    Each iteration starts from the previous iteration's skills. A failed iteration writes
    error.json and raises; rerunning the same command resumes from that iteration.
    """
    commits = get_learning_commits(item)
    if not commits:
        raise ValueError("sequential learning requires non-empty learning_commits")

    learning_dir = step1_curriculum_dir(output_root, curriculum_key)
    final_dir = learning_dir / "final_skills"
    if (final_dir / "SKILL.md").exists():
        print(f"[{item['id']}] skills cached ({skill_dir_size_kb(final_dir):.1f}KB), skipping learning", flush=True)
        return final_dir
    learning_dir.mkdir(parents=True, exist_ok=True)

    resume_index, skill_dir = discover_learning_resume_state(learning_dir, commits)
    if resume_index:
        print(f"[{item['id']}] resuming learning at commit {resume_index + 1}/{len(commits)}", flush=True)

    pbar = tqdm(total=len(commits), initial=resume_index, desc=f"[{item['id']}] learning", file=sys.stdout)
    for i, commit_info in enumerate(commits[resume_index:], start=resume_index):
        iter_dir = learning_dir / iteration_dir_name(i, commit_info)
        pbar.set_description(f"[{item['id']}] learning {i + 1}/{len(commits)} ({commit_info['sha'][:8]})")
        try:
            await run_learning_iteration(model_id, iter_dir, commit_info, skill_dir, mode=mode)
        except Exception as exc:
            save_json(iter_dir / "error.json", {
                "commit_sha": commit_info["sha"][:10],
                "error": str(exc),
                "error_type": type(exc).__name__,
            })
            pbar.close()
            raise RuntimeError(f"learning iteration {i} ({commit_info['sha'][:8]}) failed: {exc}") from exc
        skill_dir = iter_dir / "skills"
        pbar.set_postfix(skill=f"{skill_dir_size_kb(skill_dir):.1f}KB")
        pbar.update(1)
    pbar.close()

    if skill_dir and (skill_dir / "SKILL.md").exists():
        return _finalize(skill_dir, final_dir)
    return None
