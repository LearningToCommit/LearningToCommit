"""Single phases, each in a fresh sandbox directory.

    run_learning_iteration()  one learning commit: blind attempt -> contrastive reflection
                              against the oracle diff -> updated skill directory
                              (or the observe-only ablation: study the oracle diff directly)
    solve()                   one test task, with the learned skills or as a baseline

Phases communicate only through files in the experiment output directory.
"""

import asyncio
from pathlib import Path

from modules.agent import AgentSession
from modules.helpers import latest_assistant_text, load_cached_trajectory, read_text, save_json
from modules.prompt_builder import build_system_prompt, environment_context, fill_paths, load_template
from modules.sandbox import PhaseSandbox
from modules.trajectory_to_md import convert_trajectory
from workflow.config import (
    FORCED_NUDGE,
    MAX_CONSECUTIVE_NUDGES,
    PHASE_MAX_TURNS,
    PHASE_PROMPTS,
    PHASE_TIMEOUT_SECONDS,
)


def _save_trajectory(stage_dir: Path, name: str, traj: dict):
    save_json(stage_dir / f"{name}.json", traj)
    (stage_dir / f"{name}.md").write_text(convert_trajectory(traj, title=f"{stage_dir.name} / {name}"), encoding="utf-8")


async def run_learning_iteration(
    model_id: str,
    iter_dir: Path,
    commit_info: dict,
    prev_skill_dir: Path | None,
    mode: str = "attempt_reflect",
):
    """Learn from one historical commit and write the updated skills to iter_dir/skills/.

    attempt_reflect (the method):
        A. blind attempt: the agent implements the commit from its task description only;
        B. reflect: in the same session, the agent sees the oracle diff, contrasts it with its
           own attempt and updates the skill files.
    observe_only (ablation): no attempt; the agent studies the task description and the oracle
        diff in a fresh session and updates the skill files.

    Raises RuntimeError if a phase fails or no SKILL.md was written.
    """
    iter_dir.mkdir(parents=True, exist_ok=True)
    real_diff = read_text(commit_info["diff_path"])
    template_args = {
        "real_diff": real_diff,
        "sha_short": commit_info["sha"][:10],
        "subject": commit_info.get("title", ""),
        "task_desc": commit_info["synthetic_query"],
    }

    with PhaseSandbox() as sb:
        sb.setup_repo_workspace(commit_info["snapshot_path"])
        sb.seed_skills(prev_skill_dir)
        system_prompt = build_system_prompt("learn_attempt", PHASE_PROMPTS, sb.workspace, sb.skills)
        env_context = environment_context("learn_attempt", sb.workspace, sb.skills)
        agent = AgentSession(
            sb.workspace, system_prompt, model_id,
            extra_dirs=[sb.skills], max_turns=PHASE_MAX_TURNS,
            log_prefix=f"[learn {commit_info['sha'][:8]}] ",
        )

        async with agent:
            if mode == "observe_only":
                observe_query = fill_paths(load_template("learn_observe", PHASE_PROMPTS), sb.workspace, sb.skills)
                observe_query = observe_query.format(**template_args)
                await asyncio.wait_for(agent.run(f"{env_context}\n\n{observe_query}"), PHASE_TIMEOUT_SECONDS)
                reflect_traj = agent.trajectory()
                _save_trajectory(iter_dir, "reflect_trajectory", reflect_traj)
                if reflect_traj["status"] == "error":
                    raise RuntimeError(f"observe phase failed: {reflect_traj.get('error_type')}")
                attempt_status = reflect_traj["status"]
            else:
                # A. blind attempt
                attempt_query = f"Implement this change:\n\n{commit_info['synthetic_query']}"
                if commit_info.get("changed_files"):
                    attempt_query += "\n\nChanged files in this commit:\n" + "\n".join(
                        f"- {f}" for f in commit_info["changed_files"]
                    )
                await asyncio.wait_for(agent.run(f"{env_context}\n\n{attempt_query}"), PHASE_TIMEOUT_SECONDS)
                attempt_traj = agent.trajectory()
                _save_trajectory(iter_dir, "attempt_trajectory", attempt_traj)
                if attempt_traj["status"] == "error":
                    raise RuntimeError(f"attempt phase failed: {attempt_traj.get('error_type')}")
                attempt_status = attempt_traj["status"]
                # Export the attempt before reflecting: the agent may clean up its code while reflecting.
                (iter_dir / "attempt_changes.diff").write_text(sb.export_workspace_diff(), encoding="utf-8")

                # B. contrastive reflection against the oracle diff, continuing the same session
                reflect_query = fill_paths(load_template("learn_reflect", PHASE_PROMPTS), sb.workspace, sb.skills)
                reflect_query = reflect_query.format(**template_args)
                await asyncio.wait_for(agent.run(reflect_query), PHASE_TIMEOUT_SECONDS)
                reflect_traj = agent.trajectory()
                _save_trajectory(iter_dir, "reflect_trajectory", reflect_traj)
                if reflect_traj["status"] == "error":
                    raise RuntimeError(f"reflect phase failed: {reflect_traj.get('error_type')}")

        sb.collect_skills(iter_dir / "skills")

    # Known failure mode: the model says it updated the skills but never wrote the file.
    if not (iter_dir / "skills" / "SKILL.md").exists():
        raise RuntimeError(f"learning iteration did not produce SKILL.md in the skills directory ({iter_dir})")

    save_json(iter_dir / "summary.json", {
        "commit_sha": commit_info["sha"][:10],
        "commit_subject": commit_info.get("title", ""),
        "mode": mode,
        "attempt_status": attempt_status,
        "reflect_status": reflect_traj["status"],
        "cost_usd": reflect_traj.get("cost_usd"),
    })
    return reflect_traj


async def _solve_turns(agent: AgentSession, prompt: str, min_steps: int | None):
    await agent.run(prompt)
    if not min_steps:
        return
    # Forced exploration: while below min_steps, ask the agent to keep going; stop after
    # MAX_CONSECUTIVE_NUDGES nudges in a row that produce no new tool calls.
    consecutive = 0
    while agent.steps() < min_steps and consecutive < MAX_CONSECUTIVE_NUDGES and agent.error_type is None:
        before = agent.steps()
        await agent.run(FORCED_NUDGE.format(remaining=min_steps - before))
        consecutive = consecutive + 1 if agent.steps() == before else 0


async def solve(
    item: dict,
    model_id: str,
    stage_dir: Path,
    skill_dir: Path | None = None,
    variant: str = "baseline",
    min_steps: int | None = None,
) -> dict:
    """Solve one test task. Writes trajectory.json/.md, changes.diff and summary.json to stage_dir.

    skill_dir given          -> solve_with_skill (skills copied into the sandbox)
    no skill_dir             -> solve_baseline, or solve_forced if variant == "forced"
    """
    if skill_dir:
        phase = "solve_with_skill"
    elif variant == "forced":
        phase = "solve_forced"
    else:
        phase = "solve_baseline"

    cached = load_cached_trajectory(stage_dir, "changes.diff")
    if cached:
        print(f"[phase:cached] task={item['id']} phase={phase}", flush=True)
        return cached

    print(f"[phase:start] task={item['id']} phase={phase}", flush=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    with PhaseSandbox() as sb:
        sb.setup_repo_workspace(item["snapshot_path"])
        skills = None
        if phase == "solve_with_skill":
            sb.seed_skills(skill_dir)
            skills = sb.skills
        system_prompt = build_system_prompt(phase, PHASE_PROMPTS, sb.workspace, skills)
        prompt = f"{environment_context(phase, sb.workspace, skills)}\n\n{item['synthetic_query']}"
        agent = AgentSession(
            sb.workspace, system_prompt, model_id,
            extra_dirs=[skills] if skills else [], max_turns=PHASE_MAX_TURNS,
            log_prefix=f"[{item['id']}:{phase}] ",
        )

        phase_error = None
        try:
            async with agent:
                await asyncio.wait_for(
                    _solve_turns(agent, prompt, min_steps if phase == "solve_forced" else None),
                    PHASE_TIMEOUT_SECONDS,
                )
        except Exception as exc:  # keep going: the trajectory and diff are still saved
            phase_error = f"{type(exc).__name__}: {exc}"
        trajectory = agent.trajectory()
        if phase_error:
            trajectory.update(status="error", error_type="phase_execution_error", error=phase_error)
        trajectory["sandbox_id"] = sb.id

        try:
            diff = sb.export_workspace_diff()
        except Exception as exc:
            diff = ""
            trajectory.update(status="error", error_type="workspace_diff_export_error", error=str(exc))
        (stage_dir / "changes.diff").write_text(diff, encoding="utf-8")

    _save_trajectory(stage_dir, "trajectory", trajectory)
    save_json(stage_dir / "summary.json", {
        "phase": phase,
        "task_id": item["id"],
        "status": trajectory["status"],
        "error": trajectory.get("error"),
        "steps": agent.steps(),
        "cost_usd": trajectory.get("cost_usd"),
        "latest_text": latest_assistant_text(trajectory["messages"]),
    })
    print(f"[phase:end] task={item['id']} phase={phase} status={trajectory['status']} steps={agent.steps()}", flush=True)
    return trajectory
