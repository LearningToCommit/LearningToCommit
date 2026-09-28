"""Assemble system prompts and user turns for each phase.

Prompt templates live in system_prompts/*.txt and use two path placeholders,
{{WORKSPACE_DIR}} and {{SKILLS_DIR}}, filled with the phase sandbox paths. Templates with
content placeholders ({real_diff}, {task_desc}, ...) are filled with str.format.
"""

from pathlib import Path

from modules.helpers import read_text

SOLVE_PHASES = ("solve_with_skill", "solve_baseline", "solve_forced")


def fill_paths(template: str, workspace: Path, skills: Path | None) -> str:
    text = template.replace("{{WORKSPACE_DIR}}", str(workspace))
    if skills is not None:
        text = text.replace("{{SKILLS_DIR}}", str(skills))
    return text


def build_system_prompt(phase: str, phase_prompts: dict[str, str], workspace: Path, skills: Path | None) -> str:
    return fill_paths(read_text(phase_prompts[phase]), workspace, skills)


def environment_context(phase: str, workspace: Path, skills: Path | None) -> str:
    """Short description of the directory layout, prepended to the first user turn."""
    lines = ["<environment_context>"]
    if phase in SOLVE_PHASES:
        lines.append(f"workspace: {workspace}  (repo source, already extracted and git-initialized; modify files here)")
    else:
        lines.append(f"workspace: {workspace}  (repo source, git initialized, you can `git diff` to see your changes)")
    if skills is not None:
        lines.append(f"skills: {skills}/  (skill directory — read SKILL.md first, then subsystem files as needed)")
    lines.append(
        "output: the workflow exports `git diff` from the workspace after you finish; "
        "you do not need to write a diff file yourself"
    )
    lines.append("</environment_context>")
    return "\n".join(lines)


def load_template(phase: str, phase_prompts: dict[str, str]) -> str:
    return read_text(phase_prompts[phase])
