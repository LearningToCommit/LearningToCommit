"""Output directory layout of one experiment: output/<signature>/<model>/..."""

from pathlib import Path

STEP1_DIRNAME = "01_onboarding"   # learning curricula and learned skills
STEP2_DIRNAME = "02_resolution"   # solve runs (with_skill / baseline) and per-task records
STEP3_DIRNAME = "03_evaluation"   # evaluation outputs

TASK_RECORDS_DIRNAME = "task_records"
STEP2_VARIANTS = ("with_skill", "baseline", "baseline_forced")


def step1_dir(output_root: Path) -> Path:
    return output_root / STEP1_DIRNAME


def step1_curriculum_dir(output_root: Path, curriculum_key: str) -> Path:
    return step1_dir(output_root) / f"curriculum_{curriculum_key[:8]}"


def step2_dir(output_root: Path) -> Path:
    return output_root / STEP2_DIRNAME


def step2_variant_dir(output_root: Path, variant: str, task_id: str | None = None) -> Path:
    if variant not in STEP2_VARIANTS:
        raise ValueError(f"unsupported Step 2 variant: {variant}")
    path = step2_dir(output_root) / variant
    return path / task_id if task_id else path


def step2_task_records_dir(output_root: Path) -> Path:
    return step2_dir(output_root) / TASK_RECORDS_DIRNAME


def step2_task_records_index_path(output_root: Path) -> Path:
    return step2_task_records_dir(output_root) / "index.jsonl"


def step3_dir(output_root: Path) -> Path:
    return output_root / STEP3_DIRNAME


def build_task_paths(output_root: Path, task_id: str) -> dict[str, Path]:
    return {
        "record_path": step2_task_records_dir(output_root) / f"{task_id}.json",
        "solve_with_skill_dir": step2_variant_dir(output_root, "with_skill", task_id),
        "solve_baseline_dir": step2_variant_dir(output_root, "baseline", task_id),
        "solve_baseline_forced_dir": step2_variant_dir(output_root, "baseline_forced", task_id),
    }


def initialize_experiment_layout(output_root: Path):
    for path in (step1_dir(output_root), step2_task_records_dir(output_root), step3_dir(output_root)):
        path.mkdir(parents=True, exist_ok=True)
    (output_root / "README.md").write_text(
        "# Experiment layout\n\n"
        f"- `{STEP1_DIRNAME}/curriculum_<key>/`: one directory per learning commit "
        "(`iter_XX_<sha>/` with attempt/reflect trajectories, the attempt diff and the skill snapshot), "
        "plus `final_skills/`\n"
        f"- `{STEP2_DIRNAME}/with_skill|baseline|baseline_forced/<task_id>/`: solve trajectory, "
        "`changes.diff`, `summary.json`\n"
        f"- `{STEP2_DIRNAME}/{TASK_RECORDS_DIRNAME}/`: one status card per task + `index.jsonl`\n"
        f"- `{STEP3_DIRNAME}/`: evaluation outputs (programmatic metrics, pairwise judge)\n",
        encoding="utf-8",
    )
