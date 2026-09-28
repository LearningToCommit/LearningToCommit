"""Per-phase working directory: a fresh repo snapshot plus an optional skills directory.

Every phase (learning attempt/reflect, solve) runs in its own directory under
SANDBOX_ROOT:

    <SANDBOX_ROOT>/<id>/workspace/   repo snapshot, git-initialized with a "base" commit
    <SANDBOX_ROOT>/<id>/skills/      skill files the agent reads and (during learning) writes

This is NOT an isolation boundary: the agent's Bash/Write tools run with your user
permissions on the host. Run the pipeline inside a container or VM.
"""

import shutil
import subprocess
import uuid
from pathlib import Path

from loguru import logger

from modules.config import SANDBOX_KEEP, SANDBOX_ROOT

# Neutral identity and no user hooks/signing for the workflow's own git commands.
_GIT_FLAGS = [
    "-c", "user.name=workflow",
    "-c", "user.email=workflow@example.com",
    "-c", "commit.gpgsign=false",
    "-c", "core.hooksPath=/dev/null",
]


def _run(cmd: list[str], cwd: Path, timeout: int = 600) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd[:4])} ...\n{stderr[:2000]}")
    return result.stdout.decode("utf-8", errors="replace")


def _git(args: list[str], cwd: Path, timeout: int = 600) -> str:
    return _run(["git", *_GIT_FLAGS, *args], cwd=cwd, timeout=timeout)


class PhaseSandbox:
    """Context manager that creates and (unless LTC_SANDBOX_KEEP=1) removes the phase directory."""

    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.root = SANDBOX_ROOT / self.id
        self.workspace = self.root / "workspace"
        self.skills = self.root / "skills"

    def __enter__(self):
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.skills.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if SANDBOX_KEEP:
            logger.info(f"keeping sandbox {self.root}")
        else:
            shutil.rmtree(self.root, ignore_errors=True)
        return False

    # ── setup ──────────────────────────────────────────────────────────────

    def setup_repo_workspace(self, snapshot_path: str | Path):
        """Extract the snapshot tarball and create the base commit that diffs are taken against.

        `git add -A` respects the repo's own .gitignore; the prompts tell the agent to
        `git add -f` any ignored file it deliberately creates.
        """
        snapshot_path = Path(snapshot_path).resolve()
        if not snapshot_path.exists():
            raise FileNotFoundError(f"snapshot not found: {snapshot_path}")
        _run(["tar", "-xzf", str(snapshot_path), "-C", str(self.workspace)], cwd=self.root)
        _git(["init", "-q"], cwd=self.workspace)
        _git(["add", "-A"], cwd=self.workspace)
        _git(["commit", "-q", "-m", "base", "--allow-empty", "--no-verify"], cwd=self.workspace)

    def seed_skills(self, skill_dir: Path | None):
        """Copy an existing skill directory into the sandbox skills dir."""
        if skill_dir and Path(skill_dir).is_dir():
            shutil.copytree(skill_dir, self.skills, dirs_exist_ok=True)

    # ── outputs ────────────────────────────────────────────────────────────

    def export_workspace_diff(self) -> str:
        """Diff of the workspace against the base commit (captures commits the agent made, too)."""
        base = _git(["rev-list", "--max-parents=0", "HEAD"], cwd=self.workspace).split()[0]
        _git(["add", "-N", "--all", "."], cwd=self.workspace)
        return _git(["diff", base, "--no-color", "--binary"], cwd=self.workspace)

    def collect_skills(self, dest: Path):
        """Copy the sandbox skills dir to `dest` (replacing it)."""
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.skills, dest)
