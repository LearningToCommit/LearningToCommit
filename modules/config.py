"""Runtime settings, read once from environment variables (and an optional .env file)."""

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Variables already exported in the shell take precedence over .env.
load_dotenv(REPO_ROOT / ".env", override=False)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


# Model used by every agent phase unless --model is given on the CLI.
DEFAULT_MODEL = os.getenv("LTC_MODEL", "").strip() or "claude-sonnet-4-6"

# Max output tokens per LLM call (agent phases and plain Messages API calls).
MAX_OUTPUT_TOKENS = _int_env("LTC_MAX_TOKENS", 32768)

# Extended thinking for agent phases: "disabled" (default, as in the paper) or "adaptive".
AGENT_THINKING = os.getenv("LTC_THINKING", "").strip() or "disabled"

# Number of test tasks solved concurrently.
CONCURRENCY = _int_env("LTC_CONCURRENCY", 4)

# Parent directory for per-phase working directories (workspace + skills).
# resolve() matters on macOS, where the temp dir is behind the /var -> /private/var symlink.
SANDBOX_ROOT = Path(
    os.getenv("LTC_SANDBOX_ROOT", "").strip() or Path(tempfile.gettempdir()) / "ltc_sandbox"
).resolve()

# Keep per-phase working directories after the phase ends (debugging).
SANDBOX_KEEP = os.getenv("LTC_SANDBOX_KEEP", "") == "1"

# Models for the pairwise judge (stage 1 analysis, stage 2 JSON extraction).
JUDGE_MODEL = os.getenv("LTC_JUDGE_MODEL", "").strip() or DEFAULT_MODEL
JUDGE_EXTRACTOR_MODEL = os.getenv("LTC_JUDGE_EXTRACTOR_MODEL", "").strip() or JUDGE_MODEL
