"""Workflow constants: prompt templates, per-phase limits, default locations."""

import sys

from loguru import logger

from modules.config import REPO_ROOT

PROMPTS_DIR = REPO_ROOT / "system_prompts"

# Prompt template for each phase.
PHASE_PROMPTS = {
    "learn_attempt": str(PROMPTS_DIR / "learn_attempt.txt"),          # system prompt for learning
    "learn_reflect": str(PROMPTS_DIR / "learn_reflect.txt"),          # user turn after the blind attempt
    "learn_observe": str(PROMPTS_DIR / "learn_reflect_observe.txt"),  # observe-only ablation (no attempt)
    "solve_with_skill": str(PROMPTS_DIR / "solve_with_skill.txt"),
    "solve_baseline": str(PROMPTS_DIR / "solve_baseline.txt"),
    "solve_forced": str(PROMPTS_DIR / "solve_forced_exploration.txt"),  # forced-exploration baseline
}

# Per-phase limits.
PHASE_MAX_TURNS = 300          # max agent turns per session
PHASE_TIMEOUT_SECONDS = 3600   # wall-clock limit per agent turn sequence

# Forced-exploration baseline: minimum number of tool steps before the agent may stop.
DEFAULT_FORCED_MIN_STEPS = 40
FORCED_NUDGE = (
    "Are you sure you are done? You still have {remaining} tool steps available out of your budget. "
    "Please make full use of them: continue exploring the repository and verifying your approach "
    "against its conventions before finalising."
)
MAX_CONSECUTIVE_NUDGES = 3

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output"

logger.remove()
logger.add(sys.stderr, level="WARNING")
