"""Shared helpers for the data-construction scripts: git access, prefiltering, LLM calls."""

import json
import re
import subprocess
import sys
from pathlib import Path

import tiktoken

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.helpers import find_first_json  # noqa: E402
from modules.llm import complete_text  # noqa: E402

PROMPTS_DIR = REPO_ROOT / "system_prompts" / "data_construction"
TOKENIZER = tiktoken.get_encoding("cl100k_base")
MAX_DIFF_TOKENS = 180_000  # commits with larger diffs are skipped

AUTO_PATTERNS = re.compile(
    r"\b(bump\s+version|auto[- ]generated|release\s+v?\d|update\s+changelog|generated\s+by|"
    r"bot\s+commit|typo|fix\s+typo|fixup|squash)\b",
    re.IGNORECASE,
)


# ── files ──────────────────────────────────────────────────────────────────

def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(items: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ── git ────────────────────────────────────────────────────────────────────

def run_git(args: list[str], cwd: str, timeout: int = 300, binary: bool = False):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])} failed: {result.stderr.decode(errors='replace')[:500]}")
    return result.stdout if binary else result.stdout.decode("utf-8", errors="replace")


def get_commit_diff(sha: str, repo_dir: str) -> str | None:
    """`git diff <sha>~1 <sha>`, or None if unavailable or larger than MAX_DIFF_TOKENS."""
    try:
        diff = run_git(["diff", f"{sha}~1", sha, "--no-color"], cwd=repo_dir, timeout=60)
    except RuntimeError:
        return None
    if len(TOKENIZER.encode(diff, disallowed_special=())) > MAX_DIFF_TOKENS:
        return None
    return diff


def write_commit_diff(sha: str, repo_dir: str, output_path: Path) -> bool:
    try:
        diff = run_git(["diff", f"{sha}~1", sha, "--no-color"], cwd=repo_dir, timeout=60)
    except RuntimeError:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(diff, encoding="utf-8")
    return True


def write_git_archive(rev: str, repo_dir: str, output_path: Path) -> bool:
    try:
        data = run_git(["archive", "--format=tar.gz", rev], cwd=repo_dir, timeout=600, binary=True)
    except RuntimeError:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return True


_SEP, _FIELD, _BODY_START, _BODY_END = "<<COMMIT_SEP>>", "<<FIELD_SEP>>", "<<BODY_START>>", "<<BODY_END>>"


def parse_commits(repo_dir: str, max_raw: int | None = None) -> list[dict]:
    """Non-merge commits (newest first) with subject, date, body and numstat totals."""
    fmt = f"{_SEP}%H{_FIELD}%s{_FIELD}%aI{_FIELD}{_BODY_START}%B{_BODY_END}"
    cmd = ["log", "--format=" + fmt, "--no-merges", "--numstat"]
    if max_raw is not None:
        cmd.append(f"-n{max_raw}")
    commits = []
    for chunk in run_git(cmd, cwd=repo_dir, timeout=1800).split(_SEP):
        chunk = chunk.strip()
        if not chunk or f"{_FIELD}{_BODY_START}" not in chunk or _BODY_END not in chunk:
            continue
        header, rest = chunk.split(f"{_FIELD}{_BODY_START}", 1)
        body, stats = rest.split(_BODY_END, 1)
        parts = header.split(_FIELD)
        if len(parts) < 3:
            continue
        sha, subject, date = (p.strip() for p in parts[:3])
        body = body.strip()
        if body.startswith(subject):
            body = body[len(subject):].strip()
        files, ins, dels = [], 0, 0
        for line in stats.splitlines():
            cols = line.strip().split("\t")
            if len(cols) == 3:
                ins += int(cols[0]) if cols[0] != "-" else 0
                dels += int(cols[1]) if cols[1] != "-" else 0
                files.append(cols[2])
        commits.append({
            "sha": sha, "subject": subject, "date": date, "body": body,
            "changed_files": files, "insertions": ins, "deletions": dels,
            "total_lines": ins + dels, "file_count": len(files),
        })
    return commits


def prefilter(commits: list[dict]) -> list[dict]:
    """Drop tiny commits and automated/housekeeping commits."""
    kept = []
    for c in commits:
        if c["total_lines"] < 10 or (c["file_count"] == 1 and c["total_lines"] < 15):
            continue
        if AUTO_PATTERNS.search(c["subject"]):
            continue
        kept.append(c)
    return kept


# ── LLM ────────────────────────────────────────────────────────────────────

async def call_json(prompt: str, model: str | None = None, temperature: float = 0.0) -> dict | None:
    """One completion whose answer is expected to contain a JSON object."""
    for _ in range(2):
        text = await complete_text(prompt, model=model, temperature=temperature)
        parsed = find_first_json(text or "")
        if parsed is not None:
            return parsed
    return None


STRUCTIFY_PROMPT = """You are a JSON extractor. Extract a JSON object from the analysis below.

Analysis:
{analysis}

Required JSON format:
{schema}

Output JSON only, no extra prose."""


async def analyze_then_extract(prompt: str, json_schema: str, model: str | None = None) -> tuple[dict | None, str]:
    """Two stages: free-form analysis, then conversion of that analysis to JSON."""
    analysis = await complete_text(prompt, model=model, temperature=0.3)
    if not analysis:
        return None, ""
    return await call_json(STRUCTIFY_PROMPT.format(analysis=analysis, schema=json_schema), model=model), analysis


async def populate_synthetic_query(commit: dict, repo_dir: str, template: str, model: str | None = None) -> str:
    """Write an issue-style task description for a commit into commit['synthetic_query']."""
    diff = get_commit_diff(commit["sha"], repo_dir)
    if diff is None:
        commit["synthetic_query"] = ""
        return ""
    prompt = (
        template.replace("{subject}", commit.get("subject", ""))
        .replace("{body}", commit.get("body", ""))
        .replace("{patch}", diff)
    )
    commit["synthetic_query"] = await complete_text(prompt, model=model, temperature=0.3) or ""
    return commit["synthetic_query"]
