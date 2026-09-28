"""Small I/O helpers shared by the workflow and evaluation scripts."""

import json
from pathlib import Path


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def load_json_if_exists(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def upsert_jsonl_by_key(path: Path, payload: dict, key: str):
    rows = [row for row in load_jsonl(path) if row.get(key) != payload[key]]
    rows.append(payload)
    save_jsonl(path, rows)


def latest_assistant_text(messages: list[dict]) -> str:
    """Text of the last assistant message that has any text."""
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return message["content"]
    return ""


def load_cached_trajectory(stage_dir: Path, *required_paths: str):
    """Return the saved trajectory if the stage finished successfully and all outputs exist."""
    summary = load_json_if_exists(stage_dir / "summary.json")
    trajectory = load_json_if_exists(stage_dir / "trajectory.json")
    if not summary or summary.get("status") != "success" or not trajectory:
        return None
    if any(not (stage_dir / rel).exists() for rel in required_paths):
        return None
    return trajectory


def find_first_json(text: str):
    """Extract the first valid JSON object embedded in free text."""
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start: idx + 1])
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None
