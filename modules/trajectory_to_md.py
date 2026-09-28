#!/usr/bin/env python3
"""Render a trajectory.json (see modules/agent.py) as readable Markdown.

Usage:
    python modules/trajectory_to_md.py path/to/trajectory.json   # writes trajectory.md next to it
"""

import json
import sys
from pathlib import Path


def _truncate(text: str, max_len: int) -> str:
    text = text or ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... (truncated, {len(text)} chars total)"


def _format_call(name: str, args: dict) -> str:
    if name == "Bash":
        return f"`Bash`\n\n```bash\n{_truncate(args.get('command', ''), 3000)}\n```"
    if name in ("Read", "Glob", "Grep"):
        return f"`{name}` " + ", ".join(f"{k}=`{v}`" for k, v in args.items())
    if name == "Write":
        return f"`Write` → `{args.get('file_path', '?')}`\n\n```\n{_truncate(args.get('content', ''), 5000)}\n```"
    if name == "Edit":
        return (
            f"`Edit` → `{args.get('file_path', '?')}`\n\n"
            f"old:\n```\n{_truncate(args.get('old_string', ''), 1500)}\n```\n\n"
            f"new:\n```\n{_truncate(args.get('new_string', ''), 1500)}\n```"
        )
    return f"`{name}`\n\n```json\n{_truncate(json.dumps(args, ensure_ascii=False, indent=2), 3000)}\n```"


def convert_trajectory(traj: dict, title: str | None = None) -> str:
    messages = traj.get("messages", [])
    results = {m.get("tool_call_id"): m for m in messages if m.get("role") == "tool"}
    out = [f"# {title or 'Trajectory'}", ""]
    out.append(
        f"status: **{traj.get('status')}** · model: `{traj.get('model')}` · "
        f"turns: {traj.get('num_turns')} · cost: ${traj.get('cost_usd') or 0:.4f}"
    )
    if traj.get("error_type"):
        out.append(f"\nerror: `{traj['error_type']}` {traj.get('error') or ''}")
    step = 0
    for m in messages:
        role = m.get("role")
        if role == "system":
            out += ["", "## System prompt", "", "```", _truncate(m.get("content", ""), 4000), "```"]
        elif role == "user":
            out += ["", "## User", "", _truncate(m.get("content", ""), 6000)]
        elif role == "assistant":
            if m.get("content"):
                out += ["", "### Assistant", "", m["content"]]
            for call in m.get("tool_calls") or []:
                step += 1
                fn = call["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"raw": fn.get("arguments")}
                out += ["", f"### Step {step}: {fn['name']}", "", _format_call(fn["name"], args)]
                res = results.get(call["id"])
                if res is not None:
                    flag = " (error)" if res.get("is_error") else ""
                    out += ["", f"<details><summary>result{flag}</summary>", "", "```",
                            _truncate(res.get("content", ""), 2000), "```", "", "</details>"]
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    traj = json.loads(path.read_text(encoding="utf-8"))
    out_path = path.with_suffix(".md")
    out_path.write_text(convert_trajectory(traj, title=path.parent.name), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
