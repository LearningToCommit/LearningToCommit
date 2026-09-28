"""Agent sessions on top of the Claude Agent SDK.

One `AgentSession` = one Claude Code session (via `ClaudeSDKClient`) rooted in a phase
workspace. `run(prompt)` sends a user turn and consumes the message stream until the turn's
`ResultMessage`; calling it again continues the same conversation (used for
attempt -> reflect, and for the forced-exploration nudges).

The SDK owns the agent loop, prompt caching and context compaction. This module only
configures the session and records the stream as a trajectory:

    trajectory = {
        "status": "success" | "error",
        "error_type": ...,                 # only on error
        "messages": [...],                 # OpenAI-style: system / user / assistant(tool_calls) / tool
        "tools": [...],                    # built-in tools the agent could use
        "model", "session_id", "num_turns", "cost_usd", "usage", "duration_ms",
    }

Each assistant message in `messages` corresponds to one LLM response; the number of
assistant messages that carry `tool_calls` is the "trajectory steps" metric.
"""

import dataclasses
import json
import os
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from modules.config import AGENT_THINKING, MAX_OUTPUT_TOKENS, SANDBOX_ROOT

# Built-in Claude Code tools exposed to the agent (no web access, no subagents).
AGENT_TOOLS = ["Read", "Edit", "Write", "Bash", "Glob", "Grep"]

# Environment for the Claude Code subprocess: isolate it from the user's personal
# configuration (settings, CLAUDE.md, memory, skills, plugins, MCP servers).
_ISOLATION_ENV = {
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(MAX_OUTPUT_TOKENS),
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_BUNDLED_SKILLS": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    # Keep Python bytecode out of the exported diffs.
    "PYTHONDONTWRITEBYTECODE": "1",
}


def _claude_config_dir() -> str:
    """Private Claude Code config dir (session transcripts land here, not in ~/.claude)."""
    path = Path(os.getenv("LTC_CLAUDE_CONFIG_DIR", "").strip() or SANDBOX_ROOT / "claude_config")
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _tool_result_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
        else:
            parts.append(f"[{item.get('type', 'content') if isinstance(item, dict) else 'content'}]")
    return "\n".join(parts)


def count_tool_steps(messages: list[dict]) -> int:
    """Number of LLM responses that issued at least one tool call."""
    return sum(1 for m in messages if m.get("role") == "assistant" and m.get("tool_calls"))


class AgentSession:
    """A Claude Code session with a custom system prompt, rooted at `workspace`.

    Example:
        async with AgentSession(workspace, system_prompt, model, extra_dirs=[skills]) as agent:
            await agent.run("Implement ...")
            traj = agent.trajectory()
    """

    def __init__(
        self,
        workspace: Path,
        system_prompt: str,
        model: str,
        extra_dirs: list[Path] | None = None,
        max_turns: int = 300,
        log_prefix: str = "",
    ):
        self.model = model
        self.log_prefix = log_prefix
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self.result: dict = {}
        self.error_type: str | None = None
        self._assistant_index: dict[str, int] = {}
        self.options = ClaudeAgentOptions(
            cwd=str(workspace),
            add_dirs=[str(d) for d in (extra_dirs or [])],
            system_prompt=system_prompt,
            model=model,
            tools=AGENT_TOOLS,
            allowed_tools=AGENT_TOOLS,
            # Pre-approved tools run without prompts; anything else is denied.
            permission_mode="dontAsk",
            max_turns=max_turns,
            setting_sources=[],
            skills=[],
            strict_mcp_config=True,
            thinking={"type": AGENT_THINKING},
            env={**_ISOLATION_ENV, "CLAUDE_CONFIG_DIR": _claude_config_dir()},
        )
        self._client: ClaudeSDKClient | None = None

    async def __aenter__(self):
        self._client = ClaudeSDKClient(options=self.options)
        await self._client.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client is not None:
            try:
                await self._client.disconnect()
            finally:
                self._client = None
        return False

    # ── stream recording ──────────────────────────────────────────────────

    def _record_assistant(self, msg: AssistantMessage):
        # The CLI emits one AssistantMessage per content block; blocks of the same
        # LLM response share a message_id and are merged into one entry here.
        key = msg.message_id or f"anon-{len(self.messages)}"
        if key in self._assistant_index:
            entry = self.messages[self._assistant_index[key]]
        else:
            entry = {"role": "assistant", "content": None, "tool_calls": None, "model": msg.model}
            self._assistant_index[key] = len(self.messages)
            self.messages.append(entry)
        if msg.usage:
            entry["usage"] = msg.usage
        if msg.error:
            entry["error"] = msg.error
        for block in msg.content:
            if isinstance(block, TextBlock):
                entry["content"] = (entry["content"] or "") + block.text
            elif isinstance(block, ThinkingBlock):
                entry["reasoning_content"] = entry.get("reasoning_content", "") + block.thinking
            elif isinstance(block, ToolUseBlock):
                entry["tool_calls"] = (entry["tool_calls"] or []) + [{
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": json.dumps(block.input, ensure_ascii=False)},
                }]
                steps = count_tool_steps(self.messages)
                print(f"  {self.log_prefix}[step {steps}] {block.name}", flush=True)

    def _record_user(self, msg: UserMessage):
        if isinstance(msg.content, str):
            self.messages.append({"role": "user", "content": msg.content})
            return
        for block in msg.content:
            if isinstance(block, ToolResultBlock):
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": block.tool_use_id,
                    "content": _tool_result_text(block.content),
                    "is_error": bool(block.is_error),
                })
            elif isinstance(block, TextBlock):
                self.messages.append({"role": "user", "content": block.text})

    # ── public API ────────────────────────────────────────────────────────

    async def run(self, prompt: str) -> dict:
        """Send one user turn and consume the stream until the turn's ResultMessage."""
        assert self._client is not None, "use `async with AgentSession(...)`"
        self.messages.append({"role": "user", "content": prompt})
        self._assistant_index = {}
        self.error_type = None
        await self._client.query(prompt)
        async for msg in self._client.receive_response():
            if isinstance(msg, AssistantMessage):
                if msg.parent_tool_use_id is None:
                    self._record_assistant(msg)
            elif isinstance(msg, UserMessage):
                if msg.parent_tool_use_id is None:
                    self._record_user(msg)
            elif isinstance(msg, ResultMessage):
                self.result = {
                    k: v for k, v in dataclasses.asdict(msg).items()
                    if k in ("subtype", "is_error", "num_turns", "session_id", "stop_reason",
                             "total_cost_usd", "usage", "duration_ms", "errors", "result")
                }
                if msg.is_error or msg.subtype != "success":
                    self.error_type = msg.subtype if msg.subtype != "success" else "agent_error"
        return self.result

    def steps(self) -> int:
        return count_tool_steps(self.messages)

    def trajectory(self) -> dict:
        traj = {
            "status": "error" if self.error_type else "success",
            "messages": self.messages,
            "tools": AGENT_TOOLS,
            "model": self.model,
            "session_id": self.result.get("session_id"),
            "num_turns": self.result.get("num_turns"),
            # Cumulative over the whole session (all run() calls so far).
            "cost_usd": self.result.get("total_cost_usd"),
            "usage": self.result.get("usage"),
            "duration_ms": self.result.get("duration_ms"),
        }
        if self.error_type:
            traj["error_type"] = self.error_type
            traj["error"] = self.result.get("errors") or self.result.get("result")
        return traj
