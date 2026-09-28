"""Plain (non-agentic) Claude calls through the Anthropic Messages API.

Used by the pairwise judge and the data-construction scripts. Authentication comes from
ANTHROPIC_API_KEY (and optional ANTHROPIC_BASE_URL), the same variables the agent uses.
"""

import asyncio

import anthropic
from loguru import logger

from modules.config import DEFAULT_MODEL, MAX_OUTPUT_TOKENS

MAX_RETRIES = 4
_client: anthropic.AsyncAnthropic | None = None


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic()
    return _client


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError))


async def complete_text(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> str | None:
    """Single-turn completion; returns the concatenated text, or None after repeated failures.

    Streaming is used so that large max_tokens values do not hit non-streaming request limits.
    """
    kwargs = {
        "model": model or DEFAULT_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    if temperature is not None:
        # Recent SDK versions have no `temperature` argument; send it as a raw body field.
        kwargs["extra_body"] = {"temperature": temperature}

    for attempt in range(MAX_RETRIES):
        try:
            async with get_client().messages.stream(**kwargs) as stream:
                message = await stream.get_final_message()
            text = "".join(b.text for b in message.content if b.type == "text").strip()
            if text:
                return text
        except Exception as exc:  # noqa: BLE001 - surface every failure mode in the log
            if isinstance(exc, anthropic.BadRequestError) and "temperature" in str(exc) and "extra_body" in kwargs:
                kwargs.pop("extra_body")  # model does not accept sampling parameters
                continue
            if not _is_retryable(exc) or attempt == MAX_RETRIES - 1:
                logger.error(f"LLM call failed ({type(exc).__name__}): {str(exc)[:300]}")
                return None
            await asyncio.sleep(5 * 2**attempt)
    return None
