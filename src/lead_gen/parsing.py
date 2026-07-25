"""
Shared JSON parsing for crew outputs — no more silent [] on failure.

A parse failure and "the model genuinely found nothing" are different events:
the first is an error to log loudly (and optionally retry), the second is a
legitimate empty result. ParseResult keeps them distinguishable.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    items: list = field(default_factory=list)
    error: Optional[str] = None  # set only on parse failure, never on empty

    @property
    def failed(self) -> bool:
        return self.error is not None


def _try_parse(raw: str) -> Optional[list]:
    """Attempt to extract a JSON array (or single-key dict of one) from raw text."""
    candidates = []

    if "```json" in raw:
        candidates.append(raw.split("```json", 1)[1].split("```", 1)[0].strip())
    elif "```" in raw:
        candidates.append(raw.split("```", 1)[1].split("```", 1)[0].strip())

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        candidates.append(match.group(0))

    candidates.append(raw.strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and len(parsed) == 1:
            value = next(iter(parsed.values()))
            if isinstance(value, list):
                return value
    return None


def parse_json_array(
    raw: str,
    *,
    context: str,
    retry_cb: Optional[Callable[[], str]] = None,
) -> ParseResult:
    """
    Parse a JSON array out of an LLM's raw output.

    On failure: log the error with a truncated dump, call retry_cb once (a
    bounded re-kickoff of the crew) if provided, and if that also fails return
    a ParseResult with .error set — callers must not treat it as "0 results".
    """
    parsed = _try_parse(raw or "")
    if parsed is not None:
        return ParseResult(items=parsed)

    logger.error(
        "[%s] failed to parse JSON array from crew output. Raw (truncated): %r",
        context, (raw or "")[:500],
    )

    if retry_cb is not None:
        logger.info("[%s] retrying crew kickoff once after parse failure", context)
        try:
            retry_raw = retry_cb()
        except Exception as exc:
            return ParseResult(error=f"parse failed and retry raised: {exc}")
        parsed = _try_parse(retry_raw or "")
        if parsed is not None:
            return ParseResult(items=parsed)
        logger.error(
            "[%s] retry also failed to produce parseable JSON. Raw (truncated): %r",
            context, (retry_raw or "")[:500],
        )

    return ParseResult(error=f"could not parse JSON array from {context} output")
