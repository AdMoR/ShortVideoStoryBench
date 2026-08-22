"""
Parsing for pi's `--mode json` NDJSON event stream.

Shared by the judge backend (which shells out to pi for one focused answer) and
the agentic generator (which streams a long tool-using run). Each line is one
event; malformed lines are skipped rather than fatal, because a stream truncated
by a kill or a crash is still worth reading up to the break.

Event shapes used here (docs/json.md):

    {"type": "turn_start"}
    {"type": "message_end", "message": {"role": "assistant", "content": [...]}}
    {"type": "message_update", "usage": {...}, "assistantMessageEvent": {...}}
    {"type": "tool_execution_start", "toolCallId": ..., "toolName": ..., "args": ...}
    {"type": "tool_execution_end", "toolCallId": ..., "toolName": ...,
     "result": {"content": [...], "details": {...}}, "isError": false}
"""

import json
from typing import Iterable, Iterator, Optional


def parse_line(line: str) -> Optional[dict]:
    """One NDJSON line to an event, or None if it is blank or malformed."""
    line = line.strip()
    if not line:
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def iter_events(ndjson: str) -> Iterator[dict]:
    """Every well-formed event in an NDJSON stream."""
    for line in ndjson.splitlines():
        event = parse_line(line)
        if event is not None:
            yield event


def final_assistant_text(events: Iterable[dict]) -> str:
    """
    The text of the last assistant message in the stream.

    `message_end` carries the final authoritative message; `message_update`
    records are delta-only and are ignored here.
    """
    text = ""
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message", {})
        if message.get("role") != "assistant":
            continue
        parts = message.get("content", [])
        if isinstance(parts, list):
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    return text


def tool_result_details(event: dict) -> dict:
    """The `details` payload of a tool_execution_end event, or {}."""
    result = event.get("result")
    if not isinstance(result, dict):
        return {}
    details = result.get("details")
    return details if isinstance(details, dict) else {}


def is_error(event: dict) -> bool:
    """
    Whether a tool_execution_end reports failure.

    pi puts `isError` on the event; a tool that sets it on its own result should
    count too, so both are checked.
    """
    if event.get("isError"):
        return True
    result = event.get("result")
    return bool(isinstance(result, dict) and result.get("isError"))
