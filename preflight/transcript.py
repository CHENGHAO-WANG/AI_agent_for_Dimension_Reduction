"""A `claude -p --output-format stream-json` transcript, read as the tool calls it made.

The pre-flight judges the agent by what it did, and what it did is its tool calls: each
with its input and the result the harness returned. Lines that are not JSON are skipped,
so a transcript cut short by a crash still yields the calls before the cut.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: How much of a tool result a trimmed fixture keeps: enough for "moved to the
#: background" and for a refusal's first sentence.
TRIMMED_RESULT = 400


@dataclass
class Call:
    name: str
    input: dict[str, Any]
    result: str = ""
    is_error: bool = False


@dataclass
class Transcript:
    init: dict[str, Any] = field(default_factory=dict)
    calls: list[Call] = field(default_factory=list)
    result: dict[str, Any] | None = None


def _events(path: Path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def _text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or "")


def read(path: Path) -> Transcript:
    transcript = Transcript()
    by_id: dict[str, Call] = {}
    for event in _events(path):
        if event.get("type") == "system" and event.get("subtype") == "init":
            transcript.init = event
        elif event.get("type") == "result":
            transcript.result = event
        message = event.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                call = Call(block.get("name", ""), block.get("input") or {})
                transcript.calls.append(call)
                by_id[block.get("id", "")] = call
            elif block.get("type") == "tool_result" and block.get("tool_use_id") in by_id:
                call = by_id[block["tool_use_id"]]
                call.result = _text(block.get("content"))
                call.is_error = bool(block.get("is_error"))
    return transcript


def trim(source: Path, destination: Path) -> None:
    """Keep what the checks read: the init event's commands, every tool call and the
    start of its result, and the result event's totals."""
    lines = []
    for event in _events(source):
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            lines.append({"type": "system", "subtype": "init",
                          "slash_commands": event.get("slash_commands", [])})
        elif kind == "result":
            lines.append({"type": "result", **{k: event.get(k) for k in
                          ("num_turns", "duration_ms", "total_cost_usd")}})
        elif kind in ("assistant", "user") and isinstance(event.get("message"), dict):
            blocks = []
            for block in event["message"].get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    blocks.append({k: block.get(k) for k in ("type", "id", "name", "input")})
                elif block.get("type") == "tool_result":
                    blocks.append({"type": "tool_result", "tool_use_id": block.get("tool_use_id"),
                                   "is_error": bool(block.get("is_error")),
                                   "content": _text(block.get("content"))[:TRIMMED_RESULT]})
            if blocks:
                lines.append({"type": kind, "message": {"content": blocks}})
    Path(destination).write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
