"""Minimal Server-Sent Events encoding (no extra dependency)."""

from __future__ import annotations

import json
from typing import Any


def sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    # JSON never contains raw newlines, so a single `data:` line is always valid.
    return f"event: {event}\ndata: {payload}\n\n"


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # disable proxy buffering (nginx) so tokens flush immediately
}
