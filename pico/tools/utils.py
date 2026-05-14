"""Shared JSON response helpers for tool handlers.

All tool modules can import ``_success`` and ``_error`` from here
instead of duplicating them locally.
"""

from __future__ import annotations

import json
from typing import Any


def success(data: Any = None, message: str = "") -> str:
    """Return a success JSON response (new clean API).

    Nests ``data`` under a ``"data"`` key.  Prefer this for new tools.
    """
    resp: dict[str, Any] = {"success": True}
    if data is not None:
        resp["data"] = data
    if message:
        resp["message"] = message
    return json.dumps(resp, ensure_ascii=False)


def error(message: str, details: Any = None) -> str:
    """Return an error JSON response (new clean API).

    Nests extra info under a ``"details"`` key.  Prefer this for new tools.
    """
    resp: dict[str, Any] = {"success": False, "error": message}
    if details is not None:
        resp["details"] = details
    return json.dumps(resp, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Backward-compatible helpers matching the existing tool file signatures.
# These spread dict data at the top level of the response object.
# ---------------------------------------------------------------------------

def _success(data: Any) -> str:
    """Return a success JSON response (legacy flat layout).

    If *data* is a dict its keys are merged into the top-level object.
    Otherwise it is wrapped as ``{"result": data}``.
    """
    return json.dumps(
        {"success": True, **(data if isinstance(data, dict) else {"result": data})},
        ensure_ascii=False,
    )


def _error(msg: str) -> str:
    """Return an error JSON response (legacy flat layout)."""
    return json.dumps({"success": False, "error": msg}, ensure_ascii=False)
