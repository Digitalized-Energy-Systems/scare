"""Sole owner of the agent-aid grammar.

One home for building and parsing the ``node-<id>`` / ``child-<id>`` /
``branch-<hi>-<lo>`` aid strings, so the format lives in exactly one place.

Two DISTINCT parse fallbacks are preserved verbatim and must NOT be unified:
``node_id_from_aid`` returns None on a malformed id (detection's routing drops
it), while ``id_from_aid`` returns 0 (reconfiguration's branch-aid construction
tolerates it). Different callers, different intent.
"""

from __future__ import annotations

from typing import Any


def node_aid(node_id: Any) -> str:
    return f"node-{node_id}"


def child_aid(child_id: Any) -> str:
    return f"child-{child_id}"


def is_node_aid(aid: str) -> bool:
    return aid.startswith("node-")


def is_child_aid(aid: str) -> bool:
    return aid.startswith("child-")


def branch_aid(branch_id: tuple) -> str:
    a, b = branch_id[0], branch_id[1]
    hi, lo = (a, b) if a > b else (b, a)
    return f"branch-{hi}-{lo}"


def addr_aid(addr: Any) -> str:
    return getattr(addr, "aid", str(addr))


def node_id_from_aid(aid: str) -> int | None:
    """Detection semantics: ``node-<int>`` -> int, else None (malformed dropped)."""
    if not aid.startswith("node-"):
        return None
    try:
        return int(aid.split("-", 1)[1])
    except ValueError:
        return None


def id_from_aid(aid: str) -> int:
    """Reconfiguration semantics: trailing int, else 0 (tolerant of malformed)."""
    try:
        return int(aid.split("-")[-1])
    except ValueError:
        return 0


def branch_aid_from_addrs(addr_a: Any, addr_b: Any) -> str:
    a, b = id_from_aid(addr_aid(addr_a)), id_from_aid(addr_aid(addr_b))
    hi, lo = (a, b) if a > b else (b, a)
    return f"branch-{hi}-{lo}"
