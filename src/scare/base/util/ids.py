"""Reproducible id minting.

``uuid4`` draws from ``os.urandom`` and is immune to ``random.seed``, so any id
that reaches routing, dict iteration order or tie-breaking makes a run
irreproducible. Ids are derived from an explicit key instead.

The established pattern is a per-agent monotonic counter rendered as
``{aid}/{seq}`` -- unique because aids are unique, and reproducible because the
sequence is driven by the agent's own message order. ``deterministic_uuid``
projects that key into a ``UUID`` for call sites whose id is UUID-typed.
"""

from __future__ import annotations

from uuid import UUID, uuid5

# Changing this namespace changes every derived id in the system.
SCARE_ID_NAMESPACE = UUID("1f5b9c2e-6a44-5d18-9e37-0c8a2b4d6f10")


def deterministic_uuid(*parts: object) -> UUID:
    """UUID derived from ``parts``; equal parts always give an equal UUID."""
    return uuid5(SCARE_ID_NAMESPACE, "/".join(str(p) for p in parts))
