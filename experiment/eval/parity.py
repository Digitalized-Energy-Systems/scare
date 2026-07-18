"""Parity-harness primitive for the architecture refactor: canonicalize the
non-deterministic ids so a golden A/B byte-diff reflects real behavior changes,
not RNG noise.

Several coordination ids are minted from an *unseeded* ``uuid4`` — the auction id
(constraints.py), holon/community ids (holonic.py, repartition.py), and path/
community search ids (reconfiguration.py, restoration.py). Two otherwise-identical
runs therefore emit different uuid strings, so a raw byte-diff of the event /
negotiation logs is spuriously non-empty. ``normalize_ids`` rewrites every
uuid4-shaped token to a stable placeholder assigned in order of first appearance,
so identical behavior -> identical normalized text, while a genuine change (an
extra negotiation, a reordered auction) still shows up.

Mango's negotiation ``nid`` is already deterministic (``_neg_seq``), so it is not
uuid-shaped and is left untouched.

Usage (prove the harness empty on HEAD before any refactor step, then gate each
step on it):

    python -m experiment.eval.parity run_a/events.csv run_b/events.csv
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def normalize_ids(text: str) -> str:
    """Replace every uuid4-shaped token with a first-appearance-ordered
    placeholder (``uuid-0``, ``uuid-1``, ...). Deterministic and order-sensitive:
    identical behavior normalizes to identical text."""
    mapping: dict[str, str] = {}

    def _sub(m: re.Match[str]) -> str:
        tok = m.group(0)
        if tok not in mapping:
            mapping[tok] = f"uuid-{len(mapping)}"
        return mapping[tok]

    return _UUID_RE.sub(_sub, text)


def normalize_file(path: str | Path) -> str:
    return normalize_ids(Path(path).read_text(encoding="utf-8"))


def diff_normalized(path_a: str | Path, path_b: str | Path) -> list[str]:
    """Unified diff of two files after id-normalization; empty list == parity."""
    a = normalize_file(path_a).splitlines(keepends=True)
    b = normalize_file(path_b).splitlines(keepends=True)
    return list(
        difflib.unified_diff(a, b, fromfile=str(path_a), tofile=str(path_b))
    )


def main() -> int:
    import sys

    if len(sys.argv) != 3:
        print("usage: python -m experiment.eval.parity <file_a> <file_b>")
        return 2
    diff = diff_normalized(sys.argv[1], sys.argv[2])
    if not diff:
        print("PARITY: identical after id-normalization")
        return 0
    print("".join(diff), end="")
    print(f"\nPARITY FAIL: {sum(1 for d in diff if d[:1] in '+-' and d[:3] not in ('+++', '---'))} changed lines")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
