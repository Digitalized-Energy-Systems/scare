"""Write-path guards for offline re-grading of a completed campaign.

A re-grade rewrites ``result.json`` in place. Pointed at the wrong directory it
destroys a run of record that cannot be reconstructed: the oracle arm's claim
payloads are produced in memory during the solve and are *not* recomputable
from the persisted per-task artefacts. There is no undo.

Two independent conditions must both hold before any such script may write:

1. the caller passed ``--i-have-a-copy`` (an explicit human acknowledgement), and
2. the target resolves outside every protected root and off any network share.

The HPC runner is deliberately not gated — writing ``result.json`` into the
campaign directory is its job. This module is for tooling that rewrites a
campaign *after* it has been run.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePath

#: Roots holding runs of record. A re-grade must never resolve inside one.
#: Extend via ``SCARE_PROTECTED_ROOTS`` (os.pathsep-separated) rather than
#: editing this list, so a site-specific mount does not need a code change.
PROTECTED_ROOTS: tuple[str, ...] = (
    "Y:/fs/dss/home/towo7024/SCARE/scare/experiment/_runs",
)


class ProtectedCampaignError(RuntimeError):
    """Raised when a destructive operation targets a run of record."""


def _protected_roots() -> list[PurePath]:
    roots = list(PROTECTED_ROOTS)
    extra = os.environ.get("SCARE_PROTECTED_ROOTS", "")
    roots += [r for r in extra.split(os.pathsep) if r.strip()]
    return [PurePath(str(r).replace("\\", "/").rstrip("/").lower()) for r in roots]


def _is_network_path(path: Path) -> bool:
    """UNC paths, and mapped drives that Windows reports as remote."""
    text = str(path).replace("\\", "/")
    if text.startswith("//"):
        return True
    drive = os.path.splitdrive(str(path))[0]
    if not drive:
        return False
    try:  # Windows only; absent elsewhere, where UNC detection is enough.
        import ctypes

        get_drive_type = ctypes.windll.kernel32.GetDriveTypeW  # type: ignore[attr-defined]
    except (ImportError, AttributeError, OSError):
        return False
    DRIVE_REMOTE = 4
    return int(get_drive_type(f"{drive}\\")) == DRIVE_REMOTE


def is_protected(campaign_dir: Path | str) -> bool:
    """True when ``campaign_dir`` is a run of record or lives on a share."""
    path = Path(campaign_dir).expanduser()
    try:
        path = path.resolve()
    except OSError:
        path = path.absolute()
    if _is_network_path(path):
        return True
    norm = PurePath(str(path).replace("\\", "/").rstrip("/").lower())
    for root in _protected_roots():
        if norm == root or root in norm.parents:
            return True
    return False


def assert_regradable(
    campaign_dir: Path | str, *, acknowledged: bool, read_only: bool = False
) -> Path:
    """Validate a re-grade target, or raise :class:`ProtectedCampaignError`.

    Returns the resolved directory so callers can use it directly rather than
    re-deriving a path the guard has not seen.

    ``read_only`` (a ``--dry-run``) skips both checks: they exist to stop an
    unrecoverable write, and refusing to let anyone *look* at what a re-grade
    would do to the run of record only pushes that inspection into an ad-hoc
    script with no guard at all. Callers must not pass it on a writing path.
    """
    path = Path(campaign_dir).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    if not resolved.is_dir():
        raise ProtectedCampaignError(f"Not a directory: {resolved}")
    if read_only:
        return resolved
    if not acknowledged:
        raise ProtectedCampaignError(
            "Re-grading rewrites result.json in place and cannot be undone. "
            "Pass --i-have-a-copy to confirm a byte-identical copy exists."
        )
    if is_protected(resolved):
        raise ProtectedCampaignError(
            f"Refusing to re-grade {resolved}: it is a run of record or lives "
            "on a network share. Copy it to local disk and point at the copy."
        )
    return resolved


def add_regrade_arguments(parser) -> None:
    """Attach the acknowledgement flag to a re-grade script's parser."""
    parser.add_argument(
        "--i-have-a-copy",
        dest="acknowledged",
        action="store_true",
        help=(
            "Confirm a byte-identical copy of the campaign exists. Required: "
            "re-grading rewrites result.json in place."
        ),
    )
