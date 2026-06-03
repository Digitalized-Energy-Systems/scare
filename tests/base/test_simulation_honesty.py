"""Empirical guards that SCARE runs as an honest mango simulation.

Honest = every coroutine spawned in response to a message goes through
mango's scheduler (``context.schedule_instant_task`` / friends), not bare
``asyncio.create_task``.  Bare ``create_task`` would put the work on a
"side track" off the simulation clock: mango's ``step_simulation`` can
return before the task completes, so its sends land in a later step than
the message that caused them.  See the rationale block in
``distributed_resource_optimization/carrier/mango.py:132-143`` and the
upstream regression test ``test_mango_simulation_is_clock_gated_no_side_track``.

Two checks here:

1. ``test_admm_handler_routes_through_mango_scheduler`` — direct
   spy on ``ScareDistributedOptimizationRole._handle_optimization``: it
   must call ``context.schedule_instant_task`` and must not call
   ``asyncio.create_task``.

2. ``test_no_bare_create_task_in_scare_source`` — static guard that
   re-finds any ``asyncio.create_task`` / ``asyncio.ensure_future`` /
   ``loop.create_task`` introduced into ``src/scare`` in the future.
   The single legitimate exception today is the carrier driver pattern
   in dro upstream (``CoordinatorRole._handle_start``), which is
   *outside* SCARE's tree.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from distributed_resource_optimization.algorithm.admm.core import ADMMMessage
from distributed_resource_optimization.carrier.mango import _CarrierRequest

from scare.base.admm import ScareDistributedOptimizationRole


# ---------------------------------------------------------------------------
# 1) Direct handler spy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admm_handler_routes_through_mango_scheduler():
    """The handler must schedule via ``context.schedule_instant_task``,
    not ``asyncio.create_task``.  Without this, ADMM message handling
    runs off the simulation clock.
    """
    role = ScareDistributedOptimizationRole.__new__(ScareDistributedOptimizationRole)
    role.algorithm = MagicMock()
    role._carrier = MagicMock()
    role._context = MagicMock()  # context is a @property → set the backing attr

    msg = ADMMMessage.__new__(ADMMMessage)
    meta: dict = {"sender_id": "peer-1"}

    tasks_before = asyncio.all_tasks()
    with patch("asyncio.create_task") as create_task_spy:
        role._handle_optimization(msg, meta)
    tasks_after = asyncio.all_tasks()

    assert role.context.schedule_instant_task.called, (
        "handler must route through mango scheduler "
        "(context.schedule_instant_task), not bare asyncio.create_task"
    )
    assert not create_task_spy.called, (
        "handler called asyncio.create_task — that bypasses mango's "
        "scheduler and creates an off-clock side-track task"
    )

    # The coroutine passed to schedule_instant_task is the one we want
    # mango to track; verify exactly one was passed.
    (call,) = role.context.schedule_instant_task.call_args_list
    (coro,) = call.args
    assert asyncio.iscoroutine(coro), (
        f"schedule_instant_task must receive a coroutine; got {type(coro)!r}"
    )
    coro.close()  # we mocked the carrier — closing prevents 'never-awaited' warning

    new_tasks = tasks_after - tasks_before
    assert not new_tasks, (
        f"handler created orphan asyncio tasks outside mango's scheduler: "
        f"{[t.get_name() for t in new_tasks]}"
    )


@pytest.mark.asyncio
async def test_admm_handler_unwraps_carrier_request_through_scheduler():
    """The wrapped-request branch (from ``send_awaitable``) must also
    route through the scheduler — same guarantee for requests as for
    plain messages.
    """
    role = ScareDistributedOptimizationRole.__new__(ScareDistributedOptimizationRole)
    role.algorithm = MagicMock()
    role._carrier = MagicMock()
    role._context = MagicMock()  # context is a @property → set the backing attr

    inner = ADMMMessage.__new__(ADMMMessage)
    req = _CarrierRequest(content=inner, request_id="req-xyz")

    with patch("asyncio.create_task") as create_task_spy:
        role._handle_optimization(req, {"sender_id": "peer-1"})

    assert role.context.schedule_instant_task.called
    assert not create_task_spy.called

    # The scheduled coroutine carries the unwrapped meta with the request_id.
    (call,) = role.context.schedule_instant_task.call_args_list
    (coro,) = call.args
    coro.close()


# ---------------------------------------------------------------------------
# 2) Static guard over the whole src/scare tree
# ---------------------------------------------------------------------------


_FORBIDDEN = re.compile(
    r"\b(asyncio\.(?:create_task|ensure_future)|loop\.create_task)\b"
)
_TIME_FORBIDDEN = re.compile(
    r"\b(time\.(?:time|monotonic|perf_counter)|loop\.time)\("
)


def _iter_scare_sources() -> list[Path]:
    root = Path(__file__).resolve().parents[2] / "src" / "scare"
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _strip_comments_and_strings(src: str) -> str:
    # Crude scrub: drop line comments and triple-quoted docstring blocks
    # so a match inside a comment or docstring doesn't trip the guard.
    # Triple-quoted strings (greedy across lines).
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    # Line comments.
    src = re.sub(r"#[^\n]*", "", src)
    return src


def test_no_bare_create_task_in_scare_source():
    """No file under ``src/scare`` may introduce ``asyncio.create_task``,
    ``asyncio.ensure_future``, or ``loop.create_task`` — those bypass
    mango's scheduler.  Use ``self.context.schedule_instant_task(...)``
    instead.
    """
    offenders: list[str] = []
    for path in _iter_scare_sources():
        text = _strip_comments_and_strings(path.read_text(encoding="utf-8"))
        for match in _FORBIDDEN.finditer(text):
            # Recover the original line number from the unstripped source.
            line = path.read_text(encoding="utf-8").count(
                "\n", 0, _locate_in_original(path, match.group(0))
            ) + 1
            offenders.append(f"{path}:{line}: {match.group(0)}")

    assert not offenders, (
        "Found scheduler-bypassing calls in src/scare. Replace with "
        "self.context.schedule_instant_task(...).\n  "
        + "\n  ".join(offenders)
    )


def test_no_wallclock_time_in_scare_source():
    """``time.time()`` / ``time.monotonic()`` / ``loop.time()`` are
    wall-clock probes — they do not track ``world.clock.time`` under
    discrete-step simulation.  Use ``self.context.current_timestamp``
    or ``world.clock.time``.
    """
    offenders: list[str] = []
    for path in _iter_scare_sources():
        text = _strip_comments_and_strings(path.read_text(encoding="utf-8"))
        for match in _TIME_FORBIDDEN.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path}:{line_no}: {match.group(0)}")

    assert not offenders, (
        "Found wall-clock time probes in src/scare. Use "
        "self.context.current_timestamp instead.\n  "
        + "\n  ".join(offenders)
    )


def _locate_in_original(path: Path, needle: str) -> int:
    return path.read_text(encoding="utf-8").find(needle)
