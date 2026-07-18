"""Lock for CaptureWindow: one snapshot per window on the first failed solve,
and disarm/re-arm RESTORES the original _solve_physics seam (no wrapper stacking,
the gap the old dict-based disarm left open)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from scare.base.runtime.infeasibility_capture import CaptureWindow


class _FailResult:
    failed = True
    error = "infeasible LP"


class _Behavior:
    def __init__(self) -> None:
        self._net = SimpleNamespace(branches=[], nodes=[], childs=[])
        self._last_energy_flow_t = 1.5
        self.solve_calls = 0
        self._solve_physics = self._real_solve  # stable instance-attr identity

    def _real_solve(self, dt_h):
        self.solve_calls += 1
        return _FailResult()


def test_captures_one_snapshot_on_first_failure(tmp_path):
    b = _Behavior()
    out = tmp_path / "snap.json"
    win = CaptureWindow()
    win.arm(b, out)
    b._solve_physics(0.1)  # fails -> one capture
    assert b.solve_calls == 1
    data = json.loads(out.read_text())
    assert data["result"]["success"] is False
    assert data["sim_time_s"] == 1.5
    # A second failure in the same window does not re-capture.
    out.write_text("SENTINEL")
    b._solve_physics(0.1)
    assert out.read_text() == "SENTINEL"


def test_disarm_restores_original_seam(tmp_path):
    b = _Behavior()
    original = b._solve_physics
    win = CaptureWindow()
    win.arm(b, tmp_path / "s.json")
    assert b._solve_physics is not original  # patched
    win.disarm()
    assert b._solve_physics is original  # restored (not left in pass-through)


def test_rearm_does_not_stack_wrappers(tmp_path):
    b = _Behavior()
    original = b._solve_physics
    win = CaptureWindow()
    win.arm(b, tmp_path / "a.json")
    win.arm(b, tmp_path / "b.json")  # re-arm same instance
    win.disarm()
    assert b._solve_physics is original  # only one level was ever installed


def test_handle_restore_is_idempotent(tmp_path):
    b = _Behavior()
    original = b._solve_physics
    win = CaptureWindow()
    handle = win.arm(b, tmp_path / "s.json")
    handle.restore()
    handle.restore()  # no raise, no double-effect
    assert b._solve_physics is original
