"""Guard: methods the scenario builder calls on a role must exist on that role.

``restoration.py`` wires failure-event handlers through mango's ``behavior_in``,
which dispatches to a callback typed ``(role: SomeRole, event)``. Those calls are
resolved at event time, so a method that moves from the role onto one of its
helper objects raises ``AttributeError`` only when a failure actually fires --
and mango's ``Role`` has no ``__getattr__`` fallback to soften it.

That is how ``trigger_balance_negotiation`` went missing: the flex refactor left
it on ``TriggerCoordinator`` while ``restoration.py`` kept calling it on
``EnergyBalanceNegotiator``, breaking the primary failure-response path with a
green test suite.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "scare"
_RESTORATION = _SRC / "scenario" / "restoration.py"


def _methods_by_class() -> dict[str, set[str]]:
    """All method names defined on each class across ``src/scare``.

    Names are pooled per class name (not per module): the check only needs to
    know that *some* class of that name defines the attribute.
    """
    out: dict[str, set[str]] = {}
    for path in _SRC.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                out.setdefault(node.name, set()).update(
                    b.name
                    for b in node.body
                    if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
                # Attributes assigned in the body count as public surface too.
                for b in node.body:
                    if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
                        out[node.name].add(b.target.id)
    return out


def _annotated_role_calls() -> list[tuple[int, str, str]]:
    """``(line, class_name, method)`` for calls on an annotated role parameter."""
    tree = ast.parse(_RESTORATION.read_text(encoding="utf-8"))
    found: list[tuple[int, str, str]] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        annotated = {
            a.arg: a.annotation.id
            for a in fn.args.args
            if isinstance(getattr(a, "annotation", None), ast.Name)
        }
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
            ):
                cls = annotated.get(node.func.value.id)
                if cls:
                    found.append((node.lineno, cls, node.func.attr))
    return found


def test_scenario_only_calls_methods_the_role_defines():
    classes = _methods_by_class()
    missing = [
        f"restoration.py:{line}: {cls}.{meth}() -- not defined on {cls}"
        for line, cls, meth in _annotated_role_calls()
        # Only classes we can actually see; ignore inherited/framework surface.
        if cls in classes and meth not in classes[cls] and not meth.startswith("_")
    ]
    assert not missing, (
        "restoration.py calls role methods that do not exist. These raise "
        "AttributeError only when the event fires.\n  " + "\n  ".join(missing)
    )
