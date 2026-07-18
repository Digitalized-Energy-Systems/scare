"""Exception-breadth + empty-coercion lock for safe_observe.

The per-site exception tuple is behavior-relevant (some control roles swallow
broad Exception, others only AttributeError/KeyError), so this pins that only
the passed exceptions are swallowed and that empty_to_none is opt-in.
"""

from __future__ import annotations

import pytest

from scare.base.util import safe_observe


class _B:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc

    def observe(self, aid):
        if self._exc is not None:
            raise self._exc
        return self._result


def test_returns_obs_unchanged():
    assert safe_observe(_B({"vm_pu": 1.0}), "a") == {"vm_pu": 1.0}


def test_default_swallows_attribute_and_key_error():
    assert safe_observe(_B(exc=KeyError("x")), "a") is None
    assert safe_observe(_B(exc=AttributeError()), "a") is None


def test_default_propagates_other_exceptions():
    with pytest.raises(ValueError):
        safe_observe(_B(exc=ValueError("boom")), "a")


def test_broad_exc_swallows_everything():
    assert safe_observe(_B(exc=RuntimeError()), "a", exc=Exception) is None


def test_narrow_exc_propagates_unlisted():
    with pytest.raises(KeyError):
        safe_observe(_B(exc=KeyError()), "a", exc=AttributeError)


def test_empty_to_none_false_keeps_empty():
    # voltage_droop relies on an empty dict passing through to its `or {}`.
    assert safe_observe(_B({}), "a") == {}
    assert safe_observe(_B(None), "a") is None


def test_empty_to_none_true_coerces_falsy():
    assert safe_observe(_B({}), "a", empty_to_none=True) is None
    assert safe_observe(_B(None), "a", empty_to_none=True) is None
    assert safe_observe(_B({"x": 1}), "a", empty_to_none=True) == {"x": 1}
