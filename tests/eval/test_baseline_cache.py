"""Lock for BaselineCache — must preserve the deepcopy-on-write/read contract of
the module dicts it replaced (a returned served map is mutable without poisoning
the cache; regs are side-cached)."""

from __future__ import annotations

from experiment.eval.oracle import BaselineCache


def test_get_returns_a_deep_copy():
    c = BaselineCache()
    served = {"child-1": {"nested": [1, 2]}}
    c.put("k", served, {"child-1": 0.5})
    got = c.get("k")
    assert got == served
    got["child-1"]["nested"].append(99)  # mutate the returned copy
    assert c.get("k") == served  # cache not poisoned


def test_put_snapshots_the_input():
    c = BaselineCache()
    served = {"a": [1]}
    c.put("k", served, None)
    served["a"].append(2)  # mutate the original after put
    assert c.get("k") == {"a": [1]}


def test_get_regs_and_falsy_regs_not_stored():
    c = BaselineCache()
    c.put("k", {}, {"child-1": 0.7})
    assert c.get_regs("k") == {"child-1": 0.7}
    assert c.get_regs("missing") is None
    c.put("k2", {}, None)
    assert c.get_regs("k2") is None
    c.put("k3", {}, {})  # empty regs are falsy -> not stored (matches `if regs`)
    assert c.get_regs("k3") is None


def test_miss_and_clear():
    c = BaselineCache()
    assert c.get("nope") is None
    c.put("k", {"a": 1}, {"r": 1.0})
    c.clear()
    assert c.get("k") is None
    assert c.get_regs("k") is None
