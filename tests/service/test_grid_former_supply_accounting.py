"""A promoted island grid-former is supply everywhere, and is never curtailed.

``eval_full_v2_20260728-202054``'s microgrid arm lost 60 % of its gas because
``enable_grid_former_curtail_guard`` defaulted False and was set by *no* config —
so the former-rating registry stayed empty, every ``is_former`` branch was dead,
and a ``GridFormingSource``'s free ``mass_flow_kgs`` Var (init 0, flips positive
when it absorbs) read as zero supply and then as tiered demand. All 38 promoted
gas formers were curtailed to regulation 0.0 (0/38 in the clean twin) and the
holon's gas pool fell from 104 % to 51 % of demand, zeroing tiers 3 and 4.

Two halves are pinned here: the flag now defaults on (and is provably inert
without promotion), and the two L3 supply builders — which had *no* former
handling at all, guard or no guard — credit a former instead of billing it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from mango.express import topology as mango_topology

from scare.base.config import RestorationConfiguration
from scare.base.model import Sector
from scare.base.util import (
    apply_regulate,
    is_grid_former_child,
    lookup_grid_former_rating,
    register_grid_former_rating,
    register_priority,
    register_sector,
)
from scare.community import summary_coalition as sc
from scare.community import summary_publish as sp
from scare.community.summary_coalition import CoalitionManager
from scare.community.summary_publish import SummaryPublisher
from scare.service.balance.grid_former import (
    GRID_FORMER_SUPPLY_PROBE_SHARE,
    GridFormerPolicy,
)
from tests.conftest import MockBehavior

_CARRIERS = ("electricity", "water", "gas")

#: A former delivering 2.0 into a community whose single load draws 5.0. The
#: probe share is deliberately non-zero here (the shipped default is 0): the
#: pre-fix ``cap < 0`` path credits a *delivering* former by accident, so only a
#: credit that differs from plain ``|cap|`` proves the former branch is the one
#: running. Expected credit: ``2.0 + 0.25*(8.0-2.0) = 3.5``.
_FORMER_DELIVERED = 2.0
_FORMER_RATING = 8.0
_PROBE_SHARE = 0.25
_LOAD_DEMAND = 5.0


class _Ctx:
    def __init__(self) -> None:
        self.aid = "leader"
        self.current_timestamp = 5.0
        self.sent: list[Any] = []

    async def send_message(self, payload: Any, **_kw: Any) -> None:
        self.sent.append(payload)


def _behavior() -> MockBehavior:
    b = MockBehavior()
    b._scare_config = RestorationConfiguration()
    # Load: cap 5.0, fully served, tier 3.
    b.set_obs("load", {"p_mw": _LOAD_DEMAND, "regulation": 1.0, "priority": 3})
    register_sector(b, "load", Sector.ELECTRICITY)
    register_priority(b, "load", 3)
    # Former: free Var currently injecting (load convention -> negative).
    b.set_obs("gf", {"p_mw": -_FORMER_DELIVERED, "regulation": 1.0, "priority": 3})
    register_sector(b, "gf", Sector.ELECTRICITY)
    register_priority(b, "gf", 3)
    register_grid_former_rating(b, "gf", _FORMER_RATING)
    return b


def _role(
    behavior: MockBehavior,
    members: list[str],
    *,
    probe_share: float = _PROBE_SHARE,
) -> SimpleNamespace:
    return SimpleNamespace(
        sector=Sector.ELECTRICITY,
        behavior=behavior,
        context=_Ctx(),
        cp_budget_nominal=True,
        coalition_delivered_supply=True,
        inversion_tol=1e-9,
        _topology_tid="sector",
        _my_node_id=None,
        _member_node_ids={a: None for a in members},
        _version=SimpleNamespace(next=lambda: 1),
        _summary_changed=lambda *_a, **_k: True,
        _grid_former_policy=GridFormerPolicy(behavior, probe_share=probe_share),
    )


def _expected_credit(delivered: float) -> float:
    return delivered + _PROBE_SHARE * max(0.0, _FORMER_RATING - delivered)


def _patch_topology(
    module: Any, members: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        module,
        "topology_neighbors",
        lambda _role, tid=None: (
            [SimpleNamespace(aid=a) for a in members] if tid == "groups" else ["peer"]
        ),
    )
    monkeypatch.setattr(mango_topology, "topology_characteristic", lambda *_a: None)


class _NullChannel:
    def publish(self, *_a: Any, **_k: Any) -> None:
        pass


# --- The flag itself ------------------------------------------------------


def test_guard_is_on_by_default():
    """It was False and set by no config, which silently disabled the whole
    former policy on the one arm that needs it."""
    assert RestorationConfiguration().enable_grid_former_curtail_guard is True


def test_holon_summary_role_carries_the_policy_at_the_shipped_probe_share():
    """The two L3 builders reach the policy through their owning role; before
    this it existed only on the L2 negotiator."""
    from scare.community.summary import HolonSummaryRole

    role = HolonSummaryRole(MockBehavior(), Sector.GAS)
    assert isinstance(role._grid_former_policy, GridFormerPolicy)
    assert role._grid_former_policy.probe_share == GRID_FORMER_SUPPLY_PROBE_SHARE
    # Shipped default credits DELIVERED only — a positive share over-credits a
    # former sharing an island with a budgeted slack (see grid_former.py).
    assert GRID_FORMER_SUPPLY_PROBE_SHARE == 0.0


def test_guard_is_inert_without_promotion():
    """Flipping the default is only safe because nothing but
    ``apply_microgrid_islanding`` creates ``GridForming*`` models."""
    from experiment.scenarios import GRIDS

    net = GRIDS["simbench_lv"]()
    b = MockBehavior()
    b._scare_config = RestorationConfiguration()
    b._net = net
    assert not any(is_grid_former_child(b, f"child-{c.id}") for c in net.childs)
    assert lookup_grid_former_rating(b, "child-0") is None


def test_guard_sees_every_promoted_unit():
    from experiment.scenarios import GRIDS
    from scare.scenario.restoration import _maybe_register_grid_former

    net = GRIDS["simbench_lv"]()
    from experiment.scenarios import apply_microgrid_islanding

    promoted = apply_microgrid_islanding(
        net, carriers=_CARRIERS, promote_all_generators=True
    )
    b = MockBehavior()
    b._scare_config = RestorationConfiguration()
    b._net = net
    policy = GridFormerPolicy(b, probe_share=GRID_FORMER_SUPPLY_PROBE_SHARE)
    for child in net.childs:
        _maybe_register_grid_former(b, f"child-{child.id}", child)

    seen = sum(1 for c in net.childs if policy.is_former(f"child-{c.id}"))
    assert seen == sum(promoted.values()) > 0


# --- The two L3 supply builders ------------------------------------------


def test_l2_summary_credits_a_former_as_supply(monkeypatch: pytest.MonkeyPatch):
    b = _behavior()
    members = ["load", "gf"]
    _patch_topology(sp, members, monkeypatch)
    monkeypatch.setattr(
        sp.CrossSectorChannel, "for_behavior", staticmethod(lambda _b: _NullChannel())
    )
    role = _role(b, members)
    pub = SummaryPublisher(role)
    asyncio.run(pub._publish(force=True))

    summary = role.context.sent[-1]
    assert summary.supply_by_sector["electricity"] == pytest.approx(
        _expected_credit(_FORMER_DELIVERED)
    )
    # The former must not appear as demand in ANY tier — that is the shed path.
    assert summary.demand_by_sector_priority["electricity"] == {3: _LOAD_DEMAND}


def test_coalition_acceptance_credits_a_former_as_supply(
    monkeypatch: pytest.MonkeyPatch,
):
    b = _behavior()
    members = ["load", "gf"]
    _patch_topology(sc, members, monkeypatch)
    role = _role(b, members)
    acceptance = CoalitionManager(role)._local_acceptance("c1", (3,))

    assert acceptance is not None
    assert acceptance.supply_by_sector["electricity"] == pytest.approx(
        _expected_credit(_FORMER_DELIVERED)
    )
    assert acceptance.demand_by_sector_priority["electricity"] == {3: _LOAD_DEMAND}


@pytest.mark.parametrize("p_mw", [0.0, 1.5])
def test_a_former_is_never_billed_as_demand_whatever_its_var_reads(
    p_mw: float, monkeypatch: pytest.MonkeyPatch
):
    """The free Var reads 0 at init and flips positive when the former absorbs
    the island residual; neither may enter the tiered demand the holon sheds."""
    b = _behavior()
    b.set_obs("gf", {"p_mw": p_mw, "regulation": 1.0, "priority": 3})
    members = ["load", "gf"]
    _patch_topology(sp, members, monkeypatch)
    monkeypatch.setattr(
        sp.CrossSectorChannel, "for_behavior", staticmethod(lambda _b: _NullChannel())
    )
    role = _role(b, members)
    pub = SummaryPublisher(role)
    asyncio.run(pub._publish(force=True))

    summary = role.context.sent[-1]
    assert summary.demand_by_sector_priority["electricity"] == {3: _LOAD_DEMAND}
    # A non-delivering former still offers its probe headroom; the pre-fix
    # ``cap == 0 -> continue`` path contributed nothing at all.
    assert summary.supply_by_sector["electricity"] == pytest.approx(
        _expected_credit(0.0)
    )


# --- The curtail backstop -------------------------------------------------


def test_apply_regulate_refuses_to_curtail_a_former():
    from experiment.scenarios import GRIDS, apply_microgrid_islanding

    net = GRIDS["simbench_lv"]()
    apply_microgrid_islanding(net, carriers=_CARRIERS, promote_all_generators=True)
    former = next(
        c for c in net.childs if type(c.model).__name__ == "GridFormingSource"
    )
    aid = f"child-{former.id}"

    b = MockBehavior()
    b._scare_config = RestorationConfiguration()
    b._net = net
    b.set_obs(aid, {"mass_flow_kgs": 0.0, "regulation": 1.0})
    b.add_action(aid, "regulate")

    applied = apply_regulate(b, aid, 0.0, sector="gas", reason="test", timestamp=1.0)
    assert applied
    _, _, args, kwargs = b.action_log[-1]
    assert (list(args) + [kwargs.get("regulation")])[0] == 1.0
