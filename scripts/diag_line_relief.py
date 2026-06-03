"""Root-cause diagnostic for the branch-106-7 line overload (seed 0).

Builds the line_stress scenario, applies seed-0's two failures, then runs
plain power flows under counterfactual shed policies to answer: is the
residual overload priority-blocked (tier-1 through-load) or structural?
"""
from __future__ import annotations
from collections import deque, defaultdict

from monee import run_energy_flow
from monee.model.child import ExtPowerGrid, PowerLoad
from experiment.restoration import GRIDS, apply_line_stress, assign_load_priorities

SEED = 0
FAILS = [(355, 372, 0), (239, 164, 0)]   # seed-0 failures from failures.json
TARGET = (106, 7)                        # binding line branch-106-7


def build():
    net = GRIDS["simbench_lv"]()
    apply_line_stress(net, load_scale=1.8, ampacity_scale=0.5)
    return net


def load_children(net):
    """node_id -> list of (child, PowerLoad model)."""
    out = defaultdict(list)
    for c in net.childs:
        if isinstance(c.model, PowerLoad):
            out[c.node_id].append(c)
    return out


def slack_nodes(net):
    return {c.node_id for c in net.childs if isinstance(c.model, ExtPowerGrid)}


def active_el_adj(net, exclude_edge=None, failed=()):
    """Undirected electricity adjacency, excluding failed + one edge."""
    adj = defaultdict(set)
    failed_set = {tuple(f) for f in failed}
    for b in net.branches:
        bid = tuple(b.id)
        if bid in failed_set:
            continue
        a, c = b.id[0], b.id[1]
        if exclude_edge and {a, c} == set(exclude_edge[:2]):
            continue
        adj[a].add(c)
        adj[c].add(a)
    return adj


def reachable(adj, sources):
    seen = set(sources)
    dq = deque(sources)
    while dq:
        n = dq.popleft()
        for m in adj[n]:
            if m not in seen:
                seen.add(m)
                dq.append(m)
    return seen


def line_loading(net):
    """{(a,b,c): loading_percent (in %)} after a fresh power flow."""
    res = run_energy_flow(net)
    solved = getattr(res, "network", net)
    out = {}
    for b in solved.branches:
        try:
            lp = float(b.model.loading_percent)
        except Exception:
            continue
        if 0.0 < abs(lp) <= 5.0:
            lp *= 100.0
        out[tuple(b.id)] = abs(lp)
    return out


def set_reg(child, val):
    m = child.model
    if hasattr(m, "regulation"):
        m.regulation = val
    else:
        m.p_mw = m.p_mw * val


def main():
    net = build()
    prios = assign_load_priorities(net, seed=SEED, distribution="skewed")
    slack = slack_nodes(net)

    # Downstream set of TARGET = loads that lose their path to slack when
    # TARGET is removed (post-failure topology).
    adj_cut = active_el_adj(net, exclude_edge=TARGET, failed=FAILS)
    reach = reachable(adj_cut, slack)
    node_loads = load_children(net)
    downstream = []  # (child, tier, p_mw, node)
    for node, kids in node_loads.items():
        if node in reach:
            continue
        for c in kids:
            tier = prios.get(f"child-{c.id}", prios.get(c.model.__dict__.get("name", ""), 0))
            downstream.append((c, tier, float(c.model.p_mw), node))

    print(f"slack nodes: {sorted(slack)}")
    print(f"TARGET branch {TARGET}: downstream load count = {len(downstream)}")
    tier_hist = defaultdict(lambda: [0, 0.0])
    for _, t, p, _ in downstream:
        tier_hist[t][0] += 1
        tier_hist[t][1] += p
    print("downstream tier histogram (tier: count, sum_p_mw):")
    for t in sorted(tier_hist):
        print(f"   tier {t}: n={tier_hist[t][0]}  sum_p={tier_hist[t][1]:.4f} MW")
    print(f"total downstream demand = {sum(p for _,_,p,_ in downstream):.4f} MW")

    # Deactivate failed branches for the physics solves.
    for f in FAILS:
        try:
            net.deactivate_by_id(tuple(f), "branch")
        except Exception as e:
            print("deactivate failed", f, e)

    def loading_of_target():
        L = line_loading(net)
        # match TARGET regardless of the 3rd id element
        for k, v in L.items():
            if (k[0], k[1]) == TARGET or (k[1], k[0]) == TARGET:
                return v
        return None

    # (a) all loads full
    for c, *_ in downstream:
        set_reg(c, 1.0)
    La = loading_of_target()

    # (b) shed ALL downstream loads
    for c, *_ in downstream:
        set_reg(c, 0.0)
    Lb = loading_of_target()

    # (c) shed only NON-tier-1 downstream (keep tier-1 at full)
    for c, t, *_ in downstream:
        set_reg(c, 0.0 if t != 1 else 1.0)
    Lc = loading_of_target()

    # (d) restore full for reference
    for c, *_ in downstream:
        set_reg(c, 1.0)

    print()
    print(f"branch-106-7 loading_percent (post-failure power flow):")
    print(f"  (a) all downstream loads FULL        : {La:.1f} %")
    print(f"  (b) ALL downstream loads shed to 0   : {Lb:.1f} %")
    print(f"  (c) only NON-tier-1 downstream shed  : {Lc:.1f} %  (tier-1 kept full)")


if __name__ == "__main__":
    main()
