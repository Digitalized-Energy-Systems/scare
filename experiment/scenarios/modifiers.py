"""In-place scenario modifiers (``apply_*``) and the slack-budget helper."""

from monee import enable_islanding
from monee.model.child import (
    ExtHydrGrid,
    ExtPowerGrid,
    HeatLoad,
    PowerGenerator,
    PowerLoad,
    Sink,
    Source,
)
from monee.model.extension import (
    GasLinepack,
    GridFormingGenerator,
    GridFormingSource,
    LumpedThermalCapacitance,
)

_SLACK_LP_HEADROOM_FACTOR: float = 10.0


def apply_microgrid_islanding(
    mes: "object",
    *,
    carriers: "frozenset[str] | set[str] | tuple[str, ...] | list[str]" = (
        "electricity",
        "water",
        "gas",
    ),
    promote_all_generators: bool = False,
    grid_former_aids: "tuple[str, ...] | list[str] | set[str]" = (),
) -> dict[str, int]:
    """Enable monee's islanding extension on *mes* and optionally convert
    selected generator-class children into ``GridForming*`` types so the
    extension has grid-formers to anchor sub-islands on.

    Args:
        carriers:
            Carrier names to enable islanding on (``"electricity"``,
            ``"water"``, ``"gas"``). Forwards True/None per carrier;
            does not customise islanding modes.
        promote_all_generators:
            When True, every eligible generator-class child
            (PowerGenerator, gas/water Source) becomes the corresponding
            ``GridFormingGenerator`` / ``GridFormingSource`` so any
            sub-island containing one is solvable on its own. When False,
            only children whose aid is in ``grid_former_aids`` are promoted.
        grid_former_aids:
            Per-child opt-in aids of the form ``"child-{id}"``; ignored
            when ``promote_all_generators`` is True.

    Returns:
        ``{carrier: n_promoted}`` — children converted per sector.

    Side effects:
        Calls :func:`monee.enable_islanding`; the resulting
        ``NetworkIslandingConfig`` is attached and respected by every
        subsequent solve (Pyomo / Gekko).

    Notes:
        Promotion uses approximate ratings from the child's setpoint
        magnitude — sufficient for eligibility (the LP needs a leading
        reference per island, not a precise capacity).
    """
    carrier_set = frozenset(carriers)
    if not carrier_set:
        return {"electricity": 0, "water": 0, "gas": 0}

    promote_aids = set(grid_former_aids) if grid_former_aids else set()
    counters = {"electricity": 0, "water": 0, "gas": 0}

    for child in mes.childs:
        aid = f"child-{child.id}"
        if not (promote_all_generators or aid in promote_aids):
            continue
        try:
            node = mes.node_by_id(child.node_id)
        except Exception:
            continue
        grid_name = str(getattr(node.grid, "name", "") or "").lower()
        m = child.model

        if (
            isinstance(m, PowerGenerator)
            and "power" in grid_name
            and "electricity" in carrier_set
        ):
            p_max = max(1e-6, abs(float(getattr(m, "p_mw", 0.0) or 0.0)))
            # Full four-quadrant reactive capability (|Q| up to the active
            # rating). A grid-forming inverter must supply the island's whole
            # reactive demand + line charging while pinning bus voltage; the
            # old 0.1*p_max headroom made a severed island's Q-balance
            # infeasible (IIS: node Q-balance vs gf q_mvar bound), so a
            # post-failure island solve failed every step. Feasibility
            # threshold measured at ~0.2*p_max; p_max keeps ample margin.
            q_max = max(1e-6, abs(float(getattr(m, "q_mvar", 0.0) or 0.0)), p_max)
            child.model = GridFormingGenerator(
                p_mw_max=p_max,
                q_mvar_max=q_max,
                vm_pu=1.0,
            )
            counters["electricity"] += 1
        elif isinstance(m, Source):
            # Sources live on both gas and water nodes; route by parent
            # grid so the right carrier is counted and anchored.
            mass_max = max(1e-6, abs(float(getattr(m, "mass_flow_kgs", 0.0) or 0.0)))
            if "gas" in grid_name and "gas" in carrier_set:
                child.model = GridFormingSource(
                    pressure_pu=1.0,
                    mass_flow_max_kgs=mass_max,
                )
                counters["gas"] += 1
            elif "water" in grid_name and "water" in carrier_set:
                child.model = GridFormingSource(
                    pressure_pu=1.0,
                    t_k=356.0,
                    mass_flow_max_kgs=mass_max,
                )
                counters["water"] += 1

    # None per-carrier leaves that carrier's islanding disabled.
    enable_islanding(
        mes,
        electricity=True if "electricity" in carrier_set else None,
        gas=True if "gas" in carrier_set else None,
        water=True if "water" in carrier_set else None,
    )
    return counters


def apply_temporal_extensions(
    mes: "object",
    *,
    linepack: bool = False,
    ltc: bool = False,
    ltc_default_t_init: float | None = None,
    ltc_first_step_steady_state: bool = False,
) -> dict[str, int]:
    """Attach monee's temporal-storage extensions to *mes*.

    ``GasLinepack`` (per-pipe ``linepack_kg`` / ``net_pack_kgs`` vars) and
    ``LumpedThermalCapacitance`` (per-water-junction ρ·V thermal mass)
    activate their inter-step dynamics under monee's ``Stepper`` (the
    persistent physics stepper this campaign drives). The single-step
    ``energyflow`` path pins ``net_pack_kgs = 0`` and emits no LTC inertia
    term, so under it attaching them is only an obs-schema compatibility
    check.

    ``ltc_first_step_steady_state`` drops the LTC inertia equation on the
    FIRST step (emits ``net_heat == 0`` instead), so the first solve settles
    the junctions onto the network's steady-state thermal field before
    inertia kicks in on step 2. Without it, every LTC junction cold-starts at
    the ``t_pu`` per-unit default (≈ reference temperature) regardless of the
    real supply temperature, which then reads as an instantaneous temperature
    collapse rather than thermal inertia. MIP-backend only (the campaign's
    stepper resolves to gurobipy, which satisfies this).

    Returns ``{"linepack_pipes": n, "ltc_junctions": n}`` so the caller
    can confirm the extension found something to attach to.
    """
    counters = {"linepack_pipes": 0, "ltc_junctions": 0}
    if linepack:
        ext = GasLinepack()
        mes.add_extension(ext)
        # ``prepare`` materialises per-branch state; call it early for
        # the counter (the solver also calls it).
        try:
            ext.prepare(mes)
        except Exception:  # noqa: BLE001 — count is best-effort
            pass
        counters["linepack_pipes"] = len(getattr(ext, "_active_branches", ()))
    if ltc:
        kwargs: dict = {"first_step_steady_state": bool(ltc_first_step_steady_state)}
        if ltc_default_t_init is not None:
            kwargs["default_t_init"] = float(ltc_default_t_init)
        ext = LumpedThermalCapacitance(**kwargs)
        mes.add_extension(ext)
        try:
            ext.prepare(mes)
        except Exception:  # noqa: BLE001
            pass
        counters["ltc_junctions"] = len(getattr(ext, "_ltc_rho_v", {}))
    return counters


def apply_slack_budget(
    mes,
    fraction: float,
    *,
    hard_cap_carriers: "tuple[str, ...] | list[str] | set[str] | None" = None,
) -> None:
    """Register the operator's slack budget on the slack agents.

    The slack budget is a soft target the MAS enforces, not a hard LP
    bound (which could make the energy-flow LP infeasible after a failure
    shifts the imbalance).

    ``hard_cap_carriers`` (subset of ``{"electricity", "gas"}``) makes the
    budget a HARD LP import cap for those carriers instead of the wide
    feasibility envelope: the slack can supply at most ``budget`` and any
    deficit above it must come from stored energy (gas linepack) or be shed.
    This is what turns the temporal experiment's gas linepack into a
    load-bearing lever — with the soft budget the physics just backfills from
    the unbounded slack and the pack never moves. Use only where a storage
    extension can cover the transient; too tight a cap starves the pack and
    the gas LP goes infeasible. Steps:

    1. Compute the operator budget per sector from nominal load (plus
       injection) magnitudes.
    2. Set the slack Var bounds to a wide physical envelope so the LP can
       always balance: ``± _SLACK_LP_HEADROOM_FACTOR · max(budget, sector
       throughput)``. The throughput term (load + generation/source
       magnitude) keeps the LP feasible on grids where CP / generation
       flow dwarfs native load; sizing off the budget alone under-sizes
       it there. This is purely a feasibility guard; the soft budget in
       step 3 is the operator policy.
    3. Stash the budget as ``_scare_slack_budget_mw`` /
       ``_scare_slack_budget_kgs`` on the slack model; the scenario-build
       hook registers it as the slack agent's rating, then multiplies by
       ``slack_target_fraction`` for the MAS-level target.

    ``fraction`` is the share of total demand the operator wants the
    external grid to supply (e.g. 0.5 = slack target is 50 % of nominal
    demand); network generators make up the remainder and the MAS drives
    toward that distribution.
    """
    # Aggregate nominal demand per sector. Heat is excluded: heat-side
    # ExtHydrGrid imports mass-flow at fixed supply temperature, bounded
    # by WaterPipe / HeatExchanger physics rather than load magnitude.
    # Cap only power and gas external grids (no other resource limit).
    # Sinks live on both gas and water junctions, so route by parent-node
    # grid name — counting heat-side Sinks toward the gas budget would
    # inflate the cap ~4× and make the constraint inert.
    # Also track per-sector injection magnitude (PowerGenerator,
    # gas-side Source): all slack demand is equal regardless of producer,
    # so the budget mirrors physical slack throughput (load + injection).
    # On grids with gas-fed converters a load-only budget under-shoots.
    # The feasibility envelope below uses the same throughput basis.
    total_p_mw = 0.0
    total_gas_mass_kgs = 0.0
    total_p_gen_mw = 0.0
    total_gas_source_kgs = 0.0
    total_gas_conv_kgs = 0.0
    for child in mes.childs:
        m = child.model
        if isinstance(m, PowerLoad):
            total_p_mw += abs(getattr(m, "p_mw", 0.0))
        elif isinstance(m, PowerGenerator):
            total_p_gen_mw += abs(getattr(m, "p_mw", 0.0))
        elif isinstance(m, (Sink, Source)):
            try:
                grid_name = str(
                    getattr(mes.node_by_id(child.node_id).grid, "name", "")
                ).lower()
            except Exception:
                grid_name = ""
            if "gas" in grid_name:
                if isinstance(m, Sink):
                    total_gas_mass_kgs += abs(getattr(m, "mass_flow_kgs", 0.0))
                else:
                    total_gas_source_kgs += abs(getattr(m, "mass_flow_kgs", 0.0))
    # Gas-drawing cross-sector converters (CHP) are modeled at node level
    # (CHPHGControlNode), not as Sink/Source children, so the loop above
    # misses them. Their gas input is normal slack demand.
    for node in mes.nodes:
        nm = node.model
        if type(nm).__name__ == "CHPHGControlNode":
            total_gas_conv_kgs += abs(getattr(nm, "gas_mass_flow_kgs", 0.0))

    cap_p_mw = max(1e-3, fraction * (total_p_mw + total_p_gen_mw))
    cap_gas_mass_kgs = max(
        1e-4,
        fraction * (total_gas_mass_kgs + total_gas_source_kgs + total_gas_conv_kgs),
    )

    # Feasibility envelope: never below the sector's physical throughput
    # (load + injection it may absorb), so the LP always balances even
    # when the operator budget is tiny relative to CP / generation flow.
    lp_p_mw = _SLACK_LP_HEADROOM_FACTOR * max(cap_p_mw, total_p_mw + total_p_gen_mw)
    lp_gas_mass_kgs = _SLACK_LP_HEADROOM_FACTOR * max(
        cap_gas_mass_kgs,
        total_gas_mass_kgs + total_gas_source_kgs + total_gas_conv_kgs,
    )

    hard = frozenset(hard_cap_carriers or ())
    for child in mes.childs:
        m = child.model
        if (
            isinstance(m, ExtPowerGrid)
            and hasattr(m, "p_mw")
            and hasattr(m.p_mw, "min")
        ):
            # Hard-cap carriers pin the import bound at the budget so a deficit
            # must draw storage or shed; others keep the wide feasibility
            # envelope and let the MAS own the soft target.
            p_import_cap = cap_p_mw if "electricity" in hard else lp_p_mw
            m.p_mw.min = -lp_p_mw
            m.p_mw.max = p_import_cap
            # Soft target the MAS drives toward.
            m._scare_slack_budget_mw = cap_p_mw
        elif (
            isinstance(m, ExtHydrGrid)
            and hasattr(m, "mass_flow_kgs")
            and hasattr(m.mass_flow_kgs, "min")
        ):
            try:
                grid_name = str(
                    getattr(mes.node_by_id(child.node_id).grid, "name", "")
                ).lower()
            except Exception:
                grid_name = ""
            if "gas" in grid_name:
                # The gas ExtHydrGrid SUPPLIES at negative mass_flow (import is
                # the -min direction; positive is withdrawal-from-network), the
                # opposite of the power slack. Hard-cap the supply magnitude on
                # the min side; leave the positive side wide so a surplus can
                # always be dumped without infeasibility.
                gas_supply_cap = cap_gas_mass_kgs if "gas" in hard else lp_gas_mass_kgs
                m.mass_flow_kgs.min = -gas_supply_cap
                m.mass_flow_kgs.max = lp_gas_mass_kgs
                m._scare_slack_budget_kgs = cap_gas_mass_kgs
            # heat-side ExtHydrGrid left unbounded


def apply_cold_day(
    mes: "object",
    *,
    supply_t_k: float = 343.15,
    heat_load_scale: float = 1.5,
) -> None:
    """Mutate ``mes`` in place to simulate a cold-day stress scenario.

    Knobs:

    - ``supply_t_k``: heat-side ``ExtHydrGrid`` slack supply temperature
      (default ~70 °C vs unstressed ~83 °C). Lower supply shrinks the
      downstream headroom against the 60 °C / 333.15 K lower bound, so
      transport delay or pipe loss can push junctions into violation.
    - ``heat_load_scale``: multiplier on every ``HeatLoad.q_mw_heat``
      (default 1.5×); makes the heat sector harder to balance.

    Call once on a fresh net — calling twice stacks the load scale.
    """
    for child in mes.childs:
        m = child.model
        if isinstance(m, HeatLoad):
            m.q_mw_heat = float(m.q_mw_heat) * heat_load_scale
        elif isinstance(m, ExtHydrGrid):
            try:
                grid_name = str(
                    getattr(mes.node_by_id(child.node_id).grid, "name", "")
                ).lower()
            except Exception:
                grid_name = ""
            if "water" in grid_name or "heat" in grid_name:
                m.t_k = supply_t_k


def apply_pv_peak(
    mes: "object",
    *,
    gen_scale: float = 1.5,
    load_scale: float = 0.4,
    slack_vm_pu: float = 1.04,
) -> None:
    """Mutate ``mes`` in place to simulate a sunny-midday over-voltage
    stress scenario (the VDE-AR-N 4105 regime).

    Three concerted factors:

    1. High HV/MV transformer tap — slack ``ExtPowerGrid.vm_pu`` set to
       ``slack_vm_pu`` (default 1.04 pu), so the feeder sits ~1 % below
       the upper bound before any PV.
    2. PV near nameplate — every ``PowerGenerator.p_mw`` magnitude ×
       ``gen_scale`` (default 1.5×). Moderate on purpose: a 3-4× scale
       trivially imbalances and the LP curtails it away, whereas 1.5×
       stays within the slack budget so voltage depends on local Q.
    3. Daytime load trough — ``PowerLoad.p_mw`` × ``load_scale``
       (default 0.4×). Reactive load left nominal (the standard targets
       active-power imbalance).

    Call once on a fresh net — calling twice stacks the scales.
    """
    # Totals used to size the slack-budget relaxation below.
    total_gen_p = 0.0
    total_load_p = 0.0

    for child in mes.childs:
        m = child.model
        if isinstance(m, PowerGenerator):
            # p_mw < 0 (load convention); scaling preserves sign.
            m.p_mw = float(m.p_mw) * gen_scale
            total_gen_p += abs(m.p_mw)
        elif isinstance(m, PowerLoad):
            m.p_mw = float(m.p_mw) * load_scale
            total_load_p += abs(m.p_mw)

    # Widen the slack so the LP stays feasible under reverse flow. This
    # scenario is about voltage behaviour, not slack scarcity, so stress
    # should come from the tap setpoint and feeder impedance rather than
    # an artificial slack cap.
    headroom = max(total_gen_p - total_load_p, total_load_p) + 0.5
    for child in mes.childs:
        m = child.model
        if isinstance(m, ExtPowerGrid):
            # ``vm_pu`` pins the slack-bus voltage magnitude; raising it
            # shifts the whole feeder profile upward.
            try:
                m.vm_pu = float(slack_vm_pu)
            except Exception:
                pass
            # Widen the slack p_mw Var bounds to absorb reverse flow.
            try:
                if hasattr(m.p_mw, "min"):
                    m.p_mw.min = -headroom
                if hasattr(m.p_mw, "max"):
                    m.p_mw.max = +headroom
            except Exception:
                pass


def apply_line_stress(
    mes: "object",
    *,
    load_scale: float = 1.8,
    ampacity_scale: float = 0.5,
    affect_branch_fraction: float = 1.0,
) -> None:
    """Mutate ``mes`` in place to simulate a line-loading stress scenario.

    Exercises the line-loading pipeline: PowerLine agents observe
    overload, the home group leader receives a relief-MW
    ``StartBalanceNegotiation``, and the reconfigurator's path-ranking
    sees loading variation across candidate paths.

    Knobs:

    - ``load_scale`` (default 1.8×): multiplier on every
      ``PowerLoad.p_mw``; combined with reduced ampacity, guarantees an
      overload after any non-trivial branch failure.
    - ``ampacity_scale`` (default 0.5×): multiplier on every PowerLine
      ``max_i_ka``; shifts the binding constraint from voltage to
      thermal.
    - ``affect_branch_fraction`` (default 1.0): fraction of PowerLines
      to reduce, picked deterministically by sorted branch id. Sweep to
      study concentrated vs distributed reductions.

    Call once on a fresh net — calling twice stacks the scales.
    """
    for child in mes.childs:
        m = child.model
        if isinstance(m, PowerLoad):
            m.p_mw = float(m.p_mw) * load_scale

    if ampacity_scale != 1.0 and affect_branch_fraction > 0.0:
        # Reduce ampacity on the highest-``max_i_ka`` lines first, so the
        # constraint lands on the lines the reconfigurator would
        # otherwise prefer.
        powerlines = [
            b
            for b in mes.branches
            if hasattr(b.model, "max_i_ka")
            and type(b.model).__name__.lower().startswith("power")
        ]
        powerlines.sort(
            key=lambda b: (-float(getattr(b.model, "max_i_ka", 0.0)), b.id),
        )
        n_affect = max(1, int(len(powerlines) * affect_branch_fraction))
        for branch in powerlines[:n_affect]:
            try:
                branch.model.max_i_ka = float(branch.model.max_i_ka) * ampacity_scale
            except Exception:
                pass
