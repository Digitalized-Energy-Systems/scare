"""Configuration for the restoration scenario builder.

A single dataclass that the scenario builder consumes to enable / disable
architectural components and tune their parameters.  Defaults reproduce
the current (post-Gap-1..6 + options 3+4 + B/A/C) behaviour, so existing
callers that don't pass a config get the established baseline.

Used by the evaluation harness to run ablations (turn each component
off in isolation) and sensitivity sweeps (vary tunables) without
duplicating the scenario plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RestorationConfiguration:
    # ----------------------------------------------------------------
    # Architectural levels (ablation flags)
    # ----------------------------------------------------------------

    # Level-2 holonic ADMM across same-sector group leaders.  When
    # False, no HolonicCommunityRole is installed and the ``holons``
    # topology is empty; group-level rebalancing falls back to local
    # constraint-violation triggers and the local-generation fallback.
    enable_holonic: bool = True

    # Level-3 cross-sector ADMM at coupling-point agents.  When False,
    # ``EnergyConverterRole`` / ``DistributedOptimizationRole`` /
    # ``CoordinatorRole`` are not installed on CP nodes/branches and
    # the ``cps`` topology is empty.
    enable_cp_admm: bool = True

    # Replicated priority-cascaded sharing ADMM at L3 (the decentralised
    # replacement for the Option-B elected-coordinator path).  When
    # True, every CP runs the same :func:`scare.service.cp_priority_admm.
    # solve_cp_priority_admm` kernel locally on its replicated peer view
    # and commits its own regulation factor as a single self-addressed
    # ``apply_regulate`` write — no per-component coordinator, no
    # cross-CP fan-out.  Each CP sources per-sector demand/supply from
    # the extended L2 ``holon_summary_<sector>`` meshes (the CP joins
    # those meshes at scenario build time) and CP-specific slices from
    # peer ``CPSummary`` envelopes.
    #
    # Default flipped to True as of the L3 cutover: the new path
    # shadows the legacy Option-B installation through the
    # ``if priority_admm: new_path; elif cp_admm: legacy_path`` install
    # chain in ``scenario.restoration``, so ``enable_cp_admm`` stays at
    # its True default for ablation reachability without competing with
    # the new role.  Set False to fall back to the legacy elected-
    # coordinator path; both can be False to install no L3 role at all
    # (the single-level / component-level variants do this).
    enable_cp_priority_admm: bool = True

    # L3 kernel selection.  ``"lexicographic"`` (default) runs the
    # distributed lexicographic-cascade sharing ADMM from
    # ``distributed_resource_optimization`` — a Π-round cascade (one
    # round per priority tier, highest first) that *maximises* served
    # demand per tier subject to the hard per-(sector, step) constraint
    # ``σ + Σ_i r_i·c_{i,s} ≤ B_s − θ``.  Because the sector base supply
    # ``B`` folds in the slack's operator budget (the slack is counted
    # at ``|cap|`` = its budget in ``_handle_ask_flex``), this hard-caps
    # the CPs' cross-sector draw at the budget — so a CP burning gas to
    # make electricity cannot force the gas slack past its budget even
    # when the gas sector carries no demand of its own.  ``"penalty"``
    # selects the legacy ``solve_cp_priority_admm`` kernel
    # (priority-weight marginal penalty); that kernel is *formally
    # broken* for the budget case — a soft over-draw penalty either
    # limit-cycles (flat) or settles with a steady-state offset
    # (proportional), neither of which respects the hard budget — and is
    # retained only for ablation.
    cp_admm_algorithm: str = "lexicographic"

    # Proximal step-damping coefficient ``α ≥ 0`` for the lexicographic
    # cascade's per-CP closed-form projection.  Biases the *iterate
    # step* (not the saddle): at any fixed point ``r = r_prev`` so the α
    # term cancels and the asymptote is the bare sharing-ADMM optimum.
    # 0.1 is the DRO empirical sweet spot; α = 0 is still correct but
    # can oscillate on a degenerate optimal face and never trip the
    # per-iter convergence test.  Ignored when
    # ``cp_admm_algorithm == "penalty"``.
    cp_admm_r_regularization: float = 0.1

    # Distributed FailureNotice propagation through ProblemDetector.
    # When False, a centralised ``behavior_in(BranchFailureEvent)``
    # callback triggers all leaders directly (legacy behaviour, kept
    # for the ablation comparison).
    enable_distributed_failure_notice: bool = True

    # Concept C — Layer-2 dynamic holon-membership filter.  When True,
    # a ``DynamicHolonRole`` sits next to ``HolonicCommunityRole`` on
    # every holon-eligible leader and drops members that have become
    # physically unreachable through the live sector subgraph after a
    # branch failure (see scare.community.dynamic_holon).  When False
    # the holon keeps its static chunk-time membership and may try to
    # allocate flow across islanded members.  Default on so the
    # holon's allocations stay physically realisable.
    enable_dynamic_holon_topology: bool = True

    # Concept C — Layer-3 dynamic CP-connector filter.  When True, a
    # ``DynamicConnectorRole`` sits next to ``EnergyConverterRole`` on
    # every CP agent and drops group-leader peers that have become
    # physically unreachable through the cross-sector graph (incl. CP
    # bridges) after a branch failure.  Default on for the same
    # physically-realisable-allocations reason as L2.
    enable_dynamic_cp_topology: bool = True

    # Layer 2.5 — sector-wide holon-summary mesh + cross-holon priority
    # invariant detection.  Milestone 1: each leader periodically
    # publishes its per-tier served/demand summary on the
    # ``holon_summary_<sector>`` topology; every same-sector leader
    # subscribes and runs a local inversion check across received
    # summaries.  On detection, ``record_event("priority_inversion_
    # detected", ...)`` fires for post-run analysis.  Milestone 2
    # (later) will form an ad-hoc coalition that re-balances across
    # the inverted holons.  Cheap when off: no topology + no role.
    enable_holon_summary: bool = True

    # Period between HolonSummary publishes (sim-seconds).  A leader's
    # role schedules ``_publish`` every ``holon_summary_period_s`` and
    # ``_check_invariants`` at the same cadence.  Faster picks up
    # cross-holon inversions sooner but costs O(N²) extra messages
    # per sector per period.
    #
    # The L2 holon ADMM dispatches its initial allocation reactively
    # at ~ t=0.08 s (right after holon formation).  That allocation
    # is what *creates* the cross-holon inversions L2.5 then races to
    # close.  Detection needs (a) one self-publish, (b) peer publishes
    # to arrive, (c) a second tick where this leader sees ≥1 peer
    # summary and can compute the cross-holon aggregate.  At 1 s
    # period that was ≥ 2 s of inversion before the first coalition
    # ever fired — the dominant cause of the per-component priority
    # invariant lingering low even at 60 s sim runs.  0.25 s lets the
    # first coalition open by ~0.6 s sim time so the post-L2 inversion
    # gap is closed within the first second.
    holon_summary_period_s: float = 0.25

    # Per-tier served-fraction tolerance for declaring a priority
    # inversion.  A pair (tier_high, tier_low) is flagged when
    # ``frac[tier_high] < frac[tier_low] - holon_summary_inversion_tol``.
    # Mirrors the priority-invariant claim's 1e-3 tolerance so the
    # detector and the claim agree on what counts as an inversion.
    holon_summary_inversion_tol: float = 1e-3

    # Layer 2.5 milestone 2 — coalition formation.  When True, the
    # lex-smallest leader that detects a cross-holon priority
    # inversion opens an ad-hoc coalition with the affected peers,
    # runs a scoped priority-greedy allocation over their flex, and
    # broadcasts per-tier service-fraction constraints via the same
    # ``StartBalanceNegotiation(service_fraction_by_sector_priority=
    # ...)`` handler L2 uses for its supply-priority ADMM result.
    # Constraints are re-asserted every L2.5 tick until ``ttl_s`` or
    # a ``BranchFailureEvent`` invalidates them.  Off ⇒ M1 behaviour
    # (detect + record event, no scoped allocation).
    #
    # Default off as of 2026-05-23: with ``holon_admm_scope="sector"``
    # the new sector-wide L2 already produces a sector-uniform per-tier
    # service fraction across all holons in the same sector, so the
    # intra-sector coalition has nothing to fix.  Set True to opt
    # back in (e.g. when paired with ``holon_admm_scope="holon"``).
    enable_holon_coalition: bool = False

    # Window the initiator waits after broadcasting
    # ``CoalitionInvitation`` before running the allocation pass.
    # Short enough that the coalition tick cadence (``holon_summary_
    # period_s``) still dominates; long enough that even loaded peers
    # have time to reply.  Default 1 s matches the M1 tick period.
    holon_coalition_accept_window_s: float = 1.0

    # TTL on coalition constraints.  After ``issued_at + ttl_s`` the
    # constraint is dropped and the underlying L2 holon ADMM
    # allocation takes over on the next L2 rebalance.  Sized so the
    # coalition's effect outlasts a few L2.5 ticks (so re-assert keeps
    # the fraction stable) but doesn't survive past the next
    # significant grid state change.
    holon_coalition_constraint_ttl_s: float = 8.0

    # Cross-sector coalition extension (L2.5).  When True,
    # ``HolonSummaryRole`` additionally detects priority inversions
    # *across* sectors connected by a CP (e.g. tier-1 electricity
    # under-served while tier-5 heat fully served, with a P2H
    # between them) and forms a coalition spanning both sectors plus
    # the bridging CP(s).  The coalition issues a ``CPCommitment``
    # envelope to each CP member and per-sector service fractions to
    # the leader members — both written to the shared
    # ``CoalitionConstraintStore`` so L2 (per-sector) and L3 (CP
    # ADMM) honour the commitment for the TTL window.
    #
    # Off ⇒ per-sector behaviour only.  Default off as of 2026-05-23:
    # the smoke campaign showed L2.5 firing reactively after the bad
    # initial L2 dispatch has already been made (electrical balance
    # collapses ~22 % in the first 300 ms of sim time) and only
    # marginally recovers — see the 2026-05-22 eval_full_smoke phase
    # analysis.  The new sector-wide L2 (all holons as ADMM peers,
    # intra-sector) is intended to subsume L2.5's role; cross-sector
    # coupling will be re-introduced in a later milestone once the
    # in-sector path is correct.  Set True to opt back in.
    enable_cross_sector_coalitions: bool = False

    # Curtailment auction in GridConstraintMonitor on hard violations.
    # When False, violations only emit a BalanceProblem to re-trigger
    # gossip; no proportional curtailment is broadcast.
    enable_curtailment_auction: bool = True

    # Constraint-aware participation scaling inside the gossip step.
    # When False, ``participation_scale = 1`` always.
    enable_constraint_aware_gossip: bool = True

    # Multi-hop ConstraintStateMessage forwarding from
    # GridConstraintMonitor.  When False, only direct neighbours see
    # the local utilization.
    enable_multihop_constraint: bool = True

    # Priority-weighted waterfall S parameter in the holon ADMM.
    # When False, S=0 ⇒ ADMM redistributes proportionally to balance
    # only, ignoring critical-tier urgency.
    enable_priority_holon_allocation: bool = True

    # No-regret floor in EnergyBalanceNegotiator._apply_setpoint
    # during restoration directions.  When True, a load that has
    # once reached factor=X cannot drop below X in a subsequent
    # gossip round.
    #
    # Default flipped to False as of 2026-05-23: the floor blocks
    # the L3→L2→L1 cascade's re-shed semantics.  When L3 commits a
    # P2H increase to serve a high-priority heat load, the
    # source-sector L2 must shed lower-priority elec loads to free
    # the supply; the floor preserved the old (now stale) factor
    # and prevented the shed, leading to LP over-commitment or
    # priority inversions.  Set True to opt back into the no-regret
    # behaviour for ablations against the pre-Option-B path.
    enable_monotonic_floor: bool = False

    # L2 priority-floor: clamp L1 reactive sheds (gossip ``balance`` +
    # ``stability`` re-apply) up to ``min(L2 allocation,
    # constraint-allowed fraction)`` for tiers 2/3/4.  Stops a
    # supply-poor *local* gossip group from shedding a load the
    # *component*-scope holon ADMM decided to serve — the L2→L1 override
    # that produced the tier-3 < tier-4 / tier-2 < tier-3 inversions
    # (eval task-88, task-51).  The constraint-allowed cap (same util as
    # ``clamp_to_constraints``) lets curtailment/physics still shed the
    # load during a real violation, per-load and continuously, so the
    # floor and the constraint clamp never fight (the coarse-flag version
    # regressed the cold-day task-72).  Applies to all load tiers incl.
    # tier 1 (whose constraint-allowed is always 1.0, so the floor just
    # re-asserts its hard-lock against ``stability`` erosion); the
    # curtailment auction can still shed any tier when physics demands.
    # Set False to ablate against the pre-floor override behaviour.
    enable_l2_priority_floor: bool = True

    # Cold-load pickup ramp limit on regulation increases.
    # When False, factor jumps are not throttled.
    enable_clpu_ramp: bool = True

    # Heat-only periodic un-shed recovery in GridConstraintMonitor.
    # When True, every ~5 s the monitor checks local heat constraints
    # and if they're clear it bumps each load's regulation factor up
    # by ~0.2 per cycle (independent of L2/L3 priority decisions).
    #
    # Default flipped to False as of 2026-05-23: same class of bug as
    # ``enable_monotonic_floor``.  When L2 sheds heat tier 5 loads to
    # factor=0 (priority decision: serve tier <5 first), heat_recovery
    # then un-sheds them back to ~0.225 because heat constraints are
    # locally clear — overriding the priority cascade.  The 2026-05-23
    # smoke showed this driving uniform-0.225 inversions on heat
    # tier 5 vs tier 6 across multiple scenarios (cooldown_sweep,
    # cold_day_stress, ablation_thermal, etc).
    #
    # Set True to opt back into the no-regret heat un-shed behaviour
    # for ablations against the pre-Option-B path.
    enable_heat_recovery: bool = False

    # Local Q-V droop at every inverter-coupled PowerGenerator (PV).
    # Follows the VDE-AR-N 4105 Q(U) characteristic: piecewise-linear
    # with a 0.97–1.03 pu deadband, saturating at ±Q_max at 0.95 / 1.05.
    # Q_max is bounded by the apparent-power capability circle of the
    # inverter (S_n = |p_n| / cos φ_min, with cos φ_min = 0.95 for
    # S_n ≤ 13.8 kVA, else 0.90).  When False, no droop role is
    # installed and reactive dispatch stays at simbench defaults — used
    # for the ablation comparison showing the contribution of local
    # voltage support to overall restoration quality.
    enable_qv_droop: bool = True

    # Voltage reference for the Q(U) curve (per unit).  Per VDE-AR-N
    # 4105 the curve is anchored at 1.0 pu; exposed so the sensitivity
    # sweep can probe the effect of a re-centred droop.
    qv_droop_voltage_ref_pu: float = 1.0

    # F2: slack-infeed target as a fraction of the registered slack
    # rating.  Per-community gossip uses this to bias its imbalance
    # accounting away from "balance ⇒ zero slack draw" toward
    # "balance ⇒ slack draws its target fraction of rating".  In
    # practice this only matches the operator's intent when the
    # slack's community spans the full LP balance scope (i.e.
    # ``community_partition_method="connected_component"``); for
    # the holonic and label-propagation partitions the gossip target
    # derived from this setpoint contradicts the global budget.  Left
    # at 0.0 by default; budget enforcement instead routes through
    # :class:`~scare.service.slack_budget.SlackBudgetMonitor`'s
    # signed ``override_target`` path, which is partition-agnostic.
    slack_target_fraction: float = 0.0

    # P6 primal-dual QP gossip.  When True (default), the receiving
    # agent computes its δ_i in closed form from the gossiped dual
    # variable λ as ``δ_i = clamp(w_i · λ, dmin_i, dmax_i)``, with
    # priority encoded continuously in w_i and the dual updated by
    # gradient ascent on the primal residual.  When False, the legacy
    # equal-share / priority-gated update is used (intra-tier
    # serialisation via deterministic sub-rounds).  Both routes share
    # the P1 saturation flag, P2 stall detection, P3 step decay, and
    # all the other infrastructure; only the per-agent update rule
    # differs.  Exposed as an ablation flag so the harness can run
    # head-to-head comparisons between QP and equal-share gossip.
    enable_qp_gossip: bool = True

    # Branch-side line-loading monitor.  When True every PowerLine
    # branch (switchable or not) gets a GridConstraintMonitor watching
    # the line's loading_percent.  On overload the monitor sends
    # StartBalanceNegotiation with a relief-MW override target to the
    # branch's home group leader (picked at scenario setup time as the
    # endpoint with the lower priority-weighted demand, so shedding
    # falls on the less-critical side) and propagates a
    # ConstraintStateMessage to both endpoint groups so neighbouring
    # gossip agents throttle their participation.  When False the
    # branch agents are not registered and line overload is silent.
    enable_line_loading_constraint: bool = True

    # External-grid slack budget monitor.  When True, every slack-class
    # child (ExtPowerGrid / ExtHydrGrid with ``_scare_slack_budget_*``
    # stamped by ``apply_slack_budget``) carries a ``SlackBudgetMonitor``
    # role that polls the LP-chosen ``p_mw`` / ``mass_flow`` and, when
    # the absolute draw exceeds ``budget · (1 + slack_budget_violation_tol)``,
    # records a ``slack_budget_violation`` event and emits a
    # ``BalanceProblem`` so the co-located ``EnergyBalanceNegotiator``
    # triggers a rebalance round (and the optional curtailment auction
    # / multihop propagation chained off it).  Goal: make the operator-
    # policy ``slack_budget_pct`` a runtime-enforced constraint rather
    # than a passive label, while leaving the LP envelope (10× budget)
    # wide enough that the energy-flow solve stays feasible.
    enable_slack_budget_monitor: bool = True

    # Relative tolerance for the slack-budget monitor.  A draw is flagged
    # only when ``|obs| > budget · (1 + slack_budget_violation_tol)``;
    # the small margin avoids flagging the steady-state numerical wiggle
    # of an LP that's already converged inside the envelope.
    slack_budget_violation_tol: float = 0.05

    # Loss-compensating effective-budget feedback in the slack monitor.
    # The operator budget caps the slack's *actual* draw, but the
    # L1/L2/L3 control only shapes the served setpoints — network losses
    # (plus the per-leader-group vs global supply-pool mismatch) leave
    # the realized draw above budget even when the controller believes
    # it hit the target.  When True, ``SlackBudgetMonitor`` runs an
    # integral correction on the observed overage and advertises a
    # tightened *effective* budget (``B - losses``) into the supply pool
    # so the actual draw converges to the operator budget.  When False,
    # the pool advertises the nominal budget (legacy behaviour).
    enable_slack_budget_feedback: bool = True

    # GridReconfigurator path ranking by line loading.  When True the
    # reconfigurator carries a running max_loading_percent along each
    # GridPathMessage, buffers all results within a short window, and
    # picks the path with the lowest peak loading instead of the
    # first-arrived (typically shortest).  When False the legacy
    # first-arrival behaviour is preserved.
    enable_reconfig_feasibility_ranking: bool = True

    # Window during which the reconfigurator collects candidate paths
    # before picking the best.  Sized for the electricity poll period;
    # too short loses alternatives, too long delays restoration.
    reconfig_path_window_s: float = 1.5

    # Holon ADMM iteration cap.  50 was chosen historically so concurrent
    # holon ADMMs across sectors don't block discrete-time progress, but
    # smoke runs show non-convergence at this cap on simbench_lv (residuals
    # 1e-3 to 2e-2).  Exposed so the campaign can trade convergence quality
    # against wallclock cost.
    holon_admm_max_iters: int = 50

    # ADMM absolute residual tolerance — convergence quality the holon
    # coordinator stops at when |r| < this.  The package default 1e-4 is
    # tight; relaxing to 1e-3 lets typical scenarios converge inside the
    # 50-iter cap without changing the qualitative behaviour.
    holon_admm_abs_tol: float = 1e-3

    # Tier-stratified holon ADMM (Package C).  When True, the holon's
    # ADMM target vector is built per-(sector, priority_tier) instead
    # of per-sector only, and the L1 honour path dispatches per-tier
    # targets directly to member agents — preserving the holon's
    # global priority decision through the L2 → L1 handoff.  Legacy
    # path (False) uses one scalar target per (member, sector) and
    # L1 re-derives priority locally, which can invert priority on
    # finely-partitioned grids (see priority_invariant claim in
    # eval/claims.py).
    enable_tier_stratified_holon_admm: bool = True

    # Number of priority tiers the tier-stratified ADMM allocates over.
    # Mirrors ``base.util.tier_priority_weight``'s 4-tier schedule
    # (tier 1 = critical / hard-locked at the L1 leader pre-step;
    # tiers 2-4 = QP-weighted with 1e8 / 1e4 / 1.0 exponents).  Larger
    # would re-introduce the over-soft proportional split this redesign
    # was meant to fix; smaller would collapse the QP into pure tier-1
    # gating with no QP-side ordering.
    priority_tiers: int = 4

    # Holon ADMM mode (Package C variants).  Only consulted when
    # ``enable_tier_stratified_holon_admm`` is True.
    #
    # - ``"supply"`` (default, Route A): supply-side formulation.
    #   ``T`` = total demand at each (sec, tier), each actor's
    #   contribution represents supply commitment.  Coupling
    #   ``Σ x_g ≤ supply_g`` binds whenever holon-wide supply <
    #   holon-wide demand.  Priority weights then decide which
    #   tiers get the scarce supply.  Enables cross-community
    #   generation routing (e.g. shed A's tier-8 load to free
    #   supply for B's tier-2 load).  L1 dispatch interprets the
    #   result as per-tier service fractions and applies them
    #   uniformly to local loads at that tier.  This is the
    #   formulation that actually arbitrates priority across
    #   communities in end-to-end runs; ``demand`` is preserved as
    #   an ablation but does not exercise priority weighting in
    #   practice (see eval/claims.py:priority_invariant).
    # - ``"demand"`` (legacy, ablation only): Package C demand-side
    #   formulation.  ``T`` = per-cell deficit, each actor absorbs
    #   its share of the per-cell deficit (`ub = local_deficit`).
    #   Priority weights arbitrate only when the per-actor coupling
    #   (``Σ x_g ≤ flex_g``) binds, which is rare in pure-load
    #   groups where flex == deficit.  Solves the "where to spend
    #   limited flex" problem.
    holon_admm_mode: str = "supply"

    # Holon ADMM *scope* — which actors participate in a single ADMM
    # round.  Decoupled from ``holon_admm_mode`` (which picks the LP
    # formulation regardless of scope).
    #
    # - ``"component"`` (default as of 2026-05-23): every *group
    #   leader* in the same active connected component of the sector
    #   is an ADMM actor.  The elected coordinator (lex-smallest
    #   leader aid among leaders mutually reachable on the active
    #   branch subgraph) collects each leader's community flex
    #   (supply + per-tier demand), runs the ADMM, and dispatches the
    #   resulting per-tier ``service_fraction`` to every leader in
    #   the component.  Each leader then applies the fractions to
    #   its OWN community members directly (no holon hop).
    #
    #   Why per-component: priority is a global ordering whose
    #   guarantee scope is exactly the connected component of the
    #   active grid (the priority-invariant claim aggregates per
    #   ``(sector, component)``).  Aligning the optimisation scope
    #   with the claim scope means cross-leader inversions cannot
    #   arise *and* a failure that splits a component re-elects two
    #   coordinators that decide independently for their halves.
    #
    #   Why every group leader (not every holon): the previous
    #   sector-scope path dispatched only via holon leaders, leaving
    #   communities not in any holon at the LP-default factor=1.0 —
    #   the actual root cause of the inversions surfaced by the
    #   2026-05-22 smoke.  This default puts every community leader
    #   in the dispatch loop.
    #
    # - ``"sector"`` (deprecated, 2026-05-22 default): all holon
    #   leaders in the sector were ADMM actors.  Suffered the
    #   coverage gap above — kept reachable as an ablation for the
    #   campaign comparison; will be removed once the per-component
    #   path is validated.
    # - ``"holon"`` (legacy): each holon leader runs its own ADMM
    #   over its member groups only.  Produces per-holon per-tier
    #   service fractions that need not agree across holons — i.e.
    #   the cross-holon inversions the original smoke surfaced.
    #   Retained as an ablation knob.
    holon_admm_scope: str = "component"

    # Hebbian-emergent holon membership refinement (Aoki & Aoyagi 2009).
    # Leaders broadcast their normalised sector imbalance δ_g as
    # HebbianFlexBeacon, accumulate a per-peer co-variance estimate
    # H_{gh} = (1-η)·H + η·δ_g·δ_h, and after ``hebbian_warmup_s``
    # rebuild holon membership from peers with H_{gh} > threshold.  This
    # replaces / refines the static lex-chunked partition from
    # _build_topologies, so groups whose stress dynamics correlate end
    # up cooperating regardless of aid ordering.  Disable for the
    # static-partition ablation.
    enable_hebbian_formation: bool = True

    # Sim-seconds before the recluster begins (during which the co-
    # variance estimate accumulates).  Default 12 s matches the
    # original holonic.py constructor default; campaigns with shorter
    # simulations should drop this (e.g. 4 s) so reclustering has time
    # to fire within the run window.
    hebbian_warmup_s: float = 12.0

    # Co-variance threshold above which a peer is admitted to the
    # dynamically-emergent holon.  Higher = stricter (fewer peers
    # admitted, smaller holons).
    hebbian_threshold: float = 0.35

    # Level-1 (sub-community) partition method.
    #
    # - ``"label_propagation"`` (default, preserves legacy behaviour):
    #   radius-bounded min-label propagation — communities are
    #   ≤``community_label_propagation_radius``-hop balls centred on
    #   the lex-smallest reachable seed.
    # - ``"modularity"``: distributed-Louvain Phase 1 — communities
    #   form to maximise local modularity gain, respecting the graph's
    #   natural cluster structure.  Sizes vary; not bounded by radius.
    # - ``"connected_component"``: one community per connected
    #   component of the per-sector subgraph.  Used by the
    #   ``component_level`` baseline — gives the gossip negotiator a
    #   global per-component view rather than many small radius-bounded
    #   sub-communities.  Combine with ``cps_join_communities=True`` so
    #   the CPs that bridge two components actually participate.
    community_partition_method: str = "label_propagation"

    # Radius bound for ``label_propagation`` method (ignored by
    # modularity).  Mirrors the per-sector ``_LABEL_PROPAGATION_RADIUS``
    # default of 2 historically wired in scenario/restoration.py.
    community_label_propagation_radius: int = 2

    # Resolution γ for ``modularity`` method (ignored by label
    # propagation).  γ > 1 ⇒ finer partition (more, smaller
    # communities); γ < 1 ⇒ coarser.  Default 1.0 = standard
    # modularity.
    community_modularity_resolution: float = 1.0

    # Iteration cap for the modularity phase-1 sweep.  Convergence
    # typically in 3-5 rounds; 10 is a safe cap.
    community_modularity_iterations: int = 10

    # ----------------------------------------------------------------
    # ``component_level`` baseline tunables
    # ----------------------------------------------------------------
    #
    # When True, every CP agent joins the per-sector community of each
    # endpoint it bridges (a P2G bridges one electricity + one gas
    # community).  CPs become normal members of those communities' L1
    # gossip rounds and drop their separate CP-ADMM path —
    # ``EnergyConverterRole`` is replaced by
    # ``MultiCommunityCPRole``, which collects per-community signals,
    # combines them with an EMA over per-sector setpoint targets, and
    # commits via ``apply_regulate`` under a deadband + cooldown guard
    # so a CP sitting in two communities can't ping-pong between
    # contradictory asks.  Only meaningful with
    # ``enable_cp_admm=False``; the variant builder in
    # ``experiment/hpc/runner.py`` enforces the combination.
    cps_join_communities: bool = False

    # EMA blending factor for the multi-community CP guard.  Each tick
    # the new per-sector target is ``α · proposed + (1 − α) · current``;
    # higher means more reactive to incoming proposals, lower means
    # heavier filtering.  0.3 mirrors the smoothing band used by the
    # existing slack-budget cooldown plumbing.
    cp_oscillation_ema_alpha: float = 0.3

    # Minimum |target − current| (in the sector's natural units — MW
    # for electricity, MW for heat, kg/s for gas) below which a new
    # target is treated as noise and not committed.  Sits above the
    # 0.01 MW fixed-point tolerance the legacy ``EnergyConverterRole``
    # already uses on incoming setpoints.
    cp_oscillation_deadband_mw: float = 0.05

    # Minimum simulation-second gap between two regulation commits on
    # the same CP.  Modelled on the 2.0 s ``SlackBudgetMonitor`` refire
    # cooldown that successfully damps oscillation in the gossip layer.
    cp_oscillation_min_interval_s: float = 1.0

    # ----------------------------------------------------------------
    # Sensitivity-sweep tunables
    # ----------------------------------------------------------------

    # monee-side solver throttle.  0 means "solve whenever monee
    # decides".  Non-zero buffers regulate writes per aid and flushes
    # at fixed simulation-time boundaries (see comms wrapper, step 10).
    cooldown_s: float = 0.0

    # FailureNotice TTL stamped at endpoint detectors.
    ttl_hops: int = 3

    # Hop cost across CP-bridge edges in the FailureNotice
    # propagation.  Same-sector edges always cost 1.
    cp_bridge_cost: int = 2

    # Maximum members per holon (excluding initiator that's the
    # chunk's lex-smallest leader).
    holon_max_size: int = 4

    # Gossip protocol parameters.
    gossip_termination_tolerance: float = 1e-5
    gossip_max_hops: int = 100

    # ----------------------------------------------------------------
    # Robustness / scenario perturbations
    # ----------------------------------------------------------------

    # Per-message drop probability in [0, 1].  Implemented via a
    # custom mango DelayProvider; 0 means lossless.
    comms_packet_loss_pct: float = 0.0

    # Latency jitter standard deviation in milliseconds.  Implemented
    # via a Poisson-mixed delay provider; 0 means deterministic.
    comms_latency_jitter_ms: float = 0.0

    # Agent dropout: at ``agent_dropout_at_s`` simulation seconds, the
    # listed aids are unregistered from the world.  Empty tuple means
    # no dropout.
    agent_dropout_aids: tuple[str, ...] = field(default_factory=tuple)
    agent_dropout_at_s: float = float("inf")

    # ----------------------------------------------------------------
    # Logging detail
    # ----------------------------------------------------------------

    # When True, every send_message goes through a counting wrapper
    # so per-message-type counts can be reported in result.json.
    # Off by default — high volume; turned on for the comm-cost
    # campaign only.
    record_messages: bool = False

    # ----------------------------------------------------------------
    # Islanding / microgrid configuration (opt-in)
    # ----------------------------------------------------------------
    #
    # When set, monee's ``enable_islanding`` is applied at grid-build
    # time so that any ``GridFormingMixin`` child can lead an island
    # for its carrier when the main grid-former is unreachable.  The
    # default is ``None`` — only ``ExtPowerGrid`` / ``ExtHydrGrid``
    # form a grid, which mirrors realistic LV networks where black-
    # start hardware is rare.  The microgrid scenarios opt in by
    # passing the per-carrier dict here.
    #
    # Schema is a per-carrier mapping ``{"electricity": True,
    # "water": True, "gas": True}`` (None or missing carrier means
    # "leave that carrier unchanged").  Values can also be a custom
    # ``IslandingMode`` instance for fine-tuning.
    #
    # ``frozenset`` of carriers is used at the dataclass-level so the
    # frozen-dataclass invariant is preserved; a richer mapping can
    # be plumbed via a separate hashable wrapper if/when scenarios
    # need it.
    microgrid_islanding_carriers: frozenset[str] = field(default_factory=frozenset)

    # When True, the grid factory tries to convert eligible
    # PowerGenerator / HeatGenerator / Source children into the
    # corresponding ``GridForming*`` types so that monee's islanding
    # extension actually has grid-formers to anchor sub-islands on.
    # Off by default; setting True alongside ``microgrid_islanding_
    # carriers`` is the "what-if every unit could black-start" upper-
    # bound scenario.  Use ``microgrid_grid_former_aids`` instead to
    # mark specific units only.
    microgrid_promote_all_generators: bool = False

    # Aids (or node-ids) of children to promote to grid-formers for
    # their sector.  Empty means "no specific units"; if combined
    # with ``microgrid_promote_all_generators=True``, that flag wins.
    # When neither is set but ``microgrid_islanding_carriers`` is non-
    # empty, the LP can still benefit (e.g.\ adjacent sub-islands
    # connected through the surviving Ext*Grid) but distant sub-
    # islands stay ignored exactly as before.
    microgrid_grid_former_aids: tuple[str, ...] = field(default_factory=tuple)


def default_config() -> RestorationConfiguration:
    """Return a fresh default config.  Equivalent to
    ``RestorationConfiguration()``; the named function makes intent
    explicit at call sites that want to be obviously baseline.
    """
    return RestorationConfiguration()
