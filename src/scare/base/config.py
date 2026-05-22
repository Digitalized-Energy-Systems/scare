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
    # constraint-violation triggers and the islanding fallback.
    enable_holonic: bool = True

    # Level-3 cross-sector ADMM at coupling-point agents.  When False,
    # ``EnergyConverterRole`` / ``DistributedOptimizationRole`` /
    # ``CoordinatorRole`` are not installed on CP nodes/branches and
    # the ``cps`` topology is empty.
    enable_cp_admm: bool = True

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
    # per sector per period.  Default 1 s — short enough that even
    # the 5 s smoke sims fire the publisher 3–4 times before sim
    # end, long enough that per-period communication cost stays low.
    holon_summary_period_s: float = 1.0

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
    enable_holon_coalition: bool = True

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
    # Off ⇒ legacy behaviour (per-sector coalitions only).  Provided
    # as an ablation knob so evaluation campaigns can quantify the
    # cross-sector contribution in isolation from the rest of the
    # stack.
    enable_cross_sector_coalitions: bool = True

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
    # during restoration directions.  When False, loads can be
    # un-restored across negotiation rounds without a violation.
    enable_monotonic_floor: bool = True

    # Cold-load pickup ramp limit on regulation increases.
    # When False, factor jumps are not throttled.
    enable_clpu_ramp: bool = True

    # Heat-only periodic un-shed recovery in GridConstraintMonitor.
    # When False, heat regulations stay where the gossip put them.
    enable_heat_recovery: bool = True

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
    # rating (which itself is ``_bound_external_slack``'s cap when the
    # grid uses a constrained slack budget).  Each slack agent then
    # reports ``setpoint = slack_target_fraction · rating`` as its
    # contribution to the gossip's imbalance computation — driving the
    # MAS to shed / restore until the residual matches what the slack
    # is *expected* to provide, instead of treating "slack absorbs
    # everything" as the equilibrium.  Default 0.0: slack provides
    # nothing in the imbalance accounting and the MAS does all the
    # balancing locally.  1.0: slack should provide up to its full
    # rated infeed; MAS handles anything beyond that.
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
    # Mirrors ``base.util.obs_priority``'s tier range (1..10 by
    # default).  Larger means more ADMM dimensions per actor — solver
    # cost grows linearly.
    priority_tiers: int = 10

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
