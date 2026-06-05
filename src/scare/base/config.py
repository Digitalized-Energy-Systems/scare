"""Configuration for the restoration scenario builder.

A single dataclass the scenario builder consumes to enable/disable
architectural components and tune their parameters.  Defaults give the
baseline behaviour, so callers that pass no config get the baseline.
Used by the evaluation harness for ablations (turn one component off)
and sensitivity sweeps (vary tunables).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RestorationConfiguration:
    # ----------------------------------------------------------------
    # Architectural levels (ablation flags)
    # ----------------------------------------------------------------

    # Level-2 holonic ADMM across same-sector group leaders.  False:
    # no HolonicCommunityRole, empty ``holons`` topology; group-level
    # rebalancing falls back to local constraint-violation triggers and
    # the local-generation fallback.
    enable_holonic: bool = True

    # Re-establish the heat sector's link to Layer-3 CP coordination.
    # Heat MW-balance (L2/L3 supply-priority dispatch + L1 gossip) is
    # off for the temperature-limited heat sector, which severs heat's
    # L3 link.  When True, cold-load leaders signal their *delivered*
    # heat (not the unbounded heat-slack budget) as the L3 base supply,
    # so unmet demand (nominal − delivered) drives reachable heat CPs
    # (CHP/P2H) to ramp output.  Without it, L3 sees the slack's
    # effectively-infinite pool, judges heat fully supplied, and leaves
    # every heat CP idle while junctions sit below the temperature
    # floor.  Heat-scoped; el/gas keep the slack-budget base supply.
    enable_heat_cp_supply: bool = True
    # Refresh cadence (s) for a heat leader's HolonSummary so the
    # post-failure delivered-heat vector reaches L3 within the episode.
    # Needed because heat's normal summary triggers (L1/L2) are off and
    # the 30 s liveness watchdog never fires inside a short run.
    heat_cp_supply_refresh_s: float = 2.0

    # Level-3 cross-sector ADMM at coupling-point agents.  False:
    # ``EnergyConverterRole`` / ``DistributedOptimizationRole`` /
    # ``CoordinatorRole`` are not installed on CP nodes/branches and the
    # ``cps`` topology is empty.
    enable_cp_admm: bool = True

    # Replicated priority-cascaded sharing ADMM at L3 (decentralised
    # replacement for the elected-coordinator path).  True: every CP
    # runs the same L3 sharing-ADMM kernel (selected by
    # ``cp_admm_algorithm``) locally on its replicated peer view and
    # commits its own regulation factor as one self-addressed
    # ``apply_regulate`` write — no coordinator, no cross-CP fan-out.
    # Each CP sources per-sector demand/supply from the L2
    # ``holon_summary_<sector>`` meshes (joined at build time) and
    # CP-specific slices from peer ``CPSummary`` envelopes.  This path
    # shadows the legacy install via the
    # ``if priority_admm: new; elif cp_admm: legacy`` chain in
    # ``scenario.restoration``.  False falls back to the legacy
    # coordinator path; both False installs no L3 role.
    enable_cp_priority_admm: bool = True

    # L3 kernel selection.  ``"gossip"`` (default): coordinator-free
    # peer-to-peer sharing ADMM from
    # ``distributed_resource_optimization.algorithm.gossip_lexicographic_cascade``
    # — each CP runs only its own scalar x-update and rebuilds the
    # shared aggregate from peer Iter broadcasts.  Crash-fault tolerant
    # (peer death held to next round; stale rounds discarded by
    # ``round_id``; round-timeout commits a feasible-suboptimal iterate
    # rather than blocking).  ``"lexicographic"``: replicated kernel —
    # every CP solves the full N-CP problem locally and keeps only its
    # own row (retained for ablation).
    cp_admm_algorithm: str = "gossip"

    # Proximal step-damping coefficient ``α ≥ 0`` for the lexicographic
    # cascade's per-CP closed-form projection.  Biases the iterate step,
    # not the saddle: at any fixed point ``r = r_prev`` so the α term
    # cancels and the asymptote is the bare sharing-ADMM optimum.
    # α = 0 is correct but can oscillate on a degenerate optimal face.
    cp_admm_r_regularization: float = 0.1

    # Distributed FailureNotice propagation through ProblemDetector.
    # False: a centralised ``behavior_in(BranchFailureEvent)`` callback
    # triggers all leaders directly (legacy, kept for ablation).
    enable_distributed_failure_notice: bool = True

    # Layer-2 dynamic holon-membership filter.  True: a
    # ``DynamicHolonRole`` beside ``HolonicCommunityRole`` on every
    # holon-eligible leader drops members that became physically
    # unreachable through the live sector subgraph after a branch
    # failure (see scare.community.dynamic_holon).  False: holon keeps
    # static membership and may allocate flow across islanded members.
    enable_dynamic_holon_topology: bool = True

    # Layer-3 dynamic CP-connector filter.  True: a
    # ``DynamicConnectorRole`` beside ``EnergyConverterRole`` on every
    # CP drops group-leader peers that became physically unreachable
    # through the cross-sector graph (incl. CP bridges) after a failure.
    enable_dynamic_cp_topology: bool = True

    # Layer 2.5 — sector-wide holon-summary mesh + cross-holon priority
    # inversion detection.  Each leader periodically publishes its
    # per-tier served/demand summary on ``holon_summary_<sector>``;
    # every same-sector leader subscribes and checks for inversions
    # across received summaries, firing
    # ``record_event("priority_inversion_detected", ...)`` on detection.
    # Cheap when off: no topology + no role.
    enable_holon_summary: bool = True

    # Period between HolonSummary publishes (sim-seconds); also the
    # ``_check_invariants`` cadence.  Faster picks up cross-holon
    # inversions sooner but costs O(N²) extra messages per sector per
    # period.  Detection needs a self-publish, peer publishes to arrive,
    # then a tick where this leader sees ≥1 peer summary — so the period
    # bounds how fast a post-L2-dispatch inversion can be closed.
    holon_summary_period_s: float = 0.25

    # Per-tier served-fraction tolerance for declaring a priority
    # inversion.  A pair (tier_high, tier_low) is flagged when
    # ``frac[tier_high] < frac[tier_low] - holon_summary_inversion_tol``.
    # Mirrors the priority-invariant claim's 1e-3 tolerance.
    holon_summary_inversion_tol: float = 1e-3

    # Layer 2.5 coalition formation.  True: the lex-smallest leader that
    # detects a cross-holon inversion opens an ad-hoc coalition with the
    # affected peers, runs a scoped priority-greedy allocation over their
    # flex, and broadcasts per-tier service-fraction constraints via the
    # ``StartBalanceNegotiation(service_fraction_by_sector_priority=...)``
    # handler L2 uses.  Constraints re-asserted every L2.5 tick until
    # ``ttl_s`` or a ``BranchFailureEvent`` invalidates them.  False:
    # detect + record event only, no scoped allocation.
    enable_holon_coalition: bool = False

    # Window the initiator waits after broadcasting
    # ``CoalitionInvitation`` before running the allocation pass.  Short
    # enough that the tick cadence (``holon_summary_period_s``) still
    # dominates; long enough that loaded peers can reply.
    holon_coalition_accept_window_s: float = 1.0

    # TTL on coalition constraints.  After ``issued_at + ttl_s`` the
    # constraint is dropped and the L2 holon ADMM allocation takes over
    # on the next rebalance.  Sized to outlast a few L2.5 ticks but not
    # survive the next significant grid state change.
    holon_coalition_constraint_ttl_s: float = 8.0

    # Cross-sector coalition extension (L2.5).  True: ``HolonSummaryRole``
    # also detects inversions *across* sectors connected by a CP (e.g.
    # tier-1 electricity under-served while tier-5 heat fully served via
    # a P2H) and forms a coalition spanning both sectors plus the
    # bridging CP(s).  It issues a ``CPCommitment`` to each CP member and
    # per-sector service fractions to leader members, both written to the
    # shared ``CoalitionConstraintStore`` so L2 and L3 honour the
    # commitment for the TTL window.  False: per-sector behaviour only.
    enable_cross_sector_coalitions: bool = False

    # Curtailment auction in GridConstraintMonitor on hard violations.
    # False: violations only emit a BalanceProblem to re-trigger gossip;
    # no proportional curtailment is broadcast.
    enable_curtailment_auction: bool = True

    # Gate the curtailment auction so it stops firing where it cannot
    # help.  No-ops when ``enable_curtailment_auction`` is False.  True
    # adds two guards:
    #   (a) SCOPE — the auction never fires on heat ``t_k`` or line
    #       ``loading_percent`` violations.  Its component-wide
    #       priority×own-sensitivity×reducible bidding is blind to WHICH
    #       node/branch is violated, so for these it sheds the most
    #       "willing" load, not one that relieves the violation (no
    #       load's curtailment moves another junction's return
    #       temperature — the frontier controller's lever; a line
    #       overload's correct lever is the endpoint-targeted relief
    #       path).  Still fires on node-local ``vm_pu`` / ``pressure_pu``
    #       where the violating node's own load is the lever.
    #   (b) PROGRESS GATE — for any other variable, if the overshoot has
    #       not improved beyond its best-seen value for
    #       ``_CURTAIL_NO_PROGRESS_LIMIT`` consecutive rounds, stop
    #       re-arming that variable until its overshoot improves (a
    #       worsening / topology change re-engages it).
    enable_curtail_auction_gating: bool = False

    # Cross-sensitivity targeting for the curtailment auction.  The
    # auction allocates a bidder's curtail share by
    # ``priority × OWN-local sensitivity × reducible`` — the own-local
    # term measures how curtailing the bidder moves the bidder's OWN
    # variable, not the violated node/branch, so the shed spreads
    # roughly uniformly within a tier.  True: ``CurtailmentNeed`` carries
    # the violation's origin and each bidder additionally weights its
    # willingness by electrical proximity to that origin (from the cached
    # multi-hop ``ConstraintStateMessage`` hop-distance) — a bounded
    # within-tier multiplier, so priority stays lexicographically
    # dominant.  Effect: shed lands on loads nearest the violation
    # (highest ∂constraint/∂Q).  Most meaningful for MW/flow constraints
    # (``loading_percent``, ``vm_pu``, gas ``pressure_pu``); intended to
    # pair with ``enable_curtail_auction_gating``.
    enable_curtail_auction_targeting: bool = False

    # Iterative line-overload relief.  On a branch ``loading_percent``
    # violation the branch monitor asks its ``home_leader`` (the line's
    # lower-priority-demand endpoint group) to shed ``relief_mw`` via a
    # gossip round.  Legacy (False): sent ONCE per violation episode, so
    # the home leader sheds a single step and the line plateaus above
    # 100 %.  True: relief is RE-ASSERTED every poll the line is still
    # overloaded (cooldown-guarded), with magnitude recomputed from the
    # live overshoot, so it shrinks to zero as the line reaches its bound
    # and drives the line toward feasibility round-by-round.  Electricity
    # line-relief only.
    enable_line_relief_reassert: bool = False

    # Branch-downstream targeted line relief.  The only loads whose
    # curtailment reduces a radial branch's flow ~1:1 are those
    # DOWNSTREAM of it (the subtree it feeds); the component-wide auction
    # and the endpoint-relief path both shed loads that need not flow
    # through the line.  True: each electricity branch monitor gets the
    # set of loads that become slack-disconnected when the branch is
    # removed (computed once at build), and on a ``loading_percent``
    # violation runs the curtailment auction against THAT set instead of
    # the whole component — shed lands on loads that actually relieve the
    # line, lowest-priority-first.  Replaces the component-wide auction
    # AND the endpoint-relief path for line overload; branches whose
    # removal doesn't cleanly disconnect a side (meshed / not a bridge)
    # get an empty set and fall back to legacy relief.
    enable_branch_downstream_relief: bool = True

    # Strict reverse-priority WATERFALL for the branch-downstream relief
    # auction.  Only consulted when ``enable_branch_downstream_relief``
    # is True.  The willingness-proportional default
    # (``priority_weight × sensitivity × reducible``, 1e8/1e4/1 tier
    # exponents) pours ~all shed onto the lowest-priority downstream
    # loads — often too small a fraction of the through-flow to clear a
    # 10-20 % overload — and never escalates to the tier-3 bulk carrying
    # the line, so it plateaus above 100 %.  True: shed the downstream
    # set in strict reverse-priority order — drive the lowest-priority
    # tier with reducible draw to zero, then escalate, re-arming each
    # poll until the line is ≤100 %.  Tier 1 is never shed (hard-lock);
    # if tiers 2-4 are exhausted and the line is still over, the auction
    # stops and records a ``line_relief_tier1_residual`` event (the line
    # is undersized for its critical through-load — a supply/topology
    # problem, not a control one).  Keeps the priority invariant green
    # where a priority-flat shed would invert tiers 2/3.  Electricity
    # line-relief only.  False falls back to the willingness-proportional
    # auction.
    enable_line_relief_waterfall: bool = True

    # Constraint-aware participation scaling inside the gossip step.
    # False: ``participation_scale = 1`` always.
    enable_constraint_aware_gossip: bool = True

    # Multi-hop ConstraintStateMessage forwarding from
    # GridConstraintMonitor.  False: only direct neighbours see the local
    # utilization.
    enable_multihop_constraint: bool = True

    # Priority-weighted waterfall S parameter in the holon ADMM.  False:
    # S=0 ⇒ ADMM redistributes to balance only, ignoring critical-tier
    # urgency.
    enable_priority_holon_allocation: bool = True

    # No-regret floor in EnergyBalanceNegotiator._apply_setpoint during
    # restoration directions.  True: a load that once reached factor=X
    # cannot drop below X in a later gossip round.  Kept False because
    # the floor blocks the L3→L2→L1 re-shed: when L3 commits a P2H
    # increase to serve high-priority heat, the source-sector L2 must
    # shed lower-priority elec loads to free supply, but the floor
    # preserves the stale factor and prevents the shed (LP
    # over-commitment / priority inversions).
    enable_monotonic_floor: bool = False

    # L2 priority-floor: clamp L1 reactive sheds (gossip ``balance`` +
    # ``stability`` re-apply) up to ``min(L2 allocation,
    # constraint-allowed fraction)`` for tiers 2/3/4.  Stops a
    # supply-poor local gossip group from shedding a load the
    # component-scope holon ADMM decided to serve.  The constraint-allowed
    # cap (same util as ``clamp_to_constraints``) lets curtailment/physics
    # still shed the load during a real violation, per-load and
    # continuously, so the floor and the clamp never fight.  Applies to
    # all tiers incl. tier 1 (whose cap is always 1.0, so the floor just
    # re-asserts its hard-lock against ``stability`` erosion).
    enable_l2_priority_floor: bool = True

    # Write the PHYSICALLY-actuated (constraint-clamped / floored) delta back
    # into the gossip ledger after ``_apply_setpoint`` so the dual sees a
    # constrained load's true (smaller) contribution and reallocates the freed
    # supply to unconstrained loads. False: the ledger keeps the unclamped
    # requested delta, so a constraint-clamped load's freed supply is never
    # re-served (the slack/generator just draws less). The constraint stays
    # solved either way — only the gossip accounting changes. Raw-restoration /
    # capacity-utilisation gain; PWSF-neutral (the freed supply, by
    # construction, only reaches lower-priority loads).
    enable_actuated_ledger_writeback: bool = True

    # Cold-load pickup ramp limit on regulation increases.  False: factor
    # jumps are not throttled.
    enable_clpu_ramp: bool = True

    # Heat-only curtailment-auction lock.  When a heat load is curtailed
    # for a live temperature violation (``reason="curtail"``), that lever
    # becomes authoritative: the L2 holon supply-priority dispatch must
    # DEFER (skip the load) rather than claw it back up — otherwise the
    # MW-based L2 re-dispatch restores the just-curtailed cold node,
    # re-cools it below the t_k floor, and the two layers limit-cycle.
    # The lock lifts when the frontier controller's ``heat_recovery``
    # ramp restores the load to ~1.0; else it persists (a load shed for a
    # permanent failure stays shed).  Heat-scoped; el/gas L2 untouched.
    enable_heat_curtail_lock: bool = True

    # Heat-only frontier feedback controller.  Each poll, every heat
    # load's GridConstraintMonitor drives its regulation toward the point
    # where its junction temperature sits at the feasibility floor
    # (~t_k = 313.15 K) — the maximum feasible service — using local
    # dT/dreg as the gain, rate-limited with a restore-hysteresis band.
    # Replaces bang-bang heat behaviour (``constraint_allowed_fraction``
    # is a feasibility GATE returning 0 for any out-of-bounds node, so
    # the holon/clamp can only serve-full→collapse→0 or shed-to-0) with
    # oracle-style partial-frontier serving.  Applies to ALL tiers incl.
    # tier-1.  Writes use ``reason="curtail"`` (shed) / ``"heat_recovery"``
    # (restore) so the heat curtail-lock makes L2 defer.  Heat-scoped.
    enable_heat_frontier: bool = True

    # Priority-waterfall gate for the heat frontier controller.  Heat
    # shedding is otherwise tier-blind: each load drives its OWN junction
    # temperature to the frontier independently, so the shed burden falls
    # by geography, not priority.  True: a cold heat load broadcasts its
    # (tier, reducible) and defers its own shed while a strictly
    # lower-priority load in its hydraulic region still has reducible
    # draw — so shedding follows priority order (lowest first).
    # Heat-scoped.
    enable_heat_priority_waterfall: bool = True

    # Local Q-V droop at every inverter-coupled PowerGenerator (PV).
    # Follows the VDE-AR-N 4105 Q(U) characteristic: piecewise-linear
    # with a 0.97–1.03 pu deadband, saturating at ±Q_max at 0.95 / 1.05.
    # Q_max is bounded by the inverter's apparent-power capability circle
    # (S_n = |p_n| / cos φ_min, cos φ_min = 0.95 for S_n ≤ 13.8 kVA, else
    # 0.90).  False: no droop role; reactive dispatch stays at simbench
    # defaults.
    enable_qv_droop: bool = True

    # Voltage reference for the Q(U) curve (per unit); VDE-AR-N 4105
    # anchors it at 1.0 pu.  Exposed for the re-centred-droop sweep.
    qv_droop_voltage_ref_pu: float = 1.0

    # Slack-infeed target as a fraction of the registered slack rating.
    # Per-community gossip biases its imbalance accounting toward
    # "balance ⇒ slack draws its target fraction of rating".  Only
    # matches operator intent when the slack's community spans the full
    # LP balance scope (``community_partition_method="connected_component"``);
    # for holonic/label-propagation partitions the derived target
    # contradicts the global budget.  Left at 0.0; budget enforcement
    # instead routes through
    # :class:`~scare.service.slack_budget.SlackBudgetMonitor`'s signed
    # ``override_target`` path, which is partition-agnostic.
    slack_target_fraction: float = 0.0

    # P6 primal-dual QP gossip.  True: the receiving agent computes δ_i
    # in closed form from the gossiped dual λ as
    # ``δ_i = clamp(w_i · λ, dmin_i, dmax_i)``, priority encoded
    # continuously in w_i, dual updated by gradient ascent on the primal
    # residual.  False: legacy equal-share / priority-gated update
    # (intra-tier serialisation via deterministic sub-rounds).  Both
    # share the saturation flag, stall detection, and step decay; only
    # the per-agent update rule differs.
    enable_qp_gossip: bool = True

    # Branch-side line-loading monitor.  True: every PowerLine branch
    # gets a GridConstraintMonitor watching its loading_percent; on
    # overload it sends StartBalanceNegotiation with a relief-MW override
    # to the branch's home group leader (the endpoint with lower
    # priority-weighted demand, so shedding falls on the less-critical
    # side) and propagates a ConstraintStateMessage to both endpoint
    # groups so neighbouring gossip agents throttle participation.
    # False: branch agents not registered, line overload is silent.
    enable_line_loading_constraint: bool = True

    # External-grid slack budget monitor.  True: every slack-class child
    # (ExtPowerGrid / ExtHydrGrid stamped with ``_scare_slack_budget_*``
    # by ``apply_slack_budget``) carries a ``SlackBudgetMonitor`` that
    # polls the LP-chosen ``p_mw`` / ``mass_flow`` and, when the absolute
    # draw exceeds ``budget · (1 + slack_budget_violation_tol)``, records
    # a ``slack_budget_violation`` event and emits a ``BalanceProblem``
    # so the co-located ``EnergyBalanceNegotiator`` triggers a rebalance.
    # Makes the operator ``slack_budget_pct`` a runtime-enforced
    # constraint, while the LP envelope (10× budget) stays wide enough to
    # keep the energy-flow solve feasible.
    enable_slack_budget_monitor: bool = True

    # Relative tolerance for the slack-budget monitor.  A draw is flagged
    # only when ``|obs| > budget · (1 + slack_budget_violation_tol)``;
    # the margin avoids flagging steady-state numerical wiggle of an LP
    # already converged inside the envelope.
    slack_budget_violation_tol: float = 0.05

    # Loss-compensating effective-budget feedback in the slack monitor.
    # The budget caps the slack's *actual* draw, but L1/L2/L3 control
    # only shapes served setpoints — network losses leave the realized
    # draw above budget even when the controller believes it hit target.
    # True: ``SlackBudgetMonitor`` runs an integral correction on the
    # observed overage and advertises a tightened *effective* budget
    # (``B - losses``) into the supply pool so actual draw converges to
    # the operator budget.  False: pool advertises the nominal budget.
    enable_slack_budget_feedback: bool = True

    # GridReconfigurator path ranking by line loading.  True: the
    # reconfigurator carries a running max_loading_percent along each
    # GridPathMessage, buffers all results within a short window, and
    # picks the path with the lowest peak loading instead of the
    # first-arrived (typically shortest).  False: legacy first-arrival.
    enable_reconfig_feasibility_ranking: bool = True

    # Window during which the reconfigurator collects candidate paths
    # before picking the best.  Sized for the electricity poll period;
    # too short loses alternatives, too long delays restoration.
    reconfig_path_window_s: float = 1.5

    # Holon ADMM iteration cap.  Concurrent holon ADMMs across sectors
    # must not block discrete-time progress; trades convergence quality
    # against wallclock cost (can leave residuals 1e-3 to 2e-2 unconverged
    # on simbench_lv).
    holon_admm_max_iters: int = 50

    # ADMM absolute residual tolerance — the coordinator stops when
    # |r| < this.  Relaxed from the package default 1e-4 so typical
    # scenarios converge inside the iteration cap.
    holon_admm_abs_tol: float = 1e-3

    # Tier-stratified holon ADMM.  True: the ADMM target vector is built
    # per-(sector, priority_tier) and the L1 honour path dispatches
    # per-tier targets directly to members, preserving the holon's global
    # priority decision through the L2 → L1 handoff.  False: one scalar
    # target per (member, sector) and L1 re-derives priority locally,
    # which can invert priority on finely-partitioned grids (see
    # priority_invariant claim in eval/claims.py).
    enable_tier_stratified_holon_admm: bool = True

    # Number of priority tiers the tier-stratified ADMM allocates over.
    # Mirrors ``base.util.tier_priority_weight``'s 4-tier schedule
    # (tier 1 = critical / hard-locked at the L1 leader pre-step;
    # tiers 2-4 = QP-weighted with 1e8 / 1e4 / 1.0 exponents).
    priority_tiers: int = 4

    # Holon ADMM scope — which actors participate in one ADMM round.
    #
    # - ``"component"`` (default): every *group leader* in the same
    #   active connected component is an ADMM actor.  The elected
    #   coordinator (lex-smallest reachable leader aid) collects each
    #   leader's community flex (supply + per-tier demand), runs the
    #   ADMM, and dispatches the per-tier ``service_fraction`` to every
    #   leader in the component; each leader applies the fractions to its
    #   OWN community members directly (no holon hop).  Aligns the
    #   optimisation scope with the priority-invariant claim's
    #   ``(sector, component)`` scope, so cross-leader inversions cannot
    #   arise and a component split re-elects independent coordinators.
    # - ``"sector"`` (deprecated): all holon leaders in the sector are
    #   actors; leaves communities not in any holon at factor=1.0.
    #   Retained as an ablation.
    # - ``"holon"`` (legacy): each holon leader runs its own ADMM over
    #   its member groups, producing per-holon fractions that need not
    #   agree across holons (cross-holon inversions).  Retained as an
    #   ablation.
    holon_admm_scope: str = "component"

    # Level-1 (sub-community) partition method.
    #
    # - ``"label_propagation"`` (default): radius-bounded min-label
    #   propagation — communities are
    #   ≤``community_label_propagation_radius``-hop balls centred on the
    #   lex-smallest reachable seed.
    # - ``"modularity"``: distributed-Louvain Phase 1 — communities
    #   maximise local modularity gain.  Sizes vary, not radius-bounded.
    # - ``"connected_component"``: one community per connected component
    #   of the per-sector subgraph.  Used by the ``component_level``
    #   baseline; gives the gossip negotiator a global per-component view.
    #   Combine with ``cps_join_communities=True`` so bridging CPs
    #   participate.
    community_partition_method: str = "label_propagation"

    # Radius bound for ``label_propagation`` (ignored by modularity).
    community_label_propagation_radius: int = 2

    # Resolution γ for ``modularity`` (ignored by label propagation).
    # γ > 1 ⇒ finer (more, smaller communities); γ < 1 ⇒ coarser.
    community_modularity_resolution: float = 1.0

    # Iteration cap for the modularity phase-1 sweep (converges in ~3-5).
    community_modularity_iterations: int = 10

    # ----------------------------------------------------------------
    # ``component_level`` baseline tunables
    # ----------------------------------------------------------------
    #
    # True: every CP agent joins the per-sector community of each
    # endpoint it bridges (a P2G bridges one electricity + one gas
    # community), becomes a normal member of those L1 gossip rounds, and
    # drops its separate CP-ADMM path — ``EnergyConverterRole`` is
    # replaced by ``MultiCommunityCPRole``, which collects per-community
    # signals, blends them with an EMA over per-sector setpoint targets,
    # and commits via ``apply_regulate`` under a deadband + cooldown
    # guard so a CP in two communities can't ping-pong between
    # contradictory asks.  Only meaningful with ``enable_cp_admm=False``;
    # the variant builder in ``experiment/hpc/runner.py`` enforces it.
    cps_join_communities: bool = False

    # EMA blending factor for the multi-community CP guard.  Each tick
    # the new per-sector target is ``α · proposed + (1 − α) · current``;
    # higher = more reactive, lower = heavier filtering.
    cp_oscillation_ema_alpha: float = 0.3

    # Minimum |target − current| (sector natural units — MW for
    # electricity/heat, kg/s for gas) below which a new target is treated
    # as noise and not committed.
    cp_oscillation_deadband_mw: float = 0.05

    # Minimum simulation-second gap between two regulation commits on the
    # same CP.
    cp_oscillation_min_interval_s: float = 1.0

    # ----------------------------------------------------------------
    # Sensitivity-sweep tunables
    # ----------------------------------------------------------------

    # monee-side solver throttle.  0: solve whenever monee decides.
    # Non-zero buffers regulate writes per aid and flushes at fixed
    # simulation-time boundaries (see comms wrapper, step 10).
    cooldown_s: float = 0.0

    # FailureNotice TTL stamped at endpoint detectors.
    ttl_hops: int = 3

    # Hop cost across CP-bridge edges in FailureNotice propagation;
    # same-sector edges always cost 1.
    cp_bridge_cost: int = 2

    # Maximum members per holon (excluding the initiator, the chunk's
    # lex-smallest leader).
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

    # ----------------------------------------------------------------
    # Logging detail
    # ----------------------------------------------------------------

    # True: every send_message goes through a counting wrapper so
    # per-message-type counts are reported in result.json.  High volume;
    # off except for the comm-cost campaign.
    record_messages: bool = False


def default_config() -> RestorationConfiguration:
    """Return a fresh baseline config (``RestorationConfiguration()``);
    the named function makes baseline intent explicit at call sites.
    """
    return RestorationConfiguration()
