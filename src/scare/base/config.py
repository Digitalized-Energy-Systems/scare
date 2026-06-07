"""Configuration for the restoration scenario builder.

A single dataclass of enable-flags and tunables; defaults give the baseline.
Used by the eval harness for ablations and sensitivity sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RestorationConfiguration:
    # --- Architectural levels (ablation flags) ---

    # L2 holonic ADMM across same-sector group leaders. False: no holon role,
    # rebalancing falls back to local triggers + local-gen fallback.
    enable_holonic: bool = True

    # Re-link heat sector to L3 CP coordination. True: cold-load leaders signal
    # delivered heat as L3 base supply, so unmet demand ramps heat CPs (CHP/P2H).
    # False: L3 sees the slack pool, judges heat supplied, leaves CPs idle.
    enable_heat_cp_supply: bool = True
    # Refresh cadence (s) for a heat leader's HolonSummary so the delivered-heat
    # vector reaches L3 (heat's normal L1/L2 triggers are off).
    heat_cp_supply_refresh_s: float = 2.0

    # L3 cross-sector ADMM at coupling-point agents. False: CP roles not
    # installed, ``cps`` topology empty.
    enable_cp_admm: bool = True

    # Replicated priority-cascaded sharing ADMM at L3 (no elected coordinator).
    # True: every CP runs the L3 kernel on its replicated peer view and commits
    # its own factor. False: legacy coordinator path; both False = no L3 role.
    enable_cp_priority_admm: bool = True

    # L3 kernel: ``"gossip"`` (default) = coordinator-free peer-to-peer sharing
    # ADMM, crash-fault tolerant; ``"lexicographic"`` = replicated full-problem
    # solve keeping own row (ablation).
    cp_admm_algorithm: str = "gossip"

    # Proximal step-damping α ≥ 0 for the lexicographic cascade's projection.
    # Biases the step, not the fixed point. α=0 is correct but can oscillate.
    cp_admm_r_regularization: float = 0.1

    # Distributed FailureNotice propagation via ProblemDetector. False:
    # centralised callback triggers all leaders (legacy ablation).
    enable_distributed_failure_notice: bool = True

    # L2 dynamic holon-membership filter. True: drop members made physically
    # unreachable after a branch failure. False: static membership.
    enable_dynamic_holon_topology: bool = True

    # L3 dynamic CP-connector filter. True: drop group-leader peers made
    # physically unreachable through the cross-sector graph after a failure.
    enable_dynamic_cp_topology: bool = True

    # L2.5 sector-wide holon-summary mesh + cross-holon inversion detection.
    # Leaders publish per-tier summaries and flag inversions across peers.
    # Cheap when off (no topology, no role).
    enable_holon_summary: bool = True

    # Period (s) between HolonSummary publishes; also the invariant-check
    # cadence. Faster catches inversions sooner but costs O(N²) messages.
    holon_summary_period_s: float = 0.25

    # Per-tier served-fraction tolerance for flagging an inversion: pair
    # (high, low) flagged when frac[high] < frac[low] - tol.
    holon_summary_inversion_tol: float = 1e-3

    # L2.5 coalition formation. True: the detecting lex-smallest leader opens an
    # ad-hoc coalition, runs scoped priority-greedy allocation, broadcasts
    # service-fraction constraints (TTL'd). False: detect + record only.
    enable_holon_coalition: bool = False

    # Window (s) the initiator waits after CoalitionInvitation before allocating.
    # Short vs the tick cadence; long enough for loaded peers to reply.
    holon_coalition_accept_window_s: float = 1.0

    # TTL (s) on coalition constraints; after it the L2 holon ADMM takes over.
    # Outlasts a few L2.5 ticks but not the next major grid change.
    holon_coalition_constraint_ttl_s: float = 8.0

    # Cross-sector coalition extension (L2.5). True: also detect inversions
    # across CP-connected sectors and form a coalition spanning both sectors +
    # bridging CP(s). False: per-sector behaviour only.
    enable_cross_sector_coalitions: bool = False

    # Curtailment auction on hard violations. False: violations only emit a
    # BalanceProblem; no proportional curtailment broadcast.
    enable_curtailment_auction: bool = True

    # Gate the curtailment auction where it can't help (no-op if auction off).
    # True adds two guards: (a) SCOPE — never fire on heat t_k or line
    # loading_percent (its blind willingness-bidding can't target those); still
    # fires on node-local vm_pu/pressure_pu. (b) PROGRESS GATE — stop re-arming a
    # variable that hasn't improved for ``_CURTAIL_NO_PROGRESS_LIMIT`` rounds.
    enable_curtail_auction_gating: bool = False

    # Cross-sensitivity targeting for the auction. True: CurtailmentNeed carries
    # the violation origin and bidders weight willingness by electrical proximity
    # (bounded within-tier multiplier, priority stays dominant), so shed lands on
    # loads nearest the violation. Pairs with ``enable_curtail_auction_gating``.
    enable_curtail_auction_targeting: bool = False

    # Iterative line-overload relief. False (legacy): relief sent ONCE per
    # episode, line plateaus above 100%. True: RE-ASSERTED every poll while still
    # overloaded, magnitude from live overshoot, driving toward feasibility.
    # Electricity line-relief only.
    enable_line_relief_reassert: bool = False

    # Branch-downstream targeted line relief. True: each branch monitor runs the
    # auction against only the loads disconnected when the branch is removed
    # (its downstream subtree), so shed actually relieves the line. Meshed/
    # non-bridge branches get an empty set and fall back to legacy relief.
    enable_branch_downstream_relief: bool = True

    # Strict reverse-priority waterfall for the branch-downstream auction (only
    # with ``enable_branch_downstream_relief``). True: shed the downstream set
    # lowest-tier-first, escalating each poll until ≤100%. Tier 1 never shed; if
    # tiers 2-4 exhausted and still over, record ``line_relief_tier1_residual``
    # (line undersized — a supply/topology problem). False: willingness-
    # proportional auction. Electricity line-relief only.
    enable_line_relief_waterfall: bool = True

    # Constraint-aware participation scaling in the gossip step. False:
    # participation_scale = 1 always.
    enable_constraint_aware_gossip: bool = True

    # Multi-hop ConstraintStateMessage forwarding. False: only direct neighbours
    # see local utilization.
    enable_multihop_constraint: bool = True

    # Priority-weighted waterfall S in the holon ADMM. False: S=0 ⇒ balance only,
    # ignoring critical-tier urgency.
    enable_priority_holon_allocation: bool = True

    # No-regret floor in ``_apply_setpoint``: a load that reached factor X can't
    # drop below X later. Kept False — it blocks the L3→L2→L1 re-shed needed when
    # L3 commits a P2H increase (causes LP over-commitment / priority inversions).
    enable_monotonic_floor: bool = False

    # L2 priority-floor: clamp L1 sheds up to min(L2 allocation, constraint-
    # allowed fraction) for tiers 2/3/4, so a supply-poor gossip group can't shed
    # a load the holon ADMM serves. The constraint cap lets real violations still
    # shed. Applies to all tiers (tier 1 cap is 1.0, re-asserting its hard-lock).
    enable_l2_priority_floor: bool = True

    # Write the physically-actuated (clamped/floored) delta back into the gossip
    # ledger so the dual reallocates freed supply to unconstrained loads. False:
    # ledger keeps the requested delta, freed supply never re-served. Constraint
    # stays solved either way; raw-restoration gain, PWSF-neutral.
    enable_actuated_ledger_writeback: bool = True

    # Curtail-vs-ramp interlock. True: while the auction curtails a generator for
    # a live over-voltage, local-gen RESTORE paths DEFER instead of ramping back
    # to 1.0. Fixes a limit cycle (multiplicative curtail restarts from full each
    # time a restore resets the PV). TTL-lifted; electricity-only.
    enable_curtail_ramp_interlock: bool = True

    # Volt-VAR-Watt support (DEFAULT ON — best voltage-control variant). False:
    # Q(U) droop caps reactive at the VDE-AR-N 4105 displacement envelope
    # (q_max = p·tanφ, which shrinks as active power is curtailed — too weak to
    # clear over-voltage once curtailment bites). True: each inverter uses its full
    # apparent-power circle (IEEE 1547-2018), so reactive GROWS as p is curtailed.
    # Validated on simbench_lv_small/pv_peak (n=40 deterministic, once the
    # observation-lag bug was fixed via ``energy_flow_max_acts``): VVW clears
    # over-voltage on 98% of seeds (vs 68% for the cos-φ-capped droop) at ~10-12pp
    # LESS PV curtailment. The earlier "VVW makes curtailment worse" reading was an
    # artifact of stale power-flow observations, not VVW. The auction-coordination
    # variant (``enable_qv_auction_coordination``) matched plain VVW within noise,
    # so it is NOT enabled by default — plain VVW is the recommended variant.
    # See project_pv_overvoltage_levers.
    enable_vvw_coordination: bool = True

    # Coordinated Q(U)-droop / auction hand-off (experimental, OFF). Fixes the
    # naive VVW "stacking" (reactive + active both fully correct the same over-
    # voltage, so curtailment rises). With this the levers share a reactive-relief
    # ledger: (A) the auction sheds only the RESIDUAL after the local reactive
    # lever, skipping active when reactive covers it; (B) the gen curtail-lock
    # releases early once the droop reports spare reactive headroom. Levers
    # substitute instead of stack. Best paired with ``enable_vvw_coordination``.
    enable_qv_auction_coordination: bool = False

    # Phase-2 feeder-aware gate for the hand-off (only with
    # ``enable_qv_auction_coordination``). Phase-1 is purely local and can leave
    # neighbours over-volt. With this, the auction stops deferring and sheds
    # active whenever ANY feeder node is over-bound (from the gossip neighbour
    # cache). Errs safe (more curtailment) on stale data.
    enable_qv_feeder_gate: bool = True

    # Force an immediate energy-flow recompute the instant a branch fails,
    # bypassing the ``energy_flow_cooldown_s`` throttle (default 0.1 s).  A
    # topology change invalidates the cached power-flow, but the cooldown
    # otherwise makes agents react to STALE pre-failure voltages for up to a
    # cooldown window — observed driving the coordinated control to inject
    # reactive the wrong way and leave the feeder over-volt (the residual
    # catastrophic tail).  Costs ONE extra solve per failure event (failures
    # are rare), not a global cadence increase, so wallclock is unaffected.
    enable_recompute_on_failure: bool = True

    # Optional override for the env energy-flow recompute cooldown (s).  None
    # keeps the env default (0.1).  Lower = less observation lag (agents react
    # to fresher power-flow) at more solves/wallclock.  Diagnostic/tuning knob
    # for the delayed-feedback over-voltage instability.
    energy_flow_cooldown_s_override: float | None = None

    # Adaptive energy-flow recompute: force a re-solve once this many
    # setpoint-changing acts have piled up on the current solve, even within
    # the cooldown.  Bounds observation lag during a post-failure control
    # flurry (fixes the delayed-feedback over-voltage instability) while
    # staying on the cheap timer when the grid is quiet.  None/0 = disabled.
    energy_flow_max_acts: int | None = 8

    # Cold-load pickup ramp limit on regulation increases. False: factor jumps
    # not throttled.
    enable_clpu_ramp: bool = True

    # Heat-only curtailment-auction lock. When a heat load is curtailed for a
    # live temperature violation, the L2 dispatch must DEFER rather than claw it
    # back (else the two layers limit-cycle). Lifts when the frontier controller
    # restores the load to ~1.0. Heat-scoped; el/gas L2 untouched.
    enable_heat_curtail_lock: bool = True

    # Heat-only frontier feedback controller. Each poll, drive each heat load's
    # regulation toward its junction temperature sitting at the feasibility floor
    # (max feasible service), using local dT/dreg as gain. Replaces bang-bang
    # gating with partial-frontier serving. All tiers; heat-scoped.
    enable_heat_frontier: bool = True

    # Priority-waterfall gate for the heat frontier controller. True: a cold heat
    # load defers its own (otherwise tier-blind) shed while a strictly lower-
    # priority load in its hydraulic region still has reducible draw. Heat-scoped.
    enable_heat_priority_waterfall: bool = True

    # Local Q-V droop at every inverter-coupled PV. VDE-AR-N 4105 Q(U): piecewise-
    # linear, 0.97–1.03 pu deadband, ±Q_max at 0.95/1.05, Q_max bounded by the
    # inverter's apparent-power circle. False: no droop role; simbench defaults.
    enable_qv_droop: bool = True

    # Voltage reference for the Q(U) curve (p.u.); VDE-AR-N 4105 anchors at 1.0.
    # Exposed for the re-centred-droop sweep.
    qv_droop_voltage_ref_pu: float = 1.0

    # Prior |dV/dQ| (p.u./MVar) seeding the droop's sensitivity EMA, used only
    # under ``enable_qv_auction_coordination`` to size advertised reactive relief.
    # LOWER under-credits reactive (auction sheds more active, robust); HIGHER
    # credits more reactive (saves active, risks deferring to a weak droop). 0.03
    # was the only 8-seed-clean value. See ``project_pv_overvoltage_levers``.
    qv_dvdq_prior: float = 0.03

    # Slack-infeed target as a fraction of slack rating. Only matches operator
    # intent under ``connected_component`` partitioning; contradicts the global
    # budget otherwise. Left at 0.0; budget enforcement uses SlackBudgetMonitor's
    # partition-agnostic ``override_target`` instead.
    slack_target_fraction: float = 0.0

    # P6 primal-dual QP gossip. True: receiver computes δ_i = clamp(w_i·λ, dmin,
    # dmax) in closed form, priority in w_i, dual by gradient ascent. False:
    # legacy equal-share / priority-gated update. Share saturation/stall/decay.
    enable_qp_gossip: bool = True

    # Branch-side line-loading monitor. True: every PowerLine branch watches
    # loading_percent and, on overload, sends a relief-MW override to its home
    # leader (lower priority-weighted side) + ConstraintStateMessage to both
    # endpoints. False: line overload is silent.
    enable_line_loading_constraint: bool = True

    # External-grid slack budget monitor. True: each slack child polls its draw
    # and, when it exceeds budget·(1+tol), records ``slack_budget_violation`` and
    # emits a BalanceProblem to trigger a rebalance — making ``slack_budget_pct``
    # runtime-enforced while the LP envelope (10× budget) stays feasible.
    enable_slack_budget_monitor: bool = True

    # Relative tolerance for the slack-budget monitor; a draw is flagged only
    # when |obs| > budget·(1+tol). Margin avoids flagging steady-state wiggle.
    slack_budget_violation_tol: float = 0.05

    # Loss-compensating effective-budget feedback. Network losses leave realized
    # draw above budget. True: monitor runs an integral correction and advertises
    # a tightened effective budget (B − losses) so actual draw converges to the
    # operator budget. False: advertises the nominal budget.
    enable_slack_budget_feedback: bool = True

    # GridReconfigurator path ranking by line loading. True: buffer results in a
    # window and pick the path with the lowest peak loading. False: legacy
    # first-arrival (typically shortest).
    enable_reconfig_feasibility_ranking: bool = True

    # Window (s) the reconfigurator collects candidate paths before picking.
    # Too short loses alternatives, too long delays restoration.
    reconfig_path_window_s: float = 1.5

    # Holon ADMM iteration cap. Trades convergence quality against wallclock (can
    # leave residuals 1e-3 to 2e-2 unconverged on simbench_lv).
    holon_admm_max_iters: int = 50

    # ADMM absolute residual tolerance (stop when |r| < this). Relaxed from the
    # package default 1e-4 so scenarios converge within the iteration cap.
    holon_admm_abs_tol: float = 1e-3

    # Tier-stratified holon ADMM. True: target built per-(sector, tier) and
    # dispatched per-tier to members, preserving the holon's priority decision
    # through L2→L1. False: one scalar target, L1 re-derives priority (can invert
    # on finely-partitioned grids).
    enable_tier_stratified_holon_admm: bool = True

    # Number of priority tiers the tier-stratified ADMM allocates over. Mirrors
    # ``base.util.tier_priority_weight`` (tier 1 hard-locked; 2-4 QP-weighted
    # 1e8/1e4/1.0).
    priority_tiers: int = 4

    # Holon ADMM scope — actors per ADMM round.
    # - ``"component"`` (default): every group leader in the same active
    #   connected component is an actor; the lex-smallest reachable leader
    #   coordinates and dispatches per-tier service_fraction. Matches the
    #   priority-invariant claim's (sector, component) scope.
    # - ``"sector"`` (deprecated): all holon leaders in the sector; non-holon
    #   communities stay at 1.0. Ablation.
    # - ``"holon"`` (legacy): each holon leader runs its own ADMM (cross-holon
    #   inversions possible). Ablation.
    holon_admm_scope: str = "component"

    # L1 (sub-community) partition method.
    # - ``"label_propagation"`` (default): ≤radius-hop balls from the lex-smallest
    #   seed.
    # - ``"modularity"``: distributed-Louvain phase 1; sizes vary, not radius-
    #   bounded.
    # - ``"connected_component"``: one community per connected component; used by
    #   the ``component_level`` baseline (pair with ``cps_join_communities=True``).
    community_partition_method: str = "label_propagation"

    # Radius bound for ``label_propagation`` (ignored by modularity).
    community_label_propagation_radius: int = 2

    # Resolution γ for ``modularity`` (ignored by label propagation). γ>1 ⇒ finer,
    # γ<1 ⇒ coarser.
    community_modularity_resolution: float = 1.0

    # Iteration cap for the modularity phase-1 sweep (converges in ~3-5).
    community_modularity_iterations: int = 10

    # --- ``component_level`` baseline tunables ---
    #
    # True: every CP joins the per-sector community of each endpoint it bridges
    # and becomes a normal L1 member, dropping its CP-ADMM path for a
    # MultiCommunityCPRole (EMA-blended per-sector targets under a deadband +
    # cooldown guard). Only meaningful with ``enable_cp_admm=False``.
    cps_join_communities: bool = False

    # EMA blend for the multi-community CP guard: target = α·proposed +
    # (1−α)·current. Higher = more reactive, lower = heavier filtering.
    cp_oscillation_ema_alpha: float = 0.3

    # Min |target − current| (MW el/heat, kg/s gas) below which a new target is
    # treated as noise and not committed.
    cp_oscillation_deadband_mw: float = 0.05

    # Min sim-second gap between two regulation commits on the same CP.
    cp_oscillation_min_interval_s: float = 1.0

    # --- Sensitivity-sweep tunables ---

    # monee-side solver throttle. 0: solve whenever monee decides. Non-zero
    # buffers regulate writes per aid and flushes at fixed time boundaries.
    cooldown_s: float = 0.0

    # FailureNotice TTL stamped at endpoint detectors.
    ttl_hops: int = 3

    # Hop cost across CP-bridge edges (same-sector edges cost 1).
    cp_bridge_cost: int = 2

    # Max members per holon (excluding the lex-smallest initiator).
    holon_max_size: int = 4

    # Gossip protocol parameters.
    gossip_termination_tolerance: float = 1e-5
    gossip_max_hops: int = 100

    # --- Robustness / scenario perturbations ---

    # Per-message drop probability [0,1] (custom mango DelayProvider); 0 lossless.
    comms_packet_loss_pct: float = 0.0

    # Latency jitter stddev (ms) via a Poisson-mixed delay provider; 0
    # deterministic.
    comms_latency_jitter_ms: float = 0.0

    # --- Logging detail ---

    # True: route every send_message through a counter for per-type counts in
    # result.json. High volume; off except the comm-cost campaign.
    record_messages: bool = False


def default_config() -> RestorationConfiguration:
    """Return a fresh baseline config; names baseline intent at call sites."""
    return RestorationConfiguration()
