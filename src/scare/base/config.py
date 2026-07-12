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

    # Outlet-temperature guard on heat-producing CPs (P2H/G2H/CHP). A CP injects
    # q_mw_heat into its outlet junction as pure energy; on a low-flow junction
    # the resulting ΔT = Q/(ṁ·c_p) drives t_k far past the envelope ceiling, and
    # NOTHING pushes back: the L3 kernel sees only demand−delivered (delivered is
    # measured at load setpoints, which injection can never raise), the heat
    # frontier owns only the LOW side, the auction skips t_k, and CP branches are
    # BORN at regulation=1.0 so even a zero-commit run injects at rated power
    # (eval_full_v2_20260711: hot CP outlets = the dominant compliance failure).
    # True: each heat-producing CP runs a reactive AIMD controller on its outlet
    # junction's t_k that maintains a regulation CEILING, enforced against every
    # L3 commit inside ``apply_regulate`` (sector="cp"). False: no guard.
    enable_cp_heat_outlet_guard: bool = True
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
    # solve keeping own row. NOTE: the gossip cascade was previously inert (its
    # commit callback never fired in sim time, so the CP converter-curtailment
    # actuator was dead and the electricity slack over-drew on cp-heavy grids).
    # Fixed by (a) the initiator in-flight-round guard (GossipParticipant.
    # is_round_active) so rounds aren't perpetually cancelled before committing,
    # and (b) sim-cadence round/iter timeouts (2.0s/0.2s) so each round commits
    # within the sim. Validated on cp_heavy_dependent @0.15: gossip == lexico
    # (electricity slack 2.1xB -> ~0.4xB, compliant). See
    # project_slack_compliance_rootcause.
    cp_admm_algorithm: str = "gossip"

    # Proximal step-damping α ≥ 0 for the lexicographic cascade's projection.
    # Biases the step, not the fixed point. α=0 is correct but can oscillate.
    cp_admm_r_regularization: float = 0.1

    # Gossip only: build the round's demand set over the UNION of every sector
    # present in the community (all bridged sectors), not just the elected
    # initiator's. The gossip initiator is the lowest cp_id, which on cp_heavy
    # grids is always a P2G ("branch-*" < "node-*", P2G ids sort before P2H), so
    # it bridges electricity+gas only and heat NEVER enters the broadcast demand
    # set — every heat CP (CHP/P2H) then commits regulation 0 while real tier-1
    # heat demand goes unserved. The replicated lexicographic path has no such
    # gap (each CP self-includes its own bridged sectors), so this realigns
    # gossip with it. False restores the initiator-only demand set.
    enable_cp_demand_union: bool = True

    # L3 CP input cap uses the NOMINAL operator budget instead of the
    # SlackBudgetMonitor's wound-down effective budget. The eff-budget integral
    # feedback (floor 0) exists to make L1/L2 shed native load toward B; feeding
    # the same signal to the CP kernel gives every η<1 converter zero input
    # headroom, so its converged optimum is r=0 — no CP ever dispatches (v2
    # campaign: SCARE CP gas output ≈ 0 vs oracle at nameplate). The kernel's
    # cascade LP arbitrates native serving vs CP input within Σserved + B
    # (input_capped_mode), so the pool can transiently over-credit the
    # slack-fed share of served load by up to B; the SlackBudgetMonitor still
    # enforces the physical draw at B, bounding the effect to over-commit /
    # churn rather than sustained over-draw (local A/B: no violation increase).
    # False restores the starved signal.
    enable_cp_nominal_budget: bool = True

    # Gossip only: warm-start each cascade round from the previous round's
    # ADMM state (r/x/z/u) instead of zeros. A round budget of ~11 iterations
    # (round_timeout_s / iter_timeout_s) cannot converge cold for N=20-33 CPs,
    # so every commit was an early partial iterate (~0); with carry-over the
    # rounds continue one another and converge across rounds. False restores
    # cold starts.
    enable_cp_gossip_warm_start: bool = True

    # Distributed FailureNotice propagation via ProblemDetector. False:
    # centralised callback triggers all leaders (legacy ablation).
    enable_distributed_failure_notice: bool = True

    # L2 dynamic holon-membership filter. True: drop members made physically
    # unreachable after a branch failure. False: static membership.
    enable_dynamic_holon_topology: bool = True

    # L3 dynamic CP-connector filter. True: drop group-leader peers made
    # physically unreachable through the cross-sector graph after a failure.
    enable_dynamic_cp_topology: bool = True

    # L2.5 cross-holon inversion detection: leaders publish per-tier summaries
    # and flag inversions across peers. Off disables the HolonSummaryRole and
    # coalition machinery only — the holon_summary_<sector> mesh itself is
    # always built under enable_holonic because coordinator election and
    # LeaderEmerged re-registration ride on it.
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

    # Coalition-pool supply accounting fix. The coalition acceptance pool
    # (`_local_acceptance`) credited generators at RATED |cap| and slacks without
    # the CP-reserve debit, so it allocated service fractions the slack must fund
    # PAST its budget; those fractions are then merged over the L2 ADMM as a
    # priority floor that clamps the SlackBudgetMonitor's override sheds UP for
    # the coalition TTL — structurally disarming budget enforcement (eval_full_v2:
    # elec child-118 draws 1.68× budget, coalition/cross-sector arms only). True:
    # credit non-slack generators at delivered |sp| (mirrors the L2 pool,
    # balance.py) and debit the slack's CP reserve, so coalition fractions stay
    # inside B. Also gates cross-sector coalitions on the CP transfer being
    # actuatable (a CPCommitment consumer exists — only under the legacy
    # EnergyConverterRole L3), so own-sector fractions are never raised on a
    # phantom transfer that the default L3 never delivers. See
    # project_cp_fix_design. False: legacy rated-capacity crediting.
    enable_coalition_delivered_supply: bool = True

    # Curtailment auction on hard violations. False: violations only emit a
    # BalanceProblem; no proportional curtailment broadcast.
    enable_curtailment_auction: bool = True

    # Generation-prioritised relief for excess-injection violations. True: an
    # over-voltage (vm_pu > hi) auction bids GENERATORS only (by reducible
    # output) and excludes loads, and an export line overload suppresses the
    # load-shed waterfall the moment it is export-classified with curtailable
    # downstream generation (instead of after the debounce). Cutting injection is
    # the only lever that lowers voltage / reverse-flow line loading; shedding
    # load on a PV-surplus feeder does not help.
    #
    # DEFAULT FALSE — the naive form is A/B-REFUTED on LV-S (paired Δpwsf
    # -0.047 clean, -0.056 pv_peak; served DOWN, agent_shed UP, voltage no
    # better). Root cause: relieving an export line by curtailing PV also removes
    # the PV that serves LOCAL load, and the gen curtail-lock then holds it at 0
    # (the balance layer's ramp-back requests are clamped), so the downstream
    # region starves and balance sheds MORE than the waterfall did. A correct
    # form must (a) bound the curtailment so PV keeps serving local load (cut
    # only the export excess) and (b) add a line/voltage-clear gen handoff (cf.
    # the line-relief lock's always-on bounded hand-off) so PV recovers to local service once the
    # constraint clears. Kept as an opt-in flag + tested machinery for that work.
    enable_generation_priority_curtailment: bool = False

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

    # Soft congestion-price line relief (v1: export/reverse-flow radial case).
    # True: an overloaded branch with curtailable downstream generation drives a
    # per-branch congestion price (AIMD integrator on the overshoot) instead of
    # the hard curtail-to-0 + gen curtail-lock of ``_relieve_export_overload``.
    # The price becomes a REVERSIBLE generation ceiling enforced softly in the
    # gossip ``_apply_setpoint`` (min(requested, 1 - Σprice)) under
    # ``reason='line_congestion'``, which does NOT arm the gen curtail-lock — so
    # gossip can ramp PV back up to serve LOCAL load up to the export-clearing
    # level, and the ceiling lifts as the line clears (price decays). Targets the
    # A/B-refuted pathology where the lock pinned PV at 0 and starved downstream
    # load. Load-shed on export lines stays suppressed. Electricity-only; leaves
    # the forward-flow load waterfall, over-voltage interlock, and L2/L3 intact.
    #
    # DEFAULT TRUE. A/B-VALIDATED on LV-S (16 paired seeds): pv_peak Δpwsf +0.065
    # (15/15 wins, served up, agent_shed down, voltage no worse), clean neutral
    # (−0.0005) — the correct inversion of the refuted gen-priority fix. Broader
    # no-regression validation (CP grids, line_stress, heat) still advisable.
    enable_line_congestion_price: bool = True

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

    # Direction-aware constraint capping in ``constraint_allowed_fraction`` /
    # ``clamp_to_constraints``. True: a load's served fraction is capped only by
    # the bound that SERVING pushes toward (voltage/pressure/temperature all fall
    # as a load draws), so over-voltage no longer sheds load — it is relieved by
    # serving load and is instead curtailed at generators. False (legacy):
    # symmetric cap that sheds load on over-voltage too (wrong lever on a
    # PV-surplus feeder; strands load shed while the slack has headroom).
    enable_directional_constraint_cap: bool = True

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
    # Validated on the PRE-2026-07-11 stock simbench_lv_small (density 0.2 —
    # the name now denotes the tuned 0.5-density grid) under pv_peak (n=40
    # deterministic, once the observation-lag bug was fixed via
    # ``energy_flow_max_acts``): VVW clears
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

    # End-of-sim settle tail. The final flush_energy_flow() before metrics are
    # written reveals the true converged power-flow, which can differ from the
    # last (cooldown-/act-throttled) solve the controllers observed — so the
    # constraints_final snapshot may record an over-voltage the droop/auction
    # never reacted to (the reproduced end-of-sim observation desync). When ON,
    # after the main run we alternate flush + a short discrete-step chunk up to
    # ``end_of_sim_settle_max_rounds`` times so controllers act on the revealed
    # state before the snapshot, breaking early once a flush leaves nothing
    # dirty. Default OFF: it shifts the final state of EVERY task/sector, so it
    # must be A/B-validated with the deterministic n=40 aggregate methodology
    # (see ``project_pv_overvoltage_levers``) before becoming default.
    enable_end_of_sim_settle: bool = False
    end_of_sim_settle_max_rounds: int = 3
    end_of_sim_settle_chunk_s: float = 2.0

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

    # Priority waterfall for the heat frontier controller. True: a cold heat
    # load defers its own (otherwise tier-blind) shed while strictly lower-
    # priority same-component loads hold enough reducible draw to cover the
    # step (sufficiency-gated, defer-budget-bounded), AND actively sends those
    # peers bounded CurtailmentRequests each poll — the shed authority that
    # makes the deferral a real waterfall instead of a wait-until-peers-freeze
    # no-op. Heat-scoped.
    #
    # DEFAULT FALSE since the heat L2 reconnect (enable_heat_l2_dispatch)
    # subsumes it: the A/B (ab_heat_priority_v2, 16 paired seeds) shows the
    # combined arm is dominated by L2-dispatch-alone on every gate (pass rate,
    # compliance, heat PWSF) — the local peer sheds fight the global per-tier
    # allocation. Re-enable only as the fallback heat-priority lever when L2
    # dispatch is ablated off.
    enable_heat_priority_waterfall: bool = False

    # Reconnect heat to the L2 holon per-tier allocation. True: heat leaders
    # actuate the component ADMM's per-(sector, tier) service fractions
    # (gossip stays heat-excluded), holon flex reports delivered heat as the
    # sector supply (the physics-delivered total the priority waterfall can
    # reallocate across tiers), and heat holons re-run the component
    # allocation periodically (heat has no gossip/failure trigger of its
    # own). The frontier + curtail-lock keep temperature-feasibility
    # authority: L2 raises defer on locked loads, L2 sheds always pass.
    # A/B-validated (ab_heat_priority_v2, 16 paired seeds): heat_priority
    # pass 0-6% -> 50-69%, inversions 2.4 -> 0.3/task, tier-1 heat +0.09..
    # +0.12, compliance no worse; costs bounded tier-4 over-shed on well-
    # supplied grids (heat PWSF -0.01 stress / -0.04 clean; probe share is
    # the tuning knob). The frontier peer-shed waterfall FIGHTS this lever
    # (worse on every gate combined than alone) — prefer disabling
    # enable_heat_priority_waterfall wherever this is on.
    enable_heat_l2_dispatch: bool = True
    # Cadence (s) of the heat holon's periodic rebalance trigger; aligned by
    # default with heat_cp_supply_refresh_s so allocations track the summary.
    heat_l2_rebalance_s: float = 2.0

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

    # R3: L2 service-fraction dispatch (Route A) ramps dispatchable DGs toward
    # covering the served demand instead of only shedding loads. Without it,
    # enforcement is load-shed-only: the holon ADMM sizes service fractions
    # assuming the full generator pool, but _dispatch_service_fractions never
    # issues a generator setpoint, so load is shed to the un-ramped generation
    # level (the 78%-gen / 54%-slack oracle gap). Slacks excluded (grid-
    # following, no regulation knob). True: ramp; False: legacy shed-only.
    enable_l2_generator_ramp: bool = True

    # Coordination overhaul: reactive notify-on-change cascade. When True:
    # (a) UPWARD — an L1 gossip notifies L2/L3 only when its converged setpoint
    #     actually moved (balance.py _finish_negotiation), so a re-converged-to-
    #     same gossip does not re-trigger the holon ADMM;
    # (b) DOWNWARD — an unchanged L2 allocation re-asserts the per-load priority
    #     floor (set_l2_priority_floor) WITHOUT abandoning the in-flight gossip,
    #     instead of preempting it (the dominant "yielding to L2" abandonment);
    # (c) the rebalance_min_gap_s time-throttle is bypassed — the change-
    #     detection makes the holon→member→finished→holon cascade self-terminate
    #     at a fixed point, so the time fuse is no longer needed.
    # Local A/B (v3, 2026-06-29): abandon-rate −15.3pp, priority + diary
    # invariants 12/12, no feedback-storm, served/violations flat (one wart: a
    # few un-abandoned gossips run to their convergence timeout). False: legacy
    # throttled re-broadcast with mid-flight L2 preemption.
    enable_change_only_dispatch: bool = True

    # Fix 2 (opt-in): size the holon's load-serving supply pool at the slack's
    # nominal operator budget B (== abs(cap), the same hard ext-grid bound the
    # oracle uses) instead of the SlackBudgetMonitor's loss-compensation
    # eff_budget. The eff_budget feedback winds DOWN toward 0 whenever physical
    # import exceeds budget (e.g. irreducible CP coupling draw on cp-heavy
    # grids), shrinking the pool below what the network can physically serve and
    # over-shedding feasible load. The eff_budget still governs the slack's own
    # setpoint; with this flag it no longer also caps serviceable supply. False:
    # unchanged eff_budget behaviour. See root-cause "cp-dependent over-shed".
    enable_nominal_slack_supply: bool = False

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

    # Restore / serve-more lever (opt-in). The SlackBudgetMonitor normally only
    # sheds when OVER budget; nothing restores load when the slack sits UNDER
    # budget, so after curtailment the import headroom is left unused and load is
    # over-shed (the oracle uses 100% of the budget; SCARE ~54%). True: when the
    # draw is below ``budget*(1-tol)`` the monitor sends a POSITIVE
    # override_target (= headroom to B) to the home leader, whose L1 gossip
    # restores load (highest priority first) up to the budget. Symmetric with the
    # shed path and deadband-gated so it can't oscillate. False: shed-only.
    # See project_oracle_quality_gap.
    enable_slack_restore: bool = False

    # Generation ramp-to-full lever (opt-in). Generators are often left below
    # full output (mean reg ~0.78 vs the oracle's 1.0), wasting local supply that
    # would displace slack import and free budget to serve more load. True: each
    # generator's GenerationController periodically ramps its own setpoint toward
    # rated (reg=1.0) via apply_regulate with a GEN_RESTORE reason, so the
    # over-voltage curtail-ramp interlock still defers it when the auction holds
    # the generator down. Local per-generator (no reachability dependence).
    # False: generators only follow gossip/stability dispatch. See
    # project_oracle_quality_gap.
    enable_gen_ramp_to_full: bool = False

    # CP-aware slack supply (opt-in correctness fix). On cp-heavy grids the
    # cross-sector converter (power-to-heat / CHP) electricity draw rides the
    # electricity slack but is invisible to the L2 holon balance (it is not a
    # PowerLoad / community member), so the holon serves native load up to
    # ``gen + B`` while the physical slack draws ``B + CP_draw`` — over budget.
    # True: the SlackBudgetMonitor's measured over-draw is debited from the
    # slack's credited supply in the holon pool, so the holon balances native
    # load against the budget NET of the CP draw and sheds native load until the
    # physical slack lands at B (the deficit is routed through holonic balancing
    # before CP, as intended). False: legacy (CP draw uncounted). Default OFF:
    # forensics showed this debits a CROSS-sector converter draw against the
    # SAME-sector native-electricity pool, so it over-sheds native load without
    # touching the converter draw (the real lever is the L3 CP converter
    # curtailment via the CP priority ADMM — see enable_cp_priority_admm /
    # cp_admm_algorithm). See project_slack_compliance_rootcause.
    enable_cp_aware_slack_supply: bool = False

    # NOTE: a CP-facing slack-budget reserve debit (feed the SlackBudgetMonitor's
    # measured over-draw into the CP kernel's input cap) was prototyped and
    # REFUTED by A/B (ab_cp_slack_debit_20260712, steady-state grading,
    # simbench_lv_cp_heavy_dependent @0.15): the scenario was already 100%
    # slack-compliant and the debit DROPPED it to 75% — a peak-hold reserve
    # latches the un-actionable post-failure transient spike and suppresses CP
    # dispatch, and on cp-heavy-dependent grids the CPs are net producers so
    # winding them down shifts native load onto the slacks (same perverse loop as
    # enable_cp_nominal_budget=False). Removed; a correct version must exclude the
    # startup transient (e.g. N sustained over-budget polls). The paired feedback
    # target shift (slack_budget.py, target B·(1−margin)) was kept — it is
    # unconditional and the debit-off arm carrying it was 100% compliant. See
    # project_cp_fix_design.

    # Layer-0 gas pressure regulator on each gas ExtHydrGrid slack. True: the
    # slack autonomously drives its ``pressure_pu`` setpoint (the regulator-
    # station lever) to hold downstream junction pressure inside the band,
    # sensed via the constraint mesh — tried BEFORE shedding. Flow-neutral in
    # this model (loads fix withdrawals), so a near-free lever. When the profile
    # spread exceeds the band the setpoint saturates and the residual is left to
    # the existing pressure-violation shedding path. False: no regulator role
    # (legacy — pressure handled only by shedding). Gas-only.
    enable_gas_pressure_regulator: bool = True

    # Feedback step fraction toward the target setpoint per poll for the gas
    # pressure regulator. <1 because the Weymouth map is nonlinear (one-shot
    # overshoots); the loop converges over a few gas ticks.
    gas_pressure_regulator_gain: float = 0.5

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
