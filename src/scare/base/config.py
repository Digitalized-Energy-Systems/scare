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

    # Outlet-temp guard on heat-producing CPs (P2H/G2H/CHP). True: an AIMD
    # controller holds a regulation CEILING on each CP outlet junction's t_k,
    # enforced against every L3 commit in ``apply_regulate`` (sector="cp"); without
    # it nothing pushes back on injection-driven over-temp. False: no guard.
    # (eval_full_v2_20260711: hot CP outlets = the dominant compliance failure.)
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

    # L3 kernel: ``"gossip"`` (default, coordinator-free CFT sharing ADMM) or
    # ``"lexicographic"`` (replicated full solve). Gossip was previously inert
    # (commit callback never fired -> CP curtailment dead, elec slack over-drew on
    # cp-heavy); fixed by the in-flight-round guard (is_round_active) + sim-cadence
    # 2.0s/0.2s timeouts. Validated cp_heavy_dependent @0.15: gossip == lexico,
    # slack 2.1xB -> ~0.4xB. See project_slack_compliance_rootcause.
    cp_admm_algorithm: str = "gossip"

    # Proximal step-damping α ≥ 0 for the lexicographic cascade's projection.
    # Biases the step, not the fixed point. α=0 is correct but can oscillate.
    cp_admm_r_regularization: float = 0.1

    # Gossip only: build the round demand set over the UNION of all bridged
    # sectors, not just the initiator's. Initiator = lowest cp_id = a P2G on
    # cp_heavy ("branch-*" < "node-*"), which bridges elec+gas only, so heat never
    # enters the demand set and every heat CP commits reg 0 while tier-1 heat
    # starves; the lexicographic path self-includes. False: initiator-only set.
    enable_cp_demand_union: bool = True

    # L3 CP input cap uses the NOMINAL operator budget, not the wound-down
    # eff-budget. The floor-0 eff-budget feedback (for L1/L2 native shed) gives
    # every η<1 converter zero input headroom -> converged r=0, no CP dispatches
    # (v2: SCARE CP gas output ≈ 0 vs oracle nameplate). Cascade LP +
    # SlackBudgetMonitor bound the effect to over-commit/churn, not sustained
    # over-draw (local A/B: no violation increase). False: starved signal.
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

    # Period (s) of the forced re-publish / re-form / re-balance watchdogs in
    # HolonSummaryRole, HolonicCommunityRole and CPPriorityAdmmRole. These exist
    # to break a stalled delta gate: a publish is suppressed while no tier moves,
    # so an allocation that stops changing is frozen until a watchdog forces one
    # through. 30.0 is the historical hardcoded value and equals
    # ``simulation_duration_s`` in every shipped campaign config, so the watchdogs
    # fire at most once, at the horizon — i.e. never usefully. Kept as the default
    # so runs stay byte-identical to prior campaigns; set it well below the
    # horizon to actually arm the recovery path.
    holon_watchdog_s: float = 30.0

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

    # Coalition-pool supply fix. True: credit non-slack gens at delivered |sp| +
    # debit slack CP reserve (fractions stay inside B), gate cross-sector on an
    # actuatable CPCommitment consumer. Legacy rated-|cap|/no-debit let coalition
    # floors clamp SlackBudgetMonitor sheds UP, disarming enforcement (eval_full_v2:
    # child-118 drew 1.68× budget, coalition arms only). project_cp_fix_design.
    enable_coalition_delivered_supply: bool = True

    # Curtailment auction on hard violations. False: violations only emit a
    # BalanceProblem; no proportional curtailment broadcast.
    enable_curtailment_auction: bool = True

    # Gen-prioritised relief for excess-injection: over-voltage/export-overload
    # bid GENERATORS only, suppress the load-shed waterfall. DEFAULT FALSE — naive
    # form A/B-REFUTED on LV-S (Δpwsf -0.047 clean, -0.056 pv_peak; served down,
    # shed up, voltage no better): curtailing export PV starves local load +
    # curtail-lock pins it at 0. Correct form bounds to export-excess + clear-time
    # gen handoff; opt-in.
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

    # Soft congestion-price line relief (export/reverse-flow radial). True: an
    # overloaded branch drives a per-branch AIMD price used as a REVERSIBLE gen
    # ceiling in gossip ``_apply_setpoint`` (min(req, 1 - Σprice),
    # reason='line_congestion') that does NOT arm the curtail-lock, so PV ramps
    # back to serve local load. Elec-only. DEFAULT TRUE — A/B-validated LV-S
    # (16 seeds): pv_peak Δpwsf +0.065 (15/15 wins), clean neutral (−0.0005).
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

    # Age-out (s) for the proactive-utilization entries feeding that throttle.
    # ConstraintWarning fires only at util >= PROACTIVE_WARNING_FRACTION (0.85)
    # and the recovery path emits nothing, so an entry -- always in [0.85, 1.0]
    # -- otherwise pins the agent's participation_scale <= 0.15 for the rest of
    # the run. 0.0 keeps that latching behaviour (the pre-fix default); a
    # positive value expires stale entries on read. Opt-in: clearing the latch
    # changes gossip participation and therefore campaign numbers.
    proactive_util_ttl_s: float = 0.0

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

    # Volt-VAR-Watt (DEFAULT ON, best voltage variant): inverters use the full
    # apparent-power circle (IEEE 1547-2018) so reactive GROWS as active is
    # curtailed; False = cos-φ/VDE-AR-N 4105 envelope droop (q_max=p·tanφ, too weak
    # once curtailment bites). Validated pre-2026-07-11 simbench_lv_small pv_peak
    # (n=40, after the ``energy_flow_max_acts`` obs-lag fix): clears over-voltage
    # 98% vs 68% at ~10-12pp less curtailment — the earlier "VVW makes curtailment
    # worse" reading was a stale-observation artifact. See project_pv_overvoltage_levers.
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

    # Protect promoted island grid-formers from MAS curtailment. A GridForming*
    # unit is its island's free-Var slack reference at reg=1; curtailing it toward
    # reduced post-failure demand leaves a reg<1 reference that can't anchor the
    # island → every step infeasible (verified: restoring formers to full re-
    # feasibilises the exact state, load-shed does not). Guard pins the former δ=0
    # in the MW gossip (dropped from dual) + backstops ``apply_regulate`` for
    # non-gossip writes; never actuated. Inert outside islanding.
    enable_grid_former_curtail_guard: bool = False

    # NOTE (islanding gas slack): under the former guard the gas slack settles ~61%
    # of B with gas still shed — NOT reclaimable. Both prototyped fixes REFUTED
    # (recoverable_islanding, simbench_lv, seed 100000023; baseline PWSF 0.42/PASS):
    # anti-windup floor 0.5·B -> gas tier-1 0.31->1.00, PWSF 0.60 but slack 201% B;
    # damped restore 0.3 -> PWSF 0.39 (worse) + priority inversion (tier-4 0.22 >
    # tier-1 0.09), slack 126%. Formers are free-Var with no MAS dispatch lever so
    # serving gas draws the slack >1x — wind-down is correct enforcement; closing
    # the oracle gap needs former/P2G dispatch, not a slack lever.

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
    # R1 estimator lever: credit non-slack generators their AVAILABLE capacity
    # (rated |cap|, reduced by a fresh over-voltage curtail-lock — delivered
    # only — and by the line-congestion ceiling) to the L2 supply pool instead
    # of delivered |sp|. Delivered-credit makes the pool a function of its own
    # previous output (self-limiting ratchet); capacity-credit lets the
    # supply-priority allocation ask generators to ramp. False = legacy.
    enable_gen_capacity_supply: bool = False

    # R5 last-sink guard: refuse to shed the ONLY HeatLoad at a junction with
    # a co-located fixed HeatGenerator below the fraction that absorbs the
    # local injection (node-365 over-temperature class). CP outlet injection
    # excluded (variable; owned by the CP heat-outlet guard).
    # VALIDATED-NEGATIVE 2026-07-20 (task 19 cold_day A/B, fixed-monee
    # substrate): the monee CP-write fix already clears the node-365/272
    # violations in control, and enabling this guard forced draws at 25
    # junctions on a supply-starved grid — NEW under-temp violation (node 314
    # @310.0 K) and PWSF -0.006. Keep OFF; premise only matters where fixed
    # HeatGenerator injection dominates the heat balance.
    enable_heat_last_sink_guard: bool = False

    # R2 drift fix (task-17 latch): the L2 dispatch caps each load by local
    # feasibility at WRITE time; constraint release fires no event, so a load
    # capped to ~0 stays below the standing allocation until the next solve —
    # which the no-trigger gate may never run. Periodically LIFT such loads
    # back toward min(allocation, constraint_allowed). Restore-only: never
    # pulls a load down, so it cannot fight legitimate above-allocation L1
    # serving or deepen a shed; all interlocks apply via apply_regulate.
    enable_l2_allocation_reassert: bool = False
    l2_allocation_reassert_s: float = 2.0
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

    # Q(U) settling time (s), VDE-AR-N 4105 §5.7.2. The Q(U) characteristic is a
    # pure proportional law and the fleet closes the loop through one shared bus
    # voltage, so the aggregate loop gain is
    # ``|dV/dQ| · Σq_max / (V_HIGH − V_DEADBAND_HIGH)`` — with the FULL circle
    # (``enable_vvw_coordination``) that is 4.25 on LV-S, where 0.91 MVar of
    # inverter circle sits behind a 160 kVA substation, and the droop
    # limit-cycles bang-bang between the deadband edge and saturation for the
    # whole run. The first-order lag scales that gain by ``dt/tau``: at the
    # 0.5 s electricity poll, 3.0 gives an effective 0.71 (stable) and settles
    # in ~9 s, inside the 30 s horizon. The standard's own ~10 s settles slower
    # than the horizon. 0 disables the lag (legacy instantaneous droop).
    qv_droop_settling_tau_s: float = 3.0

    # Attack time constant (s) — the lag applied when |Q| is RISING (deeper
    # voltage support). None (default) = symmetric, reuse the settling time.
    # REFUTED as a default: the theory was that only the release half
    # destabilises, so a fast attack (0.0) would clear the opening pv_peak
    # over-voltage in one tick and recover the ~1pp of tier-3/4 electricity the
    # symmetric lag costs. Measured on 40 paired lv_small pv_peak seeds it does
    # NOT — served energy is no better (mean el_served 0.959 vs 0.974), runtime
    # is +2.5 s, and one seed still limit-cycles (tail ptp 0.023 vs 0.0005).
    # The served-energy gap is not a transient artifact: the bang-bang was
    # delivering ~0.147 MVar of AVERAGE absorption against the settled
    # equilibrium's 0.031, i.e. the defect was incidentally buying voltage
    # headroom. Recovering it needs a re-tuned Q(U) curve (deadband /
    # ``qv_droop_voltage_ref_pu``), not a re-shaped lag. Kept as a knob.
    qv_droop_attack_tau_s: float | None = None

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

    # Coordination overhaul: reactive notify-on-change cascade. True: (a) UPWARD an
    # L1 gossip notifies L2/L3 only when its converged setpoint moved; (b) DOWNWARD
    # an unchanged L2 alloc re-asserts the per-load priority floor without abandoning
    # the in-flight gossip (was the dominant abandonment); (c) rebalance_min_gap_s
    # throttle bypassed since change-detection self-terminates the cascade. Local
    # A/B (v3, 2026-06-29): abandon-rate −15.3pp, 12/12 invariants, served/
    # violations flat. False: legacy throttled re-broadcast + mid-flight preemption.
    enable_change_only_dispatch: bool = True

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

    # NOTE: CP-aware slack supply (the flag ``enable_cp_aware_slack_supply``,
    # which debited the SlackBudgetMonitor's measured over-draw from the slack's
    # credited holon supply) was removed. It debited a CROSS-sector converter
    # draw against the SAME-sector native-electricity pool, so it over-shed
    # native load without touching the converter draw; the real lever is the L3
    # CP converter curtailment via the CP priority ADMM (see
    # enable_cp_priority_admm / cp_admm_algorithm). See
    # project_slack_compliance_rootcause.

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
    #
    # INERT under the default ``holon_admm_scope="component"``: that path returns
    # a priority waterfall from ``supply_priority_admm`` (no ub-overrides + no CP
    # coupling + supply>0 short-circuits before any iteration), so no ADMM runs
    # and neither this nor ``holon_admm_abs_tol`` can bind. ``sweep_holon_admm``
    # measured exactly one distinct result across 7 arms for that reason. Sweep
    # them only alongside ``holon_admm_scope`` in ("holon", "sector").
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
    # INERT under the default ``holon_admm_scope="component"``: holons are only
    # chunked for the legacy holon/sector scopes, while component scope elects a
    # per-component coordinator over the holon_summary mesh. ``sweep_holon_size``
    # returned one distinct result across all 4 arms for that reason.
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


_DECLARED_DEFAULTS = RestorationConfiguration()


def cfg_value(cfg: object, name: str):
    """Read ``name`` off ``cfg``, falling back to this file's declared default.

    Call sites that spelled their own literal fallback drifted whenever a
    default changed here, so an unconfigured code path silently ran a different
    configuration than production. Resolving through the dataclass makes that
    impossible.
    """
    return getattr(cfg, name, getattr(_DECLARED_DEFAULTS, name))
