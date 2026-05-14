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


def default_config() -> RestorationConfiguration:
    """Return a fresh default config.  Equivalent to
    ``RestorationConfiguration()``; the named function makes intent
    explicit at call sites that want to be obviously baseline.
    """
    return RestorationConfiguration()
