"""Tuning constants for grid constraint monitoring and enforcement.

Collected here so the monitor and its helper controllers (state propagation,
congestion relief, the curtailment auction) share one definition of each knob.
"""

from __future__ import annotations

from scare.base.model import Sector

# Hops constraint state propagates.
_DEFAULT_MAX_HOPS = 3

# Min utilization change that triggers a fresh broadcast.
_FORWARD_VALUE_TOL: float = 0.02

# Min sim-time between re-broadcasts of an unchanged value (keeps trust
# liveness ticking without per-cycle flooding).
_FORWARD_FRESHNESS_S: float = 5.0

# Cache-gate tolerance for ``_monitor``; tighter than ``_FORWARD_VALUE_TOL``.
_VALUES_DELTA_TOL: float = 1e-4

# EMA smoothing for the local sensitivity estimate (dV/dP).
_SENSITIVITY_EMA_ALPHA: float = 0.2

# Per-sector min |ΔP| before a sample is used; below this ΔV is noise.
_SENSITIVITY_MIN_DP: dict[Sector, float] = {
    Sector.ELECTRICITY: 0.01,  # MW
    Sector.GAS: 1e-4,  # kg/s
    Sector.HEAT: 5e-4,  # MW (0.5 kW; registers ~30% regulation steps)
}

# Default sensitivity before any samples collected.
_SENSITIVITY_DEFAULT: dict[Sector, float] = {
    Sector.ELECTRICITY: 0.01,  # p.u. voltage per MW
    Sector.GAS: 0.5,  # p.u. pressure per kg/s
    Sector.HEAT: 10.0,  # K per MW (samples are dT/dP_MW; ≡ 1e-5 K/W)
}

# Measured replacement for the HEAT seed, reachable via
# ``SimulationConfig.heat_sensitivity_seed_k_per_mw``. NOT the default until the
# A/B lands: flipping it moves every heat load in every run.
#
# 10.0 makes ``HeatFrontierController`` structurally bang-bang — ``sensitivity *
# cap`` is 0.075 K per unit regulation for a 7.5 kW load against ~27 K measured,
# so on eval_full_v2_20260728-202054 99.4% of all 67 719 frontier moves saturated
# at ±MAX_STEP and the proportional term never engaged. 3500.0 is the median
# |dT_k/dP| over that campaign's 53 423 consecutive frontier poll pairs: 3022
# K/MW under SCARE, 3556 under both baselines (25-75%: 1067-6044 / 1600-6000).
# Three variants agreeing says this is a property of the DHS, not of the
# controller. The seed also gates the EMA, whose ``min(sample, 10*value+1)``
# clamp needs ~6 accepted samples to climb from 10 to 3000 — more than a 30 s
# run affords.
_SENSITIVITY_HEAT_MEASURED: float = 3500.0

# Bounds on the auction willingness sensitivity multiplier; a within-tier
# tiebreaker kept far below the tier step so priority stays lexicographic.
_SENS_MULT_MIN: float = 0.25
_SENS_MULT_MAX: float = 4.0

# Primary constraint variable per sector for sensitivity tracking.
_SECTOR_PRIMARY_VAR: dict[Sector, str] = {
    Sector.ELECTRICITY: "vm_pu",
    Sector.GAS: "pressure_pu",
    Sector.HEAT: "t_k",
}

# Sentinel bidder key for the auctioneer's OWN load (its setpoint is the
# most direct lever on its junction); distinct from any ``str(addr)`` key.
_SELF_BID_KEY: str = "__self__"

# Auction gating (``enable_curtail_auction_gating``): no-progress rounds
# before the gate suspends re-arming, and the overshoot improvement that
# counts as progress.
_CURTAIL_NO_PROGRESS_LIMIT: int = 2
_CURTAIL_PROGRESS_TOL: float = 0.01

# Coordinated hand-off (``enable_qv_auction_coordination``): defer to the
# reactive lever only while voltage is measurably dropping. ``_..._TOL`` is the
# min p.u. drop per poll counting as progress; ``_..._DEFERS`` is a backstop
# bounding pathological oscillation, not the primary stop.
_QV_MAX_CONSECUTIVE_DEFERS: int = 6
_QV_DEFER_PROGRESS_TOL: float = 1e-3

# The auction never fires on ``loading_percent`` (node-blind bidding can't
# relieve a branch; the line-relief path owns it, re-enabling it only with a
# targeted bidder set) and skips ``t_k`` only while the heat frontier owns it
# (see ``_auction_skips_var``). It still fires on ``vm_pu`` / ``pressure_pu``
# where local load is the lever.

# Targeting (``enable_curtail_auction_targeting``): proximity to the violated
# origin scales willingness within these bounds (within-tier tiebreaker, from
# cached multi-hop distance). Auctioneer is the origin, self-bids at PROX_MAX.
_CURTAIL_PROX_MIN: float = 0.25
_CURTAIL_PROX_MAX: float = 4.0

# Min sim-seconds between line-relief re-assertions per branch
# (``enable_line_relief_reassert``); never out-paces the gossip round it triggers.
_LINE_RELIEF_COOLDOWN_S: float = 2.0

# Consecutive polls classifying an overload as export (reverse flow) before
# load-shed relief is suppressed and generators are curtailed; a single
# transient reverse-flow sample must not trigger a non-reverting curtail.
_EXPORT_DEBOUNCE_POLLS: int = 2

# Sim-seconds a resolved downstream topology stays valid. No topology event
# (branch failure, tie close) reaches branch monitors, so re-resolve on a TTL
# when consulted.
_DOWNSTREAM_TOPOLOGY_TTL_S: float = 10.0

# Aggressive per-round gain for branch-downstream line relief (vs 0.3 default),
# walking a 10-20% overload down to ≤100% over rounds; priority orders WHO.
_LINE_RELIEF_GAIN: float = 1.5

# Reducible-draw threshold (MW) below which a downstream bidder is exhausted,
# escalating the waterfall to the next tier.
_LINE_RELIEF_MIN_REDUCIBLE: float = 5e-4

# Schmitt-trigger release margin (loading-% points): hold the L2-clawback lock
# until the line drops this far below bound, avoiding a relief↔L2 limit-cycle.
# Released only on genuine headroom, not the relief's own settle point.
_LINE_RELIEF_RELEASE_MARGIN: float = 15.0

# Congestion-price AIMD (``enable_line_congestion_price``): GAIN integrates the
# price up on normalized loading overshoot ((val-hi)/100); RESTORE_STEP decays it
# each headroom poll so PV ramps back. HEADROOM_MARGIN gates the decay (hold the
# last ceiling inside the band so a stalled monitor can't release curtailment and
# re-overload). PRICE_MAX < 1 keeps a downstream gen off a hard 0.
_LINE_CONGESTION_GAIN: float = 0.35
_LINE_CONGESTION_RESTORE_STEP: float = 0.05
_LINE_CONGESTION_HEADROOM_MARGIN: float = 8.0
_LINE_CONGESTION_PRICE_MAX: float = 0.95
# Freshness (sim-s) of a published congestion price; matches the line-curtail
# lock TTL so a monitor that stops publishing releases the ceiling on the same
# horizon the old hard lock aged out.
_LINE_CONGESTION_TTL_S: float = 3.0

# Heat frontier feedback period (s), faster than the heat SCADA poll so a
# rate-limited deeply-cold node converges within the run. See HeatFrontierController.
_HEAT_FRONTIER_PERIOD_S: float = 1.0
