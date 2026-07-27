"""Scenario construction for the restoration experiments.

Grid builders + the named-grid registry (:data:`grids.GRIDS`), in-place
``apply_*`` stress modifiers, and per-load priority assignment. Split out of
the former ``experiment/restoration.py``; the demo CLI now lives in
``experiment.scenarios.__main__`` (``python -m experiment.scenarios``).
"""

from experiment.scenarios.grids import (
    GRIDS,
    add_backup_lines,
    create_large_lv_simbench,
)
from experiment.scenarios.modifiers import (
    apply_cold_day,
    apply_heat_node_regulariser,
    apply_line_stress,
    apply_microgrid_islanding,
    apply_pv_peak,
    apply_slack_budget,
    apply_temporal_extensions,
)
from experiment.scenarios.priorities import assign_load_priorities

__all__ = [
    "GRIDS",
    "add_backup_lines",
    "create_large_lv_simbench",
    "apply_cold_day",
    "apply_heat_node_regulariser",
    "apply_line_stress",
    "apply_microgrid_islanding",
    "apply_pv_peak",
    "apply_slack_budget",
    "apply_temporal_extensions",
    "assign_load_priorities",
]
