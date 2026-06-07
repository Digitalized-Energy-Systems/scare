"""Unit tests for balance.py helper functions (pure, no mango deps)."""

from scare.service.balance.balance import (
    _PRIORITY_TIERS,
    _compute_actual_priority,
    _deterministic_next,
    _deterministic_sub_round,
)

# ===================================================================
# _deterministic_next
# ===================================================================


class TestDeterministicNext:
    def test_single_neighbour(self):
        assert _deterministic_next(["addr1"], "neg-1", 0) == "addr1"

    def test_empty_returns_none(self):
        assert _deterministic_next([], "neg-1", 0) is None

    def test_stable(self):
        r1 = _deterministic_next(["a", "b", "c"], "neg-1", 5)
        r2 = _deterministic_next(["a", "b", "c"], "neg-1", 5)
        assert r1 == r2

    def test_varies_with_counter(self):
        results = {
            _deterministic_next(["a", "b", "c", "d"], "neg-1", i) for i in range(20)
        }
        assert len(results) > 1

    def test_result_in_list(self):
        neighbours = ["x", "y", "z"]
        result = _deterministic_next(neighbours, "test", 42)
        assert result in neighbours


# ===================================================================
# _compute_actual_priority
# ===================================================================


class TestComputeActualPriority:
    # --- Restoration (target > 0) ---

    def test_restoration_load_priority_1(self):
        assert _compute_actual_priority(1, target=1.0) == 1

    def test_restoration_load_priority_3(self):
        # Restoration: priority maps to round number (lower = earlier).
        assert _compute_actual_priority(3, target=1.0) == 3

    def test_restoration_load_capped_at_tiers(self):
        assert _compute_actual_priority(100, target=1.0) == _PRIORITY_TIERS

    def test_restoration_generator_last(self):
        assert _compute_actual_priority(0, target=1.0) == _PRIORITY_TIERS + 1

    def test_restoration_ordering(self):
        # Lower priority number = earlier round
        p1 = _compute_actual_priority(1, target=1.0)
        p3 = _compute_actual_priority(3, target=1.0)
        assert p1 < p3

    # --- Reduction (target < 0) ---

    def test_reduction_generator_first(self):
        assert _compute_actual_priority(0, target=-1.0) == 0

    def test_reduction_high_priority_load_shed_last(self):
        p1 = _compute_actual_priority(1, target=-1.0)
        p4 = _compute_actual_priority(4, target=-1.0)
        assert p1 > p4  # more important → later round → shed last

    # --- Zero target ---

    def test_zero_target(self):
        assert _compute_actual_priority(3, target=0.0) == 3


# ===================================================================
# _deterministic_sub_round
# ===================================================================


class TestDeterministicSubRound:
    def test_deterministic(self):
        r1 = _deterministic_sub_round("addr1", "neg-1", 2, 5)
        r2 = _deterministic_sub_round("addr1", "neg-1", 2, 5)
        assert r1 == r2

    def test_in_range(self):
        for tier_size in [2, 5, 10, 100]:
            r = _deterministic_sub_round("addr1", "neg-1", 1, tier_size)
            assert 0 <= r < tier_size

    def test_varies_with_address(self):
        results = {
            _deterministic_sub_round(f"addr-{i}", "neg-1", 1, 10) for i in range(50)
        }
        assert len(results) > 1

    def test_tier_size_one(self):
        assert _deterministic_sub_round("addr1", "neg-1", 1, 1) == 0
