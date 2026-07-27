import pytest

from experiment.eval import guards
from experiment.eval.guards import (
    ProtectedCampaignError,
    assert_regradable,
    is_protected,
)


def test_unacknowledged_call_is_refused_even_on_a_safe_path(tmp_path):
    with pytest.raises(ProtectedCampaignError, match="i-have-a-copy"):
        assert_regradable(tmp_path, acknowledged=False)


def test_acknowledged_local_directory_is_allowed(tmp_path):
    assert assert_regradable(tmp_path, acknowledged=True) == tmp_path.resolve()


def test_missing_directory_is_refused(tmp_path):
    with pytest.raises(ProtectedCampaignError, match="Not a directory"):
        assert_regradable(tmp_path / "nope", acknowledged=True)


def test_unc_paths_are_protected():
    assert is_protected(r"\\hpc-share\runs\eval_full_v2")


def test_configured_protected_root_and_its_children(monkeypatch, tmp_path):
    root = tmp_path / "runs_of_record"
    (root / "eval" / "campaign").mkdir(parents=True)
    monkeypatch.setenv("SCARE_PROTECTED_ROOTS", str(root))
    assert is_protected(root)
    assert is_protected(root / "eval" / "campaign")
    assert not is_protected(tmp_path / "elsewhere")


def test_protected_root_is_refused_even_when_acknowledged(monkeypatch, tmp_path):
    root = tmp_path / "runs_of_record"
    target = root / "eval" / "campaign"
    target.mkdir(parents=True)
    monkeypatch.setenv("SCARE_PROTECTED_ROOTS", str(root))
    with pytest.raises(ProtectedCampaignError, match="run of record"):
        assert_regradable(target, acknowledged=True)


def test_read_only_inspects_a_protected_root_without_acknowledgement(
    monkeypatch, tmp_path
):
    """A dry run writes nothing; refusing it just moves the inspection into an
    unguarded ad-hoc script."""
    root = tmp_path / "runs_of_record"
    target = root / "eval" / "campaign"
    target.mkdir(parents=True)
    monkeypatch.setenv("SCARE_PROTECTED_ROOTS", str(root))
    assert (
        assert_regradable(target, acknowledged=False, read_only=True)
        == target.resolve()
    )


def test_read_only_still_refuses_a_missing_directory(tmp_path):
    with pytest.raises(ProtectedCampaignError, match="Not a directory"):
        assert_regradable(tmp_path / "nope", acknowledged=False, read_only=True)


def test_sibling_prefix_is_not_treated_as_a_child(monkeypatch, tmp_path):
    """``_runs_backup`` must not match the ``_runs`` root by string prefix."""
    root = tmp_path / "_runs"
    root.mkdir()
    sibling = tmp_path / "_runs_backup"
    sibling.mkdir()
    monkeypatch.setenv("SCARE_PROTECTED_ROOTS", str(root))
    assert is_protected(root)
    assert not is_protected(sibling)


def test_shipped_default_root_covers_the_campaign_of_record():
    assert any("_runs" in str(r) for r in guards._protected_roots())
    assert is_protected(
        "Y:/fs/dss/home/towo7024/SCARE/scare/experiment/_runs/eval/"
        "eval_full_v2_20260724-141520"
    )
