"""Tests for the id-normalization parity primitive."""

from __future__ import annotations

from experiment.eval.parity import normalize_ids

U1 = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
U2 = "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"


def test_uuids_replaced_by_first_appearance_placeholders():
    out = normalize_ids(f"open auction {U1} then {U2} then {U1} again")
    assert out == "open auction uuid-0 then uuid-1 then uuid-0 again"


def test_identical_structure_different_uuids_normalize_equal():
    a = f"holon {U1} serves {U2}"
    b = f"holon {'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'} serves {'11111111-2222-3333-4444-555555555555'}"
    assert normalize_ids(a) == normalize_ids(b)


def test_real_change_survives_normalization():
    # An extra token (a genuine behavior difference) must NOT normalize away.
    a = f"a {U1} b"
    b = f"a {U1} b {U2}"
    assert normalize_ids(a) != normalize_ids(b)


def test_non_uuid_text_untouched():
    # Deterministic negotiation nids (neg-0, neg-1) and numbers are not uuids.
    txt = "nid=neg-7 t=1.5 factor=0.42 aid=child-118"
    assert normalize_ids(txt) == txt
