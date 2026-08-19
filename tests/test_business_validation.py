"""Business validation tests. Uses existing artifacts only."""

from src.production.business_validation import validate_inventory_risk, validate_ten_questions


def test_inventory_risk_matrix_present():
    result = validate_inventory_risk()
    assert result["exists"] is True
    assert result["status"] == "PASS"
    assert result["n_rows"] > 0
    assert not result["missing_columns"]


def test_ten_questions_have_evidence():
    payload = validate_ten_questions()
    assert payload["questions_total"] == 10
    assert payload["questions_with_evidence"] == 10
    assert payload["automated_replenishment_implemented"] is False
    assert payload["automatic_retraining_enabled"] is False
