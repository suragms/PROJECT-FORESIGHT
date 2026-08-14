"""Validation helper tests."""

from __future__ import annotations

import pytest

from src.forecasting.validation import InputValidationError, records_to_frame, validate_dataset_horizon


def test_validate_horizon_ok():
    validate_dataset_horizon("UCI", 1)
    validate_dataset_horizon("SYNTHETIC", 30)


def test_validate_horizon_bad():
    with pytest.raises(InputValidationError):
        validate_dataset_horizon("UCI", 2)
    with pytest.raises(InputValidationError):
        validate_dataset_horizon("NOPE", 1)
    with pytest.raises(InputValidationError):
        validate_dataset_horizon("UCI", "x")


def test_missing_keys():
    with pytest.raises(InputValidationError):
        records_to_frame([{"source_dataset": "UCI"}])


def test_missing_date():
    with pytest.raises(InputValidationError):
        records_to_frame([{
            "source_dataset": "UCI", "entity_id": "e", "product_key": "p",
        }])


def test_empty_records():
    with pytest.raises(InputValidationError):
        records_to_frame([])
