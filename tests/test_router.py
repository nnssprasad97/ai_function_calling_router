"""Tests verifying router policies and contracts for req-3 through req-8."""

import pytest
from src.router import Router, process_request
from src.schemas import (
    FlightSearchOutput,
    WeatherLookupOutput,
)


@pytest.fixture
def router():
    return Router(backoff_duration=0.01)


def test_req_3_happy_path(router):
    """
    Verification contract for req-3:
    Execute with a sunny-day prompt (e.g. 'Book a flight from MIA to JFK on 2024-10-10').
    Verify attempts == 0 (no recovery logic triggered) and output is a successful standard JSON structure.
    """
    result = router.process_request(
        request="Book a flight from MIA to JFK on 2024-10-10.",
        use_recovery=True,
        injected_fault="NONE",
    )
    assert result["status"] == "success"
    assert result["attempts"] == 0
    assert result["recovery_policy"] is None
    assert result["tool"] == "FlightSearch"
    assert result["data"]["origin"] == "MIA"
    assert result["data"]["destination"] == "JFK"
    assert "flight_number" in result["data"]

    # Passes output schema validation
    validated = FlightSearchOutput(**result["data"])
    assert validated.status == "confirmed"


def test_req_4_recovery_missing_field(router):
    """
    Verification contract for req-4:
    Prompt intentionally missing a required field (e.g. missing date).
    Check internal trace/logs to confirm missing_field recovery policy was triggered exactly once.
    Verify final response is a success payload with the resolved argument.
    """
    result = router.process_request(
        request="Book a flight to Paris departing from London on 2024-12-01.",
        use_recovery=True,
        injected_fault="NONE",
    )
    # If initial intent extraction was missing or had missing field
    assert result["status"] == "success"
    # Even on edge case prompt missing date
    result_missing_date = router.process_request(
        request="Book a flight from Mumbai to Delhi.",
        use_recovery=True,
        injected_fault="NONE",
    )
    assert result_missing_date["status"] == "success"
    assert result_missing_date["attempts"] == 1
    assert result_missing_date["recovery_policy"] == "missing_field"
    assert "recovery_policy_triggered:missing_field" in result_missing_date["trace"]
    assert "recovery_passed:missing_field" in result_missing_date["trace"]
    assert "date" in result_missing_date["data"]


def test_req_5_recovery_type_error(router):
    """
    Verification contract for req-5:
    Prompt containing a known type violation (e.g. 'Book a flight from JFK to LHR tomorrow').
    Check trace to confirm type_error recovery policy was triggered.
    Verify successful execution after LLM correction.
    """
    result = router.process_request(
        request="Book a flight from JFK to LHR tomorrow.",
        use_recovery=True,
        injected_fault="NONE",
    )
    assert result["status"] == "success"
    assert result["attempts"] == 1
    assert result["recovery_policy"] == "type_error"
    assert "recovery_policy_triggered:type_error" in result["trace"]
    assert "recovery_passed:type_error" in result["trace"]
    assert result["data"]["origin"] == "JFK"
    # The date was translated into a valid ISO date
    FlightSearchOutput(**result["data"])


def test_req_6_recovery_timeout(router):
    """
    Verification contract for req-6:
    Injected fault: TIMEOUT.
    Monitor execution trace to confirm Python-native retry occurred without additional LLM API call.
    Verify successful completion on retry.
    """
    result = router.process_request(
        request="What is the weather in Tokyo in C?",
        use_recovery=True,
        injected_fault="TIMEOUT",
    )
    assert result["status"] == "success"
    assert result["attempts"] == 1
    assert result["recovery_policy"] == "timeout"
    assert "recovery_policy_triggered:timeout" in result["trace"]
    assert "recovery_passed:timeout" in result["trace"]
    # Verify zero additional LLM calls for timeout retry (only 1 for initial intent)
    assert result["llm_call_count"] == 1
    WeatherLookupOutput(**result["data"])


def test_req_7_recovery_malformed_response(router):
    """
    Verification contract for req-7:
    Injected fault: MALFORMED_RESPONSE.
    Verify schema_repair recovery policy is triggered.
    Verify final returned object passes output schema validation.
    """
    result = router.process_request(
        request="What is the weather in Singapore in C?",
        use_recovery=True,
        injected_fault="MALFORMED_RESPONSE",
    )
    assert result["status"] == "success"
    assert result["attempts"] == 1
    assert result["recovery_policy"] == "schema_repair"
    assert "recovery_policy_triggered:schema_repair" in result["trace"]
    assert "recovery_passed:schema_repair" in result["trace"]
    # Final payload conforms to WeatherLookupOutput
    validated = WeatherLookupOutput(**result["data"])
    assert validated.location == "Singapore"
    assert validated.unit == "C"


def test_req_8_explicit_failure_timeout(router):
    """
    Verification contract for req-8:
    Persistently failing test case (TIMEOUT_PERSISTENT).
    Verify response strictly matches {"status": "error", "reason": "timeout_unrecoverable"}.
    """
    result = router.process_request(
        request="Find flights from SFO to JFK on 2024-11-11.",
        use_recovery=True,
        injected_fault="TIMEOUT_PERSISTENT",
    )
    assert result["status"] == "error"
    assert result["reason"] == "timeout_unrecoverable"
    assert result["attempts"] == 1


def test_req_8_explicit_failure_schema_repair(router):
    """
    Verification contract for req-8:
    Persistently failing malformed response (MALFORMED_PERSISTENT).
    Verify response strictly matches {"status": "error", "reason": "schema_repair_unrecoverable"}.
    """
    result = router.process_request(
        request="What is the weather in Dublin in C?",
        use_recovery=True,
        injected_fault="MALFORMED_PERSISTENT",
    )
    assert result["status"] == "error"
    assert result["reason"] == "schema_repair_unrecoverable"
    assert result["attempts"] == 1


def test_req_8_explicit_failure_unresolvable_missing_field(router):
    """
    Verification contract for req-8:
    Unresolvable missing field where targeted recovery fails after 1 retry.
    Verify response strictly matches {"status": "error", "reason": "missing_field_unrecoverable"}.
    """
    result = router.process_request(
        request="Book a flight with unresolvable missing parameters.",
        use_recovery=True,
        injected_fault="NONE",
    )
    assert result["status"] == "error"
    assert result["reason"] == "missing_field_unrecoverable"
    assert result["attempts"] == 1


def test_baseline_mode_no_recovery(router):
    """Verify that baseline mode gives up immediately on any fault with 0 attempts."""
    # Timeout in baseline gives up immediately with 0 attempts
    result = router.process_request(
        request="What is the weather in Tokyo in C?",
        use_recovery=False,
        injected_fault="TIMEOUT",
    )
    assert result["status"] == "error"
    assert result["attempts"] == 0
