"""Tests verifying req-2-fault-injector: fault injection middleware bypassing LLM."""

import pytest
from pydantic import ValidationError

from src.fault_injector import FaultContext, fault_scope
from src.schemas import (
    FlightSearchOutput,
    WeatherLookupOutput,
)
from src.tools import (
    flight_search,
    weather_lookup,
)


def test_fault_injector_none_directive():
    """Verify NONE directive executes normally without faults."""
    with fault_scope("NONE"):
        result = weather_lookup(location="Tokyo", unit="C")
        assert result["location"] == "Tokyo"
        assert result["unit"] == "C"
        # Conforms to expected output schema
        validated = WeatherLookupOutput(**result)
        assert validated.location == "Tokyo"


def test_fault_injector_timeout_directive():
    """
    Verification contract for req-2:
    Call mock tool with fault configuration set to TIMEOUT and verify TimeoutError is raised.
    """
    with fault_scope("TIMEOUT"):
        with pytest.raises(TimeoutError) as exc_info:
            flight_search(origin="JFK", destination="LHR", date="2024-11-15")
        assert "timed out" in str(exc_info.value).lower()


def test_fault_injector_malformed_response_directive():
    """
    Verification contract for req-2:
    Call mock tool with fault configuration set to MALFORMED_RESPONSE and verify
    output schema is violated (raises ValidationError when parsed against output model).
    """
    with fault_scope("MALFORMED_RESPONSE"):
        result = weather_lookup(location="Tokyo", unit="C")
        # Result should be mutated and missing standard fields
        assert isinstance(result, dict)
        with pytest.raises(ValidationError) as exc_info:
            WeatherLookupOutput(**result)
        # Verify schema violation occurred
        assert exc_info.value.error_count() > 0


def test_fault_injector_persistent_timeout():
    """Verify persistent timeout fails continuously."""
    with fault_scope("TIMEOUT_PERSISTENT"):
        with pytest.raises(TimeoutError):
            flight_search(origin="JFK", destination="LHR", date="2024-11-15")
        with pytest.raises(TimeoutError):
            flight_search(origin="JFK", destination="LHR", date="2024-11-15")
