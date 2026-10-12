"""Tests verifying req-1-schemas: 4 Pydantic tool schemas for argument validation."""

import datetime as dt
import pytest
from pydantic import BaseModel, ValidationError

from src.schemas import (
    CalendarBooking,
    CalendarBookingInput,
    CalendarBookingOutput,
    FlightSearch,
    FlightSearchInput,
    FlightSearchOutput,
    TemperatureUnit,
    UnitConversion,
    UnitConversionInput,
    UnitConversionOutput,
    WeatherLookup,
    WeatherLookupInput,
    WeatherLookupOutput,
)


def test_schema_classes_inherit_from_pydantic_base_model():
    """Verify that all four required domain schemas subclass pydantic.BaseModel."""
    assert issubclass(FlightSearch, BaseModel)
    assert issubclass(CalendarBooking, BaseModel)
    assert issubclass(WeatherLookup, BaseModel)
    assert issubclass(UnitConversion, BaseModel)

    assert issubclass(FlightSearchInput, BaseModel)
    assert issubclass(CalendarBookingInput, BaseModel)
    assert issubclass(WeatherLookupInput, BaseModel)
    assert issubclass(UnitConversionInput, BaseModel)

    assert issubclass(FlightSearchOutput, BaseModel)
    assert issubclass(CalendarBookingOutput, BaseModel)
    assert issubclass(WeatherLookupOutput, BaseModel)
    assert issubclass(UnitConversionOutput, BaseModel)


def test_flight_search_schema_validation():
    """Verify FlightSearch input validation: requires origin, destination, date."""
    # Valid instance
    valid = FlightSearch(
        origin="JFK",
        destination="LHR",
        date=dt.date(2024, 11, 15),
    )
    assert valid.origin == "JFK"
    assert valid.destination == "LHR"
    assert valid.date == dt.date(2024, 11, 15)

    # Valid string date parsed by Pydantic
    valid_str = FlightSearch(origin="JFK", destination="LHR", date="2024-11-15")
    assert valid_str.date == dt.date(2024, 11, 15)

    # Missing required field raises ValidationError
    with pytest.raises(ValidationError) as exc_info:
        FlightSearch(destination="LHR", date="2024-11-15")
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("origin",) and e["type"] == "missing" for e in errors)

    # Invalid date format (natural language) raises ValidationError
    with pytest.raises(ValidationError) as exc_info:
        FlightSearch(origin="JFK", destination="LHR", date="tomorrow")
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("date",) for e in errors)


def test_calendar_booking_schema_validation():
    """Verify CalendarBooking input validation: requires event_title, start_time, duration_minutes."""
    # Valid instance
    valid = CalendarBooking(
        event_title="Sprint Planning",
        start_time=dt.datetime(2024, 10, 15, 10, 0, 0),
        duration_minutes=45,
    )
    assert valid.event_title == "Sprint Planning"
    assert valid.duration_minutes == 45

    # Missing duration raises ValidationError
    with pytest.raises(ValidationError) as exc_info:
        CalendarBooking(
            event_title="Sync",
            start_time="2024-10-15T10:00:00",
        )
    assert any(e["loc"] == ("duration_minutes",) and e["type"] == "missing" for e in exc_info.value.errors())

    # Invalid duration type or constraint (e.g. <= 0) raises ValidationError
    with pytest.raises(ValidationError):
        CalendarBooking(
            event_title="Sync",
            start_time="2024-10-15T10:00:00",
            duration_minutes=0,
        )


def test_weather_lookup_schema_validation():
    """Verify WeatherLookup input validation: requires location, unit (Enum C, F)."""
    # Valid instance
    valid_c = WeatherLookup(location="Tokyo", unit=TemperatureUnit.C)
    assert valid_c.location == "Tokyo"
    assert valid_c.unit == TemperatureUnit.C

    valid_f = WeatherLookup(location="New York", unit="F")
    assert valid_f.unit == TemperatureUnit.F

    # Missing location
    with pytest.raises(ValidationError) as exc_info:
        WeatherLookup(unit="C")
    assert any(e["loc"] == ("location",) and e["type"] == "missing" for e in exc_info.value.errors())

    # Invalid unit value
    with pytest.raises(ValidationError) as exc_info:
        WeatherLookup(location="Tokyo", unit="Kelvin")
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("unit",) for e in errors)


def test_unit_conversion_schema_validation():
    """Verify UnitConversion input validation: requires value, from_unit, to_unit."""
    # Valid instance
    valid = UnitConversion(value=100.5, from_unit="km", to_unit="miles")
    assert valid.value == 100.5
    assert valid.from_unit == "km"
    assert valid.to_unit == "miles"

    # Missing value
    with pytest.raises(ValidationError) as exc_info:
        UnitConversion(from_unit="km", to_unit="miles")
    assert any(e["loc"] == ("value",) and e["type"] == "missing" for e in exc_info.value.errors())

    # Wrong type for value (non-numeric string)
    with pytest.raises(ValidationError) as exc_info:
        UnitConversion(value="one-hundred", from_unit="km", to_unit="miles")
    assert any(e["loc"] == ("value",) for e in exc_info.value.errors())


def test_output_schemas_validation():
    """Verify output schemas enforce contract shapes."""
    flight_out = FlightSearchOutput(
        flight_number="AI-101",
        origin="JFK",
        destination="LHR",
        date="2024-11-15",
        price_usd=249.5,
        status="confirmed",
    )
    assert flight_out.price_usd == 249.5

    # Missing flight_number raises ValidationError
    with pytest.raises(ValidationError):
        FlightSearchOutput(
            origin="JFK",
            destination="LHR",
            date="2024-11-15",
            price_usd=249.5,
            status="confirmed",
        )
