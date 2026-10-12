"""Pydantic schemas for AI function-calling router tools and outputs."""

import datetime as dt
from enum import Enum
from typing import Dict, Type
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TemperatureUnit(str, Enum):
    """Supported temperature units."""
    C = "C"
    F = "F"


# ==============================================================================
# Tool Input Schemas (Validation Boundary)
# ==============================================================================

class FlightSearchInput(BaseModel):
    """Schema for flight search queries."""
    model_config = ConfigDict(extra="forbid")

    origin: str = Field(..., min_length=1, description="Departure airport or city")
    destination: str = Field(..., min_length=1, description="Destination airport or city")
    date: dt.date = Field(..., description="Flight date in ISO-8601 format (YYYY-MM-DD)")


class CalendarBookingInput(BaseModel):
    """Schema for booking calendar events."""
    model_config = ConfigDict(extra="forbid")

    event_title: str = Field(..., min_length=1, description="Event or meeting title")
    start_time: dt.datetime = Field(..., description="Event start timestamp in ISO-8601 format (YYYY-MM-DDTHH:MM:SS)")
    duration_minutes: int = Field(..., gt=0, description="Meeting duration in minutes")


class WeatherLookupInput(BaseModel):
    """Schema for weather lookups."""
    model_config = ConfigDict(extra="forbid")

    location: str = Field(..., min_length=1, description="City or geographic location")
    unit: TemperatureUnit = Field(..., description="Temperature unit: C or F")


class UnitConversionInput(BaseModel):
    """Schema for unit conversions."""
    model_config = ConfigDict(extra="forbid")

    value: float = Field(..., description="Numeric value to convert")
    from_unit: str = Field(..., min_length=1, description="Source measurement unit")
    to_unit: str = Field(..., min_length=1, description="Target measurement unit")


# Aliases for direct class lookup / parsing compatibility
FlightSearch = FlightSearchInput
CalendarBooking = CalendarBookingInput
WeatherLookup = WeatherLookupInput
UnitConversion = UnitConversionInput


# ==============================================================================
# Tool Output Schemas (Response Schema Repair Boundary)
# ==============================================================================

class FlightSearchOutput(BaseModel):
    """Expected schema for flight search tool response."""
    flight_number: str = Field(..., min_length=1)
    origin: str = Field(..., min_length=1)
    destination: str = Field(..., min_length=1)
    date: str = Field(..., min_length=1)
    price_usd: float = Field(..., ge=0)
    status: str = Field(..., min_length=1)


class CalendarBookingOutput(BaseModel):
    """Expected schema for calendar booking tool response."""
    booking_id: str = Field(..., min_length=1)
    event_title: str = Field(..., min_length=1)
    start_time: str = Field(..., min_length=1)
    end_time: str = Field(..., min_length=1)
    duration_minutes: int = Field(..., gt=0)
    status: str = Field(..., min_length=1)


class WeatherLookupOutput(BaseModel):
    """Expected schema for weather lookup tool response."""
    location: str = Field(..., min_length=1)
    temperature: float = Field(...)
    unit: str = Field(..., min_length=1)
    condition: str = Field(..., min_length=1)


class UnitConversionOutput(BaseModel):
    """Expected schema for unit conversion tool response."""
    original_value: float = Field(...)
    from_unit: str = Field(..., min_length=1)
    converted_value: float = Field(...)
    to_unit: str = Field(..., min_length=1)


# Schema mappings for router lookup
TOOL_INPUT_SCHEMAS: Dict[str, Type[BaseModel]] = {
    "FlightSearch": FlightSearchInput,
    "CalendarBooking": CalendarBookingInput,
    "WeatherLookup": WeatherLookupInput,
    "UnitConversion": UnitConversionInput,
}

TOOL_OUTPUT_SCHEMAS: Dict[str, Type[BaseModel]] = {
    "FlightSearch": FlightSearchOutput,
    "CalendarBooking": CalendarBookingOutput,
    "WeatherLookup": WeatherLookupOutput,
    "UnitConversion": UnitConversionOutput,
}
