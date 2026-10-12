"""AI Function-Calling Router with Pydantic and Injected-Failure Recovery."""

from src.fault_injector import FaultContext, fault_scope, inject_fault
from src.router import Router, process_request
from src.schemas import (
    CalendarBooking,
    CalendarBookingInput,
    CalendarBookingOutput,
    FlightSearch,
    FlightSearchInput,
    FlightSearchOutput,
    UnitConversion,
    UnitConversionInput,
    UnitConversionOutput,
    WeatherLookup,
    WeatherLookupInput,
    WeatherLookupOutput,
)
from src.tools import (
    calendar_booking,
    flight_search,
    unit_conversion,
    weather_lookup,
)

__all__ = [
    "Router",
    "process_request",
    "FaultContext",
    "fault_scope",
    "inject_fault",
    "FlightSearch",
    "FlightSearchInput",
    "FlightSearchOutput",
    "CalendarBooking",
    "CalendarBookingInput",
    "CalendarBookingOutput",
    "WeatherLookup",
    "WeatherLookupInput",
    "WeatherLookupOutput",
    "UnitConversion",
    "UnitConversionInput",
    "UnitConversionOutput",
    "flight_search",
    "calendar_booking",
    "weather_lookup",
    "unit_conversion",
]
