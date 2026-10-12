"""Mock tool implementations decorated with fault injection middleware."""

from datetime import date, datetime, timedelta
from typing import Any, Dict, Union
from src.fault_injector import inject_fault
from src.schemas import (
    CalendarBookingInput,
    CalendarBookingOutput,
    FlightSearchInput,
    FlightSearchOutput,
    TemperatureUnit,
    UnitConversionInput,
    UnitConversionOutput,
    WeatherLookupInput,
    WeatherLookupOutput,
)


def _convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Helper unit converter for common conversions."""
    fu = from_unit.strip().lower()
    tu = to_unit.strip().lower()

    if fu == tu:
        return float(value)

    # Length
    if fu in ("km", "kilometer", "kilometers") and tu in ("miles", "mile", "mi"):
        return value * 0.621371
    if fu in ("miles", "mile", "mi") and tu in ("km", "kilometer", "kilometers"):
        return value / 0.621371
    if fu in ("meters", "meter", "m") and tu in ("feet", "foot", "ft"):
        return value * 3.28084
    if fu in ("feet", "foot", "ft") and tu in ("meters", "meter", "m"):
        return value / 3.28084

    # Weight
    if fu in ("kg", "kilogram", "kilograms") and tu in ("lbs", "pounds", "pound", "lb"):
        return value * 2.20462
    if fu in ("lbs", "pounds", "pound", "lb") and tu in ("kg", "kilogram", "kilograms"):
        return value / 2.20462

    # Temperature
    if fu in ("c", "celsius") and tu in ("f", "fahrenheit"):
        return (value * 9 / 5) + 32
    if fu in ("f", "fahrenheit") and tu in ("c", "celsius"):
        return (value - 32) * 5 / 9

    # Default fallback conversion factor
    return value * 1.0


@inject_fault(tool_name="FlightSearch")
def flight_search(origin: str, destination: str, date: Union[date, str]) -> Dict[str, Any]:
    """Mock search for flights between two locations."""
    date_str = date.isoformat() if hasattr(date, "isoformat") else str(date)
    flight_hash = abs(hash(f"{origin}->{destination}")) % 800 + 100
    
    return {
        "flight_number": f"AI-{flight_hash}",
        "origin": origin.strip().upper(),
        "destination": destination.strip().upper(),
        "date": date_str,
        "price_usd": 249.50,
        "status": "confirmed",
    }


@inject_fault(tool_name="CalendarBooking")
def calendar_booking(
    event_title: str,
    start_time: Union[datetime, str],
    duration_minutes: int,
) -> Dict[str, Any]:
    """Mock calendar event booking."""
    if isinstance(start_time, datetime):
        dt = start_time
    else:
        dt = datetime.fromisoformat(str(start_time))
    
    end_dt = dt + timedelta(minutes=int(duration_minutes))
    booking_id = f"cal-{abs(hash(event_title + str(dt))) % 9000 + 1000}"

    return {
        "booking_id": booking_id,
        "event_title": event_title.strip(),
        "start_time": dt.isoformat(),
        "end_time": end_dt.isoformat(),
        "duration_minutes": int(duration_minutes),
        "status": "confirmed",
    }


@inject_fault(tool_name="WeatherLookup")
def weather_lookup(location: str, unit: Union[TemperatureUnit, str]) -> Dict[str, Any]:
    """Mock current weather lookup."""
    unit_str = unit.value if hasattr(unit, "value") else str(unit)
    norm_unit = unit_str.upper()
    temperature = 22.0 if norm_unit == "C" else 71.6

    return {
        "location": location.strip(),
        "temperature": temperature,
        "unit": norm_unit,
        "condition": "Partly Cloudy",
    }


@inject_fault(tool_name="UnitConversion")
def unit_conversion(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
    """Mock unit conversion service."""
    converted = _convert_units(float(value), from_unit, to_unit)

    return {
        "original_value": float(value),
        "from_unit": from_unit.strip(),
        "converted_value": round(converted, 2),
        "to_unit": to_unit.strip(),
    }


# Tool Registry
TOOL_REGISTRY: Dict[str, Any] = {
    "FlightSearch": flight_search,
    "CalendarBooking": calendar_booking,
    "WeatherLookup": weather_lookup,
    "UnitConversion": unit_conversion,
}

# Aliases
mock_flight_search = flight_search
mock_calendar_booking = calendar_booking
mock_weather_lookup = weather_lookup
mock_unit_conversion = unit_conversion
