"""Fault injection middleware for testing LLM function-calling failure recovery."""

import functools
import time
from contextvars import ContextVar
from typing import Any, Callable, Optional


class FaultContext:
    """Manages active fault injection state per context and thread."""
    # Class-level fallback for global access
    current_fault: str = "NONE"

    _context_var: ContextVar[str] = ContextVar("current_fault", default="NONE")
    _trigger_count_var: ContextVar[int] = ContextVar("trigger_count", default=0)

    @classmethod
    def get_fault(cls) -> str:
        """Get the currently active fault directive."""
        val = cls._context_var.get()
        return val if val != "NONE" else cls.current_fault

    @classmethod
    def set_fault(cls, fault: str) -> None:
        """Set the active fault directive and reset retry counters."""
        norm_fault = fault.strip().upper() if fault else "NONE"
        cls.current_fault = norm_fault
        cls._context_var.set(norm_fault)
        cls._trigger_count_var.set(0)

    @classmethod
    def get_trigger_count(cls) -> int:
        """Get number of times fault has triggered in current scope."""
        return cls._trigger_count_var.get()

    @classmethod
    def increment_trigger_count(cls) -> int:
        """Increment and return trigger count."""
        count = cls._trigger_count_var.get() + 1
        cls._trigger_count_var.set(count)
        return count

    @classmethod
    def reset(cls) -> None:
        """Reset fault to NONE."""
        cls.set_fault("NONE")


class fault_scope:
    """Context manager for scoping fault injection to a single test or request."""
    def __init__(self, fault: str):
        self.fault = fault
        self.previous_fault = "NONE"

    def __enter__(self):
        self.previous_fault = FaultContext.get_fault()
        FaultContext.set_fault(self.fault)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        FaultContext.set_fault(self.previous_fault)


def alter_payload_for_malformed(tool_name: str, result: dict) -> dict:
    """
    Mutate a valid tool result payload so it violates the expected Pydantic output schema,
    while preserving enough raw semantically relevant fields for LLM schema repair.
    """
    clean_name = tool_name.replace("Input", "").replace("Output", "").replace("_", "")
    
    if "FlightSearch" in clean_name or clean_name == "flightsearch":
        return {
            "msg": "flight_found_ok",
            "flight_code": result.get("flight_number", "AI-204"),
            "departure": result.get("origin", "NYC"),
            "arrival": result.get("destination", "LON"),
            "flight_date": result.get("date", "2024-10-10"),
            "fare_usd": result.get("price_usd", 249.5),
            "booking_state": result.get("status", "confirmed"),
        }
    elif "WeatherLookup" in clean_name or clean_name == "weatherlookup":
        return {
            "msg": "success",
            "loc": result.get("location", "Tokyo"),
            "temp_str": f"{result.get('temperature', 72.0)} degrees",
            "degrees_num": result.get("temperature", 72.0),
            "temp_scale": result.get("unit", "C"),
            "sky_condition": result.get("condition", "Partly Cloudy"),
        }
    elif "CalendarBooking" in clean_name or clean_name == "calendarbooking":
        return {
            "status_text": "booked",
            "meeting_id": result.get("booking_id", "cal-101"),
            "title": result.get("event_title", "Sync"),
            "starts_at": result.get("start_time", "2024-10-10T10:00:00"),
            "ends_at": result.get("end_time", "2024-10-10T10:30:00"),
            "length_mins": result.get("duration_minutes", 30),
            "confirmation": result.get("status", "confirmed"),
        }
    elif "UnitConversion" in clean_name or clean_name == "unitconversion":
        return {
            "status": "ok",
            "input_val": result.get("original_value", 10.0),
            "source_unit": result.get("from_unit", "km"),
            "output_val": result.get("converted_value", 6.21),
            "target_unit": result.get("to_unit", "miles"),
        }
    else:
        return {
            "unrecognized_structure": True,
            "raw_payload": result,
        }


def inject_fault(func: Optional[Callable] = None, *, tool_name: Optional[str] = None):
    """
    Decorator to apply to mock tools.
    Checks FaultContext.get_fault() and alters behavior accordingly:
      - NONE: Normal tool execution.
      - TIMEOUT / TIMEOUT_TRANSIENT: Raises TimeoutError on attempt 1; succeeds on retry.
      - TIMEOUT_PERSISTENT: Raises TimeoutError on all attempts (exhausts retry bound).
      - MALFORMED / MALFORMED_RESPONSE: Alters returned dict so it violates output schema.
      - MALFORMED_PERSISTENT: Returns completely corrupted unrepairable payload.
    """
    def decorator(fn: Callable) -> Callable:
        resolved_name = tool_name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            active_fault = FaultContext.get_fault().upper()

            # Transient Timeout
            if active_fault in ("TIMEOUT", "TIMEOUT_TRANSIENT"):
                count = FaultContext.get_trigger_count()
                if count == 0:
                    FaultContext.increment_trigger_count()
                    time.sleep(0.01)
                    raise TimeoutError(f"Upstream service timed out for {resolved_name}")
                # On retry, transient fault succeeds!

            # Persistent Timeout
            elif active_fault == "TIMEOUT_PERSISTENT":
                FaultContext.increment_trigger_count()
                time.sleep(0.01)
                raise TimeoutError(f"Persistent upstream timeout for {resolved_name}")

            # Execute underlying mock tool
            result = fn(*args, **kwargs)

            # Malformed Response (repairable by LLM)
            if active_fault in ("MALFORMED", "MALFORMED_RESPONSE"):
                return alter_payload_for_malformed(resolved_name, result)

            # Persistent Malformed (unrepairable garbage)
            elif active_fault == "MALFORMED_PERSISTENT":
                return {
                    "corrupt_binary": "0xDEADBEEF",
                    "error": "corrupted_payload_unrecoverable",
                    "data": None,
                }

            return result

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
