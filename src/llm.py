"""LLM interface supporting deterministic offline simulation and OpenAI API."""

import json
import os
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional, Tuple


class LLMClient:
    """Base interface for LLM operations in the function-calling router."""

    def __init__(self):
        self.call_count = 0

    def parse_intent(self, request: str) -> Tuple[str, Dict[str, Any]]:
        """Select tool and extract raw arguments from user natural language request."""
        raise NotImplementedError

    def recover_missing_field(self, tool_name: str, field_name: str, request: str) -> Any:
        """Targeted re-prompt to recover missing field from original request."""
        raise NotImplementedError

    def recover_type_error(self, field_name: str, expected_format: str, bad_value: Any) -> Any:
        """Targeted re-prompt to translate bad value into expected format."""
        raise NotImplementedError

    def repair_schema(self, malformed_payload: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Targeted re-prompt to extract schema-compliant payload from malformed response."""
        raise NotImplementedError


class DeterministicLLM(LLMClient):
    """
    Deterministic rule-based LLM engine for reproducible, offline evaluation.
    Accurately models LLM intent classification, argument extraction, and targeted re-prompts.
    """

    def __init__(self, reference_date: Optional[date] = None):
        super().__init__()
        # Reference date for relative date resolution (e.g., "tomorrow")
        self.reference_date = reference_date or date(2026, 10, 12)

    def parse_intent(self, request: str) -> Tuple[str, Dict[str, Any]]:
        self.call_count += 1
        req_lower = request.lower()

        # 1. FlightSearch intent
        if any(w in req_lower for w in ["flight", "fly", "plane", "ticket"]) or ("from " in req_lower and "to " in req_lower and not "convert" in req_lower):
            tool = "FlightSearch"
            args: Dict[str, Any] = {}

            # Origin extraction
            origin_match = re.search(r"from\s+([A-Za-z\s]+?)(?:\s+to|\s+on|\s+tomorrow|\s+next|\.|$)", request, re.IGNORECASE)
            if origin_match:
                orig = origin_match.group(1).strip()
                if orig:
                    args["origin"] = orig

            # Destination extraction
            dest_match = re.search(r"to\s+([A-Za-z\s]+?)(?:\s+from|\s+on|\s+tomorrow|\s+next|\.|$)", request, re.IGNORECASE)
            if dest_match:
                dest = dest_match.group(1).strip()
                if dest:
                    args["destination"] = dest

            # Date extraction
            if "tomorrow" in req_lower:
                args["date"] = "tomorrow"  # Intentionally triggers type error if ISO required
            elif "next tuesday" in req_lower:
                args["date"] = "next Tuesday"
            elif "next week" in req_lower:
                args["date"] = "next week"
            else:
                iso_date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", request)
                if iso_date_match:
                    args["date"] = iso_date_match.group(1)
                else:
                    # Check for "on <date>"
                    on_date_match = re.search(r"on\s+([A-Za-z0-9\-]+)", request, re.IGNORECASE)
                    if on_date_match and on_date_match.group(1).lower() not in ("tomorrow", "tuesday"):
                        args["date"] = on_date_match.group(1)

            return tool, args

        # 2. WeatherLookup intent
        if any(w in req_lower for w in ["weather", "temperature", "forecast", "how hot", "how cold", "degrees"]):
            tool = "WeatherLookup"
            args: Dict[str, Any] = {}

            # Location extraction
            loc_match = re.search(r"\b(?:in|for|at)\s+([A-Za-z\s]+?)(?:\s+(?:in|with|using)|[.?]|$)", request, re.IGNORECASE)
            if loc_match:
                loc = loc_match.group(1).strip()
                # filter out unit words
                if loc.lower() not in ("celsius", "fahrenheit", "c", "f"):
                    args["location"] = loc

            # Unit extraction
            if any(w in req_lower for w in ["fahrenheit", "in f", "in fahrenheit", "°f"]):
                args["unit"] = "F"
            elif any(w in req_lower for w in ["celsius", "in c", "in celsius", "°c"]):
                args["unit"] = "C"
            elif "degrees" in req_lower and "f" in req_lower:
                args["unit"] = "F"
            elif "location" in args:
                # Default to C if location is specified but unit omitted
                args["unit"] = "C"

            # Check if type error should be provoked (e.g. natural language unit "celsius" instead of "C")
            if "unit: celsius" in req_lower or "unit=celsius" in req_lower:
                args["unit"] = "celsius"

            return tool, args

        # 3. CalendarBooking intent
        if any(w in req_lower for w in ["meeting", "schedule", "calendar", "book a call", "appointment", "event", "sync"]):
            tool = "CalendarBooking"
            args: Dict[str, Any] = {}

            # Title
            title_match = re.search(r'(?:titled|called|named|for)\s+["\']?([^"\',]+?)["\']?(?:\s+at|\s+on|\s+for|\.|$)', request, re.IGNORECASE)
            if title_match:
                args["event_title"] = title_match.group(1).strip()
            else:
                args["event_title"] = "Team Sync"

            # Start time
            iso_dt_match = re.search(r"\b(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\b", request)
            if iso_dt_match:
                args["start_time"] = iso_dt_match.group(1)
            elif "tomorrow at 10am" in req_lower or "tomorrow at 10:00" in req_lower:
                args["start_time"] = "tomorrow at 10am"  # provoke type error
            elif "next monday at 2pm" in req_lower:
                args["start_time"] = "next Monday at 2pm"
            else:
                date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", request)
                if date_match:
                    args["start_time"] = f"{date_match.group(1)}T09:00:00"

            # Duration
            dur_match = re.search(r"(\d+)\s*(?:min|minute|minutes|m\b)", req_lower)
            if dur_match:
                args["duration_minutes"] = int(dur_match.group(1))
            elif "1 hour" in req_lower or "one hour" in req_lower:
                # If explicit type error requested:
                if "string_duration" in req_lower:
                    args["duration_minutes"] = "1 hour"
                else:
                    args["duration_minutes"] = 60

            return tool, args

        # 4. UnitConversion intent
        if any(w in req_lower for w in ["convert", "conversion", "miles to km", "km to miles", "celsius to fahrenheit"]):
            tool = "UnitConversion"
            args: Dict[str, Any] = {}

            # Value
            val_match = re.search(r"(?:convert\s+)?([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z°]+)?\s+(?:to|into)\s+([a-zA-Z°]+)", request, re.IGNORECASE)
            if val_match:
                args["value"] = float(val_match.group(1))
                if val_match.group(2):
                    args["from_unit"] = val_match.group(2).strip()
                args["to_unit"] = val_match.group(3).strip()

            # Separate patterns if not caught above
            if "value" not in args:
                num_match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\b", request)
                if num_match:
                    args["value"] = float(num_match.group(1))

            if "from_unit" not in args:
                from_match = re.search(r"from\s+([a-zA-Z]+)", request, re.IGNORECASE)
                if from_match:
                    args["from_unit"] = from_match.group(1).strip()

            if "to_unit" not in args:
                to_match = re.search(r"to\s+([a-zA-Z]+)", request, re.IGNORECASE)
                if to_match:
                    args["to_unit"] = to_match.group(1).strip()

            return tool, args

        # Default fallback
        return "WeatherLookup", {"location": "San Francisco", "unit": "C"}

    def recover_missing_field(self, tool_name: str, field_name: str, request: str) -> Any:
        """
        Implements targeted re-prompt:
        'The tool [ToolName] is missing the field [FieldName]. Based on the user's original request '[Request]', what should this value be? Return only the value.'
        """
        self.call_count += 1
        req_lower = request.lower()

        # If original request contains unresolvable marker, simulate unrecoverable
        if "unresolvable" in req_lower or "impossible" in req_lower:
            return None

        if field_name == "date":
            # Check if user mentioned tomorrow, next week, or ISO date
            if "tomorrow" in req_lower:
                return (self.reference_date + timedelta(days=1)).isoformat()
            if "next tuesday" in req_lower:
                return (self.reference_date + timedelta(days=8)).isoformat()
            iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", request)
            if iso_match:
                return iso_match.group(1)
            # Default fallback date if request had vague travel context
            return (self.reference_date + timedelta(days=7)).isoformat()

        if field_name == "origin":
            origin_match = re.search(r"from\s+([A-Za-z\s]+?)(?:\s+to|\s+on|\.|$)", request, re.IGNORECASE)
            if origin_match:
                return origin_match.group(1).strip()
            # If user said e.g. "Paris to Tokyo", origin is Paris
            pair_match = re.search(r"([A-Za-z]+)\s+to\s+([A-Za-z]+)", request, re.IGNORECASE)
            if pair_match:
                return pair_match.group(1).strip()
            return "JFK"  # Standard default departure hub

        if field_name == "destination":
            dest_match = re.search(r"to\s+([A-Za-z\s]+?)(?:\s+on|\.|$)", request, re.IGNORECASE)
            if dest_match:
                return dest_match.group(1).strip()
            return "LHR"

        if field_name == "duration_minutes":
            dur_match = re.search(r"(\d+)\s*(?:min|minute)", req_lower)
            if dur_match:
                return int(dur_match.group(1))
            if "1 hour" in req_lower:
                return 60
            return 30

        if field_name == "start_time":
            iso_dt = re.search(r"\b(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\b", request)
            if iso_dt:
                return iso_dt.group(1)
            return f"{(self.reference_date + timedelta(days=1)).isoformat()}T09:00:00"

        if field_name == "location":
            loc_match = re.search(r"\b(?:in|for|at)\s+([A-Za-z\s]+?)(?:\s+(?:in|with)|[.?]|$)", request, re.IGNORECASE)
            if loc_match:
                return loc_match.group(1).strip()
            return "London"

        if field_name == "unit":
            if any(w in req_lower for w in ["fahrenheit", " f", "°f"]):
                return "F"
            return "C"

        if field_name == "value":
            num_match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\b", request)
            if num_match:
                return float(num_match.group(1))
            return 1.0

        if field_name == "from_unit":
            return "miles"

        if field_name == "to_unit":
            return "km"

        return None

    def recover_type_error(self, field_name: str, expected_format: str, bad_value: Any) -> Any:
        """
        Implements targeted re-prompt:
        'The field [FieldName] requires format [Format], but you provided [BadValue]. Translate this into the correct format.'
        """
        self.call_count += 1
        bad_str = str(bad_value).strip().lower()

        # If bad_value is specifically marked unrecoverable
        if "uncastable" in bad_str or "unrecoverable" in bad_str:
            return bad_value  # Will fail validation again!

        if field_name == "date":
            if bad_str == "tomorrow":
                return (self.reference_date + timedelta(days=1)).isoformat()
            if bad_str in ("next tuesday", "tuesday"):
                return (self.reference_date + timedelta(days=8)).isoformat()
            if bad_str in ("next week", "in a week"):
                return (self.reference_date + timedelta(days=7)).isoformat()
            # Try to resolve relative or formatted date
            return (self.reference_date + timedelta(days=1)).isoformat()

        if field_name == "start_time":
            if "tomorrow" in bad_str:
                return f"{(self.reference_date + timedelta(days=1)).isoformat()}T10:00:00"
            if "next monday" in bad_str:
                return f"{(self.reference_date + timedelta(days=7)).isoformat()}T14:00:00"
            return f"{(self.reference_date + timedelta(days=1)).isoformat()}T09:00:00"

        if field_name == "duration_minutes":
            if "hour" in bad_str:
                nums = re.findall(r"\d+", bad_str)
                hours = int(nums[0]) if nums else 1
                return hours * 60
            nums = re.findall(r"\d+", bad_str)
            if nums:
                return int(nums[0])
            return 30

        if field_name == "unit":
            if "c" in bad_str:
                return "C"
            if "f" in bad_str:
                return "F"
            return "C"

        if field_name == "value":
            # Strip dollar signs or commas
            clean = re.sub(r"[^\d.]", "", bad_str)
            if clean:
                return float(clean)
            return 0.0

        return bad_value

    def repair_schema(self, malformed_payload: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implements targeted re-prompt:
        'The tool returned this malformed payload: [Payload]. Extract the data to match this JSON schema: [Schema].'
        """
        self.call_count += 1

        # Check if the payload is intentionally corrupted and unrecoverable
        if malformed_payload.get("error") == "corrupted_payload_unrecoverable" or "corrupt_binary" in malformed_payload:
            return {"unrepairable": True}

        repaired: Dict[str, Any] = {}
        target_properties = schema.get("properties", {})

        # FlightSearch Output schema repair
        if "flight_number" in target_properties:
            repaired["flight_number"] = malformed_payload.get("flight_code", malformed_payload.get("flight_number", "AI-204"))
            repaired["origin"] = malformed_payload.get("departure", malformed_payload.get("origin", "NYC"))
            repaired["destination"] = malformed_payload.get("arrival", malformed_payload.get("destination", "LON"))
            repaired["date"] = malformed_payload.get("flight_date", malformed_payload.get("date", "2024-10-10"))
            fare = malformed_payload.get("fare_usd", malformed_payload.get("price_usd", 249.5))
            if isinstance(fare, str):
                fare = float(re.sub(r"[^\d.]", "", fare))
            repaired["price_usd"] = float(fare)
            repaired["status"] = malformed_payload.get("booking_state", malformed_payload.get("status", "confirmed"))
            return repaired

        # WeatherLookup Output schema repair
        if "temperature" in target_properties and "condition" in target_properties:
            repaired["location"] = malformed_payload.get("loc", malformed_payload.get("location_name", malformed_payload.get("location", "Tokyo")))
            temp = malformed_payload.get("degrees_num", malformed_payload.get("temperature", 72.0))
            if isinstance(temp, str) or "temp_str" in malformed_payload:
                raw_temp = str(malformed_payload.get("temp_str", temp))
                num_match = re.search(r"[-+]?\d*\.?\d+", raw_temp)
                temp = float(num_match.group(0)) if num_match else 72.0
            repaired["temperature"] = float(temp)
            repaired["unit"] = malformed_payload.get("temp_scale", malformed_payload.get("unit", "C"))
            repaired["condition"] = malformed_payload.get("sky_condition", malformed_payload.get("sky", malformed_payload.get("condition", "Partly Cloudy")))
            return repaired

        # CalendarBooking Output schema repair
        if "booking_id" in target_properties and "event_title" in target_properties:
            repaired["booking_id"] = malformed_payload.get("meeting_id", malformed_payload.get("ref_id", malformed_payload.get("booking_id", "cal-101")))
            repaired["event_title"] = malformed_payload.get("title", malformed_payload.get("summary", malformed_payload.get("event_title", "Sync")))
            repaired["start_time"] = malformed_payload.get("starts_at", malformed_payload.get("begins", malformed_payload.get("start_time", "2024-10-10T10:00:00")))
            repaired["end_time"] = malformed_payload.get("ends_at", malformed_payload.get("end_time", "2024-10-10T10:30:00"))
            length = malformed_payload.get("length_mins", malformed_payload.get("duration_minutes", 30))
            if isinstance(length, str):
                nums = re.findall(r"\d+", length)
                length = int(nums[0]) if nums else 30
            repaired["duration_minutes"] = int(length)
            repaired["status"] = malformed_payload.get("confirmation", malformed_payload.get("status_text", malformed_payload.get("status", "confirmed")))
            return repaired

        # UnitConversion Output schema repair
        if "original_value" in target_properties and "converted_value" in target_properties:
            repaired["original_value"] = float(malformed_payload.get("input_val", malformed_payload.get("original_value", 10.0)))
            repaired["from_unit"] = malformed_payload.get("source_unit", malformed_payload.get("unit_from", malformed_payload.get("from_unit", "km")))
            out_val = malformed_payload.get("output_val", malformed_payload.get("converted_value", 6.21))
            if isinstance(out_val, str) or "result_str" in malformed_payload:
                raw_out = str(malformed_payload.get("result_str", out_val))
                num_match = re.search(r"[-+]?\d*\.?\d+", raw_out)
                out_val = float(num_match.group(0)) if num_match else 6.21
            repaired["converted_value"] = float(out_val)
            repaired["to_unit"] = malformed_payload.get("target_unit", malformed_payload.get("to_unit", "miles"))
            return repaired

        return malformed_payload


class OpenAILLM(LLMClient):
    """OpenAI API implementation using targeted prompts."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        super().__init__()
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def parse_intent(self, request: str) -> Tuple[str, Dict[str, Any]]:
        self.call_count += 1
        system_prompt = (
            "You are a function calling agent. Given the user's request, select the best tool from: "
            "['FlightSearch', 'CalendarBooking', 'WeatherLookup', 'UnitConversion']. "
            "Extract arguments as a JSON object: {'tool': '<ToolName>', 'arguments': {...}}. "
            "Emit only valid JSON."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(response.choices[0].message.content or "{}")
        return data.get("tool", "WeatherLookup"), data.get("arguments", {})

    def recover_missing_field(self, tool_name: str, field_name: str, request: str) -> Any:
        self.call_count += 1
        prompt = (
            f"The tool {tool_name} is missing the field {field_name}. "
            f"Based on the user's original request '{request}', what should this value be? "
            f"Return only the value."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = (response.choices[0].message.content or "").strip().strip('"').strip("'")
        try:
            return json.loads(content)
        except Exception:
            return content

    def recover_type_error(self, field_name: str, expected_format: str, bad_value: Any) -> Any:
        self.call_count += 1
        prompt = (
            f"The field {field_name} requires format {expected_format}, but you provided {bad_value}. "
            f"Translate this into the correct format."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = (response.choices[0].message.content or "").strip().strip('"').strip("'")
        try:
            return json.loads(content)
        except Exception:
            return content

    def repair_schema(self, malformed_payload: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        self.call_count += 1
        prompt = (
            f"The tool returned this malformed payload: {json.dumps(malformed_payload)}. "
            f"Extract the data to match this JSON schema: {json.dumps(schema)}."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(response.choices[0].message.content or "{}")


def get_llm_client() -> LLMClient:
    """Factory to get configured LLM client."""
    use_openai = os.getenv("USE_OPENAI", "false").lower() in ("true", "1")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if use_openai and api_key and not api_key.startswith("your_") and not api_key.startswith("replace_"):
        try:
            model = os.getenv("LLM_MODEL_ID", "gpt-4o-mini")
            return OpenAILLM(api_key=api_key, model=model)
        except Exception:
            return DeterministicLLM()

    return DeterministicLLM()
