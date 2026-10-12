"""Targeted recovery strategies for AI function-calling router."""

import time
from typing import Any, Callable, Dict, Optional, Tuple, Type
from pydantic import BaseModel, ValidationError
from src.llm import LLMClient


class RecoveryResult:
    """Represents the outcome of a recovery strategy attempt."""
    def __init__(
        self,
        success: bool,
        recovered_data: Optional[Any] = None,
        error_payload: Optional[Dict[str, Any]] = None,
        policy_name: str = "none",
        attempts: int = 1,
    ):
        self.success = success
        self.recovered_data = recovered_data
        self.error_payload = error_payload
        self.policy_name = policy_name
        self.attempts = attempts


def get_field_format_description(model_cls: Type[BaseModel], field_name: str) -> str:
    """Extract human-readable format description for a field in a Pydantic model."""
    schema = model_cls.model_json_schema()
    props = schema.get("properties", {})
    prop = props.get(field_name, {})
    
    if prop.get("format") == "date" or field_name == "date":
        return "ISO date string (YYYY-MM-DD)"
    if prop.get("format") == "date-time" or field_name == "start_time":
        return "ISO datetime string (YYYY-MM-DDTHH:MM:SS)"
    if prop.get("type") == "integer":
        return "integer (int)"
    if prop.get("type") == "number":
        return "number (float)"
    if "enum" in prop:
        return f"Enum: {', '.join(prop['enum'])}"
    return prop.get("type", "valid format")


def recover_missing_field(
    llm: LLMClient,
    tool_name: str,
    raw_args: Dict[str, Any],
    validation_error: ValidationError,
    original_request: str,
    input_model_cls: Type[BaseModel],
    max_retries: int = 1,
) -> RecoveryResult:
    """
    Policy 1: Missing Field Recovery
    Prompt: 'The tool [ToolName] is missing the field [FieldName]. Based on the user's original request '[Request]', what should this value be? Return only the value.'
    Strictly bounded to 1 retry.
    """
    errors = validation_error.errors()
    # Find first missing field
    missing_err = next((e for e in errors if e.get("type") == "missing"), errors[0])
    field_name = str(missing_err["loc"][0])

    updated_args = dict(raw_args)

    for attempt in range(1, max_retries + 1):
        # Targeted narrow LLM query
        extracted_value = llm.recover_missing_field(
            tool_name=tool_name,
            field_name=field_name,
            request=original_request,
        )

        if extracted_value is not None:
            updated_args[field_name] = extracted_value

        # Re-validate with merged arguments
        try:
            validated_model = input_model_cls(**updated_args)
            return RecoveryResult(
                success=True,
                recovered_data=validated_model,
                policy_name="missing_field",
                attempts=attempt,
            )
        except ValidationError:
            pass  # Retry bound reached

    return RecoveryResult(
        success=False,
        error_payload={"status": "error", "reason": "missing_field_unrecoverable"},
        policy_name="missing_field",
        attempts=max_retries,
    )


def recover_type_error(
    llm: LLMClient,
    tool_name: str,
    raw_args: Dict[str, Any],
    validation_error: ValidationError,
    input_model_cls: Type[BaseModel],
    max_retries: int = 1,
) -> RecoveryResult:
    """
    Policy 2: Wrong Type Recovery
    Prompt: 'The field [FieldName] requires format [Format], but you provided [BadValue]. Translate this into the correct format.'
    Strictly bounded to 1 retry.
    """
    errors = validation_error.errors()
    # Find first type/format error
    type_err = next((e for e in errors if e.get("type") != "missing"), errors[0])
    field_name = str(type_err["loc"][0])
    bad_value = raw_args.get(field_name, type_err.get("input"))
    expected_format = get_field_format_description(input_model_cls, field_name)

    updated_args = dict(raw_args)

    for attempt in range(1, max_retries + 1):
        # Targeted narrow LLM query
        corrected_value = llm.recover_type_error(
            field_name=field_name,
            expected_format=expected_format,
            bad_value=bad_value,
        )

        updated_args[field_name] = corrected_value

        # Re-validate with corrected arguments
        try:
            validated_model = input_model_cls(**updated_args)
            return RecoveryResult(
                success=True,
                recovered_data=validated_model,
                policy_name="type_error",
                attempts=attempt,
            )
        except ValidationError:
            pass  # Retry bound reached

    return RecoveryResult(
        success=False,
        error_payload={"status": "error", "reason": "type_error_unrecoverable"},
        policy_name="type_error",
        attempts=max_retries,
    )


def recover_timeout(
    tool_func: Callable,
    tool_args: Dict[str, Any],
    backoff_duration: float = 0.1,
    max_retries: int = 1,
) -> RecoveryResult:
    """
    Policy 3: Timeout Recovery
    Wait specified backoff duration and re-invoke tool directly in Python without LLM.
    Strictly bounded to 1 retry.
    """
    for attempt in range(1, max_retries + 1):
        time.sleep(backoff_duration)
        try:
            # Re-invoke tool function directly
            result = tool_func(**tool_args)
            return RecoveryResult(
                success=True,
                recovered_data=result,
                policy_name="timeout",
                attempts=attempt,
            )
        except TimeoutError:
            pass  # Retry bound reached

    return RecoveryResult(
        success=False,
        error_payload={"status": "error", "reason": "timeout_unrecoverable"},
        policy_name="timeout",
        attempts=max_retries,
    )


def recover_malformed_response(
    llm: LLMClient,
    tool_name: str,
    raw_payload: Dict[str, Any],
    output_model_cls: Type[BaseModel],
    max_retries: int = 1,
) -> RecoveryResult:
    """
    Policy 4: Response Schema Repair Recovery
    Prompt: 'The tool returned this malformed payload: [Payload]. Extract the data to match this JSON schema: [Schema].'
    Strictly bounded to 1 retry.
    """
    target_schema = output_model_cls.model_json_schema()

    for attempt in range(1, max_retries + 1):
        repaired_dict = llm.repair_schema(
            malformed_payload=raw_payload,
            schema=target_schema,
        )

        try:
            validated_output = output_model_cls(**repaired_dict)
            return RecoveryResult(
                success=True,
                recovered_data=validated_output.model_dump(),
                policy_name="schema_repair",
                attempts=attempt,
            )
        except ValidationError:
            pass  # Retry bound reached

    return RecoveryResult(
        success=False,
        error_payload={"status": "error", "reason": "schema_repair_unrecoverable"},
        policy_name="schema_repair",
        attempts=max_retries,
    )
