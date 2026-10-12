"""AI Function-Calling Router with Pydantic boundaries and targeted recovery policies."""

import logging
import os
from typing import Any, Dict, List, Optional
from pydantic import ValidationError

from src.fault_injector import FaultContext, fault_scope
from src.llm import LLMClient, get_llm_client
from src.recovery import (
    recover_malformed_response,
    recover_missing_field,
    recover_timeout,
    recover_type_error,
)
from src.schemas import TOOL_INPUT_SCHEMAS, TOOL_OUTPUT_SCHEMAS
from src.tools import TOOL_REGISTRY

logger = logging.getLogger("router")


class Router:
    """Orchestration router handling tool selection, Pydantic validation, and recovery."""

    def __init__(self, llm: Optional[LLMClient] = None, backoff_duration: float = 0.05):
        self.llm = llm or get_llm_client()
        self.backoff_duration = backoff_duration

    def process_request(
        self,
        request: str,
        use_recovery: bool = True,
        injected_fault: str = "NONE",
    ) -> Dict[str, Any]:
        """
        Process a natural language request through the function-calling pipeline.

        Args:
            request: The natural language query from the user.
            use_recovery: If True, activates targeted recovery policies; if False,
                          fails immediately on any error (baseline comparison mode).
            injected_fault: Fault directive for middleware (NONE, TIMEOUT, MALFORMED_RESPONSE, etc.)

        Returns:
            Dict matching standardized success payload or explicit error payload:
            Success: {"status": "success", "tool": str, "data": dict, "attempts": int, ...}
            Failure: {"status": "error", "reason": str, ...}
        """
        trace: List[str] = []
        recovery_policy_triggered: Optional[str] = None
        recovery_attempts = 0

        initial_llm_calls = self.llm.call_count

        with fault_scope(injected_fault):
            trace.append("intent_classification_start")
            tool_name, raw_args = self.llm.parse_intent(request)
            trace.append(f"intent_selected:{tool_name}")

            if tool_name not in TOOL_INPUT_SCHEMAS:
                trace.append("unknown_tool_error")
                return {
                    "status": "error",
                    "reason": "unknown_tool_unrecoverable",
                    "attempts": 0,
                    "trace": trace,
                    "llm_call_count": self.llm.call_count - initial_llm_calls,
                }

            input_schema_cls = TOOL_INPUT_SCHEMAS[tool_name]
            output_schema_cls = TOOL_OUTPUT_SCHEMAS[tool_name]
            tool_func = TOOL_REGISTRY[tool_name]

            # ------------------------------------------------------------------
            # Phase 3 & 4: Input Validation Boundary & Targeted Recovery
            # ------------------------------------------------------------------
            trace.append("input_validation_start")
            validated_input = None

            try:
                validated_input = input_schema_cls(**raw_args)
                trace.append("input_validation_passed")
            except ValidationError as val_err:
                trace.append("input_validation_failed")

                if not use_recovery:
                    trace.append("baseline_give_up:input_validation")
                    return {
                        "status": "error",
                        "reason": "validation_error_baseline",
                        "attempts": 0,
                        "trace": trace,
                        "llm_call_count": self.llm.call_count - initial_llm_calls,
                    }

                # Determine whether missing field or type error
                errors = val_err.errors()
                has_missing = any(e.get("type") == "missing" for e in errors)

                if has_missing:
                    recovery_policy_triggered = "missing_field"
                    trace.append("recovery_policy_triggered:missing_field")
                    rec_res = recover_missing_field(
                        llm=self.llm,
                        tool_name=tool_name,
                        raw_args=raw_args,
                        validation_error=val_err,
                        original_request=request,
                        input_model_cls=input_schema_cls,
                        max_retries=1,
                    )
                    recovery_attempts += rec_res.attempts

                    if not rec_res.success:
                        trace.append("recovery_failed:missing_field_unrecoverable")
                        err_out = dict(rec_res.error_payload or {})
                        err_out.update({
                            "attempts": recovery_attempts,
                            "recovery_policy": recovery_policy_triggered,
                            "trace": trace,
                            "llm_call_count": self.llm.call_count - initial_llm_calls,
                        })
                        return err_out

                    validated_input = rec_res.recovered_data
                    trace.append("recovery_passed:missing_field")

                else:
                    recovery_policy_triggered = "type_error"
                    trace.append("recovery_policy_triggered:type_error")
                    rec_res = recover_type_error(
                        llm=self.llm,
                        tool_name=tool_name,
                        raw_args=raw_args,
                        validation_error=val_err,
                        input_model_cls=input_schema_cls,
                        max_retries=1,
                    )
                    recovery_attempts += rec_res.attempts

                    if not rec_res.success:
                        trace.append("recovery_failed:type_error_unrecoverable")
                        err_out = dict(rec_res.error_payload or {})
                        err_out.update({
                            "attempts": recovery_attempts,
                            "recovery_policy": recovery_policy_triggered,
                            "trace": trace,
                            "llm_call_count": self.llm.call_count - initial_llm_calls,
                        })
                        return err_out

                    validated_input = rec_res.recovered_data
                    trace.append("recovery_passed:type_error")

            # ------------------------------------------------------------------
            # Phase 2 & 4: Tool Execution Layer & Timeout Recovery
            # ------------------------------------------------------------------
            trace.append("tool_execution_start")
            tool_kwargs = validated_input.model_dump()
            raw_tool_output = None

            try:
                raw_tool_output = tool_func(**tool_kwargs)
                trace.append("tool_execution_success")
            except TimeoutError:
                trace.append("tool_execution_timeout")

                if not use_recovery:
                    trace.append("baseline_give_up:timeout")
                    return {
                        "status": "error",
                        "reason": "timeout_baseline",
                        "attempts": 0,
                        "trace": trace,
                        "llm_call_count": self.llm.call_count - initial_llm_calls,
                    }

                recovery_policy_triggered = "timeout"
                trace.append("recovery_policy_triggered:timeout")
                rec_res = recover_timeout(
                    tool_func=tool_func,
                    tool_args=tool_kwargs,
                    backoff_duration=self.backoff_duration,
                    max_retries=1,
                )
                recovery_attempts += rec_res.attempts

                if not rec_res.success:
                    trace.append("recovery_failed:timeout_unrecoverable")
                    err_out = dict(rec_res.error_payload or {})
                    err_out.update({
                        "attempts": recovery_attempts,
                        "recovery_policy": recovery_policy_triggered,
                        "trace": trace,
                        "llm_call_count": self.llm.call_count - initial_llm_calls,
                    })
                    return err_out

                raw_tool_output = rec_res.recovered_data
                trace.append("recovery_passed:timeout")

            # ------------------------------------------------------------------
            # Phase 4 & 7: Output Schema Validation & Schema Repair Recovery
            # ------------------------------------------------------------------
            trace.append("output_validation_start")
            validated_output = None

            try:
                validated_output = output_schema_cls(**raw_tool_output)
                trace.append("output_validation_passed")
            except (ValidationError, TypeError):
                trace.append("output_validation_failed")

                if not use_recovery:
                    trace.append("baseline_give_up:output_schema_mismatch")
                    return {
                        "status": "error",
                        "reason": "malformed_response_baseline",
                        "attempts": 0,
                        "trace": trace,
                        "llm_call_count": self.llm.call_count - initial_llm_calls,
                    }

                recovery_policy_triggered = "schema_repair"
                trace.append("recovery_policy_triggered:schema_repair")
                rec_res = recover_malformed_response(
                    llm=self.llm,
                    tool_name=tool_name,
                    raw_payload=raw_tool_output,
                    output_model_cls=output_schema_cls,
                    max_retries=1,
                )
                recovery_attempts += rec_res.attempts

                if not rec_res.success:
                    trace.append("recovery_failed:schema_repair_unrecoverable")
                    err_out = dict(rec_res.error_payload or {})
                    err_out.update({
                        "attempts": recovery_attempts,
                        "recovery_policy": recovery_policy_triggered,
                        "trace": trace,
                        "llm_call_count": self.llm.call_count - initial_llm_calls,
                    })
                    return err_out

                validated_output = rec_res.recovered_data
                trace.append("recovery_passed:schema_repair")

            # Final Standardized Success Response
            return {
                "status": "success",
                "tool": tool_name,
                "data": validated_output.model_dump() if hasattr(validated_output, "model_dump") else validated_output,
                "attempts": recovery_attempts,
                "recovery_policy": recovery_policy_triggered,
                "trace": trace,
                "llm_call_count": self.llm.call_count - initial_llm_calls,
            }


# Module-level convenience function matching instructions skeleton
_default_router: Optional[Router] = None


def process_request(request: str, use_recovery: bool = True, injected_fault: str = "NONE") -> Dict[str, Any]:
    """Default module-level entrypoint for router request processing."""
    global _default_router
    if _default_router is None:
        _default_router = Router()
    return _default_router.process_request(request, use_recovery=use_recovery, injected_fault=injected_fault)
