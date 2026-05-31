import json
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import requests

from lisanbench.providers.common import (
    append_stream_event,
    coerce_nonnegative_float as _coerce_nonnegative_float,
    extract_openai_usage_tokens,
    extract_usage_cost as _extract_usage_cost,
    iter_sse_events,
    truncate as _truncate,
)
from lisanbench.providers.types import CompletionResult
from lisanbench.model_catalog import get_openrouter_provider_config, get_openrouter_service_tier
from lisanbench.model_names import (
    OPENROUTER_REASONING_EFFORT_LEVELS,
    OPENROUTER_VERBOSITY_LEVELS,
    extract_word_boundary_reasoning_effort,
    implicit_reasoning_label,
    split_model_suffix,
    split_reasoning_suffix,
)

if TYPE_CHECKING:
    from lisanbench.completions import CompletionAPI


OpenRouterCostRecord = Dict[str, Any]


def _format_stream_error_details(
    error_block: Dict[str, Any],
    parsed_event: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    metadata = error_block.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    provider_name = metadata.get("provider_name")
    raw_payload = metadata.get("raw")

    upstream_message = None
    upstream_code = None
    upstream_type = None
    if isinstance(raw_payload, str) and raw_payload.strip():
        try:
            raw_obj = json.loads(raw_payload)
            if isinstance(raw_obj, dict):
                raw_error = raw_obj.get("error")
                if isinstance(raw_error, dict):
                    upstream_message = raw_error.get("message")
                    upstream_code = raw_error.get("code")
                    upstream_type = raw_error.get("type")
        except json.JSONDecodeError:
            pass

    error_info: Dict[str, Any] = {
        "message": error_block.get("message"),
        "code": error_block.get("code"),
        "provider": parsed_event.get("provider"),
        "provider_name": provider_name,
        "finish_reason": extract_finish_reason(parsed_event),
        "upstream_type": upstream_type,
        "upstream_code": upstream_code,
        "upstream_message": upstream_message,
    }

    detail_parts: List[str] = []
    if error_info["message"] is not None:
        detail_parts.append(f"message='{_truncate(error_info['message'])}'")
    if error_info["code"] is not None:
        detail_parts.append(f"code={error_info['code']}")
    if provider_name:
        detail_parts.append(f"provider='{_truncate(provider_name)}'")
    if upstream_type:
        detail_parts.append(f"upstream_type='{_truncate(upstream_type)}'")
    if upstream_code:
        detail_parts.append(f"upstream_code='{_truncate(upstream_code)}'")
    if upstream_message:
        detail_parts.append(f"upstream_message='{_truncate(upstream_message)}'")

    detail = ", ".join(detail_parts) if detail_parts else "unknown error"
    return error_info, detail


def call_openrouter_api(
    api: "CompletionAPI",
    model_name: str,
    prompt: str,
) -> CompletionResult:
    """Call OpenRouter API for models using requests."""
    openrouter_api_key = api._require_openrouter_key()
    sent_model_name = model_name.split(":", 1)[0]

    request_data: Dict[str, Any] = {
        "model": sent_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": api.temperature,
        "max_tokens": api.max_tokens,
    }

    provider_config = get_openrouter_provider_config(model_name)
    if provider_config:
        request_data["provider"] = provider_config

    service_tier = get_openrouter_service_tier(model_name)
    if service_tier:
        request_data["service_tier"] = service_tier

    if api.stream_responses:
        request_data["stream"] = True
        request_data["stream_options"] = {"include_usage": True}

    reasoning_payload = build_reasoning_payload(model_name)
    if reasoning_payload:
        request_data["reasoning"] = reasoning_payload

    verbosity = build_verbosity(model_name)
    if verbosity:
        request_data["verbosity"] = verbosity

    request_payload_for_logging = dict(request_data, model=model_name)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
    }

    try:
        session = api._get_session()
        response = session.post(
            url,
            headers=headers,
            json=request_data,
            timeout=(60, 14400),
            stream=api.stream_responses,
        )
    except requests.RequestException as exc:
        error_info = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        api._log_raw_interaction(
            model_name=model_name,
            prompt=prompt,
            request_payload=request_payload_for_logging,
            error=error_info,
        )
        raise Exception(f"OpenRouter API error: {exc}") from exc

    if api.stream_responses:
        return consume_streaming_response(
            api,
            response,
            model_name,
            prompt,
            request_payload_for_logging,
        )
    return consume_json_response(
        api,
        response,
        model_name,
        prompt,
        request_payload_for_logging,
    )


def consume_json_response(
    api: "CompletionAPI",
    response: requests.Response,
    model_name: str,
    prompt: str,
    request_payload: Dict[str, Any],
) -> CompletionResult:
    response_text = response.text
    if not response.ok:
        body_preview = response_text[:700]
        detailed_error = None
        try:
            err_json = response.json()
            if isinstance(err_json, dict):
                err_block = err_json.get("error")
                if isinstance(err_block, dict):
                    err_msg = err_block.get("message")
                    err_code = err_block.get("code")
                    err_type = err_block.get("type")
                    detail_parts: List[str] = []
                    if err_msg is not None:
                        detail_parts.append(f"message='{_truncate(err_msg)}'")
                    if err_type is not None:
                        detail_parts.append(f"type='{_truncate(err_type)}'")
                    if err_code is not None:
                        detail_parts.append(f"code='{_truncate(err_code)}'")
                    if detail_parts:
                        detailed_error = ", ".join(detail_parts)
        except Exception:
            pass

        error_info = {
            "status_code": response.status_code,
            "reason": response.reason,
            "body_preview": body_preview,
            "parsed_error": detailed_error,
        }
        api._log_raw_interaction(
            model_name=model_name,
            prompt=prompt,
            request_payload=request_payload,
            raw_response_text=response_text,
            error=error_info,
        )
        response.close()
        detail_suffix = f" ({detailed_error})" if detailed_error else ""
        raise Exception(
            f"OpenRouter API error: HTTP {response.status_code} {response.reason}{detail_suffix}"
        )

    try:
        completion_data = response.json()
    except json.JSONDecodeError as exc:
        response.close()
        error_info = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        api._log_raw_interaction(
            model_name=model_name,
            prompt=prompt,
            request_payload=request_payload,
            raw_response_text=response_text,
            error=error_info,
        )
        raise Exception(
            f"OpenRouter API error: Failed to decode JSON response: {exc}"
        ) from exc

    response.close()
    generation_id = completion_data.get("id") or "no-id"

    api._log_raw_interaction(
        model_name=model_name,
        prompt=prompt,
        request_payload=request_payload,
        generation_id=generation_id,
        response_payload=completion_data,
        raw_response_text=response_text,
    )

    response_text = completion_data["choices"][0]["message"]["content"]

    usage = completion_data.get("usage") or {}
    usage_tokens = extract_openai_usage_tokens(usage)

    preliminary_cost = _extract_usage_cost(usage)
    return CompletionResult(
        text=response_text,
        cost_usd=preliminary_cost,
        generation_id=generation_id,
        input_tokens=usage_tokens.input_tokens,
        output_tokens=usage_tokens.output_tokens,
        reasoning_tokens=usage_tokens.reasoning_tokens,
    )


def consume_streaming_response(
    api: "CompletionAPI",
    response: requests.Response,
    model_name: str,
    prompt: str,
    request_payload: Dict[str, Any],
) -> CompletionResult:
    if not response.ok:
        error_body = response.text
        error_info = {
            "status_code": response.status_code,
            "reason": response.reason,
            "body_preview": error_body[:500],
        }

        api._log_raw_interaction(
            model_name=model_name,
            prompt=prompt,
            request_payload=request_payload,
            raw_response_text=error_body,
            error=error_info,
        )
        response.close()
        raise Exception(
            f"OpenRouter API error: HTTP {response.status_code} {response.reason}"
        )

    response_chunks: List[str] = []
    stream_events: List[Dict[str, Any]] = []
    usage: Dict[str, Any] = {}
    generation_id = "no-id"

    def log_stream_event(event: Dict[str, Any]) -> None:
        append_stream_event(stream_events, event)

    try:
        for event_kind, event_payload in iter_sse_events(response):
            if event_kind == "raw_line":
                log_stream_event({"raw_line": event_payload})
                continue
            if event_kind == "done":
                break
            if event_kind == "malformed":
                log_stream_event({"malformed": event_payload})
                continue

            parsed_event = event_payload
            event_obj = parsed_event if isinstance(parsed_event, dict) else {"event": parsed_event}
            log_stream_event(event_obj)
            if isinstance(parsed_event, dict):
                generation_id = parsed_event.get("id", generation_id)
                usage = parsed_event.get("usage") or usage

                error_block = parsed_event.get("error")
                if error_block:
                    error_info, detail = _format_stream_error_details(error_block, parsed_event)
                    partial_response = "".join(response_chunks).strip()
                    api._log_raw_interaction(
                        model_name=model_name,
                        prompt=prompt,
                        request_payload=request_payload,
                        generation_id=generation_id,
                        raw_response_text=partial_response,
                        error=error_info,
                        stream_events=stream_events,
                    )
                    raise Exception(
                        f"OpenRouter streaming error: {detail}"
                    )

                choices = parsed_event.get("choices") or []
            else:
                choices = []
            if choices:
                chunk = extract_text_from_choice(
                    choices[0],
                    has_existing_chunks=bool(response_chunks),
                )
                if chunk:
                    response_chunks.append(chunk)
    finally:
        response.close()

    response_text = "".join(response_chunks).strip()
    if not response_text:
        error_info = {
            "message": "Streaming response yielded no text",
            "logged_events": len(stream_events),
        }
        api._log_raw_interaction(
            model_name=model_name,
            prompt=prompt,
            request_payload=request_payload,
            error=error_info,
            stream_events=stream_events,
        )
        raise Exception("OpenRouter API error: Empty streaming response")

    api._log_raw_interaction(
        model_name=model_name,
        prompt=prompt,
        request_payload=request_payload,
        generation_id=generation_id,
        response_payload={"usage": usage},
        raw_response_text=response_text,
        stream_events=stream_events,
    )

    usage_tokens = extract_openai_usage_tokens(usage)

    preliminary_cost = _extract_usage_cost(usage)
    return CompletionResult(
        text=response_text,
        cost_usd=preliminary_cost,
        generation_id=generation_id,
        input_tokens=usage_tokens.input_tokens,
        output_tokens=usage_tokens.output_tokens,
        reasoning_tokens=usage_tokens.reasoning_tokens,
    )


def extract_text_from_choice(
    choice: Dict[str, Any],
    *,
    has_existing_chunks: bool,
) -> str:
    chunk = extract_text_block(choice.get("delta"))
    if chunk:
        return chunk
    if not has_existing_chunks:
        return extract_text_block(choice.get("message"))
    return ""


def extract_text_block(block: Any) -> str:
    if not block:
        return ""
    if isinstance(block, str):
        return block
    if isinstance(block, list):
        parts = [extract_text_block(part) for part in block]
        return "".join(parts)
    if isinstance(block, dict):
        if isinstance(block.get("content"), str):
            return block["content"]
        if isinstance(block.get("content"), list):
            return "".join(extract_text_block(part) for part in block["content"])
        if isinstance(block.get("parts"), list):
            return "".join(extract_text_block(part) for part in block["parts"])
        if isinstance(block.get("text"), str):
            return block["text"]
    return ""


def extract_finish_reason(event: Dict[str, Any]) -> Optional[str]:
    choices = event.get("choices") or []
    if choices:
        return choices[0].get("finish_reason")
    return None


def get_openrouter_costs(
    api: "CompletionAPI",
    generation_ids: List[str],
) -> Dict[str, OpenRouterCostRecord]:
    """Get actual costs from OpenRouter generation API.

    Retries generation lookups with backoff waits of 1 min, 2 min, and 4 min.
    Returns resolved generation records keyed by generation id.
    Unresolved ids are omitted from the returned mapping.
    """
    openrouter_api_key = api._require_openrouter_key()

    costs: Dict[str, OpenRouterCostRecord] = {}
    url = "https://openrouter.ai/api/v1/generation"
    headers = {"Authorization": f"Bearer {openrouter_api_key}"}
    wait_schedule_seconds = [60, 120, 240]
    pending_ids = list(generation_ids)

    for attempt, wait_seconds in enumerate(wait_schedule_seconds, start=1):
        if not pending_ids:
            break

        print(
            f"Waiting {wait_seconds}s before OpenRouter cost fetch attempt "
            f"{attempt}/{len(wait_schedule_seconds)} for {len(pending_ids)} pending IDs..."
        )
        time.sleep(wait_seconds)

        unresolved_next: List[str] = []
        for generation_id in pending_ids:
            try:
                session = api._get_session()
                response = session.get(
                    url,
                    headers=headers,
                    params={"id": generation_id},
                    timeout=60,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                print(
                    f"Warning: Could not retrieve OpenRouter cost for {generation_id} "
                    f"(attempt {attempt}/{len(wait_schedule_seconds)}): {exc}"
                )
                unresolved_next.append(generation_id)
                continue

            try:
                data = response.json()
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: Invalid JSON while retrieving cost for {generation_id} "
                    f"(attempt {attempt}/{len(wait_schedule_seconds)}): {exc}"
                )
                unresolved_next.append(generation_id)
                continue

            data_block = data.get("data") if isinstance(data, dict) else None
            total_cost = data_block.get("total_cost") if isinstance(data_block, dict) else None
            usage_cost = data_block.get("usage") if isinstance(data_block, dict) else None

            resolved_cost = _coerce_nonnegative_float(total_cost)
            if resolved_cost is None:
                resolved_cost = _coerce_nonnegative_float(usage_cost)

            if resolved_cost is None:
                print(
                    f"Warning: OpenRouter did not return a usable total_cost for {generation_id} "
                    f"(attempt {attempt}/{len(wait_schedule_seconds)})."
                )
                unresolved_next.append(generation_id)
                continue

            token_count = 0
            if isinstance(data_block, dict):
                for key in (
                    "tokens_prompt",
                    "tokens_completion",
                    "native_tokens_prompt",
                    "native_tokens_completion",
                    "native_tokens_reasoning",
                ):
                    value = data_block.get(key)
                    if isinstance(value, int) and value > 0:
                        token_count += value

            is_byok = bool(data_block.get("is_byok")) if isinstance(data_block, dict) else False
            if resolved_cost == 0.0 and token_count > 0 and not is_byok:
                print(
                    f"Warning: OpenRouter returned zero cost for {generation_id} "
                    f"despite non-zero token usage (attempt {attempt}/{len(wait_schedule_seconds)}). "
                    f"Will retry."
                )
                unresolved_next.append(generation_id)
                continue

            costs[generation_id] = {
                "cost": resolved_cost,
                "is_byok": is_byok,
                "provider_name": data_block.get("provider_name") if isinstance(data_block, dict) else None,
            }

        pending_ids = unresolved_next

    if pending_ids:
        print(
            f"Warning: OpenRouter cost still unavailable after retries for "
            f"{len(pending_ids)} generation IDs. Later fallbacks may still fill them."
        )

    return costs


# Models where manual thinking budget/effort is rejected or ignored and only
# adaptive thinking is supported. On these, `:thinking-<level>` maps to the
# top-level `verbosity` field (which OpenRouter forwards as Anthropic's
# `output_config.effort`) instead of `reasoning.effort` / `reasoning.max_tokens`.
_ADAPTIVE_ONLY_MODELS = frozenset({
    "anthropic/claude-opus-4.7",
    "anthropic/claude-opus-4.8",
})

def _is_adaptive_only(base_model: str) -> bool:
    return base_model in _ADAPTIVE_ONLY_MODELS


def build_reasoning_payload(model_name: str) -> Optional[Dict[str, Any]]:
    """Return the appropriate reasoning payload for models that require it."""
    base_model, suffix = split_model_suffix(model_name, lower_suffix=True)

    if "none" in suffix or "off" in suffix:
        return {"enabled": False}

    if _is_adaptive_only(base_model):
        # Opus 4.7+ ignores reasoning.max_tokens and reasoning.effort — the
        # model always uses adaptive thinking. Only the on/off signal matters
        # here; depth is steered via build_verbosity.
        return {"enabled": "thinking" in suffix}

    if "thinking" not in suffix:
        reasoning_base, reasoning_suffix = split_reasoning_suffix(model_name.lower())
        implicit_label = implicit_reasoning_label(reasoning_base, reasoning_suffix)
        if not implicit_label:
            return {"enabled": False}

        payload: Dict[str, Any] = {"enabled": True}
        if implicit_label in OPENROUTER_REASONING_EFFORT_LEVELS:
            payload["effort"] = implicit_label
        return payload

    payload: Dict[str, Any] = {"enabled": True}

    match = re.search(r"(\d+)\s*k", suffix)
    if match:
        budget_k = int(match.group(1))
        payload["max_tokens"] = budget_k * 1024
        return payload

    for level in OPENROUTER_REASONING_EFFORT_LEVELS:
        if level in suffix:
            payload["effort"] = level
            return payload

    return payload


def build_verbosity(model_name: str) -> Optional[str]:
    """Return a top-level `verbosity` value for adaptive-only models.

    For Claude Opus 4.7 and similar adaptive-only models, `:thinking-<level>`
    suffixes steer thinking depth via OpenRouter's `verbosity` field (which
    maps to Anthropic's `output_config.effort`). Returns None when no verbosity
    should be sent.
    """
    base_model, suffix = split_model_suffix(model_name, lower_suffix=True)

    if not _is_adaptive_only(base_model):
        return None
    if "thinking" not in suffix:
        return None

    return extract_word_boundary_reasoning_effort(suffix, OPENROUTER_VERBOSITY_LEVELS)
