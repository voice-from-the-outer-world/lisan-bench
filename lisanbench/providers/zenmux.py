import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import requests

from lisanbench.providers.common import (
    append_stream_event,
    extract_openai_usage_tokens,
    iter_sse_events,
    truncate as _truncate,
)
from lisanbench.providers.types import CompletionResult
from lisanbench.model_catalog import ZENMUX_MODEL_MAPPING
from lisanbench.model_names import ZENMUX_REASONING_EFFORT_LEVELS, extract_reasoning_effort, split_model_suffix

if TYPE_CHECKING:
    from lisanbench.completions import CompletionAPI


DEFAULT_ZENMUX_API_BASE = "https://zenmux.ai/api/v1"


def _resolve_zenmux_api_base() -> str:
    configured = os.getenv("ZENMUX_API_BASE", DEFAULT_ZENMUX_API_BASE)
    trimmed = configured.strip() if isinstance(configured, str) else ""
    return trimmed or DEFAULT_ZENMUX_API_BASE


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _should_expose_reasoning_text() -> bool:
    return _env_flag("ZENMUX_INCLUDE_REASONING_TEXT", default=False)


def _build_reasoning_payload(model_name: str) -> Optional[Dict[str, Any]]:
    """Return the appropriate reasoning payload for models that require it."""
    payload: Dict[str, Any] = {"enabled": True}
    if not _should_expose_reasoning_text():
        # Benchmarks only consume the final answer. Hide verbose reasoning by
        # default so reasoning-heavy models do not spend their visible output
        # budget on chain-of-thought.
        payload["exclude"] = True

    _, suffix = split_model_suffix(model_name, lower_suffix=True)

    if "thinking" not in suffix:
        return {"enabled": False}

    match = re.search(r"(\d+)\s*k", suffix)
    if match:
        budget_k = int(match.group(1))
        payload["max_tokens"] = budget_k * 1024
        return payload

    effort = extract_reasoning_effort(suffix, ZENMUX_REASONING_EFFORT_LEVELS)
    if effort:
        payload["effort"] = effort
        return payload

    return payload


def _extract_reasoning_text_from_block(api: "CompletionAPI", block: Any) -> str:
    if not isinstance(block, dict):
        return ""
    for key in ("reasoning_details", "reasoning_content", "reasoning"):
        text = api._extract_text_block(block.get(key))
        if text:
            return text
    return ""


def _extract_reasoning_text_from_choice(
    api: "CompletionAPI",
    choice: Dict[str, Any],
) -> str:
    if not isinstance(choice, dict):
        return ""
    for key in ("delta", "message"):
        text = _extract_reasoning_text_from_block(api, choice.get(key))
        if text:
            return text
    return ""


def _consume_json_response(
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
            f"ZenMux API error: HTTP {response.status_code} {response.reason}{detail_suffix}"
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
            f"ZenMux API error: Failed to decode JSON response: {exc}"
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

    choices = completion_data.get("choices") if isinstance(completion_data, dict) else None
    choice: Dict[str, Any] = choices[0] if isinstance(choices, list) and choices else {}
    response_content = api._extract_text_from_choice(
        choice,
        has_existing_chunks=False,
    ).strip()
    if not response_content and isinstance(choice, dict):
        response_content = api._extract_text_block(choice.get("message")).strip()
    if not response_content:
        response_content = ""

    usage = completion_data.get("usage") or {}
    usage_tokens = extract_openai_usage_tokens(usage)

    cost_usd = api._estimate_cost_usd(
        "zenmux",
        model_name,
        usage_tokens.input_tokens,
        usage_tokens.output_tokens,
    )

    return CompletionResult(
        text=response_content,
        cost_usd=cost_usd,
        generation_id="",
        input_tokens=usage_tokens.input_tokens,
        output_tokens=usage_tokens.output_tokens,
        reasoning_tokens=usage_tokens.reasoning_tokens,
    )


def _consume_streaming_response(
    api: "CompletionAPI",
    response: requests.Response,
    model_name: str,
    prompt: str,
    request_payload: Dict[str, Any],
) -> CompletionResult:
    if not response.ok:
        error_body = response.text
        detailed_error = None
        try:
            err_json = json.loads(error_body)
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
            "body_preview": error_body[:700],
            "parsed_error": detailed_error,
        }
        api._log_raw_interaction(
            model_name=model_name,
            prompt=prompt,
            request_payload=request_payload,
            raw_response_text=error_body,
            error=error_info,
        )
        response.close()
        detail_suffix = f" ({detailed_error})" if detailed_error else ""
        raise Exception(
            f"ZenMux API error: HTTP {response.status_code} {response.reason}{detail_suffix}"
        )

    response_chunks: List[str] = []
    reasoning_chunks: List[str] = []
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
                    err_msg = error_block.get("message", "unknown error") if isinstance(error_block, dict) else str(error_block)
                    partial_response = "".join(response_chunks).strip()
                    api._log_raw_interaction(
                        model_name=model_name,
                        prompt=prompt,
                        request_payload=request_payload,
                        generation_id=generation_id,
                        raw_response_text=partial_response,
                        error=error_block if isinstance(error_block, dict) else {"message": str(error_block)},
                        stream_events=stream_events,
                    )
                    raise Exception(f"ZenMux streaming error: {err_msg}")

                choices = parsed_event.get("choices") or []
            else:
                choices = []
            if choices:
                chunk = api._extract_text_from_choice(
                    choices[0],
                    has_existing_chunks=bool(response_chunks),
                )
                if chunk:
                    response_chunks.append(chunk)
                reasoning_chunk = _extract_reasoning_text_from_choice(api, choices[0])
                if reasoning_chunk:
                    reasoning_chunks.append(reasoning_chunk)
    finally:
        response.close()

    response_text = "".join(response_chunks).strip()
    reasoning_text = "".join(reasoning_chunks).strip()
    if not response_text:
        error_message = "Streaming response yielded no final text"
        if reasoning_text:
            error_message = "Streaming response yielded reasoning but no final text"
        error_info = {
            "message": error_message,
            "logged_events": len(stream_events),
        }
        if reasoning_text:
            error_info["reasoning_text_preview"] = reasoning_text[:500]
        api._log_raw_interaction(
            model_name=model_name,
            prompt=prompt,
            request_payload=request_payload,
            generation_id=generation_id,
            response_payload={"usage": usage} if usage else None,
            error=error_info,
            stream_events=stream_events,
        )
        raise Exception(f"ZenMux API error: {error_message}")

    api._log_raw_interaction(
        model_name=model_name,
        prompt=prompt,
        request_payload=request_payload,
        generation_id=generation_id,
        response_payload={
            "usage": usage,
            "reasoning_text_preview": reasoning_text[:500],
        },
        raw_response_text=response_text,
        stream_events=stream_events,
    )

    usage_tokens = extract_openai_usage_tokens(usage)

    cost_usd = api._estimate_cost_usd(
        "zenmux",
        model_name,
        usage_tokens.input_tokens,
        usage_tokens.output_tokens,
    )

    return CompletionResult(
        text=response_text,
        cost_usd=cost_usd,
        generation_id="",
        input_tokens=usage_tokens.input_tokens,
        output_tokens=usage_tokens.output_tokens,
        reasoning_tokens=usage_tokens.reasoning_tokens,
    )


def call_zenmux_api(
    api: "CompletionAPI",
    model_name: str,
    prompt: str,
) -> CompletionResult:
    """Call ZenMux API for models using requests (OpenAI-compatible)."""
    zenmux_api_key = api._require_zenmux_key()

    provider_model, suffix = api._resolve_provider_model(
        "zenmux",
        model_name,
        ZENMUX_MODEL_MAPPING,
    )

    base_url = _resolve_zenmux_api_base().rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {zenmux_api_key}",
        "Content-Type": "application/json",
    }

    request_data: Dict[str, Any] = {
        "model": provider_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": api.temperature,
        "max_completion_tokens": api.max_tokens,
    }

    if api.stream_responses:
        request_data["stream"] = True
        request_data["stream_options"] = {"include_usage": True}

    reasoning_payload = _build_reasoning_payload(model_name)
    if reasoning_payload and reasoning_payload.get("enabled") is not False:
        request_data["reasoning"] = reasoning_payload

    request_payload_for_logging = dict(
        request_data,
        model=model_name,
        resolved_model=provider_model,
    )

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
            "resolved_model": provider_model,
        }
        api._log_raw_interaction(
            model_name=model_name,
            prompt=prompt,
            request_payload=request_payload_for_logging,
            error=error_info,
        )
        raise Exception(f"ZenMux API error: {exc}") from exc

    if api.stream_responses:
        return _consume_streaming_response(
            api,
            response,
            model_name,
            prompt,
            request_payload_for_logging,
        )
    return _consume_json_response(
        api,
        response,
        model_name,
        prompt,
        request_payload_for_logging,
    )
