import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import requests

from lisanbench.providers.common import (
    append_stream_event,
    extract_error_block as _extract_error_block,
    format_error_detail as _format_error_detail,
    iter_sse_events,
    to_int as _to_int,
    truncate as _truncate,
)
from lisanbench.providers.types import CompletionResult
from lisanbench.model_catalog import ZAI_MODEL_MAPPING

if TYPE_CHECKING:
    from lisanbench.completions import CompletionAPI


DEFAULT_ZAI_API_BASE = "https://api.z.ai/api/paas/v4"
ZAI_MAX_TOKENS_LIMIT = 131072


def _coerce_token_value(value: Any) -> int:
    if isinstance(value, (int, float, str, bool)):
        return _to_int(value)
    if isinstance(value, list):
        coerced = [_coerce_token_value(item) for item in value]
        return max(coerced) if coerced else 0
    if isinstance(value, dict):
        for key in (
            "reasoning_tokens",
            "thinking_tokens",
            "thought_tokens",
            "token_count",
            "tokens",
            "count",
            "value",
            "total",
        ):
            if key in value:
                coerced = _coerce_token_value(value.get(key))
                if coerced > 0:
                    return coerced
        coerced = [_coerce_token_value(item) for item in value.values()]
        return max(coerced) if coerced else 0
    return 0


def _extract_usage_token_counts(usage: Any) -> Tuple[int, int, int]:
    if not isinstance(usage, dict):
        usage = {}

    def _first_positive(container: Dict[str, Any], keys: Tuple[str, ...]) -> int:
        for key in keys:
            if key in container:
                value = _coerce_token_value(container.get(key))
                if value > 0:
                    return value
        return 0

    input_tokens = _first_positive(
        usage,
        (
            "prompt_tokens",
            "input_tokens",
            "promptTokenCount",
            "prompt_token_count",
        ),
    )
    output_tokens = _first_positive(
        usage,
        (
            "completion_tokens",
            "output_tokens",
            "completionTokenCount",
            "completion_token_count",
            "outputTokenCount",
            "output_token_count",
            "candidatesTokenCount",
            "candidates_token_count",
        ),
    )

    reasoning_tokens = _first_positive(
        usage,
        (
            "reasoning_tokens",
            "thinking_tokens",
            "thought_tokens",
            "reasoningTokenCount",
            "reasoning_token_count",
            "thinkingTokenCount",
            "thinking_token_count",
        ),
    )

    for details_key in (
        "completion_tokens_details",
        "completionTokenDetails",
        "completion_token_details",
        "output_tokens_details",
        "outputTokenDetails",
        "output_token_details",
        "tokens_details",
        "token_details",
    ):
        details = usage.get(details_key)
        if not isinstance(details, dict):
            continue
        if reasoning_tokens <= 0:
            reasoning_tokens = _first_positive(
                details,
                (
                    "reasoning_tokens",
                    "thinking_tokens",
                    "thought_tokens",
                    "reasoningTokenCount",
                    "reasoning_token_count",
                    "thinkingTokenCount",
                    "thinking_token_count",
                ),
            )

    if output_tokens <= 0:
        total_tokens = _first_positive(
            usage,
            (
                "total_tokens",
                "totalTokenCount",
                "total_token_count",
            ),
        )
        if total_tokens > 0 and input_tokens > 0:
            output_tokens = max(total_tokens - input_tokens, 0)

    return input_tokens, output_tokens, reasoning_tokens


def _resolve_zai_api_base() -> str:
    configured = os.getenv("ZAI_API_BASE", DEFAULT_ZAI_API_BASE)
    trimmed = configured.strip() if isinstance(configured, str) else ""
    return trimmed or DEFAULT_ZAI_API_BASE


def _is_invalid_parameter_error_response(response: requests.Response) -> bool:
    if response.status_code != 400:
        return False
    try:
        payload = response.json()
    except Exception:
        return False
    err = _extract_error_block(payload)
    code = str(err.get("code") or "").strip().lower()
    message = str(err.get("message") or "").strip().lower()
    if code == "1210":
        return True
    return "invalid api parameter" in message


def _extract_reasoning_text_from_choice(api: "CompletionAPI", choice: Dict[str, Any]) -> str:
    if not isinstance(choice, dict):
        return ""
    for key in ("delta", "message"):
        block = choice.get(key)
        if isinstance(block, dict):
            reasoning_content = block.get("reasoning_content")
            if reasoning_content is None:
                reasoning_content = block.get("reasoning")
            text = api._extract_text_block(reasoning_content)
            if text:
                return text
    return ""


def _should_use_tokenizer_fallback() -> bool:
    raw = os.getenv("ZAI_TOKENIZER_FALLBACK", "1")
    if raw is None:
        return True
    normalized = str(raw).strip().lower()
    return normalized not in {"0", "false", "no", "off"}


def _count_tokens_with_tokenizer(
    api: "CompletionAPI",
    resolved_model: str,
    text: str,
) -> int:
    if not text or not resolved_model:
        return 0
    try:
        zai_api_key = api._require_zai_key()
    except Exception:
        return 0

    base_url = _resolve_zai_api_base().rstrip("/")
    url = f"{base_url}/tokenizer"
    headers = {
        "Authorization": f"Bearer {zai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": resolved_model,
        "messages": [{"role": "user", "content": text}],
    }

    response: Optional[requests.Response] = None
    try:
        session = api._get_session()
        response = session.post(
            url,
            headers=headers,
            json=payload,
            timeout=(30, 120),
        )
        if not response.ok:
            response.close()
            return 0
        data = response.json()
    except Exception:
        return 0
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

    entries = data.get("data") if isinstance(data, dict) else None
    usage = data.get("usage") if isinstance(data, dict) else None
    if isinstance(usage, dict):
        prompt_tokens = _coerce_token_value(usage.get("prompt_tokens"))
        if prompt_tokens > 0:
            return prompt_tokens

    if not isinstance(entries, list) or not entries:
        return 0
    first = entries[0]
    if not isinstance(first, dict):
        return 0

    token_ids = first.get("token_ids")
    if isinstance(token_ids, list):
        return len(token_ids)

    for key in ("token_count", "tokens", "count", "total"):
        if key in first:
            count = _coerce_token_value(first.get(key))
            if count > 0:
                return count
    return 0


def _resolve_reasoning_tokens(
    api: "CompletionAPI",
    *,
    reasoning_tokens_from_usage: int,
    output_tokens: int,
    reasoning_text: str,
    resolved_model: str,
) -> int:
    if reasoning_tokens_from_usage > 0:
        return reasoning_tokens_from_usage
    if not reasoning_text:
        return 0

    estimated = 0
    if _should_use_tokenizer_fallback():
        estimated = _count_tokens_with_tokenizer(api, resolved_model, reasoning_text)

    if estimated <= 0 and output_tokens > 0:
        # Conservative fallback when provider omits reasoning token accounting:
        # reasoning content exists and completion tokens are known.
        estimated = output_tokens

    if output_tokens > 0 and estimated > output_tokens:
        estimated = output_tokens
    return max(estimated, 0)


def _build_thinking_payload(suffix: str) -> Dict[str, Any]:
    suffix_l = suffix.lower()
    if "thinking" not in suffix_l:
        return {"type": "disabled"}

    payload: Dict[str, Any] = {"type": "enabled"}
    clear_false_flags = (
        "keep-thinking",
        "preserve-thinking",
        "clear-thinking-false",
        "clear_thinking_false",
        "clearthinkingfalse",
        "clear-thinking=0",
        "clear_thinking=0",
        "clearthinking=0",
    )
    clear_true_flags = (
        "clear-thinking-true",
        "clear_thinking_true",
        "clearthinkingtrue",
        "clear-thinking=1",
        "clear_thinking=1",
        "clearthinking=1",
    )
    if any(flag in suffix_l for flag in clear_false_flags):
        payload["clear_thinking"] = False
    elif any(flag in suffix_l for flag in clear_true_flags):
        payload["clear_thinking"] = True
    return payload


def _resolve_zai_max_tokens_limit(resolved_model: str) -> int:
    normalized = (resolved_model or "").strip().lower()
    if normalized.startswith("glm-4.5"):
        return 96 * 1024
    return ZAI_MAX_TOKENS_LIMIT


def _consume_zai_json_response(
    api: "CompletionAPI",
    response: requests.Response,
    model_name: str,
    prompt: str,
    request_payload: Dict[str, Any],
) -> CompletionResult:
    response_text = response.text
    if not response.ok:
        reason_text = str(response.reason or "").strip() or "Unknown"
        detail = ""
        err_block: Dict[str, Any] = {}
        try:
            parsed = response.json()
            err_block = _extract_error_block(parsed)
            detail = _format_error_detail(err_block) if err_block else ""
        except Exception:
            body_preview = response_text.strip()
            if body_preview:
                detail = f"body='{_truncate(body_preview)}'"

        error_info = {
            "status_code": response.status_code,
            "reason": reason_text,
            "body_preview": response_text[:700],
            "parsed_error": detail or None,
        }
        api._log_raw_interaction(
            model_name=model_name,
            prompt=prompt,
            request_payload=request_payload,
            raw_response_text=response_text,
            error=error_info,
        )
        response.close()
        detail_suffix = f" ({detail})" if detail else ""
        raise Exception(f"Z.AI API error: HTTP {response.status_code} {reason_text}{detail_suffix}")

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
        raise Exception(f"Z.AI API error: Failed to decode JSON response: {exc}") from exc
    finally:
        response.close()

    generation_id_for_log = (
        completion_data.get("id")
        if isinstance(completion_data, dict)
        else "no-id"
    ) or "no-id"

    api._log_raw_interaction(
        model_name=model_name,
        prompt=prompt,
        request_payload=request_payload,
        generation_id=generation_id_for_log,
        response_payload=completion_data if isinstance(completion_data, dict) else {},
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

    usage = completion_data.get("usage") if isinstance(completion_data, dict) else {}
    input_tokens, output_tokens, reasoning_tokens_from_usage = _extract_usage_token_counts(usage)
    resolved_model = str(request_payload.get("resolved_model") or "")
    reasoning_text = _extract_reasoning_text_from_choice(api, choice)
    reasoning_tokens = _resolve_reasoning_tokens(
        api,
        reasoning_tokens_from_usage=reasoning_tokens_from_usage,
        output_tokens=output_tokens,
        reasoning_text=reasoning_text,
        resolved_model=resolved_model,
    )

    cost_usd = api._estimate_cost_usd(
        "z-ai",
        model_name,
        input_tokens,
        output_tokens,
    )
    return CompletionResult(
        text=response_content,
        cost_usd=cost_usd,
        generation_id="",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _consume_zai_streaming_response(
    api: "CompletionAPI",
    response: requests.Response,
    model_name: str,
    prompt: str,
    request_payload: Dict[str, Any],
) -> CompletionResult:
    if not response.ok:
        error_body = response.text
        detail = ""
        try:
            parsed = json.loads(error_body)
            err_block = _extract_error_block(parsed)
            detail = _format_error_detail(err_block) if err_block else ""
        except Exception:
            pass

        error_info = {
            "status_code": response.status_code,
            "reason": response.reason,
            "body_preview": error_body[:700],
            "parsed_error": detail or None,
        }
        api._log_raw_interaction(
            model_name=model_name,
            prompt=prompt,
            request_payload=request_payload,
            raw_response_text=error_body,
            error=error_info,
        )
        response.close()
        detail_suffix = f" ({detail})" if detail else ""
        raise Exception(
            f"Z.AI API error: HTTP {response.status_code} {response.reason}{detail_suffix}"
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
            if not isinstance(parsed_event, dict):
                continue

            generation_id = parsed_event.get("id", generation_id)
            usage = parsed_event.get("usage") or usage

            error_block = _extract_error_block(parsed_event)
            if error_block:
                detail = _format_error_detail(error_block)
                partial_response = "".join(response_chunks).strip()
                api._log_raw_interaction(
                    model_name=model_name,
                    prompt=prompt,
                    request_payload=request_payload,
                    generation_id=generation_id,
                    raw_response_text=partial_response,
                    error=error_block,
                    stream_events=stream_events,
                )
                raise Exception(f"Z.AI streaming error: {detail}")

            choices = parsed_event.get("choices") or []
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
    if not response_text and not reasoning_text:
        error_info = {
            "message": "Streaming response yielded no text or reasoning",
            "logged_events": len(stream_events),
        }
        api._log_raw_interaction(
            model_name=model_name,
            prompt=prompt,
            request_payload=request_payload,
            error=error_info,
            stream_events=stream_events,
        )
        raise Exception("Z.AI API error: Empty streaming response")

    api._log_raw_interaction(
        model_name=model_name,
        prompt=prompt,
        request_payload=request_payload,
        generation_id=generation_id,
        response_payload={"usage": usage, "reasoning_text_preview": reasoning_text[:500]},
        raw_response_text=response_text,
        stream_events=stream_events,
    )

    input_tokens, output_tokens, reasoning_tokens_from_usage = _extract_usage_token_counts(usage)
    resolved_model = str(request_payload.get("resolved_model") or "")
    reasoning_tokens = _resolve_reasoning_tokens(
        api,
        reasoning_tokens_from_usage=reasoning_tokens_from_usage,
        output_tokens=output_tokens,
        reasoning_text=reasoning_text,
        resolved_model=resolved_model,
    )

    cost_usd = api._estimate_cost_usd(
        "z-ai",
        model_name,
        input_tokens,
        output_tokens,
    )
    return CompletionResult(
        text=response_text,
        cost_usd=cost_usd,
        generation_id="",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def call_zai_api(
    api: "CompletionAPI",
    model_name: str,
    prompt: str,
) -> CompletionResult:
    zai_api_key = api._require_zai_key()
    provider_model, suffix = api._resolve_provider_model(
        "z-ai",
        model_name,
        ZAI_MODEL_MAPPING,
    )

    session = api._get_session()
    base_url = _resolve_zai_api_base().rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {zai_api_key}",
        "Content-Type": "application/json",
    }

    thinking_payload = _build_thinking_payload(suffix)
    thinking_enabled = thinking_payload.get("type") == "enabled"

    request_data: Dict[str, Any] = {
        "model": provider_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": min(
            api.max_tokens,
            _resolve_zai_max_tokens_limit(provider_model),
        ),
        "stream": bool(api.stream_responses),
        "temperature": api.temperature,
    }

    # Only include thinking param when actually enabled; omit for non-thinking models
    if thinking_enabled:
        request_data["thinking"] = thinking_payload

    # Build fallback attempts for invalid-parameter retries
    request_attempts: List[Dict[str, Any]] = [dict(request_data)]

    # Attempt 2: drop temperature but preserve thinking/max_tokens.
    fallback = dict(request_data)
    fallback.pop("temperature", None)
    request_attempts.append(fallback)

    # Attempt 3: drop max_tokens too, while still preserving thinking if enabled.
    fallback = dict(fallback)
    fallback.pop("max_tokens", None)
    request_attempts.append(fallback)

    # Attempt 4: bare minimum payload.
    bare = {
        "model": provider_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": bool(api.stream_responses),
    }
    request_attempts.append(bare)

    for attempt_idx, request_attempt in enumerate(request_attempts):
        request_payload_for_logging = dict(
            request_attempt,
            model=model_name,
            resolved_model=provider_model,
        )
        try:
            response = session.post(
                url,
                headers=headers,
                json=request_attempt,
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
            raise Exception(f"Z.AI API error: {exc}") from exc

        if attempt_idx < len(request_attempts) - 1 and _is_invalid_parameter_error_response(response):
            error_info = {
                "status_code": response.status_code,
                "reason": response.reason,
                "body_preview": response.text[:700],
                "parsed_error": f"invalid_parameter_retry_attempt_{attempt_idx + 1}",
            }
            api._log_raw_interaction(
                model_name=model_name,
                prompt=prompt,
                request_payload=request_payload_for_logging,
                raw_response_text=response.text,
                error=error_info,
            )
            response.close()
            continue

        if api.stream_responses:
            return _consume_zai_streaming_response(
                api,
                response,
                model_name,
                prompt,
                request_payload_for_logging,
            )
        return _consume_zai_json_response(
            api,
            response,
            model_name,
            prompt,
            request_payload_for_logging,
        )

    raise Exception("Z.AI API error: Exhausted request retries.")
