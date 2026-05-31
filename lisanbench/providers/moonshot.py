import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import requests

from lisanbench.providers.common import (
    append_stream_event,
    extract_error_block as _extract_error_block,
    iter_sse_events,
    to_int as _to_int,
    truncate as _truncate,
)
from lisanbench.providers.types import CompletionResult
from lisanbench.model_catalog import MOONSHOT_MODEL_MAPPING

if TYPE_CHECKING:
    from lisanbench.completions import CompletionAPI


DEFAULT_MOONSHOT_API_BASE = "https://api.moonshot.ai/v1"


def _resolve_moonshot_api_base() -> str:
    configured = os.getenv("MOONSHOT_API_BASE", DEFAULT_MOONSHOT_API_BASE)
    trimmed = configured.strip() if isinstance(configured, str) else ""
    return trimmed or DEFAULT_MOONSHOT_API_BASE


def _is_model_not_found_error(error_code: Any, error_type: Any, error_message: Any) -> bool:
    blob = " ".join(
        part
        for part in (
            str(error_code or "").lower(),
            str(error_type or "").lower(),
            str(error_message or "").lower(),
        )
        if part
    )
    hints = (
        "model_not_found",
        "invalid model",
        "unknown model",
        "model does not exist",
        "not found",
    )
    return any(h in blob for h in hints)


def _build_moonshot_model_candidates(provider_model: str, suffix: str) -> List[str]:
    # Deterministic behavior: use the explicit catalog mapping.
    return [provider_model]


def _resolve_moonshot_temperature(
    provider_model: str,
    suffix: str,
    requested_temperature: float,
) -> float:
    # Moonshot enforces temperature=0.6 only for non-thinking kimi-k2.5.
    if provider_model.startswith("kimi-k2.5") and "thinking" not in suffix.lower():
        return 0.6
    return requested_temperature


def _build_kimi25_thinking_payload(provider_model: str, suffix: str) -> Optional[Dict[str, Any]]:
    # Per mapping contract:
    # - moonshotai/kimi-k2.5 -> kimi-k2.5 with thinking explicitly disabled
    # - moonshotai/kimi-k2.5:thinking -> kimi-k2.5 with default thinking behavior
    if provider_model != "kimi-k2.5":
        return None
    if "thinking" in suffix.lower():
        return None
    return {"type": "disabled"}


def _consume_moonshot_streaming_response(
    api: "CompletionAPI",
    response: requests.Response,
    model_name: str,
    prompt: str,
    request_payload: Dict[str, Any],
    resolved_model: str,
) -> CompletionResult:
    if not response.ok:
        error_body = response.text
        detailed_error = ""
        reason_text = str(response.reason or "").strip()
        if not reason_text or reason_text.lower() == "<none>":
            reason_text = "Unknown"
        try:
            parsed = json.loads(error_body)
            err_block = _extract_error_block(parsed)
            err_message = err_block.get("message")
            err_type = err_block.get("type")
            err_code = err_block.get("code")
            detail_parts: List[str] = []
            if err_message is not None:
                detail_parts.append(f"message='{_truncate(err_message)}'")
            if err_type is not None:
                detail_parts.append(f"type='{_truncate(err_type)}'")
            if err_code is not None:
                detail_parts.append(f"code='{_truncate(err_code)}'")
            detailed_error = ", ".join(detail_parts)
        except Exception:
            body_preview = error_body.strip()
            if body_preview:
                if body_preview.lstrip().lower().startswith(
                    "<!doctype html"
                ) or body_preview.lstrip().lower().startswith("<html"):
                    detailed_error = "upstream_html_error"
                else:
                    detailed_error = f"body='{_truncate(body_preview)}'"

        error_info = {
            "status_code": response.status_code,
            "reason": reason_text,
            "body_preview": error_body[:700],
            "parsed_error": detailed_error or None,
            "resolved_model": resolved_model,
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
            f"Moonshot API error: HTTP {response.status_code} {reason_text}{detail_suffix}"
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
            if not isinstance(parsed_event, dict):
                continue

            generation_id = parsed_event.get("id", generation_id)
            event_usage = parsed_event.get("usage")
            if isinstance(event_usage, dict):
                usage = event_usage

            error_block = _extract_error_block(parsed_event)
            if error_block:
                err_msg = error_block.get("message", "unknown error")
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
                raise Exception(f"Moonshot streaming error: {err_msg}")

            choices = parsed_event.get("choices") or []
            if choices:
                choice_usage = choices[0].get("usage")
                if isinstance(choice_usage, dict):
                    usage = choice_usage
                chunk = api._extract_text_from_choice(
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
            "resolved_model": resolved_model,
        }
        api._log_raw_interaction(
            model_name=model_name,
            prompt=prompt,
            request_payload=request_payload,
            error=error_info,
            stream_events=stream_events,
        )
        raise Exception("Moonshot API error: Empty streaming response")

    api._log_raw_interaction(
        model_name=model_name,
        prompt=prompt,
        request_payload=request_payload,
        generation_id=generation_id,
        response_payload={"usage": usage},
        raw_response_text=response_text,
        stream_events=stream_events,
    )

    if not isinstance(usage, dict):
        usage = {}
    input_tokens = _to_int(usage.get("prompt_tokens"))
    output_tokens = _to_int(usage.get("completion_tokens"))
    reasoning_tokens = _to_int(
        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        if isinstance(usage.get("completion_tokens_details"), dict)
        else usage.get("reasoning_tokens")
    )
    if output_tokens <= 0:
        total_tokens = _to_int(usage.get("total_tokens"))
        if total_tokens > 0 and input_tokens > 0:
            output_tokens = max(total_tokens - input_tokens, 0)

    cost_usd = api._estimate_cost_usd(
        "moonshotai",
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


def call_moonshot_api(
    api: "CompletionAPI",
    model_name: str,
    prompt: str,
) -> CompletionResult:
    moonshot_api_key = api._require_moonshot_key()
    provider_model, suffix = api._resolve_provider_model(
        "moonshotai",
        model_name,
        MOONSHOT_MODEL_MAPPING,
        allow_fallback=False,
    )

    session = api._get_session()
    base_url = _resolve_moonshot_api_base().rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {moonshot_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    model_candidates = _build_moonshot_model_candidates(provider_model, suffix)
    last_error_detail: Optional[str] = None

    for idx, resolved_model in enumerate(model_candidates):
        request_data: Dict[str, Any] = {
            "model": resolved_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": _resolve_moonshot_temperature(
                resolved_model,
                suffix,
                api.temperature,
            ),
            "max_completion_tokens": api.max_tokens,
            "stream": api.stream_responses,
        }
        if api.stream_responses:
            request_data["stream_options"] = {"include_usage": True}
        thinking_payload = _build_kimi25_thinking_payload(resolved_model, suffix)
        if thinking_payload is not None:
            request_data["thinking"] = thinking_payload
        request_payload_for_logging = dict(
            request_data,
            model=model_name,
            resolved_model=resolved_model,
        )

        try:
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
                "resolved_model": resolved_model,
            }
            api._log_raw_interaction(
                model_name=model_name,
                prompt=prompt,
                request_payload=request_payload_for_logging,
                error=error_info,
            )
            raise Exception(f"Moonshot API error: {exc}") from exc

        if api.stream_responses:
            return _consume_moonshot_streaming_response(
                api,
                response,
                model_name,
                prompt,
                request_payload_for_logging,
                resolved_model,
            )

        response_text = response.text
        if not response.ok:
            detailed_error = ""
            reason_text = str(response.reason or "").strip()
            if not reason_text or reason_text.lower() == "<none>":
                reason_text = "Unknown"
            try:
                parsed = response.json()
                err_block = _extract_error_block(parsed)
                err_message = err_block.get("message")
                err_code = err_block.get("code")
                err_type = err_block.get("type")
                detail_parts: List[str] = []
                if err_message is not None:
                    detail_parts.append(f"message='{_truncate(err_message)}'")
                if err_type is not None:
                    detail_parts.append(f"type='{_truncate(err_type)}'")
                if err_code is not None:
                    detail_parts.append(f"code='{_truncate(err_code)}'")
                detailed_error = ", ".join(detail_parts)
            except Exception:
                err_block = {}
                err_message = None
                err_code = None
                err_type = None
                body_preview = response_text.strip()
                if body_preview:
                    if body_preview.lstrip().lower().startswith("<!doctype html") or body_preview.lstrip().lower().startswith("<html"):
                        detailed_error = "upstream_html_error"
                    else:
                        detailed_error = f"body='{_truncate(body_preview)}'"

            error_info = {
                "status_code": response.status_code,
                "reason": reason_text,
                "body_preview": response_text[:700],
                "parsed_error": detailed_error or None,
                "resolved_model": resolved_model,
            }
            api._log_raw_interaction(
                model_name=model_name,
                prompt=prompt,
                request_payload=request_payload_for_logging,
                raw_response_text=response_text,
                error=error_info,
            )
            response.close()

            # If suffix requested a thinking variant and candidate model is unsupported,
            # fall back to the non-thinking model before failing the request.
            if (
                idx < len(model_candidates) - 1
                and _is_model_not_found_error(err_code, err_type, err_message)
            ):
                last_error_detail = detailed_error or f"HTTP {response.status_code} {response.reason}"
                continue

            detail_suffix = f" ({detailed_error})" if detailed_error else ""
            raise Exception(
                f"Moonshot API error: HTTP {response.status_code} {reason_text}{detail_suffix}"
            )

        try:
            completion_data = response.json()
        except json.JSONDecodeError as exc:
            response.close()
            error_info = {
                "type": type(exc).__name__,
                "message": str(exc),
                "resolved_model": resolved_model,
            }
            api._log_raw_interaction(
                model_name=model_name,
                prompt=prompt,
                request_payload=request_payload_for_logging,
                raw_response_text=response_text,
                error=error_info,
            )
            raise Exception(
                f"Moonshot API error: Failed to decode JSON response: {exc}"
            ) from exc
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
            request_payload=request_payload_for_logging,
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
        if not isinstance(usage, dict):
            usage = {}
        input_tokens = _to_int(usage.get("prompt_tokens"))
        output_tokens = _to_int(usage.get("completion_tokens"))
        reasoning_tokens = _to_int(
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            if isinstance(usage.get("completion_tokens_details"), dict)
            else usage.get("reasoning_tokens")
        )
        if output_tokens <= 0:
            total_tokens = _to_int(usage.get("total_tokens"))
            if total_tokens > 0 and input_tokens > 0:
                output_tokens = max(total_tokens - input_tokens, 0)

        cost_usd = api._estimate_cost_usd(
            "moonshotai",
            model_name,
            input_tokens,
            output_tokens,
        )

        # Keep generation_id empty for non-OpenRouter providers so OpenRouter cost backfill
        # does not try to resolve it later.
        generation_id = ""
        return CompletionResult(
            text=response_content,
            cost_usd=cost_usd,
            generation_id=generation_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    detail = f" Last model selection error: {last_error_detail}" if last_error_detail else ""
    raise Exception(f"Moonshot API error: No usable model candidate succeeded.{detail}")
