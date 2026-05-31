import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

from lisanbench.providers.types import UsageTokens


STREAM_EVENT_LOG_LIMIT = 500


def truncate(value: Any, max_len: int = 300) -> str:
    text = str(value) if value is not None else ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def to_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def coerce_nonnegative_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return max(float(value), 0.0)
    if isinstance(value, str):
        try:
            return max(float(value), 0.0)
        except ValueError:
            return None
    return None


def extract_usage_cost(usage: Any) -> float:
    if not isinstance(usage, dict):
        return 0.0
    cost = coerce_nonnegative_float(usage.get("cost"))
    return cost if cost is not None else 0.0


def extract_error_block(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            return err
    return {}


def format_error_detail(error_block: Dict[str, Any]) -> str:
    parts: List[str] = []
    message = error_block.get("message")
    error_type = error_block.get("type")
    code = error_block.get("code")
    if message is not None:
        parts.append(f"message='{truncate(message)}'")
    if error_type is not None:
        parts.append(f"type='{truncate(error_type)}'")
    if code is not None:
        parts.append(f"code='{truncate(code)}'")
    return ", ".join(parts) if parts else "unknown error"


def parse_json_error_detail(response_text: str, response: Any = None) -> Optional[str]:
    try:
        payload = response.json() if response is not None else json.loads(response_text)
    except Exception:
        return None
    err_block = extract_error_block(payload)
    if not err_block:
        return None
    return format_error_detail(err_block)


def extract_openai_usage_tokens(usage: Any) -> UsageTokens:
    if not isinstance(usage, dict):
        usage = {}
    completion_tokens_details = usage.get("completion_tokens_details")
    if not isinstance(completion_tokens_details, dict):
        completion_tokens_details = {}
    return UsageTokens(
        input_tokens=to_int(usage.get("prompt_tokens")),
        output_tokens=to_int(usage.get("completion_tokens")),
        reasoning_tokens=to_int(completion_tokens_details.get("reasoning_tokens")),
    )


def append_stream_event(events: List[Dict[str, Any]], event: Dict[str, Any]) -> None:
    if len(events) < STREAM_EVENT_LOG_LIMIT:
        events.append(event)


def iter_sse_events(response: Any) -> Iterator[Tuple[str, Any]]:
    for raw_line in response.iter_lines(decode_unicode=True, chunk_size=8192):
        if not raw_line:
            continue
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(":"):
            continue
        if not stripped.startswith("data:"):
            yield "raw_line", stripped
            continue

        payload = stripped[5:].strip()
        if payload == "[DONE]":
            yield "done", None
            break

        try:
            parsed_event = json.loads(payload)
        except json.JSONDecodeError:
            yield "malformed", payload[:200]
            continue

        yield "event", parsed_event
