import datetime
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from openai import OpenAI

from lisanbench.providers.common import extract_openai_usage_tokens
from lisanbench.providers.types import CompletionResult
from lisanbench.model_catalog import OPENAI_MODEL_MAPPING, get_model_service_tier
from lisanbench.model_names import OPENAI_REASONING_EFFORT_LEVELS, extract_reasoning_effort

if TYPE_CHECKING:
    from lisanbench.completions import CompletionAPI


logger = logging.getLogger(__name__)

OPENAI_FLEX_TIMEOUT_SECONDS = 3600.0


def _extract_error_message(error: Any) -> str:
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message
        return json.dumps(error, ensure_ascii=False)
    if isinstance(error, str) and error.strip():
        return error
    return "unknown_error"


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
                continue

            if not isinstance(part, dict):
                continue

            text_value = part.get("text")
            if isinstance(text_value, str):
                texts.append(text_value)
                continue

            inner_text = part.get("content")
            if isinstance(inner_text, str):
                texts.append(inner_text)

        if texts:
            return "".join(texts)

    if content is None:
        return ""

    return json.dumps(content, ensure_ascii=False)


def _to_plain_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        dumped = to_dict()
        if isinstance(dumped, dict):
            return dumped
    return {}


def build_openai_chat_request(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    suffix: str,
    service_tier: str | None = None,
) -> Dict[str, Any]:
    request_data: Dict[str, Any] = {
        "model": model,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }

    reasoning_effort = extract_reasoning_effort(suffix, OPENAI_REASONING_EFFORT_LEVELS)
    if reasoning_effort:
        request_data["reasoning_effort"] = reasoning_effort

    if service_tier:
        request_data["service_tier"] = service_tier

    return request_data


def parse_openai_chat_completion(
    response_data: Dict[str, Any],
) -> Tuple[str, str, int, int, int]:
    choices = response_data.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(first_choice, dict):
        first_choice = {}

    message = first_choice.get("message")
    if not isinstance(message, dict):
        message = {}

    usage_tokens = extract_openai_usage_tokens(response_data.get("usage"))

    return (
        _extract_text_content(message.get("content")),
        str(response_data.get("id") or ""),
        usage_tokens.input_tokens,
        usage_tokens.output_tokens,
        usage_tokens.reasoning_tokens,
    )


def call_openai_api(
    api: "CompletionAPI",
    model_name: str,
    prompt: str,
) -> CompletionResult:
    """Call the direct OpenAI Chat Completions API for one prompt."""
    model, suffix = api._resolve_provider_model(
        "openai", model_name, OPENAI_MODEL_MAPPING, allow_fallback=False
    )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    request_data = build_openai_chat_request(
        model=model,
        prompt=prompt,
        max_tokens=api.max_tokens,
        temperature=api.temperature,
        suffix=suffix,
        service_tier=get_model_service_tier(model_name),
    )
    request_payload_for_logging = dict(request_data, model=model_name)

    openai_client = OpenAI(api_key=openai_api_key, timeout=OPENAI_FLEX_TIMEOUT_SECONDS)
    try:
        completion = openai_client.chat.completions.create(
            **request_data,
            timeout=OPENAI_FLEX_TIMEOUT_SECONDS,
        )
    except Exception as exc:
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
        raise Exception(f"OpenAI API error: {exc}") from exc

    response_data = _to_plain_dict(completion)
    response_text, generation_id, input_tokens, output_tokens, reasoning_tokens = (
        parse_openai_chat_completion(response_data)
    )

    api._log_raw_interaction(
        model_name=model_name,
        prompt=prompt,
        request_payload=request_payload_for_logging,
        generation_id=generation_id or "no-id",
        response_payload=response_data,
    )

    cost_usd = api.estimate_billed_cost_usd(
        "openai",
        model_name,
        input_tokens,
        output_tokens,
    )

    # OpenAI response IDs are not OpenRouter generation IDs. Keep this empty
    # so final OpenRouter cost reconciliation only queries OpenRouter rows.
    return CompletionResult(
        text=response_text,
        cost_usd=cost_usd,
        generation_id="",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _download_jsonl_records(
    openai_client: OpenAI,
    file_id: str,
    output_path: str,
) -> List[Dict[str, Any]]:
    file_bytes = openai_client.files.content(file_id).content

    with open(output_path, "wb") as file:
        file.write(file_bytes)

    records: List[Dict[str, Any]] = []
    with open(output_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed OpenAI batch JSONL line %d in %s.",
                    line_number,
                    output_path,
                )
                continue

            if not isinstance(parsed, dict):
                logger.warning(
                    "Skipping non-object OpenAI batch record on line %d in %s.",
                    line_number,
                    output_path,
                )
                continue

            records.append(parsed)

    return records


def _parse_batch_record(record: Dict[str, Any]) -> Dict[str, Any]:
    custom_id = str(record.get("custom_id") or "")
    if not custom_id:
        raise ValueError(f"Batch record missing custom_id: {record}")

    error = record.get("error")
    if error:
        return {
            "custom_id": custom_id,
            "text": f"[ERROR] {_extract_error_message(error)}",
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "generation_id": "",
        }

    response = record.get("response")
    if not isinstance(response, dict):
        return {
            "custom_id": custom_id,
            "text": "[ERROR] Missing response payload in OpenAI batch result.",
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "generation_id": "",
        }

    status_code = response.get("status_code")
    body = response.get("body") if isinstance(response.get("body"), dict) else {}
    if status_code != 200:
        body_error = body.get("error")
        message = _extract_error_message(body_error) if body_error else f"HTTP {status_code}"
        return {
            "custom_id": custom_id,
            "text": f"[ERROR] {message}",
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "generation_id": "",
        }

    text, generation_id, input_tokens, output_tokens, reasoning_tokens = (
        parse_openai_chat_completion(body)
    )

    return {
        "custom_id": custom_id,
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "generation_id": generation_id,
    }


def call_openai_batch_api(
    api: "CompletionAPI",
    model_name: str,
    prompts: List[str],
) -> List[CompletionResult]:
    """Call OpenAI API with batch processing."""
    if not prompts:
        return []

    model, suffix = api._resolve_provider_model(
        "openai", model_name, OPENAI_MODEL_MAPPING, allow_fallback=False
    )

    output_dir = "openai_batches"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = str(model).replace("/", "_")
    filename = f"{timestamp}_{safe_model}.jsonl"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            body = build_openai_chat_request(
                model=model,
                prompt=prompt,
                max_tokens=api.max_tokens,
                temperature=api.temperature,
                suffix=suffix,
            )

            entry = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(entry) + "\n")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    openai_client = OpenAI(api_key=openai_api_key, timeout=3600)

    with open(filepath, "rb") as batch_input_file:
        batch_file = openai_client.files.create(
            file=batch_input_file,
            purpose="batch",
        )

    batch_job = openai_client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    terminal_failure_states = {"failed", "expired", "cancelled"}
    while True:
        results_response = openai_client.batches.retrieve(batch_job.id)
        status = getattr(results_response, "status", None)
        if status == "completed":
            break
        if status in terminal_failure_states:
            raise Exception(
                f"OpenAI batch failed with status='{status}' and errors={getattr(results_response, 'errors', None)!r}"
            )
        time.sleep(30)

    output_records: List[Dict[str, Any]] = []
    output_file_id = getattr(results_response, "output_file_id", None)
    if output_file_id:
        result_file_name = f"{output_dir}/batch_result_{timestamp}_{safe_model}.jsonl"
        output_records = _download_jsonl_records(
            openai_client,
            output_file_id,
            result_file_name,
        )

    error_records: List[Dict[str, Any]] = []
    error_file_id = getattr(results_response, "error_file_id", None)
    if error_file_id:
        error_file_name = f"{output_dir}/batch_error_{timestamp}_{safe_model}.jsonl"
        error_records = _download_jsonl_records(
            openai_client,
            error_file_id,
            error_file_name,
        )

    if not output_records and not error_records:
        raise Exception(
            "OpenAI batch completed but neither output_file_id nor error_file_id produced any records."
        )

    records_by_key: Dict[str, Dict[str, Any]] = {}
    for record in output_records + error_records:
        parsed = _parse_batch_record(record)
        records_by_key[parsed["custom_id"]] = parsed

    outputs: List[CompletionResult] = []
    for i in range(len(prompts)):
        key = f"request-{i}"
        record = records_by_key.get(
            key,
            {
                "text": f"[ERROR] Missing OpenAI batch result for {key}.",
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "generation_id": "",
            },
        )
        response_text = str(record["text"])
        input_tokens = int(record["input_tokens"])
        output_tokens = int(record["output_tokens"])
        reasoning_tokens = int(record["reasoning_tokens"])
        cost_usd = api.estimate_billed_cost_usd(
            "openai",
            model_name,
            input_tokens,
            output_tokens,
            batch=True,
        )
        # OpenAI response IDs are not OpenRouter generation IDs. Keep this empty
        # so final OpenRouter cost reconciliation only queries OpenRouter rows.
        generation_id = ""
        outputs.append(
            CompletionResult(
                text=response_text,
                cost_usd=cost_usd,
                generation_id=generation_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
            )
        )

    return outputs
