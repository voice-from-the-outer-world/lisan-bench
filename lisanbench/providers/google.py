import datetime
import json
import os
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from google.genai.types import HttpOptions

from lisanbench.providers.types import CompletionResult
from lisanbench.model_catalog import GOOGLE_MODEL_MAPPING, get_model_service_tier
from lisanbench.model_names import GOOGLE_THINKING_LEVELS, extract_reasoning_effort

if TYPE_CHECKING:
    from lisanbench.completions import CompletionAPI


GOOGLE_FLEX_TIMEOUT_MS = 3600 * 1000


def _extract_text_from_payload(payload: Dict[str, Any]) -> str:
    try:
        candidates = payload.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            texts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
            if texts:
                return "".join(texts)
        if isinstance(payload.get("text"), str):
            return payload["text"]
    except Exception:
        pass
    return json.dumps(payload, ensure_ascii=False)


def _extract_usage_from_payload(payload: Dict[str, Any]) -> Tuple[int, int, int]:
    usage = payload.get("usage_metadata") or payload.get("usageMetadata") or {}
    if not isinstance(usage, dict):
        usage = {}

    input_tokens = usage.get("prompt_token_count", usage.get("promptTokenCount", 0))
    output_tokens = usage.get(
        "candidates_token_count",
        usage.get("candidatesTokenCount", 0),
    )
    total_tokens = usage.get("total_token_count", usage.get("totalTokenCount", 0))
    reasoning_tokens = usage.get(
        "thoughts_token_count",
        usage.get("thoughtsTokenCount", 0),
    )

    if not output_tokens and total_tokens and input_tokens:
        # Some payload variants only include total tokens; infer visible output tokens.
        output_tokens = max(total_tokens - input_tokens - int(reasoning_tokens or 0), 0)

    return int(input_tokens or 0), int(output_tokens or 0), int(reasoning_tokens or 0)


def _response_to_dict(response: Any) -> Dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json", exclude_none=True)
    if isinstance(response, dict):
        return response
    try:
        return dict(response)
    except Exception:
        return {}


def _build_google_client_kwargs(api_key: str, service_tier: Optional[str]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if service_tier == "flex":
        kwargs["http_options"] = HttpOptions(timeout=GOOGLE_FLEX_TIMEOUT_MS)
    return kwargs


def _google_thinking_levels_for_model(provider_model: str) -> Tuple[str, ...]:
    model_l = provider_model.lower()
    if (
        model_l.startswith("gemini-3.1-pro")
        or model_l.startswith("gemini-3-pro")
        or model_l.startswith("gemini-3.0-pro")
    ):
        return ("low", "high")
    if model_l.startswith("gemini-3-flash") or model_l.startswith("gemini-3.5-flash"):
        return GOOGLE_THINKING_LEVELS
    return GOOGLE_THINKING_LEVELS


def _extract_google_thinking_level(provider_model: str, suffix: str) -> Optional[str]:
    return extract_reasoning_effort(suffix, _google_thinking_levels_for_model(provider_model))


def _extract_google_thinking_budget(suffix: str) -> Optional[int]:
    suffix_l = suffix.lower()
    match = re.search(r"thinking-(\d+)k\b", suffix_l)
    if match:
        return int(match.group(1)) * 1024
    if "thinking-none" in suffix_l or "thinking-off" in suffix_l:
        return 0
    if "thinking-dynamic" in suffix_l:
        return -1
    return None


def _build_google_generation_config(
    temperature: float,
    max_tokens: int,
    provider_model: str,
    suffix: str,
    service_tier: Optional[str] = None,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if service_tier:
        config["service_tier"] = service_tier

    if provider_model.startswith("gemini-3"):
        thinking_level = _extract_google_thinking_level(provider_model, suffix)
        if thinking_level:
            config["thinking_config"] = {"thinking_level": thinking_level}
    elif provider_model.startswith("gemini-2.5"):
        thinking_budget = _extract_google_thinking_budget(suffix)
        if thinking_budget is not None:
            config["thinking_config"] = {"thinking_budget": thinking_budget}

    return config


def call_google_batch_api(
    api: "CompletionAPI",
    model_name: str,
    prompts: List[str],
) -> List[CompletionResult]:
    """Call Google API with batch processing."""
    if not prompts:
        return []

    model, suffix = api._resolve_provider_model(
        "google", model_name, GOOGLE_MODEL_MAPPING, allow_fallback=False
    )

    output_dir = "google_batches"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = f"{timestamp}_{model.replace('/', '_')}.jsonl"
    input_filepath = os.path.join(output_dir, input_filename)

    generation_config = _build_google_generation_config(
        api.temperature,
        api.max_tokens,
        model,
        suffix,
    )

    with open(input_filepath, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            entry = {
                "key": f"request-{i}",
                "request": {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generation_config": generation_config,
                },
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    google_client = genai.Client(
        api_key=google_api_key,
        http_options=HttpOptions(timeout=3600 * 1000),
    )

    uploaded_file = google_client.files.upload(
        file=input_filepath,
        config=types.UploadFileConfig(
            display_name=f"batch-input-{timestamp}",
            mime_type="jsonl",
        ),
    )

    batch_job = google_client.batches.create(
        model=model,
        src=uploaded_file.name,
        config={"display_name": f"file-upload-job-{timestamp}"},
    )

    completed_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    while True:
        job = google_client.batches.get(name=batch_job.name)
        if job.state.name in completed_states:
            break
        time.sleep(30)

    if job.state.name != "JOB_STATE_SUCCEEDED":
        err_msg = getattr(job, "error", None)
        raise Exception(f"Gemini batch failed: {job.state.name}. {err_msg or ''}")

    if not (job.dest and getattr(job.dest, "file_name", None)):
        raise Exception("Gemini batch succeeded but no result file was returned.")

    result_bytes = google_client.files.download(file=job.dest.file_name)

    result_file_name = os.path.join(
        output_dir, f"batch_result_{timestamp}_{model.replace('/', '_')}.jsonl"
    )
    with open(result_file_name, "wb") as f:
        f.write(result_bytes)

    responses_by_key: Dict[str, Dict[str, int | str]] = {}
    with open(result_file_name, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("key")
            payload = obj.get("response", obj)

            if isinstance(payload, dict) and "error" in payload:
                msg = payload["error"].get("message") or json.dumps(payload["error"], ensure_ascii=False)
                responses_by_key[key or f"request-{len(responses_by_key)}"] = {
                    "text": f"[ERROR] {msg}",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                }
                continue

            input_tokens, output_tokens, reasoning_tokens = _extract_usage_from_payload(
                payload if isinstance(payload, dict) else {}
            )
            responses_by_key[key or f"request-{len(responses_by_key)}"] = {
                "text": _extract_text_from_payload(payload),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
            }

    outputs: List[CompletionResult] = []
    for i in range(len(prompts)):
        rec = responses_by_key.get(
            f"request-{i}",
            {
                "text": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
            },
        )
        response_text = str(rec["text"])
        input_tokens = int(rec["input_tokens"])
        output_tokens = int(rec["output_tokens"])
        reasoning_tokens = int(rec["reasoning_tokens"])
        billable_output_tokens = output_tokens + reasoning_tokens
        cost_usd = api.estimate_billed_cost_usd(
            "google",
            model_name,
            input_tokens,
            billable_output_tokens,
            batch=True,
        )
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


def call_google_aistudio(
    api: "CompletionAPI",
    model_name: str,
    prompt: str,
) -> CompletionResult:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    model_id, suffix = api._resolve_provider_model(
        "google",
        model_name,
        GOOGLE_MODEL_MAPPING,
        allow_fallback=False,
    )
    service_tier = get_model_service_tier(model_name)
    client = genai.Client(**_build_google_client_kwargs(google_api_key, service_tier))
    request_config = _build_google_generation_config(
        api.temperature,
        api.max_tokens,
        model_id,
        suffix,
        service_tier=service_tier,
    )

    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(**request_config),
    )

    response_payload = _response_to_dict(response)
    response_text = _extract_text_from_payload(response_payload)
    input_tokens, output_tokens, reasoning_tokens = _extract_usage_from_payload(
        response_payload
    )
    billable_output_tokens = output_tokens + reasoning_tokens
    preliminary_cost = api.estimate_billed_cost_usd(
        "google",
        model_name,
        input_tokens,
        billable_output_tokens,
    )
    response_id = str(response_payload.get("response_id") or "no-id")

    api._log_raw_interaction(
        model_name=model_name,
        prompt=prompt,
        request_payload={
            "model": model_id,
            "config": request_config,
        },
        generation_id=response_id,
        response_payload=response_payload,
        raw_response_text=response_text,
    )

    return CompletionResult(
        text=response_text,
        cost_usd=preliminary_cost,
        generation_id="",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )
