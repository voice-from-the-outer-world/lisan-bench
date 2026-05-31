import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from lisanbench.providers.types import CompletionResult
from lisanbench.model_catalog import ANTHROPIC_MODEL_MAPPING
from lisanbench.model_names import ANTHROPIC_EFFORT_LEVELS, extract_word_boundary_reasoning_effort

if TYPE_CHECKING:
    from lisanbench.completions import CompletionAPI


logger = logging.getLogger(__name__)

_MANUAL_THINKING_SUFFIX = "thinking-16k"
_THINKING_BUDGET_TOKENS = 16_384

# Provider model ids that only accept adaptive thinking. Manual
# `thinking: {type: "enabled", budget_tokens: N}` returns 400 on these.
# For these models `:thinking-<level>` maps to `output_config.effort`.
_ADAPTIVE_ONLY_PROVIDER_IDS = frozenset({"claude-opus-4-7", "claude-opus-4-8"})

def _extract_adaptive_effort(suffix: str) -> Optional[str]:
    return extract_word_boundary_reasoning_effort(suffix, ANTHROPIC_EFFORT_LEVELS)

# Anthropic enforces model-specific `max_tokens` limits, but this runner keeps
# `MAX_TOKENS` user-configurable and lets the API be the source of truth. We
# only log when a configured value exceeds current documented limits.
_DOCUMENTED_ANTHROPIC_MAX_TOKENS_BY_MODEL: Dict[str, int] = {
    "claude-3-haiku-20240307": 4_096,
    "claude-3-opus-20240229": 4_096,
    "claude-3-5-haiku-20241022": 8_192,
    "claude-3-5-sonnet-20240620": 8_192,
    "claude-3-5-sonnet-20241022": 8_192,
    "claude-3-7-sonnet-20250219": 64_000,
    "claude-sonnet-4-20250514": 64_000,
    "claude-sonnet-4-5-20250929": 64_000,
    "claude-sonnet-4-6": 64_000,
    "claude-haiku-4-5-20251001": 64_000,
    "claude-opus-4-20250514": 32_000,
    "claude-opus-4-1": 32_000,
    "claude-opus-4-6": 128_000,
    "claude-opus-4-7": 128_000,
    "claude-opus-4-8": 128_000,
}


def _warn_if_documented_max_tokens_exceeded(model: str, requested_max_tokens: int) -> None:
    model_cap = _DOCUMENTED_ANTHROPIC_MAX_TOKENS_BY_MODEL.get(model)
    if model_cap is None or requested_max_tokens <= model_cap:
        return
    logger.warning(
        "Configured MAX_TOKENS=%d exceeds Anthropic's current documented max_tokens=%d "
        "for model '%s'. The request remains user-configurable and will be sent as-is.",
        requested_max_tokens,
        model_cap,
        model,
    )


def _build_anthropic_request_params(
    api: "CompletionAPI",
    model: str,
    prompt: str,
    suffix: str,
) -> Dict[str, Any]:
    max_tokens = int(api.max_tokens)
    _warn_if_documented_max_tokens_exceeded(model, max_tokens)

    request_params: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

    adaptive_only = model in _ADAPTIVE_ONLY_PROVIDER_IDS
    thinking_requested = "thinking" in suffix

    if adaptive_only and thinking_requested:
        # Opus 4.7+ only supports `thinking: {type: "adaptive"}`. Depth is
        # steered via `output_config.effort`, not `budget_tokens`.
        request_params["thinking"] = {"type": "adaptive"}
        effort = _extract_adaptive_effort(suffix)
        if effort:
            request_params["output_config"] = {"effort": effort}
        return request_params

    if not adaptive_only and _MANUAL_THINKING_SUFFIX in suffix:
        if max_tokens <= _THINKING_BUDGET_TOKENS:
            raise ValueError(
                "Anthropic thinking requests require max_tokens greater than the "
                f"thinking budget ({_THINKING_BUDGET_TOKENS}). "
                f"Resolved max_tokens={max_tokens} for model '{model}'."
            )
        request_params["thinking"] = {
            "type": "enabled",
            "budget_tokens": _THINKING_BUDGET_TOKENS,
        }
        return request_params

    temperature = float(api.temperature)
    if not 0.0 <= temperature <= 1.0:
        raise ValueError(
            f"Anthropic temperature must be between 0.0 and 1.0 inclusive; got {temperature}."
        )
    request_params["temperature"] = temperature
    return request_params


def _extract_text_from_message(msg: Dict[str, Any]) -> str:
    try:
        parts = msg.get("content", []) or []
        texts: List[str] = []
        for p in parts:
            if isinstance(p, dict) and p.get("type") == "text" and "text" in p:
                texts.append(p["text"])
        return "".join(texts) if texts else json.dumps(msg, ensure_ascii=False)
    except Exception:
        return json.dumps(msg, ensure_ascii=False)


def _describe_batch_result_error(res: Dict[str, Any]) -> str:
    rtype = res.get("type")
    err = res.get("error") or {}
    if isinstance(err, dict):
        message = err.get("message")
        if message:
            return str(message)
        if err:
            return json.dumps(err, ensure_ascii=False)
    if rtype:
        return str(rtype)
    return "unknown_error"


def call_anthropic_batch_api(
    api: "CompletionAPI",
    model_names: List[str],
    prompts: List[str],
) -> List[CompletionResult]:
    """Call Anthropic Messages Batch API."""
    if not prompts:
        return []

    if len(model_names) != len(prompts):
        raise ValueError(
            "model_names and prompts must have the same length, "
            f"got {len(model_names)} models and {len(prompts)} prompts"
        )

    requests: List[Request] = []
    for i, (model_name, prompt) in enumerate(zip(model_names, prompts)):
        model, suffix = api._resolve_provider_model(
            "anthropic", model_name, ANTHROPIC_MODEL_MAPPING, allow_fallback=False
        )
        request_params = _build_anthropic_request_params(api, model, prompt, suffix)

        params = MessageCreateParamsNonStreaming(**request_params)
        requests.append(Request(custom_id=f"request-{i}", params=params))

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    anthropic_client = Anthropic(api_key=anthropic_api_key, timeout=3600)

    message_batch = anthropic_client.messages.batches.create(requests=requests)

    batch_id = message_batch.id
    logger.info("Created Anthropic batch %s: %s", batch_id, message_batch)

    while True:
        message_batch = anthropic_client.messages.batches.retrieve(batch_id)
        if message_batch.processing_status == "ended":
            break
        time.sleep(30)

    responses_by_key: Dict[str, Dict[str, Any]] = {}

    for entry in anthropic_client.messages.batches.results(batch_id):
        result = entry.to_dict()

        key = result.get("custom_id")
        if not key:
            logger.warning("Batch result entry missing custom_id: %s", result)
            continue

        res = result.get("result") or {}
        rtype = res.get("type")

        if rtype == "succeeded":
            msg = res.get("message") or {}
            text = _extract_text_from_message(msg)
            usage = msg.get("usage") or {}
            responses_by_key[key] = {
                "text": text,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "reasoning_tokens": 0,
            }
        else:
            emsg = _describe_batch_result_error(res)
            responses_by_key[key] = {
                "text": f"[ERROR] {emsg}",
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
            }

    outputs: List[CompletionResult] = []
    for i in range(len(prompts)):
        key = f"request-{i}"
        rec = responses_by_key.get(
            key,
            {
                "text": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
            },
        )
        text = rec["text"]
        input_tokens = rec["input_tokens"]
        output_tokens = rec["output_tokens"]
        reasoning_tokens = rec.get("reasoning_tokens", 0)
        cost_usd = api.estimate_billed_cost_usd(
            "anthropic",
            model_names[i],
            input_tokens,
            output_tokens,
            batch=True,
        )
        generation_id = ""
        outputs.append(
            CompletionResult(
                text=text,
                cost_usd=cost_usd,
                generation_id=generation_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
            )
        )

    return outputs
