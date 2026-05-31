import datetime
import json
import logging
import os
from dataclasses import dataclass
from threading import Lock, local
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from lisanbench.providers.anthropic import call_anthropic_batch_api
from lisanbench.providers.types import CompletionResult
from lisanbench.providers.google import (
    call_google_aistudio,
    call_google_batch_api,
)
from lisanbench.providers.moonshot import call_moonshot_api
from lisanbench.providers.zenmux import call_zenmux_api
from lisanbench.providers.openai import call_openai_api, call_openai_batch_api
from lisanbench.providers.zai import call_zai_api
from lisanbench.providers.openrouter import (
    build_reasoning_payload,
    call_openrouter_api,
    consume_json_response,
    consume_streaming_response,
    extract_finish_reason,
    extract_text_block,
    extract_text_from_choice,
    get_openrouter_costs,
)
from lisanbench.model_catalog import (
    ANTHROPIC_PRICING,
    GOOGLE_PRICING,
    MODEL_PRICING,
    MOONSHOT_PRICING,
    OPENAI_PRICING,
    ZAI_PRICING,
    ZENMUX_PRICING,
    get_model_service_tier,
)
from lisanbench.model_names import OPENAI_REASONING_EFFORT_LEVELS, extract_reasoning_effort, split_model_suffix
from lisanbench.model_routing import resolve_model_route

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FLEX_COST_MULTIPLIER = 0.5
BATCH_COST_MULTIPLIER = 0.5
FLEX_DISCOUNT_PROVIDERS = frozenset({"openai", "google"})

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _read_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid integer value for %s=%r; using default %d.", name, raw, default)
        return default
    if value <= 0:
        logger.warning("Expected positive integer for %s=%r; using default %d.", name, raw, default)
        return default
    return value


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float value for %s=%r; using default %s.", name, raw, default)
        return default


def _coerce_positive_int_setting(name: str, value: Optional[int], default: int) -> int:
    if value is None:
        return default
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.") from None
    if coerced <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return coerced


def _coerce_float_setting(name: str, value: Optional[float], default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number; got {value!r}.") from None


@dataclass
class APIUsage:
    """Track API usage statistics."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0


class ThreadSafeAPIUsage:
    """Thread-safe API usage tracking."""

    def __init__(self):
        self.lock = Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_cost_usd = 0.0

    def add_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        cost: float,
    ) -> None:
        """Add usage statistics in a thread-safe manner."""
        with self.lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_reasoning_tokens += reasoning_tokens
            self.total_cost_usd += cost


class CompletionAPI:
    """General completion coordinator with provider-specific backends."""

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_batching: bool = False,
        stream_responses: bool = True,
        force_openrouter: bool = False,
    ):
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            logger.info(
                "OPENROUTER_API_KEY not set. "
                "OpenRouter calls and cost lookups will fail until it is provided."
            )
        self.zai_api_key = os.getenv("ZAI_API_KEY")
        self.moonshot_api_key = os.getenv("MOONSHOT_API_KEY")
        self.zenmux_api_key = os.getenv("ZENMUX_API_KEY")
        env_max_tokens = _read_positive_int_env("MAX_TOKENS", 64000)
        self.max_tokens = _coerce_positive_int_setting(
            "max_tokens",
            max_tokens,
            env_max_tokens,
        )
        env_temperature = _read_float_env("TEMPERATURE", 1.0)
        self.temperature = _coerce_float_setting(
            "temperature",
            temperature,
            env_temperature,
        )

        self.use_batching = use_batching
        self.api_usage = ThreadSafeAPIUsage()
        self._session_local = local()
        self.stream_responses = stream_responses
        self.force_openrouter = force_openrouter

    @staticmethod
    def _split_model_and_suffix(model_name: str) -> Tuple[str, str]:
        return split_model_suffix(model_name)

    @staticmethod
    def _extract_reasoning_effort(suffix: str) -> Optional[str]:
        return extract_reasoning_effort(suffix, OPENAI_REASONING_EFFORT_LEVELS)

    def _resolve_provider_model(
        self,
        provider: str,
        model_name: str,
        mapping: Dict[str, str],
        *,
        allow_fallback: bool = True,
    ) -> Tuple[str, str]:
        """Resolve a benchmark model name into a provider-native model id."""
        base_model, suffix = self._split_model_and_suffix(model_name)
        mapped = mapping.get(base_model)
        if mapped:
            return mapped, suffix

        if allow_fallback:
            fallback = base_model.split("/", 1)[1] if "/" in base_model else base_model
            if provider == "anthropic":
                fallback = fallback.replace(".", "-")
            logger.warning(
                "No mapping found for %s model '%s'; falling back to provider model id '%s'.",
                provider,
                model_name,
                fallback,
            )
            return fallback, suffix

        raise ValueError(
            f"No mapping found for {provider} model '{model_name}'. "
            "Add completion_model_id in model_catalog.yaml for this model."
        )

    def _require_openrouter_key(self) -> str:
        if not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment variables. "
                "Set it to use OpenRouter completion or generation-cost APIs."
            )
        return self.openrouter_api_key

    def _require_moonshot_key(self) -> str:
        if not self.moonshot_api_key:
            raise ValueError(
                "MOONSHOT_API_KEY not found in environment variables. "
                "Set it to use moonshotai/* direct completion APIs."
            )
        return self.moonshot_api_key

    def _require_zai_key(self) -> str:
        if not self.zai_api_key:
            raise ValueError(
                "ZAI_API_KEY not found in environment variables. "
                "Set it to use z-ai/* direct completion APIs."
            )
        return self.zai_api_key

    def _require_zenmux_key(self) -> str:
        if not self.zenmux_api_key:
            raise ValueError(
                "ZENMUX_API_KEY not found in environment variables. "
                "Set it to use zenmux/* direct completion APIs."
            )
        return self.zenmux_api_key


    def _estimate_cost_usd(
        self,
        provider: str,
        canonical_model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate provider cost using static per-1M-token pricing tables."""
        base_model, _ = self._split_model_and_suffix(canonical_model_name)
        short_model = base_model.split("/", 1)[1] if "/" in base_model else base_model
        pricing = MODEL_PRICING.get(canonical_model_name) or MODEL_PRICING.get(base_model)
        pricing_tables = {
            "openai": OPENAI_PRICING,
            "google": GOOGLE_PRICING,
            "anthropic": ANTHROPIC_PRICING,
            "moonshotai": MOONSHOT_PRICING,
            "z-ai": ZAI_PRICING,
            "zenmux": ZENMUX_PRICING,
        }
        if pricing is None:
            pricing = pricing_tables.get(provider, {}).get(short_model)
        if not pricing:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * float(pricing.get("in", 0.0))
        output_cost = (output_tokens / 1_000_000) * float(pricing.get("out", 0.0))
        return input_cost + output_cost

    def estimate_billed_cost_usd(
        self,
        provider: str,
        canonical_model_name: str,
        input_tokens: int,
        output_tokens: int,
        *,
        batch: bool = False,
    ) -> float:
        """Estimate the amount billed for a request from catalog list prices."""
        list_price_cost = self._estimate_cost_usd(
            provider,
            canonical_model_name,
            input_tokens,
            output_tokens,
        )
        return list_price_cost * self._billing_cost_multiplier(
            provider,
            canonical_model_name,
            batch=batch,
        )

    def _billing_cost_multiplier(
        self,
        provider: str,
        canonical_model_name: str,
        *,
        batch: bool = False,
    ) -> float:
        if batch:
            return BATCH_COST_MULTIPLIER
        if (
            provider in FLEX_DISCOUNT_PROVIDERS
            and get_model_service_tier(canonical_model_name) == "flex"
        ):
            return FLEX_COST_MULTIPLIER
        return 1.0

    def create_batch(self, provider: str, model_names: List[str], prompts: List[str]):
        """Dispatch a batch call to the provider-specific implementation."""
        if len(model_names) != len(prompts):
            raise ValueError(
                "model_names and prompts must have the same length, "
                f"got {len(model_names)} models and {len(prompts)} prompts"
            )

        if provider == "openai":
            unique_models = set(model_names)
            if len(unique_models) != 1:
                raise ValueError(
                    f"OpenAI batch API requires a single model per batch file, got: {unique_models}"
                )
            return self._record_completion_results(self._call_openai_batch_api(model_names[0], prompts))

        if provider == "google":
            unique_models = set(model_names)
            if len(unique_models) != 1:
                raise ValueError(
                    "Google Gemini batch API currently assumes a single model per batch, "
                    f"got: {unique_models}"
                )
            return self._record_completion_results(self._call_google_batch_api(model_names[0], prompts))

        if provider == "anthropic":
            return self._record_completion_results(self._call_anthropic_batch_api(model_names, prompts))

        raise ValueError(f"Unknown provider for batching: {provider!r}")

    def create_completion(self, model_name: str, prompt: str):
        """Call a model API and return text, cost, ids, and token usage."""
        route = resolve_model_route(
            model_name,
            force_openrouter=self.force_openrouter,
        )
        if route.backend == "zenmux":
            return self._record_completion_result(self._call_zenmux_api(model_name, prompt))
        if route.backend == "moonshotai":
            return self._record_completion_result(self._call_moonshot_api(model_name, prompt))
        if route.backend == "z-ai":
            return self._record_completion_result(self._call_zai_api(model_name, prompt))
        if route.backend == "google":
            return self._record_completion_result(self._call_google_aistudio(model_name, prompt))
        if route.backend == "openai":
            return self._record_completion_result(self._call_openai_api(model_name, prompt))

        return self._record_completion_result(self._call_openrouter_api(model_name, prompt))

    def _record_completion_result(self, result: CompletionResult) -> CompletionResult:
        completion = CompletionResult.from_value(result)
        self.api_usage.add_usage(
            completion.input_tokens,
            completion.output_tokens,
            completion.reasoning_tokens,
            completion.cost_usd,
        )
        return completion

    def _record_completion_results(self, results: List[CompletionResult]) -> List[CompletionResult]:
        completions = [CompletionResult.from_value(result) for result in results]
        for completion in completions:
            self.api_usage.add_usage(
                completion.input_tokens,
                completion.output_tokens,
                completion.reasoning_tokens,
                completion.cost_usd,
            )
        return completions

    def _get_session(self) -> requests.Session:
        """Return a per-thread requests.Session instance."""
        session = getattr(self._session_local, "session", None)
        if session is None:
            session = requests.Session()
            self._session_local.session = session
        return session

    def _log_raw_interaction(
        self,
        *,
        model_name: str,
        prompt: str,
        request_payload: Dict,
        generation_id: str = "no-id",
        response_payload: Optional[Dict] = None,
        raw_response_text: Optional[str] = None,
        error: Optional[Dict] = None,
        stream_events: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist the full prompt/response/error for auditability."""
        now_utc = datetime.datetime.now(datetime.UTC)
        timestamp_for_name = now_utc.strftime("%Y%m%dT%H%M%SZ")
        iso_timestamp = now_utc.isoformat()
        safe_model_name = model_name.replace(":", "_").replace("/", "_")
        base_filename = f"{timestamp_for_name}_{safe_model_name}_{generation_id or 'no-id'}"
        json_path = os.path.join("raw", f"{base_filename}.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        log_payload: Dict[str, Any] = {
            "timestamp": iso_timestamp,
            "model": model_name,
            "generation_id": generation_id or "no-id",
            "prompt": prompt,
            "request": request_payload,
        }

        if response_payload is not None:
            log_payload["response_data"] = response_payload
        if raw_response_text is not None:
            log_payload["raw_response_text"] = raw_response_text
        if error is not None:
            log_payload["error"] = error
        if stream_events is not None:
            log_payload["stream_events"] = stream_events

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(log_payload, f, ensure_ascii=False, indent=2)
        except Exception as logging_exc:
            fallback_path = os.path.join("raw", f"{base_filename}.log")
            try:
                with open(fallback_path, "w", encoding="utf-8") as f:
                    f.write("Failed to write JSON log; falling back to text.\n")
                    f.write(f"Logging error: {logging_exc}\n\n")
                    f.write(json.dumps(log_payload, ensure_ascii=False, indent=2))
            except Exception:
                pass

    def _call_openai_batch_api(self, model_name: str, prompts: List[str]):
        return call_openai_batch_api(self, model_name, prompts)

    def _call_openai_api(self, model_name: str, prompt: str):
        return call_openai_api(self, model_name, prompt)

    def _call_google_batch_api(self, model_name: str, prompts: List[str]):
        return call_google_batch_api(self, model_name, prompts)

    def _call_anthropic_batch_api(self, model_names: List[str], prompts: List[str]):
        return call_anthropic_batch_api(self, model_names, prompts)

    def _call_openrouter_api(self, model_name: str, prompt: str):
        return call_openrouter_api(self, model_name, prompt)

    def _call_moonshot_api(self, model_name: str, prompt: str):
        return call_moonshot_api(self, model_name, prompt)

    def _call_zenmux_api(self, model_name: str, prompt: str):
        return call_zenmux_api(self, model_name, prompt)

    def _call_zai_api(self, model_name: str, prompt: str):
        return call_zai_api(self, model_name, prompt)


    def _consume_json_response(
        self,
        response: requests.Response,
        model_name: str,
        prompt: str,
        request_payload: Dict,
    ):
        return consume_json_response(self, response, model_name, prompt, request_payload)

    def _consume_streaming_response(
        self,
        response: requests.Response,
        model_name: str,
        prompt: str,
        request_payload: Dict,
    ):
        return consume_streaming_response(self, response, model_name, prompt, request_payload)

    def _extract_text_from_choice(
        self,
        choice: Dict[str, Any],
        *,
        has_existing_chunks: bool,
    ) -> str:
        return extract_text_from_choice(choice, has_existing_chunks=has_existing_chunks)

    def _extract_text_block(self, block: Any) -> str:
        return extract_text_block(block)

    def _extract_finish_reason(self, event: Dict[str, Any]) -> Optional[str]:
        return extract_finish_reason(event)

    def get_openrouter_costs(self, generation_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        return get_openrouter_costs(self, generation_ids)

    def estimate_model_catalog_cost_usd(
        self,
        canonical_model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        provider = (
            canonical_model_name.split("/", 1)[0]
            if "/" in canonical_model_name
            else canonical_model_name
        )
        return self.estimate_billed_cost_usd(
            provider,
            canonical_model_name,
            input_tokens,
            output_tokens,
        )

    def _build_reasoning_payload(self, model_name: str) -> Optional[Dict[str, Any]]:
        return build_reasoning_payload(model_name)

    def _call_google_aistudio(self, model_name: str, prompt: str):
        return call_google_aistudio(self, model_name, prompt)
