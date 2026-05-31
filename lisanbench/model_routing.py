"""Central routing decisions for benchmark model ids."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from lisanbench.model_catalog import (
    ANTHROPIC_MODEL_MAPPING,
    BATCH_COMPLETION_PROVIDER_BY_MODEL,
    COMPLETION_PROVIDER_BY_MODEL,
    GOOGLE_MODEL_MAPPING,
    OPENAI_MODEL_MAPPING,
)
from lisanbench.model_names import parse_model_name

API_KEY_ENV_BY_BACKEND = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "moonshotai": "MOONSHOT_API_KEY",
    "z-ai": "ZAI_API_KEY",
    "zenmux": "ZENMUX_API_KEY",
}

BATCH_PROVIDER_MODEL_MAPPINGS: Dict[str, Dict[str, str]] = {
    "openai": OPENAI_MODEL_MAPPING,
    "google": GOOGLE_MODEL_MAPPING,
    "anthropic": ANTHROPIC_MODEL_MAPPING,
}

BATCH_SUPPORTED_PROVIDERS = frozenset(BATCH_PROVIDER_MODEL_MAPPINGS)


@dataclass(frozen=True)
class ModelRoute:
    model_name: str
    backend: str
    api_key_env: Optional[str]
    batching: bool = False
    batch_supported: bool = False

    @property
    def uses_openrouter(self) -> bool:
        return self.backend == "openrouter"


def catalog_completion_provider(model_name: str) -> str:
    return COMPLETION_PROVIDER_BY_MODEL.get(model_name, "openrouter")


def catalog_batch_completion_provider(model_name: str) -> str:
    return BATCH_COMPLETION_PROVIDER_BY_MODEL.get(
        model_name,
        catalog_completion_provider(model_name),
    )


def catalog_routing_provider(model_name: str) -> str:
    """Backward-compatible wrapper for older imports."""
    return catalog_batch_completion_provider(model_name)


def resolve_model_route(
    model_name: str,
    *,
    force_openrouter: bool = False,
    batching: bool = False,
) -> ModelRoute:
    if batching:
        provider = catalog_batch_completion_provider(model_name)
        return ModelRoute(
            model_name=model_name,
            backend=provider,
            api_key_env=API_KEY_ENV_BY_BACKEND.get(provider),
            batching=True,
            batch_supported=provider in BATCH_SUPPORTED_PROVIDERS,
        )

    if force_openrouter:
        backend = "openrouter"
    else:
        backend = catalog_completion_provider(model_name)

    return ModelRoute(
        model_name=model_name,
        backend=backend,
        api_key_env=API_KEY_ENV_BY_BACKEND.get(backend),
    )


def required_api_key_envs(
    model_names: List[str],
    *,
    force_openrouter: bool = False,
    batching: bool = False,
) -> List[str]:
    if batching:
        providers = {
            catalog_batch_completion_provider(model_name)
            for model_name in model_names
            if "/" in model_name
        }
        return [
            API_KEY_ENV_BY_BACKEND[provider]
            for provider in ("openai", "google", "anthropic")
            if provider in providers
        ]

    if force_openrouter:
        return ["OPENROUTER_API_KEY"]

    routes = [
        resolve_model_route(model_name, force_openrouter=False)
        for model_name in model_names
    ]
    ordered_backends = ("openrouter", "openai", "google", "moonshotai", "z-ai", "zenmux")
    return [
        API_KEY_ENV_BY_BACKEND[backend]
        for backend in ordered_backends
        if any(route.backend == backend for route in routes)
    ]


def batch_model_has_completion_provider_mapping(
    model_name: str,
    completion_provider: str,
) -> bool:
    mapping = BATCH_PROVIDER_MODEL_MAPPINGS.get(completion_provider)
    if not mapping:
        return False
    return parse_model_name(model_name).base_model in mapping


def batch_model_has_provider_mapping(model_name: str, provider: str) -> bool:
    """Backward-compatible wrapper for older imports."""
    return batch_model_has_completion_provider_mapping(model_name, provider)
