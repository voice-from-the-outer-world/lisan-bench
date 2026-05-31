"""
Runtime model catalog loaded exclusively from model_catalog.yaml.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from lisanbench.model_names import split_model_suffix

MODEL_CATALOG_FILENAME = "model_catalog.yaml"
MAPPED_COMPLETION_PROVIDERS = frozenset(
    {"anthropic", "google", "moonshotai", "openai", "z-ai", "zenmux"}
)
FLEX_SERVICE_TIER_COMPANIES = frozenset({"google", "openai"})
OPENROUTER_FLEX_PROVIDER_BY_COMPANY = {
    "google": "google-ai-studio",
    "openai": "openai",
}


def _parse_nonempty_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed if trimmed else None
    return None


def _parse_provider_list(value: Any) -> List[str]:
    single = _parse_nonempty_str(value)
    if single:
        return [single]
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        parsed = _parse_nonempty_str(item)
        if parsed:
            out.append(parsed)
    return out


def _parse_openrouter_provider_only(entry: Dict[str, Any]) -> List[str]:
    openrouter = entry.get("openrouter")
    if isinstance(openrouter, dict):
        provider_only = _parse_provider_list(openrouter.get("provider_only"))
        if provider_only:
            return provider_only
    return []


def _parse_service_tier(entry: Dict[str, Any], *, for_openrouter: bool = False) -> Optional[str]:
    tiers = _parse_provider_list(entry.get("service_tiers"))
    service_tier = _parse_nonempty_str(entry.get("service_tier"))
    if service_tier:
        tiers.insert(0, service_tier)

    if for_openrouter:
        openrouter = entry.get("openrouter")
        if isinstance(openrouter, dict):
            tiers.extend(_parse_provider_list(openrouter.get("service_tiers")))
            openrouter_service_tier = _parse_nonempty_str(openrouter.get("service_tier"))
            if openrouter_service_tier:
                tiers.insert(0, openrouter_service_tier)

    seen = set()
    for tier in tiers:
        normalized = tier.lower()
        if normalized == "priority":
            raise RuntimeError(
                f"{MODEL_CATALOG_FILENAME}: priority service tier is "
                "not supported by this benchmark; use 'flex' or omit service tier metadata"
            )
        if normalized != "flex":
            raise RuntimeError(
                f"{MODEL_CATALOG_FILENAME}: unsupported service tier "
                f"{tier!r}; expected 'flex'"
            )
        if normalized not in seen:
            seen.add(normalized)
            return "flex"
    return None


def _default_service_tier_for_company(company: str) -> Optional[str]:
    return "flex" if company in FLEX_SERVICE_TIER_COMPANIES else None


def _default_openrouter_provider_only_for_company(company: str) -> List[str]:
    provider = OPENROUTER_FLEX_PROVIDER_BY_COMPANY.get(company)
    return [provider] if provider else []


def _company_for(entry: Dict[str, Any], base_model: str) -> str:
    explicit = _parse_nonempty_str(entry.get("company"))
    if explicit:
        return explicit
    legacy_provider = _parse_nonempty_str(entry.get("provider"))
    if legacy_provider:
        return legacy_provider
    return base_model.split("/", 1)[0] if "/" in base_model else "other"


def _completion_provider_for(entry: Dict[str, Any], base_model: str) -> str:
    explicit = _parse_nonempty_str(entry.get("completion_provider"))
    if explicit:
        return explicit
    legacy_routing_provider = _parse_nonempty_str(entry.get("routing_provider"))
    if legacy_routing_provider:
        return legacy_routing_provider
    return "openrouter"


def _batch_completion_provider_for(
    entry: Dict[str, Any],
    completion_provider: str,
) -> str:
    explicit = _parse_nonempty_str(entry.get("batch_completion_provider"))
    if explicit:
        return explicit
    return completion_provider


def _completion_model_id_for(entry: Dict[str, Any]) -> Optional[str]:
    return _parse_nonempty_str(entry.get("completion_model_id")) or _parse_nonempty_str(
        entry.get("provider_model_id")
    )


def _routing_provider_for(entry: Dict[str, Any], base_model: str) -> str:
    """Backward-compatible name for older tests/imports."""
    return _completion_provider_for(entry, base_model)


def _load_catalog_entries() -> List[Dict[str, Any]]:
    catalog_path = Path(__file__).resolve().parent.parent / MODEL_CATALOG_FILENAME
    if not catalog_path.exists():
        raise RuntimeError(f"Required catalog file not found: {MODEL_CATALOG_FILENAME}")

    try:
        with catalog_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse {MODEL_CATALOG_FILENAME}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"{MODEL_CATALOG_FILENAME} must contain a top-level mapping")

    entries = payload.get("models")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"{MODEL_CATALOG_FILENAME} must contain a non-empty 'models' list")

    out: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"{MODEL_CATALOG_FILENAME}: models[{idx}] must be a mapping"
            )
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise RuntimeError(
                f"{MODEL_CATALOG_FILENAME}: models[{idx}] missing valid 'id'"
            )
        out.append(entry)
    return out


def _build_runtime_structures(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    model_names: List[str] = []
    company_by_model: Dict[str, str] = {}
    completion_provider_by_model: Dict[str, str] = {}
    batch_completion_provider_by_model: Dict[str, str] = {}

    anthropic_mapping: Dict[str, str] = {}
    google_mapping: Dict[str, str] = {}
    moonshot_mapping: Dict[str, str] = {}
    openai_mapping: Dict[str, str] = {}
    zai_mapping: Dict[str, str] = {}
    zenmux_mapping: Dict[str, str] = {}

    anthropic_pricing: Dict[str, Dict[str, float]] = {}
    google_pricing: Dict[str, Dict[str, float]] = {}
    moonshot_pricing: Dict[str, Dict[str, float]] = {}
    openai_pricing: Dict[str, Dict[str, float]] = {}
    zai_pricing: Dict[str, Dict[str, float]] = {}
    zenmux_pricing: Dict[str, Dict[str, float]] = {}
    model_pricing: Dict[str, Dict[str, float]] = {}

    release_dates: Dict[str, Dict[str, str]] = {}
    openrouter_provider_only_by_model: Dict[str, List[str]] = {}
    service_tier_by_model: Dict[str, str] = {}
    openrouter_service_tier_by_model: Dict[str, str] = {}
    for entry in entries:
        model_id = str(entry["id"]).strip()
        if bool(entry.get("listed", True)):
            model_names.append(model_id)

        base_model = model_id.split(":", 1)[0]
        company = _company_for(entry, base_model)
        completion_provider = _completion_provider_for(entry, base_model)
        batch_completion_provider = _batch_completion_provider_for(
            entry,
            completion_provider,
        )
        company_by_model[model_id] = company
        completion_provider_by_model[model_id] = completion_provider
        batch_completion_provider_by_model[model_id] = batch_completion_provider

        completion_model_id = _completion_model_id_for(entry)
        if completion_model_id:
            mapped_providers = {
                completion_provider,
                batch_completion_provider,
            } & MAPPED_COMPLETION_PROVIDERS
            for provider in mapped_providers:
                if provider == "anthropic":
                    anthropic_mapping[base_model] = completion_model_id
                elif provider == "google":
                    google_mapping[base_model] = completion_model_id
                elif provider == "moonshotai":
                    moonshot_mapping[base_model] = completion_model_id
                elif provider == "openai":
                    openai_mapping[base_model] = completion_model_id
                elif provider == "z-ai":
                    zai_mapping[base_model] = completion_model_id
                elif provider == "zenmux":
                    zenmux_mapping[base_model] = completion_model_id

        pricing = entry.get("pricing")
        if isinstance(pricing, dict):
            p_in = pricing.get("in")
            p_out = pricing.get("out")
            if isinstance(p_in, (int, float)) and isinstance(p_out, (int, float)):
                short_model_key = base_model.split("/", 1)[1] if "/" in base_model else base_model
                rec = {"in": float(p_in), "out": float(p_out)}
                model_pricing[model_id] = rec
                priced_providers = {
                    completion_provider,
                    batch_completion_provider,
                } & MAPPED_COMPLETION_PROVIDERS
                for provider in priced_providers:
                    if provider == "anthropic":
                        anthropic_pricing[short_model_key] = rec
                    elif provider == "google":
                        google_pricing[short_model_key] = rec
                    elif provider == "moonshotai":
                        moonshot_pricing[short_model_key] = rec
                    elif provider == "openai":
                        openai_pricing[short_model_key] = rec
                    elif provider == "z-ai":
                        zai_pricing[short_model_key] = rec
                    elif provider == "zenmux":
                        zenmux_pricing[short_model_key] = rec

        release = entry.get("release")
        if isinstance(release, dict):
            date = release.get("date")
            source = release.get("source")
            if isinstance(date, str) and date.strip() and isinstance(source, str) and source.strip():
                release_dates[model_id] = {"date": date.strip(), "source": source.strip()}

        provider_only = _parse_openrouter_provider_only(entry)
        if not provider_only and completion_provider == "openrouter":
            provider_only = _default_openrouter_provider_only_for_company(company)

        if provider_only:
            openrouter_provider_only_by_model[base_model] = provider_only

        default_service_tier = _default_service_tier_for_company(company)

        service_tier = _parse_service_tier(entry) or default_service_tier
        if service_tier:
            service_tier_by_model[base_model] = service_tier

        service_tier = _parse_service_tier(entry, for_openrouter=True) or default_service_tier
        if service_tier:
            openrouter_service_tier_by_model[base_model] = service_tier

    return {
        "models": model_names,
        "COMPANY_BY_MODEL": company_by_model,
        "COMPLETION_PROVIDER_BY_MODEL": completion_provider_by_model,
        "BATCH_COMPLETION_PROVIDER_BY_MODEL": batch_completion_provider_by_model,
        "PROVIDER_BY_MODEL": batch_completion_provider_by_model,
        "ANTHROPIC_MODEL_MAPPING": anthropic_mapping,
        "GOOGLE_MODEL_MAPPING": google_mapping,
        "MOONSHOT_MODEL_MAPPING": moonshot_mapping,
        "OPENAI_MODEL_MAPPING": openai_mapping,
        "ZAI_MODEL_MAPPING": zai_mapping,
        "ZENMUX_MODEL_MAPPING": zenmux_mapping,
        "ANTHROPIC_PRICING": anthropic_pricing,
        "GOOGLE_PRICING": google_pricing,
        "MOONSHOT_PRICING": moonshot_pricing,
        "OPENAI_PRICING": openai_pricing,
        "ZAI_PRICING": zai_pricing,
        "ZENMUX_PRICING": zenmux_pricing,
        "MODEL_PRICING": model_pricing,
        "model_release_dates": release_dates,
        "SERVICE_TIER_BY_MODEL": service_tier_by_model,
        "OPENROUTER_PROVIDER_ONLY_BY_MODEL": openrouter_provider_only_by_model,
        "OPENROUTER_SERVICE_TIER_BY_MODEL": openrouter_service_tier_by_model,
    }


_RUNTIME_DATA = _build_runtime_structures(_load_catalog_entries())

models = _RUNTIME_DATA["models"]
COMPANY_BY_MODEL = _RUNTIME_DATA["COMPANY_BY_MODEL"]
COMPLETION_PROVIDER_BY_MODEL = _RUNTIME_DATA["COMPLETION_PROVIDER_BY_MODEL"]
BATCH_COMPLETION_PROVIDER_BY_MODEL = _RUNTIME_DATA["BATCH_COMPLETION_PROVIDER_BY_MODEL"]
PROVIDER_BY_MODEL = _RUNTIME_DATA["PROVIDER_BY_MODEL"]

ANTHROPIC_MODEL_MAPPING = _RUNTIME_DATA["ANTHROPIC_MODEL_MAPPING"]
GOOGLE_MODEL_MAPPING = _RUNTIME_DATA["GOOGLE_MODEL_MAPPING"]
MOONSHOT_MODEL_MAPPING = _RUNTIME_DATA["MOONSHOT_MODEL_MAPPING"]
OPENAI_MODEL_MAPPING = _RUNTIME_DATA["OPENAI_MODEL_MAPPING"]
ZAI_MODEL_MAPPING = _RUNTIME_DATA["ZAI_MODEL_MAPPING"]
ZENMUX_MODEL_MAPPING = _RUNTIME_DATA["ZENMUX_MODEL_MAPPING"]

ANTHROPIC_PRICING = _RUNTIME_DATA["ANTHROPIC_PRICING"]
GOOGLE_PRICING = _RUNTIME_DATA["GOOGLE_PRICING"]
MOONSHOT_PRICING = _RUNTIME_DATA["MOONSHOT_PRICING"]
OPENAI_PRICING = _RUNTIME_DATA["OPENAI_PRICING"]
ZAI_PRICING = _RUNTIME_DATA["ZAI_PRICING"]
ZENMUX_PRICING = _RUNTIME_DATA["ZENMUX_PRICING"]
MODEL_PRICING = _RUNTIME_DATA["MODEL_PRICING"]

model_release_dates = _RUNTIME_DATA["model_release_dates"]
SERVICE_TIER_BY_MODEL = _RUNTIME_DATA["SERVICE_TIER_BY_MODEL"]
OPENROUTER_PROVIDER_ONLY_BY_MODEL = _RUNTIME_DATA["OPENROUTER_PROVIDER_ONLY_BY_MODEL"]
OPENROUTER_SERVICE_TIER_BY_MODEL = _RUNTIME_DATA["OPENROUTER_SERVICE_TIER_BY_MODEL"]


def get_model_service_tier(model_name: str) -> Optional[str]:
    base_model, _ = split_model_suffix(model_name)
    return SERVICE_TIER_BY_MODEL.get(base_model)


def get_openrouter_provider_config(model_name: str) -> Optional[Dict[str, List[str]]]:
    base_model, _ = split_model_suffix(model_name)
    providers = OPENROUTER_PROVIDER_ONLY_BY_MODEL.get(base_model)
    if not providers:
        return None
    return {"only": list(providers)}


def get_openrouter_service_tier(model_name: str) -> Optional[str]:
    base_model, _ = split_model_suffix(model_name)
    return OPENROUTER_SERVICE_TIER_BY_MODEL.get(base_model)
