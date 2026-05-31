"""Shared helpers for benchmark model ids, suffixes, and reasoning labels."""

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


REASONING_SUFFIX_ALIASES = {
    "kimi-k2-thinking": "kimi-k2:thinking",
    "olmo-3-32b-think": "olmo-3-32b:thinking",
}

# Provider-specific parsing orders. `xhigh` must be checked before `high`
# for substring-based APIs because "xhigh" contains "high".
OPENAI_REASONING_EFFORT_LEVELS = ("none", "minimal", "low", "medium", "xhigh", "high")
OPENROUTER_REASONING_EFFORT_LEVELS = OPENAI_REASONING_EFFORT_LEVELS
OPENROUTER_VERBOSITY_LEVELS = ("low", "medium", "high", "xhigh", "max")
ANTHROPIC_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
GOOGLE_THINKING_LEVELS = ("minimal", "low", "medium", "high")
ZENMUX_REASONING_EFFORT_LEVELS = ("none", "minimal", "low", "medium", "high")

IMPLICIT_HIGH_REASONING_BASES = {
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-3.0-pro",
}

IMPLICIT_MEDIUM_REASONING_BASES = {
    "o1-mini",
    "o3-mini",
    "o4-mini",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
}

IMPLICIT_THINKING_BASES = {
    "deepseek-r1-0528",
    "glm-4.5",
    "glm-4.5-air",
    "glm-4.6",
    "glm-4.7",
    "minimax-m1",
}


@dataclass(frozen=True)
class ModelNameParts:
    original: str
    base_model: str
    suffix: str
    suffix_lower: str
    short_base: str
    short_name: str


def strip_provider_prefix(model_name: str) -> str:
    return model_name.split("/", 1)[1] if "/" in model_name else model_name


def split_model_suffix(model_name: str, *, lower_suffix: bool = False) -> Tuple[str, str]:
    if ":" in model_name:
        base, suffix = model_name.split(":", 1)
        return base, suffix.lower() if lower_suffix else suffix
    return model_name, ""


def parse_model_name(model_name: str) -> ModelNameParts:
    base_model, suffix = split_model_suffix(model_name)
    short_base = strip_provider_prefix(base_model)
    short_name = strip_provider_prefix(model_name)
    return ModelNameParts(
        original=model_name,
        base_model=base_model,
        suffix=suffix,
        suffix_lower=suffix.lower(),
        short_base=short_base,
        short_name=short_name,
    )


def split_reasoning_suffix(model_name: str) -> Tuple[str, str]:
    raw_name = strip_provider_prefix(model_name)
    canonical = REASONING_SUFFIX_ALIASES.get(raw_name, raw_name)
    if ":" in canonical:
        return canonical.split(":", 1)

    release_thinking_match = re.fullmatch(r"(.+)-thinking-(\d{4,}.*)", canonical)
    if release_thinking_match:
        return f"{release_thinking_match.group(1)}-{release_thinking_match.group(2)}", "thinking"
    if canonical.endswith("-thinking"):
        return canonical[:-9], "thinking"
    if canonical.endswith("-think"):
        return canonical[:-6], "thinking"
    return canonical, ""


def explicit_reasoning_disabled(suffix: str) -> bool:
    return suffix.lower() in {"thinking-none", "thinking-off"}


def explicit_reasoning_enabled(suffix: str) -> bool:
    suffix_l = suffix.lower()
    return suffix_l == "thinking" or suffix_l.startswith("thinking-")


def implicit_reasoning_label(base_model: str, suffix: str) -> Optional[str]:
    base_l = base_model.lower()
    suffix_l = suffix.lower()

    if base_l in IMPLICIT_HIGH_REASONING_BASES and not suffix_l:
        return "high"

    force_medium = base_l in IMPLICIT_MEDIUM_REASONING_BASES or base_l.startswith("gpt-oss-")
    if force_medium and not suffix_l:
        return "medium"

    force_thinking = base_l in IMPLICIT_THINKING_BASES
    if (
        (base_l.startswith("grok-") or force_thinking)
        and not suffix_l.startswith("thinking")
        and base_l != "grok-4.1-fast"
    ):
        return "thinking"

    return None


def is_reasoning_model_name(model_name: str) -> bool:
    base, suffix = split_reasoning_suffix(model_name.lower())
    suffix_l = suffix.lower()

    if explicit_reasoning_disabled(suffix_l):
        return False
    if explicit_reasoning_enabled(suffix_l):
        return True

    return implicit_reasoning_label(base, suffix_l) is not None


def thinking_suffix_parameter(suffix: str) -> Optional[str]:
    suffix_l = suffix.lower()
    if suffix_l.startswith("thinking-"):
        return suffix_l.removeprefix("thinking-")
    return None


def extract_reasoning_effort(
    suffix: str,
    levels: Iterable[str],
) -> Optional[str]:
    suffix_l = suffix.lower()
    if "thinking" not in suffix_l:
        return None
    for level in levels:
        if level in suffix_l:
            return level
    return None


def extract_word_boundary_reasoning_effort(
    suffix: str,
    levels: Iterable[str],
) -> Optional[str]:
    suffix_l = suffix.lower()
    for level in levels:
        if re.search(rf"\b{re.escape(level)}\b", suffix_l):
            return level
    return None
