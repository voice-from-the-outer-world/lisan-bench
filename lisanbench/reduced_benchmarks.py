"""Reduced benchmark presets used by the CLI and result printer."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from lisanbench.model_names import is_reasoning_model_name

REDUCED_PRESET_NAME = "reduced_15x2_path_length"
REDUCED_NUM_TRIALS = 2
REDUCED_ESTIMATORS_PATH = Path(__file__).resolve().parent.parent / "reduced_benchmark_estimators.json"

FALLBACK_REDUCED_WORDS = [
    "all",
    "beat",
    "brass",
    "created",
    "flags",
    "jobs",
    "keep",
    "lamp",
    "leaders",
    "moves",
    "not",
    "one",
    "reached",
    "sex",
    "shipping",
]


def is_reasoning_model(model_name: str) -> bool:
    return is_reasoning_model_name(model_name)


def reduced_estimator_key(model_name: str, config: Optional[Dict[str, Any]] = None) -> str:
    if config is None:
        config = load_reduced_config()
    if config and config.get("estimator_mode") == "global":
        return "global"
    return "reasoning" if is_reasoning_model(model_name) else "non_reasoning"


def load_reduced_config(
    path: Path = REDUCED_ESTIMATORS_PATH,
) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    config = payload.get(REDUCED_PRESET_NAME)
    if not isinstance(config, dict):
        return None
    estimators = config.get("estimators")
    if not isinstance(estimators, dict):
        return None
    shared_words = config.get("words")
    if not isinstance(shared_words, list) or not all(isinstance(word, str) for word in shared_words):
        return None
    required_keys = ("global",) if config.get("estimator_mode") == "global" else ("non_reasoning", "reasoning")
    for key in required_keys:
        estimator = estimators.get(key)
        if not isinstance(estimator, dict):
            return None
        coefficients = estimator.get("coefficients")
        intercept = estimator.get("intercept")
        if (
            not isinstance(coefficients, list)
            or not all(isinstance(value, (int, float)) for value in coefficients)
            or not isinstance(intercept, (int, float))
            or len(shared_words) != len(coefficients)
        ):
            return None
        estimator["words"] = list(shared_words)
    return config


def get_reduced_estimator(model_name: str) -> Optional[Dict[str, Any]]:
    config = load_reduced_config()
    if not config:
        return None
    return config["estimators"][reduced_estimator_key(model_name, config)]


def get_reduced_words(model_name: Optional[str] = None) -> List[str]:
    if model_name is not None:
        estimator = get_reduced_estimator(model_name)
        if estimator:
            return list(estimator["words"])
    config = load_reduced_config()
    if config:
        words: List[str] = []
        for estimator in config["estimators"].values():
            for word in estimator["words"]:
                if word not in words:
                    words.append(word)
        return words
    return list(FALLBACK_REDUCED_WORDS)


# Backward-compatible aliases for code paths added when this preset was 10x3.
REDUCED_10X3_PRESET_NAME = REDUCED_PRESET_NAME
FALLBACK_REDUCED_10X3_WORDS = FALLBACK_REDUCED_WORDS
reduced_10x3_estimator_key = reduced_estimator_key
load_reduced_10x3_config = load_reduced_config
get_reduced_10x3_estimator = get_reduced_estimator
get_reduced_10x3_words = get_reduced_words
