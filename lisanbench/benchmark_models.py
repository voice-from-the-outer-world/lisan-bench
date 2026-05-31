from dataclasses import MISSING, asdict, dataclass, fields
from typing import Any, Dict, List


@dataclass
class Results:
    """Store results for a single benchmark run."""

    model_name: str
    starting_word: str
    chain_length: int
    word_chain: List[str]
    longest_valid_chain: int
    total_valid_links: int
    total_invalid_links: int
    validity_ratio: float
    execution_time: float
    timestamp: str
    temperature: float
    raw_response: str
    api_cost_usd: float
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    generation_id: str = ""
    response_api_cost_usd: float = 0.0
    cost_source: str = "response"

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "Results":
        if not isinstance(payload, dict):
            raise TypeError(f"Result row must be a mapping, got {type(payload).__name__}")

        payload = dict(payload)
        if "chain_length" not in payload and "longest_valid_chain" in payload:
            payload["chain_length"] = payload["longest_valid_chain"]
        if "longest_valid_chain" not in payload and "chain_length" in payload:
            payload["longest_valid_chain"] = payload["chain_length"]
        payload.setdefault("word_chain", [])
        payload.setdefault("raw_response", "")

        values: Dict[str, Any] = {}
        for field in fields(cls):
            if field.name in payload:
                values[field.name] = payload[field.name]
            elif field.default is not MISSING:
                values[field.name] = field.default
            elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
                values[field.name] = field.default_factory()  # type: ignore[misc,attr-defined]
            else:
                raise TypeError(f"Result row missing required field {field.name!r}")
        return cls(**values)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrialJob:
    model_name: str
    provider: str
    starting_word: str
    trial_number: int

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "TrialJob":
        return cls(
            model_name=str(payload["model_name"]),
            provider=str(payload["provider"]),
            starting_word=str(payload["starting_word"]),
            trial_number=int(payload["trial_number"]),
        )


@dataclass(frozen=True)
class RunConfig:
    models: List[str]
    starting_words_tested: Any
    num_trials: int
    max_workers: int
