from dataclasses import dataclass
from typing import Any, Iterator, Tuple


@dataclass(frozen=True)
class UsageTokens:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class CompletionResult:
    text: str
    cost_usd: float
    generation_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @classmethod
    def from_value(cls, value: Any) -> "CompletionResult":
        if isinstance(value, cls):
            return value
        if isinstance(value, tuple) and len(value) == 6:
            text, cost_usd, generation_id, input_tokens, output_tokens, reasoning_tokens = value
            return cls(
                text=str(text),
                cost_usd=float(cost_usd or 0.0),
                generation_id=str(generation_id or ""),
                input_tokens=int(input_tokens or 0),
                output_tokens=int(output_tokens or 0),
                reasoning_tokens=int(reasoning_tokens or 0),
            )
        raise TypeError(f"Expected CompletionResult or 6-item tuple, got {type(value).__name__}")

    def as_tuple(self) -> Tuple[str, float, str, int, int, int]:
        return (
            self.text,
            self.cost_usd,
            self.generation_id,
            self.input_tokens,
            self.output_tokens,
            self.reasoning_tokens,
        )

    def __iter__(self) -> Iterator[Any]:
        return iter(self.as_tuple())

    def __getitem__(self, index: int) -> Any:
        return self.as_tuple()[index]
