import unittest

from lisanbench.providers.anthropic import _build_anthropic_request_params
from lisanbench.providers.google import (
    GOOGLE_FLEX_TIMEOUT_MS,
    _build_google_client_kwargs,
    _build_google_generation_config,
)
from lisanbench.providers.openrouter import build_reasoning_payload, build_verbosity
from lisanbench.model_names import OPENAI_REASONING_EFFORT_LEVELS, extract_reasoning_effort


class _FakeAnthropicAPI:
    max_tokens = 64_000
    temperature = 1.0


class ReasoningEffortTests(unittest.TestCase):
    def test_openai_effort_levels_do_not_include_max(self):
        self.assertEqual(
            "xhigh",
            extract_reasoning_effort("thinking-xhigh", OPENAI_REASONING_EFFORT_LEVELS),
        )
        self.assertIsNone(
            extract_reasoning_effort("thinking-max", OPENAI_REASONING_EFFORT_LEVELS)
        )

    def test_anthropic_adaptive_effort_accepts_max(self):
        params = _build_anthropic_request_params(
            _FakeAnthropicAPI(),
            "claude-opus-4-8",
            "prompt",
            "thinking-max",
        )

        self.assertEqual({"type": "adaptive"}, params["thinking"])
        self.assertEqual({"effort": "max"}, params["output_config"])

    def test_openrouter_opus_uses_verbosity_for_max(self):
        model = "anthropic/claude-opus-4.8:thinking-max"

        self.assertEqual({"enabled": True}, build_reasoning_payload(model))
        self.assertEqual("max", build_verbosity(model))

    def test_openrouter_generic_reasoning_effort_does_not_emit_max(self):
        self.assertEqual(
            {"enabled": True},
            build_reasoning_payload("openai/gpt-5.4:thinking-max"),
        )

    def test_openrouter_implicit_reasoning_models_are_enabled(self):
        cases = {
            "openai/gpt-5-nano": {"enabled": True, "effort": "medium"},
            "deepseek/deepseek-r1-0528": {"enabled": True},
            "z-ai/glm-4.7": {"enabled": True},
            "minimax/minimax-m1": {"enabled": True},
            "x-ai/grok-4": {"enabled": True},
            "google/gemini-3.0-pro": {"enabled": True, "effort": "high"},
        }

        for model_name, expected in cases.items():
            with self.subTest(model_name=model_name):
                self.assertEqual(expected, build_reasoning_payload(model_name))

    def test_openrouter_implicit_reasoning_can_be_explicitly_disabled(self):
        self.assertEqual(
            {"enabled": False},
            build_reasoning_payload("openai/gpt-5-nano:thinking-none"),
        )

    def test_openrouter_adaptive_thinking_none_is_disabled(self):
        self.assertEqual(
            {"enabled": False},
            build_reasoning_payload("anthropic/claude-opus-4.8:thinking-none"),
        )

    def test_google_thinking_levels_are_model_specific(self):
        self.assertEqual(
            {
                "temperature": 1.0,
                "max_output_tokens": 64_000,
                "thinking_config": {"thinking_level": "low"},
            },
            _build_google_generation_config(
                1.0, 64_000, "gemini-3-pro-preview", "thinking-low"
            ),
        )
        self.assertEqual(
            {"temperature": 1.0, "max_output_tokens": 64_000},
            _build_google_generation_config(
                1.0, 64_000, "gemini-3-pro-preview", "thinking-medium"
            ),
        )
        self.assertEqual(
            {"temperature": 1.0, "max_output_tokens": 64_000},
            _build_google_generation_config(
                1.0, 64_000, "gemini-3.1-pro-preview", "thinking-medium"
            ),
        )
        self.assertEqual(
            {
                "temperature": 1.0,
                "max_output_tokens": 64_000,
                "thinking_config": {"thinking_level": "minimal"},
            },
            _build_google_generation_config(
                1.0, 64_000, "gemini-3.5-flash", "thinking-minimal"
            ),
        )

    def test_google_generation_config_can_request_flex_service_tier(self):
        self.assertEqual(
            {"temperature": 1.0, "max_output_tokens": 64_000, "service_tier": "flex"},
            _build_google_generation_config(
                1.0,
                64_000,
                "gemini-2.0-flash",
                "",
                service_tier="flex",
            ),
        )

    def test_google_flex_client_uses_one_hour_timeout(self):
        kwargs = _build_google_client_kwargs("test-key", "flex")

        self.assertEqual("test-key", kwargs["api_key"])
        self.assertEqual(GOOGLE_FLEX_TIMEOUT_MS, kwargs["http_options"].timeout)


if __name__ == "__main__":
    unittest.main()
