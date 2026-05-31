import unittest

from lisanbench.providers.anthropic import _build_anthropic_request_params
from lisanbench.providers.openrouter import build_reasoning_payload, build_verbosity


class _FakeAnthropicAPI:
    max_tokens = 64_000
    temperature = 1.0


class ReasoningRoutingTests(unittest.TestCase):
    def test_openrouter_opus_adaptive_effort_levels_use_verbosity(self):
        levels = ("low", "medium", "high", "xhigh", "max")

        for model in ("anthropic/claude-opus-4.7", "anthropic/claude-opus-4.8"):
            for level in levels:
                model_name = f"{model}:thinking-{level}"
                with self.subTest(model_name=model_name):
                    self.assertEqual({"enabled": True}, build_reasoning_payload(model_name))
                    self.assertEqual(level, build_verbosity(model_name))

    def test_anthropic_opus_adaptive_effort_levels_use_output_config(self):
        levels = ("low", "medium", "high", "xhigh", "max")
        api = _FakeAnthropicAPI()

        for provider_model in ("claude-opus-4-7", "claude-opus-4-8"):
            for level in levels:
                with self.subTest(provider_model=provider_model, level=level):
                    params = _build_anthropic_request_params(
                        api,
                        provider_model,
                        "prompt",
                        f"thinking-{level}",
                    )
                    self.assertEqual({"type": "adaptive"}, params["thinking"])
                    self.assertEqual({"effort": level}, params["output_config"])
                    self.assertNotIn("temperature", params)


if __name__ == "__main__":
    unittest.main()
