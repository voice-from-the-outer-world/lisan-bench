import unittest
from unittest.mock import patch

from lisanbench.providers.types import CompletionResult
from lisanbench.completions import CompletionAPI
from lisanbench.model_routing import (
    batch_model_has_completion_provider_mapping,
    required_api_key_envs,
    resolve_model_route,
)
from lisanbench.reduced_benchmarks import is_reasoning_model
from lisanbench.visualization import _normalize_model_label


class CompletionRoutingBehaviorTests(unittest.TestCase):
    def _api_with_marked_routes(self, *, force_openrouter: bool = False) -> CompletionAPI:
        def marked(name):
            return lambda model, prompt: CompletionResult(
                text=name,
                cost_usd=0.0,
                generation_id="",
            )

        api = CompletionAPI(force_openrouter=force_openrouter)
        api._call_openrouter_api = marked("openrouter")
        api._call_openai_api = marked("openai")
        api._call_google_aistudio = marked("google")
        api._call_moonshot_api = marked("moonshotai")
        api._call_zai_api = marked("z-ai")
        api._call_zenmux_api = marked("zenmux")
        return api

    def test_non_batch_completion_routing_uses_catalog_completion_provider(self):
        cases = {
            "moonshotai/kimi-k2.5": "moonshotai",
            "z-ai/glm-4.7": "z-ai",
            "z-ai/glm-5": "openrouter",
            "zenmux/deepseek-v4-pro": "zenmux",
            "google/gemini-3.0-pro": "google",
            "google/gemini-3.5-flash:thinking-high": "openrouter",
            "openai/gpt-4.1": "openrouter",
            "anthropic/claude-opus-4.8:thinking-high": "openrouter",
        }
        api = self._api_with_marked_routes()

        for model_name, expected_route in cases.items():
            with self.subTest(model_name=model_name):
                self.assertEqual(
                    expected_route,
                    api.create_completion(model_name, "prompt")[0],
                )

    def test_route_resolver_matches_non_batch_dispatch_rules(self):
        cases = {
            "moonshotai/kimi-k2.5": "moonshotai",
            "z-ai/glm-4.7": "z-ai",
            "z-ai/glm-5": "openrouter",
            "zenmux/deepseek-v4-pro": "zenmux",
            "google/gemini-3.0-pro": "google",
            "google/gemini-3.5-flash:thinking-high": "openrouter",
            "openai/gpt-4.1": "openrouter",
            "anthropic/claude-opus-4.8:thinking-high": "openrouter",
        }

        for model_name, expected_route in cases.items():
            with self.subTest(model_name=model_name):
                self.assertEqual(expected_route, resolve_model_route(model_name).backend)

    def test_route_resolver_reports_required_keys_in_current_order(self):
        self.assertEqual(
            [
                "OPENROUTER_API_KEY",
                "GOOGLE_API_KEY",
                "MOONSHOT_API_KEY",
                "ZAI_API_KEY",
                "ZENMUX_API_KEY",
            ],
            required_api_key_envs(
                [
                    "openai/gpt-4.1",
                    "google/gemini-3.0-pro",
                    "moonshotai/kimi-k2.5",
                    "z-ai/glm-4.7",
                    "zenmux/deepseek-v4-pro",
                ]
            ),
        )

    def test_openai_direct_route_is_available_when_catalog_selects_it(self):
        api = self._api_with_marked_routes()

        with patch.dict(
            "lisanbench.model_routing.COMPLETION_PROVIDER_BY_MODEL",
            {"openai/gpt-4.1": "openai"},
        ):
            self.assertEqual(
                "openai",
                resolve_model_route("openai/gpt-4.1").backend,
            )
            self.assertEqual(
                "openai",
                api.create_completion("openai/gpt-4.1", "prompt")[0],
            )
            self.assertEqual(
                ["OPENAI_API_KEY"],
                required_api_key_envs(["openai/gpt-4.1"]),
            )

    def test_batch_route_resolver_uses_catalog_provider_and_mapping_support(self):
        openai_route = resolve_model_route("openai/gpt-4.1", batching=True)
        qwen_route = resolve_model_route("qwen/qwen3-32b", batching=True)

        self.assertEqual("openai", openai_route.backend)
        self.assertTrue(openai_route.batch_supported)
        self.assertTrue(batch_model_has_completion_provider_mapping("openai/gpt-4.1", "openai"))
        self.assertEqual("openrouter", qwen_route.backend)
        self.assertFalse(qwen_route.batch_supported)

    def test_create_batch_records_batch_usage_once(self):
        api = CompletionAPI()
        api._call_openai_batch_api = lambda model, prompts: [
            CompletionResult(
                text="cat, bat",
                cost_usd=0.25,
                generation_id="",
                input_tokens=10,
                output_tokens=5,
                reasoning_tokens=2,
            )
        ]

        outputs = api.create_batch("openai", ["openai/gpt-4.1"], ["prompt"])

        self.assertEqual(outputs[0].input_tokens, 10)
        self.assertEqual(api.api_usage.total_input_tokens, 10)
        self.assertEqual(api.api_usage.total_output_tokens, 5)
        self.assertEqual(api.api_usage.total_reasoning_tokens, 2)
        self.assertEqual(api.api_usage.total_cost_usd, 0.25)

    def test_force_openrouter_overrides_direct_routes(self):
        api = self._api_with_marked_routes(force_openrouter=True)

        for model_name in (
            "moonshotai/kimi-k2.5",
            "z-ai/glm-4.7",
            "zenmux/deepseek-v4-pro",
            "google/gemini-3.0-pro",
        ):
            with self.subTest(model_name=model_name):
                self.assertEqual(
                    "openrouter",
                    api.create_completion(model_name, "prompt")[0],
                )


class ModelReasoningBehaviorTests(unittest.TestCase):
    def test_reasoning_classification_matches_current_rules(self):
        cases = {
            "openai/gpt-5": True,
            "openai/gpt-5.4-mini:thinking-none": False,
            "x-ai/grok-4.1-fast": False,
            "x-ai/grok-4.1-fast:thinking": True,
            "qwen/qwen3-235b-a22b-thinking-2507": True,
            "moonshotai/kimi-k2-thinking": True,
            "allenai/olmo-3-32b-think": True,
            "deepseek/deepseek-r1-0528": True,
            "qwen/qwen3-coder": False,
        }

        for model_name, expected in cases.items():
            with self.subTest(model_name=model_name):
                self.assertIs(expected, is_reasoning_model(model_name))

    def test_visualization_labels_match_current_rules(self):
        cases = {
            "claude-opus-4.8:thinking-max": "Opus 4.8 (max)",
            "gpt-5": "GPT 5 (medium)",
            "gpt-5.4-mini:thinking-none": "GPT 5.4 Mini",
            "grok-4.1-fast": "Grok 4.1 Fast",
            "grok-4.1-fast:thinking": "Grok 4.1 Fast (thinking)",
            "qwen3-235b-a22b-thinking-2507": "Qwen3 235B A22B 2507 (thinking)",
            "kimi-k2-thinking": "Kimi K2 (thinking)",
            "olmo-3-32b-think": "Olmo 3 32B (thinking)",
            "deepseek-r1-0528": "Deepseek R1 0528 (thinking)",
            "qwen3-coder": "Qwen3 Coder",
        }

        for model_name, expected in cases.items():
            with self.subTest(model_name=model_name):
                self.assertEqual(expected, _normalize_model_label(model_name))


if __name__ == "__main__":
    unittest.main()
