import unittest

from lisanbench.completions import CompletionAPI


class CostEstimationTests(unittest.TestCase):
    def test_openrouter_only_models_use_catalog_pricing_for_fallback_estimates(self):
        api = CompletionAPI()

        self.assertAlmostEqual(
            0.75,
            api.estimate_model_catalog_cost_usd(
                "qwen/qwen3-32b",
                1_000_000,
                1_000_000,
            ),
        )
        self.assertAlmostEqual(
            0.7,
            api.estimate_model_catalog_cost_usd(
                "x-ai/grok-4.1-fast",
                1_000_000,
                1_000_000,
            ),
        )
        self.assertAlmostEqual(
            0.0,
            api.estimate_model_catalog_cost_usd(
                "x-ai/grok-4.1-fast:free",
                1_000_000,
                1_000_000,
            ),
        )


if __name__ == "__main__":
    unittest.main()
