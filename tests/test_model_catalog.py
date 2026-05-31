import unittest

from lisanbench.model_catalog import (
    _build_runtime_structures,
    _completion_provider_for,
    _load_catalog_entries,
    get_model_service_tier,
    get_openrouter_service_tier,
)


class ModelCatalogRoutingTests(unittest.TestCase):
    def test_openrouter_provider_only_uses_explicit_or_flex_provider_defaults(self):
        data = _build_runtime_structures(
            [
                {
                    "id": "openai/o3:thinking-medium",
                    "company": "openai",
                    "completion_provider": "openrouter",
                    "completion_model_id": "o3-2025-04-16",
                },
                {
                    "id": "anthropic/claude-test",
                    "company": "anthropic",
                    "completion_provider": "openrouter",
                    "completion_model_id": "claude-test-native",
                    "openrouter": {"provider_only": ["anthropic"]},
                },
                {
                    "id": "qwen/qwen-test",
                    "company": "qwen",
                    "completion_provider": "openrouter",
                    "openrouter": {"provider_only": "alibaba"},
                },
            ]
        )

        provider_only = data["OPENROUTER_PROVIDER_ONLY_BY_MODEL"]
        self.assertEqual(provider_only["openai/o3"], ["openai"])
        self.assertEqual(provider_only["anthropic/claude-test"], ["anthropic"])
        self.assertEqual(provider_only["qwen/qwen-test"], ["alibaba"])

    def test_service_tier_defaults_to_flex_for_openai_and_google(self):
        data = _build_runtime_structures(
            [
                {
                    "id": "google/gemini-flex",
                    "company": "google",
                    "completion_model_id": "gemini-flex",
                },
                {
                    "id": "openai/gpt-unmarked",
                    "company": "openai",
                    "completion_model_id": "gpt-unmarked",
                },
                {
                    "id": "qwen/qwen-unmarked",
                    "company": "qwen",
                },
            ]
        )

        self.assertEqual(data["SERVICE_TIER_BY_MODEL"]["google/gemini-flex"], "flex")
        self.assertEqual(data["SERVICE_TIER_BY_MODEL"]["openai/gpt-unmarked"], "flex")
        self.assertEqual(data["OPENROUTER_SERVICE_TIER_BY_MODEL"]["google/gemini-flex"], "flex")
        self.assertEqual(data["OPENROUTER_SERVICE_TIER_BY_MODEL"]["openai/gpt-unmarked"], "flex")
        self.assertNotIn("qwen/qwen-unmarked", data["SERVICE_TIER_BY_MODEL"])
        self.assertNotIn("qwen/qwen-unmarked", data["OPENROUTER_SERVICE_TIER_BY_MODEL"])

    def test_openrouter_service_tier_rejects_priority(self):
        with self.assertRaisesRegex(RuntimeError, "priority service tier"):
            _build_runtime_structures(
                [
                    {
                        "id": "openai/gpt-priority",
                        "company": "openai",
                        "completion_model_id": "gpt-priority",
                        "openrouter": {"service_tier": "priority"},
                    },
                ]
            )

    def test_openrouter_service_tier_rejects_unknown_values(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported service tier"):
            _build_runtime_structures(
                [
                    {
                        "id": "google/gemini-test",
                        "company": "google",
                        "completion_model_id": "gemini-test",
                        "openrouter": {"service_tier": "turbo"},
                    },
                ]
            )

    def test_company_does_not_control_completion_dispatch(self):
        data = _build_runtime_structures(
            [
                {
                    "id": "google/gemini-test:thinking-high",
                    "company": "google",
                    "completion_model_id": "gemini-test-native",
                }
            ]
        )

        self.assertEqual(
            data["COMPLETION_PROVIDER_BY_MODEL"]["google/gemini-test:thinking-high"],
            "openrouter",
        )
        self.assertEqual(
            data["BATCH_COMPLETION_PROVIDER_BY_MODEL"]["google/gemini-test:thinking-high"],
            "openrouter",
        )
        self.assertNotIn("google/gemini-test", data["GOOGLE_MODEL_MAPPING"])
        self.assertEqual(
            data["OPENROUTER_PROVIDER_ONLY_BY_MODEL"]["google/gemini-test"],
            ["google-ai-studio"],
        )

    def test_completion_provider_is_the_explicit_internal_dispatch_override(self):
        data = _build_runtime_structures(
            [
                {
                    "id": "custom/openai-compatible-model",
                    "company": "metadata-company",
                    "completion_provider": "openai",
                    "batch_completion_provider": "openai",
                    "completion_model_id": "native-openai-compatible-model",
                    "pricing": {"in": 1.25, "out": 2.5},
                }
            ]
        )

        self.assertEqual(data["COMPLETION_PROVIDER_BY_MODEL"]["custom/openai-compatible-model"], "openai")
        self.assertEqual(data["BATCH_COMPLETION_PROVIDER_BY_MODEL"]["custom/openai-compatible-model"], "openai")
        self.assertEqual(
            data["OPENAI_MODEL_MAPPING"]["custom/openai-compatible-model"],
            "native-openai-compatible-model",
        )
        self.assertEqual(data["OPENAI_PRICING"]["openai-compatible-model"], {"in": 1.25, "out": 2.5})
        self.assertNotIn("custom/openai-compatible-model", data["OPENROUTER_PROVIDER_ONLY_BY_MODEL"])

    def test_legacy_catalog_keys_still_parse(self):
        data = _build_runtime_structures(
            [
                {
                    "id": "legacy/direct-model",
                    "provider": "legacy-company",
                    "routing_provider": "zenmux",
                    "provider_model_id": "legacy/native-model",
                }
            ]
        )

        self.assertEqual(data["COMPANY_BY_MODEL"]["legacy/direct-model"], "legacy-company")
        self.assertEqual(data["COMPLETION_PROVIDER_BY_MODEL"]["legacy/direct-model"], "zenmux")
        self.assertEqual(data["ZENMUX_MODEL_MAPPING"]["legacy/direct-model"], "legacy/native-model")

    def test_real_catalog_completion_model_ids_have_supported_completion_provider(self):
        supported_completion_providers = {
            "anthropic",
            "google",
            "moonshotai",
            "openai",
            "openrouter",
            "z-ai",
            "zenmux",
        }
        unsupported = []
        for entry in _load_catalog_entries():
            if "completion_model_id" not in entry:
                continue
            model_id = str(entry["id"]).strip()
            base_model = model_id.split(":", 1)[0]
            completion_provider = _completion_provider_for(entry, base_model)
            if completion_provider not in supported_completion_providers:
                unsupported.append((model_id, completion_provider))

        self.assertEqual([], unsupported)

    def test_real_catalog_company_and_completion_provider_metadata_is_normalized(self):
        entries = _load_catalog_entries()
        by_id = {str(entry["id"]).strip(): entry for entry in entries}
        metadata_errors = []

        for entry in entries:
            model_id = str(entry["id"]).strip()
            company = entry.get("company")
            completion_provider = entry.get("completion_provider")
            prefix = model_id.split("/", 1)[0] if "/" in model_id else model_id

            for legacy_key in ("provider", "routing_provider", "provider_model_id"):
                if legacy_key in entry:
                    metadata_errors.append((model_id, f"legacy key {legacy_key}"))
            if company is None:
                metadata_errors.append((model_id, "missing company"))
            if completion_provider is None:
                metadata_errors.append((model_id, "missing completion_provider"))
            if company in {"google", "openai"} and entry.get("completion_model_id"):
                if entry.get("batch_completion_provider") != company:
                    metadata_errors.append((model_id, "missing direct batch_completion_provider"))
            if prefix == "openai" and company != "openai":
                metadata_errors.append((model_id, company))
            if prefix == "anthropic" and company != "anthropic":
                metadata_errors.append((model_id, company))
            if prefix == "google" and company != "google":
                metadata_errors.append((model_id, company))
            if prefix == "x-ai" and company != "x-ai":
                metadata_errors.append((model_id, company))
            if prefix.lower() == "qwen" and company != "qwen":
                metadata_errors.append((model_id, company))
            if prefix == "z-ai" and company != "z-ai":
                metadata_errors.append((model_id, company))
            if company in {"google-ai-studio", "xai", "alibaba", "Qwen", "novita"}:
                metadata_errors.append((model_id, f"stale company {company}"))
            if company in {"openai", "google"}:
                if get_model_service_tier(model_id) != "flex":
                    metadata_errors.append((model_id, "missing direct flex service_tier"))
                if get_openrouter_service_tier(model_id) != "flex":
                    metadata_errors.append((model_id, "missing OpenRouter flex service_tier"))

        self.assertEqual([], metadata_errors)
        self.assertEqual(
            {"provider_only": "novita"},
            by_id["z-ai/glm-5"]["openrouter"],
        )
        self.assertEqual(
            {"provider_only": "novita"},
            by_id["z-ai/glm-5:thinking"]["openrouter"],
        )
        self.assertEqual("openrouter", by_id["z-ai/glm-5"]["completion_provider"])
        self.assertEqual("openrouter", by_id["z-ai/glm-5:thinking"]["completion_provider"])
        for model_id in (
            "zenmux/deepseek-v4-pro",
            "zenmux/deepseek-v4-pro:thinking-high",
            "zenmux/deepseek-v4-flash",
            "zenmux/deepseek-v4-flash:thinking-high",
        ):
            entry = by_id[model_id]
            base_model = model_id.split(":", 1)[0]
            self.assertEqual("deepseek", entry["company"])
            self.assertEqual("zenmux", entry["completion_provider"])
            self.assertEqual("zenmux", _completion_provider_for(entry, base_model))


if __name__ == "__main__":
    unittest.main()
