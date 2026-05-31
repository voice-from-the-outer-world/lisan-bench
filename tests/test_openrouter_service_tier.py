import json
import unittest

from lisanbench.providers.openrouter import call_openrouter_api


class _FakeUsage:
    def __init__(self) -> None:
        self.calls = []

    def add_usage(self, input_tokens, output_tokens, reasoning_tokens, cost) -> None:
        self.calls.append((input_tokens, output_tokens, reasoning_tokens, cost))


class _FakeResponse:
    ok = True
    status_code = 200
    reason = "OK"

    def __init__(self) -> None:
        self.payload = {
            "id": "gen-test",
            "service_tier": "flex",
            "choices": [{"message": {"content": "cat, bat"}}],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "cost": 0.0001,
            },
        }
        self.text = json.dumps(self.payload)
        self.closed = False

    def json(self):
        return self.payload

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self) -> None:
        self.last_json = None

    def post(self, url, *, headers, json, timeout, stream):
        self.last_json = json
        return _FakeResponse()


class _FakeAPI:
    def __init__(self) -> None:
        self.temperature = 1.0
        self.max_tokens = 256
        self.stream_responses = False
        self.api_usage = _FakeUsage()
        self.session = _FakeSession()
        self.logged_request = None

    def _require_openrouter_key(self) -> str:
        return "test-key"

    def _get_session(self):
        return self.session

    def _log_raw_interaction(self, **kwargs) -> None:
        self.logged_request = kwargs.get("request_payload")


class OpenRouterServiceTierTests(unittest.TestCase):
    def test_openai_requests_flex_tier_and_openai_provider(self):
        api = _FakeAPI()

        call_openrouter_api(api, "openai/gpt-4.1", "Generate a chain.")

        self.assertEqual(api.session.last_json["service_tier"], "flex")
        self.assertEqual(api.session.last_json["provider"], {"only": ["openai"]})
        self.assertEqual(api.logged_request["service_tier"], "flex")
        self.assertEqual(api.logged_request["provider"], {"only": ["openai"]})

    def test_gemini_35_flash_requests_flex_tier(self):
        api = _FakeAPI()

        result = call_openrouter_api(
            api,
            "google/gemini-3.5-flash:thinking-none",
            "Generate a chain.",
        )

        self.assertEqual(api.session.last_json["service_tier"], "flex")
        self.assertEqual(api.session.last_json["provider"], {"only": ["google-ai-studio"]})
        self.assertEqual(api.logged_request["service_tier"], "flex")
        self.assertEqual(api.logged_request["provider"], {"only": ["google-ai-studio"]})
        self.assertEqual(result[0], "cat, bat")
        self.assertEqual(result[1], 0.0001)

    def test_unmarked_models_do_not_send_service_tier(self):
        api = _FakeAPI()

        call_openrouter_api(api, "qwen/qwen3-32b", "Generate a chain.")

        self.assertNotIn("service_tier", api.session.last_json)
        self.assertNotIn("service_tier", api.logged_request)


if __name__ == "__main__":
    unittest.main()
