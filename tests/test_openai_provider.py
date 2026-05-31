import os
import unittest
from unittest.mock import patch

from lisanbench.providers.openai import (
    build_openai_chat_request,
    call_openai_api,
    parse_openai_chat_completion,
)


class _FakeOpenAIResponse:
    def model_dump(self):
        return {
            "id": "chatcmpl-test",
            "choices": [{"message": {"content": "cat, bat"}}],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        }


class _FakeChatCompletions:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeOpenAIResponse()


class _FakeOpenAIClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _FakeChatCompletions()})()


class _FakeAPI:
    max_tokens = 256
    temperature = 0.7

    def __init__(self):
        self.logged_request = None
        self.logged_response = None

    def _resolve_provider_model(self, provider, model_name, mapping, *, allow_fallback):
        self.resolved = (provider, model_name, allow_fallback)
        return "gpt-5.4", "thinking-xhigh"

    def estimate_billed_cost_usd(self, provider, model_name, input_tokens, output_tokens):
        self.cost_args = (provider, model_name, input_tokens, output_tokens)
        return 0.123

    def _log_raw_interaction(self, **kwargs):
        self.logged_request = kwargs.get("request_payload")
        self.logged_response = kwargs.get("response_payload")


class OpenAIProviderTests(unittest.TestCase):
    def test_build_openai_chat_request_includes_reasoning_and_service_tier(self):
        request = build_openai_chat_request(
            model="gpt-5.4",
            prompt="prompt",
            max_tokens=256,
            temperature=0.7,
            suffix="thinking-xhigh",
            service_tier="flex",
        )

        self.assertEqual(request["model"], "gpt-5.4")
        self.assertEqual(request["messages"], [{"role": "user", "content": "prompt"}])
        self.assertEqual(request["max_completion_tokens"], 256)
        self.assertEqual(request["temperature"], 0.7)
        self.assertEqual(request["reasoning_effort"], "xhigh")
        self.assertEqual(request["service_tier"], "flex")

    def test_parse_openai_chat_completion_extracts_usage(self):
        text, generation_id, input_tokens, output_tokens, reasoning_tokens = (
            parse_openai_chat_completion(_FakeOpenAIResponse().model_dump())
        )

        self.assertEqual(text, "cat, bat")
        self.assertEqual(generation_id, "chatcmpl-test")
        self.assertEqual(input_tokens, 11)
        self.assertEqual(output_tokens, 7)
        self.assertEqual(reasoning_tokens, 3)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_call_openai_api_uses_direct_chat_completions_shape(self):
        fake_client = _FakeOpenAIClient()

        with patch("lisanbench.providers.openai.OpenAI", return_value=fake_client) as openai_cls:
            result = call_openai_api(
                _FakeAPI(),
                "openai/gpt-5.4:thinking-xhigh",
                "Generate a chain.",
            )

        openai_cls.assert_called_once_with(api_key="test-key", timeout=3600.0)
        request = fake_client.chat.completions.last_kwargs
        self.assertEqual(request["model"], "gpt-5.4")
        self.assertEqual(request["messages"], [{"role": "user", "content": "Generate a chain."}])
        self.assertEqual(request["max_completion_tokens"], 256)
        self.assertEqual(request["temperature"], 0.7)
        self.assertEqual(request["reasoning_effort"], "xhigh")
        self.assertEqual(request["service_tier"], "flex")
        self.assertEqual(request["timeout"], 3600.0)

        self.assertEqual(result.text, "cat, bat")
        self.assertEqual(result.cost_usd, 0.123)
        self.assertEqual(result.generation_id, "")
        self.assertEqual(result.input_tokens, 11)
        self.assertEqual(result.output_tokens, 7)
        self.assertEqual(result.reasoning_tokens, 3)


if __name__ == "__main__":
    unittest.main()
