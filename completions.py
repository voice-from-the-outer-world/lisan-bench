import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from threading import Lock
from typing import Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()


@dataclass
class APIUsage:
    """Track API usage statistics."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0


class ThreadSafeAPIUsage:
    """Thread-safe API usage tracking."""

    def __init__(self):
        self.lock = Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    def add_usage(self, input_tokens: int, output_tokens: int, cost: float) -> None:
        """Add usage statistics in a thread-safe manner."""
        with self.lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost_usd += cost


class CompletionAPI:
    """Handle API completions for both OpenRouter"""

    # Models that require special reasoning parameters
    REASONING_MODELS = {
        'anthropic/claude-sonnet-4:thinking-16k',
        'anthropic/claude-opus-4:thinking-16k'
    }

    def __init__(self, temperature: float = 1.0):
        """
        Initialize the completion API handler.

        Args:
            temperature: Temperature setting for model responses

        Raises:
            ValueError: If required API keys are not found
        """
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        self.temperature = temperature
        self.api_usage = ThreadSafeAPIUsage()

    def call_model(self, model_name: str, prompt: str) -> Tuple[str, float, str, int, int]:
        """
        Call a model via appropriate API and return response with cost and token counts.

        Args:
            model_name: Name of the model to call
            prompt: The prompt to send to the model

        Returns:
            Tuple of (response_text, cost_usd, generation_id, input_tokens, output_tokens)

        Raises:
            Exception: If API call fails
        """
        return self._call_openrouter_api(model_name, prompt)

    def _call_openrouter_api(self, model_name: str, prompt: str) -> Tuple[str, float, str, int, int]:
        """
        Call OpenRouter API for models using urllib.request.

        Args:
            model_name: Name of the model
            prompt: The prompt to send

        Returns:
            Tuple of (response_text, cost_usd, generation_id, input_tokens, output_tokens)

        Raises:
            Exception: If API call fails
        """
        try:
            # Prepare request data
            request_data = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.temperature
            }

            # Add reasoning parameters for specific models
            if model_name in self.REASONING_MODELS:
                request_data["reasoning"] = {
                    "max_tokens": 16384
                }

            # Convert data to JSON bytes
            json_data = json.dumps(request_data).encode('utf-8')

            # Create the request
            req = urllib.request.Request(
                url="https://openrouter.ai/api/v1/chat/completions",
                data=json_data,
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                method="POST"
            )

            # Make the request
            with urllib.request.urlopen(req) as response:
                response_data = response.read().decode('utf-8')
                completion_data = json.loads(response_data)

            response_text = completion_data["choices"][0]["message"]["content"]
            generation_id = completion_data["id"]

            # Track token usage
            input_tokens = 0
            output_tokens = 0
            usage = completion_data.get("usage")
            if usage:
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

            # For now, we'll use 0 cost and update it later with get_openrouter_costs
            # Store generation_id for later cost retrieval
            preliminary_cost = 0.0

            # Update thread-safe API usage (cost will be updated later)
            self.api_usage.add_usage(input_tokens, output_tokens, preliminary_cost)

            return response_text, preliminary_cost, generation_id, input_tokens, output_tokens

        except Exception as e:
            raise Exception(f"OpenRouter API error: {e}")

    def get_openrouter_costs(self, generation_ids: List[str]) -> Dict[str, float]:
        """
        Get actual costs from OpenRouter generation API for multiple generation IDs.

        Args:
            generation_ids: List of generation IDs to get costs for

        Returns:
            Dictionary mapping generation_id to cost in USD
        """
        costs = {}

        for generation_id in generation_ids:
            try:
                # Build URL with query parameters
                params = urllib.parse.urlencode({"id": generation_id})
                url = f"https://openrouter.ai/api/v1/generation?{params}"

                # Create the request
                req = urllib.request.Request(
                    url=url,
                    headers={"Authorization": f"Bearer {self.openrouter_api_key}"}
                )

                # Make the request
                with urllib.request.urlopen(req) as response:
                    response_data = response.read().decode('utf-8')
                    data = json.loads(response_data)

                cost = data.get("data", {}).get("total_cost", 0.0)
                costs[generation_id] = cost

            except Exception as e:
                print(f"Warning: Could not retrieve OpenRouter cost for {generation_id}: {e}")
                costs[generation_id] = 0.0

        return costs
