import json
import tempfile
import unittest
from pathlib import Path

from lisanbench.providers.types import CompletionResult
from lisanbench.lisan_bench import LisanBench


class ReasoningMetadataTests(unittest.TestCase):
    def _write_words_file(self, directory: str) -> Path:
        words_file = Path(directory) / "words.txt"
        words_file.write_text("cat\nbat\n", encoding="utf-8")
        return words_file

    def test_new_results_file_initializes_reasoning_token_total(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            words_file = self._write_words_file(tmpdir)
            results_file = Path(tmpdir) / "results.json"

            bench = LisanBench(words_file=str(words_file))
            bench.initialize_results_file(str(results_file))

            data = json.loads(results_file.read_text(encoding="utf-8"))
            self.assertEqual(0, data["metadata"]["total_reasoning_tokens"])

    def test_loading_old_results_backfills_reasoning_token_total_from_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            words_file = self._write_words_file(tmpdir)
            results_file = Path(tmpdir) / "old_results.json"
            results_file.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "total_api_cost_usd": 0.01,
                            "total_input_tokens": 10,
                            "total_output_tokens": 20,
                        },
                        "results": {
                            "model-a": [
                                {
                                    "model_name": "model-a",
                                    "starting_word": "cat",
                                    "chain_length": 1,
                                    "word_chain": ["cat", "bat"],
                                    "longest_valid_chain": 1,
                                    "total_valid_links": 1,
                                    "total_invalid_links": 0,
                                    "validity_ratio": 1.0,
                                    "execution_time": 0.1,
                                    "timestamp": "2026-01-01T00:00:00",
                                    "temperature": 1.0,
                                    "raw_response": "cat, bat",
                                    "api_cost_usd": 0.01,
                                    "input_tokens": 10,
                                    "output_tokens": 20,
                                    "reasoning_tokens": 7,
                                    "generation_id": "",
                                    "response_api_cost_usd": 0.01,
                                    "cost_source": "response",
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            bench = LisanBench(words_file=str(words_file))
            bench.load_existing_results(str(results_file))

            self.assertEqual(7, bench.completion_api.api_usage.total_reasoning_tokens)
            self.assertEqual(7, bench.results_data["metadata"]["total_reasoning_tokens"])

    def test_batch_result_usage_is_recorded_once_by_lisan_bench(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            words_file = self._write_words_file(tmpdir)
            bench = LisanBench(words_file=str(words_file), use_batching=True)
            bench.completion_api._call_openai_batch_api = lambda model_name, prompts: [
                CompletionResult(
                    text="cat, bat",
                    cost_usd=0.25,
                    generation_id="",
                    input_tokens=10,
                    output_tokens=5,
                    reasoning_tokens=2,
                )
            ]

            results = bench._run_batch_for_provider(
                "openai",
                [
                    {
                        "model_name": "openai/gpt-4.1",
                        "provider": "openai",
                        "starting_word": "cat",
                        "trial_number": 1,
                    }
                ],
            )

            self.assertEqual(1, len(results))
            self.assertEqual(10, bench.completion_api.api_usage.total_input_tokens)
            self.assertEqual(5, bench.completion_api.api_usage.total_output_tokens)
            self.assertEqual(2, bench.completion_api.api_usage.total_reasoning_tokens)
            self.assertEqual(0.25, bench.completion_api.api_usage.total_cost_usd)


if __name__ == "__main__":
    unittest.main()
