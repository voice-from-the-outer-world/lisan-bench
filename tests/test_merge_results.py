import json
import tempfile
import unittest
from pathlib import Path

import lisanbench.merge_results as merge_module


def _payload(
    model: str,
    cost: float,
    *,
    starting_words=None,
    total_reasoning_tokens: int = 0,
) -> dict:
    if starting_words is None:
        starting_words = ["lamp"]
    return {
        "metadata": {
            "timestamp": "2026-05-28T10:00:00",
            "temperature": 1.0,
            "total_api_cost_usd": cost,
            "total_input_tokens": 10,
            "total_output_tokens": 20,
            "total_reasoning_tokens": total_reasoning_tokens,
            "models_tested": [model],
            "starting_words_tested": starting_words,
            "last_updated": "2026-05-28T10:01:00",
            "num_trials": 1,
            "threads": 1,
            "completion_time": "2026-05-28T10:02:00",
            "status": "completed",
        },
        "results": {
            model: [
                {
                    "model_name": model,
                    "starting_word": "lamp",
                    "longest_valid_chain": 1,
                }
            ]
        },
    }


class MergeResultsAtomicWriteTests(unittest.TestCase):
    def test_merge_files_writes_valid_json_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            first = tmpdir / "first.json"
            second = tmpdir / "second.json"
            output = tmpdir / "merged.json"
            first.write_text(
                json.dumps(_payload("model-a", 1.25, total_reasoning_tokens=30)),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(_payload("model-b", 2.5, total_reasoning_tokens=300)),
                encoding="utf-8",
            )

            merge_module.merge_files(first, second, output, auto_yes=True)

            merged = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(["model-a", "model-b"], merged["metadata"]["models_tested"])
            self.assertEqual(3.75, merged["metadata"]["total_api_cost_usd"])
            self.assertEqual(330, merged["metadata"]["total_reasoning_tokens"])
            self.assertEqual(1, len(merged["results"]["model-a"]))
            self.assertEqual(1, len(merged["results"]["model-b"]))

    def test_merge_files_rejects_different_word_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            first = tmpdir / "first.json"
            second = tmpdir / "second.json"
            output = tmpdir / "merged.json"
            first.write_text(
                json.dumps(_payload("model-a", 1.25, starting_words=["lamp", "cat"])),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(_payload("model-b", 2.5, starting_words=["lamp", "dog"])),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "starting_words_tested do not match"):
                merge_module.merge_files(first, second, output, auto_yes=True)

            self.assertFalse(output.exists())

    def test_merge_files_allows_same_word_set_with_model_specific_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            first = tmpdir / "first.json"
            second = tmpdir / "second.json"
            output = tmpdir / "merged.json"
            first.write_text(
                json.dumps(
                    _payload(
                        "model-a",
                        1.25,
                        starting_words={"model-a": ["lamp", "cat"]},
                    )
                ),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    _payload(
                        "model-b",
                        2.5,
                        starting_words={"model-b": ["cat", "lamp"]},
                    )
                ),
                encoding="utf-8",
            )

            merge_module.merge_files(first, second, output, auto_yes=True)

            merged = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(["model-a", "model-b"], merged["metadata"]["models_tested"])

    def test_failed_write_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            first = tmpdir / "first.json"
            second = tmpdir / "second.json"
            output = tmpdir / "merged.json"
            first.write_text(json.dumps(_payload("model-a", 1.25)), encoding="utf-8")
            second.write_text(json.dumps(_payload("model-b", 2.5)), encoding="utf-8")
            output.write_text('{"existing": true}\n', encoding="utf-8")

            original_merge_results = merge_module.merge_results
            try:
                merge_module.merge_results = lambda *_args, **_kwargs: {"bad": object()}
                with self.assertRaises(TypeError):
                    merge_module.merge_files(first, second, output, auto_yes=True)
            finally:
                merge_module.merge_results = original_merge_results

            self.assertEqual({"existing": True}, json.loads(output.read_text(encoding="utf-8")))
            self.assertEqual([], list(tmpdir.glob(".merged.json.*.tmp")))


if __name__ == "__main__":
    unittest.main()
