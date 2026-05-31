import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lisanbench.main as main_module


class FakeBench:
    captured_run_kwargs = None

    def __init__(self, **_kwargs):
        self.temperature = 1.0
        self.max_tokens = 100
        self.results_filename = "unused.json"

    def run_benchmark(self, **kwargs):
        type(self).captured_run_kwargs = kwargs
        return {}

    def print_results(self, _results):
        return None


class MainOutputHandlingTests(unittest.TestCase):
    def setUp(self):
        FakeBench.captured_run_kwargs = None

    def test_output_is_not_passed_as_resume_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new-results.json"
            argv = [
                "main.py",
                "--models",
                "openai/gpt-4.1",
                "--words",
                "lamp",
                "--trials",
                "1",
                "--output",
                str(output),
            ]
            route = SimpleNamespace(backend="openrouter")

            with patch("sys.argv", argv), patch.dict(
                "os.environ",
                {"OPENROUTER_API_KEY": "test-key"},
                clear=False,
            ), patch.object(
                main_module,
                "ensure_default_word_dictionary",
            ), patch.object(
                main_module,
                "resolve_model_route",
                return_value=route,
            ), patch.object(
                main_module,
                "LisanBench",
                FakeBench,
            ):
                main_module.main()

            self.assertIsNotNone(FakeBench.captured_run_kwargs)
            self.assertIsNone(FakeBench.captured_run_kwargs["resume_from_file"])
            self.assertEqual(str(output), FakeBench.captured_run_kwargs["output_file"])

    def test_existing_output_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "existing-results.json"
            output.write_text("already here\n", encoding="utf-8")
            argv = [
                "main.py",
                "--models",
                "openai/gpt-4.1",
                "--words",
                "lamp",
                "--trials",
                "1",
                "--output",
                str(output),
            ]

            with patch("sys.argv", argv), patch.object(
                main_module,
                "ensure_default_word_dictionary",
            ), patch.object(
                main_module,
                "LisanBench",
                FakeBench,
            ):
                with self.assertRaises(SystemExit) as cm:
                    main_module.main()

            self.assertEqual(1, cm.exception.code)
            self.assertIsNone(FakeBench.captured_run_kwargs)


if __name__ == "__main__":
    unittest.main()
