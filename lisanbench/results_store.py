import json
import os
import tempfile
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

from lisanbench.benchmark_models import Results


class ResultsStore:
    """Owns benchmark result JSON persistence and resume compatibility."""

    def __init__(
        self,
        *,
        completion_api: Any,
        words_file: str,
        use_batching: bool,
        stream_responses: bool,
        force_openrouter: bool,
    ) -> None:
        self.completion_api = completion_api
        self.words_file = words_file
        self.use_batching = use_batching
        self.stream_responses = stream_responses
        self.force_openrouter = force_openrouter
        self.filename: Optional[str] = None
        self.data: Dict[str, Any] = {}
        self.lock = Lock()

    def initialize(self, filename: Optional[str], *, temperature: float, max_tokens: int) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lisan_bench_results_{timestamp}.json"

        self.filename = filename
        self.data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "total_api_cost_usd": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_reasoning_tokens": 0,
                "models_tested": [],
                "starting_words_tested": [],
                "status": "in_progress",
                "words_file": self.words_file,
                "batching": self.use_batching,
                "streaming": self.stream_responses,
                "force_openrouter": self.force_openrouter,
            },
            "results": {},
        }

        self.save()
        print(f"Initialized results file: {filename}")
        return filename

    def save_unsafe(self) -> None:
        if self.filename is None:
            return

        self.data.setdefault("metadata", {}).update({
            "last_updated": datetime.now().isoformat(),
            "total_api_cost_usd": self.completion_api.api_usage.total_cost_usd,
            "total_input_tokens": self.completion_api.api_usage.total_input_tokens,
            "total_output_tokens": self.completion_api.api_usage.total_output_tokens,
            "total_reasoning_tokens": self.completion_api.api_usage.total_reasoning_tokens,
        })

        try:
            self.atomic_write_json(self.filename, self.data)
        except Exception as e:
            print(f"Warning: Failed to save results to {self.filename}: {e}")

    def save(self) -> None:
        with self.lock:
            self.save_unsafe()

    def load(self, filename: str, generation_ids: List[str]) -> Dict[str, List[Results]]:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"File {filename} not found - starting fresh.")
            return {}
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return {}

        self.filename = filename
        self.data = data if isinstance(data, dict) else {}
        self.data.setdefault("metadata", {})
        self.data.setdefault("results", {})

        results: Dict[str, List[Results]] = {}
        result_groups = self.data.get("results", {})
        if isinstance(result_groups, dict):
            for model, trial_dicts in result_groups.items():
                if not isinstance(trial_dicts, list):
                    continue
                trial_objs: List[Results] = []
                for td in trial_dicts:
                    try:
                        result = Results.from_json(td)
                    except Exception as exc:
                        print(f"Warning: Skipping malformed result row for {model}: {exc}")
                        continue
                    trial_objs.append(result)
                    gen_id = str(getattr(result, "generation_id", "") or "")
                    if gen_id:
                        generation_ids.append(gen_id)
                results[str(model)] = trial_objs

        meta = self.data.get("metadata", {})
        self.completion_api.api_usage.total_cost_usd = self._metadata_float_or_sum(
            meta,
            "total_api_cost_usd",
            "api_cost_usd",
        )
        self.completion_api.api_usage.total_input_tokens = self._metadata_int_or_sum(
            meta,
            "total_input_tokens",
            "input_tokens",
        )
        self.completion_api.api_usage.total_output_tokens = self._metadata_int_or_sum(
            meta,
            "total_output_tokens",
            "output_tokens",
        )

        reasoning_total_present = "total_reasoning_tokens" in meta
        reasoning_total = self._metadata_int_or_sum(
            meta,
            "total_reasoning_tokens",
            "reasoning_tokens",
        )
        if not reasoning_total_present:
            self.data.setdefault("metadata", {})["total_reasoning_tokens"] = reasoning_total
        self.completion_api.api_usage.total_reasoning_tokens = reasoning_total

        print(f"Loaded {sum(len(v) for v in results.values())} trials from {filename}")
        return results

    def append_results_unsafe(self, results_by_model: Dict[str, List[Results]], new_results: List[Results]) -> None:
        self.data.setdefault("results", {})
        for result in new_results:
            results_by_model.setdefault(result.model_name, []).append(result)
            self.data["results"].setdefault(result.model_name, []).append(result.to_json())
        self.save_unsafe()

    def sum_result_int_field(self, field_name: str) -> int:
        total = 0
        result_groups = self.data.get("results", {})
        if not isinstance(result_groups, dict):
            return 0

        for model_results in result_groups.values():
            if not isinstance(model_results, list):
                continue
            for result in model_results:
                if not isinstance(result, dict):
                    continue
                try:
                    total += int(result.get(field_name, 0) or 0)
                except (TypeError, ValueError):
                    continue

        return total

    def sum_result_float_field(self, field_name: str) -> float:
        total = 0.0
        result_groups = self.data.get("results", {})
        if not isinstance(result_groups, dict):
            return 0.0

        for model_results in result_groups.values():
            if not isinstance(model_results, list):
                continue
            for result in model_results:
                if not isinstance(result, dict):
                    continue
                try:
                    total += float(result.get(field_name, 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue

        return total

    def _metadata_int_or_sum(self, meta: Dict[str, Any], metadata_field: str, result_field: str) -> int:
        if metadata_field in meta:
            try:
                return int(meta.get(metadata_field, 0) or 0)
            except (TypeError, ValueError):
                pass
        return self.sum_result_int_field(result_field)

    def _metadata_float_or_sum(self, meta: Dict[str, Any], metadata_field: str, result_field: str) -> float:
        if metadata_field in meta:
            try:
                return float(meta.get(metadata_field, 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
        return self.sum_result_float_field(result_field)

    @staticmethod
    def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
        dir_path = os.path.dirname(path) or "."
        base = os.path.basename(path)
        fd, tmp_path = tempfile.mkstemp(prefix=f".{base}.", suffix=".tmp", dir=dir_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
            raise
