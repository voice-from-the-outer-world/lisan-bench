import os
import time
from collections import deque
from datetime import datetime
from threading import Event, Lock
from typing import Any, Dict, List, Optional

from lisanbench.benchmark_models import Results
from lisanbench.benchmark_runner import BenchmarkRunner
from lisanbench.completions import CompletionAPI
from lisanbench.cost_reconciliation import CostReconciler
from lisanbench.reporting import average_lengths_by_word, print_reduced_extrapolation, print_results
from lisanbench.results_store import ResultsStore
from lisanbench.trial_evaluator import TrialEvaluator, build_trial_key, truncate_error
from lisanbench.utils import DEFAULT_WORDS_FILE, ensure_default_word_dictionary, load_word_dictionary


class LisanBench:
    """Compatibility facade for benchmark orchestration."""

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        words_file: str = DEFAULT_WORDS_FILE,
        use_batching: bool = False,
        stream_responses: bool = True,
        force_openrouter: bool = False,
    ):
        self.completion_api: CompletionAPI = CompletionAPI(
            temperature=temperature,
            max_tokens=max_tokens,
            use_batching=use_batching,
            stream_responses=stream_responses,
            force_openrouter=force_openrouter,
        )
        self.temperature: float = self.completion_api.temperature
        self.max_tokens: int = self.completion_api.max_tokens
        self.stream_responses: bool = stream_responses
        if words_file == DEFAULT_WORDS_FILE:
            ensure_default_word_dictionary()
        self.words_file: str = words_file
        self.valid_words: set = load_word_dictionary(words_file)
        print(f"Loaded {len(self.valid_words):,} valid English words from {words_file}")

        self.use_batching: bool = use_batching
        self.force_openrouter: bool = force_openrouter
        self.print_lock: Lock = Lock()
        self.generation_ids: List[str] = []
        self.generation_ids_lock: Lock = Lock()
        self.inflight_lock: Lock = Lock()
        self.inflight_requests: Dict[str, Dict[str, Any]] = {}
        self.recent_results_lock: Lock = Lock()
        self.recent_results: deque = deque(maxlen=8)
        self.heartbeat_interval_seconds: int = self._read_positive_int_env("BENCH_HEARTBEAT_SECONDS", 300)
        self.stop_event: Event = Event()

        self.results_store = ResultsStore(
            completion_api=self.completion_api,
            words_file=self.words_file,
            use_batching=self.use_batching,
            stream_responses=self.stream_responses,
            force_openrouter=self.force_openrouter,
        )
        self.file_lock = self.results_store.lock
        self.cost_reconciler = CostReconciler(
            completion_api=self.completion_api,
            results_store=self.results_store,
            generation_ids=self.generation_ids,
        )
        self.trial_evaluator = TrialEvaluator(
            completion_api=self.completion_api,
            valid_words=self.valid_words,
            temperature=self.temperature,
            generation_ids=self.generation_ids,
            generation_ids_lock=self.generation_ids_lock,
            print_lock=self.print_lock,
            stop_event=self.stop_event,
            mark_inflight_start=self._mark_inflight_start,
            mark_inflight_end=self._mark_inflight_end,
        )
        self.runner = BenchmarkRunner(self)

    @property
    def results_filename(self) -> Optional[str]:
        return self.results_store.filename

    @results_filename.setter
    def results_filename(self, value: Optional[str]) -> None:
        self.results_store.filename = value

    @property
    def results_data(self) -> Dict[str, Any]:
        return self.results_store.data

    @results_data.setter
    def results_data(self, value: Dict[str, Any]) -> None:
        self.results_store.data = value

    @staticmethod
    def _read_positive_int_env(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        raw = raw.strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            return default
        return value if value > 0 else default

    @staticmethod
    def _build_trial_key(model_name: str, starting_word: str, trial_number: int) -> str:
        return build_trial_key(model_name, starting_word, trial_number)

    @staticmethod
    def _preview_response(text: str, limit: int = 80) -> str:
        if not text:
            return ""
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    @staticmethod
    def _truncate_error(text: str, limit: int = 160) -> str:
        return truncate_error(text, limit)

    def _mark_inflight_start(
        self,
        trial_key: str,
        model_name: str,
        starting_word: str,
        trial_number: int,
        attempt_number: int,
    ) -> None:
        with self.inflight_lock:
            self.inflight_requests[trial_key] = {
                "model": model_name,
                "word": starting_word,
                "trial": trial_number,
                "attempt": attempt_number,
                "started_at": time.time(),
            }

    def _mark_inflight_end(self, trial_key: str) -> None:
        with self.inflight_lock:
            self.inflight_requests.pop(trial_key, None)

    def _record_recent_result(self, result: Results) -> None:
        with self.recent_results_lock:
            self.recent_results.append({
                "timestamp": datetime.now().isoformat(),
                "model": result.model_name,
                "word": result.starting_word,
                "trial": None,
                "valid": result.longest_valid_chain,
                "out_tokens": result.output_tokens,
                "exec_s": result.execution_time,
                "preview": self._preview_response(result.raw_response),
            })

    def initialize_results_file(self, filename: Optional[str] = None) -> str:
        return self.results_store.initialize(filename, temperature=self.temperature, max_tokens=self.max_tokens)

    def _atomic_write_json(self, path: str, data: Dict[str, Any]) -> None:
        self.results_store.atomic_write_json(path, data)

    def _save_current_results_unsafe(self) -> None:
        self.results_store.save_unsafe()

    def save_current_results(self) -> None:
        self.results_store.save()

    @staticmethod
    def _sum_result_int_field(data: Dict[str, Any], field_name: str) -> int:
        total = 0
        result_groups = data.get("results", {})
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

    def load_existing_results(self, filename: str) -> Dict[str, List[Results]]:
        return self.results_store.load(filename, self.generation_ids)

    def _lookup_response_cost_from_raw(self, generation_id: Optional[str]) -> float:
        return self.cost_reconciler.lookup_response_cost_from_raw(generation_id)

    def update_costs_from_openrouter(self) -> None:
        self.cost_reconciler.update_costs_from_openrouter()

    def run_benchmark(
        self,
        models: List[str],
        starting_words: Optional[List[str] | Dict[str, List[str]]] = None,
        num_trials: int = 1,
        max_workers: int = 50,
        resume_from_file: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> Dict[str, List[Results]]:
        return self.runner.run_benchmark(
            models=models,
            starting_words=starting_words,
            num_trials=num_trials,
            max_workers=max_workers,
            resume_from_file=resume_from_file,
            output_file=output_file,
        )

    def _evaluate_single_trial(self, model_name: str, starting_word: str, trial_number: int) -> Optional[Results]:
        return self.trial_evaluator.evaluate_single_trial(model_name, starting_word, trial_number)

    def _run_batch_for_provider(
        self,
        provider: str,
        jobs: List[Dict[str, Any]],
    ) -> Optional[List[Results]]:
        return self.trial_evaluator.run_batch_for_provider(provider, jobs)

    @staticmethod
    def _average_lengths_by_word(trials: List[Any]) -> Dict[str, float]:
        return average_lengths_by_word(trials)

    def _print_reduced_extrapolation(self, results: Dict[str, List[Results]]) -> None:
        print_reduced_extrapolation(results)

    def print_results(self, results: Dict[str, List[Results]]) -> None:
        print_results(
            results=results,
            temperature=self.temperature,
            max_tokens=self.completion_api.max_tokens,
            api_usage=self.completion_api.api_usage,
            valid_words=self.valid_words,
        )
