import time
from datetime import datetime
from threading import Event, Lock
from typing import Any, Callable, Dict, List, Optional

from lisanbench.benchmark_models import Results, TrialJob
from lisanbench.providers.types import CompletionResult
from lisanbench.utils import create_prompt, extract_word_chain, validate_chain


def build_trial_key(model_name: str, starting_word: str, trial_number: int) -> str:
    return f"{model_name}|{starting_word}|t{trial_number}"


def truncate_error(text: str, limit: int = 160) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


class TrialEvaluator:
    """Evaluates single and provider-batched benchmark trials."""

    def __init__(
        self,
        *,
        completion_api: Any,
        valid_words: set,
        temperature: float,
        generation_ids: List[str],
        generation_ids_lock: Lock,
        print_lock: Lock,
        stop_event: Event,
        mark_inflight_start: Callable[[str, str, str, int, int], None],
        mark_inflight_end: Callable[[str], None],
    ) -> None:
        self.completion_api = completion_api
        self.valid_words = valid_words
        self.temperature = temperature
        self.generation_ids = generation_ids
        self.generation_ids_lock = generation_ids_lock
        self.print_lock = print_lock
        self.stop_event = stop_event
        self.mark_inflight_start = mark_inflight_start
        self.mark_inflight_end = mark_inflight_end

    def evaluate_single_trial(self, model_name: str, starting_word: str, trial_number: int) -> Optional[Results]:
        if self.stop_event.is_set():
            return None

        trial_key = build_trial_key(model_name, starting_word, trial_number)
        max_attempts = 3
        attempts = 0
        transient_failures = 0
        max_transient_failures_before_counting_attempt = 12
        non429_backoff_cap = 10
        rate_limit_backoff_cap = 30
        transient_backoff_cap = 60

        while not self.stop_event.is_set():
            attempt_number = attempts + 1
            self.mark_inflight_start(
                trial_key,
                model_name,
                starting_word,
                trial_number,
                attempt_number,
            )
            try:
                prompt = create_prompt(starting_word)
                start_time = time.time()

                completion = CompletionResult.from_value(
                    self.completion_api.create_completion(model_name, prompt)
                )

                if self.stop_event.is_set():
                    return None

                execution_time = time.time() - start_time

                if completion.generation_id:
                    with self.generation_ids_lock:
                        self.generation_ids.append(completion.generation_id)

                return self._result_from_completion(
                    model_name=model_name,
                    starting_word=starting_word,
                    execution_time=execution_time,
                    completion=completion,
                )

            except Exception as e:
                msg = str(e)
                msg_l = msg.lower()

                if "429" in msg or "Too Many Requests" in msg:
                    backoff = min(rate_limit_backoff_cap, 1 << min(attempts, 6))
                    if not self.stop_event.is_set():
                        with self.print_lock:
                            print(f"[429] {model_name}/{starting_word} t{trial_number}: backoff {backoff}s...")
                    self._sleep_with_cancel(backoff)
                    continue

                transient_http_codes = ("500", "502", "503", "504", "520", "522", "524", "408")
                is_transient_http = any(f"http {code}" in msg_l for code in transient_http_codes)
                is_transient_network = any(
                    token in msg_l
                    for token in (
                        "readtimeout",
                        "connecttimeout",
                        "connection reset",
                        "connection aborted",
                        "remote disconnected",
                        "temporarily unavailable",
                        "server_interrupted",
                        "bad gateway",
                        "gateway timeout",
                        "service unavailable",
                    )
                )
                if is_transient_http or is_transient_network:
                    transient_failures += 1
                    backoff = min(transient_backoff_cap, 1 << min(transient_failures, 6))
                    if not self.stop_event.is_set():
                        with self.print_lock:
                            print(
                                f"[transient] {model_name}/{starting_word} t{trial_number}: "
                                f"{truncate_error(msg)}; backoff {backoff}s "
                                f"(transient {transient_failures}/{max_transient_failures_before_counting_attempt})"
                            )

                    if transient_failures >= max_transient_failures_before_counting_attempt:
                        attempts += 1
                        transient_failures = 0
                        if attempts >= max_attempts or self.stop_event.is_set():
                            return None

                    self._sleep_with_cancel(backoff)
                    continue

                transient_failures = 0
                attempts += 1
                if not self.stop_event.is_set():
                    with self.print_lock:
                        print(
                            f"Error in trial {trial_number} for {model_name} with word '{starting_word}' "
                            f"(attempt {attempts}/{max_attempts}): {e}"
                        )

                if attempts >= max_attempts or self.stop_event.is_set():
                    return None

                backoff = min(non429_backoff_cap, 1 << attempts)
                self._sleep_with_cancel(backoff)
            finally:
                self.mark_inflight_end(trial_key)

        return None

    def run_batch_for_provider(
        self,
        provider: str,
        jobs: List[Dict[str, Any]],
    ) -> Optional[List[Results]]:
        if self.stop_event.is_set() or not jobs:
            return None

        trial_jobs = [TrialJob.from_mapping(job) for job in jobs]
        for job in trial_jobs:
            if job.provider != provider:
                raise ValueError(
                    f"_run_batch_for_provider got mixed providers: expected '{provider}', "
                    f"found '{job.provider}' for model '{job.model_name}'"
                )

        model_names = [job.model_name for job in trial_jobs]
        prompts = [create_prompt(job.starting_word) for job in trial_jobs]

        start_time = time.time()
        batch_outputs = self.completion_api.create_batch(provider, model_names, prompts)
        total_time = time.time() - start_time

        if self.stop_event.is_set():
            return None

        if not isinstance(batch_outputs, list) or len(batch_outputs) != len(trial_jobs):
            raise RuntimeError(
                f"create_batch for provider '{provider}' returned "
                f"{len(batch_outputs) if isinstance(batch_outputs, list) else 'non-list'} items, "
                f"expected {len(trial_jobs)}"
            )

        per_job_time = total_time / max(len(trial_jobs), 1)
        results: List[Results] = []
        error_rows: List[str] = []

        for job, output in zip(trial_jobs, batch_outputs):
            if self.stop_event.is_set():
                break

            completion = CompletionResult.from_value(output)
            if completion.text.lstrip().startswith("[ERROR]"):
                error_rows.append(
                    f"{job.model_name}/{job.starting_word} t{job.trial_number}: "
                    f"{truncate_error(completion.text)}"
                )
                continue

            if completion.generation_id:
                with self.generation_ids_lock:
                    self.generation_ids.append(completion.generation_id)

            results.append(
                self._result_from_completion(
                    model_name=job.model_name,
                    starting_word=job.starting_word,
                    execution_time=per_job_time,
                    completion=completion,
                )
            )

        if error_rows and not self.stop_event.is_set():
            preview = "; ".join(error_rows[:3])
            suffix = "" if len(error_rows) <= 3 else f"; +{len(error_rows) - 3} more"
            with self.print_lock:
                print(
                    f"Batch provider '{provider}' returned {len(error_rows)} error row(s); "
                    f"skipped them: {preview}{suffix}"
                )

        return results or None

    def _result_from_completion(
        self,
        *,
        model_name: str,
        starting_word: str,
        execution_time: float,
        completion: CompletionResult,
    ) -> Results:
        word_chain = extract_word_chain(completion.text)
        longest_valid_from_start, total_valid_links, total_invalid_links = validate_chain(
            word_chain,
            self.valid_words,
            starting_word,
        )
        validity_ratio = (
            total_valid_links / (total_valid_links + total_invalid_links)
            if (total_valid_links + total_invalid_links) > 0
            else 0
        )

        return Results(
            model_name=model_name,
            starting_word=starting_word,
            chain_length=longest_valid_from_start,
            word_chain=word_chain,
            longest_valid_chain=longest_valid_from_start,
            total_valid_links=total_valid_links,
            total_invalid_links=total_invalid_links,
            validity_ratio=validity_ratio,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            temperature=self.temperature,
            raw_response=completion.text,
            api_cost_usd=completion.cost_usd,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            reasoning_tokens=completion.reasoning_tokens,
            generation_id=completion.generation_id,
            response_api_cost_usd=completion.cost_usd,
            cost_source="response" if completion.cost_usd > 0 else "pending",
        )

    def _sleep_with_cancel(self, seconds: float) -> None:
        slept = 0.0
        while slept < seconds and not self.stop_event.is_set():
            time.sleep(0.25)
            slept += 0.25
