import os
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Event, Thread
from typing import Any, Dict, List, Optional, Tuple

from lisanbench.model_routing import batch_model_has_completion_provider_mapping, resolve_model_route


class BenchmarkRunner:
    """Schedules trials, drains futures, persists progress, and finalizes metadata."""

    def __init__(self, bench: Any) -> None:
        self.bench = bench

    def run_benchmark(
        self,
        models: List[str],
        starting_words: Optional[List[str] | Dict[str, List[str]]] = None,
        num_trials: int = 1,
        max_workers: int = 50,
        resume_from_file: Optional[str] = None,
        output_file: Optional[str] = None,
    ):
        bench = self.bench
        if not starting_words:
            starting_words = []
        if isinstance(starting_words, dict):
            starting_words_by_model = {
                model: list(words)
                for model, words in starting_words.items()
            }
            metadata_starting_words: Any = starting_words_by_model
        else:
            common_words = list(starting_words)
            starting_words_by_model = {
                model: common_words
                for model in models
            }
            metadata_starting_words = common_words

        if output_file and os.path.exists(output_file):
            raise FileExistsError(
                f"Output file already exists: {output_file}. "
                "Use --resume to continue an existing results file."
            )

        if resume_from_file:
            if not os.path.exists(resume_from_file):
                raise FileNotFoundError(f"Resume file not found: {resume_from_file}")
            results = bench.load_existing_results(resume_from_file)
            if output_file:
                bench.results_filename = output_file
        else:
            results = {}
            bench.initialize_results_file(output_file)

        bench.results_data["metadata"].update({
            "temperature": bench.temperature,
            "max_tokens": bench.max_tokens,
            "num_trials": num_trials,
            "threads": max_workers,
            "models_tested": models,
            "starting_words_tested": metadata_starting_words,
            "batching": bench.use_batching,
            "streaming": bench.stream_responses,
            "force_openrouter": bench.force_openrouter,
            "words_file": bench.words_file,
        })
        bench.save_current_results()

        print(f"Starting benchmark with temperature={bench.temperature}")
        print(f"Max output tokens per request: {bench.max_tokens}")
        print(f"Max tokens per request: {bench.completion_api.max_tokens}")
        print(f"Max parallel threads: {max_workers}")
        total_evaluations = sum(
            len(starting_words_by_model.get(model, [])) * num_trials
            for model in models
        )
        print(
            f"Total combinations: {len(models)} models x model-specific words x {num_trials} trials = "
            f"{total_evaluations} evaluations"
        )

        completed_count = sum(len(model_results) for model_results in results.values())
        if completed_count > 0:
            print(f"Resuming from existing results: {completed_count}/{total_evaluations} evaluations completed")

        provider_by_model: Dict[str, str] = {}
        if bench.use_batching:
            unsupported: List[Tuple[str, str]] = []
            unsupported_model_api: List[Tuple[str, str]] = []
            for m in models:
                route = resolve_model_route(m, batching=True)
                provider = route.backend
                provider_by_model[m] = provider
                if not route.batch_supported:
                    unsupported.append((m, provider))
                    continue
                if not batch_model_has_completion_provider_mapping(m, provider):
                    unsupported_model_api.append((m, provider))
            if unsupported:
                detail = ", ".join(f"{m} (provider={p})" for m, p in unsupported)
                raise ValueError(
                    "self.use_batching=True is only supported for openai, google, or anthropic models.\n"
                    f"Unsupported models: {detail}"
                )
            if unsupported_model_api:
                detail = ", ".join(f"{m} (provider={p})" for m, p in unsupported_model_api)
                raise ValueError(
                    "These models are not supported by the direct provider batch APIs in this project. "
                    "Use --no-batching (OpenRouter path) for them.\n"
                    f"Unsupported batch models: {detail}"
                )

        interrupted = False
        executor = ThreadPoolExecutor(max_workers=max_workers)
        future_to_key = {}
        heartbeat_stop_event = Event()
        heartbeat_thread: Optional[Thread] = None

        try:
            if not bench.use_batching:
                for model in models:
                    for word in starting_words_by_model.get(model, []):
                        existing_trials = [t for t in results.get(model, []) if t.starting_word == word]
                        trials_needed = num_trials - len(existing_trials)
                        for trial_idx in range(trials_needed):
                            fut = executor.submit(
                                bench._evaluate_single_trial,
                                model,
                                word,
                                len(existing_trials) + trial_idx + 1,
                            )
                            future_to_key[fut] = ("single", model, word)
            else:
                jobs_by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

                for model in models:
                    provider = provider_by_model[model]
                    for word in starting_words_by_model.get(model, []):
                        existing_trials = [t for t in results.get(model, []) if t.starting_word == word]
                        trials_needed = num_trials - len(existing_trials)
                        if trials_needed <= 0:
                            continue

                        for trial_idx in range(trials_needed):
                            jobs_by_group.setdefault((provider, model), []).append({
                                "model_name": model,
                                "provider": provider,
                                "starting_word": word,
                                "trial_number": len(existing_trials) + trial_idx + 1,
                            })

                for (provider, model), jobs in jobs_by_group.items():
                    if not jobs:
                        continue
                    fut = executor.submit(bench._run_batch_for_provider, provider, jobs)
                    future_to_key[fut] = ("batch", provider, model)

            if future_to_key:
                heartbeat_thread = self._start_heartbeat(
                    executor=executor,
                    future_to_key=future_to_key,
                    results=results,
                    total_evaluations=total_evaluations,
                    completed_count=completed_count,
                    max_workers=max_workers,
                    heartbeat_stop_event=heartbeat_stop_event,
                )

            for future in as_completed(future_to_key):
                mode, a, b = future_to_key[future]

                try:
                    result_or_results = future.result()
                    if not result_or_results:
                        continue

                    if isinstance(result_or_results, list):
                        new_results = result_or_results
                    else:
                        new_results = [result_or_results]

                    with bench.file_lock:
                        bench.results_store.append_results_unsafe(results, new_results)
                        for result in new_results:
                            bench._record_recent_result(result)

                    current_completed = sum(len(trials) for trials in results.values())
                    pct = (current_completed / total_evaluations * 100) if total_evaluations else 100.0

                    with bench.print_lock:
                        if mode == "single":
                            last = new_results[-1]
                            print(
                                f"{last.model_name}: '{last.starting_word}' - "
                                f"{last.longest_valid_chain:2d} valid - "
                                f"out={last.output_tokens} tok - "
                                f"time={last.execution_time:.1f}s - "
                                f"Progress: ({pct:.1f}%)"
                            )
                        else:
                            for last in new_results:
                                print(
                                    f"{last.model_name}: '{last.starting_word}' - "
                                    f"{last.longest_valid_chain:2d} valid - "
                                    f"out={last.output_tokens} tok - "
                                    f"time={last.execution_time:.1f}s - "
                                    f"Progress: ({pct:.1f}%)"
                                )

                except CancelledError:
                    with bench.print_lock:
                        if mode == "single":
                            print(f"Task {a} - '{b}' cancelled.")
                        else:
                            print(f"Batched task for provider '{a}' cancelled.")
                except Exception as e:
                    with bench.print_lock:
                        if mode == "single":
                            print(f"Task {a} - '{b}' failed: {e}")
                        else:
                            print(f"Batched task for provider '{a}' failed: {e}")

        except KeyboardInterrupt:
            interrupted = True
            bench.stop_event.set()
            print("\nKeyboardInterrupt received - cancelling pending tasks...")
            for f in future_to_key:
                f.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

        finally:
            heartbeat_stop_event.set()
            if heartbeat_thread and heartbeat_thread.is_alive():
                heartbeat_thread.join(timeout=2.0)
            try:
                executor.shutdown(wait=True)
            except Exception:
                pass

        if interrupted:
            bench.results_data["metadata"]["status"] = "interrupted"
            bench.results_data["metadata"]["completion_time"] = datetime.now().isoformat()
            bench.save_current_results()
            print(f"Partial results saved to: {bench.results_filename}")
        else:
            bench.update_costs_from_openrouter()
            actual_results = sum(len(trials) for trials in results.values())
            if actual_results < total_evaluations:
                bench.results_data["metadata"]["status"] = "completed_partial"
                bench.results_data["metadata"]["missing_trials"] = total_evaluations - actual_results
                print(
                    f"\nWARNING: Benchmark finished with {actual_results}/{total_evaluations} trials. "
                    f"{total_evaluations - actual_results} trials failed."
                )
            else:
                bench.results_data["metadata"]["status"] = "completed"
            bench.results_data["metadata"]["completion_time"] = datetime.now().isoformat()
            bench.save_current_results()
            print(f"\nBenchmark results saved to: {bench.results_filename}")
            print(f"Final total cost: ${bench.completion_api.api_usage.total_cost_usd:.4f}")

        return results

    def _start_heartbeat(
        self,
        *,
        executor: ThreadPoolExecutor,
        future_to_key: Dict[Any, Any],
        results: Dict[str, Any],
        total_evaluations: int,
        completed_count: int,
        max_workers: int,
        heartbeat_stop_event: Event,
    ) -> Thread:
        bench = self.bench
        heartbeat_interval = bench.heartbeat_interval_seconds
        last_completed_snapshot = completed_count
        last_heartbeat_at = time.time()

        print(
            f"Heartbeat logging enabled: every {heartbeat_interval}s "
            f"(set BENCH_HEARTBEAT_SECONDS to override)."
        )

        def heartbeat_loop() -> None:
            nonlocal last_completed_snapshot, last_heartbeat_at
            while not heartbeat_stop_event.wait(heartbeat_interval):
                if bench.stop_event.is_set():
                    continue

                done_count = sum(1 for f in future_to_key if f.done())
                running_count = sum(1 for f in future_to_key if f.running())
                pending_count = len(future_to_key) - done_count - running_count

                with bench.file_lock:
                    current_completed = sum(len(trials) for trials in results.values())

                now = time.time()
                elapsed = max(now - last_heartbeat_at, 1e-9)
                completed_delta = current_completed - last_completed_snapshot
                per_min = completed_delta / elapsed * 60.0

                with bench.inflight_lock:
                    inflight_items = list(bench.inflight_requests.values())
                oldest = sorted(
                    inflight_items,
                    key=lambda item: item.get("started_at", 0.0),
                )[:3]

                oldest_descriptions: List[str] = []
                for item in oldest:
                    age_s = max(0.0, now - float(item.get("started_at", now)))
                    oldest_descriptions.append(
                        f"{item.get('model')}/'{item.get('word')}'"
                        f" t{item.get('trial')} a{item.get('attempt')} "
                        f"for {age_s:.0f}s"
                    )
                oldest_text = "; ".join(oldest_descriptions) if oldest_descriptions else "none"

                with bench.recent_results_lock:
                    recent_items = list(bench.recent_results)[-2:]
                recent_text = " | ".join(
                    (
                        f"{item['model']}/'{item['word']}' "
                        f"valid={item['valid']} out={item['out_tokens']} "
                        f"time={item['exec_s']:.1f}s preview='{item['preview']}'"
                    )
                    for item in recent_items
                )
                if not recent_text:
                    recent_text = "none"

                alive_workers = "n/a"
                try:
                    worker_threads = getattr(executor, "_threads", None)
                    if worker_threads is not None:
                        alive_workers = str(sum(1 for t in worker_threads if t.is_alive()))
                except Exception:
                    pass

                with bench.print_lock:
                    print(
                        "[heartbeat] "
                        f"completed={current_completed}/{total_evaluations} "
                        f"(+{completed_delta} in {elapsed:.0f}s, {per_min:.2f}/min) | "
                        f"futures running={running_count} pending={pending_count} done={done_count} | "
                        f"inflight={len(inflight_items)} | "
                        f"workers_alive={alive_workers}/{max_workers}"
                    )
                    print(f"[heartbeat] oldest_inflight: {oldest_text}")
                    print(f"[heartbeat] recent_results: {recent_text}")

                last_completed_snapshot = current_completed
                last_heartbeat_at = now

        heartbeat_thread = Thread(
            target=heartbeat_loop,
            name="lisanbench-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        return heartbeat_thread
