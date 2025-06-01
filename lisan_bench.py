import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional

from completions import CompletionAPI
from utils import (
    create_prompt,
    extract_word_chain,
    load_word_dictionary,
    validate_chain
)


@dataclass
class Results:
    """Store results for a single benchmark run."""
    model_name: str
    starting_word: str
    chain_length: int
    word_chain: List[str]
    longest_valid_chain: int
    total_valid_links: int
    total_invalid_links: int
    validity_ratio: float
    execution_time: float
    timestamp: str
    temperature: float
    raw_response: str
    api_cost_usd: float
    generation_id: str


class LisanBench:
    """Main benchmarking class"""

    def __init__(self, temperature: float = 1.0, words_file: str = "words_alpha.txt"):
        """
        Initialize the benchmark.

        Args:
            temperature: Temperature setting for model responses
            words_file: Path to the dictionary file

        Raises:
            FileNotFoundError: If words file is not found
        """
        self.temperature: float = temperature
        self.completion_api: CompletionAPI = CompletionAPI(temperature)
        self.valid_words: set = load_word_dictionary(words_file)
        print(f"Loaded {len(self.valid_words):,} valid English words from {words_file}")

        self.results_filename: Optional[str] = None
        self.results_data: Dict = {}
        self.file_lock: Lock = Lock()
        self.print_lock: Lock = Lock()
        self.generation_ids: List[str] = []
        self.generation_ids_lock: Lock = Lock()

    def initialize_results_file(self, filename: Optional[str] = None) -> str:
        """
        Initialize the results file for continuous saving.

        Args:
            filename: Optional filename to use

        Returns:
            The filename being used
        """
        with self.file_lock:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"lisan_bench_results_{timestamp}.json"

            self.results_filename = filename

            self.results_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "temperature": self.temperature,
                    "total_api_cost_usd": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "models_tested": [],
                    "starting_words_tested": [],
                    "status": "in_progress"
                },
                "results": {}
            }

            self._save_current_results_unsafe()
            print(f"Initialized results file: {filename}")
            return filename

    def _save_current_results_unsafe(self) -> None:
        """
        Save current results to file - MUST be called with file_lock held.

        Returns:
            None
        """
        if self.results_filename is None:
            return

        self.results_data["metadata"].update({
            "last_updated": datetime.now().isoformat(),
            "total_api_cost_usd": self.completion_api.api_usage.total_cost_usd,
            "total_input_tokens": self.completion_api.api_usage.total_input_tokens,
            "total_output_tokens": self.completion_api.api_usage.total_output_tokens,
        })

        try:
            with open(self.results_filename, 'w') as f:
                json.dump(self.results_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save results to {self.results_filename}: {e}")

    def save_current_results(self) -> None:
        """
        Thread-safe save of current results to file.

        Returns:
            None
        """
        with self.file_lock:
            self._save_current_results_unsafe()

    def load_existing_results(self, filename: str) -> Dict[str, List[Results]]:
        """
        Load an existing JSON results file so we can resume without losing
        any trials.

        Args:
            filename: Path to the results file

        Returns:
            Dict[str, List[Results]]  – model → list of Results (all trials)
        """
        try:
            with open(filename, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"File {filename} not found – starting fresh.")
            return {}
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return {}

        with self.file_lock:
            self.results_filename = filename
            self.results_data = data

        results: Dict[str, List[Results]] = {}

        for model, trial_dicts in data.get("results", {}).items():
            trial_objs: List[Results] = []
            for td in trial_dicts:
                trial_objs.append(Results(**td))
                gen_id = td.get("generation_id")
                if gen_id:
                    with self.generation_ids_lock:
                        self.generation_ids.append(gen_id)
            results[model] = trial_objs

        meta = data.get("metadata", {})
        with self.completion_api.api_usage.lock:
            self.completion_api.api_usage.total_cost_usd = meta.get("total_api_cost_usd", 0.0)
            self.completion_api.api_usage.total_input_tokens = meta.get("total_input_tokens", 0)
            self.completion_api.api_usage.total_output_tokens = meta.get("total_output_tokens", 0)

        print(f"Loaded {sum(len(v) for v in results.values())} trials from {filename}")
        return results

    def evaluate_model_single(
            self,
            model_name: str,
            starting_word: str,
            num_trials: int = 1
    ) -> List[Results]:
        """
        Evaluate a single model with a single starting word, saving results from ALL trials.

        Args:
            model_name: Name of the model to evaluate
            starting_word: Word to start the chain with
            num_trials: Number of successful trials to run

        Returns:
            List of Results objects, one for each successful trial

        Raises:
            Exception: If unable to produce any successful trials
        """
        trial_results: List[Results] = []
        trial = 0
        start_time = time.time()
        attempts = 0

        while trial < num_trials:
            try:
                prompt = create_prompt(starting_word)
                response, api_cost, generation_id, input_tokens, output_tokens = self.completion_api.call_model(
                    model_name, prompt
                )

                with self.generation_ids_lock:
                    self.generation_ids.append(generation_id)

                word_chain = extract_word_chain(response)

                longest_valid_from_start, total_valid_links, total_invalid_links = validate_chain(
                    word_chain, self.valid_words
                )
                validity_ratio = (
                    total_valid_links / (total_valid_links + total_invalid_links)
                    if (total_valid_links + total_invalid_links) > 0
                    else 0
                )

                execution_time = time.time() - start_time

                result = Results(
                    model_name=model_name,
                    starting_word=starting_word,
                    chain_length=len(word_chain),
                    word_chain=word_chain,
                    longest_valid_chain=longest_valid_from_start,
                    total_valid_links=total_valid_links,
                    total_invalid_links=total_invalid_links,
                    validity_ratio=validity_ratio,
                    execution_time=execution_time,
                    timestamp=datetime.now().isoformat(),
                    temperature=self.temperature,
                    raw_response=response,
                    api_cost_usd=api_cost,
                    generation_id=generation_id
                )

                trial_results.append(result)
                trial += 1  # Only increment on success

            except Exception as e:
                with self.print_lock:
                    print(f"Error in trial {trial + 1} for {model_name} with word '{starting_word}': {e}")
            finally:
                attempts += 1

            if attempts > 2 * num_trials:
                print(f"Attempts exceeded. Stopping evaluation for model {model_name} with word '{starting_word} ")
                break

        if not trial_results:
            raise Exception(f"All trials failed for {model_name} with word '{starting_word}'")

        return trial_results

    def evaluate_model_all_words(
            self,
            model_name: str,
            starting_words: List[str],
            completed_results: Dict[str, List[Results]],
            num_trials: int = 1
    ) -> List[Results]:
        """
        Run the remaining trials for each starting word of *model_name*.

        If a word already has >= num_trials trials on disk, we skip it entirely.
        Every new trial is appended to JSON immediately.

        Args:
            model_name: Name of the model to evaluate
            starting_words: List of starting words for the model
            completed_results: Dict of existing results by model name
            num_trials: Number of trials per word

        Returns:
            List of all Results (old and new) for this model
        """
        previous_trials: List[Results] = completed_results.get(model_name, [])
        trials_by_word: Dict[str, List[Results]] = defaultdict(list)
        for t in previous_trials:
            trials_by_word[t.starting_word].append(t)

        new_trials: List[Results] = []

        for word in starting_words:
            already = len(trials_by_word[word])
            if already >= num_trials:
                continue

            try:
                needed = num_trials - already
                fresh = self.evaluate_model_single(model_name, word, needed)
            except Exception as e:
                with self.print_lock:
                    print(f"❌ {model_name} failed on '{word}': {e}")
                continue

            for res in fresh:
                new_trials.append(res)
                trials_by_word[word].append(res)

                with self.file_lock:
                    self.results_data["results"].setdefault(model_name, []).append(
                        asdict(res)
                    )
                    self._save_current_results_unsafe()

                with self.print_lock:
                    print(
                        f"{model_name}: '{word}' - "
                        f"trial {len(trials_by_word[word])}/{num_trials} - "
                        f"valid_chain={res.longest_valid_chain:2d}"
                    )

        return previous_trials + new_trials

    def update_costs_from_openrouter(self) -> None:
        """
        Update costs from OpenRouter API for all generation IDs.

        Returns:
            None
        """
        print("\nRetrieving actual costs from OpenRouter...")

        with self.generation_ids_lock:
            openrouter_ids = list(self.generation_ids)

        if not openrouter_ids:
            print("No generation IDs to update costs for")
            return

        # First, collect only the generation IDs that have zero cost
        openrouter_ids_to_fetch = []

        with self.file_lock:
            for model_name, model_results in self.results_data["results"].items():
                for result in model_results:
                    gen_id = result.get("generation_id")
                    current_cost = result.get("api_cost_usd", 0.0)
                    if gen_id and current_cost == 0.0:
                        openrouter_ids_to_fetch.append(gen_id)

        # Only fetch costs if there are any zero-cost entries to update
        if openrouter_ids_to_fetch:
            costs: Dict[str, float] = self.completion_api.get_openrouter_costs(openrouter_ids_to_fetch)

            with self.file_lock:
                total_cost_update = 0.0

                for model_name, model_results in self.results_data["results"].items():
                    for result in model_results:
                        gen_id = result.get("generation_id")
                        if gen_id and gen_id in costs:
                            old_cost = result.get("api_cost_usd", 0.0)
                            new_cost = costs[gen_id]
                            result["api_cost_usd"] = new_cost
                            total_cost_update += (new_cost - old_cost)

                self.results_data["metadata"]["total_api_cost_usd"] += total_cost_update
                self.completion_api.api_usage.total_cost_usd += total_cost_update

                self._save_current_results_unsafe()

            print(f"Updated costs from OpenRouter. Total cost adjustment: ${total_cost_update:.4f}")
        else:
            print("No zero-cost entries found. Skipping cost fetch from OpenRouter.")

    def run_benchmark(
            self,
            models: List[str],
            starting_words: Optional[List[str]] = None,
            num_trials: int = 1,
            max_workers: int = 59,
            resume_from_file: Optional[str] = None
    ) -> Dict[str, List[Results]]:
        """
        Run the full benchmark with parallelization.

        Args:
            models: List of model names to evaluate
            starting_words: List of starting words
            num_trials: Number of trials per word
            max_workers: Maximum parallel threads
            resume_from_file: Optional file to resume from

        Returns:
            Dictionary of model name to list of Results
        """
        if starting_words is None:
            starting_words = ["hat", "mine", "lung", "layer", "pattern", "camping", "avoid", "traveller", "origin",
                              "abysmal"]

        if resume_from_file and os.path.exists(resume_from_file):
            results = self.load_existing_results(resume_from_file)
        else:
            results = {}
            self.initialize_results_file(resume_from_file)

        with self.file_lock:
            self.results_data["metadata"].update({
                "temperature": self.temperature,
                "num_trials": num_trials,
                "threads": max_workers,
                "models_tested": models,
                "starting_words_tested": starting_words
            })
            self._save_current_results_unsafe()

        print(f"Starting benchmark with temperature={self.temperature}")
        print(f"Max parallel threads: {max_workers}")
        print(
            f"Total combinations: {len(models)} models × {len(starting_words)} words x {num_trials} trials = {len(models) * len(starting_words) * num_trials} evaluations"
        )

        completed_count = sum(len(model_results) for model_results in results.values())
        total_evaluations = len(models) * len(starting_words) * num_trials

        if completed_count > 0:
            print(f"Resuming from existing results: {completed_count}/{total_evaluations} evaluations completed")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_model = {
                executor.submit(
                    self.evaluate_model_all_words,
                    model,
                    starting_words,
                    results,
                    num_trials
                ): model
                for model in models
            }

            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    model_results = future.result()
                    results[model] = model_results

                    current_completed = sum(len(trials) for trials in results.values())
                    with self.print_lock:
                        pct = current_completed / total_evaluations * 100
                        print(
                            f"✅  Model {model} finished! "
                            f"Progress: {current_completed}/{total_evaluations} "
                            f"({pct:.1f} %)"
                        )

                except Exception as e:
                    with self.print_lock:
                        print(f"❌ Model {model} failed completely: {e}")

        self.update_costs_from_openrouter()

        with self.file_lock:
            self.results_data["metadata"]["status"] = "completed"
            self.results_data["metadata"]["completion_time"] = datetime.now().isoformat()
            self._save_current_results_unsafe()

        print(f"\nBenchmark completed! Results saved to: {self.results_filename}")
        print(f"Final total cost: ${self.completion_api.api_usage.total_cost_usd:.4f}")
        return results

    def print_results(self, results: Dict[str, List[Results]]) -> None:
        """
        Pretty-print every trial plus aggregate statistics.

        Args:
            results: Dictionary mapping model name to list of Results

        Returns:
            None
        """
        total_trials = sum(len(v) for v in results.values())
        print("\n" + "=" * 120)
        print("LISAN-BENCH RESULTS  (all trials)")
        print("=" * 120)
        print(f"Temperature         : {self.temperature}")
        print(f"Trials recorded     : {total_trials}")
        print(f"Total API Cost      : ${self.completion_api.api_usage.total_cost_usd:.4f}")
        total_tokens = (
                self.completion_api.api_usage.total_input_tokens
                + self.completion_api.api_usage.total_output_tokens
        )
        print(f"Total Tokens        : {total_tokens:,}")

        starting_words = {
            r.starting_word for trials in results.values() for r in trials
        }

        for sw in sorted(starting_words):
            print(f"\nStarting Word: '{sw}'")
            print("-" * 120)
            rows = []
            for model, trials in results.items():
                these = [t for t in trials if t.starting_word == sw]
                if not these:
                    continue
                best = max(
                    these,
                    key=lambda t: (t.longest_valid_chain, t.total_valid_links),
                )
                rows.append((model, best, len(these)))

            rows.sort(key=lambda x: (x[1].longest_valid_chain, x[1].total_valid_links), reverse=True)

            for rank, (model, best, n_trials) in enumerate(rows, 1):
                print(
                    f"{rank:2d}. {model:<30} | "
                    f"Best Valid Chain: {best.longest_valid_chain:3d} | "
                    f"Trials: {n_trials:2d} | "
                    f"Best Total Valid Links: {best.total_valid_links:3d}"
                )

        print("\n" + "=" * 120)
        print("GLOBAL AGGREGATES  (sum of best trial per word, per model)")
        print("=" * 120)
        for model, trials in results.items():
            if not trials:
                continue
            # Aggregate bests per word
            words = set(t.starting_word for t in trials)
            sum_longest = sum(
                max(t.longest_valid_chain for t in trials if t.starting_word == word)
                for word in words
            )
            sum_valid = sum(
                max(t.total_valid_links for t in trials if t.starting_word == word)
                for word in words
            )
            # Validity ratio uses all trials (sum)
            sum_invalid = sum(t.total_invalid_links for t in trials)
            ratio = sum_valid / (sum_valid + sum_invalid) if (sum_valid + sum_invalid) else 0
            print(
                f"{model:<30} | "
                f"Σ Best Chains: {sum_longest:4d} | "
                f"Σ Best Valid Links: {sum_valid:5d} | "
                f"Validity Ratio: {ratio:.3f}"
            )

        print("\n" + "=" * 120)
        print("GLOBAL AGGREGATES  (sum of averages per word, per model)")
        print("=" * 120)
        for model, trials in results.items():
            if not trials:
                continue
            words = set(t.starting_word for t in trials)
            avg_longest_per_word = []
            avg_valid_per_word = []
            for word in words:
                word_trials = [t for t in trials if t.starting_word == word]
                avg_longest = sum(t.longest_valid_chain for t in word_trials) / len(word_trials)
                avg_valid = sum(t.total_valid_links for t in word_trials) / len(word_trials)
                avg_longest_per_word.append(avg_longest)
                avg_valid_per_word.append(avg_valid)
            sum_avg_longest = sum(avg_longest_per_word)
            sum_avg_valid = sum(avg_valid_per_word)
            sum_invalid = sum(t.total_invalid_links for t in trials)
            sum_valid = sum(t.total_valid_links for t in trials)
            ratio = sum_valid / (sum_valid + sum_invalid) if (sum_valid + sum_invalid) else 0
            print(
                f"{model:<30} | "
                f"Σ Avg Chains: {sum_avg_longest:6.2f} | "
                f"Σ Avg Valid Links: {sum_avg_valid:8.2f} | "
                f"Validity Ratio: {ratio:.3f}"
            )
