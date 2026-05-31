from dataclasses import replace
from typing import Any, Dict, List, Optional, Set

from lisanbench.benchmark_models import Results
from lisanbench.reduced_benchmarks import get_reduced_estimator, is_reasoning_model, reduced_estimator_key
from lisanbench.utils import coerce_word_chain, validate_chain


def average_lengths_by_word(trials: List[Any]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for trial in trials:
        if isinstance(trial, dict):
            word = trial.get("starting_word")
            value = trial.get("longest_valid_chain")
        else:
            word = trial.starting_word
            value = trial.longest_valid_chain
        if not isinstance(word, str) or value is None:
            continue
        try:
            grouped.setdefault(word, []).append(float(value))
        except (TypeError, ValueError):
            continue
    return {
        word: sum(values) / len(values)
        for word, values in grouped.items()
        if values
    }


def print_reduced_extrapolation(results: Dict[str, List[Results]]) -> None:
    rows = []
    for model, trials in results.items():
        estimator = get_reduced_estimator(model)
        if not estimator:
            continue
        estimator_words = estimator["words"]
        averages = average_lengths_by_word(trials)
        if any(word not in averages for word in estimator_words):
            continue
        features = [averages[word] for word in estimator_words]
        predicted = estimator["intercept"] + sum(
            coefficient * value
            for coefficient, value in zip(estimator["coefficients"], features)
        )
        rows.append((model, estimator, predicted, sum(features), len(trials)))

    if not rows:
        return

    rows.sort(key=lambda item: item[2], reverse=True)

    print("\n" + "=" * 120)
    print("REDUCED 15x2 EXTRAPOLATION  (estimated full 50-word Path Length)")
    print("=" * 120)
    print("Estimator source    : reduced_benchmark_estimators.json")
    print("Uncertainty shown   : +/- saved estimator MAE")
    for rank, (model, estimator, predicted, reduced_sum, trial_count) in enumerate(rows, 1):
        model_class = reduced_estimator_key(model)
        loo_r2 = estimator.get("loo_r2")
        loo_mae = (
            estimator.get("simulated_thinking_mae")
            if is_reasoning_model(model) and estimator.get("simulated_thinking_mae") is not None
            else estimator.get("simulated_mae", estimator.get("loo_mae"))
        )
        mae_text = f"{float(loo_mae):.2f}" if isinstance(loo_mae, (int, float)) else "n/a"
        r2_text = f"{float(loo_r2):.6f}" if isinstance(loo_r2, (int, float)) else "n/a"
        print(
            f"{rank:2d}. {model:<30} | "
            f"Estimated Full Score: {predicted:8.2f} +/- {mae_text:<8} | "
            f"Observed 15-word Sum: {reduced_sum:7.2f} | "
            f"Estimator: {model_class:<13} | "
            f"Clean LOO R^2: {r2_text} | "
            f"Trials: {trial_count:3d}"
        )


def print_results(
    *,
    results: Dict[str, List[Results]],
    temperature: float,
    max_tokens: int,
    api_usage: Any,
    valid_words: Optional[Set[str]] = None,
) -> None:
    results = _score_results(results, valid_words)
    total_trials = sum(len(v) for v in results.values())
    print("\n" + "=" * 120)
    print("LISAN-BENCH RESULTS  (all trials)")
    print("=" * 120)
    print(f"Temperature         : {temperature}")
    print(f"Max Tokens          : {max_tokens}")
    print(f"Trials recorded     : {total_trials}")
    print(f"Total API Cost      : ${api_usage.total_cost_usd:.4f}")
    total_tokens = api_usage.total_input_tokens + api_usage.total_output_tokens
    print(f"Total Tokens        : {total_tokens:,}")

    starting_words = {r.starting_word for trials in results.values() for r in trials}

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
        words = set(t.starting_word for t in trials)
        best_trials = []
        for word in words:
            word_trials = [t for t in trials if t.starting_word == word]
            best_trial = max(
                word_trials,
                key=lambda t: (t.longest_valid_chain, t.total_valid_links, -t.total_invalid_links),
            )
            best_trials.append(best_trial)

        sum_longest = sum(t.longest_valid_chain for t in best_trials)
        sum_valid = sum(t.total_valid_links for t in best_trials)
        sum_invalid = sum(t.total_invalid_links for t in best_trials)
        ratio = sum_valid / (sum_valid + sum_invalid) if (sum_valid + sum_invalid) else 0
        print(
            f"{model:<30} | "
            f"Sum Best Chains: {sum_longest:4d} | "
            f"Sum Best Valid Links: {sum_valid:5d} | "
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
            f"Sum Avg Chains: {sum_avg_longest:6.2f} | "
            f"Sum Avg Valid Links: {sum_avg_valid:8.2f} | "
            f"Validity Ratio: {ratio:.3f}"
        )

    print_reduced_extrapolation(results)


def _score_results(
    results: Dict[str, List[Results]],
    valid_words: Optional[Set[str]],
) -> Dict[str, List[Results]]:
    if valid_words is None:
        return results

    scored: Dict[str, List[Results]] = {}
    for model, trials in results.items():
        scored_trials: List[Results] = []
        for trial in trials:
            chain = coerce_word_chain(trial.word_chain, trial.raw_response)
            longest, valid_links, invalid_links = validate_chain(
                chain,
                valid_words,
                trial.starting_word,
            )
            total_links = valid_links + invalid_links
            ratio = valid_links / total_links if total_links else 0.0
            scored_trials.append(
                replace(
                    trial,
                    chain_length=longest,
                    word_chain=chain,
                    longest_valid_chain=longest,
                    total_valid_links=valid_links,
                    total_invalid_links=invalid_links,
                    validity_ratio=ratio,
                )
            )
        scored[model] = scored_trials
    return scored
