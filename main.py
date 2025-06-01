import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from lisan_bench import LisanBench
from model_list import models as all_models
from utils import download_file


def parse_arguments() -> argparse.Namespace:
    """
    Parse and return command line arguments for the LisanBench benchmark.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="LisanBench - Evaluate language models on constrained word chain generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark on all models with default settings
  python main.py

  # Run on specific models with custom temperature
  python main.py --models "openai/gpt-4" "anthropic/claude-3.5-sonnet" --temperature 0.7

  # Run on a single model with more trials
  python main.py --models "openai/gpt-4" --trials 5

  # Resume from existing results file
  python main.py --resume results_20240115_120000.json

  # Use custom number of threads
  python main.py --threads 20

  # Use custom starting words
  python main.py --words "cat" "dog" "bird"
        """
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to benchmark (default: all models). Can specify multiple models.",
        default=None
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for model responses (default: 1.0)"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials per word (default: 1)"
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=59,
        help="Maximum number of parallel threads (default: 59)"
    )

    parser.add_argument(
        "--words",
        nargs="+",
        help="Starting words for the benchmark (default: standard set)",
        default=None
    )

    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from existing results file",
        default=None
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output filename for results (default: auto-generated with timestamp)",
        default=None
    )

    parser.add_argument(
        "--words-file",
        type=str,
        default="words_alpha.txt",
        help="Path to the words dictionary file (default: words_alpha.txt)"
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit"
    )

    return parser.parse_args()


def validate_models(model_names: List[str]) -> List[str]:
    """
    Ensure all provided model names exist in the set of available models.

    Args:
        model_names (List[str]): List of model names to check.

    Returns:
        List[str]: List of valid model names.

    Raises:
        ValueError: If any model name is not available.
    """
    invalid_models = [m for m in model_names if m not in all_models]

    if invalid_models:
        print(f"ERROR: The following models are not available: {', '.join(invalid_models)}")
        print(f"\nAvailable models:")
        for model in sorted(all_models):
            print(f"  - {model}")
        raise ValueError(f"Invalid models: {invalid_models}")

    return model_names


def main() -> None:
    """
    Main entry point for the LisanBench CLI tool.
    Parses arguments, validates configuration, handles resume logic, and runs the benchmark.
    """
    args = parse_arguments()

    # Fast path for listing models: just print and exit
    if args.list_models:
        print("Available models:")
        for model in sorted(all_models):
            print(f"  - {model}")
        sys.exit(0)

    # If resuming, override CLI args with metadata from the provided results file
    if args.resume:
        try:
            with open(args.resume, 'r') as f:
                data = json.load(f)
            metadata = data.get("metadata", {})
            args.temperature = metadata.get("temperature", args.temperature)
            args.trials = metadata.get("num_trials", args.trials)
            args.threads = metadata.get("threads", args.threads)
            saved_models = metadata.get("models_tested")
            saved_words = metadata.get("starting_words_tested")
            if saved_models:
                args.models = saved_models
            if saved_words:
                args.words = saved_words
            args.words_file = metadata.get("words_file", args.words_file)
        except Exception as e:
            print(f"ERROR loading resume file metadata: {e}")
            sys.exit(1)

    # Download dictionary if not present and not overridden
    if args.words_file != "words_alpha.txt":
        if not os.path.exists(args.words_file):
            print(f"ERROR: {args.words_file} not found!")
            sys.exit(1)
    else:
        # Pull fresh dictionary from GitHub repo (if not cached locally)
        download_file("https://github.com/dwyl/english-words/raw/master/words_alpha.txt", Path(args.words_file))

    # Select which models to test, validating against available set
    if args.models:
        try:
            models_to_test = validate_models(args.models)
        except ValueError:
            sys.exit(1)
    else:
        models_to_test = all_models
        print(f"No models specified, using all {len(models_to_test)} available models")

    # Use custom starting words or fallback to default list (somewhat diverse for coverage)
    if args.words:
        starting_words = args.words
    else:
        starting_words = [
            "hat", "mine", "lung", "layer", "pattern", "camping", "avoid", "traveller", "origin", "abysmal"
        ]

    print(f"\nStarting LisanBench")
    print(f"Models to test: {len(models_to_test)}")
    print(f"Starting words: {starting_words}")
    print(f"Temperature: {args.temperature}")
    print(f"Trials per word: {args.trials}")
    print(f"Max threads: {args.threads}")

    if args.resume:
        print(f"Resuming from: {args.resume}")

    # Benchmark setup: may fail if dictionary is missing or file is invalid
    try:
        bench = LisanBench(temperature=args.temperature, words_file=args.words_file)
    except Exception as e:
        print(f"ERROR initializing benchmark: {e}")
        sys.exit(1)

    # Main execution: run the benchmark and handle errors/interrupts gracefully
    try:
        results = bench.run_benchmark(
            models=models_to_test,
            starting_words=starting_words,
            num_trials=args.trials,
            max_workers=args.threads,
            resume_from_file=args.resume or args.output
        )

        bench.print_results(results)

    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        print(f"Partial results saved to: {bench.results_filename}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR during benchmark: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
