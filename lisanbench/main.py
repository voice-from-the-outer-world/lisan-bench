import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from lisanbench.lisan_bench import LisanBench
from lisanbench.model_catalog import models as all_models
from lisanbench.model_routing import required_api_key_envs, resolve_model_route
from lisanbench.reduced_benchmarks import REDUCED_NUM_TRIALS, get_reduced_words
from lisanbench.utils import DEFAULT_WORDS_FILE, ensure_default_word_dictionary, resolve_words_file


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
  uv run python main.py

  # Run on specific models with custom temperature
  uv run python main.py --models "openai/gpt-4" "anthropic/claude-3.5-sonnet" --temperature 0.7

  # Run on a single model with more trials
  uv run python main.py --models "openai/gpt-4" --trials 5

  # Resume from existing results file
  uv run python main.py --resume results_20240115_120000.json

  # Use custom number of threads
  uv run python main.py --threads 20

  # Use custom starting words
  uv run python main.py --words "cat" "dog" "bird"

  # Use provider batch APIs instead of regular completion routing
  uv run python main.py --batching
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
        default=None,
        help="Temperature for model responses. Defaults to TEMPERATURE from the environment, or 1.0."
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum output tokens per request. Overrides MAX_TOKENS from the environment when set."
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per word (default: 3)"
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=50,
        help="Maximum number of parallel threads (default: 50)"
    )

    parser.add_argument(
        "--words",
        nargs="+",
        help="Starting words for the benchmark (default: standard set)",
        default=None
    )

    parser.add_argument(
        "--reduced-15x2",
        "--reduced-10x3",
        dest="reduced_15x2",
        action="store_true",
        default=False,
        help=(
            "Run the documented reduced benchmark preset: 15 predictive starting "
            "words and 2 trials per word. Cannot be combined with --words. "
            "--reduced-10x3 is accepted as a deprecated alias."
        ),
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
        default=DEFAULT_WORDS_FILE,
        help=f"Path to the words dictionary file (default: {DEFAULT_WORDS_FILE})"
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit"
    )

    parser.add_argument(
        "--batching",
        action="store_true",
        help="Use provider batch APIs (OpenAI Batch, Anthropic Message Batches, Gemini Batch Mode). "
             "Set provider keys for selected models (OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY)."
    )

    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable incremental streaming responses from OpenRouter (default: enabled). "
             "Use --no-streaming to buffer full responses instead."
    )

    parser.add_argument(
        "--force-openrouter",
        action="store_true",
        default=False,
        help="Route ALL model requests through OpenRouter, ignoring direct provider APIs "
             "(OpenAI, Moonshot, Z.AI, Google AI Studio)."
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
    trials_flag_passed = any(
        arg == "--trials" or arg.startswith("--trials=")
        for arg in sys.argv[1:]
    )

    if args.reduced_15x2 and args.words:
        print("ERROR: --reduced-15x2 cannot be combined with --words.")
        sys.exit(1)
    if args.reduced_15x2 and trials_flag_passed and args.trials != REDUCED_NUM_TRIALS:
        print(
            f"ERROR: --reduced-15x2 uses exactly {REDUCED_NUM_TRIALS} trials; "
            f"omit --trials or pass --trials {REDUCED_NUM_TRIALS}."
        )
        sys.exit(1)

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
            reduced_flag_passed = any(a in {"--reduced-15x2", "--reduced-10x3"} for a in sys.argv[1:])
            temperature_flag_passed = any(
                arg == "--temperature" or arg.startswith("--temperature=")
                for arg in sys.argv[1:]
            )
            if "temperature" in metadata and not temperature_flag_passed:
                args.temperature = metadata.get("temperature", args.temperature)
            max_tokens_flag_passed = any(arg.startswith("--max-tokens") for arg in sys.argv[1:])
            if "max_tokens" in metadata and not max_tokens_flag_passed:
                args.max_tokens = metadata.get("max_tokens", args.max_tokens)
            args.trials = metadata.get("num_trials", args.trials)
            if reduced_flag_passed:
                args.trials = REDUCED_NUM_TRIALS
            args.threads = metadata.get("threads", args.threads)
            saved_models = metadata.get("models_tested")
            saved_words = metadata.get("starting_words_tested")
            if saved_models:
                args.models = saved_models
            if saved_words:
                args.words = saved_words
            if reduced_flag_passed:
                args.words = None
            args.words_file = metadata.get("words_file", args.words_file)
            streaming_flag_passed = any(
                arg in ("--streaming", "--no-streaming") for arg in sys.argv[1:]
            )
            if "streaming" in metadata and not streaming_flag_passed:
                args.streaming = bool(metadata.get("streaming", args.streaming))
            # carry batching flag from metadata only if user didn't pass it
            if "batching" in metadata and not any(a == "--batching" for a in sys.argv[1:]):
                args.batching = bool(metadata["batching"])
            if "force_openrouter" in metadata and not any(a == "--force-openrouter" for a in sys.argv[1:]):
                args.force_openrouter = bool(metadata["force_openrouter"])
        except Exception as e:
            print(f"ERROR loading resume file metadata: {e}")
            sys.exit(1)

    if args.output and Path(args.output).exists():
        print(
            f"ERROR: Output file already exists: {args.output}. "
            "Use --resume to continue an existing results file."
        )
        sys.exit(1)

    # Download/regenerate the pinned default dictionary if needed.
    if args.words_file != DEFAULT_WORDS_FILE:
        if not resolve_words_file(args.words_file).exists():
            print(f"ERROR: {args.words_file} not found!")
            sys.exit(1)
    else:
        ensure_default_word_dictionary()

    # Select models
    if args.models:
        try:
            models_to_test = validate_models(args.models)
        except ValueError:
            sys.exit(1)
    else:
        models_to_test = all_models
        print(f"No models specified, using all {len(models_to_test)} available models")

    if args.batching:
        args.streaming = False
        args.threads = 1

    # Validate required API keys for the selected routing mode/models.
    if args.batching:
        missing = [
            env_name
            for env_name in required_api_key_envs(models_to_test, batching=True)
            if not os.getenv(env_name)
        ]
        if missing:
            print(f"ERROR: Missing required API keys for batching mode: {', '.join(missing)}")
            sys.exit(1)
    elif args.force_openrouter:
        if not os.getenv("OPENROUTER_API_KEY"):
            print("ERROR: OPENROUTER_API_KEY is required for --force-openrouter mode.")
            sys.exit(1)
    else:
        routes = [resolve_model_route(m) for m in models_to_test]
        uses_openrouter = any(route.backend == "openrouter" for route in routes)
        uses_openai_direct = any(route.backend == "openai" for route in routes)
        uses_google_direct = any(route.backend == "google" for route in routes)
        uses_moonshot_direct = any(route.backend == "moonshotai" for route in routes)
        uses_zai_direct = any(route.backend == "z-ai" for route in routes)
        uses_zenmux_direct = any(route.backend == "zenmux" for route in routes)

        if uses_openrouter and not os.getenv("OPENROUTER_API_KEY"):
            print("ERROR: OPENROUTER_API_KEY is required unless all selected models use direct provider APIs.")
            sys.exit(1)
        if uses_openai_direct and not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY is required for the selected OpenAI direct models.")
            sys.exit(1)
        if uses_google_direct and not os.getenv("GOOGLE_API_KEY"):
            print("ERROR: GOOGLE_API_KEY is required for the selected Google AI Studio direct models.")
            sys.exit(1)
        if uses_moonshot_direct and not os.getenv("MOONSHOT_API_KEY"):
            print("ERROR: MOONSHOT_API_KEY is required for moonshotai/* direct API models.")
            sys.exit(1)
        if uses_zai_direct and not os.getenv("ZAI_API_KEY"):
            print("ERROR: ZAI_API_KEY is required for z-ai/* direct API models.")
            sys.exit(1)
        if uses_zenmux_direct and not os.getenv("ZENMUX_API_KEY"):
            print("ERROR: ZENMUX_API_KEY is required for zenmux/* direct API models.")
            sys.exit(1)

    # Starting words
    if args.reduced_15x2:
        starting_words = {
            model: get_reduced_words(model)
            for model in models_to_test
        }
        args.trials = REDUCED_NUM_TRIALS
    elif args.words:
        starting_words = args.words
    else:
        starting_words = ['countries', 'selection', 'computers', 'following', 'relations', 'completed', 'careers',
                          'carried', 'shipping', 'reached', 'tracking', 'created', 'housing', 'leaders', 'flags',
                          'score', 'yards', 'means', 'camping', 'spam', 'brass', 'stage', 'moves', 'buy', 'selling',
                          'mailing', 'better', 'king', 'lamp', 'list', 'masters', 'keep', 'layer', 'parking', 'jobs',
                          'back', 'shares', 'sex', 'him', 'dates', 'runs', 'all', 'make', 'tips', 'pills', 'rating',
                          'one', 'beat', 'can', 'not']

    os.makedirs("raw", exist_ok=True)
    try:
        bench = LisanBench(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            words_file=args.words_file,
            use_batching=args.batching,
            stream_responses=args.streaming,
            force_openrouter=args.force_openrouter,
        )
    except Exception as e:
        print(f"ERROR initializing benchmark: {e}")
        sys.exit(1)

    print(f"\nStarting LisanBench")
    print(f"Models to test: {len(models_to_test)}")
    print(f"Starting words: {starting_words}")
    print(f"Temperature: {bench.temperature}")
    print(f"Max tokens: {bench.max_tokens}")
    print(f"Trials per word: {args.trials}")
    print(f"Max threads: {args.threads}")
    print(f"Batching mode: {args.batching}")
    print(f"Streaming: {args.streaming}")
    if args.force_openrouter:
        print(f"Force OpenRouter: True")
    if args.resume:
        print(f"Resuming from: {args.resume}")

    try:
        results = bench.run_benchmark(
            models=models_to_test,
            starting_words=starting_words,
            num_trials=args.trials,
            max_workers=args.threads,
            resume_from_file=args.resume,
            output_file=args.output,
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
