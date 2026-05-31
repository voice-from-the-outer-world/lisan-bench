# LisanBench

Website: [lisanbench.com](https://lisanbench.com)

> **“Lisan”** (as in *Lisan al-Gaib*) means “tongue” or “language” in Arabic. So, LisanBench literally translates to
*Language Bench*.

**LisanBench** is a lightweight, *cheap-to-run* benchmark for large language models that stresses **forward planning**,
**vocabulary depth**, **constraint adherence**, **attention**, and long-context **persistence** all at once.

Most traditional academic benchmarks like MMLU, MATH, AIME, or HumanEval are saturated and skewed toward niche domains.
They also
don’t correlate well with general intelligence or practical model utility.

Community-developed benchmarks
like [AidanBench](https://github.com/aidanmclaughlin/AidanBench), [SOLOBench](https://github.com/jd-3d/SOLOBench), [SimpleBench](https://github.com/simple-bench/SimpleBench),
or [Thematic Generalization](https://github.com/lechmazur/generalization) tend to capture those
qualities better. A core theme in those is **emergent complexity** from **simple problem scaling**.

LisanBench follows that principle. Inspired by these and co-developed with Claude-4 Opus, it introduces a novel twist on
Lewis Carroll’s 1877 game *Word Ladder*.

📖 **[Full announcement thread on X for more details](https://x.com/scaling01/status/1928510435164037342)**

---

## How It Works

In classic *Word Ladder*, you transform a start word into a target word, changing one letter at a time to create valid
intermediate words.

> “Each step consists of a single letter substitution, forming a new valid word.”  
> *[Wikipedia](https://en.wikipedia.org/wiki/Word_ladder)*

LisanBench uses an **open-ended variant** without a target word and Levenshtein-, instead of Hamming-distance:  
Each word in the chain must differ from the previous one by a **Levenshtein distance of 1** (i.e., one insert, delete,
or substitution), all intermediate words have to be in the dictionary and **no repetitions are allowed**.

**Example chain:**

```
cat -> bat -> bit -> sit -> wit -> win -> bin -> ...
```

By default, models are scored on **50 starting words** with **3 trials per word**. Earlier LisanBench runs used only 10
starting words, but the current 50-word, 3-trial setup gives a more stable estimate of model performance across word
difficulty and generation variance.

The benchmark also moved from the older `words_alpha.txt` dictionary to a pinned SCOWL/ESDB word list, due to common words missing in the older dictionary.

The main score is the **sum of valid transitions in the longest valid chain prefix** across all starting words. For
multi-trial runs, the benchmark reports both per-word best-trial sums and per-word average sums.

### Scoring Rules

The scorer extracts words from the model response and validates the longest prefix that satisfies all benchmark rules:

- The first extracted word must exactly match the requested starting word, case-insensitively.
- Every next word must be a valid dictionary word.
- Each transition must have Levenshtein/edit distance exactly 1.
- A word can appear only once in the chain.

If the response does not begin with the requested starting word, the trial receives `0` valid transitions for the main
score. The run still records secondary validity statistics for the extracted response so malformed outputs remain
auditable.

### Dictionary

The default validator uses a pinned SCOWL/ESDB word list:

```text
dictionaries/scowl/scowl_2026_02_25_huge_us_gb_ca_au_ascii.txt
```

If that file is missing, the runner regenerates it from the saved SCOWL source URL and verifies the exact SHA-256:

```text
7c0d7f4f19bacfa3ba95038a048e6b084c9d73e8ba5ebb82130c795bcea3f0f6
```

The generated dictionary lowercases alphabetic entries, sorts unique words, and keeps only `a` and `i` as one-letter
words. The other single-letter alphabet entries are excluded to avoid trivial one-letter reward hacks.

---

## Why Is This Challenging?

At its theoretical limit, LisanBench
approaches the **NP-hard "[Longest Simple Path Problem](https://en.wikipedia.org/wiki/Longest_path_problem)"**, giving
it practically indefinite scaling potential and skill ceiling.
The benchmark operates within the largest connected component of the
English language graph built from the pinned SCOWL/ESDB dictionary,
where nodes are words and edges connect words with Levenshtein distance = 1.

**Scaling advantage:**  
This benchmark rewards both **test-time compute** and **model size**, but models with explicit reasoning capabilities
have a distinct edge. Non-reasoning models must depend purely on implicit, latent reasoning and are therefore much more
likely to hit dead-ends. Larger models get boosts from broader vocabulary memorization, deeper
task comprehension, smarter heuristics, and better tokenization. But raw size isn’t enough: models also need precise
recall of previously used words and **output persistence** - the ability to keep generating long, uninterrupted sequences
without cutting off too soon.

To excel, top models need more than brute force: they have to leverage heuristics, local search, and backtracking
to escape dead-end traps and low-connectivity regions. This requires extensive multi-step planning to navigate the
narrow paths
through the word graph.

**What makes LisanBench unique is the simultaneous stress-testing of several core abilities:**

- **Strategic Planning:** Models must anticipate several moves ahead to avoid dead-ends and discover viable routes in a
  tight word space.
- **Vocabulary Depth:** Knowledge of both common and obscure words is vital for extending chains.
- **Recall:** The inability to repeat words across potentially hundreds of steps makes this a test of long-range memory.
- **Constraint Adherence & Task Understanding:** Models must strictly enforce the Levenshtein distance = 1 rule - no
  shortcuts or loose approximations.
- **Sustained Generation:** Cohesive, rule-abiding output must be maintained over long sequences without early stopping
  or breaking constraints.

![Leaderboard Results](assets/sum_chain_avg.png)
*Complete leaderboard results using the current average-score view*

## Repository Structure

| Path | Purpose |
|------|---------|
| `lisanbench/` | Runtime package for benchmark orchestration, CLI parsing, routing, scoring, result storage, reporting, visualization, reduced benchmark helpers, and shared utilities. |
| `lisanbench/providers/` | Provider integrations for OpenAI, Anthropic, Google, Moonshot, Z.AI, ZenMux, and OpenRouter. |
| `main.py`, `visualization.py`, `starting_word_picker.py`, `merge_results.py`, `cost_reconciliation.py` | Stable root compatibility wrappers for existing `uv run python ...` workflows. |
| `model_catalog.yaml` | Root model catalog with pricing, release dates, provider routing, service tiers, and direct provider mappings. |
| `reduced_benchmark_estimators.json` | Root reduced 15x2 estimator configuration. |
| `assets/` | Pre-rendered README figures. |
| `dictionaries/` | Pinned SCOWL dictionary files. |

## Quick Start

### 1. Install

```bash
git clone https://github.com/voice-from-the-outer-world/lisan-bench.git
cd lisan-bench
uv sync
```

Install `uv` first if it is not already available:
https://docs.astral.sh/uv/getting-started/installation/

### 2. Configure API keys

Create a local `.env` in the project root with the keys for the providers you plan to use. This file is ignored and
must not be committed:

```bash
OPENROUTER_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
MOONSHOT_API_KEY=...
ZAI_API_KEY=...
ZENMUX_API_KEY=...
```

You only need keys for the selected routes. In regular mode, the CLI resolves the selected catalog entries and checks
only the API keys for those backends. Catalog entries can route through OpenRouter or direct provider adapters such as
Google, Moonshot, Z.AI, and ZenMux, so `uv run python main.py --list-models` plus the catalog metadata determine which
keys are required. `--batching` uses the configured batch providers for the selected models, so set only the needed
batch keys among `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GOOGLE_API_KEY`. `--force-openrouter` requires only
`OPENROUTER_API_KEY`.

Optional provider settings:

- `ZAI_API_BASE=https://api.z.ai/api/paas/v4` overrides the default Z.AI base URL.
- `TEMPERATURE` and `MAX_TOKENS` can be set in `.env`; CLI flags override them.
- For OpenRouter evaluation of OpenAI models that need your OpenAI account, add your OpenAI key to OpenRouter:
  https://openrouter.ai/settings/integrations

### 3. Inspect and run

```bash
# See model IDs available in model_catalog.yaml
uv run python main.py --list-models

# Run the default benchmark: all listed models, 50 words, 3 trials per word
uv run python main.py

# Run the reduced preset: 15 held-out predictive words, 2 trials per word
uv run python main.py --reduced-15x2
```

Results are written as `lisan_bench_results_YYYYMMDD_HHMMSS.json` unless `--output` is provided. `--output`
must name a new file; it does not resume or overwrite existing results. Use `--resume` to continue an existing
results file.

Model metadata and routing live in `model_catalog.yaml`. These examples are copied from the current catalog:

```yaml
models:
  - id: "openai/gpt-5.4:thinking-medium"
    company: "openai"
    completion_provider: "openrouter"
    batch_completion_provider: "openai"
    completion_model_id: "gpt-5.4"
    pricing:
      in: 2.5
      out: 15.0
    release:
      date: "2026-03-05"
      source: "https://openrouter.ai/openai/gpt-5.4"
  - id: "anthropic/claude-sonnet-4.5"
    company: "anthropic"
    completion_provider: "openrouter"
    batch_completion_provider: "anthropic"
    completion_model_id: "claude-sonnet-4-5-20250929"
    pricing:
      in: 3.0
      out: 15.0
    release:
      date: "2025-09-29"
      source: "https://www.anthropic.com/news/claude-sonnet-4-5"
```

For entries with `completion_provider: "openrouter"`, `completion_model_id` is used by the direct batch provider mapping
when `batch_completion_provider` is set; it is not the OpenRouter route id. `--batching` uses provider batch APIs instead
of Flex. Some OpenRouter routes do not support Flex; completed OpenRouter runs reconcile against the cost returned by
OpenRouter, so recorded costs can differ from the requested tier.

## Reduced Benchmark

The full default run uses 50 starting words with 3 trials each, or 150 model calls per model. The documented reduced
preset uses 15 held-out predictive words with 2 trials each, or 30 model calls per model. In practice this is about
4-5x cheaper while still tracking the full benchmark well enough for scaling curves, ablations, and broad model
ordering. It is not meant to adjudicate very close leaderboard ties.

Run it with:

```bash
uv run python main.py --reduced-15x2
```

The reduced words were chosen from existing full-result data by optimizing a 15-word subset for leave-one-out predictive
quality, then selecting the global linear estimator that gave the best 30-call practical ordering and absolute-error
tradeoff in Monte Carlo simulations. The reduced word identities are intentionally not published here.

Current estimator quality:

| Check | Value |
|-------|------:|
| Calibration models | 137 |
| Clean leave-one-out R^2 | 0.99743 |
| Clean leave-one-out MAE | 72.90 |
| Clean leave-one-out RMSE | 117.27 |
| Monte Carlo overall Spearman | 0.9775 |
| Monte Carlo thinking-model Spearman | 0.9826 |
| Monte Carlo thinking-model Kendall tau | 0.9002 |
| Monte Carlo overall MAE | 164.5 |
| Monte Carlo thinking-model MAE | 302.1 |

For the GPT-5.4-mini reasoning-effort sweep, the reduced 30-call estimate was compared against full 150-call scores
across six settings. The mean absolute error was 130.7 score points, with a median absolute error of 117.3 and maximum
absolute error of 349.4.

![GPT-5.4-mini Reduced Benchmark Dashboard](assets/gpt54_mini_reasoning_sweep_dashboard.png)
*Reduced 15x2 estimates versus full 50x3 scores for the GPT-5.4-mini reasoning sweep*

## Usage Examples

```bash
# Full benchmark with the pinned SCOWL dictionary
uv run python main.py

# Run specific catalog model IDs
uv run python main.py --models "openai/gpt-5.4:thinking-medium" "anthropic/claude-opus-4.5"

# Control generation and concurrency
uv run python main.py --models "openai/gpt-5.4:thinking-medium" --temperature 0.8 --max-tokens 12000 --threads 10

# More trials per starting word
uv run python main.py --models "openai/gpt-5.4:thinking-medium" --trials 5

# Small smoke test with your own custom starting words
uv run python main.py --models "openai/gpt-5.4-mini:thinking-none" --words WORD1 WORD2 WORD3 --trials 1

# Resume interrupted benchmark in-place
uv run python main.py --resume lisan_bench_results_20260531_120000.json

# Save to a known new filename
uv run python main.py --models "openai/gpt-5.4-mini:thinking-none" --output smoke_results.json

# Use provider batch APIs where supported
uv run python main.py --models "openai/gpt-5.4-mini:thinking-none" "anthropic/claude-sonnet-4.5" --batching

# Force every selected model through OpenRouter, ignoring direct provider routes
uv run python main.py --models "openai/gpt-5.4-mini:thinking-none" --force-openrouter

# Disable incremental OpenRouter streaming
uv run python main.py --models "openai/gpt-5.4-mini:thinking-none" --no-streaming
```

Use `uv run python main.py --help` for the full option list.

The benchmark validates model IDs against `model_catalog.yaml`. If a model name fails, run
`uv run python main.py --list-models` and copy the exact catalog ID.

## Generating Custom Starting Words

`starting_word_picker.py` selects common English words from the largest connected component of the edit-distance graph,
then spreads them across word length, local connectivity, and edit-distance diversity. Use it to design custom word sets
before passing them to `main.py --words`.

```bash
# Generate 20 diverse starting words
uv run python starting_word_picker.py --num-words 20

# Shorter words only
uv run python starting_word_picker.py --num-words 15 --min-length 4 --max-length 8

# More difficulty buckets and stronger separation between chosen words
uv run python starting_word_picker.py -n 50 --difficulty-levels 10 --min-edit-distance 4

# Consider a larger slice of the common-word list
uv run python starting_word_picker.py -n 50 --max-common-words 10000
```

After choosing words, pass them directly to the benchmark:

```bash
uv run python main.py --models "openai/gpt-5.4-mini:thinking-none" --words WORD1 WORD2 WORD3 --trials 3
```

Use `uv run python starting_word_picker.py --help` for all picker options.

## Visualizations

`visualization.py` reads a result JSON file, recomputes strict valid-chain prefixes with the current validator, prints
rankings, and generates plots. If no file is supplied, it uses `latest_results.json`.

```bash
# Show interactive plots for a results file
uv run python visualization.py lisan_bench_results_20260531_120000.json

# Save regular plots to the default output directory
uv run python visualization.py lisan_bench_results_20260531_120000.json --save

# Save plots to a custom directory
uv run python visualization.py lisan_bench_results_20260531_120000.json --save --output-dir OUTPUT_DIR

# Limit leaderboard-style plots to the top 15 models
uv run python visualization.py latest_results.json --save --top_k 15

# Plot ranks 10 through 25 instead of the top-k window
uv run python visualization.py latest_results.json --save --rank-range 10,25

# Choose models for the difficulty trajectory plot
uv run python visualization.py latest_results.json --save --trajectory-models gpt-5.4 claude-opus-4.5

# Generate only Levenshtein connectivity graphs for the starting words in a result file
uv run python visualization.py latest_results.json --word-graphs-only --word-graphs-dir WORD_GRAPH_DIR

# Include all edit-1 links among discovered graph nodes; this is much denser
uv run python visualization.py latest_results.json --word-graphs-only --full-graph
```

Saved regular plots include:

- `sum_chain_avg.png`: main multi-trial leaderboard, averaging each word across trials before summing.
- `sum_difficulty_chain_avg.png`: difficulty-weighted score.
- `sum_chain_max.png`: best-of-word score.
- `avg_validity_per_word.png`: average valid-transition ratio.
- `scatter_chain_vs_validity.png`: score vs. adherence.
- `boxplot_by_word_colored.png`: per-word distribution colored by starting word.
- `results_over_time.png`: score frontier by model release date from `model_catalog.yaml`.
- `reasoning_efficiency.png`: reasoning-token efficiency when token metadata is available.
- `gpt54_mini_reasoning_sweep_dashboard.png`: reduced 15x2 versus full 50x3 comparison for GPT-5.4-mini.

Use `uv run python visualization.py --help` for the full visualization option list.

## Benchmark Results

![Average Validity](assets/avg_validity_per_word.png)
*Average link validity across models (OpenAI models show exceptional precision)*

![Difficulty Distribution](assets/boxplot_by_word_colored.png)
*Performance distribution per starting word, showing difficulty variance and outliers*

![Path Length vs Accuracy](assets/scatter_chain_vs_validity.png)
*Trade-off between valid-transition score and constraint adherence*

![Model Scores Over Time](assets/results_over_time.png)
*Score frontier by model release date*

![Reasoning Efficiency](assets/reasoning_efficiency.png)
*Relationship between reasoning-token usage and valid-chain length*

### Word Difficulty Visualization

The visualization script can generate local connectivity graphs for the starting words in a result file. Words sit on
concentric circles by Levenshtein distance, and same-distance links can be omitted for readability. If you run
`visualization.py --full-graph`, you will get every connection, but the resulting graph is usually too dense to inspect.

The graphs below are retained as legacy illustrations from the old 10-word LisanBench subset. They are not the current
50-word benchmark set and should not be interpreted as current leaderboard coverage.

<details>
<summary><b>Click to expand legacy 10-word connectivity graphs</b></summary>

| Legacy Word | Local Connectivity Graph |
|-------------|--------------------------|
| **abysmal** | ![abysmal graph](assets/levenshtein_graph_abysmal.png) |
| **avoid** | ![avoid graph](assets/levenshtein_graph_avoid.png) |
| **camping** | ![camping graph](assets/levenshtein_graph_camping.png) |
| **hat** | ![hat graph](assets/levenshtein_graph_hat.png) |
| **layer** | ![layer graph](assets/levenshtein_graph_layer.png) |
| **lung** | ![lung graph](assets/levenshtein_graph_lung.png) |
| **mine** | ![mine graph](assets/levenshtein_graph_mine.png) |
| **origin** | ![origin graph](assets/levenshtein_graph_origin.png) |
| **pattern** | ![pattern graph](assets/levenshtein_graph_pattern.png) |
| **traveller** | ![traveller graph](assets/levenshtein_graph_traveller.png) |

</details>

## Details

**Verification:** The benchmark validates output with the pinned SCOWL/ESDB dictionary described above. The prompt text
still mentions `words_alpha.txt` for historical continuity, but the default runner, scorer, visualization metrics, and
website metrics use the pinned SCOWL file unless `--words-file` is supplied.

The SCOWL migration replaced the older default `words_alpha.txt` dictionary because an unpinned live download makes
scores harder to reproduce and includes many entries that are undesirable for this task. The pinned SCOWL/ESDB file is
derived from the saved SCOWL source URL in `lisanbench/utils.py`, normalized to lowercase alphabetic tokens, sorted, deduplicated,
written with stable line endings, and checked against the SHA-256 listed above. This means a missing local dictionary can
be regenerated byte-for-byte, while a modified dictionary fails fast.

This migration can change historical scores. Some chains that were valid under `words_alpha.txt` are invalid under the
pinned SCOWL list, and some previously invalid chains may become valid. Current leaderboard and visualization code uses
the strict current scorer so old result files can be re-evaluated consistently from their saved `word_chain` or
`raw_response`.

**Word Selection:** The starting word picker uses
the [Google 10k English words list](https://github.com/first20hours/google-10000-english) to ensure commonly-used words
are prioritized when generating diverse starting word sets.

**Inspiration:** LisanBench draws from AidanBench and SOLO-Bench but offers key advantages:

- **Scalable evaluation**: The benchmark can be expanded by adding more starting words, trials, or harder word sets.
- **Trivially verifiable**: No embedding models required. No ambiguity in evaluations.
- **Extreme difficulty and resolution scaling**: No skill ceiling and trivially extensible by adding more words and
  trials per word.
- **Knowledge-focused**: Unlike SOLO-Bench it explicitly tests vocabulary depth and has much clearer constraints.

**Important limitation:** Unlike AidanBench which operates on paragraph level, LisanBench works at the character level
and is therefore **affected by tokenization**. Models with better tokenizers should perform better,
all else being equal.

**Collaborative Creation:** This benchmark emerged from a human-AI collaboration. After multiple failed attempts at
creating divergent thinking tests, I prompted Claude 4 Opus with the new benchmark objectives and a recap of what hadn’t
worked. It responded with several "Top 10 ideas" lists, one of which included the core idea of a "Chain Link Bench,"
essentially a variant of the "Word Ladder" game. I handled problem definition, provided context, selected and refined
the concept (shifting from basic one-letter edits to Levenshtein distance = 1), and implemented it.

## Usage Terms

When publishing new results or building on top of LisanBench you are required to:

1. **Credit the creator:** Mention [@scaling01 (Lisan al Gaib)](https://x.com/scaling01/) on X (formerly Twitter)
2. **Reference the source:** Link to this repository or
   the [original announcement post](https://x.com/scaling01/status/1928510435164037342)

**Note:** These are usage terms and do not constitute a standard open source license. For additional permissions,
contact the creator.
