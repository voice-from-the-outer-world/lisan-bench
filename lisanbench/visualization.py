import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import math

from lisanbench.model_catalog import model_release_dates
from lisanbench.model_names import (
    implicit_reasoning_label,
    split_reasoning_suffix,
    strip_provider_prefix,
    thinking_suffix_parameter,
)
from lisanbench.utils import (
    SparsityWeighter,
    build_tree,
    coerce_word_chain,
    generate_edit1,
    get_sparse_move_weights,
    load_word_dictionary,
    score_sparse_valid_prefix,
    validate_chain,
)

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import tiktoken
from adjustText import adjust_text

_TIKTOKEN_ENC = None


class _ApproxTokenEncoding:
    def encode(self, text: str, disallowed_special=()) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", str(text))


def _get_tiktoken_enc():
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        try:
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_ENC = _ApproxTokenEncoding()
    return _TIKTOKEN_ENC

# --------------------------
# Style
# --------------------------
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.figsize': (20, 15),
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'legend.title_fontsize': 20
})

COLOR_MAP = {
    'openai': '#74AA9C',     # teal
    'anthropic': '#C97845',  # warm orange
    'google': '#AA4256',     # muted red
    'x-ai': '#7A7A9A',       # medium gray-purple
    'deepseek': '#355167',   # deep teal
    'other': '#8D8389',      # warm gray
}

# ==========================
# Data prep + existing plots
# ==========================


def extract_company(model_name: str) -> str:
    if model_name.startswith('zenmux/deepseek'):
        return 'deepseek'
    if model_name.startswith('zenmux/doubao'):
        return 'other'
    mappings = {
        'amazon/': 'other', 'anthropic/': 'anthropic', 'bytedance/': 'other',
        'deepseek/': 'deepseek', 'google/': 'google', 'meta-llama': 'other',
        'meta-llama/': 'other', 'mistralai/': 'other', 'moonshotai/': 'other',
        'openai/': 'openai', 'qwen/': 'other', 'x-ai/': 'x-ai'
    }
    for prefix, comp in mappings.items():
        if model_name.startswith(prefix):
            return comp
    return 'other'

def clean_name(model_name: str) -> str:
    return strip_provider_prefix(model_name)

def _estimate_plot_reasoning_tokens(output_tokens: object, raw_response: object) -> int:
    """Estimate hidden reasoning tokens from billed output minus visible response tokens."""
    try:
        output_total = int(float(output_tokens))
    except (TypeError, ValueError):
        return 0

    if output_total <= 0:
        return 0

    if raw_response is None or pd.isna(raw_response):
        response_text = ""
    else:
        response_text = str(raw_response)

    visible_tokens = len(_get_tiktoken_enc().encode(response_text, disallowed_special=()))
    return max(0, output_total - visible_tokens)

def prepare_data(results_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    file_path = Path(results_path)
    data = json.loads(file_path.read_text())
    results = data['results']
    valid_words = load_word_dictionary()
    sparsity_weighter = SparsityWeighter(valid_words)

    agg_records, run_records, efficiency_records = [], [], []
    multi_trials_exist = False
    for model_full, runs in results.items():
        company = extract_company(model_full)
        model = clean_name(model_full)
        df = pd.DataFrame(runs)
        if df.empty:
            continue

        # ensure required columns exist
        if 'starting_word' not in df:
            continue

        chains = df['word_chain'] if 'word_chain' in df else pd.Series([None] * len(df), index=df.index)
        raw_responses = df['raw_response'] if 'raw_response' in df else pd.Series([""] * len(df), index=df.index)
        strict_metrics = [
            validate_chain(coerce_word_chain(chain, raw_response), valid_words, starting_word)
            for starting_word, chain, raw_response in zip(df['starting_word'], chains, raw_responses)
        ]
        df['chain_use'] = [metric[0] for metric in strict_metrics]
        df['strict_validity_ratio'] = [
            (metric[1] / (metric[1] + metric[2])) if (metric[1] + metric[2]) else np.nan
            for metric in strict_metrics
        ]
        df['sparse_chain_score'] = [
            score_sparse_valid_prefix(starting_word, chain, raw_response, sparsity_weighter)
            for starting_word, chain, raw_response in zip(df['starting_word'], chains, raw_responses)
        ]
        df['sparse_move_weights'] = [
            get_sparse_move_weights(starting_word, chain, raw_response, sparsity_weighter)
            for starting_word, chain, raw_response in zip(df['starting_word'], chains, raw_responses)
        ]
        if 'output_tokens' in df:
            df['plot_output_tokens'] = pd.to_numeric(df['output_tokens'], errors='coerce').fillna(0)
            df['plot_reasoning_tokens'] = [
                _estimate_plot_reasoning_tokens(output_tokens, raw_response)
                for output_tokens, raw_response in zip(df['plot_output_tokens'], raw_responses)
            ]
        words = df['starting_word'].unique().tolist()

        sum_avg = 0.0
        sum_max = 0.0
        sum_sparse = 0.0
        avg_validity_per_word = []

        for word in words:
            dsub = df[df['starting_word'] == word]
            if len(dsub) > 1:
                multi_trials_exist = True
            # Sum of averages per word (existing)
            sum_avg += float(dsub['chain_use'].mean())
            # Sum of per-word maxima across trials (new)
            try:
                sum_max += float(dsub['chain_use'].max())
            except Exception:
                pass
            try:
                sum_sparse += float(dsub['sparse_chain_score'].mean())
            except Exception:
                pass
            avg_validity_per_word.append(float(dsub['strict_validity_ratio'].mean()))

            longest_chain = float(dsub['chain_use'].mean())
            if 'plot_output_tokens' in dsub:
                output_tokens = float(dsub['plot_output_tokens'].mean())
                reasoning_tokens = float(dsub['plot_reasoning_tokens'].mean())
                efficiency_records.append({
                    'model': model,
                    'company': company,
                    'starting_word': word,
                    'longest_chain': longest_chain,
                    'output_tokens': output_tokens,
                    'reasoning_tokens': reasoning_tokens,
                })

        agg_records.append({
            'model_full': model_full,
            'model': model,
            'company': company,
            'sum_chain_avg': sum_avg,
            'sum_chain_max': sum_max,
            'sum_sparse_chain_avg': sum_sparse,
            'avg_hardness_per_move': (sum_sparse / sum_avg) if sum_avg > 0 else np.nan,
            'avg_validity_per_word': (pd.Series(avg_validity_per_word).mean()
                                      if avg_validity_per_word else np.nan),
            'overall_avg_validity': float(df['strict_validity_ratio'].mean())
        })

        for _, r in df.iterrows():
            run_records.append({
                'starting_word': r['starting_word'],
                'model': model,
                'company': company,
                'chain': r.get('chain_use', np.nan),
                'sparse_chain': r.get('sparse_chain_score', np.nan),
                'sparse_move_weights': r.get('sparse_move_weights', []),
            })

    df_agg = pd.DataFrame(agg_records)
    df_runs = pd.DataFrame(run_records).dropna(subset=['starting_word', 'model', 'chain'])
    df_eff = pd.DataFrame(efficiency_records)
    return df_agg, df_runs, df_eff, multi_trials_exist


def plot_results_over_time(
        df_agg: pd.DataFrame,
        save_dir: str = None,
        score_col: str = "sum_chain_avg",
):
    """
    Plot benchmark score over model release date and highlight frontier models.
    Release dates come from `model_release_dates` loaded from model_catalog.yaml.
    """
    required_cols = {"model_full", "company", score_col}
    if df_agg.empty or (required_cols - set(df_agg.columns)):
        return

    records = []
    for row in df_agg.itertuples(index=False):
        model_full = getattr(row, "model_full", None)
        if not isinstance(model_full, str):
            continue
        meta = model_release_dates.get(model_full, {})
        date_str = meta.get("date")
        if not isinstance(date_str, str) or not date_str.strip():
            continue
        release_date = pd.to_datetime(date_str.strip(), errors="coerce")
        if pd.isna(release_date):
            continue

        score = getattr(row, score_col)
        try:
            score_val = float(score)
        except (TypeError, ValueError):
            continue

        records.append({
            "model_full": model_full,
            "model_label": _normalize_model_label(clean_name(model_full)),
            "company": getattr(row, "company", "other"),
            "release_date": release_date,
            "score": score_val,
        })

    if not records:
        return

    df_time = pd.DataFrame(records).sort_values("release_date").reset_index(drop=True)
    if df_time.empty:
        return

    # Frontier = first model at each strictly higher score in chronological order.
    frontier_flags: List[bool] = []
    best_so_far = -float("inf")
    for score in df_time["score"].tolist():
        is_frontier = score > best_so_far
        frontier_flags.append(is_frontier)
        if is_frontier:
            best_so_far = score
    df_time["is_frontier"] = frontier_flags

    fig, ax = plt.subplots(figsize=(20, 15))

    non_frontier = df_time[~df_time["is_frontier"]]
    if not non_frontier.empty:
        ax.scatter(
            non_frontier["release_date"],
            non_frontier["score"],
            c="#C7C7C7",
            alpha=0.45,
            s=75,
            edgecolors="none",
            label="Non-frontier models",
            zorder=1,
        )

    frontier = df_time[df_time["is_frontier"]]
    for company, grp in frontier.groupby("company"):
        ax.scatter(
            grp["release_date"],
            grp["score"],
            color=COLOR_MAP.get(company, COLOR_MAP["other"]),
            alpha=0.95,
            s=190,
            edgecolor="black",
            linewidth=1.2,
            label=company,
            zorder=3,
        )
        for _, item in grp.iterrows():
            ax.annotate(
                item["model_label"],
                (item["release_date"], item["score"]),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=12,
                zorder=4,
            )

    frontier_sorted = frontier.sort_values("release_date")
    if len(frontier_sorted) >= 2:
        ax.step(
            frontier_sorted["release_date"],
            frontier_sorted["score"],
            where="post",
            color="#6E6E6E",
            linewidth=2.0,
            alpha=0.7,
            linestyle="--",
            zorder=2,
        )

    y_label_by_score_col = {
        "sum_chain_avg": "Score (sum of per-word averages)",
        "sum_chain_max": "Score (sum of best trial per word)",
    }

    ax.set_title("LisanBench: Score Over Time (Release-Date Frontier)", pad=20)
    ax.set_xlabel("Release Date")
    ax.set_ylabel(y_label_by_score_col.get(score_col, f"Score ({score_col})"))
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Company", loc="best")
    plt.tight_layout()

    if save_dir:
        plt.savefig(Path(save_dir) / "results_over_time.png", bbox_inches="tight", dpi=300)
    else:
        plt.show()
    plt.close(fig)

def _apply_rank_window(df: pd.DataFrame,
                       sort_col: str,
                       descending: bool = True,
                       top_k: int = 25,
                       rank_range: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
    """Return a copy of *df* limited to the requested rank window."""
    if df.empty:
        return df.copy()

    df_sorted = df.sort_values(sort_col, ascending=not descending)

    if not rank_range:
        return df_sorted.head(top_k).copy()

    start, end = rank_range
    start = max(start, 1)
    end = max(end, start)

    start_idx = start - 1
    df_window = df_sorted.iloc[start_idx:end].copy()
    return df_window


def plot_barh(df: pd.DataFrame, y_col: str, title: str, xlabel: str, fname=None, fmt=".2f", save_dir=None, top_k=25, rank_range: Optional[Tuple[int, int]] = None):
    if df.empty or y_col not in df:
        return
    dfp = _apply_rank_window(df, y_col, descending=True, top_k=top_k, rank_range=rank_range)
    if dfp.empty:
        return
    # Apply human-friendly labels for y-axis
    dfp = dfp.copy()
    dfp["model"] = dfp["model"].apply(_normalize_model_label)
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.barplot(data=dfp, x=y_col, y="model", hue="company", palette=COLOR_MAP, dodge=False, ax=ax)
    ax.set_title(title, pad=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    for i, v in enumerate(dfp[y_col]):
        ax.text(v + (max(dfp[y_col]) * 0.005), i, f"{v:{fmt}}", va='center', fontsize=16)
    ax.legend(title="Company", loc='lower right')
    plt.tight_layout()
    if save_dir and fname:
        plt.savefig(Path(save_dir) / fname, bbox_inches="tight", dpi=300)
    else:
        plt.show()
    plt.close(fig)

def _normalize_model_label(name: str) -> str:
    alias_map = {
        "chatgpt-4o-latest": "gpt-4o",
        "claude-3.5-sonnet": "Sonnet 3.6",
        "claude-3.5-sonnet-20240620": "Sonnet 3.5",
        "deepseek-chat": "deepseek-v3",
    }

    def _format_token(token: str) -> str:
        low = token.lower()
        if low == "gpt":
            return "GPT"
        if re.fullmatch(r"\d+(\.\d+)?", low):
            return low
        if re.fullmatch(r"\d+(\.\d+)?b", low):
            return low[:-1] + "B"
        if re.fullmatch(r"\d+[a-z]", low):
            return low
        if re.fullmatch(r"[a-z]\d+(\.\d+)?", low):
            return low.upper()
        if re.fullmatch(r"a\d+b", low):
            return low.upper()
        if re.fullmatch(r"[a-z]{2,4}", low) and not re.search(r"[aeiou]", low):
            return low.upper()
        return low[0].upper() + low[1:]

    def _format_claude_base(base_name: str) -> str:
        raw_tokens = [tok for tok in re.split(r"[-_]", base_name) if tok]
        if not raw_tokens or raw_tokens[0].lower() != "claude":
            return ""

        tokens = raw_tokens[1:]
        if tokens and tokens[-1].isdigit() and len(tokens[-1]) >= 6:
            tokens = tokens[:-1]

        family_idx = next((i for i, tok in enumerate(tokens) if tok.lower() in {"haiku", "sonnet", "opus"}), None)
        if family_idx is None:
            return ""

        family = tokens[family_idx].capitalize()
        version_tokens = [tok for i, tok in enumerate(tokens) if i != family_idx]
        if not version_tokens:
            return family

        if all(re.fullmatch(r"\d+", tok) for tok in version_tokens):
            version = ".".join(version_tokens)
        else:
            version = " ".join(version_tokens)

        return f"{family} {version}".strip()

    def _drop_release_tag(tokens: List[str]) -> List[str]:
        if len(tokens) < 2 or not tokens[-1].isdigit():
            return tokens
        if len(tokens[-1]) >= 6 or tokens[-2].lower() in {"turbo", "flash", "sonnet", "haiku", "opus"}:
            return tokens[:-1]
        return tokens

    def _format_base(base_name: str) -> str:
        claude_label = _format_claude_base(base_name)
        if claude_label:
            return claude_label

        raw_tokens = [tok for tok in re.split(r"[-_]", base_name) if tok]
        raw_tokens = [
            tok for tok in _drop_release_tag(raw_tokens)
            if tok.lower() not in {"it", "instruct", "latest"} and not re.fullmatch(r"fp\d+", tok.lower())
        ]
        if not raw_tokens:
            return ""

        if raw_tokens[0].lower() == "gpt":
            label = "GPT"
            rest = raw_tokens[1:]
            if rest and rest[0].lower() == "oss":
                oss_tokens = [_format_token(tok).upper() for tok in rest[1:]]
                return "-".join(["GPT", "OSS", *oss_tokens]).strip("-")
            if rest and re.fullmatch(r"\d+(\.\d+)?[a-z]?", rest[0].lower()):
                label = f"GPT {rest[0].lower()}"
                rest = rest[1:]
            formatted_rest = [_format_token(tok) for tok in rest]
            return " ".join([label, *formatted_rest]).strip()

        return " ".join(_format_token(tok) for tok in raw_tokens).strip()

    canonical_name = alias_map.get(name, name)
    base, suffix = split_reasoning_suffix(canonical_name)
    label = _format_base(base)

    suffix_l = suffix.lower()
    if suffix_l == "thinking":
        label = f"{label} (thinking)".strip()
    elif suffix_l.startswith("thinking-"):
        thinking_param = thinking_suffix_parameter(suffix_l)
        if thinking_param == "none":
            pass
        elif thinking_param in {"low", "medium", "high", "xhigh"}:
            label = f"{label} ({thinking_param})".strip()
        elif re.fullmatch(r"\d+k", thinking_param):
            label = f"{label} ({thinking_param})".strip()
        else:
            label = f"{label} ({thinking_param})".strip()
    elif suffix_l == "free":
        pass
    elif suffix_l:
        label = f"{label} {_format_token(suffix_l)}".strip()

    implicit_label = implicit_reasoning_label(base, suffix)
    if implicit_label:
        label = f"{label} ({implicit_label})".strip()

    return label or name


def _normalize_reasoning_efficiency_label(name: str) -> str:
    label = _normalize_model_label(name)
    # For the reasoning efficiency chart, remove explicit thinking suffixes.
    label = re.sub(r"\s+\((thinking)\)$", "", label, flags=re.IGNORECASE).strip()
    return label

def plot_scatter(df_agg: pd.DataFrame, save_dir: str = None, top_k=25, rank_range: Optional[Tuple[int, int]] = None):
    if df_agg.empty:
        return
    dfp = _apply_rank_window(df_agg, "sum_chain_avg", descending=True, top_k=top_k, rank_range=rank_range)
    if dfp.empty:
        return
    dfp["model"] = dfp["model"].apply(_normalize_model_label)
    fig, ax = plt.subplots(figsize=(20, 15))
    texts = []
    for comp, grp in dfp.groupby('company'):
        ax.scatter(grp['sum_chain_avg'], grp['avg_validity_per_word'],
                   label=comp, s=200, color=COLOR_MAP.get(comp, COLOR_MAP['other']), alpha=0.85,
                   edgecolor='black', linewidth=1.2)
        for _, row in grp.iterrows():
            texts.append(ax.text(row['sum_chain_avg'], row['avg_validity_per_word'], row['model'],
                    fontsize=16))
    adjust_text(texts, ax=ax, max_move=40,
                arrowprops=dict(arrowstyle='-', color='grey', lw=0.5))
    ax.set_title("Score vs Validity", pad=20)
    ax.set_xlabel("Score")
    ax.set_ylabel("Average Validity Ratio")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title="Company", loc='lower right')
    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / "scatter_chain_vs_validity.png", bbox_inches="tight", dpi=300)
    else:
        plt.show()
    plt.close(fig)


def plot_sparse_trajectory(df_runs: pd.DataFrame,
                           models: Optional[List[str]] = None,
                           save_dir: str = None,
                           fname: str = "difficulty_trajectory.png",
                           title: str = "Difficulty Weight per Move (Trajectory)"):
    """
    Plot the average per-move difficulty weight as a trajectory for each requested model.

    Parameters
    ----------
    models : list of model name strings (as they appear in results, without provider prefix).
             If None or empty, the plot is skipped.
    """
    required_cols = {"model", "company", "sparse_move_weights"}
    if df_runs.empty or (required_cols - set(df_runs.columns)):
        return
    if not models:
        return

    requested = {m.strip().lower() for m in models}
    df = df_runs[df_runs['model'].str.lower().isin(requested)].copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(20, 15))

    for model_name, grp in df.groupby('model'):
        company = grp['company'].iloc[0]
        color = COLOR_MAP.get(company, COLOR_MAP['other'])
        sequences = [w for w in grp['sparse_move_weights'] if isinstance(w, list) and len(w) > 0]
        if not sequences:
            continue

        max_len = max(len(s) for s in sequences)
        avg_weights = [np.mean([s[i] for s in sequences if i < len(s)]) for i in range(max_len)]
        # Shade ±1 std where enough samples exist
        std_weights = [np.std([s[i] for s in sequences if i < len(s)]) for i in range(max_len)]
        counts = [sum(1 for s in sequences if i < len(s)) for i in range(max_len)]

        move_indices = list(range(1, max_len + 1))
        label = _normalize_model_label(model_name)
        ax.plot(move_indices, avg_weights, label=label, color=color, linewidth=2.0, alpha=0.85,
                marker='o', markersize=4)
        if len(sequences) > 1:
            lower = [a - s for a, s in zip(avg_weights, std_weights)]
            upper = [a + s for a, s in zip(avg_weights, std_weights)]
            ax.fill_between(move_indices, lower, upper, color=color, alpha=0.12)

        # Fade out where < 20 % of runs still have data
        threshold = max(counts) * 0.2
        fade_start = next((i for i, c in enumerate(counts) if c < threshold), None)
        if fade_start is not None and fade_start < max_len:
            ax.axvline(move_indices[fade_start], color=color, linewidth=1.0,
                       linestyle=':', alpha=0.5)

    ax.set_title(title, pad=20)
    ax.set_xlabel("Move Index")
    ax.set_ylabel("Avg Difficulty Weight (sparsity × rarity)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title="Model", loc='upper right', fontsize=14)
    plt.tight_layout()

    if save_dir:
        plt.savefig(Path(save_dir) / fname, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close(fig)


def plot_score_vs_sparse_score(df_agg: pd.DataFrame,
                               save_dir: str = None,
                               top_k: int = 25,
                               rank_range: Optional[Tuple[int, int]] = None,
                               fname: str = "score_vs_difficulty_score.png",
                               title: str = "Path Length vs Difficulty-Weighted Score"):
    required_cols = {"model", "company", "sum_chain_avg", "sum_sparse_chain_avg"}
    if df_agg.empty or (required_cols - set(df_agg.columns)):
        return

    dfp = _apply_rank_window(df_agg, "sum_chain_avg", descending=True, top_k=top_k, rank_range=rank_range)
    dfp = dfp.dropna(subset=["sum_chain_avg", "sum_sparse_chain_avg"]).copy()
    if len(dfp) < 2:
        return

    pearson_r = float(dfp["sum_chain_avg"].corr(dfp["sum_sparse_chain_avg"]))
    spearman_r = float(dfp["sum_chain_avg"].rank().corr(dfp["sum_sparse_chain_avg"].rank()))
    dfp["model_plot"] = dfp["model"].apply(_normalize_model_label)

    fig, ax = plt.subplots(figsize=(20, 15))
    sns.regplot(
        data=dfp,
        x="sum_chain_avg",
        y="sum_sparse_chain_avg",
        scatter=False,
        ci=None,
        line_kws={"color": "#5A5A5A", "linestyle": "--", "linewidth": 2.5},
        ax=ax,
    )

    texts = []
    for comp, grp in dfp.groupby("company"):
        ax.scatter(
            grp["sum_chain_avg"],
            grp["sum_sparse_chain_avg"],
            label=comp,
            s=200,
            color=COLOR_MAP.get(comp, COLOR_MAP["other"]),
            alpha=0.85,
            edgecolor="black",
            linewidth=1.2,
        )
        for _, row in grp.iterrows():
            texts.append(ax.text(row["sum_chain_avg"], row["sum_sparse_chain_avg"], row["model_plot"], fontsize=16))

    adjust_text(texts, ax=ax, max_move=40, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

    ax.set_title(title, pad=20)
    ax.set_xlabel("Path Length Score (valid transitions; sum of per-word averages)")
    ax.set_ylabel("Difficulty-Weighted Score")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Company", loc="best")
    ax.text(
        0.02, 0.98,
        f"Pearson r = {pearson_r:.3f}\nSpearman r = {spearman_r:.3f}",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=16,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#B0B0B0"),
    )
    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / fname, bbox_inches="tight", dpi=300)
    else:
        plt.show()
    plt.close(fig)


def plot_distribution_by_word(df_runs: pd.DataFrame, save_dir: str = None):
    required_cols = {"starting_word", "model", "company", "chain"}
    if df_runs.empty or (required_cols - set(df_runs.columns)):
        return

    df = df_runs.copy()
    df["model"] = df["model"].apply(_normalize_model_label)

    perf = (df.groupby(['starting_word', 'model', 'company'], as_index=False)['chain']
            .mean()
            .rename(columns={'chain': 'mean_chain'}))
    if perf.empty:
        return

    top_idx = perf.groupby('starting_word')['mean_chain'].idxmax()
    tops = perf.loc[top_idx].set_index('starting_word')

    medians = df.groupby('starting_word')['chain'].median().sort_values(ascending=True)
    ord_words = [w for w in medians.index.tolist() if w in tops.index]
    if not ord_words:
        return

    def company_color(comp: str) -> str:
        return COLOR_MAP.get(comp, COLOR_MAP['other'])

    word_colors = [company_color(tops.loc[w, 'company']) for w in ord_words]

    df['starting_word'] = pd.Categorical(df['starting_word'], categories=ord_words, ordered=True)
    df_plot = df[df['starting_word'].isin(ord_words)]

    fig, ax = plt.subplots(figsize=(20, 15))
    sns.boxplot(data=df_plot, x="starting_word", y="chain", order=ord_words,
                showcaps=True, boxprops=dict(alpha=0.85),
                whiskerprops=dict(linestyle='--', alpha=0.7),
                medianprops=dict(color="black", linewidth=2), ax=ax)

    boxes = list(ax.artists)
    if len(boxes) != len(ord_words):
        boxes = [p for p in ax.patches if isinstance(p, mpatches.PathPatch)]
        boxes = boxes[:len(ord_words)]
    for patch, color in zip(boxes, word_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    y_max_by_word = df_plot.groupby('starting_word', observed=False)['chain'].max().reindex(ord_words)
    ymin, ymax = df_plot['chain'].min(), df_plot['chain'].max()
    ypad = 0.02 * (ymax - ymin) if ymax > ymin else 0.5

    label_texts = []
    for i, w in enumerate(ord_words):
        label = str(tops.loc[w, 'model'])
        color = company_color(tops.loc[w, 'company'])
        y_val = y_max_by_word.loc[w] if pd.notna(y_max_by_word.loc[w]) else ymax

        ax.scatter(i, y_val, s=150, color=color, edgecolor='black', linewidth=1.2, zorder=5)
        label_texts.append(
            ax.text(i, y_val + ypad, label, ha='center', va='bottom', fontsize=16,
                    fontweight='bold', color=color, rotation=90)
        )

    ax.set_ylim(None, max(ax.get_ylim()[1], (ymax + 3 * ypad)))
    title = ax.set_title("Path Length Distribution by Word", pad=24)
    ax.set_xlabel("Starting Word")
    ax.set_ylabel("Longest Valid Chain")
    ax.set_xticks(range(len(ord_words)))
    ax.set_xticklabels(ord_words, rotation=90, ha='center')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    for _ in range(4):
        if not label_texts:
            break
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        title_bbox = title.get_window_extent(renderer=renderer)
        required_px = max(
            0.0,
            max((txt.get_window_extent(renderer=renderer).y1 - (title_bbox.y0 - 12))
                for txt in label_texts)
        )
        if required_px <= 0:
            break
        y0_data = ax.transData.inverted().transform((0, 0))[1]
        y1_data = ax.transData.inverted().transform((0, required_px))[1]
        delta_y = max(y1_data - y0_data, ypad)
        lower, upper = ax.get_ylim()
        ax.set_ylim(lower, upper + 1.1 * delta_y)
        plt.tight_layout()

    if save_dir:
        plt.savefig(Path(save_dir) / "boxplot_by_word_colored.png", bbox_inches="tight", dpi=300)
    else:
        plt.show()
    plt.close(fig)

def plot_reasoning_efficiency(df_eff: pd.DataFrame, save_dir: str = None, top_k: int = 25, rank_range: Optional[Tuple[int, int]] = None):
    if df_eff.empty:
        return
    df_agg_eff = (df_eff.groupby(['model', 'company'], as_index=False)
                  .agg(avg_output_tokens=('output_tokens', 'mean'),
                       avg_reasoning_tokens=('reasoning_tokens', 'mean'),
                       avg_longest_chain=('longest_chain', 'mean')))
    # Exclude non-reasoning models (< 2000 avg output tokens)
    df_agg_eff = df_agg_eff[df_agg_eff['avg_output_tokens'] >= 2000]
    df_agg_eff = _apply_rank_window(df_agg_eff, 'avg_longest_chain', descending=True, top_k=top_k, rank_range=rank_range)
    if df_agg_eff.empty:
        return
    df_agg_eff["model"] = df_agg_eff["model"].apply(_normalize_reasoning_efficiency_label)

    fig, ax = plt.subplots(figsize=(20, 15))
    texts = []
    for comp, grp in df_agg_eff.groupby('company'):
        ax.scatter(grp['avg_reasoning_tokens'], grp['avg_longest_chain'],
                   color=COLOR_MAP.get(comp, COLOR_MAP['other']), label=comp, s=200, alpha=0.85,
                   edgecolor='black', linewidth=1.2)
        for _, row in grp.iterrows():
            texts.append(ax.text(row['avg_reasoning_tokens'], row['avg_longest_chain'], row['model'],
                    fontsize=16))
    adjust_text(texts, ax=ax, max_move=40,
                arrowprops=dict(arrowstyle='-', color='grey', lw=0.5))
    ax.set_title("Reasoning Efficiency", pad=20)
    ax.set_xlabel("Average Reasoning Tokens")
    ax.set_ylabel("Average Longest Valid Chain")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title="Company", loc='lower right')
    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / "reasoning_efficiency.png", bbox_inches="tight", dpi=300)
    else:
        plt.show()
    plt.close(fig)

def _result_starting_words(results_path: str) -> List[str]:
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    metadata_words = results.get("metadata", {}).get("starting_words_tested")
    words: List[str] = []

    if isinstance(metadata_words, list):
        words = [str(word) for word in metadata_words if word]
    elif isinstance(metadata_words, dict):
        for values in metadata_words.values():
            if isinstance(values, list):
                words.extend(str(word) for word in values if word)

    if not words:
        result_entries = results.get("results", [])
        if isinstance(result_entries, dict):
            result_entries = result_entries.values()
        for entries in result_entries:
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict) and entry.get("starting_word"):
                        words.append(str(entry["starting_word"]))
            elif isinstance(entries, dict) and entries.get("starting_word"):
                words.append(str(entries["starting_word"]))

    return list(dict.fromkeys(words))


def radial_layout(G: nx.Graph, layer_gap: float = 10.0) -> Dict[str, Tuple[float, float]]:
    """
    Compute a radial layout for a graph with node 'depth' attributes.

    Args:
        G: A NetworkX graph where each node has a 'depth' attribute.
        layer_gap: Radial distance between successive layers (depths).

    Returns:
        A dict mapping each node to an (x, y) position on concentric circles.
    """
    depths = nx.get_node_attributes(G, "depth")
    pos: Dict[str, Tuple[float, float]] = {}
    max_depth = max(depths.values(), default=0)

    for d in range(max_depth + 1):
        nodes_at_d = [n for n, dep in depths.items() if dep == d]
        if not nodes_at_d:
            continue
        radius = d * layer_gap
        for i, node in enumerate(nodes_at_d):
            angle = 2 * math.pi * i / len(nodes_at_d)
            pos[node] = (radius * math.cos(angle), radius * math.sin(angle))

    return pos


def draw_and_save(
        word: str,
        english_words: set[str],
        save_dir: str = None,
        max_nodes: int = 2000,
        layer_gap: float = 10.0,
        figsize: Tuple[int, int] = (16, 16),
        full_graph: bool = False
) -> None:
    """
    Build a Levenshtein-distance graph for the given word and save a radial visualization.

    Args:
        word: The starting word.
        english_words: A set of valid English words.
        save_dir: Optional directory to save plot.
        max_nodes: Soft cap on the total number of nodes in the graph.
        layer_gap: Radial distance between concentric circles (per depth).
        figsize: Size of the figure (width, height).
        full_graph: Build full graph if True (rarely needed).
    """
    G, depths = build_tree(word, english_words,
                           max_nodes=max_nodes,
                           full_graph=full_graph)
    pos = radial_layout(G, layer_gap=layer_gap)

    # Identify dead ends: nodes of degree 1 whose neighbors generate no new valid words
    dead_ends: List[str] = []
    for n in G.nodes:
        if n == word:
            continue
        if G.degree(n) != 1:
            continue
        # If every edit-1 of n is already in depths, it's a true dead end
        if not any(cand not in depths for cand in generate_edit1(n, english_words)):
            dead_ends.append(n)

    node_colors = [
        "green" if n == word else
        "red" if n in dead_ends else
        "grey"
        for n in G.nodes
    ]
    node_sizes = [500 if n == word else 80 for n in G.nodes]

    edge_colors = [
        "red" if (u in dead_ends or v in dead_ends) else "black"
        for u, v in G.edges
    ]

    # Label root and sparse layers
    depth_counts: Dict[int, int] = {}
    for dep in depths.values():
        depth_counts[dep] = depth_counts.get(dep, 0) + 1
    labels: Dict[str, str] = {}
    for n, dep in depths.items():
        if dep == 0 or depth_counts.get(dep, 0) < 10 * dep:
            labels[n] = n

    fig, ax = plt.subplots(figsize=figsize)

    # Draw background rings
    for d in range(1, max(depths.values(), default=0) + 1):
        circle = plt.Circle(
            (0, 0),
            radius=layer_gap * d,
            color="black",
            alpha=0.08,
            fill=True,
            lw=1,
            zorder=0
        )
        ax.add_patch(circle)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=0.6, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    # Legend for node color meaning
    legend_handles = [
        mpatches.Patch(color="green", label="start"),
        mpatches.Patch(color="red", label="dead end (+ edge)"),
        mpatches.Patch(color="grey", label="intermediate")
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.0, 0.93), fontsize=14, title_fontsize=16)

    # Annotate plot with dead-end ratio
    dead_pct = len(dead_ends) / max(len(G), 1) * 100
    ax.set_title(
        f"Simplified Levenshtein graph for {word}\n{dead_pct:.1f}% of nodes are dead-ends",
        pad=20,
        fontsize=20
    )
    ax.axis("off")
    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(save_dir) / f"levenshtein_graph_{word}.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}")
    else:
        plt.show()


def visualize_levenshtein_graphs(
    words: List[str],
    words_file: str = "dictionaries/scowl/scowl_2026_02_25_huge_us_gb_ca_au_ascii.txt",
    save_dir: str = "plots",
    full_graph: bool = False,
) -> List[Path]:
    english_words = load_word_dictionary(words_file)
    outputs = []
    for word in words:
        draw_and_save(word, english_words, save_dir=save_dir, full_graph=full_graph)
        outputs.append(Path(save_dir) / f"levenshtein_graph_{word}.png")
    return outputs


# ==========================
# Driver
# ==========================

def visualize(results_path: str, save: bool = False, top_k: int = 25,
              rank_range: Optional[Tuple[int, int]] = None,
              trajectory_models: Optional[List[str]] = None,
              output_dir: str = "plots"):
    df_agg, df_runs, df_eff, multi_trials_exist = prepare_data(results_path)
    save_dir = output_dir if save else None
    if save_dir:
        Path(save_dir).mkdir(exist_ok=True)

    if multi_trials_exist and not df_agg.empty:
        dfr = _apply_rank_window(df_agg, 'sum_chain_avg', descending=True, top_k=top_k, rank_range=rank_range)
        start = rank_range[0] if rank_range else 1
        print("\n--- LisanBench Ranking (Path Length) ---")
        for i, row in enumerate(dfr.itertuples(index=False), start=start):
            label = _normalize_model_label(row.model)
            print(f"#{i:>3}  {label:<40s}  {row.sum_chain_avg:.2f}")
        print()

        dfr_sparse = _apply_rank_window(df_agg, 'sum_sparse_chain_avg', descending=True, top_k=top_k, rank_range=rank_range)
        print("--- LisanBench Ranking (Difficulty-Weighted) ---")
        for i, row in enumerate(dfr_sparse.itertuples(index=False), start=start):
            label = _normalize_model_label(row.model)
            print(f"#{i:>3}  {label:<40s}  {row.sum_sparse_chain_avg:.2f}")
        print()

        plot_barh(df_agg, 'sum_chain_avg', "LisanBench", "Score",
                  fname="sum_chain_avg.png", fmt=".2f", save_dir=save_dir, top_k=top_k, rank_range=rank_range)
    if not df_agg.empty:
        plot_barh(
            df_agg,
            'sum_sparse_chain_avg',
            "LisanBench: Difficulty-Weighted Score",
            "Difficulty-Weighted Score",
            fname="sum_difficulty_chain_avg.png",
            fmt=".2f",
            save_dir=save_dir,
            top_k=top_k,
            rank_range=rank_range,
        )
        plot_barh(
            df_agg,
            'sum_chain_max',
            "LisanBench: Best-of-Word Score",
            "Score",
            fname="sum_chain_max.png",
            fmt=".2f",
            save_dir=save_dir,
            top_k=top_k,
            rank_range=rank_range,
        )
        default_score_col = "sum_chain_avg" if multi_trials_exist else "sum_chain_max"
        plot_results_over_time(df_agg, save_dir=save_dir, score_col=default_score_col)
        plot_score_vs_sparse_score(
            df_agg,
            save_dir=save_dir,
            top_k=top_k,
            rank_range=rank_range,
        )
    plot_barh(df_agg, 'avg_validity_per_word', "LisanBench: Average Validity Ratio",
              "Average Validity Ratio", fname="avg_validity_per_word.png", fmt=".2f",
              save_dir=save_dir, top_k=top_k, rank_range=rank_range)
    plot_scatter(df_agg, save_dir=save_dir, top_k=top_k, rank_range=rank_range)
    plot_sparse_trajectory(df_runs, models=trajectory_models, save_dir=save_dir)
    plot_distribution_by_word(df_runs, save_dir=save_dir)
    if not df_eff.empty:
        plot_reasoning_efficiency(df_eff, save_dir=save_dir, top_k=top_k, rank_range=rank_range)

def _parse_rank_range(value: str) -> Tuple[int, int]:
    try:
        start_str, end_str = (part.strip() for part in value.split(',', 1))
        start, end = int(start_str), int(end_str)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError("rank range must be in 'start,end' format with integers")

    if start < 1 or end < start:
        raise argparse.ArgumentTypeError("rank range must satisfy 1 <= start <= end")

    return start, end


def main():
    parser = argparse.ArgumentParser(description="LisanBench visualization")
    parser.add_argument("file", nargs="?", default="latest_results.json", help="Path to the results JSON file")
    parser.add_argument("--save", action="store_true", help="Save plots to 'plots' dir as PNG")
    parser.add_argument("--output-dir", default="plots",
                        help="Directory for saved regular plots when --save is used (default: plots)")
    parser.add_argument("--top_k", type=int, default=25, help="Show only top_k best models (default: 25)")
    parser.add_argument("--rank-range", type=_parse_rank_range, help="Limit plots to models ranked between start,end (1-based, inclusive); example: 10,20")
    parser.add_argument("--trajectory-models", nargs="+", metavar="MODEL",
                        help="Models to include in the difficulty trajectory plot (space-separated, no provider prefix)")
    parser.add_argument("--word-graphs-only", action="store_true",
                        help="Generate only Levenshtein graph PNGs for starting words in the results file")
    parser.add_argument("--word-graphs-dir", default="plots",
                        help="Directory for --word-graphs-only output (default: plots)")
    parser.add_argument("--full-graph", "--full_graph", action="store_true",
                        help="Add all edit-1 links among discovered graph nodes")
    args = parser.parse_args()
    try:
        if args.word_graphs_only:
            words = _result_starting_words(args.file)
            if not words:
                raise ValueError(f"No starting words found in {args.file}")
            print(f"Generating {len(words)} Levenshtein graph plots...")
            visualize_levenshtein_graphs(words, save_dir=args.word_graphs_dir, full_graph=args.full_graph)
        else:
            visualize(args.file, save=args.save, top_k=args.top_k, rank_range=args.rank_range,
                      trajectory_models=args.trajectory_models, output_dir=args.output_dir)
    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
