import argparse
import json
import math
import sys
from pathlib import Path
from typing import Set, Tuple, List, Dict

import matplotlib
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.patches import Patch

from utils import load_word_dictionary, generate_edit1, build_tree

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import pandas as pd

COLOR_MAP = {
    'openai': '#74AA9C', 'meta-llama': '#044EAB', 'anthropic': '#D4C5B9',
    'google': '#669DF7', 'x-ai': '#000000', 'mistralai': '#F54E42',
    'deepseek': '#006994', 'amazon': '#FF9900', 'qwen': '#800080', 'other': '#808080'
}
MARKER_MAP = {
    'openai': 'o', 'meta-llama': 's', 'anthropic': '^',
    'google': 'X', 'x-ai': 'P', 'mistralai': 'v',
    'deepseek': 'D', 'amazon': '>', 'qwen': '<', 'other': '8'
}

plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'legend.title_fontsize': 16
})


def extract_company(model_name: str) -> str:
    """
    Extract the company name from the model name string.

    Args:
        model_name: The full model name string.

    Returns:
        The company name.
    """
    mappings = {
        'amazon/': 'amazon', 'anthropic/': 'anthropic', 'deepseek/': 'deepseek',
        'google/': 'google', 'meta-llama/': 'meta-llama', 'mistralai/': 'mistralai',
        'openai/': 'openai', 'qwen/': 'qwen', 'x-ai/': 'x-ai'
    }
    for prefix, comp in mappings.items():
        if model_name.startswith(prefix):
            return comp
    return model_name.split('/')[0].lower() if '/' in model_name else 'other'


def clean_name(model_name: str) -> str:
    """
    Remove company prefix from model name.

    Args:
        model_name: The full model name string.

    Returns:
        Model name without company prefix.
    """
    return model_name.split('/', 1)[1] if '/' in model_name else model_name


def prepare_data(results_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, bool, List[str]]:
    """
    Prepare data for visualization from results JSON file.

    Args:
        results_path: Path to the results JSON file.

    Returns:
        Tuple containing aggregated DataFrame, runs DataFrame, and a boolean indicating multi-trial existence.
    """
    file_path = Path(results_path)
    data = json.loads(file_path.read_text())
    results = data['results']

    agg_records = []
    run_records = []
    words = []
    multi_trials_exist = False
    for model_full, runs in results.items():
        company = extract_company(model_full)
        model = clean_name(model_full)
        df = pd.DataFrame(runs)
        if df.empty:
            continue
        sum_best = 0
        sum_avg = 0
        avg_validity_per_word = []
        words = df['starting_word'].unique()
        for word in words:
            dsub = df[df['starting_word'] == word]
            if len(dsub) > 1:
                multi_trials_exist = True
            sum_best += dsub['longest_valid_chain'].max()
            sum_avg += dsub['longest_valid_chain'].mean()
            avg_validity_per_word.append(dsub['validity_ratio'].mean())
        agg_records.append({
            'model': model,
            'company': company,
            'sum_chain_best': sum_best,
            'sum_chain_avg': sum_avg,
            'avg_validity_per_word': pd.Series(avg_validity_per_word).mean(),
            'overall_avg_validity': df['validity_ratio'].mean()
        })
        for _, r in df.iterrows():
            run_records.append({
                'starting_word': r['starting_word'],
                'model': model,
                'company': company,
                'chain': r['longest_valid_chain']
            })
    df_agg = pd.DataFrame(agg_records)
    df_runs = pd.DataFrame(run_records)
    return df_agg, df_runs, multi_trials_exist, words


def add_watermark() -> None:
    """
    Add watermark to the plot.
    """
    plt.figtext(
        0.99, 0.01,
        "X: @scaling01 | Lisan al Gaib",
        fontsize=16, color="#888888",
        ha='right', va='bottom', alpha=0.8
    )


def plot_barh(
        df: pd.DataFrame,
        y_col: str,
        color_col: str,
        title: str,
        xlabel: str,
        fname: str = None,
        fmt: str = ".0f",
        save_dir: str = None
) -> None:
    """
    Plot a horizontal bar chart.

    Args:
        df: DataFrame with data to plot.
        y_col: Column name for y-axis values.
        color_col: Column name for color categories.
        title: Plot title.
        xlabel: X-axis label.
        fname: Optional filename to save plot.
        fmt: Format string for value annotations.
        save_dir: Optional directory to save plot.
    """
    handles = [Patch(color=c, label=comp) for comp, c in COLOR_MAP.items()]
    df = df.sort_values(y_col, ascending=False)
    plt.figure(figsize=(20, 14))
    plt.barh(df['model'], df[y_col], color=[COLOR_MAP[c] for c in df[color_col]])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel(xlabel)
    max_val = df[y_col].max()
    for idx, val in enumerate(df[y_col]):
        plt.text(val + max_val * 0.005, idx, f"{val:{fmt}}", va='center', fontsize=14)
    plt.legend(handles=handles, title='Company', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    add_watermark()
    plt.tight_layout()
    if save_dir and fname:
        outpath = Path(save_dir) / fname
        plt.savefig(outpath, bbox_inches='tight')
    else:
        plt.show()


def plot_scatter(df_agg: pd.DataFrame, save_dir: str = None) -> None:
    """
    Plot scatter plot of sum_chain_best vs avg_validity_per_word.

    Args:
        df_agg: Aggregated DataFrame.
        save_dir: Optional directory to save plot.
    """
    handles = [Patch(color=c, label=comp) for comp, c in COLOR_MAP.items()]
    plt.figure(figsize=(20, 14))
    for comp, grp in df_agg.groupby('company'):
        plt.scatter(grp['sum_chain_best'], grp['avg_validity_per_word'],
                    label=comp, c=[COLOR_MAP[comp]], marker=MARKER_MAP[comp],
                    edgecolors='k', s=250)
        for _, row in grp.iterrows():
            plt.text(row['sum_chain_best'], row['avg_validity_per_word'], row['model'],
                     fontsize=12, ha='left', va='bottom')
    plt.title("LisanBench: Chain vs Validity (Per Word)")
    plt.xlabel("Sum of Best Chain per Word")
    plt.ylabel("Average Validity Ratio (Per Word)")
    plt.legend(handles=handles, title='Company', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.grid(True, linestyle='--', alpha=0.5)
    add_watermark()
    plt.tight_layout()
    if save_dir:
        outpath = Path(save_dir) / "scatter_chain_vs_validity.png"
        plt.savefig(outpath, bbox_inches='tight')
    else:
        plt.show()


def plot_boxplots(df_runs: pd.DataFrame, save_dir: str = None) -> None:
    """
    Plot boxplots of chain lengths by starting word.

    Args:
        df_runs: DataFrame of all runs.
        save_dir: Optional directory to save plot.
    """
    best_by_word = df_runs.loc[df_runs.groupby('starting_word')['chain'].idxmax()]

    # Order by median chain length, most difficult/easy words at top
    med = df_runs.groupby('starting_word')['chain'].median().sort_values(ascending=False)
    ord_words = med.index.tolist()
    data_plot2 = [df_runs[df_runs['starting_word'] == w]['chain'].values for w in ord_words]

    plt.figure(figsize=(20, 14))
    bp2 = plt.boxplot(data_plot2, tick_labels=ord_words, vert=False, patch_artist=True)
    for box in bp2['boxes']:
        box.set(facecolor='#ececec')
    for i, w in enumerate(ord_words):
        y = [i + 1] * len(data_plot2[i])
        plt.scatter(data_plot2[i], y, alpha=0.6, color='grey', s=50)
        row = best_by_word[best_by_word['starting_word'] == w].iloc[0]
        plt.scatter(row['chain'], i + 1, color=COLOR_MAP[row['company']],
                    edgecolors='k', s=200, marker='D')
        plt.text(row['chain'] + 1, i + 1, row['model'], va='center', fontsize=14)
    plt.title("LisanBench: Distribution by Word (Ordered by Median)")
    plt.xlabel("Longest Valid Chain")
    plt.ylabel("Starting Word")
    add_watermark()
    plt.tight_layout()
    if save_dir:
        outpath = Path(save_dir) / "boxplot_distribution_by_word.png"
        plt.savefig(outpath, bbox_inches='tight')
    else:
        plt.show()


def plot_boxplots_per_model_by_word(df_runs: pd.DataFrame, save_dir: str = None) -> None:
    """
    For each model, plot a boxplot: x-axis = starting words,
    each box = distribution of chain lengths (all trials) for that word for this model.

    Args:
        df_runs: DataFrame of all runs.
        save_dir: Optional directory to save plot.
    """
    models = sorted(df_runs['model'].unique())
    for model in models:
        sub = df_runs[df_runs['model'] == model]
        words = sorted(sub['starting_word'].unique())
        data_plot = [sub[sub['starting_word'] == w]['chain'].values for w in words]
        company = sub['company'].iloc[0]
        plt.figure(figsize=(20, 10))
        bp = plt.boxplot(data_plot, tick_labels=words, vert=False, patch_artist=True)
        for box in bp['boxes']:
            box.set(facecolor=COLOR_MAP[company])
        for i, w in enumerate(words):
            y = [i + 1] * len(data_plot[i])
            plt.scatter(data_plot[i], y, alpha=0.6, color='grey', s=50)
        plt.title(f"LisanBench: {model} Distribution by Word")
        plt.xlabel("Chain Length")
        plt.ylabel("Starting Word")
        add_watermark()
        plt.tight_layout()
        if save_dir:
            fname = f"boxplot_{model.replace('/', '_').replace(' ', '_')}_by_word.png"
            plt.savefig(Path(save_dir) / fname, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


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
        english_words: Set[str],
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
    add_watermark()
    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(save_dir) / f"levenshtein_graph_{word}.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize_levenshtein_graphs(
        words: List[str],
        words_file: str = "words_alpha.txt",
        save_dir: str = None,
        full_graph: bool = False
) -> None:
    """
    Generate and save visualizations for Levenshtein graphs of a list of words.

    Args:
        words: List of words to visualize.
        words_file: Path to word dictionary.
        save_dir: Optional directory to save plot.
        full_graph: Build full graph if True (rarely needed).
    """
    words_set = load_word_dictionary(words_file)
    for w in words:
        draw_and_save(w, words_set, save_dir=save_dir, full_graph=full_graph)


def visualize(results_path: str, save: bool = False, full_graph: bool = False) -> None:
    """
    Show all LisanBench visualizations for the results JSON file.
    Optionally, save all plots in 'plots/' directory.

    Args:
        results_path: Path to results JSON file.
        save: If True, saves plots, otherwise shows interactively.
    """
    df_agg, df_runs, multi_trials_exist, words = prepare_data(results_path)
    save_dir = "plots" if save else None
    if save_dir:
        Path(save_dir).mkdir(exist_ok=True)
    plot_barh(df_agg, 'sum_chain_best', 'company',
              "LisanBench: Sum of Best Chain per Word",
              "Sum of Best Chain per Word", fname="sum_chain_best.png", fmt=".0f", save_dir=save_dir)
    if multi_trials_exist:
        plot_barh(df_agg, 'sum_chain_avg', 'company',
                  "LisanBench: Sum of Average Chain per Word",
                  "Sum of Average Chain per Word", fname="sum_chain_avg.png", fmt=".2f", save_dir=save_dir)
        plot_boxplots_per_model_by_word(df_runs, save_dir=save_dir)

    plot_barh(df_agg, 'avg_validity_per_word', 'company',
              "LisanBench: Average Validity Ratio (Per Word)",
              "Average Validity Ratio (Per Word)", fname="avg_validity_per_word.png", fmt=".2f", save_dir=save_dir)
    plot_scatter(df_agg, save_dir=save_dir)
    plot_boxplots(df_runs, save_dir=save_dir)
    visualize_levenshtein_graphs(words, save_dir=save_dir, full_graph=full_graph)


def main() -> None:
    """
    Command-line entrypoint for LisanBench visualization tool.
    Parses arguments and generates plots.
    """
    parser = argparse.ArgumentParser(
        description="LisanBench visualization tool",
        epilog="Example: python visualize.py results.json"
    )

    parser.add_argument(
        "file",
        nargs="?",
        default="results.json",
        help="Path to the results JSON file (default: results.json)"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save plots to a directory called 'plots' (default: show interactively only)"
    )

    parser.add_argument(
        "--full_graph",
        action="store_true",
        help="Render the entire graph. Warning: this can be extremely cluttered and is usually not recommended."
    )

    args = parser.parse_args()

    try:
        visualize(args.file, save=args.save, full_graph=args.full_graph)

    except Exception as e:
        print(f"Error visualizing {args.file}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
