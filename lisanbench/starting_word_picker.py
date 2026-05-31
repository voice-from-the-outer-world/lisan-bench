import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Set, List, Dict

from lisanbench.utils import (
    ensure_default_word_dictionary,
    load_word_dictionary,
    generate_edit1,
    edit_distance,
    download_file,
    largest_connected_component_nx,
    compute_neighbor_levels
)


def categorize_by_difficulty(words: List[str], english_words: Set[str], num_levels: int = 5) -> Dict[int, List[str]]:
    """
    Categorize words into difficulty levels based on their neighbor connectivity.
    Uses fixed-width bins over the connectivity score so each bucket covers the same score range.
    Bucket 0 = easiest (highest connectivity), bucket num_levels-1 = hardest (lowest connectivity).

    Args:
        words: List of candidate words
        english_words: Set of all valid English words
        num_levels: Number of difficulty levels to create

    Returns:
        Dict mapping difficulty level (0=easiest, num_levels-1=hardest) to list of words
    """
    # Compute connectivity (level-1 + level-2 neighbors) for each word
    connectivity_scores = []
    for word in words:
        level1 = set(generate_edit1(word, english_words))
        total_connectivity = len(level1)
        connectivity_scores.append((word, total_connectivity))

    # Fixed-width binning in score space
    max_score = max((score for _, score in connectivity_scores), default=0)
    width = max(1, max_score // num_levels)
    difficulty_buckets = defaultdict(list)

    for word, score in connectivity_scores:
        raw_idx = score // width if width > 0 else 0
        if raw_idx >= num_levels:
            raw_idx = num_levels - 1
        # Invert so highest connectivity -> bucket 0 (easiest)
        bucket_idx = (num_levels - 1) - raw_idx
        difficulty_buckets[bucket_idx].append(word)

    return dict(difficulty_buckets)


def categorize_by_length(words: List[str], min_len: int, max_len: int) -> Dict[int, List[str]]:
    """
    Categorize words by their length.

    Args:
        words: List of words to categorize
        min_len: Minimum word length
        max_len: Maximum word length

    Returns:
        Dict mapping word length to list of words of that length
    """
    length_buckets = defaultdict(list)
    for word in words:
        if min_len <= len(word) <= max_len:
            length_buckets[len(word)].append(word)
    return dict(length_buckets)


def compute_min_edit_distance_to_set(candidate: str, selected_words: List[str]) -> float:
    """
    Compute minimum edit distance from candidate to any word in selected_words.

    Args:
        candidate: Word to evaluate
        selected_words: List of already selected words

    Returns:
        Minimum edit distance (returns infinity if selected_words is empty)
    """
    if not selected_words:
        return float('inf')
    return min(edit_distance(candidate, selected) for selected in selected_words)


def pick_diverse_starting_words(
        english_words: Set[str],
        common_list: List[str],
        common_set: Set[str],
        largest_cc: Set[str],
        num_words: int = 10,
        min_length: int = 3,
        max_length: int = 8,
        difficulty_levels: int = 5,
        min_edit_distance: float = 3.0,
        max_common_words: int = 5000
) -> List[str]:
    """
    Select diverse starting words with equal distribution across difficulty and length.

    Args:
        english_words: Set of all valid English words
        common_list: List of common words ordered by frequency
        common_set: Set version of common_list for O(1) lookup
        largest_cc: Set of words in largest connected component
        num_words: Number of words to select
        min_length: Minimum word length
        max_length: Maximum word length
        difficulty_levels: Number of difficulty levels to distribute across
        min_edit_distance: Minimum edit distance between selected words
        max_common_words: Maximum number of common words to consider

    Returns:
        List of selected starting words
    """
    print(f"Filtering candidates from top {max_common_words} common words...")

    # Filter candidates
    candidates = []
    for word in common_list[:max_common_words]:
        if (min_length <= len(word) <= max_length and
                word in english_words and
                word in common_set and
                word in largest_cc):
            # Must have at least one edit-1 neighbor
            if generate_edit1(word, english_words):
                candidates.append(word)

    print(f"Found {len(candidates)} valid candidates")

    if len(candidates) < num_words:
        print(f"Warning: Only {len(candidates)} candidates found, less than requested {num_words}")
        return candidates

    # Categorize by difficulty and length
    print("Categorizing by difficulty levels...")
    difficulty_buckets = categorize_by_difficulty(candidates, english_words, difficulty_levels)

    print("Categorizing by word length...")
    length_buckets = categorize_by_length(candidates, min_length, max_length)

    # Calculate target distribution
    words_per_difficulty = max(1, num_words // difficulty_levels)
    words_per_length = max(1, num_words // len(length_buckets))

    print(f"Target: ~{words_per_difficulty} words per difficulty level")
    print(f"Target: ~{words_per_length} words per length")

    selected_words = []
    used_words = set()

    # Strategy: Alternate between difficulty-based and length-based selection
    selection_attempts = 0
    max_attempts = num_words * 10  # Prevent infinite loops

    while len(selected_words) < num_words and selection_attempts < max_attempts:
        selection_attempts += 1

        # Try difficulty-based selection
        if len(selected_words) % 2 == 0:
            # Find difficulty level with fewest selected words
            difficulty_counts = defaultdict(int)
            for word in selected_words:
                for level, words in difficulty_buckets.items():
                    if word in words:
                        difficulty_counts[level] += 1
                        break

            target_difficulty = min(difficulty_counts.keys() if difficulty_counts else difficulty_buckets.keys(),
                                    key=lambda x: difficulty_counts[x])
            candidate_pool = [w for w in difficulty_buckets[target_difficulty] if w not in used_words]
        else:
            # Find length with fewest selected words
            length_counts = defaultdict(int)
            for word in selected_words:
                length_counts[len(word)] += 1

            target_length = min(length_buckets.keys(),
                                key=lambda x: length_counts[x])
            candidate_pool = [w for w in length_buckets[target_length] if w not in used_words]

        if not candidate_pool:
            # Fall back to any unused candidate
            candidate_pool = [w for w in candidates if w not in used_words]

        if not candidate_pool:
            break

        # From candidate pool, select word with maximum minimum edit distance to selected words
        best_word = None
        best_min_distance = -1

        for candidate in candidate_pool:
            min_dist = compute_min_edit_distance_to_set(candidate, selected_words)
            if min_dist >= min_edit_distance and min_dist > best_min_distance:
                best_min_distance = min_dist
                best_word = candidate

        # If no word meets min_edit_distance requirement, relax it
        if best_word is None and selected_words:
            for candidate in candidate_pool:
                min_dist = compute_min_edit_distance_to_set(candidate, selected_words)
                if min_dist > best_min_distance:
                    best_min_distance = min_dist
                    best_word = candidate

        # If still no word found, pick first available
        if best_word is None and candidate_pool:
            best_word = candidate_pool[0]

        if best_word:
            selected_words.append(best_word)
            used_words.add(best_word)

    return selected_words


def compute_average_pairwise_distance(words: List[str]) -> float:
    """Compute average pairwise edit distance between words."""
    if len(words) < 2:
        return 0.0

    total_distance = 0
    pairs = 0
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            total_distance += edit_distance(words[i], words[j])
            pairs += 1

    return total_distance / pairs


def print_word_analysis(words: List[str], common_list: List[str], english_words: Set[str]):
    """Print detailed analysis of selected words."""
    print("\n" + "=" * 60)
    print("SELECTED STARTING WORDS ANALYSIS")
    print("=" * 60)

    for i, word in enumerate(words, 1):
        # Get frequency rank
        try:
            rank = common_list.index(word) + 1
        except ValueError:
            rank = "N/A"

        # Get neighbor levels
        levels = compute_neighbor_levels(word, english_words, threshold=8000)
        levels_str = ",".join(str(x) for x in levels)

        print(f"{i:2d}. {word:<12} len={len(word):<2d} rank={rank:<6} levels={levels_str}")

    # Summary statistics
    avg_distance = compute_average_pairwise_distance(words)
    length_distribution = defaultdict(int)
    for word in words:
        length_distribution[len(word)] += 1

    print(f"\nAverage pairwise edit distance: {avg_distance:.2f}")
    print("Length distribution:", dict(length_distribution))

    # Python list format for easy copying
    print(f"\nPython list format:")
    print(f"{words}")


def main():
    parser = argparse.ArgumentParser(
        description="Pick diverse starting words for word chain game",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--num-words", "-n", type=int, default=20,
        help="Number of starting words to select"
    )
    parser.add_argument(
        "--min-length", type=int, default=3,
        help="Minimum word length"
    )
    parser.add_argument(
        "--max-length", type=int, default=9,
        help="Maximum word length"
    )
    parser.add_argument(
        "--difficulty-levels", type=int, default=10,
        help="Number of difficulty levels to distribute across"
    )
    parser.add_argument(
        "--min-edit-distance", type=float, default=3.0,
        help="Minimum edit distance between selected words"
    )
    parser.add_argument(
        "--max-common-words", type=int, default=5000,
        help="Maximum number of common words to consider as candidates"
    )
    parser.add_argument(
        "--common-words-url", default=(
            "https://raw.githubusercontent.com/"
            "first20hours/google-10000-english/master/google-10000-english.txt"
        ),
        help="URL for Google 10k common words list"
    )

    args = parser.parse_args()

    # Download and load common words
    google_filename = "google-10000-english.txt"
    google_path = Path(google_filename)

    download_file(args.common_words_url, google_path)
    ensure_default_word_dictionary()

    common_list = []
    with google_path.open("r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                common_list.append(word)
    common_set = set(common_list)

    # Load English dictionary
    print("Loading English word dictionary...")
    english_words = load_word_dictionary()

    # Find largest connected component
    print("Finding largest connected component (this may take a moment)...")
    largest_cc = largest_connected_component_nx(english_words)
    print(f"Largest connected component contains {len(largest_cc)} words")

    # Select diverse starting words
    print(f"\nSelecting {args.num_words} diverse starting words...")
    selected_words = pick_diverse_starting_words(
        english_words=english_words,
        common_list=common_list,
        common_set=common_set,
        largest_cc=largest_cc,
        num_words=args.num_words,
        min_length=args.min_length,
        max_length=args.max_length,
        difficulty_levels=args.difficulty_levels,
        min_edit_distance=args.min_edit_distance,
        max_common_words=args.max_common_words
    )

    if not selected_words:
        print("No suitable words found!")
        sys.exit(1)

    # Sort by difficulty (ascending connectivity)
    print("Sorting by difficulty...")
    word_difficulties = []
    for word in selected_words:
        level1 = set(generate_edit1(word, english_words))
        level2 = set()
        for neighbor in level1:
            level2.update(generate_edit1(neighbor, english_words))
        level2 -= level1
        level2.discard(word)
        connectivity = len(level1) + len(level2)
        word_difficulties.append((word, connectivity))

    word_difficulties.sort(key=lambda x: x[1])  # Sort by connectivity (easiest first)
    sorted_words = [word for word, _ in word_difficulties]

    # Print results
    print_word_analysis(sorted_words, common_list, english_words)


if __name__ == "__main__":
    main()
