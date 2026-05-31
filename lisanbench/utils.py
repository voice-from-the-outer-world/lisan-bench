from dataclasses import dataclass
import re
import string
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict

import networkx as nx
from wordfreq import zipf_frequency


DEFAULT_WORDS_FILE = "dictionaries/scowl/scowl_2026_02_25_huge_us_gb_ca_au_ascii.txt"
DEFAULT_SCOWL_RAW_FILE = "dictionaries/scowl/scowl-multi-huge.txt"
LEGACY_WORDS_ALPHA_FILE = "dictionaries/words_alpha.txt"
LEGACY_WORDS_ALPHA_ROOT_FILE = "words_alpha.txt"
DEFAULT_SCOWL_URL = (
    "https://app.aspell.net/create?"
    "max_size=80&spelling=US&spelling=GBs&spelling=GBz&spelling=CA&spelling=AU&"
    "max_variant=3&diacritic=strip&download=wordlist&encoding=utf-8&format=inline"
)
DEFAULT_SCOWL_SHA256 = "7c0d7f4f19bacfa3ba95038a048e6b084c9d73e8ba5ebb82130c795bcea3f0f6"


class SparsityWeighter:
    """Assign per-move weights combining sparsity with inverse word frequency."""

    _MAX_ZIPF = 8.0

    def __init__(self, valid_words: set[str]):
        self.valid_words = valid_words
        self._neighbor_cache: Dict[str, set[str]] = {}
        self._rarity_cache: Dict[str, float] = {}

    def neighbors(self, word: str) -> set[str]:
        word = word.strip().lower()
        cached = self._neighbor_cache.get(word)
        if cached is None:
            cached = generate_edit1(word, self.valid_words)
            self._neighbor_cache[word] = cached
        return cached

    def legal_next_moves(self, word: str, seen: set[str]) -> set[str]:
        word = word.strip().lower()
        if not word or word not in self.valid_words:
            return set()
        return self.neighbors(word) - seen

    def rarity(self, word: str) -> float:
        """Higher means rarer in English. Always greater than zero."""
        cached = self._rarity_cache.get(word)
        if cached is not None:
            return cached
        z = zipf_frequency(word, "en")
        if z > 0:
            val = (self._MAX_ZIPF + 1) - z
        else:
            val = self._MAX_ZIPF + 1 + len(word) / 10.0
        self._rarity_cache[word] = val
        return val

    def weight(self, branching_factor: int, next_word: str = "") -> float:
        if branching_factor <= 0:
            return 0.0
        sparsity = 1.0 / branching_factor
        if next_word:
            return sparsity * self.rarity(next_word)
        return sparsity


@dataclass(slots=True)
class ChainAnalysis:
    valid_prefix: List[str]
    stop_reason: str
    sparse_score: float
    sparsity_score: float
    weights_seq: List[float]
    sparsity_seq: List[float]


@dataclass(slots=True)
class _PrefixWalk:
    valid_prefix: List[str]
    longest_valid_from_start: int
    stop_reason: str


def coerce_word_chain(word_chain, raw_response: str = "") -> List[str]:
    if isinstance(word_chain, list):
        words = []
        for word in word_chain:
            normalized = str(word).strip().lower()
            if normalized:
                words.append(normalized)
        return words
    if isinstance(raw_response, str) and raw_response.strip():
        return [word.strip().lower() for word in extract_word_chain(raw_response) if word.strip()]
    return []


def _walk_valid_prefix(
        word_chain: List[str],
        valid_words: Set[str],
        starting_word: Optional[str] = None) -> _PrefixWalk:
    if not word_chain:
        return _PrefixWalk([], 0, "no_output")

    if starting_word is not None:
        expected_start = starting_word.strip().lower()
        actual_start = word_chain[0].strip().lower() if word_chain else ""
        if actual_start != expected_start:
            return _PrefixWalk([], 0, "wrong_start")

    seen_words = set()
    valid_prefix: List[str] = []
    longest_valid_from_start = 0

    for i, current_word in enumerate(word_chain):
        if current_word in seen_words:
            return _PrefixWalk(valid_prefix, longest_valid_from_start, "repeated_word")

        seen_words.add(current_word)

        if not is_valid_english_word(current_word, valid_words):
            return _PrefixWalk(valid_prefix, longest_valid_from_start, "invalid_word")

        valid_prefix.append(current_word)

        if i < len(word_chain) - 1:
            next_word = word_chain[i + 1]
            if next_word in seen_words:
                return _PrefixWalk(valid_prefix, longest_valid_from_start, "repeated_word")
            if not is_valid_english_word(next_word, valid_words):
                return _PrefixWalk(valid_prefix, longest_valid_from_start, "invalid_word")
            if not is_valid_link(current_word, next_word):
                return _PrefixWalk(valid_prefix, longest_valid_from_start, "wrong_edit_distance")
            longest_valid_from_start = i + 1
        else:
            longest_valid_from_start = i

    return _PrefixWalk(valid_prefix, longest_valid_from_start, "end_of_chain")


def _count_total_links(word_chain: List[str],
                       valid_words: Set[str],
                       initial_invalid_links: int = 0) -> Tuple[int, int]:
    total_valid_links = 0
    total_invalid_links = initial_invalid_links
    seen = set()

    for word1, word2 in zip(word_chain, word_chain[1:]):
        if word1 in seen or word2 in seen:
            total_invalid_links += 1
        elif (is_valid_link(word1, word2) and is_valid_english_word(word1, valid_words)
              and is_valid_english_word(word2, valid_words)):
            total_valid_links += 1
        else:
            total_invalid_links += 1
        seen.add(word1)

    return total_valid_links, total_invalid_links


def analyze_word_chain(starting_word: str,
                       word_chain,
                       raw_response: str,
                       sparsity_weighter: SparsityWeighter) -> ChainAnalysis:
    expected_start = str(starting_word).strip().lower()
    chain = coerce_word_chain(word_chain, raw_response)
    valid_words = sparsity_weighter.valid_words
    empty = ChainAnalysis([], "no_output", 0.0, 0.0, [], [])
    if not expected_start or not chain:
        return empty
    if chain[0] != expected_start or expected_start not in valid_words:
        return empty

    walk = _walk_valid_prefix(chain, valid_words, expected_start)
    prefix = walk.valid_prefix
    score = 0.0
    sparsity_score = 0.0
    weights_seq = []
    sparsity_seq = []

    seen = {prefix[0]} if prefix else set()
    prev = prefix[0] if prefix else ""
    for next_word in prefix[1:]:
        neighbors = sparsity_weighter.neighbors(prev)
        legal_next = neighbors - seen
        branching_factor = len(legal_next)
        sparsity_val = 1.0 / branching_factor
        weight_val = sparsity_weighter.weight(branching_factor, next_word)
        score += weight_val
        sparsity_score += sparsity_val
        weights_seq.append(weight_val)
        sparsity_seq.append(sparsity_val)
        seen.add(next_word)
        prev = next_word

    if walk.stop_reason == "end_of_chain":
        if len(prefix) < 2:
            stop_reason = "no_output"
        else:
            stop_reason = "dead_end" if not (sparsity_weighter.neighbors(prev) - seen) else "clean_stop"
    elif walk.stop_reason == "wrong_start":
        stop_reason = "no_output"
    else:
        stop_reason = walk.stop_reason

    return ChainAnalysis(prefix, stop_reason, score, sparsity_score, weights_seq, sparsity_seq)


def get_sparse_move_weights(starting_word: str,
                            word_chain,
                            raw_response: str,
                            sparsity_weighter: SparsityWeighter) -> List[float]:
    """Return per-move sparsity weights for the valid prefix of a chain."""
    return analyze_word_chain(starting_word, word_chain, raw_response, sparsity_weighter).weights_seq


def score_sparse_valid_prefix(starting_word: str,
                              word_chain,
                              raw_response: str,
                              sparsity_weighter: SparsityWeighter) -> float:
    """Score the valid prefix from the required starting word."""
    return analyze_word_chain(starting_word, word_chain, raw_response, sparsity_weighter).sparse_score


def resolve_words_file(words_file: str) -> Path:
    """Resolve benchmark dictionary paths, including the legacy words_alpha location."""
    normalized = words_file.replace("\\", "/")
    if normalized in {"words_alpha.txt", LEGACY_WORDS_ALPHA_FILE}:
        legacy_path = Path(LEGACY_WORDS_ALPHA_FILE)
        if legacy_path.exists():
            return legacy_path
        root_legacy_path = Path(LEGACY_WORDS_ALPHA_ROOT_FILE)
        if root_legacy_path.exists():
            return root_legacy_path

    path = Path(words_file)
    if path.exists():
        return path

    return path


def create_prompt(starting_word: str) -> str:
    """
    Create the prompt for the word chain task.

    Args:
        starting_word: The word to start the chain with

    Returns:
        Formatted prompt string
    """

    return f"""Your goal is to create the longest possible chain of words, where each consecutive word differs by exactly one letter (levenshtein/edit distance = 1).

CRITICAL RULES:
- Start with the word: "{starting_word}"
- Each next word must differ by exactly one letter from the previous word
- You can ADD one letter, REMOVE one letter, or CHANGE one letter
- All words must be valid English words without digits, punctuation, apostrophes, hyphens and whitespaces
- The words are verified using words_alpha.txt from the dwyl/english-words GitHub repo
- NO REPEATED WORDS in the chain! Each word can only appear once!
- Generate as many valid links as possible! Just keep on generating, there is no length limit!

VALID chain examples (with explanation):
- cat, bat (change 'c' to 'b')
- cat, cats (add 's')
- cats, cat (remove 's')

INVALID chain examples (with explanation):
- cat, dog (edit distance = 3, too many changes)
- cat, cat (repeated word cat)
- mat, rat, mat (repeated word mat)

Your task: Starting with "{starting_word}", create the longest possible word chain following these rules.

Format your response as a comma-separated list like this:
{starting_word}, [next_word1], [next_word2], [next_word3], ...

DO NOT RESPOND WITH ANYTHING ELSE BUT THIS LIST!"""


def load_word_dictionary(words_file: str = DEFAULT_WORDS_FILE) -> Set[str]:
    """
    Load valid English words from a benchmark dictionary file.

    Args:
        words_file: Path to the words file

    Returns:
        Set of valid English words (lowercase)

    Raises:
        Exception: For other file reading errors
    """
    try:
        resolved_words_file = resolve_words_file(words_file)
        if words_file == DEFAULT_WORDS_FILE and not resolved_words_file.exists():
            ensure_default_word_dictionary()
            resolved_words_file = resolve_words_file(words_file)
        with open(resolved_words_file, 'r', encoding='utf-8') as f:
            words = set(word.strip().lower() for word in f.readlines() if word.strip())
        return words
    except Exception as e:
        print(f"ERROR loading word dictionary: {e}")
        raise


def ensure_default_word_dictionary() -> None:
    """
    Ensure the pinned default SCOWL dictionary exists locally.

    The derived dictionary is byte-reproducible from the raw ESDB/SCOWL output:
    accept alphabetic tokens, lowercase, sort unique, and write CRLF line endings.
    """
    import hashlib

    dst = Path(DEFAULT_WORDS_FILE)
    if dst.exists():
        digest = hashlib.sha256(dst.read_bytes()).hexdigest()
        if digest != DEFAULT_SCOWL_SHA256:
            print(f"ERROR: {dst} hash mismatch: expected {DEFAULT_SCOWL_SHA256}, got {digest}")
            sys.exit(1)
        return

    raw = Path(DEFAULT_SCOWL_RAW_FILE)
    if not raw.exists():
        download_file(DEFAULT_SCOWL_URL, raw)

    lines = raw.read_text(encoding="utf-8", errors="replace").splitlines()
    words = sorted(
        {
            line.strip().lower()
            for line in lines
            if re.fullmatch(r"[A-Za-z]+", line.strip())
        }
    )
    words = [word for word in words if len(word) != 1 or word in {"a", "i"}]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(("\r\n".join(words) + "\r\n").encode("utf-8"))

    digest = hashlib.sha256(dst.read_bytes()).hexdigest()
    if digest != DEFAULT_SCOWL_SHA256:
        print(f"ERROR: regenerated {dst} hash mismatch: expected {DEFAULT_SCOWL_SHA256}, got {digest}")
        sys.exit(1)


def is_valid_english_word(word: str, valid_words: Set[str]) -> bool:
    """
    Check if a word is a valid English word.

    Args:
        word: Word to check
        valid_words: Set of valid English words

    Returns:
        True if the word is valid, False otherwise
    """
    return word.lower() in valid_words


def download_file(download_url: str, local_path: Path) -> None:
    """
    Download a file from URL if not already present at the local path.

    Args:
        download_url: URL to download from.
        local_path: Local path to save the file.

    Exits:
        On failure to download or save.
    """
    if local_path.exists():
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {local_path.name}...")
    try:
        urllib.request.urlretrieve(download_url, str(local_path))
        print(f"Saved to {local_path}")
    except Exception as e:
        print(f"ERROR: Could not download {download_url} -> {e}")
        sys.exit(1)


def edit_distance(word1: str, word2: str) -> int:
    """
    Calculate the edit distance (Levenshtein distance) between two words.

    Args:
        word1: First word
        word2: Second word

    Returns:
        Edit distance between the two words
    """
    if len(word1) != len(word2):
        # Use dynamic programming for differing lengths.
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return dp[m][n]
    else:
        # For equal length, just count the differing letters.
        return sum(c1 != c2 for c1, c2 in zip(word1, word2))


def generate_edit1(word: str, english_words: Set[str]) -> Set[str]:
    """
    Generate all valid English words at Levenshtein distance 1 from the input word.

    Args:
        word: Base word.
        english_words: Set of valid English words.

    Returns:
        Set of valid words at edit distance 1.
    """
    candidates: Set[str] = set()
    LETTERS = string.ascii_lowercase

    # Deletions
    for i in range(len(word)):
        cand = word[:i] + word[i + 1:]
        if cand in english_words:
            candidates.add(cand)

    # Insertions
    for i in range(len(word) + 1):
        for c in LETTERS:
            cand = word[:i] + c + word[i:]
            if cand in english_words:
                candidates.add(cand)

    # Substitutions
    for i in range(len(word)):
        for c in LETTERS:
            if c == word[i]:
                continue
            cand = word[:i] + c + word[i + 1:]
            if cand in english_words:
                candidates.add(cand)

    candidates.discard(word)
    return candidates


def is_valid_link(word1: str, word2: str) -> bool:
    """
    Check if two words form a valid chain (edit distance = 1).

    Args:
        word1: First word
        word2: Second word

    Returns:
        True if edit distance is 1, else False.
    """
    return edit_distance(word1, word2) == 1


def extract_word_chain(text: str) -> List[str]:
    """
    Extract raw word chain from model response using regex.

    Args:
        text: Raw model response text

    Returns:
        List of words extracted from the response
    """
    lines = text.strip().split('\n')
    words: List[str] = []

    for line in lines:
        # Remove line numbers and arrows for robustness
        clean_line = re.sub(r'^\d+\.\s*', '', line.strip())
        clean_line = re.sub(r'\s*->\s*', ' ', clean_line)
        clean_line = re.sub(r'[^\w\s]', ' ', clean_line)
        words.extend(re.findall(r'\b[a-zA-Z]+\b', clean_line.lower()))

    return words


def validate_chain(
        word_chain: List[str],
        valid_words: Set[str],
        starting_word: Optional[str] = None) -> Tuple[int, int, int]:
    """
    Validate a chain of words, checking all required constraints.

    Args:
        word_chain: List of words in the chain.
        valid_words: Set of valid English words.
        starting_word: If provided, the chain must start with this exact word.

    Returns:
        Tuple:
            - longest_valid_chain_from_start (number of valid transitions from the
              required starting word, up to first invalidity)
            - total_valid_links (total number of valid edit-1 transitions)
            - total_invalid_links (total invalid transitions)
    """
    if len(word_chain) < 2:
        return 0, 0, 0

    if starting_word is not None:
        expected_start = starting_word.strip().lower()
        actual_start = word_chain[0].strip().lower() if word_chain else ""
        if actual_start != expected_start:
            total_valid_links, total_invalid_links = _count_total_links(
                word_chain,
                valid_words,
                initial_invalid_links=1,
            )
            return 0, total_valid_links, total_invalid_links

    walk = _walk_valid_prefix(word_chain, valid_words, starting_word)
    total_valid_links, total_invalid_links = _count_total_links(word_chain, valid_words)

    return walk.longest_valid_from_start, total_valid_links, total_invalid_links


def _bfs_levels(
        root: str,
        english_words: Set[str],
        max_total_nodes: int = None,
        max_frontier_size: int = None
) -> Tuple[List[int], Dict[str, int], nx.Graph]:
    """
    Level-by-level BFS from the root, optionally tracking a graph object.

    Args:
        root: Starting word.
        english_words: Set of valid English words.
        max_total_nodes: If set, soft cap for node expansion.
        max_frontier_size: If set, stop when a frontier exceeds this size.

    Returns:
        levels: Number of nodes at each BFS level.
        depths: Mapping word -> BFS depth.
        G: NetworkX graph of discovered nodes (if max_total_nodes set).
    """
    visited: Set[str] = {root}
    depths: Dict[str, int] = {root: 0}
    levels: List[int] = []

    G: nx.Graph | None = None
    if max_total_nodes is not None:
        G = nx.Graph()
        G.add_node(root, depth=0)

    frontier: List[str] = sorted(generate_edit1(root, english_words) - visited)
    for w in frontier:
        visited.add(w)
        depths[w] = 1
        if G is not None:
            G.add_node(w, depth=1)
            G.add_edge(root, w)

    while frontier:
        nbr_count = len(frontier)
        levels.append(nbr_count)

        if max_frontier_size is not None and nbr_count > max_frontier_size:
            break
        if G is not None and G.number_of_nodes() > max_total_nodes:
            break

        next_frontier: List[str] = []
        for w in frontier:
            for nbr in sorted(generate_edit1(w, english_words)):
                if nbr in visited:
                    continue
                visited.add(nbr)
                depths[nbr] = depths[w] + 1
                next_frontier.append(nbr)
                if G is not None:
                    G.add_node(nbr, depth=depths[nbr])
                    G.add_edge(w, nbr)

        frontier = next_frontier

    return levels, depths, G


def build_tree(
        word: str,
        english_words: Set[str],
        max_nodes: int = 2000,
        full_graph: bool = False
) -> Tuple[nx.Graph, Dict[str, int]]:
    """
    Build a BFS tree of reachable words from a starting word via edit-1 steps.
    Optionally augment with all edit-1 edges among the discovered nodes.

    Args:
        word: Starting/root word.
        english_words: Set of valid English words.
        max_nodes: Cap for nodes to include.
        full_graph: If True, add all edit-1 links among found nodes.

    Returns:
        NetworkX graph (BFS tree or augmented subgraph), and depth dict.
    """
    levels, depths, G = _bfs_levels(root=word,
                                    english_words=english_words,
                                    max_total_nodes=max_nodes,
                                    max_frontier_size=None)

    if full_graph:
        # Add all missing edit-1 edges within the explored node set
        for n in list(G.nodes):
            for nbr in generate_edit1(n, english_words):
                if nbr in depths and not G.has_edge(n, nbr):
                    G.add_edge(n, nbr)

    return G, depths


def compute_neighbor_levels(
        root: str,
        english_words: Set[str],
        threshold: int = 2000
) -> List[int]:
    """
    BFS from `root`, recording the size of each frontier level until one exceeds `threshold`.

    Args:
        root: Starting word.
        english_words: Set of valid English words.
        threshold: Stop after this many nodes in a level.

    Returns:
        List of newly discovered node counts per BFS level (stopping at threshold).
    """
    levels, depths, G = _bfs_levels(root=root,
                                    english_words=english_words,
                                    max_total_nodes=None,
                                    max_frontier_size=threshold)
    return levels


def largest_connected_component_nx(
        english_words: Set[str]
) -> Set[str]:
    """
    Build the full edit-distance-1 graph over all given words,
    then return the largest connected component as a set.

    Args:
        english_words: Set of valid English words.

    Returns:
        Set of words in the largest connected component.

    WARNING: Slow on large word lists; checks every possible edit-1 connection.
    """
    G = nx.Graph()
    G.add_nodes_from(english_words)

    seen_edges = set()
    for w in english_words:
        nbrs = generate_edit1(w, english_words)
        for cand in nbrs:
            # Only add each undirected edge once
            if (w, cand) in seen_edges or (cand, w) in seen_edges:
                continue
            G.add_edge(w, cand)
            seen_edges.add((w, cand))

    largest_cc = max(nx.connected_components(G), key=len)
    return set(largest_cc)
