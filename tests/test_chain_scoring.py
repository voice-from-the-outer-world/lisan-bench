import unittest
from unittest.mock import patch

from lisanbench.utils import analyze_word_chain, resolve_words_file, validate_chain


VALID_WORDS = {"bat", "cat", "dog", "hat"}


class DummyWeighter:
    valid_words = VALID_WORDS
    _neighbors = {
        "cat": {"bat"},
        "bat": {"cat", "hat"},
        "hat": {"bat"},
        "dog": set(),
    }

    def neighbors(self, word: str) -> set[str]:
        return set(self._neighbors.get(word, set()))

    def weight(self, branching_factor: int, next_word: str = "") -> float:
        return branching_factor + 0.25


class ChainScoringCompatibilityTests(unittest.TestCase):
    def test_legacy_words_alpha_prefers_dictionaries_path(self):
        def fake_exists(path):
            return str(path).replace("\\", "/") in {
                "words_alpha.txt",
                "dictionaries/words_alpha.txt",
            }

        with patch("pathlib.Path.exists", fake_exists):
            self.assertEqual(
                "dictionaries/words_alpha.txt",
                str(resolve_words_file("words_alpha.txt")).replace("\\", "/"),
            )

    def test_validate_chain_preserves_current_edge_case_scores(self):
        cases = [
            (["cat", "bat", "hat"], "cat", (2, 2, 0)),
            (["hat", "bat", "hat"], "hat", (1, 1, 1)),
            (["bat", "cat"], "cat", (0, 1, 1)),
            (["cat", "cot"], "cat", (0, 0, 1)),
            (["cat", "dog"], "cat", (0, 0, 1)),
            (["cat"], "cat", (0, 0, 0)),
            ([], "cat", (0, 0, 0)),
        ]

        for chain, starting_word, expected in cases:
            with self.subTest(chain=chain, starting_word=starting_word):
                self.assertEqual(expected, validate_chain(chain, VALID_WORDS, starting_word))

    def test_sparse_analysis_uses_same_valid_prefix_for_valid_starting_words(self):
        weighter = DummyWeighter()
        cases = [
            (["cat", "bat", "hat"], "cat"),
            (["hat", "bat", "hat"], "hat"),
            (["cat", "cot"], "cat"),
            (["cat", "dog"], "cat"),
            (["cat"], "cat"),
        ]

        for chain, starting_word in cases:
            with self.subTest(chain=chain, starting_word=starting_word):
                longest, _, _ = validate_chain(chain, VALID_WORDS, starting_word)
                analysis = analyze_word_chain(starting_word, chain, "", weighter)
                self.assertEqual(longest, max(len(analysis.valid_prefix) - 1, 0))

    def test_sparse_analysis_preserves_stop_reasons_and_weights(self):
        weighter = DummyWeighter()

        analysis = analyze_word_chain("cat", ["cat", "bat", "hat"], "", weighter)
        self.assertEqual(["cat", "bat", "hat"], analysis.valid_prefix)
        self.assertEqual("dead_end", analysis.stop_reason)
        self.assertEqual([1.25, 1.25], analysis.weights_seq)
        self.assertEqual(2.5, analysis.sparse_score)

        repeated = analyze_word_chain("hat", ["hat", "bat", "hat"], "", weighter)
        self.assertEqual(["hat", "bat"], repeated.valid_prefix)
        self.assertEqual("repeated_word", repeated.stop_reason)
        self.assertEqual([1.25], repeated.weights_seq)

        wrong_start = analyze_word_chain("cat", ["bat", "cat"], "", weighter)
        self.assertEqual([], wrong_start.valid_prefix)
        self.assertEqual("no_output", wrong_start.stop_reason)


if __name__ == "__main__":
    unittest.main()
