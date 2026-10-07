import unittest
from pathlib import Path
from types import SimpleNamespace

from trustmebro.stage1.extract import (
    CorpusStats,
    _atomic_theorem_records,
    _proof_length_summary,
    tactic_head,
)


class TacticHeadTests(unittest.TestCase):
    def test_identifier_and_punctuation(self) -> None:
        self.assertEqual(tactic_head("  simp [Nat.add_comm]"), "simp")
        self.assertEqual(tactic_head("rw? [h]"), "rw?")
        self.assertEqual(tactic_head("exact' h"), "exact'")
        self.assertIsNone(tactic_head("· simp"))


class AtomicRecordTests(unittest.TestCase):
    def test_only_normalized_fields_are_emitted(self) -> None:
        repository = SimpleNamespace(url="repo", commit="0" * 40)
        tactics = [
            SimpleNamespace(tactic="  rw [h]  ", state_before="x : α\n⊢ P x"),
            SimpleNamespace(tactic="exact h", state_before="⊢ Q"),
        ]
        theorem = SimpleNamespace(
            repo=repository,
            file_path=Path("Mathlib/A.lean"),
            theorem=SimpleNamespace(full_name="Example.theorem"),
            get_traced_tactics=lambda *, atomic_only: tactics,
        )
        records = list(_atomic_theorem_records([theorem], repository))

        self.assertEqual(
            records,
            [
                {
                    "file_path": "Mathlib/A.lean",
                    "theorem": "Example.theorem",
                    "tactic_index": 0,
                    "state_before": "x : α\n⊢ P x",
                    "tactic": "rw [h]",
                    "tactic_head": "rw",
                },
                {
                    "file_path": "Mathlib/A.lean",
                    "theorem": "Example.theorem",
                    "tactic_index": 1,
                    "state_before": "⊢ Q",
                    "tactic": "exact h",
                    "tactic_head": "exact",
                },
            ],
        )


class CorpusStatisticsTests(unittest.TestCase):
    def test_theorem_and_proof_length_statistics(self) -> None:
        stats = CorpusStats()
        for theorem, length in (("a", 1), ("b", 2), ("c", 3), ("d", 4)):
            for index in range(length):
                stats.add(
                    {
                        "file_path": "Mathlib/A.lean",
                        "theorem": theorem,
                        "tactic": "simp" if index else "·",
                        "tactic_head": "simp" if index else None,
                    }
                )

        self.assertEqual(stats.transitions, 10)
        self.assertEqual(len(stats.files), 1)
        self.assertEqual(len(stats.theorem_lengths), 4)
        self.assertEqual(stats.tactic_heads, {"simp": 6})
        self.assertEqual(
            _proof_length_summary(stats),
            {
                "definition": "retained transition records per represented theorem",
                "minimum": 1,
                "first_quartile": 1.75,
                "median": 2.5,
                "mean": 2.5,
                "third_quartile": 3.25,
                "p90": 3.7,
                "p95": 3.85,
                "p99": 3.97,
                "maximum": 4,
                "single_record_theorems": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
