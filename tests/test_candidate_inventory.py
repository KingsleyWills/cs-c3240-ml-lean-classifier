"""Candidate archive → global inventory → selective public rendering.

These fixture-scale checks establish counting and representation contracts, not
full-corpus memory/runtime, final vocabulary usefulness or classifier quality.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from compression import zstd
from pathlib import Path
from unittest.mock import patch

import msgspec
import numpy as np
from test_candidates import fixture, renumber, store

from trustmebro.extraction.storage import encode_msgpack
from trustmebro.preprocessing import archives, candidates, inventory, scheduler


class InventoryTests(unittest.TestCase):
    def test_source_to_inventory_and_four_diagnostic_groups(self) -> None:
        # Failures: depth duplicates counted in union; nested shapes collapsed;
        # sharing/frontiers altered; role weights/support confused; renumbering
        # changes identities; sources modified; graph dispatch requires a DB or
        # unrelated analysis; atlas loses parallel edges. Establishes the full
        # public artifact/render path and independent exact counting oracle.
        # It does not measure rendering speed or assert pixel-exact layouts.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, source, output = root / "source.db", root / "candidates.zst", root / "inventory.zst"
            store(db, (fixture(), renumber(fixture())))
            original_db = db.read_bytes()
            scheduler.scan_candidates(db, source, depths=(1, 2, 3, 4, 5))
            original_src = source.read_bytes()
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from trustmebro.preprocessing.scheduler import inventory_main; inventory_main()",
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("uncompressed stream", result.stderr)
            actual = archives.read_inventory(output)
            expected: dict[bytes, Counter[str]] = {}
            edges: dict[bytes, bytes] = {}
            for theorem in archives.read_candidates(source):
                weights = {node.ref: (node.goal_count, node.hyp_count) for node in theorem.nodes}
                seen: set[tuple[bytes, int]] = set()
                support: set[bytes] = set()
                role_support: set[tuple[bytes, str]] = set()
                depth_support: set[tuple[bytes, int]] = set()
                for occ in theorem.occurrences:
                    shape = theorem.shapes[occ.shape]
                    edges[shape.ident] = msgspec.msgpack.encode(shape.edges)
                    count = expected.setdefault(shape.ident, Counter())
                    goal, hyp = weights[occ.nodes[0]]
                    for key, val in (("anchors", 1), ("goals", goal), ("hyps", hyp)):
                        count[f"{occ.depth}:{key}"] += val
                        if (shape.ident, occ.nodes[0]) not in seen:
                            count[key] += val
                    seen.add((shape.ident, occ.nodes[0]))
                    support.add(shape.ident)
                    depth_support.add((shape.ident, occ.depth))
                    if goal:
                        role_support.add((shape.ident, "goal_theorems"))
                    if hyp:
                        role_support.add((shape.ident, "hyp_theorems"))
                for ident in support:
                    expected[ident]["theorems"] += 1
                for ident, role in role_support:
                    expected[ident][role] += 1
                for ident, depth in depth_support:
                    expected[ident][f"{depth}:theorems"] += 1
            self.assertEqual(actual.summary.theorems, 2)
            self.assertEqual(actual.summary.shapes, len(expected))
            for row, (ident, packed) in enumerate(actual.shapes):
                self.assertEqual(packed, edges[ident])
                canonical = inventory.shape_edges(packed)
                np.testing.assert_array_equal(
                    actual.counts[row, :3],
                    (
                        len(canonical),
                        sum(len(refs) for refs in canonical if refs is not None),
                        sum(refs is None for refs in canonical),
                    ),
                )
                self.assertEqual(actual.col("theorems")[row], 2)
                for key in inventory.BASE_COLS[3:]:
                    self.assertEqual(actual.col(key)[row], expected[ident][key])
                for depth in actual.summary.selection.depths:
                    for key in inventory.DEPTH_COLS:
                        self.assertEqual(actual.col(key, depth)[row], expected[ident][f"{depth}:{key}"])
            self.assertTrue(
                any(
                    sum(actual.col("anchors", depth)[row] for depth in (1, 2, 3, 4, 5)) > actual.col("anchors")[row]
                    for row in range(len(actual.shapes))
                )
            )
            self.assertTrue(any(None in inventory.shape_edges(packed) for _, packed in actual.shapes))
            self.assertTrue(any(() in inventory.shape_edges(packed) for _, packed in actual.shapes))
            self.assertEqual(db.read_bytes(), original_db)
            self.assertEqual(source.read_bytes(), original_src)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "trustmebro.visualization.cli",
                    "graphs",
                    "--graphs",
                    "candidates",
                    "--inventory",
                    str(output),
                    "--stats",
                    str(root / "no-analysis"),
                    "--output",
                    str(root / "plots"),
                    "--atlas-min-nodes",
                    "2",
                    "--node-types",
                    "7",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                {path.name for path in (root / "plots").glob("*.png")},
                {
                    "candidate-growth.png",
                    "candidate-size-occurrences.png",
                    "candidate-size-support.png",
                    "candidate-coverage.png",
                    "candidate-atlas.png",
                },
            )
            self.assertFalse((root / "no-analysis").exists())

    def test_oversized_role_weights_remain_exact_through_aggregation_and_archive(self) -> None:
        # Failures: local reduction/global addition wraps uint64, casts to float,
        # or loses MessagePack natural extensions/corrections at block boundaries.
        # Real fixture-sized transition
        # lists cannot produce >2**64 root occurrences, so this deliberately
        # supplies weighted observations at the archive boundary. It establishes
        # exact counter/storage behavior, not source-state/weight consistency.
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source.zst", Path(tmp) / "inventory.zst"
            from trustmebro.preprocessing.candidates import extract_candidates

            original = extract_candidates(fixture(), (1, 2))
            for huge in ((1 << 63) + 11, 1 << 100):
                with self.subTest(weight=huge):
                    theorem = msgspec.structs.replace(
                        original,
                        nodes=tuple(
                            msgspec.structs.replace(node, goal_count=huge, hyp_count=huge) for node in original.nodes
                        ),
                    )
                    with zstd.open(source, "wb") as stream:
                        archives.write_frame(
                            stream, encode_msgpack(candidates.Selection("fixture", 0, 0, (1, 2), None, None))
                        )
                        for name in ("first", "second", "third"):
                            archives.write_frame(stream, encode_msgpack(msgspec.structs.replace(theorem, name=name)))
                    with patch.object(archives, "BLOCK_ROWS", 2):
                        scheduler.build_inventory(source, output, replace=True)
                    actual = archives.read_inventory(output)
                    np.testing.assert_array_equal(actual.col("goals"), actual.col("anchors").astype(object) * huge)
                    np.testing.assert_array_equal(actual.col("hyps", 1), actual.col("anchors", 1).astype(object) * huge)
                    self.assertTrue(actual.large)

    def test_failed_inventory_keeps_sources_and_previous_output(self) -> None:
        # Failures: guard silently drops shapes, partial publication replaces an
        # old artifact, source/output collision destroys source, truncated input
        # is accepted. Establishes failure atomicity, not an RSS memory bound.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, source, output = root / "source.db", root / "source.zst", root / "inventory.zst"
            store(db, (fixture(),))
            scheduler.scan_candidates(db, source)
            scheduler.build_inventory(source, output)
            previous, original = output.read_bytes(), source.read_bytes()
            with self.assertRaises(FileExistsError):
                scheduler.build_inventory(source, output)
            with self.assertRaises(ValueError):
                scheduler.build_inventory(source, source, replace=True)
            with patch.object(inventory.ShapeCounts, "index_bytes", return_value=2**30), self.assertRaises(MemoryError):
                scheduler.build_inventory(source, output, index_memory_mib=1, replace=True)
            bad = root / "truncated.zst"
            with zstd.open(bad, "wb") as stream:
                stream.write(b"\0")
            with self.assertRaises(ValueError):
                scheduler.build_inventory(bad, output, replace=True)
            self.assertEqual(output.read_bytes(), previous)
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse(list(root.glob(".inventory-*")))


if __name__ == "__main__":
    unittest.main()
