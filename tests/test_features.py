"""Fixture-scale export → fixed sparse features → archives/diagnostic rendering.

No corpus benchmark, label quality, fitted preprocessing or predictive claims.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from compression import zstd
from pathlib import Path
from random import Random
from unittest.mock import patch

import msgspec
import numpy as np
from test_candidates import fixture, reachable, reference_fragment, renumber, store

from trustmebro.extraction import records as r
from trustmebro.extraction.storage import encode_msgpack
from trustmebro.preprocessing import archives, candidates, features, scheduler


def oracle(theorem: r.Theorem, layout: features.Layout) -> np.ndarray:
    """Independent small-fixture traversal/counting, not production sparse products."""
    rows = np.zeros((len(theorem.trns), 2 * layout.block_width), dtype=np.int64)
    for step, trn in enumerate(theorem.trns):
        roles = ((trn.state.target,), tuple(local.type for local in trn.state.locals))
        for role, roots in enumerate(roles):
            for root in roots:
                seen: set[tuple[int, bytes]] = set()
                for anchor in reachable(theorem.exprs, root):
                    for depth in layout.vocab.depths:
                        edges, nodes = reference_fragment(theorem.exprs, anchor, depth)
                        ident = hashlib.sha256(msgspec.msgpack.encode(edges)).digest()
                        entry = layout.index.get(ident)
                        if entry is None or (anchor, ident) in seen:
                            continue
                        seen.add((anchor, ident))
                        for pos, node in enumerate(nodes):
                            kind = layout.vocab.node_kinds.index(type(theorem.exprs[node]).__name__)
                            col = (
                                role * layout.block_width
                                + layout.offsets[entry]
                                + pos * len(layout.vocab.node_kinds)
                                + kind
                            )
                            rows[step, col] += 1
    return rows


def leaf_theorem() -> r.Theorem:
    return r.Theorem(
        "Heldout.leaf",
        "Heldout",
        None,
        (r.Const("unknown", ()),),
        (r.Trn(r.Tactic("fresh", "fresh"), (0, 1), 1, r.ProofState(0, (), (), ())),),
    )


class FeatureTests(unittest.TestCase):
    def test_export_to_sparse_archives_diagnostics_and_public_rendering(self) -> None:
        # Failures: depth/overlap/role/sharing weights confused; variable numbering
        # changes columns; training vocabulary grows on held-out input; uncovered
        # rows dropped; serial/process/reuse paths disagree; provenance lost;
        # diagnostics count positions instead of patterns or double-count roles;
        # renderer requires source DB/analysis. Establishes exact fixture values,
        # immutable artifacts, shared prep, CLI and archive-to-image integration.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_db, source_db = root / "train.db", root / "source.db"
            observed = root / "candidates.zst"
            inv, vocab_path = root / "inventory.zst", root / "vocab.zst"
            training = (fixture(), renumber(fixture()))
            theorems = (*training, leaf_theorem())
            store(train_db, training)
            store(source_db, theorems)
            scheduler.scan_candidates(train_db, observed, depths=(1, 2, 3))
            scheduler.build_inventory(observed, inv)
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from trustmebro.preprocessing.scheduler import select_main; select_main()",
                    "--inventory",
                    str(inv),
                    "--output",
                    str(vocab_path),
                    "--max-shapes",
                    "100",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            layout = features.compile_vocabulary(archives.read_vocabulary(vocab_path))
            immutable = {path: path.read_bytes() for path in (train_db, source_db, observed, inv, vocab_path)}
            expected = {theorem.name: oracle(theorem, layout) for theorem in theorems}
            with patch.object(candidates, "prepare_adjacency", wraps=candidates.prepare_adjacency) as prepare:
                direct = features.encode_theorem(training[0], layout)
                self.assertEqual(prepare.call_count, 1)
            np.testing.assert_array_equal(direct.matrix.toarray(), expected[training[0].name])
            np.testing.assert_array_equal(expected[training[0].name], expected[training[1].name])
            np.testing.assert_array_equal(expected["Heldout.leaf"], np.zeros_like(expected["Heldout.leaf"]))
            for workers in (1, 2):
                out, stats = root / f"features-{workers}.zst", root / f"stats-{workers}.zst"
                scheduler.convert_corpus(source_db, vocab_path, out, stats, workers=workers, pair_limit=3)
                actual = {rows.name: rows for rows in archives.read_features(out)}
                self.assertEqual(set(actual), set(expected))
                for theorem in theorems:
                    rows = actual[theorem.name]
                    np.testing.assert_array_equal(rows.matrix.toarray(), expected[theorem.name])
                    np.testing.assert_array_equal(rows.steps, np.arange(len(theorem.trns)))
                    self.assertEqual(rows.tactics, tuple(trn.tactic for trn in theorem.trns))
                    self.assertEqual(rows.matrix.dtype, np.dtype(np.int64))
                report = archives.read_diagnostics(stats)
                all_rows = np.vstack(list(expected.values()))
                patterns = np.column_stack(
                    [
                        np.any(all_rows[:, role * layout.block_width + lo : role * layout.block_width + hi], axis=1)
                        for role in (0, 1)
                        for lo, hi in zip(layout.offsets[:-1], layout.offsets[1:], strict=True)
                    ]
                )
                count = len(layout.vocab.entries)
                union = (patterns[:, :count] | patterns[:, count:]).astype(np.int64)
                prefix = union[:, : report.pair_limit]
                np.testing.assert_array_equal(
                    np.frombuffer(report.pairs, dtype=archives.ARRAY_DTYPE).reshape(
                        report.pair_limit, report.pair_limit
                    ),
                    prefix.T @ prefix,
                )
                np.testing.assert_array_equal(
                    np.frombuffer(report.support, dtype=archives.ARRAY_DTYPE), patterns.sum(axis=0)
                )
                self.assertEqual(dict(report.active_dims), Counter(np.count_nonzero(all_rows, axis=1)))
                self.assertEqual(dict(report.active_patterns), Counter(patterns.sum(axis=1)))
                self.assertEqual(
                    (report.theorems, report.states, report.covered_theorems, report.covered_states), (3, 5, 2, 4)
                )
            scheduler.scan_candidates(source_db, root / "all-candidates.zst", depths=(1, 2, 3))
            for workers in (1, 2):
                # Failures: decoding large naturals in the coordinator/worker
                # differs; process transport changes CSR columns or diagnostics;
                # jobs are dropped; candidate reuse invokes graph discovery.
                # Public CLI runs the actual persistent-child initialization.
                out, stats = root / f"reused-{workers}.zst", root / f"reuse-stats-{workers}.zst"
                if workers == 1:
                    with patch.object(features, "extract_candidates", side_effect=AssertionError("rediscovery")):
                        scheduler.convert_corpus(
                            root / "all-candidates.zst", vocab_path, out, stats, candidates=True, pair_limit=3
                        )
                else:
                    subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            'import sys; from trustmebro.preprocessing.pipeline import main; main(["convert", *sys.argv[1:]])',
                            "--candidates",
                            str(root / "all-candidates.zst"),
                            "--vocab",
                            str(vocab_path),
                            "--output",
                            str(out),
                            "--stats",
                            str(stats),
                            "--workers",
                            str(workers),
                            "--cooccurrence-shapes",
                            "3",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                actual = {rows.name: rows for rows in archives.read_features(out)}
                self.assertEqual(set(actual), set(expected))
                for rows in actual.values():
                    np.testing.assert_array_equal(rows.matrix.toarray(), expected[rows.name])
                reused_report = msgspec.structs.asdict(archives.read_diagnostics(stats))
                db_report = msgspec.structs.asdict(report)
                del reused_report["header"], db_report["header"]
                self.assertEqual(reused_report, db_report)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "trustmebro.visualization.cli",
                    "graphs",
                    "--graphs",
                    "features",
                    "--feature-stats",
                    str(root / "reuse-stats-2.zst"),
                    "--output",
                    str(root / "plots"),
                    "--stats",
                    str(root / "no-analysis"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                {path.name for path in (root / "plots").glob("*.png")},
                {"feature-coverage.png", "feature-activation.png", "feature-cooccurrence.png"},
            )
            self.assertFalse((root / "no-analysis").exists())
            for path, original in immutable.items():
                self.assertEqual(path.read_bytes(), original)

    def test_tactic_names_values_and_out_of_scope_state_fields_do_not_change_features(self) -> None:
        # Failures: labels leak into X; names/literal values unexpectedly become
        # channels; local-let values/instance flags or other goals are included;
        # removing duplicate hypotheses doesn't alter context counts correctly.
        # Compact column IDs escape into output when early/middle vocabulary
        # entries are absent. Establishes the agreed feature boundary and global
        # column mapping, not future named channels or throughput.
        theorem = fixture()
        observed = candidates.extract_candidates(theorem, (1, 2, 3))
        entries = tuple(features.Entry(shape.ident, shape.edges) for shape in observed.shapes)
        unused_edges: candidates.Edges = ((1, 2, 3, 4, 5, 6), (), (), (), (), (), ())
        unused = features.Entry(hashlib.sha256(msgspec.msgpack.encode(unused_edges)).digest(), unused_edges)
        vocab = features.Vocabulary((1, 2, 3), (unused, *entries))
        layout = features.compile_vocabulary(vocab)
        original = features.encode_theorem(theorem, layout).matrix.toarray()
        np.testing.assert_array_equal(original, oracle(theorem, layout))
        exprs = list(theorem.exprs)
        exprs[0] = r.Const("completely.different", ())
        exprs[7] = r.NatLiteral(19)
        trns = tuple(
            msgspec.structs.replace(
                trn,
                tactic=r.Tactic("different", "different arguments"),
                open_goal_count=5,
                state=msgspec.structs.replace(
                    trn.state,
                    locals=tuple(
                        r.LocalLet(local.id, local.type, r.LocalDeclKind.AUXILIARY, True, 7, True)
                        for local in trn.state.locals
                    ),
                ),
            )
            for trn in theorem.trns
        )
        changed = msgspec.structs.replace(theorem, exprs=tuple(exprs), trns=trns)
        np.testing.assert_array_equal(features.encode_theorem(changed, layout).matrix.toarray(), original)
        single_hyp = msgspec.structs.replace(
            theorem,
            trns=tuple(
                msgspec.structs.replace(trn, state=msgspec.structs.replace(trn.state, locals=trn.state.locals[1:]))
                for trn in theorem.trns
            ),
        )
        actual = features.encode_theorem(single_hyp, layout).matrix.toarray()
        np.testing.assert_array_equal(actual, oracle(single_hyp, layout))
        np.testing.assert_array_equal(actual[:, : layout.block_width], original[:, : layout.block_width])
        self.assertTrue(np.any(actual[:, layout.block_width :] < original[:, layout.block_width :]))

    def test_failures_and_fixed_vocabulary_on_a_reproducible_source_sample(self) -> None:
        # Failures: output clobbers input/previous artifacts; bad theorem reaches
        # publication; insufficient reuse depths silently miss matches; malformed
        # layout/buffers accepted; source sample uses ordered prefix; incomplete
        # row archive/header accepted; inconsistent diagnostics accepted.
        # Establishes failure atomicity of each artifact
        # and reproducible selection, not an atomic rows+diagnostics transaction.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, observed, inv, vocab = (
                root / name for name in ("src.db", "candidates.zst", "inventory.zst", "vocab.zst")
            )
            theorems = (fixture(), renumber(fixture()), leaf_theorem())
            store(db, theorems)
            scheduler.scan_candidates(db, observed, depths=(1, 2))
            scheduler.build_inventory(observed, inv)
            scheduler.select_archive(inv, vocab)
            out, stats = root / "features.zst", root / "stats.zst"
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    'import sys; from trustmebro.preprocessing.pipeline import main; main(["convert", *sys.argv[1:]])',
                    "--db",
                    str(db),
                    "--vocab",
                    str(vocab),
                    "--output",
                    str(out),
                    "--stats",
                    str(stats),
                    "--limit",
                    "2",
                    "--seed",
                    "5",
                    "--cooccurrence-shapes",
                    "0",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            header = archives.read_feature_header(out)
            self.assertEqual(
                header.selection.theorems, tuple(Random(5).sample([theorem.name for theorem in theorems], 2))
            )
            self.assertEqual(archives.read_diagnostics(stats).pair_limit, 0)
            old_out, old_stats = out.read_bytes(), stats.read_bytes()
            with self.assertRaises(FileExistsError):
                scheduler.convert_corpus(db, vocab, out, stats)
            with self.assertRaises(ValueError):
                scheduler.convert_corpus(db, vocab, db, stats, replace=True)
            with self.assertRaises(ValueError):
                scheduler.convert_corpus(db, vocab, out, stats, theorems=("missing",), replace=True)
            scheduler.scan_candidates(db, root / "shallow.zst", depths=(1,))
            with self.assertRaisesRegex(ValueError, "depths"):
                scheduler.convert_corpus(root / "shallow.zst", vocab, out, stats, candidates=True, replace=True)
            # Failures: a child decoding failure publishes a partial archive or
            # clobbers existing outputs; jobs/futures conceal the exception.
            # Establishes per-artifact failure atomicity for candidate workers.
            with zstd.open(root / "broken-candidates.zst", "wb") as stream:
                archives.write_frame(stream, encode_msgpack(archives.read_selection(observed)))
                archives.write_frame(stream, encode_msgpack(next(archives.read_candidates(observed))))
                archives.write_frame(stream, b"\xc1")  # Reserved, invalid MessagePack code.
            with self.assertRaises(msgspec.DecodeError):
                scheduler.convert_corpus(
                    root / "broken-candidates.zst", vocab, out, stats, candidates=True, workers=2, replace=True
                )
            cyclic = msgspec.structs.replace(
                fixture(),
                name="Bad.cycle",
                exprs=(r.App(0, (0,)),),
                trns=(msgspec.structs.replace(fixture().trns[0], state=r.ProofState(0, (), (), ())),),
            )
            store(root / "bad.db", (fixture(), cyclic))
            with self.assertRaisesRegex(ValueError, "post-order"):
                scheduler.convert_corpus(root / "bad.db", vocab, out, stats, replace=True)
            bad = msgspec.structs.replace(header.vocab, entries=(header.vocab.entries[0], header.vocab.entries[0]))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                features.compile_vocabulary(bad)
            with zstd.open(root / "incomplete.zst", "wb") as stream:
                archives.write_frame(stream, encode_msgpack(header))
            with self.assertRaisesRegex(ValueError, "footer"):
                list(archives.read_features(root / "incomplete.zst"))
            # A valid empty compressed stream, not a zero-byte truncated file.
            (root / "empty.zst").write_bytes(zstd.compress(b""))
            with self.assertRaisesRegex(ValueError, "header"):
                list(archives.read_features(root / "empty.zst"))
            invalid_stats = msgspec.structs.replace(archives.read_diagnostics(stats), covered_states=99)
            with zstd.open(root / "invalid-stats.zst", "wb") as stream:
                archives.write_frame(stream, encode_msgpack(invalid_stats))
            with self.assertRaisesRegex(ValueError, "totals"):
                archives.read_diagnostics(root / "invalid-stats.zst")
            row = next(archives.read_features(out))
            broken = (row.name, row.steps.tobytes(), row.tactics, b"x", b"", b"")
            with self.assertRaises(ValueError):
                archives.unpack_rows(encode_msgpack(broken), row.matrix.shape[1])
            self.assertEqual(out.read_bytes(), old_out)
            self.assertEqual(stats.read_bytes(), old_stats)
            self.assertFalse(list(root.glob(".features-*")))


if __name__ == "__main__":
    unittest.main()
