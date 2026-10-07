"""Tiny candidate-archive -> coverage -> feature-equivalence checks; no tracing/JIT."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from compression import zstd
from pathlib import Path
from unittest.mock import patch

import msgspec
import numpy as np
from scipy.sparse import csr_array

from trustmebro.extraction import records as r
from trustmebro.extraction.storage import encode_msgpack
from trustmebro.preprocessing import archives, coverage, features, scheduler
from trustmebro.preprocessing import candidates as c


def fixture(name: str) -> c.Candidates:
    edges = (((),), ((1, 2), (), ()), ((1, 2, 2), (), ()))
    shapes = tuple(c.Shape(hashlib.sha256(msgspec.msgpack.encode(edge)).digest(), edge) for edge in edges)
    nodes = tuple(
        c.Node(idx, expr, 0, 0)
        for idx, expr in enumerate((r.Const("f", ()), r.Const("g", ()), r.Fvar(0), r.App(0, (2,)), r.App(1, (2, 2))))
    )
    occurrences = tuple(c.Occurrence(0, depth, (ref,)) for ref in (0, 1, 2) for depth in (1, 2))
    occurrences += (c.Occurrence(1, 1, (3, 0, 2)), c.Occurrence(1, 2, (3, 0, 2)), c.Occurrence(2, 1, (4, 1, 2)))
    roots = (c.Root(3, (0, 2, 3)), c.Root(4, (1, 2, 4)), c.Root(0, (0,)))
    tactic = r.Tactic("Fixture.tactic", "fixture")
    return c.Candidates(
        name, nodes, shapes, occurrences, roots, (c.State(0, tactic, 3, (4, 0, 4)), c.State(1, tactic, 4, (3, 0)))
    )


class CoverageTests(unittest.TestCase):
    def test_projected_filtered_archive_preserves_memberships_and_full_reader(self) -> None:
        # Failures: omitted expression data is still decoded; held-out payloads
        # reach the full decoder; request order changes the positional join;
        # unsorted/repeated closures alter binary membership or occurrence counts;
        # subset duplicates/missing names go unnoticed; large naturals regress in
        # the full reader. Establishes archive -> coverage parity and reader scope,
        # not a timing claim or validation of omitted feature-only fields.
        first = fixture("first")
        first = msgspec.structs.replace(
            first,
            nodes=(
                first.nodes[0],
                msgspec.structs.replace(first.nodes[1], expr=r.NatLiteral(1 << 100)),
                *first.nodes[2:],
            ),
            roots=(c.Root(3, (3, 2, 0, 2)), c.Root(4, (4, 2, 1, 2)), first.roots[2]),
        )
        second = msgspec.structs.replace(first, name="second")
        expected = coverage.build_coverage_index((first, second), (1, 2))
        held_out = ["held-out", "not decoded as candidate fields"]
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "candidates.zst", Path(tmp) / "coverage.zst"
            selection = c.Selection("unused.db", 0, 0, (1, 2), None, None)
            archives.publish(
                src,
                (encode_msgpack(selection), encode_msgpack(first), encode_msgpack(held_out), encode_msgpack(second)),
                sources=(),
                replace=False,
            )
            before = src.read_bytes()
            with (
                patch.object(archives, "decode_candidates", side_effect=AssertionError("full candidate decoding")),
                patch.object(archives, "decode_nat_ext", side_effect=AssertionError("omitted expression decoding")),
            ):
                actual = scheduler.prepare_coverage_archive(src, dst, theorems=("second", "first"))
            header, restored = archives.read_coverage(dst)
            self.assertEqual(header.selection.theorems, ("first", "second"))
            self.assertEqual(actual.names, expected.names)
            for index in (actual, restored):
                np.testing.assert_array_equal(index.matches.toarray(), expected.matches.toarray())
                np.testing.assert_array_equal(
                    index.matches.toarray(), np.tile(((1, 1, 0), (1, 0, 1), (1, 0, 0)), (2, 1))
                )
                for col in ("goals", "hyps", "refs", "offsets", "support"):
                    np.testing.assert_array_equal(getattr(index, col), getattr(expected, col))
                np.testing.assert_array_equal(index.shapes.idents, expected.shapes.idents)
                self.assertEqual(index.shapes.data, expected.shapes.data)
                self.assertTrue(index.matches.has_canonical_format)
            decoder = archives.decode_candidates
            with patch.object(archives, "decode_candidates", wraps=decoder) as decoded:
                full = tuple(archives.read_candidates(src, names=("second", "first")))
            self.assertEqual(decoded.call_count, 2)
            self.assertEqual(full, (first, second))
            for names, message in (
                (("missing",), "lacks requested"),
                (("first", "first"), "distinct"),
                ((), "nonempty"),
            ):
                with self.subTest(names=names), self.assertRaisesRegex(ValueError, message):
                    tuple(archives.read_coverage_candidates(src, names=names))
            with self.assertRaises(msgspec.ValidationError):
                tuple(archives.read_candidates(src))
            self.assertEqual(src.read_bytes(), before)

    def test_projected_shape_validation_and_failed_publication(self) -> None:
        # Failures: raw adjacency bypasses conflict/identity checks; an equivalent
        # noncanonical integer encoding is rejected; malformed relevant fields
        # or duplicate selected names replace an existing artifact. Establishes
        # projected-reader validation/publication, not full feature validation.
        first, second = fixture("first"), fixture("second")
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "candidates.zst", Path(tmp) / "coverage.zst"
            selection = c.Selection("unused.db", 0, 0, (1, 2), None, None)

            def source(record: object) -> None:
                archives.publish(
                    src,
                    (encode_msgpack(selection), encode_msgpack(first), msgspec.msgpack.encode(record)),
                    sources=(),
                    replace=True,
                )

            record = msgspec.msgpack.decode(encode_msgpack(second))
            # Same edges ((1, 2), (), ()), with 1 encoded as uint8, not fixint.
            record[2][1][1] = msgspec.Raw(b"\x93\x92\xcc\x01\x02\x90\x90")
            source(record)
            actual = scheduler.prepare_coverage_archive(src, dst)
            expected = coverage.build_coverage_index((first, second), (1, 2))
            self.assertEqual(actual.shapes.data, expected.shapes.data)
            np.testing.assert_array_equal(actual.matches.toarray(), expected.matches.toarray())
            before = dst.read_bytes()
            record[2][1][1] = [[2, 1], [], []]
            source(record)
            with self.assertRaisesRegex(ValueError, "conflicting"):
                scheduler.prepare_coverage_archive(src, dst, replace=True)
            self.assertEqual(dst.read_bytes(), before)
            for field in ("identity", "reference", "anchor", "adjacency", "duplicate"):
                record = msgspec.msgpack.decode(encode_msgpack(second))
                match field:
                    case "identity":
                        record[2][1][0] = b"invalid"
                    case "reference":
                        record[1][0][0] = "invalid"
                    case "anchor":
                        record[3][0][2] = []
                    case "adjacency":
                        record[2][1][1] = "invalid"
                    case "duplicate":
                        record[0] = "first"
                source(record)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    scheduler.prepare_coverage_archive(src, dst, theorems=("first", "second"), replace=True)
                self.assertEqual(dst.read_bytes(), before)

    def test_public_coverage_selection_and_conversion(self) -> None:
        # Failures: CLI dispatches to ranking, leaf-only hypotheses disappear,
        # eligible roots lose rich coverage, selection settings/provenance are
        # lost, archives don't compile/convert, or publication alters sources.
        # Establishes the complete candidate -> cover -> frozen feature path;
        # not predictive quality, optimality, or corpus-scale memory/runtime.
        theorem = fixture("first")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, cov, vocab = (root / name for name in ("candidates.zst", "coverage.zst", "vocab.zst"))
            selection = c.Selection("training.db", 0, 0, (1, 2), None, (theorem.name,))
            with zstd.open(src, "wb") as stream:
                for record in (selection, theorem):
                    archives.write_frame(stream, encode_msgpack(record))
            original = src.read_bytes()

            def command(fn: str, *args: str) -> dict:
                invocation = (
                    'import sys; from trustmebro.preprocessing.pipeline import main; main(["convert", *sys.argv[1:]])'
                    if fn == "convert"
                    else f"from trustmebro.preprocessing.scheduler import {fn}; {fn}()"
                )
                run = subprocess.run(
                    [sys.executable, "-c", invocation, *args], capture_output=True, text=True, check=True, timeout=30
                )
                return json.loads(run.stdout)

            command("coverage_main", "--candidates", str(src), "--output", str(cov))
            before = cov.read_bytes()
            report = command(
                "select_main",
                "--coverage",
                str(cov),
                "--output",
                str(vocab),
                "--objective",
                "dims",
                "--improvement-steps",
                "10",
            )
            self.assertEqual(report["eligible_roots"], 2)
            self.assertEqual(report["fallback_only_roots"], 1)
            self.assertEqual((report["rich"]["covered_goals"], report["rich"]["covered_hyps"]), (2, 3))
            self.assertEqual((report["total"]["covered_goals"], report["total"]["covered_hyps"]), (2, 5))
            self.assertLessEqual(report["selected_cost"], report["greedy_cost"])
            loaded = archives.read_vocabulary(vocab)
            self.assertEqual(loaded.selection, selection)
            self.assertEqual(loaded.coverage, features.CoverPolicy("dims", 10))
            self.assertEqual({entry.ident for entry in loaded.entries}, {shape.ident for shape in theorem.shapes})
            out, stats = root / "features.zst", root / "stats.zst"
            command(
                "convert", "--candidates", str(src), "--vocab", str(vocab), "--output", str(out), "--stats", str(stats)
            )
            (rows,) = archives.read_features(out)
            layout = features.compile_vocabulary(loaded)
            np.testing.assert_array_equal(
                rows.matrix.toarray(), features.encode_candidates(theorem, layout).matrix.toarray()
            )
            self.assertTrue(np.all(rows.matrix[:, : layout.block_width].sum(axis=1) > 0))
            self.assertEqual(archives.read_diagnostics(stats).states, 2)
            self.assertEqual(src.read_bytes(), original)
            self.assertEqual(cov.read_bytes(), before)
            previous = vocab.read_bytes()
            # A native result can be structurally valid but miss an eligible
            # hypothesis; independent verification must prevent publication.
            with (
                patch.object(coverage, "_native_cover", return_value=(np.empty(0, dtype=np.int64), 0)),
                self.assertRaisesRegex(ValueError, "eligible individual"),
            ):
                scheduler.select_coverage_archive(cov, vocab, replace=True)
            with self.assertRaises(FileExistsError):
                scheduler.select_coverage_archive(cov, vocab)
            with self.assertRaises(ValueError):
                scheduler.select_coverage_archive(cov, cov, replace=True)
            self.assertEqual(vocab.read_bytes(), previous)

    def test_cost_objectives_fallback_and_native_pruning(self) -> None:
        # A narrow incidence fixture is needed to distinguish entry costs from
        # dimension costs unambiguously, independent of candidate discovery.
        # Failures: costs/rows/columns transposed, fallback trivializes selection,
        # support thresholds discard population, search worsens cost, or local
        # column numbering changes the result. Does not establish optimality.
        base = coverage.build_coverage_index((fixture("first"),), (1, 2))
        chain = tuple((idx + 1,) if idx < 6 else () for idx in range(7))
        wide = c.Shape(hashlib.sha256(msgspec.msgpack.encode(chain)).digest(), chain)
        index = coverage.CoverageIndex(
            coverage.packed_shapes((*fixture("first").shapes, wide)),
            csr_array(np.asarray(((1, 1, 0, 1), (1, 0, 1, 1), (1, 0, 0, 0)), dtype=bool)),
            base.goals,
            base.hyps,
            base.names,
            base.offsets,
            base.refs,
            np.ones(4, dtype=np.int64),
        )
        for steps in (0, 10):
            entries = coverage.select_coverage(index, policy=features.CoverPolicy("entries", steps))
            dims = coverage.select_coverage(index, policy=features.CoverPolicy("dims", steps))
            self.assertEqual(set(entries.cols), {0, 3})
            self.assertEqual(set(dims.cols), {0, 1, 2})
            for result in (entries, dims):
                self.assertEqual(result.summary.total.covered_roots, 3)
                self.assertEqual(result.summary.rich.covered_roots, 2)
                self.assertLessEqual(result.summary.selected_cost, result.summary.greedy_cost)
                rich = result.cols[1:]
                for col in rich:
                    others = rich[rich != col]
                    self.assertTrue(np.any(coverage.coverage_counts(index, others)[:2] == 0))
        fallback = coverage.select_coverage(index, min_support=2)
        self.assertEqual(tuple(fallback.cols), (0,))
        self.assertEqual(fallback.summary.fallback_only_roots, 3)
        self.assertEqual(fallback.summary.rich.covered_roots, 0)
        for policy in (features.CoverPolicy("bad"), features.CoverPolicy("dims", -1)):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                coverage.select_coverage(index, policy=policy)
        with self.assertRaisesRegex(ValueError, "min_nodes"):
            coverage.select_coverage(index, min_nodes=1)
        # The real leaf is not interchangeable with a frontier wildcard.
        missing = coverage.CoverageIndex(
            coverage.packed_shapes((*fixture("first").shapes[1:], wide)),
            index.matches[:, 1:],
            index.goals,
            index.hyps,
            index.names,
            index.offsets,
            index.refs,
            index.support[1:],
        )
        with self.assertRaisesRegex(ValueError, "real-leaf"):
            coverage.select_coverage(missing)
        unary_edges = ((1,), ())
        unary = c.Shape(hashlib.sha256(msgspec.msgpack.encode(unary_edges)).digest(), unary_edges)
        pruning = coverage.CoverageIndex(
            coverage.packed_shapes((*fixture("first").shapes, unary)),
            csr_array(np.asarray(((1, 1, 0, 1), (1, 0, 1, 1), (1, 1, 0, 0), (1, 0, 1, 0)), dtype=bool)),
            np.ones(4, dtype=np.int64),
            np.zeros(4, dtype=np.int64),
            ("pruning",),
            np.asarray((0, 4), dtype=np.int64),
            np.arange(4, dtype=np.int64),
            np.ones(4, dtype=np.int64),
        )
        # Greedy first selects the cheaper unary pattern; later necessary
        # entries cover its roots too, so pruning must remove it.
        result = coverage.select_coverage(pruning)
        self.assertEqual(set(result.cols), {0, 1, 2})
        self.assertEqual((result.summary.greedy_cost, result.summary.selected_cost), (208, 156))

    def test_archive_to_coverage_preserves_individual_hypotheses_and_converter_membership(self) -> None:
        # Failures: a covered hypothesis hides another; duplicate depths/anchors
        # inflate coverage; repeats lose occurrence weight; theorem-local IDs
        # merge across proofs; global column remapping changes shapes; archive
        # codecs alter raw buffers or shape identities; fallback is misreported
        # as rich coverage. Establishes membership parity with feature conversion,
        # not solver behavior, full-corpus resource use or predictive quality.
        first = fixture("first")
        second = msgspec.structs.replace(
            fixture("second"),
            shapes=tuple(reversed(first.shapes)),
            occurrences=tuple(msgspec.structs.replace(occ, shape=2 - occ.shape) for occ in first.occurrences),
        )
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "candidates.zst", Path(tmp) / "coverage.zst"
            selection = c.Selection("unread-source.db", 0, 0, (1, 2), None, None)
            with zstd.open(src, "wb") as stream:
                for record in (selection, first, second):
                    archives.write_frame(stream, encode_msgpack(record))
            before = src.read_bytes()
            index = scheduler.prepare_coverage_archive(src, dst)
            header, loaded = archives.read_coverage(dst)
            self.assertEqual(header.selection, selection)
            self.assertEqual(src.read_bytes(), before)
            np.testing.assert_array_equal(index.shapes.idents, loaded.shapes.idents)
            np.testing.assert_array_equal(index.shapes.offsets, loaded.shapes.offsets)
            np.testing.assert_array_equal(index.shapes.nodes, loaded.shapes.nodes)
            self.assertEqual(index.shapes.data, loaded.shapes.data)
            np.testing.assert_array_equal(loaded.matches.toarray(), np.tile(((1, 1, 0), (1, 0, 1), (1, 0, 0)), (2, 1)))
            np.testing.assert_array_equal(loaded.goals, (1, 1, 0, 1, 1, 0))
            np.testing.assert_array_equal(loaded.hyps, (1, 2, 2, 1, 2, 2))
            np.testing.assert_array_equal(loaded.offsets, (0, 3, 6))
            np.testing.assert_array_equal(loaded.support, (2, 2, 2))
            rich = coverage.eligible_shapes(loaded)
            report = coverage.coverage_report(loaded, rich)
            self.assertEqual((report.goals, report.covered_goals, report.hyps, report.covered_hyps), (4, 4, 10, 6))
            with self.assertRaisesRegex(ValueError, "4 individual hypothesis"):
                coverage.require_coverage(loaded, rich)
            coverage.require_coverage(loaded, (0,))
            self.assertEqual(coverage.coverage_report(loaded, ()).covered_roots, 0)
            np.testing.assert_array_equal(coverage.shape_costs(loaded, "dims"), (26, 78, 78))
            self.assertEqual(coverage.coverage_report(loaded, (0,)).hyps, 10)
            for col, shape in enumerate(first.shapes):
                layout = features.compile_vocabulary(
                    features.Vocabulary((1, 2), (features.Entry(shape.ident, shape.edges),))
                )
                rows = features.encode_candidates(first, layout)
                goal_match = rows.matrix[:, : layout.block_width].sum(axis=1) > 0
                hyp_match = rows.matrix[:, layout.block_width :].sum(axis=1) > 0
                matches = index.matches[:3, col].toarray().ravel().astype(bool)
                root_rows = {root.ref: idx for idx, root in enumerate(first.roots)}
                np.testing.assert_array_equal(goal_match, [matches[root_rows[state.goal]] for state in first.states])
                np.testing.assert_array_equal(
                    hyp_match, [any(matches[root_rows[ref]] for ref in state.hyps) for state in first.states]
                )

    def test_depth_filter_packed_chunks_and_selection(self) -> None:
        # Failures: higher-depth-only/unused shapes survive registration; shared
        # shapes are removed; compact columns misaddress archived shape IDs;
        # a root without a rich match is dropped; chunks change exact incidence
        # or weights; all adjacency is decoded again during selection; incomplete
        # archives appear successful. Establishes the complete prepare/archive/
        # select/convert path, not a full-corpus peak-memory or timing guarantee.
        first = fixture("first")
        extra_edges = ((None,), ((1,), ()))
        extra = tuple(c.Shape(hashlib.sha256(msgspec.msgpack.encode(edges)).digest(), edges) for edges in extra_edges)
        first = msgspec.structs.replace(
            first,
            shapes=(*first.shapes, *extra),
            occurrences=tuple(
                msgspec.structs.replace(occ, depth=9) if occ.shape == 2 else occ for occ in first.occurrences
            )
            + (c.Occurrence(3, 9, (3,)),),
        )
        with tempfile.TemporaryDirectory() as tmp:
            src, dst, vocab = (Path(tmp) / name for name in ("candidates.zst", "coverage.zst", "vocab.zst"))
            selection = c.Selection("unused.db", 0, 0, (1, 2, 9), None, None)
            with zstd.open(src, "wb") as stream:
                archives.write_frame(stream, encode_msgpack(selection))
                for idx in range(12):
                    archives.write_frame(stream, encode_msgpack(msgspec.structs.replace(first, name=f"proof{idx}")))
            before = src.read_bytes()
            with patch.object(archives, "COVERAGE_CHUNK_BYTES", 64):
                prepared = scheduler.prepare_coverage_archive(src, dst, depths=(1, 2))
                header, loaded = archives.read_coverage(dst)
                report = scheduler.select_coverage_archive(dst, vocab)
            self.assertEqual(header.depths, (1, 2))
            self.assertEqual(src.read_bytes(), before)
            self.assertEqual(len(loaded.support), 2)
            np.testing.assert_array_equal(loaded.shapes.idents, [np.void(shape.ident) for shape in first.shapes[:2]])
            np.testing.assert_array_equal(loaded.matches.toarray(), np.tile(((1, 1), (1, 0), (1, 0)), (12, 1)))
            np.testing.assert_array_equal(loaded.support, (12, 12))
            np.testing.assert_array_equal(loaded.goals, prepared.goals)
            np.testing.assert_array_equal(loaded.hyps, prepared.hyps)
            self.assertEqual((report.total.goals, report.total.hyps, report.total.covered_roots), (24, 60, 36))
            self.assertEqual(report.fallback_only_roots, 24)
            layout = features.compile_vocabulary(archives.read_vocabulary(vocab))
            rows = features.encode_candidates(first, layout)
            self.assertEqual(rows.matrix.shape[0], 2)
            frames = list(archives.read_frames(dst))
            decoder = msgspec.msgpack.Decoder(
                type=archives.CoverageBlock | archives.CoverageNames | archives.CoverageEnd
            )
            blocks = [
                record for data in frames[1:] if isinstance(record := decoder.decode(data), archives.CoverageBlock)
            ]
            self.assertGreater(sum(block.col == "indices" for block in blocks), 1)
            self.assertTrue(all(len(block.data) <= 64 for block in blocks))
            damaged = Path(tmp) / "incomplete.zst"
            archives.publish(damaged, frames[:-1], sources=(dst,), replace=False)
            with self.assertRaisesRegex(ValueError, "completion"):
                archives.read_coverage(damaged)
            previous = vocab.read_bytes()
            with self.assertRaises(ValueError):
                scheduler.select_coverage_archive(damaged, vocab, replace=True)
            self.assertEqual(vocab.read_bytes(), previous)
            # Only selected adjacency needs decoding, but corrupted selected
            # cost metadata must not silently change the published vocabulary.
            altered = []
            for data in frames[1:]:
                record = decoder.decode(data)
                if isinstance(record, archives.CoverageBlock) and record.col == "nodes":
                    vals = np.frombuffer(record.data, dtype=archives.ARRAY_DTYPE).copy()
                    vals[-1] = 2
                    record = msgspec.structs.replace(record, data=vals.tobytes())
                    data = encode_msgpack(record)
                altered.append(data)
            damaged = Path(tmp) / "bad-cost.zst"
            archives.publish(damaged, (frames[0], *altered), sources=(dst,), replace=False)
            with self.assertRaisesRegex(ValueError, "node count"):
                scheduler.select_coverage_archive(damaged, vocab, replace=True)
            self.assertEqual(vocab.read_bytes(), previous)

    def test_invalid_input_and_failed_publication_preserve_existing_artifacts(self) -> None:
        # Successful public-path fixtures cannot exercise corrupt references,
        # identity conflicts, index guard exhaustion or incomplete archives.
        # This narrow failure check establishes rejection/publication behavior,
        # not a total-memory cap or comprehensive adversarial archive validation.
        first = fixture("first")
        with self.assertRaisesRegex(ValueError, "unknown"):
            coverage.root_coverage(
                msgspec.structs.replace(first, states=(msgspec.structs.replace(first.states[0], goal=999),)), (1, 2)
            )
        with self.assertRaisesRegex(ValueError, "duplicate coverage theorem"):
            coverage.build_coverage_index((first, first), (1, 2))
        invalid = msgspec.structs.replace(
            first, shapes=(msgspec.structs.replace(first.shapes[0], ident=b"bad"), *first.shapes[1:])
        )
        with self.assertRaisesRegex(ValueError, "identity"):
            coverage.build_coverage_index((invalid,), (1, 2))
        # Inject accounting pressure: testing actual exhaustion would require
        # an unnecessarily large workload during the active corpus extraction.
        with patch.object(coverage.sys, "getsizeof", return_value=2**20), self.assertRaises(MemoryError):
            coverage.build_coverage_index((first,), (1, 2), index_memory_mib=1)
        index = coverage.build_coverage_index((first,), (1, 2))
        for columns in ((1, 1), (-1,), (99,), (0.5,)):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                coverage.coverage_counts(index, columns)
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "candidates.zst", Path(tmp) / "coverage.zst"
            selection = c.Selection("unused.db", 0, 0, (1, 2), None, None)
            with zstd.open(src, "wb") as stream:
                for record in (selection, first):
                    archives.write_frame(stream, encode_msgpack(record))
            scheduler.prepare_coverage_archive(src, dst)
            previous = dst.read_bytes()
            with self.assertRaises(FileExistsError):
                scheduler.prepare_coverage_archive(src, dst)
            with self.assertRaises(ValueError):
                scheduler.prepare_coverage_archive(src, src, replace=True)
            with self.assertRaisesRegex(ValueError, "depths"):
                scheduler.prepare_coverage_archive(src, dst, depths=(5,), replace=True)
            with zstd.open(src, "ab") as stream:
                archives.write_frame(stream, b"\xc1")
            with self.assertRaises(msgspec.DecodeError):
                scheduler.prepare_coverage_archive(src, dst, replace=True)
            self.assertEqual(dst.read_bytes(), previous)
            self.assertFalse(list(Path(tmp).glob(".archive-*")))
            with zstd.open(dst, "wb") as stream:
                archives.write_frame(
                    stream,
                    encode_msgpack(archives.CoverageHeader(selection, (1, 2), str(src), 0, 0, "packed-coverage")),
                )
            with self.assertRaises(ValueError):
                archives.read_coverage(dst)

    def test_packed_array_length_boundaries(self) -> None:
        # Narrow codec boundary: naturally obtaining a >65,535-position fragment
        # end to end would require disproportionate graph construction. Failures:
        # fixarray/array16/array32 length confusion, signed byte/shift overflow,
        # a header reading into the next shape, or trusting bad cost metadata.
        # Establishes shallow length validation, not adjacency graph validity.
        edges = tuple(((),) * count for count in (1, 15, 16, 65535, 65536))
        shapes = coverage.packed_shapes(
            c.Shape(hashlib.sha256(msgspec.msgpack.encode(row)).digest(), row) for row in edges
        )
        archives._check_shape_counts(shapes)
        bad = coverage.PackedShapes(shapes.idents, shapes.data, shapes.offsets, shapes.nodes + 1)
        with self.assertRaisesRegex(ValueError, "node count"):
            archives._check_shape_counts(bad)
        for data, message in ((b"\xdc", "truncated"), (b"\xdd\x00\x00", "truncated"), (b"\x01", "array")):
            bad = coverage.PackedShapes(
                shapes.idents[:1], memoryview(data), np.asarray((0, len(data)), dtype=np.int64), shapes.nodes[:1]
            )
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, message):
                archives._check_shape_counts(bad)


if __name__ == "__main__":
    unittest.main()
