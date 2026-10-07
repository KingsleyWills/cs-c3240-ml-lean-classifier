"""Fixture DB → theorem-disjoint partitions → labeled queries and fixed sparse rows."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import numpy as np
from scipy.sparse import csr_array
from test_candidates import fixture

from trustmebro.extraction import records as r
from trustmebro.extraction.storage import (
    encode_exprs,
    encode_trns,
    open_extraction_db,
    recompress_db,
    stored_dicts,
    train_dicts,
)
from trustmebro.preprocessing import candidates, partition
from trustmebro.preprocessing.corpus import LabelPolicy, labeled_features, open_corpus, read_label_policy
from trustmebro.preprocessing.features import Entry, Vocabulary, compile_vocabulary, encode_theorem

EXACT = "Lean.Parser.Tactic.exact"
RW = "Lean.Parser.Tactic.rwSeq"
UNKNOWN = "Fixture.unsupported"
POLICY = LabelPolicy({EXACT: "rule_application", RW: "rewrite"}, "drop")


def theorem(name: str, kinds: tuple[str, ...]) -> r.Theorem:
    base = fixture()
    trns = tuple(
        msgspec.structs.replace(
            base.trns[step % len(base.trns)], tactic=r.Tactic(kind, f"source action {step}"), src_span=(step, step + 1)
        )
        for step, kind in enumerate(kinds)
    )
    return msgspec.structs.replace(base, name=name, trns=trns)


def store(path: Path, theorems: tuple[r.Theorem, ...], *, dicts: bool = False) -> None:
    with closing(open_extraction_db(path)) as db:
        db.executemany(
            "INSERT INTO theorems(name,module,src_path,src_start,src_end,expr_count,trn_count,exprs,trns) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                (
                    item.name,
                    item.module,
                    "Fixture/shared.lean",
                    *item.src_span,
                    len(item.exprs),
                    len(item.trns),
                    encode_exprs(item.exprs),
                    encode_trns(item.trns),
                )
                for item in theorems
            ),
        )
        if dicts:
            recompress_db(db, train_dicts(path, dict_size=512))


def population() -> tuple[r.Theorem, ...]:
    return (
        *(theorem(f"Proof.{idx}", (EXACT, UNKNOWN, RW if idx < 4 else EXACT)) for idx in range(8)),
        theorem("Drop.only", (UNKNOWN,)),
    )


class CorpusTests(unittest.TestCase):
    def test_partition_command_inherits_nested_split_defaults(self) -> None:
        # Failures: CLI shadows changed split/quota defaults; overriding one
        # quota resets another. Checks public command configuration, not repair.
        cfg = partition.SplitCfg(0.35, 42, partition.MinSupport(7, 3), partition.MinSupport(9, 4), 19)
        for flags, expected in (
            ([], cfg),
            (["--test-min-trns", "12"], partition.SplitCfg(0.35, 42, cfg.train_min, partition.MinSupport(12, 4), 19)),
        ):
            with (
                self.subTest(flags=flags),
                patch.object(partition, "DEFAULT_SPLIT_CFG", cfg),
                patch.object(partition, "read_label_policy", return_value=POLICY),
                patch.object(partition, "partition_corpus", side_effect=RuntimeError("stop before partition")) as split,
                self.assertRaisesRegex(RuntimeError, "stop before partition"),
            ):
                partition.main(["--db", "unused.db", "--labels", "unused.json", "--output", "unused"] + flags)
            self.assertEqual(split.call_args.args[3], expected)

    def test_state_queries_align_sparse_rows_labels_and_theorem_local_exprs(self) -> None:
        # Failures: dropping a tactic misaligns states/IDs/features; class IDs
        # depend on selected theorems; tables cross theorem boundaries; streaming
        # prepares once per state; raw states stay in feature-only batches;
        # excluded theorems prepare graphs; queries densify or write the database.
        # Establishes public fixture-DB query equivalence and shared preparation,
        # not full-corpus throughput, memory bounds, or classifier quality.
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source.db"
            first = theorem("First", (EXACT, UNKNOWN, RW))
            second = msgspec.structs.replace(
                theorem("Second", (RW, UNKNOWN, EXACT)), exprs=(r.Const("different.fn", ()), *first.exprs[1:])
            )
            dropped = theorem("Dropped", (UNKNOWN,))
            items = (first, second, dropped)
            store(src, items)
            original = src.read_bytes()
            policy = LabelPolicy({**POLICY.kinds, "Fixture.never_seen": "absent"}, "drop")
            observed = candidates.extract_candidates(first, (1, 2))
            layout = compile_vocabulary(Vocabulary((1, 2), tuple(Entry(s.ident, s.edges) for s in observed.shapes)))
            index = dict(layout.index)
            with open_corpus(src) as corpus:
                feature_only = list(corpus.features(layout, policy))
                self.assertTrue(all(batch.states is None for batch in feature_only))
                # A DB feature query must validate once per theorem, including
                # dropped actions, and prepare only the two retained graphs.
                with (
                    patch.object(candidates, "prepare_adjacency", wraps=candidates.prepare_adjacency) as prepare,
                    patch.object(r, "_validate_refs", wraps=r._validate_refs) as validate,
                ):
                    batches = list(
                        corpus.features(layout, policy, [second.name, dropped.name, first.name], include_states=True)
                    )
                    self.assertEqual(prepare.call_count, 2)
                    self.assertEqual(validate.call_count, 3)
                for batch, item in zip(batches, (second, dropped, first), strict=True):
                    self.assertIsNotNone(batch.states)
                    assert batch.states is not None
                    self.assertEqual(batch.states.exprs, item.exprs)
                    self.assertEqual(batch.classes, policy.labels)
                    self.assertEqual(tuple(batch.classes[idx] for idx in batch.label_ids), batch.labels)
                    self.assertNotIn(policy.label_ids["absent"], batch.label_ids)
                    steps = () if item is dropped else (0, 2)
                    np.testing.assert_array_equal(batch.rows.steps, steps)
                    self.assertEqual(batch.states.trns, tuple(item.trns[step] for step in steps))
                    self.assertIsInstance(batch.rows.matrix, csr_array)
                    if steps:
                        expected = encode_theorem(item, layout)
                        np.testing.assert_array_equal(
                            batch.rows.matrix.toarray(), expected.matrix[list(steps)].toarray()
                        )
                    else:
                        self.assertEqual(batch.rows.matrix.shape, (0, 2 * layout.block_width))
                with patch.object(candidates, "prepare_adjacency", wraps=candidates.prepare_adjacency) as prepare:
                    states = list(corpus.proof_states(layout, policy, [second.name, first.name, dropped.name]))
                    self.assertEqual(prepare.call_count, 2)
                self.assertEqual(
                    [(state.theorem, state.step) for state in states],
                    [("Second", 0), ("Second", 2), ("First", 0), ("First", 2)],
                )
                self.assertIs(states[0].exprs, states[1].exprs)
                self.assertIs(states[2].exprs, states[3].exprs)
                for state, batch in zip(states, (batches[0], batches[0], batches[2], batches[2]), strict=True):
                    assert batch.states is not None
                    idx = 0 if state.step == 0 else 1
                    self.assertEqual(state.trn, batch.states.trns[idx])
                    self.assertIs(state.state, state.trn.state)
                    self.assertEqual(state.exprs[state.state.target], batch.states.exprs[state.state.target])
                    self.assertEqual(state.label_id, policy.label_ids[state.label])
                    self.assertEqual(state.features.shape, (1, 2 * layout.block_width))
                    np.testing.assert_array_equal(state.features.toarray(), batch.rows.matrix[idx : idx + 1].toarray())
                single = corpus.proof_state("Second", 2, layout, policy)
                self.assertEqual(single.trn, states[1].trn)
                self.assertEqual(single.exprs, states[1].exprs)
                np.testing.assert_array_equal(single.features.toarray(), states[1].features.toarray())
                self.assertEqual(corpus.proof_state("First", 2, layout, policy).label_id, policy.label_ids["rewrite"])
                with patch("trustmebro.extraction.storage.decode_exprs", side_effect=AssertionError("expr decode")):
                    transitions = list(corpus.labeled_transitions(policy, [second.name, first.name]))
                    self.assertEqual(
                        [(trn.theorem, trn.step) for trn in transitions],
                        [(state.theorem, state.step) for state in states],
                    )
            pure = labeled_features(first, layout, policy, include_states=True)
            assert pure.states is not None
            self.assertIs(pure.states.exprs, first.exprs)
            self.assertEqual(layout.index, index)
            self.assertEqual(src.read_bytes(), original)

    def test_state_queries_reject_excluded_steps_missing_names_and_bad_selections(self) -> None:
        # Failures: negative/dropped indices silently select a different row;
        # absent names are skipped; malformed selections are accepted; unknown
        # tactic policy behavior or fallback class numbering changes in queries.
        # Exercises public DB query failures, not private helper implementations.
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source.db"
            item = theorem("Proof", (EXACT, UNKNOWN, RW))
            store(src, (item,))
            observed = candidates.extract_candidates(item, (1, 2))
            layout = compile_vocabulary(Vocabulary((1, 2), tuple(Entry(s.ident, s.edges) for s in observed.shapes)))
            with open_corpus(src) as corpus:
                for step in (-1, 3):
                    with self.assertRaises(IndexError):
                        corpus.proof_state(item.name, step, layout, POLICY)
                with self.assertRaisesRegex(ValueError, "excluded"):
                    corpus.proof_state(item.name, 1, layout, POLICY)
                with self.assertRaises(KeyError):
                    corpus.proof_state("missing", 0, layout, POLICY)
                for names, message in (([item.name, item.name], "duplicates"), (item.name, "single string")):
                    with self.assertRaisesRegex(ValueError, message):
                        list(corpus.proof_states(layout, POLICY, names))
                    with self.assertRaisesRegex(ValueError, message):
                        list(corpus.labeled_transitions(POLICY, names))
                with self.assertRaises(KeyError):
                    list(corpus.proof_states(layout, POLICY, ["missing"]))
                with self.assertRaises(KeyError):
                    list(corpus.labeled_transitions(POLICY, ["missing"]))
                self.assertEqual(list(corpus.proof_states(layout, POLICY, [])), [])
                self.assertEqual(list(corpus.labeled_transitions(POLICY, [])), [])
                strict = LabelPolicy(POLICY.kinds, "error")
                with self.assertRaisesRegex(ValueError, "unmapped tactic"):
                    corpus.proof_state(item.name, 1, layout, strict)
                fallback = LabelPolicy(POLICY.kinds, "other", "OTHER")
                state = corpus.proof_state(item.name, 1, layout, fallback)
                self.assertEqual(state.label, "OTHER")
                self.assertEqual(state.label_id, fallback.label_ids["OTHER"])

    def test_partition_cli_and_labeled_feature_api_preserve_records_and_row_identity(self) -> None:
        # Failures: transitions cross theorem partitions; repeated invocations
        # satisfy distinct-theorem minima; repair violates another label; blobs,
        # dictionaries/provenance/naturals/size summaries change; source is written; partitions
        # resume extraction; seeded assignments drift; dropped actions renumber steps; per-transition graph
        # preparation repeats or fits held-out vocabulary; empty batches prepare;
        # class IDs or feature columns drift across partition databases.
        # Establishes exact fixture counts and public workflow, not ML quality or
        # full-corpus solver performance. Shapes are fitted from a train theorem.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, out = root / "source.db", root / "split"
            items = population()
            store(src, items, dicts=True)
            original = src.read_bytes()
            policy_path = root / "labels.json"
            policy_path.write_bytes(msgspec.json.encode(POLICY))
            process = subprocess.run(
                [
                    "trustmebro-partition-corpus",
                    "--db",
                    str(src),
                    "--output",
                    str(out),
                    "--labels",
                    str(policy_path),
                    "--test-frac",
                    "0.25",
                    "--seed",
                    "0",
                    "--train-min-trns",
                    "2",
                    "--test-min-trns",
                    "2",
                    "--train-min-theorems",
                    "2",
                    "--test-min-theorems",
                    "2",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads(process.stdout)
            self.assertEqual(report["repair"], "minimum-move")
            self.assertGreater(report["dur_sec"], 0)
            self.assertEqual({path.name for path in out.iterdir()}, {"train.db", "test.db"})
            names: dict[str, set[str]] = {}
            raw_cols = "id,name,module,src_path,src_start,src_end,expr_count,trn_count,exprs,trns"
            with open_corpus(src) as source:
                originals = {row[1]: row for row in source.db.execute(f"SELECT {raw_cols} FROM theorems")}
                dicts = stored_dicts(source.db)
                self.assertIsNotNone(dicts)
                raw_sizes = {
                    col: sum(
                        len(source.codec.decompress(row[idx], getattr(source.codec, col))) for row in originals.values()
                    )
                    for col, idx in (("exprs", 8), ("trns", 9))
                }
                for col, raw_size in raw_sizes.items():
                    self.assertEqual(
                        sum(report["blob_sizes"][role][col]["uncompressed_bytes"] for role in ("train", "test")),
                        raw_size,
                    )
                with self.assertRaises(sqlite3.OperationalError):
                    source.db.execute("DELETE FROM theorems")
            for role in ("train", "test"):
                path = out / f"{role}.db"
                with open_corpus(path) as corpus:
                    names[role] = set(corpus.names())
                    self.assertEqual(stored_dicts(corpus.db), dicts)
                    self.assertEqual(corpus.db.execute("SELECT COUNT(*) FROM completed_files").fetchone(), (0,))
                    part_role, metadata = corpus.db.execute("SELECT role,report FROM partition_info").fetchone()
                    self.assertEqual(part_role, role)
                    saved = json.loads(metadata)
                    self.assertEqual(saved["blob_sizes"], report["blob_sizes"])
                    self.assertEqual(saved["policy"], json.loads(policy_path.read_bytes()))
                    self.assertEqual(saved["cfg"]["seed"], 0)
                    self.assertTrue(saved["solver_optimal"])
                    for label in POLICY.labels:
                        self.assertGreaterEqual(saved[role]["labels"][label]["trns"], 2)
                        self.assertGreaterEqual(saved[role]["labels"][label]["theorems"], 2)
                    for row in corpus.db.execute(f"SELECT {raw_cols} FROM theorems"):
                        self.assertEqual(row, originals[row[1]])
                    for item in corpus.theorems():
                        self.assertEqual(item, next(original for original in items if original.name == item.name))
                with self.assertRaisesRegex(r.SchemaError, "cannot be resumed"):
                    open_extraction_db(path)
            self.assertFalse(names["train"] & names["test"])
            self.assertEqual(names["train"] | names["test"], {item.name for item in items})
            partition.partition_corpus(
                src,
                root / "repeat",
                POLICY,
                partition.SplitCfg(0.25, 0, partition.MinSupport(2, 2), partition.MinSupport(2, 2)),
            )
            for role, expected_names in names.items():
                with open_corpus(root / "repeat" / f"{role}.db") as repeated:
                    self.assertEqual(set(repeated.names()), expected_names)

            with open_corpus(out / "train.db") as train:
                sample = next(item for item in train.theorems() if item.name != "Drop.only")
                observed = candidates.extract_candidates(sample, (1, 2))
                layout = compile_vocabulary(Vocabulary((1, 2), tuple(Entry(s.ident, s.edges) for s in observed.shapes)))
            for role in names:
                with open_corpus(out / f"{role}.db") as split:
                    for batch in split.features(layout, POLICY):
                        self.assertEqual(batch.classes, POLICY.labels)
                        self.assertEqual(batch.rows.matrix.shape[1], 2 * layout.block_width)
                        np.testing.assert_array_equal(
                            batch.label_ids, [POLICY.label_ids[label] for label in batch.labels]
                        )
            with open_corpus(src) as corpus:
                with patch(
                    "trustmebro.preprocessing.corpus.decode_theorem_row", side_effect=AssertionError("expr decode")
                ):
                    labeled = list(corpus.labeled_transitions(POLICY))
                self.assertEqual(len(labeled), 16)
                self.assertEqual({trn.step for trn in labeled}, {0, 2})
                self.assertEqual(len(list(corpus.labeled_transitions(LabelPolicy(POLICY.kinds, "other", "OTHER")))), 25)
                with patch.object(candidates, "prepare_adjacency", wraps=candidates.prepare_adjacency) as prepare:
                    batches = list(corpus.features(layout, POLICY))
                    self.assertEqual(prepare.call_count, 8)
                for batch, item in zip(batches, items, strict=True):
                    rows = batch.rows
                    if item.name == "Drop.only":
                        self.assertEqual(rows.matrix.shape, (0, 2 * layout.block_width))
                        self.assertEqual(batch.labels, ())
                    else:
                        expected = encode_theorem(item, layout)
                        np.testing.assert_array_equal(rows.matrix.toarray(), expected.matrix[[0, 2]].toarray())
                        np.testing.assert_array_equal(rows.steps, [0, 2])
                        self.assertEqual(rows.tactics, (item.trns[0].tactic, item.trns[2].tactic))
                        self.assertEqual(batch.labels, tuple(POLICY.label(tactic) for tactic in rows.tactics))
                self.assertEqual(list(corpus.theorems([items[3].name, items[0].name])), [items[3], items[0]])
                with self.assertRaises(KeyError):
                    corpus.theorem("missing")
                with self.assertRaisesRegex(ValueError, "duplicates"):
                    list(corpus.theorems([items[0].name] * 2))
            self.assertEqual(src.read_bytes(), original)

    def test_unsatisfied_labels_corruption_and_output_collisions_do_not_publish(self) -> None:
        # Failures: impossible global/combined quotas accepted; unmapped actions
        # silently dropped; duplicate policy keys overwrite mappings; corrupt
        # counts copied; missing input created; an existing artifact overwritten.
        # Exercises public boundaries; infeasibility is distinguished from timeout.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, out = root / "source.db", root / "split"
            cases = (
                ((theorem("Only.rare", (RW,) * 8), theorem("Only.common", (EXACT,))), POLICY, "insufficient"),
                (
                    (theorem("AB", ("a", "b")), theorem("BC", ("b", "c")), theorem("AC", ("a", "c"))),
                    LabelPolicy({"a": "A", "b": "B", "c": "C"}, "error"),
                    "infeasible",
                ),
                (population(), LabelPolicy(POLICY.kinds, "error"), "unmapped"),
            )
            for idx, (items, policy, message) in enumerate(cases):
                current = root / f"case-{idx}.db"
                store(current, items)
                original = current.read_bytes()
                with self.assertRaisesRegex(ValueError, message):
                    partition.partition_corpus(current, out, policy)
                self.assertFalse(out.exists())
                self.assertEqual(current.read_bytes(), original)
            store(src, population())
            with closing(sqlite3.connect(src)) as db:
                db.execute("UPDATE theorems SET trn_count=99 WHERE id=1")
                db.commit()
            original = src.read_bytes()
            with self.assertRaisesRegex(r.SchemaError, "count disagrees"):
                partition.partition_corpus(src, out, POLICY)
            self.assertFalse(out.exists())
            self.assertEqual(src.read_bytes(), original)
            out.mkdir()
            marker = out / "keep.txt"
            marker.write_text("existing user artifact")
            with self.assertRaises(FileExistsError):
                partition.partition_corpus(src, out, POLICY)
            self.assertEqual(marker.read_text(), "existing user artifact")
            missing = root / "missing.db"
            with self.assertRaises(sqlite3.OperationalError), open_corpus(missing):
                self.fail("missing source was created")
            self.assertFalse(missing.exists())
            duplicate = root / "bad.json"
            duplicate.write_text('{"kinds":{"same":"A","same":"B"},"unmapped":"drop"}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_label_policy(duplicate)
            duplicate.write_text('{"kinds":{"k":"A"},"unmapped":"other"}')
            with self.assertRaises((ValueError, msgspec.ValidationError)):
                read_label_policy(duplicate)

    def test_solver_boundary_and_interrupted_copy_do_not_create_misleading_partitions(self) -> None:
        # Failures: timeout called infeasible; valid incumbent rejected solely
        # for timeout; fractional/invalid incumbent accepted; first database
        # published before the second succeeds. Fault injection is necessary:
        # real solver timing/disk exhaustion cannot reliably exercise these paths.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "source.db"
            store(src, population())
            cfg = partition.SplitCfg(
                test_frac=0.25, train_min=partition.MinSupport(2, 2), test_min=partition.MinSupport(2, 2)
            )
            real_solver = partition.milp

            def timed_incumbent(*args, **kwargs):
                result = real_solver(*args, **kwargs)
                result.status = 1
                return result

            with patch.object(partition, "milp", side_effect=timed_incumbent):
                result = partition.partition_corpus(src, root / "valid-incumbent", POLICY, cfg)
            self.assertFalse(result.solver_optimal)
            self.assertTrue(all(support.theorems >= 2 for support in result.test.labels.values()))
            for idx, x in enumerate((None, np.full(9, 0.5), np.zeros(9))):
                out = root / f"failed-{idx}"
                answer = SimpleNamespace(x=x, status=1, message="time limit")
                with patch.object(partition, "milp", return_value=answer), self.assertRaises(ValueError) as caught:
                    partition.partition_corpus(src, out, POLICY, cfg)
                self.assertNotIn("infeasible", str(caught.exception))
                self.assertFalse(out.exists())
            real_open = partition.open_extraction_db

            def fail_second(path, dicts=None):
                if path.name == "test.db":
                    raise sqlite3.OperationalError("simulated disk failure")
                return real_open(path, dicts)

            with (
                patch.object(partition, "open_extraction_db", side_effect=fail_second),
                self.assertRaises(sqlite3.OperationalError),
            ):
                partition.partition_corpus(src, root / "failed-copy", POLICY, cfg)
            self.assertFalse((root / "failed-copy").exists())
            self.assertFalse(list(root.glob(".partition-*")))
