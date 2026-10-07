"""Synthetic model checks and cached-feature archive-to-checkpoint integration."""

import csv
import importlib.util
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from hashlib import file_digest, sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import msgspec
import numpy as np
import torch
from scipy.sparse import csr_array, hstack
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from torch import nn

from trustmebro.extraction import records as r
from trustmebro.extraction.storage import encode_msgpack
from trustmebro.preprocessing import archives, candidates, features, partition
from trustmebro.preprocessing.corpus import LabelPolicy

MODULE_PATH = Path(__file__).parents[1] / "src/trustmebro/mlp-pytorch/mlp.py"
SPEC = importlib.util.spec_from_file_location("trustmebro_mlp_pytorch", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mlp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mlp
SPEC.loader.exec_module(mlp)
TRAIN_PATH = MODULE_PATH.with_name("train.py")
TRAIN_SPEC = importlib.util.spec_from_file_location("trustmebro_mlp_train", TRAIN_PATH)
assert TRAIN_SPEC is not None and TRAIN_SPEC.loader is not None
adapter = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = adapter
with patch.dict(sys.modules, {"mlp": mlp}):
    TRAIN_SPEC.loader.exec_module(adapter)
EXPERIMENT_SPEC = importlib.util.spec_from_file_location(
    "trustmebro_mlp_experiments", MODULE_PATH.with_name("experiments.py")
)
assert EXPERIMENT_SPEC is not None and EXPERIMENT_SPEC.loader is not None
experiments = importlib.util.module_from_spec(EXPERIMENT_SPEC)
sys.modules[EXPERIMENT_SPEC.name] = experiments
with patch.dict(sys.modules, {"mlp": mlp, "train": adapter}):
    EXPERIMENT_SPEC.loader.exec_module(experiments)


def fixture(prefix: str, repeats: int = 8, *, classes: int = 3):
    labels = np.tile(np.arange(classes, dtype=np.int64), repeats)
    # Scramble rows to expose independent shuffling of labels and features.
    labels = labels[np.random.default_rng(7).permutation(len(labels))]
    matrix = np.zeros((len(labels), 8), dtype=np.float64)
    matrix[np.arange(len(labels)), labels] = 2
    matrix[:, 6] = 1
    space = mlp.FeatureSpace(8, tuple(f"class-{idx}" for idx in range(classes)), "fixture-vocab")
    return mlp.Data(csr_array(matrix), labels, tuple(f"{prefix}-{idx // 2}" for idx in range(len(labels))), space)


def archive_fixture(root: Path):
    train, val = fixture("train", repeats=4, classes=2), fixture("val", repeats=2, classes=2)
    policy = LabelPolicy({f"kind-{idx}": f"class-{idx}" for idx in range(3)}, "error")
    train_names, val_names = tuple(sorted(set(train.theorems))), tuple(sorted(set(val.theorems)))
    selection = candidates.Selection("absent-source.db", 1, 2, (1,), None, train_names + val_names, 0)

    def stats(data):
        support = {
            label: partition.LabelSupport(
                int((data.label_ids == idx).sum()),
                len({name for name, ident in zip(data.theorems, data.label_ids) if ident == idx}),
            )
            for idx, label in enumerate(policy.labels)
        }
        return partition.SubsetStats(len(set(data.theorems)), len(data.label_ids), len(data.label_ids), 0, support)

    split = partition.SplitManifest(
        selection, policy, partition.SplitCfg(), 1, 1, train_names, val_names, stats(train), stats(val), "fixture", None
    )
    fitted = msgspec.structs.replace(selection, theorems=train_names)
    edges = ((),)
    representation = features.Representation(
        features.AttributePolicy(hyp_slots=1, name_dims=0), (), (), fitted, "fixture", 0, 0, msgspec.json.encode(policy)
    )
    vocab = features.Vocabulary(
        (1,),
        (features.Entry(sha256(msgspec.msgpack.encode(edges)).digest(), edges),),
        selection=fitted,
        representation=representation,
    )
    width = features.compile_vocabulary(vocab).width
    train = replace(train, matrix=hstack((train.matrix, csr_array((len(train.label_ids), width - 8))), format="csr"))
    val = replace(val, matrix=hstack((val.matrix, csr_array((len(val.label_ids), width - 8))), format="csr"))
    rows = []
    for data in (train, val):
        for name in sorted(set(data.theorems)):
            keep = np.flatnonzero(np.asarray(data.theorems) == name)
            tactics = tuple(r.Tactic(f"kind-{ident}", "fixture") for ident in data.label_ids[keep])
            rows.append(features.FeatureRows(name, np.arange(len(keep)), tactics, data.matrix[keep]))
    header = archives.FeatureHeader(
        vocab, selection, "absent-source.db", 1, 2, msgspec.json.encode(policy), partition.split_id(split)
    )
    archive, manifest, vocab_path = root / "features.zst", root / "split.json", root / "vocab.zst"
    archives.publish(
        archive,
        (
            encode_msgpack(header),
            *(archives.pack_rows(row) for row in rows),
            encode_msgpack(archives.Footer(len(rows), 12)),
        ),
        sources=(),
        replace=False,
    )
    archives.publish(vocab_path, (encode_msgpack(vocab),), sources=(), replace=False)
    manifest.write_bytes(msgspec.json.encode(split))
    return archive, manifest, vocab_path, train, val


class SparseMLPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_ordered_experiments_reuse_preparation_and_resume(self):
        # Failures: overrides reset defaults, execution order changes, matrices/scaling are
        # rebuilt per run, source values change, reports disagree with saved best weights,
        # or resume silently skips changed settings/inputs or a missing checkpoint.
        # Establishes the public plan-to-artifact CPU workflow, not full-width GPU throughput.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, split, vocab, _, _ = archive_fixture(root)
            originals = {path: path.read_bytes() for path in (archive, split, vocab)}
            plan = {
                "instance": str(root),
                "output": str(root / "experiments"),
                "defaults": {
                    "device": "cpu",
                    "hidden": [8],
                    "epochs": 2,
                    "patience": 2,
                    "batch_rows": 3,
                    "seed": 9,
                    "label_smoothing": 0.05,
                },
                "runs": [
                    {"name": "baseline", "label_smoothing": 0},
                    {"name": "dropout", "dropout": 0.25, "hidden": [4], "class_weight": "inverse_sqrt"},
                ],
            }
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            with (
                patch.object(experiments, "load_training_data", wraps=adapter.load_training_data) as loading,
                patch.object(experiments, "prepare_training", wraps=mlp.prepare_training) as preparation,
                patch.object(experiments, "fit_prepared", wraps=mlp.fit_prepared) as fitting,
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(experiments.main(["--config", str(path)]), 0)
            self.assertEqual(loading.call_count, 1)
            self.assertEqual(preparation.call_count, 1)
            self.assertEqual([call.args[1].model.hidden for call in fitting.call_args_list], [(8,), (4,)])
            self.assertEqual([call.args[1].model.dropout for call in fitting.call_args_list], [0, 0.25])
            self.assertEqual([call.args[1].label_smoothing for call in fitting.call_args_list], [0, 0.05])
            self.assertTrue(all(call.args[1].seed == 9 for call in fitting.call_args_list))
            self.assertIs(fitting.call_args_list[0].args[0], fitting.call_args_list[1].args[0])
            output = root / "experiments"
            with (output / "summary.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["name"] for row in rows], ["baseline", "dropout"])
            for row in rows:
                run = output / row["name"]
                resolved = json.loads((run / "config.json").read_text())
                report = json.loads((run / "results.json").read_text())
                self.assertEqual(report["status"], "complete")
                self.assertEqual(resolved["cfg"]["seed"], 9)
                self.assertEqual(int(row["best_epoch"]), report["best_epoch"])
                self.assertEqual(float(row["val_cross_entropy"]), report["val_cross_entropy"])
                self.assertEqual(float(row["label_smoothing"]), resolved["cfg"]["label_smoothing"])
                saved = torch.load(run / "checkpoint.pt", weights_only=True)
                self.assertEqual(saved["best_epoch"], report["best_epoch"])
                self.assertEqual(
                    saved["history"][report["best_epoch"] - 1]["val_cross_entropy"], report["val_cross_entropy"]
                )
                self.assertEqual(sum(report["support"]), 4)
                self.assertEqual(len(report["precision"]), 3)
                self.assertEqual(report["vocab_id"], resolved["identity"]["inputs"]["vocab.zst"]["sha256"])
                self.assertGreaterEqual(report["duration_s"], report["fit_duration_s"])
                self.assertGreaterEqual(report["val_top5_accuracy"], report["val_top3_accuracy"])
                if resolved["cfg"]["class_weight"] != "none":
                    weighted = report["weighted_checkpoint"]
                    saved_weighted = torch.load(run / "checkpoint.weighted.pt", weights_only=True)
                    self.assertEqual(saved_weighted["criterion"], "weighted")
                    self.assertEqual(saved_weighted["best_epoch"], weighted["best_epoch"])
                    self.assertEqual(int(row["weighted_best_epoch"]), weighted["best_epoch"])
                    self.assertEqual(float(row["val_top3_accuracy"]), report["val_top3_accuracy"])
                    self.assertEqual(float(row["weighted_val_top5_accuracy"]), weighted["val_top5_accuracy"])
                    np.testing.assert_array_equal(saved_weighted["class_weights"].numpy(), [1, 1, 0])
                else:
                    self.assertIsNone(report["weighted_checkpoint"])
                    self.assertIsNone(report["val_weighted_cross_entropy"])
            saved_outputs = {file: file.read_bytes() for file in output.glob("*/*")}
            with (
                patch.object(experiments, "load_training_data", side_effect=AssertionError("resume reloaded data")),
                patch.object(experiments, "fit_prepared", side_effect=AssertionError("resume retrained")),
                redirect_stderr(io.StringIO()),
            ):
                experiments.main(["--config", str(path)])
                plan["defaults"]["lr"] = 0.02
                path.write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    experiments.main(["--config", str(path)])
                del plan["defaults"]["lr"]
                plan["defaults"]["label_smoothing"] = 0.1
                path.write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    experiments.main(["--config", str(path)])
                plan["defaults"]["label_smoothing"] = 0.05
                path.write_text(json.dumps(plan))
                weighted_path = output / "dropout/checkpoint.weighted.pt"
                weighted_path.write_bytes(saved_outputs[weighted_path] + b"changed")
                with self.assertRaisesRegex(ValueError, "checkpoint is missing or changed"):
                    experiments.main(["--config", str(path)])
                weighted_path.write_bytes(saved_outputs[weighted_path])
                archive.write_bytes(originals[archive] + b"changed")
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    experiments.main(["--config", str(path)])
                archive.write_bytes(originals[archive])
                checkpoint = output / "dropout/checkpoint.pt"
                checkpoint.write_bytes(saved_outputs[checkpoint] + b"changed")
                with self.assertRaisesRegex(ValueError, "checkpoint is missing or changed"):
                    experiments.main(["--config", str(path)])
                checkpoint.write_bytes(saved_outputs[checkpoint])
                (output / "dropout/checkpoint.pt").unlink()
                with self.assertRaisesRegex(ValueError, "checkpoint is missing or changed"):
                    experiments.main(["--config", str(path)])
            for file, data in saved_outputs.items():
                if file.exists():
                    self.assertEqual(file.read_bytes(), data)
            for file, data in originals.items():
                self.assertEqual(file.read_bytes(), data)

    def test_experiment_failure_stops_queue_and_retry_preserves_completed_runs(self):
        # Failures: an error starts later fits, discards earlier results, is marked successful,
        # or resumption repeats the successful fit rather than retrying the failed/pending runs.
        # A failure injected at the fit boundary tests durable orchestration; successful fits
        # and checkpoint/report publication still use the real tiny CPU model.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_fixture(root)
            output = root / "experiments"
            path = root / "plan.json"
            path.write_text(
                json.dumps(
                    {
                        "instance": str(root),
                        "output": str(output),
                        "defaults": {"device": "cpu", "hidden": [4], "epochs": 1, "batch_rows": 4},
                        "runs": [{"name": "first"}, {"name": "second", "scaling": "none"}, {"name": "third"}],
                    }
                )
            )
            calls = 0

            def fail_second(data, cfg, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("deliberate fit failure")
                return mlp.fit_prepared(data, cfg, **kwargs)

            with (
                patch.object(experiments, "fit_prepared", side_effect=fail_second),
                redirect_stderr(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "deliberate fit failure"),
            ):
                experiments.main(["--config", str(path)])
            self.assertEqual(calls, 2)
            first = (output / "first/checkpoint.pt").read_bytes()
            failure = json.loads((output / "second/results.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertIn("deliberate fit failure", failure["error"])
            self.assertFalse((output / "third").exists())
            with (output / "summary.csv").open(newline="") as stream:
                self.assertEqual([row["status"] for row in csv.DictReader(stream)], ["complete", "failed"])
            with (
                patch.object(experiments, "fit_prepared", wraps=mlp.fit_prepared) as fitting,
                patch.object(experiments, "prepare_training", wraps=mlp.prepare_training) as preparing,
                redirect_stderr(io.StringIO()),
            ):
                experiments.main(["--config", str(path)])
            self.assertEqual(fitting.call_count, 2)
            self.assertEqual(preparing.call_count, 2)  # maxabs and none each prepared once.
            self.assertEqual((output / "first/checkpoint.pt").read_bytes(), first)
            self.assertTrue(
                all(
                    json.loads((output / name / "results.json").read_text())["status"] == "complete"
                    for name in ("first", "second", "third")
                )
            )
            # The same durable stop boundary must handle Ctrl-C without calling it success.
            plan = json.loads(path.read_text())
            plan["runs"].append({"name": "interrupted"})
            path.write_text(json.dumps(plan))
            with (
                patch.object(experiments, "fit_prepared", side_effect=KeyboardInterrupt),
                redirect_stderr(io.StringIO()),
                self.assertRaises(KeyboardInterrupt),
            ):
                experiments.main(["--config", str(path)])
            self.assertEqual(json.loads((output / "interrupted/results.json").read_text())["status"], "interrupted")
            self.assertEqual((output / "first/checkpoint.pt").read_bytes(), first)

    def test_experiment_plan_errors_fail_before_any_fit(self):
        # Failures: unknown/ill-typed settings are silently ignored, unsafe/duplicate names
        # escape the output tree, or a late invalid run is discovered after expensive fits.
        # Tests the public preflight boundary, not training quality or concurrent writers.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_fixture(root)
            path = root / "plan.json"
            base = {"instance": str(root), "output": str(root / "experiments"), "defaults": {"device": "cpu"}}
            cases = (
                [{"name": "valid"}, {"name": "typo", "dropuot": 0.3}],
                [{"name": "same"}, {"name": "same"}],
                [{"name": "../escape"}],
                [{"name": "bad", "hidden": [True]}],
                [{"name": "bad", "epochs": 0}],
                [{"name": "bad", "dropout": 1}],
                [{"name": "bad", "scaling": "wrong"}],
                [{"name": "bad", "class_weight": "wrong"}],
                [{"name": "bad", "label_smoothing": -0.1}],
                [{"name": "bad", "label_smoothing": 1.1}],
                [{"name": "bad", "label_smoothing": True}],
                [{"name": "bad", "device": "meta"}],
                [],
            )
            for runs in cases:
                with self.subTest(runs=runs), self.assertRaises((ValueError, msgspec.ValidationError)):
                    path.write_text(json.dumps(base | {"runs": runs}))
                    experiments.main(["--config", str(path)])
            self.assertFalse((root / "experiments").exists())

    def test_command_inherits_configuration_defaults_and_preserves_other_fields(self):
        # Failures: parser literals shadow FitCfg edits, nested model defaults diverge,
        # or overriding one option resets another. Successful training alone cannot
        # diagnose these settings; this narrow check inspects the real CLI-to-fit call.
        # The mocked fitting boundary establishes configuration propagation, not training.
        cfg = mlp.FitCfg(
            model=mlp.ModelCfg((5, 3), 0.25),
            batch_rows=4,
            max_epochs=2,
            patience=7,
            min_delta=0.05,
            lr=0.002,
            weight_decay=0.03,
            seed=17,
            device="cpu",
            scaling="none",
            class_weight="balanced",
            label_smoothing=0.15,
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, split, vocab, _, _ = archive_fixture(root)
            command = [
                "--features",
                str(archive),
                "--split",
                str(split),
                "--vocab",
                str(vocab),
                "--checkpoint",
                str(root / "model.pt"),
            ]
            cases = (
                ([], cfg),
                (
                    ["--patience", "2", "--lr", "0.02", "--hidden", "7", "--dropout", "0.4"],
                    replace(cfg, patience=2, lr=0.02, model=mlp.ModelCfg((7,), 0.4)),
                ),
                (["--class-weight", "none"], replace(cfg, class_weight="none")),
                (
                    ["--class-weight", "inverse_sqrt", "--label-smoothing", "0.05"],
                    replace(cfg, class_weight="inverse_sqrt", label_smoothing=0.05),
                ),
            )
            for flags, expected in cases:
                with (
                    self.subTest(flags=flags),
                    patch.object(adapter, "DEFAULT_FIT_CFG", cfg),
                    patch.object(adapter, "fit_archive", side_effect=RuntimeError("stop before training")) as fitting,
                    redirect_stderr(io.StringIO()),
                    self.assertRaisesRegex(RuntimeError, "stop before training"),
                ):
                    adapter.main(command + flags)
                self.assertEqual(fitting.call_args.kwargs["cfg"], expected)
                self.assertEqual(fitting.call_args.kwargs["max_bytes"], adapter.DEFAULT_MAX_BYTES)

    def test_training_evaluation_and_checkpoint(self):
        # Failures: densified inputs, misaligned shuffled labels, validation-fitted scaling,
        # mutated source values, dropped tail rows, batch-averaged loss, reordered checkpoint
        # classes or overwritten artifacts. Establishes the public CPU workflow, not GPU speed.
        train, val = fixture("train"), fixture("val", repeats=3)
        val.matrix.data *= 5
        before = train.matrix.data.copy()
        rng_before = torch.random.get_rng_state().clone()
        cfg = mlp.FitCfg(model=mlp.ModelCfg((16, 8)), device="cpu", max_epochs=40, patience=10, lr=0.03, batch_rows=7)
        with patch.object(csr_array, "toarray", side_effect=AssertionError("must not densify features")):
            result = mlp.fit(train, val, cfg)
            metrics = mlp.evaluate(result.predictor, val, batch_rows=4)
            probs = mlp.predict_proba(result.predictor, val.matrix, batch_rows=5)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        np.testing.assert_array_equal(before, train.matrix.data)
        np.testing.assert_array_equal(result.predictor.scale, [2, 2, 2, 1, 1, 1, 1, 1])
        self.assertLess(result.history[-1].train_cross_entropy, result.history[0].train_cross_entropy)
        self.assertGreaterEqual(metrics.accuracy, 0.95)
        self.assertEqual(metrics.rows, len(val.label_ids))
        pred = probs.argmax(axis=1)
        self.assertAlmostEqual(metrics.cross_entropy, log_loss(val.label_ids, probs), places=5)
        self.assertAlmostEqual(metrics.accuracy, accuracy_score(val.label_ids, pred))
        self.assertAlmostEqual(metrics.macro_f1, f1_score(val.label_ids, pred, average="macro"))
        np.testing.assert_array_equal(metrics.confusion, confusion_matrix(val.label_ids, pred))
        self.assertAlmostEqual(
            metrics.cross_entropy, min(epoch.validation.cross_entropy for epoch in result.history), places=6
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            mlp.save_checkpoint(path, result)
            restored = mlp.load_checkpoint(path, expected_space=train.space)
            np.testing.assert_array_equal(probs, mlp.predict_proba(restored, val.matrix, batch_rows=5))
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved["best_epoch"], result.best_epoch)
            self.assertEqual(saved["train_split"]["theorems"], len(set(train.theorems)))
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                mlp.save_checkpoint(path, result)
            self.assertEqual(original, path.read_bytes())
            self.assertEqual(list(Path(tmp).iterdir()), [path])
            wrong_space = replace(train.space, classes=tuple(reversed(train.space.classes)))
            with self.assertRaisesRegex(ValueError, "identity differs"):
                mlp.load_checkpoint(path, expected_space=wrong_space)

    def test_top_accuracies_include_tail_rows_and_break_ties_consistently(self):
        # Failures: top-k counts are averaged per batch rather than per row, rank ties exclude
        # the top-1 prediction, or k greater than class count fails. This public evaluation
        # fixture has analytically known predictions, not a claim about model accuracy.
        for classes in (3, 4, 7):
            with self.subTest(classes=classes):
                data = fixture("evaluation", repeats=2, classes=classes)
                model = mlp.SparseMLP(data.space.width, classes, mlp.ModelCfg((4,)))
                with torch.no_grad():
                    for param in model.parameters():
                        param.zero_()
                predictor = mlp.Predictor(model, None, data.space)
                metrics = mlp.evaluate(predictor, data, batch_rows=5)
                self.assertAlmostEqual(metrics.accuracy, 1 / classes)
                self.assertAlmostEqual(metrics.top3_accuracy, min(3, classes) / classes)
                self.assertAlmostEqual(metrics.top5_accuracy, min(5, classes) / classes)
                self.assertEqual(metrics.rows, 2 * classes)
                self.assertIsNone(metrics.weighted_cross_entropy)

    def test_balanced_loss_uses_training_support_and_global_weight_mass(self):
        # Failures: weights come from validation labels, inverse_sqrt uses the wrong exponent
        # or normalization, absent classes cause infinities, smoothing is ignored or applied
        # to validation, batch means are averaged without weight mass, or checkpoints lose
        # the settings/weights and either criterion. Almost-zero learning
        # leaves the logits fixed, allowing a full-epoch loss oracle despite shuffled batches.
        train = fixture("train", repeats=8, classes=2)
        keep = np.concatenate((np.flatnonzero(train.label_ids == 0), np.flatnonzero(train.label_ids == 1)[:2]))
        space = replace(train.space, classes=("class-0", "class-1", "class-2"))
        train = replace(
            train,
            matrix=train.matrix[keep],
            label_ids=train.label_ids[keep],
            theorems=tuple(train.theorems[idx] for idx in keep),
            space=space,
        )
        val = fixture("validation", repeats=3)
        cfg = mlp.FitCfg(
            model=mlp.ModelCfg((4,)),
            device="cpu",
            class_weight="balanced",
            max_epochs=3,
            patience=2,
            batch_rows=3,
            lr=1e-30,
        )
        for mode in ("none", "balanced", "inverse_sqrt"):
            for smoothing in (0, 0.05):
                with self.subTest(mode=mode, smoothing=smoothing):
                    result = mlp.fit(train, val, replace(cfg, class_weight=mode, label_smoothing=smoothing))
                    weights = result.predictor.class_weights
                    expected_weights = {
                        "none": [1, 1, 1],
                        "balanced": [0.625, 2.5, 0],
                        "inverse_sqrt": [5 / 6, 5 / 3, 0],
                    }[mode]
                    if mode == "none":
                        self.assertIsNone(weights)
                    else:
                        np.testing.assert_allclose(weights, expected_weights)
                        self.assertAlmostEqual(np.dot([8, 2, 0], weights), 10)
                    weights = np.asarray(expected_weights)
                    train_probs = mlp.predict_proba(result.predictor, train.matrix)
                    targets = np.full(train_probs.shape, smoothing / len(space.classes))
                    targets[np.arange(len(train.label_ids)), train.label_ids] += 1 - smoothing
                    # Built-in integer-target smoothing weights each component, but divides
                    # the mean by the hard-target weight mass. Independent NumPy oracle.
                    losses = (targets * weights * -np.log(train_probs.astype(np.float64))).sum(axis=1)
                    expected_train = losses.sum() / weights[train.label_ids].sum()
                    self.assertTrue(
                        all(abs(epoch.train_cross_entropy - expected_train) < 1e-6 for epoch in result.history)
                    )
                    val_probs = mlp.predict_proba(result.predictor, val.matrix)
                    row_ce = -np.log(val_probs[np.arange(len(val.label_ids)), val.label_ids].astype(np.float64))
                    expected_val = np.average(row_ce, weights=weights[val.label_ids])
                    for batch_rows in (2, 5, len(val.label_ids)):
                        metrics = mlp.evaluate(result.predictor, val, batch_rows=batch_rows)
                        self.assertAlmostEqual(metrics.cross_entropy, log_loss(val.label_ids, val_probs), places=5)
                        if mode == "none":
                            self.assertIsNone(metrics.weighted_cross_entropy)
                        else:
                            self.assertAlmostEqual(metrics.weighted_cross_entropy, expected_val, places=6)
                    if mode != "none":
                        with self.assertRaisesRegex(ValueError, "no positive training-derived class weight"):
                            mlp.evaluate(result.predictor, replace(val, label_ids=np.full(len(val.label_ids), 2)))
                        self.assertIsNotNone(result.weighted_best)
                    with TemporaryDirectory() as tmp:
                        criteria = ("unweighted",) if mode == "none" else ("unweighted", "weighted")
                        for criterion in criteria:
                            path = Path(tmp) / f"{criterion}.pt"
                            mlp.save_checkpoint(path, result, criterion=criterion)
                            saved = torch.load(path, weights_only=True)
                            self.assertEqual(saved["cfg"]["class_weight"], mode)
                            self.assertEqual(saved["cfg"]["label_smoothing"], smoothing)
                            self.assertEqual(saved["criterion"], criterion)
                            self.assertEqual(saved["best_epoch"], mlp.checkpoint_epoch(result, criterion))
                            restored = mlp.load_checkpoint(path)
                            if mode == "none":
                                self.assertIsNone(restored.class_weights)
                            else:
                                np.testing.assert_allclose(restored.class_weights, weights)
                            actual = mlp.evaluate(restored, val, batch_rows=2)
                            expected = result.history[saved["best_epoch"] - 1].validation
                            self.assertAlmostEqual(actual.cross_entropy, expected.cross_entropy, places=6)
                            if mode != "none":
                                self.assertAlmostEqual(
                                    actual.weighted_cross_entropy, expected.weighted_cross_entropy, places=6
                                )

    def test_dual_patience_resets_independently_and_keeps_distinct_checkpoints(self):
        # Failures: stopping when only one criterion is stale, never resetting an exhausted
        # counter, sharing mutable snapshots across epochs, or using min_delta to suppress
        # best-checkpoint updates. Contrasting trajectories are injected at evaluation because
        # reliably inducing these exact patience cases with real learning would be fragile.
        # Training, snapshot ownership, publication and restored predictions remain real.
        train, val = fixture("train"), fixture("validation")
        cfg = mlp.FitCfg(
            model=mlp.ModelCfg((4,)),
            device="cpu",
            class_weight="balanced",
            max_epochs=12,
            patience=2,
            batch_rows=7,
            lr=0.03,
        )
        cases = (
            ([(1, 5), (0.8, 4), (0.9, 3), (1.1, 2), (1.2, 2.1), (1.3, 2.2)], 2, 4, 0),
            ([(1, 5), (0.8, 4), (0.9, 3), (1.1, 2), (0.7, 2.1), (0.8, 2.2), (0.9, 2.3)], 5, 4, 0),
            ([(1, 5), (0.8, 4), (0.7, 3)], 3, 3, 1e9),
            ([(20 - idx, 30 - idx) for idx in range(12)], 12, 12, 0),
        )
        original_evaluate = mlp._evaluate
        for trajectory, unweighted_epoch, weighted_epoch, min_delta in cases:
            with self.subTest(trajectory=trajectory):
                pairs = iter(trajectory)
                snapshots = []

                def observed(predictor, data, batch_rows, weights, *, snapshots=snapshots, pairs=pairs):
                    snapshots.append(mlp.predict_proba(predictor, data.matrix).copy())
                    metrics = original_evaluate(predictor, data, batch_rows, weights)
                    ce, weighted_ce = next(pairs)
                    return replace(metrics, cross_entropy=ce, weighted_cross_entropy=weighted_ce)

                with patch.object(mlp, "_evaluate", side_effect=observed):
                    result = mlp.fit(train, val, replace(cfg, min_delta=min_delta))
                self.assertEqual(len(result.history), len(trajectory))
                self.assertEqual(result.best_epoch, unweighted_epoch)
                self.assertEqual(result.weighted_best.epoch, weighted_epoch)
                np.testing.assert_allclose(
                    mlp.predict_proba(result.predictor, val.matrix), snapshots[unweighted_epoch - 1]
                )
                with TemporaryDirectory() as tmp:
                    for criterion, epoch in (("unweighted", unweighted_epoch), ("weighted", weighted_epoch)):
                        path = Path(tmp) / f"{criterion}.pt"
                        mlp.save_checkpoint(path, result, criterion=criterion)
                        restored = mlp.load_checkpoint(path)
                        np.testing.assert_allclose(mlp.predict_proba(restored, val.matrix), snapshots[epoch - 1])
                if unweighted_epoch == weighted_epoch:
                    for name, tensor in result.predictor.model.state_dict().items():
                        torch.testing.assert_close(tensor.cpu(), result.weighted_best.state[name])

    def test_early_stopping_restores_owned_best_snapshot(self):
        # Failures: storing aliased state_dict tensors, returning final rather than best weights,
        # min_delta suppressing best-model updates, or off-by-one patience. This deliberately
        # trains against the validation labels to make later checkpoints worse.
        train, val = fixture("train", classes=2), fixture("val", classes=2)
        val = replace(val, label_ids=1 - val.label_ids)
        cfg = mlp.FitCfg(model=mlp.ModelCfg((8,)), device="cpu", batch_rows=16, max_epochs=12, patience=2, lr=0.1)
        result = mlp.fit(train, val, cfg)
        self.assertLess(len(result.history), cfg.max_epochs)
        self.assertLess(result.best_epoch, len(result.history))
        expected = min(epoch.validation.cross_entropy for epoch in result.history)
        self.assertAlmostEqual(mlp.evaluate(result.predictor, val).cross_entropy, expected, places=6)
        unchanged = mlp.fit(train, val, replace(cfg, lr=1e-30))
        self.assertEqual(len(unchanged.history), 3)
        self.assertEqual(unchanged.best_epoch, 1)
        improving = mlp.fit(train, fixture("val", classes=2), replace(cfg, min_delta=1e9))
        self.assertEqual(len(improving.history), 3)
        self.assertGreater(improving.best_epoch, 1)

    def test_rejects_invalid_data_and_split_identity(self):
        # Failures: index corruption reaching unchecked Torch CSR construction, accepting NaNs,
        # label range/length/type mismatches, or theorem leakage. These public-boundary failures
        # cannot be inferred from a successful separable training fixture.
        train, val = fixture("train"), fixture("val")
        cfg = mlp.FitCfg(device="cpu", max_epochs=1)
        invalid = [
            replace(train, label_ids=train.label_ids.astype(float)),
            replace(train, label_ids=train.label_ids[:-1]),
            replace(train, label_ids=np.full(len(train.label_ids), 3)),
            replace(train, label_ids=np.full(len(train.label_ids), -1)),
            replace(train, theorems=train.theorems[:-1]),
            replace(train, matrix=csr_array(np.ones((len(train.label_ids), 7)))),
            replace(train, matrix=csr_array(np.full(train.matrix.shape, np.nan))),
        ]
        duplicate = csr_array(([1.0, 2.0], [0, 0], [0, 2]), shape=(1, 8))
        invalid.append(replace(train, matrix=duplicate, label_ids=np.array([0]), theorems=("duplicate",)))
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                mlp.fit(data, val, cfg)
        with self.assertRaisesRegex(ValueError, "overlapping theorems"):
            mlp.fit(train, train, cfg)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            mlp.fit(train, replace(val, space=replace(val.space, vocab_id="different")), cfg)
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "CUDA was requested"),
        ):
            mlp.fit(train, val, replace(cfg, device="cuda"))

    def test_sparse_outputs_and_gradients_match_dense_reference(self):
        # Failure: sparse first-layer operand layout or backward differs from a dense linear
        # reference. Narrow numerical comparison diagnoses a boundary E2E accuracy could miss;
        # fixture-only dense input is not the production execution path.
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                torch.manual_seed(4)
                model = mlp.SparseMLP(8, 3, mlp.ModelCfg((5,))).to(device)
                dense = torch.tensor(
                    [[0, 2, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 3, 0, 0]], dtype=torch.float32, device=device
                )
                sparse = dense.to_sparse_csr()
                labels = torch.tensor([0, 2], device=device)
                logits = model(sparse)
                nn.functional.cross_entropy(logits, labels).backward()
                grads = {name: param.grad.clone() for name, param in model.named_parameters()}
                model.zero_grad(set_to_none=True)
                reference = model.tail(dense @ model.weight + model.bias)
                nn.functional.cross_entropy(reference, labels).backward()
                torch.testing.assert_close(logits, reference)
                for name, param in model.named_parameters():
                    torch.testing.assert_close(grads[name], param.grad)
                torch.optim.AdamW(model.parameters(), foreach=False).step()

    def test_cuda_training_and_checkpoint(self):
        # Failures: CPU-only batching, label/device mismatch, sparse CUDA optimizer integration,
        # or inability to restore CUDA-trained weights on CPU. This tiny E2E fixture does not
        # establish full-width GPU memory consumption or throughput.
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable in this process")
        train, val = fixture("train"), fixture("val", repeats=2)
        cfg = mlp.FitCfg(model=mlp.ModelCfg((8,)), device="cuda", max_epochs=8, lr=0.05, batch_rows=7)
        result = mlp.fit(train, val, cfg)
        self.assertEqual(result.predictor.model.weight.device.type, "cuda")
        self.assertLess(result.history[-1].train_cross_entropy, result.history[0].train_cross_entropy)
        probs = mlp.predict_proba(result.predictor, val.matrix)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cuda.pt"
            mlp.save_checkpoint(path, result)
            restored = mlp.load_checkpoint(path, device="cpu", expected_space=train.space)
            np.testing.assert_allclose(probs, mlp.predict_proba(restored, val.matrix), rtol=1e-5, atol=1e-6)

    def test_declared_absent_classes_and_scaling_disabled(self):
        # Failures: dropping absent declared classes from macro-F1, wrong confusion orientation,
        # treating scaling='none' as maxabs, or failing to include rows in partial minibatches.
        train, val = fixture("train"), fixture("val", repeats=2)
        space = replace(train.space, classes=(*train.space.classes, "absent"))
        train, val = replace(train, space=space), replace(val, space=space)
        cfg = mlp.FitCfg(device="cpu", max_epochs=2, batch_rows=5, scaling="none")
        result = mlp.fit(train, val, cfg)
        self.assertIsNone(result.predictor.scale)
        metrics = mlp.evaluate(result.predictor, val, batch_rows=4)
        probs = mlp.predict_proba(result.predictor, val.matrix)
        self.assertEqual(metrics.confusion.shape, (4, 4))
        self.assertEqual(metrics.support[-1], 0)
        self.assertEqual(metrics.recall[-1], 0)
        self.assertAlmostEqual(
            metrics.macro_f1,
            f1_score(val.label_ids, probs.argmax(axis=1), labels=np.arange(4), average="macro", zero_division=0),
        )
        np.testing.assert_array_equal(
            metrics.confusion, confusion_matrix(val.label_ids, probs.argmax(axis=1), labels=np.arange(4))
        )

    def test_cached_archive_to_training_and_command(self):
        # Failures: reconstructing features, swapping logical roles/labels, losing frozen absent
        # classes, converting to the wrong precision, or rereading per epoch. The
        # real CLI must preserve input artifacts and publish a usable inference checkpoint.
        # Also verifies that weighted CLI runs publish both criteria and top-k metrics.
        # Establishes tiny archive integration, not full-corpus loader memory or GPU speed.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, split, vocab_path, expected_train, expected_val = archive_fixture(root)
            originals = {path: path.read_bytes() for path in (archive, split, vocab_path)}
            with vocab_path.open("rb") as stream:
                vocab_id = file_digest(stream, "sha256").hexdigest()
            cfg = mlp.FitCfg(device="cpu", model=mlp.ModelCfg((8,)), max_epochs=3, batch_rows=3)
            with (
                patch.object(features, "encode_theorem", side_effect=AssertionError("feature reconstruction")),
                patch.object(features, "encode_candidates", side_effect=AssertionError("feature reconstruction")),
                patch.object(candidates, "extract_candidates", side_effect=AssertionError("graph reconstruction")),
                patch.object(adapter, "load_training_matrix", wraps=adapter.load_training_matrix) as loader,
            ):
                data = adapter.load_training_data(archive, split, vocab_id=vocab_id)
                self.assertEqual(loader.call_count, 2)
                for actual, expected in ((data.train, expected_train), (data.validation, expected_val)):
                    self.assertEqual(actual.matrix.dtype, np.float32)
                    np.testing.assert_array_equal(actual.matrix.toarray(), expected.matrix.toarray())
                    np.testing.assert_array_equal(actual.label_ids, expected.label_ids)
                    self.assertEqual(actual.theorems, expected.theorems)
                    self.assertEqual(actual.space.classes, ("class-0", "class-1", "class-2"))
                result = adapter.fit_archive(archive, split, vocab_id=vocab_id, cfg=cfg)
                self.assertEqual(loader.call_count, 4)
                self.assertEqual(result.predictor.space.vocab_id, vocab_id)
                self.assertEqual(result.history[-1].validation.rows, len(expected_val.label_ids))
            checkpoint = root / "model.pt"
            run = subprocess.run(
                [
                    sys.executable,
                    str(TRAIN_PATH),
                    "--features",
                    str(archive),
                    "--split",
                    str(split),
                    "--vocab",
                    str(vocab_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--device",
                    "cpu",
                    "--epochs",
                    "2",
                    "--hidden",
                    "8",
                    "--batch-rows",
                    "3",
                    "--class-weight",
                    "inverse_sqrt",
                    "--label-smoothing",
                    "0.05",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            report = json.loads(run.stdout)
            self.assertEqual(report["train"]["rows"], len(expected_train.label_ids))
            self.assertEqual(report["validation"]["rows"], len(expected_val.label_ids))
            self.assertEqual(report["vocab_id"], vocab_id)
            self.assertIn("Epoch 1:", run.stderr)
            self.assertIn("top-3=", run.stderr)
            self.assertIn("weighted val CE=", run.stderr)
            self.assertEqual(report["class_weight"], "inverse_sqrt")
            self.assertEqual(report["label_smoothing"], 0.05)
            self.assertEqual(report["val_top5_accuracy"], 1)
            weighted_path = adapter.weighted_checkpoint(checkpoint)
            self.assertEqual(report["weighted_checkpoint"]["checkpoint"], str(weighted_path))
            saved_weighted = torch.load(weighted_path, weights_only=True)
            self.assertEqual(saved_weighted["best_epoch"], report["weighted_checkpoint"]["best_epoch"])
            self.assertEqual(saved_weighted["criterion"], "weighted")
            restored = mlp.load_checkpoint(checkpoint, expected_space=result.predictor.space)
            self.assertEqual(mlp.evaluate(restored, data.validation).rows, len(expected_val.label_ids))
            for path, before in originals.items():
                self.assertEqual(path.read_bytes(), before)

    def test_archive_binding_budget_and_input_protection(self):
        # Failures: accepting a manifest from another experiment, silently bypassing numeric
        # buffer limits, or overwriting a feature archive via the checkpoint --replace option.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, split_path, vocab_path, _, _ = archive_fixture(root)
            split = partition.read_split(split_path)
            bad_split = root / "bad-split.json"
            bad_split.write_bytes(msgspec.json.encode(msgspec.structs.replace(split, instance=2)))
            with self.assertRaisesRegex(ValueError, "split manifest differs"):
                adapter.load_training_data(archive, bad_split, vocab_id="fixture")
            with self.assertRaises(MemoryError):
                adapter.load_training_data(archive, split_path, vocab_id="fixture", max_bytes=1)
            original = archive.read_bytes()
            stderr = io.StringIO()
            with (
                patch.object(adapter, "fit_archive", side_effect=AssertionError("must reject before training")),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as failure,
            ):
                adapter.main(
                    [
                        "--features",
                        str(archive),
                        "--split",
                        str(split_path),
                        "--vocab",
                        str(vocab_path),
                        "--checkpoint",
                        str(archive),
                        "--replace",
                    ]
                )
            self.assertEqual(failure.exception.code, 2)
            self.assertIn("must not overwrite an input artifact", stderr.getvalue())
            self.assertEqual(archive.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
