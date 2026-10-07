"""Ordered MLP experiments over one immutable feature instance; no grid or test-set selection."""

import argparse
import csv
import gc
import json
import os
import platform
import sys
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import file_digest
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Literal, TextIO

import msgspec
import numpy as np
import scipy
import sklearn
import torch
from mlp import (
    DEFAULT_FIT_CFG,
    FitCfg,
    FitResult,
    ModelCfg,
    PreparedData,
    fit_prepared,
    prepare_training,
    save_checkpoint,
    validate_cfg,
)
from train import _report, checkpoint_summary, load_training_data, weighted_checkpoint

from trustmebro.preprocessing.corpus import DEFAULT_MAX_BYTES


class Settings(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    hidden: tuple[int, ...] | msgspec.UnsetType = msgspec.UNSET
    dropout: float | msgspec.UnsetType = msgspec.UNSET
    batch_rows: int | msgspec.UnsetType = msgspec.UNSET
    epochs: int | msgspec.UnsetType = msgspec.UNSET
    patience: int | msgspec.UnsetType = msgspec.UNSET
    min_delta: float | msgspec.UnsetType = msgspec.UNSET
    lr: float | msgspec.UnsetType = msgspec.UNSET
    weight_decay: float | msgspec.UnsetType = msgspec.UNSET
    seed: int | msgspec.UnsetType = msgspec.UNSET
    device: str | msgspec.UnsetType = msgspec.UNSET
    scaling: Literal["maxabs", "none"] | msgspec.UnsetType = msgspec.UNSET
    class_weight: Literal["none", "balanced", "inverse_sqrt"] | msgspec.UnsetType = msgspec.UNSET
    label_smoothing: float | msgspec.UnsetType = msgspec.UNSET


class Run(Settings):
    name: str


class Plan(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    instance: str
    output: str
    runs: tuple[Run, ...]
    defaults: Settings = msgspec.field(default_factory=Settings)
    max_bytes: int = DEFAULT_MAX_BYTES


@dataclass(frozen=True, slots=True)
class Experiment:
    name: str
    cfg: FitCfg
    dir: Path


SUMMARY_FIELDS = (
    "name",
    "status",
    "hidden",
    "dropout",
    "lr",
    "weight_decay",
    "batch_rows",
    "scaling",
    "class_weight",
    "label_smoothing",
    "seed",
    "epochs",
    "best_epoch",
    "val_cross_entropy",
    "val_accuracy",
    "val_top3_accuracy",
    "val_top5_accuracy",
    "val_weighted_cross_entropy",
    "val_macro_f1",
    "weighted_best_epoch",
    "weighted_val_cross_entropy",
    "weighted_val_weighted_cross_entropy",
    "weighted_val_accuracy",
    "weighted_val_top3_accuracy",
    "weighted_val_top5_accuracy",
    "weighted_val_macro_f1",
    "fit_duration_s",
    "duration_s",
    "error",
)


def _cfg(defaults: Settings, run: Run) -> FitCfg:
    resolved = asdict(DEFAULT_FIT_CFG)
    resolved.update(resolved.pop("model"))
    resolved["epochs"] = resolved.pop("max_epochs")
    for settings in (defaults, run):
        resolved.update({key: val for key, val in msgspec.structs.asdict(settings).items() if val is not msgspec.UNSET})
    resolved.pop("name")
    model = ModelCfg(tuple(resolved.pop("hidden")), resolved.pop("dropout"))
    resolved["max_epochs"] = resolved.pop("epochs")
    cfg = FitCfg(model=model, **resolved)
    # Reject the whole plan before starting any fit, not halfway through a queue.
    validate_cfg(cfg)
    return cfg


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _write(path: Path, emit: Callable[[TextIO], None]) -> None:
    """Publish each report atomically; completed runs survive later failures/interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            emit(stream)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _json(path: Path, data: dict) -> None:
    _write(path, lambda stream: json.dump(data, stream, indent=2, allow_nan=False))


def _summary(output: Path, runs: tuple[Experiment, ...]) -> None:
    def emit(stream: TextIO) -> None:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for run in runs:
            path = run.dir / "results.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            cfg = asdict(run.cfg)
            cfg.update(cfg.pop("model"))
            cfg["hidden"] = "x".join(map(str, cfg["hidden"]))
            row = cfg | data | {"name": run.name}
            row.update({f"weighted_{key}": val for key, val in (data.get("weighted_checkpoint") or {}).items()})
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})

    _write(output / "summary.csv", emit)


def _results(result: FitResult, duration_s: float, checkpoint: Path) -> dict:
    weighted = None
    if result.weighted_best is not None:
        path = weighted_checkpoint(checkpoint)
        weighted = checkpoint_summary(result, "weighted") | {"checkpoint_sha256": _hash(path)}
    return {
        "status": "complete",
        "checkpoint_sha256": _hash(checkpoint),
        "vocab_id": result.predictor.space.vocab_id,
        "width": result.predictor.space.width,
        "classes": result.predictor.space.classes,
        "train": result.train_split,
        "validation": result.val_split,
        "epochs": len(result.history),
        **checkpoint_summary(result),
        "class_weights": None if result.predictor.class_weights is None else result.predictor.class_weights.tolist(),
        "weighted_checkpoint": weighted,
        "history": [
            {
                "epoch": epoch.epoch,
                "train_cross_entropy": epoch.train_cross_entropy,
                "val_cross_entropy": epoch.validation.cross_entropy,
                "val_weighted_cross_entropy": epoch.validation.weighted_cross_entropy,
                "val_accuracy": epoch.validation.accuracy,
                "val_top3_accuracy": epoch.validation.top3_accuracy,
                "val_top5_accuracy": epoch.validation.top5_accuracy,
                "val_macro_f1": epoch.validation.macro_f1,
                "duration_s": epoch.duration_s,
            }
            for epoch in result.history
        ],
        "duration_s": duration_s,
    }


def run_plan(path: Path) -> None:
    """Single-writer output; paths in plans are relative to the invoking working directory."""
    plan = msgspec.json.decode(path.read_bytes(), type=Plan)
    if not plan.runs or plan.max_bytes < 1:
        raise ValueError("at least one run and a positive max_bytes limit are required")
    names = [run.name for run in plan.runs]
    if len(set(names)) != len(names) or any(
        not name
        or name in (".", "..")
        or any(not (char.isascii() and (char.isalnum() or char in "_-")) for char in name)
        for name in names
    ):
        raise ValueError("run names must be unique and contain only ASCII letters, digits, underscores or hyphens")
    instance, output = Path(plan.instance).resolve(), Path(plan.output).resolve()
    inputs = {name: instance / name for name in ("features.zst", "split.json", "vocab.zst")}
    runs = tuple(Experiment(run.name, _cfg(plan.defaults, run), output / run.name) for run in plan.runs)
    # Prevent report publication over inputs, even when the user chooses an overlapping output tree.
    destinations = {output / "summary.csv"}
    for run in runs:
        destinations.update(run.dir / name for name in ("config.json", "results.json", "checkpoint.pt"))
        if run.cfg.class_weight != "none":
            destinations.add(weighted_checkpoint(run.dir / "checkpoint.pt"))
    protected = (*inputs.values(), path.resolve())
    if any(
        dst.resolve() == src.resolve() or dst.resolve() in src.resolve().parents
        for dst in destinations
        for src in protected
    ):
        raise ValueError("experiment outputs must not overwrite or enclose input artifacts")
    print("Fingerprinting immutable inputs...", file=sys.stderr, flush=True)
    stamps = {name: (src.stat().st_size, src.stat().st_mtime_ns) for name, src in inputs.items()}
    identity = {
        "inputs": {name: {"path": str(src), "sha256": _hash(src)} for name, src in inputs.items()},
        "code": {name: _hash(Path(__file__).with_name(name)) for name in ("experiments.py", "mlp.py", "train.py")},
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    pending: list[Experiment] = []
    configs: dict[str, dict] = {}
    for run in runs:
        config = json.loads(
            json.dumps({"name": run.name, "cfg": asdict(run.cfg), "max_bytes": plan.max_bytes, "identity": identity})
        )
        configs[run.name] = config
        config_path, result_path = run.dir / "config.json", run.dir / "results.json"
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError(f"{run.name}: settings or input/code identity changed; choose a new output directory")
        elif run.dir.exists() and any(run.dir.iterdir()):
            raise ValueError(f"{run.name}: nonempty run directory has no matching config.json")
        saved = json.loads(result_path.read_text()) if result_path.exists() else {}
        if saved.get("status") == "complete":
            checkpoint = run.dir / "checkpoint.pt"
            hashes = [(checkpoint, saved["checkpoint_sha256"])]
            if run.cfg.class_weight != "none":
                hashes.append((weighted_checkpoint(checkpoint), saved["weighted_checkpoint"]["checkpoint_sha256"]))
            if any(not path.is_file() or _hash(path) != ident for path, ident in hashes):
                raise ValueError(f"{run.name}: completed checkpoint is missing or changed")
            print(f"Skip completed run: {run.name}", file=sys.stderr, flush=True)
        else:
            pending.append(run)
    _summary(output, runs)
    if not pending:
        return
    print("Loading cached training and validation vectors once...", file=sys.stderr, flush=True)
    data = load_training_data(
        inputs["features.zst"],
        inputs["split.json"],
        vocab_id=identity["inputs"]["vocab.zst"]["sha256"],
        max_bytes=plan.max_bytes,
    )
    if stamps != {name: (src.stat().st_size, src.stat().st_mtime_ns) for name, src in inputs.items()}:
        raise ValueError("input artifacts changed during fingerprinting/loading")
    prepared: dict[str, PreparedData] = {}
    for idx, run in enumerate(pending, 1):
        print(f"Run {idx}/{len(pending)}: {run.name}", file=sys.stderr, flush=True)
        _json(run.dir / "config.json", configs[run.name])
        _json(run.dir / "results.json", {"status": "running"})
        started = perf_counter()
        try:
            if run.cfg.scaling not in prepared:
                prepared[run.cfg.scaling] = prepare_training(data.train, data.validation, scaling=run.cfg.scaling)
            fit_started = perf_counter()
            result = fit_prepared(prepared[run.cfg.scaling], run.cfg, on_epoch=_report)
            fit_duration = perf_counter() - fit_started
            checkpoint = run.dir / "checkpoint.pt"
            save_checkpoint(checkpoint, result, replace=True)
            if result.weighted_best is not None:
                save_checkpoint(weighted_checkpoint(checkpoint), result, replace=True, criterion="weighted")
            report = _results(result, perf_counter() - started, checkpoint)
            report.update(fit_duration_s=fit_duration, duration_s=perf_counter() - started)
            del result  # Do not retain a previous full-sized network while allocating the next one.
            _json(run.dir / "results.json", report)
        except (Exception, KeyboardInterrupt) as error:
            _json(
                run.dir / "results.json",
                {
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "duration_s": perf_counter() - started,
                },
            )
            _summary(output, runs)
            raise
        _summary(output, runs)
        gc.collect()  # Once per fit, never in the minibatch loop.
        print(
            f"Finished {run.name}: val CE={report['val_cross_entropy']:.5f}; "
            f"accuracy={report['val_accuracy']:.4f}; macro-F1={report['val_macro_f1']:.4f}; "
            f"top-3={report['val_top3_accuracy']:.4f}; top-5={report['val_top5_accuracy']:.4f}; "
            f"{report['duration_s']:.2f}s",
            file=sys.stderr,
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an explicit ordered list of MLP configurations")
    parser.add_argument(
        "--config", type=Path, required=True, help="JSON plan; paths are relative to the working directory"
    )
    args = parser.parse_args(argv)
    run_plan(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
