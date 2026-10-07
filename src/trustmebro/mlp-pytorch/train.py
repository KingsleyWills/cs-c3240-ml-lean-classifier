"""Archive-to-MLP adapter using the documented cached-feature and split APIs."""

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path
from time import perf_counter
from typing import Literal

from mlp import (
    DEFAULT_FIT_CFG,
    Data,
    Epoch,
    FeatureSpace,
    FitCfg,
    FitResult,
    ModelCfg,
    checkpoint_epoch,
    fit,
    save_checkpoint,
)

from trustmebro.preprocessing.corpus import DEFAULT_MAX_BYTES, TrainingCfg, load_training_matrix
from trustmebro.preprocessing.partition import SplitManifest, read_split


@dataclass(frozen=True, slots=True)
class TrainingData:
    train: Data
    validation: Data


def load_archive(
    path: Path,
    *,
    vocab_id: str,
    split: SplitManifest | None = None,
    subset: Literal["train", "validation"] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Data:
    """Load cached float32 CSR vectors once; never rebuild or fit graph features.

    vocab_id is the caller's assertion of the frozen representation identity:
    the public matrix loader verifies the split but does not expose that hash.
    max_bytes bounds this partition's numeric buffers, not total process memory.
    """
    if not vocab_id:
        raise ValueError("a frozen vocabulary identity is required")
    batch = load_training_matrix(
        path, cfg=TrainingCfg(dtype="float32"), max_bytes=max_bytes, split=split, subset=subset
    )
    space = FeatureSpace(batch.matrix.shape[1], batch.classes, vocab_id)
    return Data(batch.matrix, batch.label_ids, batch.theorems, space)


def load_training_data(
    features: Path, split: Path, *, vocab_id: str, max_bytes: int = DEFAULT_MAX_BYTES
) -> TrainingData:
    """Select both logical partitions from one completed, immutable archive."""
    manifest = read_split(split)
    stamp = features.stat().st_size, features.stat().st_mtime_ns
    train = load_archive(features, vocab_id=vocab_id, split=manifest, subset="train", max_bytes=max_bytes)
    val = load_archive(features, vocab_id=vocab_id, split=manifest, subset="validation", max_bytes=max_bytes)
    if stamp != (features.stat().st_size, features.stat().st_mtime_ns):
        raise ValueError("feature archive changed while loading training and validation")
    return TrainingData(train, val)


def fit_archive(
    features: Path,
    split: Path,
    *,
    vocab_id: str,
    cfg: FitCfg = DEFAULT_FIT_CFG,
    max_bytes: int = DEFAULT_MAX_BYTES,
    on_epoch: Callable[[Epoch], None] | None = None,
) -> FitResult:
    data = load_training_data(features, split, vocab_id=vocab_id, max_bytes=max_bytes)
    return fit(data.train, data.validation, cfg, on_epoch=on_epoch)


def _report(epoch: Epoch) -> None:
    val = epoch.validation
    weighted = "" if val.weighted_cross_entropy is None else f"weighted val CE={val.weighted_cross_entropy:.5f}; "
    train_name = "train CE" if val.weighted_cross_entropy is None else "weighted train CE"
    print(
        f"Epoch {epoch.epoch}: {train_name}={epoch.train_cross_entropy:.5f}; val CE={val.cross_entropy:.5f}; "
        f"{weighted}accuracy={val.accuracy:.4f}; top-3={val.top3_accuracy:.4f}; top-5={val.top5_accuracy:.4f}; "
        f"macro-F1={val.macro_f1:.4f}; {epoch.duration_s:.2f}s",
        file=sys.stderr,
        flush=True,
    )


def weighted_checkpoint(path: Path) -> Path:
    return path.with_name(f"{path.stem}.weighted{path.suffix}")


def checkpoint_summary(result: FitResult, criterion: Literal["unweighted", "weighted"] = "unweighted") -> dict:
    """Metrics at that criterion's selected epoch, not unrelated maxima over the run."""
    epoch = checkpoint_epoch(result, criterion)
    best = result.history[epoch - 1].validation
    return {
        "criterion": criterion,
        "best_epoch": epoch,
        "val_cross_entropy": best.cross_entropy,
        "val_weighted_cross_entropy": best.weighted_cross_entropy,
        "val_accuracy": best.accuracy,
        "val_top3_accuracy": best.top3_accuracy,
        "val_top5_accuracy": best.top5_accuracy,
        "val_macro_f1": best.macro_f1,
        "precision": best.precision.tolist(),
        "recall": best.recall.tolist(),
        "f1": best.f1.tolist(),
        "support": best.support.tolist(),
        "confusion": best.confusion.tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the sparse-input MLP on cached train/validation features")
    parser.add_argument("--features", type=Path, required=True, help="completed features.zst archive")
    parser.add_argument("--split", type=Path, required=True, help="matching split.json manifest")
    parser.add_argument(
        "--vocab", type=Path, required=True, help="matching frozen vocabulary, hashed for checkpoint identity"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replace", action="store_true", help="allow replacing the checkpoint, never input files")
    parser.add_argument("--device", default=DEFAULT_FIT_CFG.device)
    parser.add_argument("--hidden", type=int, nargs="+", default=DEFAULT_FIT_CFG.model.hidden)
    parser.add_argument("--dropout", type=float, default=DEFAULT_FIT_CFG.model.dropout)
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_FIT_CFG.batch_rows)
    parser.add_argument("--epochs", type=int, default=DEFAULT_FIT_CFG.max_epochs)
    parser.add_argument("--patience", type=int, default=DEFAULT_FIT_CFG.patience)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_FIT_CFG.min_delta)
    parser.add_argument("--lr", type=float, default=DEFAULT_FIT_CFG.lr)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_FIT_CFG.weight_decay)
    parser.add_argument("--seed", type=int, default=DEFAULT_FIT_CFG.seed)
    parser.add_argument("--scaling", choices=("maxabs", "none"), default=DEFAULT_FIT_CFG.scaling)
    parser.add_argument(
        "--class-weight", choices=("none", "balanced", "inverse_sqrt"), default=DEFAULT_FIT_CFG.class_weight
    )
    parser.add_argument("--label-smoothing", type=float, default=DEFAULT_FIT_CFG.label_smoothing)
    parser.add_argument(
        "--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="numeric-buffer limit per partition, not total RSS"
    )
    args = parser.parse_args(argv)
    inputs = (args.features, args.split, args.vocab)
    checkpoints = (
        (args.checkpoint,) if args.class_weight == "none" else (args.checkpoint, weighted_checkpoint(args.checkpoint))
    )
    if any(path.resolve() in {src.resolve() for src in inputs} for path in checkpoints):
        parser.error("checkpoint must not overwrite an input artifact")
    if any(path.exists() for path in checkpoints) and not args.replace:
        parser.error("checkpoint exists; choose another path or supply --replace")
    started = perf_counter()
    with args.vocab.open("rb") as stream:
        vocab_id = file_digest(stream, "sha256").hexdigest()
    cfg = FitCfg(
        model=ModelCfg(tuple(args.hidden), args.dropout),
        batch_rows=args.batch_rows,
        max_epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        scaling=args.scaling,
        class_weight=args.class_weight,
        label_smoothing=args.label_smoothing,
    )
    print("Loading cached training and validation vectors...", file=sys.stderr, flush=True)
    result = fit_archive(
        args.features, args.split, vocab_id=vocab_id, cfg=cfg, max_bytes=args.max_bytes, on_epoch=_report
    )
    save_checkpoint(args.checkpoint, result, replace=args.replace)
    weighted = None
    if result.weighted_best is not None:
        path = weighted_checkpoint(args.checkpoint)
        save_checkpoint(path, result, replace=args.replace, criterion="weighted")
        weighted = checkpoint_summary(result, "weighted") | {"checkpoint": str(path)}
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "vocab_id": vocab_id,
                "width": result.predictor.space.width,
                "classes": result.predictor.space.classes,
                "train": result.train_split,
                "validation": result.val_split,
                "epochs": len(result.history),
                **checkpoint_summary(result),
                "class_weight": args.class_weight,
                "label_smoothing": args.label_smoothing,
                "class_weights": None
                if result.predictor.class_weights is None
                else result.predictor.class_weights.tolist(),
                "weighted_checkpoint": weighted,
                "duration_s": perf_counter() - started,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
