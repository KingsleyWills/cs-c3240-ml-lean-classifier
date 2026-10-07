"""Shared discovery, split-specific fitting, and reusable development vectors.

Candidates are label-neutral observations of the development pool. Every fitted
coverage/support/association/name product belongs to one training population.
Completed stages are reused only with identical inputs and output stamps; an
interrupted stage may replace its own unpublished/partial products on resume.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import MISSING, dataclass, fields
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic

import msgspec

from trustmebro.extraction.scheduler import Phase

from .archives import read_selection
from .corpus import read_label_policy
from .features import SUPERVISED_SCREENING
from .partition import DEFAULT_VALIDATION_CFG, MinSupport, SplitCfg, logical_splits, read_split, split_id
from .scheduler import VocabularyBuildCfg, check_vocabulary_cfg, convert_corpus, fit_vocabulary, scan_candidates


@dataclass(frozen=True, slots=True)
class PreparationCfg:
    vocabulary: VocabularyBuildCfg
    split: SplitCfg = DEFAULT_VALIDATION_CFG
    folds: int = 1
    pair_limit: int = 0


def _stamp(path: Path) -> dict[str, str | int]:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "modified_ns": stat.st_mtime_ns}


def _write_json(path: Path, data: object) -> None:
    """Publish small metadata atomically; compressed bulk products use archives."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".preparation-", delete=False) as stream:
        pending = Path(stream.name)
        try:
            stream.write(msgspec.json.encode(data))
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    try:
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def _stage(
    record: Path, outputs: tuple[Path, ...], sources: tuple[Path, ...], fn: Callable[[], object], title: str
) -> dict:
    inputs = [_stamp(path) for path in sources]
    if record.exists():
        previous = json.loads(record.read_text())
        if previous["inputs"] != inputs or any(not path.is_file() for path in outputs):
            raise ValueError(f"{title}: completed stage inputs or outputs changed; use a new output directory")
        if previous["outputs"] != [_stamp(path) for path in outputs]:
            raise ValueError(f"{title}: completed artifacts changed; use a new output directory")
        print(f"{title}: reuse completed stage", file=sys.stderr)
        return {**previous, "reused": True}
    started = monotonic()
    print(f"{title}...", file=sys.stderr)
    # Nested computation owns progress; do not compete with its live line.
    result = fn()
    if inputs != [_stamp(path) for path in sources]:
        raise ValueError(f"{title}: source changed during stage")
    report = {
        "inputs": inputs,
        "outputs": [_stamp(path) for path in outputs],
        "duration_sec": monotonic() - started,
        "result": msgspec.to_builtins(result),
    }
    _write_json(record, report)
    print(f"{title}: finished in {report['duration_sec']:.2f}s", file=sys.stderr)
    return {**report, "reused": False}


def prepare_features(
    db: Path, labels: Path, output: Path, cfg: PreparationCfg, *, candidates: Path | None = None
) -> dict:
    """One holdout or K fitted instances, without copying the source database.

    Reuse requires unchanged run configuration and sources. Choose another output
    directory for another experiment, supplying the same candidates to avoid
    discovery. Existing unrelated files are never adopted or overwritten.
    """
    if cfg.folds < 1 or cfg.vocabulary.workers < 1 or not 0 <= cfg.pair_limit <= 2048:
        raise ValueError("folds/workers must be positive and co-occurrence prefix must lie in 0..2048")
    check_vocabulary_cfg(cfg.vocabulary)
    policy = read_label_policy(labels)
    request = msgspec.json.decode(
        msgspec.json.encode(
            {
                "db": _stamp(db),
                "labels": _stamp(labels),
                "cfg": cfg,
                "shape_selection": SUPERVISED_SCREENING,
                "candidates": None if candidates is None else _stamp(candidates),
            }
        )
    )
    run = output / "run.json"
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"output is not a directory: {output}")
    if run.exists():
        if json.loads(run.read_text()) != request:
            raise ValueError("preparation inputs/settings changed; use a new output directory")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("output contains unrelated artifacts; choose an empty/new directory")
    else:
        _write_json(run, request)
    started = monotonic()
    reports: dict[str, object] = {}
    summaries: list[dict[str, object]] = []
    if candidates is None:
        candidates = output / "candidates.zst"
        reports["candidates"] = _stage(
            output / "candidates.done.json",
            (candidates,),
            (db,),
            lambda: scan_candidates(
                db, candidates, depths=cfg.vocabulary.depths, workers=cfg.vocabulary.workers, replace=True
            ),
            "Extract shared candidates",
        )
    selection = read_selection(candidates)
    if (
        Path(selection.db).resolve() != db.resolve()
        or (selection.size, selection.modified_ns) != (db.stat().st_size, db.stat().st_mtime_ns)
        or selection.limit is not None
        or selection.theorems is not None
        or not set(cfg.vocabulary.depths).issubset(selection.depths)
    ):
        raise ValueError("candidates must describe the complete, unchanged development database and requested depths")
    manifests = tuple(output / f"instance-{idx + 1:02d}" / "split.json" for idx in range(cfg.folds))

    def assign() -> dict[str, object]:
        with Phase("Choose logical train/validation roles") as progress:
            splits = logical_splits(db, policy, cfg.split, folds=cfg.folds)
            for path, split in zip(manifests, splits, strict=True):
                _write_json(path, split)
            progress.details = f"{len(splits)} instances; theorems remain grouped"
        return {"instances": len(splits)}

    reports["splits"] = _stage(output / "splits.done.json", manifests, (db, labels, candidates), assign, "Assign roles")
    for path in manifests:
        split = read_split(path)
        root = path.parent
        vocab, coverage = root / "vocab.zst", root / "coverage.zst"
        prefix = f"Instance {split.instance}/{cfg.folds}"
        instance_report = {
            "fitting": _stage(
                root / "vocabulary.done.json",
                (vocab, coverage),
                (candidates, labels, path),
                partial(
                    fit_vocabulary,
                    candidates,
                    labels,
                    vocab,
                    cfg.vocabulary,
                    coverage=coverage,
                    theorems=split.train,
                    replace=True,
                ),
                f"{prefix}: fit training vocabulary",
            ),
            "conversion": _stage(
                root / "features.done.json",
                (root / "features.zst", root / "feature-stats.zst"),
                (candidates, vocab, labels, path),
                partial(
                    convert_corpus,
                    candidates,
                    vocab,
                    root / "features.zst",
                    root / "feature-stats.zst",
                    candidates=True,
                    workers=cfg.vocabulary.workers,
                    labels=labels,
                    pair_limit=cfg.pair_limit,
                    split_id=split_id(split),
                    expected_theorems=split.train + split.validation,
                    replace=True,
                ),
                f"{prefix}: convert development pool",
            ),
        }
        reports[root.name] = instance_report
        conversion = instance_report["conversion"]["result"]
        summaries.append(
            {
                "instance": split.instance,
                "train": msgspec.to_builtins(split.train_stats),
                "validation": msgspec.to_builtins(split.validation_stats),
                "dimensions": conversion["dimensions"],
                "nnz": conversion["nnz"],
                "mean_active_per_observed_state": conversion["nnz"] / max(conversion["states"], 1),
            }
        )
    if _stamp(db) != request["db"] or _stamp(labels) != request["labels"]:
        raise ValueError("development database or label policy changed during preparation")
    report = {
        "instances": cfg.folds,
        "output": str(output.resolve()),
        "duration_sec": monotonic() - started,
        "instance_summaries": summaries,
        "stages": reports,
    }
    _write_json(output / "summary.json", report)
    return report


def _conversion_parser(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--db", type=Path)
    source.add_argument("--candidates", type=Path)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int)
    selection.add_argument("--theorems-from", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cooccurrence-shapes", type=int, default=0)
    parser.add_argument("--replace", action="store_true")


def main(argv: list[str] | None = None) -> int:
    # dims has no default; inspect the optional fields without inventing a budget.
    defaults = {field.name: field.default for field in fields(VocabularyBuildCfg) if field.default is not MISSING}
    parser = argparse.ArgumentParser(description="Prepare split-specific vocabularies and cached learning features.")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="shared candidates -> logical splits -> per-instance fitting/conversion")
    build.add_argument("--db", type=Path, required=True, help="outer training/development database, never the test DB")
    build.add_argument("--labels", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True, help="new directory, or unchanged run to resume")
    build.add_argument("--candidates", type=Path, help="reuse complete development-pool candidates")
    build.add_argument("--dims", type=int, required=True, help="total feature dimension cap per instance")
    build.add_argument("--workers", type=int, default=defaults["workers"])
    build.add_argument("--depths", type=int, nargs="+", default=defaults["depths"])
    modes = build.add_mutually_exclusive_group()
    modes.add_argument("--folds", type=int, help="K disjoint grouped validation folds; K must be >= 2")
    modes.add_argument(
        "--validation-frac", type=float, help=f"holdout theorem fraction; default {DEFAULT_VALIDATION_CFG.test_frac}"
    )
    build.add_argument("--seed", type=int, default=1)
    build.add_argument("--train-min-trns", type=int, default=DEFAULT_VALIDATION_CFG.train_min.trns)
    build.add_argument("--train-min-theorems", type=int, default=DEFAULT_VALIDATION_CFG.train_min.theorems)
    build.add_argument("--validation-min-trns", type=int, default=DEFAULT_VALIDATION_CFG.test_min.trns)
    build.add_argument("--validation-min-theorems", type=int, default=DEFAULT_VALIDATION_CFG.test_min.theorems)
    build.add_argument("--solver-seconds", type=float, default=DEFAULT_VALIDATION_CFG.solver_seconds)
    build.add_argument("--hyp-slots", type=int, default=defaults["hyp_slots"])
    build.add_argument("--name-share", type=float, default=defaults["name_share"])
    build.add_argument("--min-support", type=int, default=defaults["min_support"])
    build.add_argument("--index-memory-mib", type=int, default=defaults["index_memory_mib"])
    build.add_argument("--score-memory-mib", type=int, default=defaults["score_memory_mib"])
    build.add_argument("--name-memory-mib", type=int, default=defaults["name_memory_mib"])
    build.add_argument("--cooccurrence-shapes", type=int, default=0)
    _conversion_parser(commands.add_parser("convert", help="apply a frozen vocabulary; never select columns"))
    cfg = parser.parse_args(argv)
    if cfg.command == "convert":
        names = (
            None
            if cfg.theorems_from is None
            else tuple(line.strip() for line in cfg.theorems_from.read_text().splitlines() if line.strip())
        )
        report = convert_corpus(
            cfg.candidates if cfg.candidates is not None else cfg.db,
            cfg.vocab,
            cfg.output,
            cfg.stats,
            candidates=cfg.candidates is not None,
            workers=cfg.workers,
            limit=cfg.limit,
            seed=cfg.seed,
            theorems=names,
            pair_limit=cfg.cooccurrence_shapes,
            labels=cfg.labels,
            replace=cfg.replace,
        )
    else:
        if cfg.folds is not None and cfg.folds < 2:
            parser.error("--folds must be at least two; omit it for a holdout")
        folds = cfg.folds or 1
        vocabulary = VocabularyBuildCfg(
            cfg.dims,
            tuple(sorted(set(cfg.depths))),
            cfg.workers,
            cfg.hyp_slots,
            cfg.name_share,
            cfg.min_support,
            cfg.index_memory_mib,
            cfg.score_memory_mib,
            cfg.name_memory_mib,
        )
        split = SplitCfg(
            1 / folds
            if folds > 1
            else (DEFAULT_VALIDATION_CFG.test_frac if cfg.validation_frac is None else cfg.validation_frac),
            cfg.seed,
            MinSupport(cfg.train_min_trns, cfg.train_min_theorems),
            MinSupport(cfg.validation_min_trns, cfg.validation_min_theorems),
            cfg.solver_seconds,
        )
        report = prepare_features(
            cfg.db,
            cfg.labels,
            cfg.output,
            PreparationCfg(vocabulary, split, folds, cfg.cooccurrence_shapes),
            candidates=cfg.candidates,
        )
    print(json.dumps({key: val for key, val in report.items() if key != "stages"}))
    return 0
