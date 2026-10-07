"""Schedule candidate extraction, inventory construction and feature conversion.

Extraction/conversion workers prepare one theorem per job; selection workers
reduce byte/count-bounded theorem batches. Results arrive in completion order.
Computation lives in candidates/inventory/features/supervised; archives owns serialization
and publication. The source DB is read-only and must remain completed/immutable.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from contextlib import ExitStack, closing
from dataclasses import dataclass
from functools import partial
from itertools import islice
from multiprocessing import get_context
from pathlib import Path
from random import Random
from time import monotonic
from typing import cast

import msgspec
import numpy as np
from graph_tool import openmp_set_num_threads

from trustmebro.extraction import records as r
from trustmebro.extraction.scheduler import Phase
from trustmebro.extraction.storage import (
    THEOREM_COLS,
    BlobCodec,
    Dicts,
    TheoremRow,
    decode_nat_ext,
    decode_theorem_row,
    encode_msgpack,
    stored_dicts,
)

from .archives import (
    ARRAY_DTYPE,
    CoverageHeader,
    Diagnostics,
    FeatureHeader,
    Footer,
    candidate_frames,
    decode_candidates,
    first_frame,
    pack_rows,
    publish,
    read_candidates,
    read_coverage,
    read_coverage_candidates,
    read_frames,
    read_inventory,
    read_selection,
    read_vocabulary,
    unpack_rows,
    write_coverage,
    write_inventory,
)
from .attributes import stat_width
from .candidates import DEFAULT_DEPTHS, Candidates, Selection, extract_candidates
from .corpus import LabelPolicy, read_label_policy
from .coverage import (
    DEFAULT_INDEX_MEMORY_MIB,
    CoverageCandidates,
    CoverageIndex,
    CoverSummary,
    PackedShapes,
    build_coverage_index,
    coverage_buffer_sizes,
    coverage_shape,
    require_coverage,
    select_coverage,
)
from .features import (
    DEFAULT_ATTRIBUTE_POLICY,
    DEFAULT_COVER_POLICY,
    NODE_NAMES,
    AttributePolicy,
    CoverPolicy,
    DimensionBudget,
    Entry,
    FeatureRows,
    FeatureStats,
    Layout,
    Representation,
    SupervisedPolicy,
    Supervision,
    Vocabulary,
    add_stats,
    compile_vocabulary,
    encode_candidates,
    encode_theorem,
    entry_dimensions,
    feature_blocks,
    fixed_dimensions,
    prepare_stats,
    select_vocabulary,
)
from .inventory import ShapeCounts, Summary, add_theorem
from .supervised import SelectionBatch, select_attributes, select_supervised

DEFAULT_INVENTORY_MEMORY_MIB = 4096
SELECTION_BATCH_BYTES = 2 * 2**20
SELECTION_BATCH_THEOREMS = 16


class CandidateCounts(msgspec.Struct):
    theorems: int = 0
    nodes: int = 0
    shape_records: int = 0
    occurrences: int = 0
    states: int = 0
    stored_bytes: int = 0
    uncompressed_stream_bytes: int = 0


@dataclass(frozen=True, slots=True)
class VocabularyBuildCfg:
    dims: int  # complete vector, not an enrichment-only allowance
    depths: tuple[int, ...] = (1, 2, 3, 4, 5)
    workers: int = 1
    hyp_slots: int = DEFAULT_ATTRIBUTE_POLICY.hyp_slots
    name_share: float = 0.2
    min_support: int = DEFAULT_ATTRIBUTE_POLICY.min_support  # labeled theorem support for optional shapes/names
    index_memory_mib: int = DEFAULT_INDEX_MEMORY_MIB
    score_memory_mib: int = 1024
    name_memory_mib: int = DEFAULT_ATTRIBUTE_POLICY.memory_mib


@dataclass(frozen=True)
class CandidateWorkerCfg:
    codec: BlobCodec
    depths: tuple[int, ...]


@dataclass(frozen=True)
class FeatureWorkerCfg:
    layout: Layout
    codec: BlobCodec
    decoder: msgspec.msgpack.Decoder


@dataclass(frozen=True, slots=True)
class SelectionWorkerCfg:
    fn: Callable[[Iterable[Candidates]], SelectionBatch]
    decoder: msgspec.msgpack.Decoder


# Fixed configuration is initialized once per persistent child, never per job.
# Each child owns its decoder/layout; no shared corpus aggregation lives here.
_candidate_cfg: CandidateWorkerCfg | None = None
_feature_cfg: FeatureWorkerCfg | None = None
_selection_cfg: SelectionWorkerCfg | None = None


# Bounded scheduling and read-only source selection.


def ready_results[Row, Result](
    pool: Executor, rows: Iterable[Row], workers: int, task: Callable[[Row], Result]
) -> Iterator[Result]:
    """Refill every freed slot before the consumer processes a completed batch.

    At most 2*workers jobs are outstanding. A completed batch can additionally
    retain up to 2*workers results; this is not a total-memory bound.
    """
    rows = iter(rows)
    pending: set[Future[Result]] = set()
    try:
        pending.update(pool.submit(task, row) for row in islice(rows, 2 * workers))
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            # Several jobs may finish while the consumer aggregates/writes.
            # Refill the whole batch now, not one slot per later consumer call.
            pending.update(pool.submit(task, row) for row in islice(rows, len(finished)))
            for future in finished:
                yield future.result()
    finally:
        for future in pending:
            future.cancel()


def _sample_theorems(db: sqlite3.Connection, limit: int, seed: int | None) -> tuple[str, ...]:
    # Sample the small name index, not decoded graphs or SQLite RANDOM() rows.
    # Stable input ordering makes seeded selection independent of worker count.
    with closing(db.execute("SELECT name FROM theorems ORDER BY id")) as rows:
        names = [row[0] for row in rows]
    return tuple(Random(seed).sample(names, min(limit, len(names))))


def _src_rows(db: sqlite3.Connection, selection: Selection) -> Iterator[TheoremRow]:
    if selection.theorems is not None:
        for name in selection.theorems:
            row = db.execute(f"SELECT {THEOREM_COLS} FROM theorems WHERE name = ?", (name,)).fetchone()
            if row is None:
                raise ValueError(f"selected theorem not found: {name!r}")
            yield row
    else:
        with closing(db.execute(f"SELECT {THEOREM_COLS} FROM theorems ORDER BY id")) as rows:
            yield from rows


def _theorem_names(path: Path | None) -> tuple[str, ...] | None:
    if path is None:
        return None
    return tuple(line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#"))


def _decode_src_row(row: TheoremRow, codec: BlobCodec) -> r.Theorem:
    """Shared DB boundary: check blob counts/spans before theorem preparation."""
    # Discovery/encoding validates spans and DAG references immediately after
    # this shared storage decode, so do not traverse the expression table twice.
    theorem = decode_theorem_row(row, codec, check_refs=False)
    if not theorem.trns:
        raise ValueError(f"theorem {theorem.name!r} has no transitions")
    return theorem


# Persistent theorem workers; discovery and conversion remain separate tasks.


def _init_candidate_worker(dicts: Dicts | None, depths: tuple[int, ...]) -> None:
    global _candidate_cfg
    openmp_set_num_threads(1)
    _candidate_cfg = CandidateWorkerCfg(BlobCodec(dicts), depths)


def _extract_row(row: TheoremRow, cfg: CandidateWorkerCfg) -> tuple[bytes, CandidateCounts]:
    candidates = extract_candidates(_decode_src_row(row, cfg.codec), cfg.depths)
    counts = CandidateCounts(
        1, len(candidates.nodes), len(candidates.shapes), len(candidates.occurrences), len(candidates.states)
    )
    return encode_msgpack(candidates), counts


def _candidate_worker(row: TheoremRow) -> tuple[bytes, CandidateCounts]:
    if _candidate_cfg is None:
        raise RuntimeError("candidate worker has not been initialized")
    return _extract_row(row, _candidate_cfg)


def _init_feature_worker(vocab: Vocabulary, dicts: Dicts | None) -> None:
    global _feature_cfg
    openmp_set_num_threads(1)
    _feature_cfg = FeatureWorkerCfg(
        compile_vocabulary(vocab), BlobCodec(dicts), msgspec.msgpack.Decoder(ext_hook=decode_nat_ext)
    )


def _convert_row(row: TheoremRow | bytes, cfg: FeatureWorkerCfg) -> FeatureRows:
    if isinstance(row, bytes):
        return encode_candidates(decode_candidates(row, cfg.decoder), cfg.layout)
    return encode_theorem(_decode_src_row(row, cfg.codec), cfg.layout)


def _feature_worker(row: TheoremRow | bytes) -> bytes:
    if _feature_cfg is None:
        raise RuntimeError("feature worker has not been initialized")
    return pack_rows(_convert_row(row, _feature_cfg))


def _conversion_results(
    rows: Iterable[TheoremRow | bytes], layout: Layout, workers: int, dicts: Dicts | None = None
) -> Iterator[FeatureRows | bytes]:
    if workers == 1:
        cfg = FeatureWorkerCfg(layout, BlobCodec(dicts), msgspec.msgpack.Decoder(ext_hook=decode_nat_ext))
        for row in rows:
            yield _convert_row(row, cfg)
    else:
        with (
            ProcessPoolExecutor(
                workers,
                mp_context=get_context("spawn"),
                initializer=_init_feature_worker,
                initargs=(layout.vocab, dicts),
            ) as pool,
            closing(ready_results(pool, rows, workers, _feature_worker)) as results,
        ):
            yield from results


def _init_selection_worker(fn: Callable[[Iterable[Candidates]], SelectionBatch]) -> None:
    global _selection_cfg
    openmp_set_num_threads(1)
    _selection_cfg = SelectionWorkerCfg(fn, msgspec.msgpack.Decoder(ext_hook=decode_nat_ext))


def _selection_worker(frames: tuple[bytes, ...]) -> SelectionBatch:
    if _selection_cfg is None:
        raise RuntimeError("selection worker has not been initialized")
    cfg = _selection_cfg
    return cfg.fn(decode_candidates(frame, cfg.decoder) for frame in frames)


def _selection_jobs(frames: Iterable[bytes]) -> Iterator[tuple[bytes, ...]]:
    """Bound encoded jobs by bytes/count; an oversized theorem travels alone."""
    batch: list[bytes] = []
    size = 0
    for frame in frames:
        if batch and size + len(frame) > SELECTION_BATCH_BYTES:
            yield tuple(batch)
            batch, size = [], 0
        batch.append(frame)
        size += len(frame)
        if len(batch) == SELECTION_BATCH_THEOREMS or size >= SELECTION_BATCH_BYTES:
            yield tuple(batch)
            batch, size = [], 0
    if batch:
        yield tuple(batch)


def _selection_results[Result: SelectionBatch](
    fn: Callable[[Iterable[Candidates]], Result],
    phase: str,
    *,
    candidates: Path,
    names: tuple[str, ...] | None,
    workers: int,
    total: int | None,
) -> Iterator[Result]:
    """A persistent pool per pass; only encoded input/reduced arrays cross IPC.

    ready_results bounds outstanding jobs and replenishes before aggregation.
    Each child owns one immutable pass configuration; the corpus index remains
    in the coordinator. Queue bounds exclude decoder/native scratch and config
    copies, and are not a process-tree memory limit.
    """
    with ExitStack() as stack:
        progress = stack.enter_context(Phase(phase))
        frames = stack.enter_context(closing(candidate_frames(candidates, names)))
        jobs = _selection_jobs(frames)
        count = 0
        denominator = "" if total is None else f"/{total:,}"
        worker_label = "worker" if workers == 1 else "workers"
        if workers == 1:
            decoder = msgspec.msgpack.Decoder(ext_hook=decode_nat_ext)
            results = (fn(decode_candidates(frame, decoder) for frame in job) for job in jobs)
        else:
            pool = stack.enter_context(
                ProcessPoolExecutor(
                    workers, mp_context=get_context("spawn"), initializer=_init_selection_worker, initargs=(fn,)
                )
            )
            results = ready_results(pool, jobs, workers, _selection_worker)
        stack.enter_context(closing(results))
        for batch in results:
            count += len(batch.scope)
            rate = count / max(monotonic() - progress.started, 1e-9)
            progress.details = f"{count:,}{denominator} theorems; {rate:.1f}/s; {workers} {worker_label}"
            yield cast(Result, batch)


# Coordinator progress and diagnostics.


def _show_progress(count: int, total: int, elapsed: float) -> None:
    print(f"\r\033[2KCandidates: {elapsed:.1f}s; theorems: {count:,}/{total:,}", end="", file=sys.stderr, flush=True)


def _candidate_frames(
    results: Iterable[tuple[bytes, CandidateCounts]], counts: CandidateCounts, started: float, total: int
) -> Iterator[bytes]:
    terminal = sys.stderr.isatty()
    last_update = monotonic()
    if terminal:
        _show_progress(counts.theorems, total, last_update - started)
    for data, batch in results:
        counts.theorems += batch.theorems
        counts.nodes += batch.nodes
        counts.shape_records += batch.shape_records
        counts.occurrences += batch.occurrences
        counts.states += batch.states
        yield data
        if terminal:
            now = monotonic()
            if now - last_update >= 0.25:
                _show_progress(counts.theorems, total, now - started)
                last_update = now
    if terminal:
        _show_progress(counts.theorems, total, monotonic() - started)


def _report(stats: FeatureStats, header: FeatureHeader, nnz: int) -> Diagnostics:
    return Diagnostics(
        header,
        stats.theorems,
        stats.covered_theorems,
        stats.states,
        stats.covered_states,
        stats.goal_covered,
        stats.hyp_covered,
        nnz,
        tuple(sorted(stats.active_dims.items())),
        tuple(sorted(stats.active_patterns.items())),
        stats.support.astype(ARRAY_DTYPE, copy=False).tobytes(),
        stats.pair_limit,
        stats.pairs.astype(ARRAY_DTYPE, copy=False).tobytes(),
    )


# Publication workflows.


def prepare_coverage_archive(
    src: Path,
    output: Path,
    *,
    depths: tuple[int, ...] | None = None,
    index_memory_mib: int = DEFAULT_INDEX_MEMORY_MIB,
    replace: bool = False,
    theorems: tuple[str, ...] | None = None,
) -> CoverageIndex:
    """Prepare coverage from a completed candidate archive, not the active DB.

    No solver is invoked. The retained-index guard is not a peak-RSS limit;
    full-corpus execution requires an agreed resource budget.
    """
    if output.resolve() == src.resolve() or (output.exists() and output.samefile(src)):
        raise ValueError("coverage output cannot replace the candidate archive")
    if output.exists() and not replace:
        raise FileExistsError(f"coverage archive exists: {output}; use --replace")
    stamp = src.stat()
    selection = read_selection(src)
    if theorems is not None:
        selection = msgspec.structs.replace(selection, theorems=theorems, limit=None, seed=None)
    depths = tuple(sorted(set(selection.depths if depths is None else depths)))
    if not depths or not set(depths).issubset(selection.depths):
        raise ValueError("coverage depths must be present in the candidate archive")
    with (
        Phase("Prepare expression coverage") as progress,
        closing(read_coverage_candidates(src, names=theorems)) as observations,
    ):

        def observed() -> Iterator[CoverageCandidates]:
            for count, theorem in enumerate(observations, start=1):
                progress.details = f"{count:,} theorems"
                yield theorem

        index = build_coverage_index(observed(), depths, index_memory_mib=index_memory_mib)
    if (src.stat().st_size, src.stat().st_mtime_ns) != (stamp.st_size, stamp.st_mtime_ns):
        raise ValueError("candidate archive changed during coverage preparation")
    if theorems is not None:
        selection = msgspec.structs.replace(selection, theorems=index.names)
    header = CoverageHeader(selection, depths, str(src.resolve()), stamp.st_size, stamp.st_mtime_ns, "packed-coverage")
    write_coverage(output, header, index, src=src, replace=replace)
    return index


def scan_candidates(
    db_path: Path,
    output: Path,
    *,
    depths: tuple[int, ...] = DEFAULT_DEPTHS,
    workers: int = 1,
    limit: int | None = None,
    seed: int | None = None,
    theorems: tuple[str, ...] | None = None,
    replace: bool = False,
) -> CandidateCounts:
    """Publish only after all selected rows succeed; never modify the source DB.

    At most 2*workers jobs are buffered, each with one theorem. This bounds the
    job queue, not theorem size, decoded graphs, native search scratch, encoded
    frames, compression buffers, or total process-tree memory.
    """
    if not depths or any(depth < 1 for depth in depths):
        raise ValueError("candidate depths must be positive and nonempty")
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError("workers and limit must be positive")
    if seed is not None and limit is None:
        raise ValueError("seed requires a random sample selected with limit")
    if theorems is not None and (limit is not None or not theorems):
        raise ValueError("a nonempty theorem selection cannot be combined with a limit")
    if theorems is not None and len(set(theorems)) != len(theorems):
        raise ValueError("the theorem selection contains duplicates")
    db_path = db_path.resolve()
    if output.resolve() == db_path or (output.exists() and output.samefile(db_path)):
        raise ValueError("candidate output cannot replace the source database")
    if output.exists() and not replace:
        raise FileExistsError(f"candidate archive exists: {output}; use --replace explicitly")
    stat = db_path.stat()
    selection = Selection(
        str(db_path), stat.st_size, stat.st_mtime_ns, tuple(sorted(set(depths))), limit, theorems, seed
    )
    counts = CandidateCounts()
    started = monotonic()
    try:
        with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as db:
            dicts = stored_dicts(db)
            if limit is not None:
                selection = msgspec.structs.replace(selection, theorems=_sample_theorems(db, limit, seed))
            total = (
                len(selection.theorems)
                if selection.theorems is not None
                else db.execute("SELECT COUNT(*) FROM theorems").fetchone()[0]
            )

            def frames(rows: Iterable[TheoremRow]) -> Iterator[bytes]:
                yield encode_msgpack(selection)
                if workers == 1:
                    cfg = CandidateWorkerCfg(BlobCodec(dicts), selection.depths)
                    results = (_extract_row(row, cfg) for row in rows)
                    yield from _candidate_frames(results, counts, started, total)
                else:
                    with (
                        ProcessPoolExecutor(
                            workers,
                            mp_context=get_context("spawn"),
                            initializer=_init_candidate_worker,
                            initargs=(dicts, selection.depths),
                        ) as pool,
                        closing(ready_results(pool, rows, workers, _candidate_worker)) as results,
                    ):
                        yield from _candidate_frames(results, counts, started, total)

            with closing(_src_rows(db, selection)) as rows:
                counts.uncompressed_stream_bytes = publish(output, frames(rows), sources=(db_path,), replace=replace)
        counts.stored_bytes = output.stat().st_size
    finally:
        if sys.stderr.isatty():
            print(file=sys.stderr)
    return counts


def build_inventory(
    src: Path, output: Path, *, index_memory_mib: int = DEFAULT_INVENTORY_MEMORY_MIB, replace: bool = False
) -> Summary:
    if src.resolve() == output.resolve():
        raise ValueError("inventory cannot overwrite the candidate archive")
    if output.exists() and not replace:
        raise FileExistsError(f"inventory exists: {output}; use --replace")
    if index_memory_mib < 1:
        raise ValueError("index memory budget must be positive")
    stamp = src.stat()
    selection = read_selection(src)
    counts = ShapeCounts(selection.depths, index_memory_mib * 2**20)
    started = monotonic()
    theorems = 0
    with Phase("Inventory candidates") as progress:
        for theorem in read_candidates(src):
            add_theorem(counts, theorem)
            theorems += 1
            progress.details = f"{theorems:,} theorems; {len(counts.shapes):,} distinct shapes"
    if (src.stat().st_size, src.stat().st_mtime_ns) != (stamp.st_size, stamp.st_mtime_ns):
        raise ValueError("candidate source changed during inventory construction")
    summary = Summary(
        str(src.resolve()),
        stamp.st_size,
        stamp.st_mtime_ns,
        selection,
        theorems,
        len(counts.shapes),
        monotonic() - started,
        counts.index_bytes(),
    )
    write_inventory(output, summary, counts, src=src, replace=replace)
    return summary


def select_archive(
    src: Path,
    output: Path,
    *,
    max_shapes: int = 10_000,
    min_support: int = 1,
    min_nodes: int = 2,
    replace: bool = False,
) -> Layout:
    vocab = select_vocabulary(read_inventory(src), max_shapes=max_shapes, min_support=min_support, min_nodes=min_nodes)
    layout = compile_vocabulary(vocab)
    publish(output, (encode_msgpack(vocab),), sources=(src,), replace=replace)
    return layout


def select_coverage_archive(
    src: Path,
    output: Path,
    *,
    policy: CoverPolicy = DEFAULT_COVER_POLICY,
    min_support: int = 1,
    min_nodes: int = 2,
    replace: bool = False,
) -> CoverSummary:
    """Publish an unsupervised structural baseline for exploratory comparisons.

    Full supervised fitting belongs to fit_vocabulary. This selector retains
    custom cover objectives without preparing statistics or name channels.
    The coverage archive is fully loaded; native model copies add to peak memory.
    """
    if output.resolve() == src.resolve() or (output.exists() and output.samefile(src)):
        raise ValueError("vocabulary output cannot replace a selection source")
    if output.exists() and not replace:
        raise FileExistsError(f"vocabulary exists: {output}; use --replace")
    stamp = src.stat()
    with Phase("Read expression coverage"):
        header, index = read_coverage(src)
    with Phase("Select richer cover and verify fallback"):
        selected = select_coverage(index, policy=policy, min_nodes=min_nodes, min_support=min_support)
    vocab = Vocabulary(
        header.depths,
        _coverage_entries(index.shapes, selected.cols),
        selection=header.selection,
        min_support=min_support,
        min_nodes=min_nodes,
        coverage=policy,
    )
    compile_vocabulary(vocab)
    if (src.stat().st_size, src.stat().st_mtime_ns) != (stamp.st_size, stamp.st_mtime_ns):
        raise ValueError("selection source changed during vocabulary selection")
    publish(output, (encode_msgpack(vocab),), sources=(src,), replace=replace)
    return selected.summary


def _fit_representation(
    candidates: Path, vocab: Vocabulary, label_policy: LabelPolicy, policy: AttributePolicy, workers: int
) -> Vocabulary:
    """Fit names once; callers own source guards and final publication."""
    stamp = candidates.stat()
    selection = read_selection(candidates)
    if vocab.representation is not None:
        raise ValueError("supply a structural vocabulary, not an already fitted representation")
    if not set(vocab.depths).issubset(selection.depths):
        raise ValueError("candidate archive lacks vocabulary discovery depths")
    if vocab.selection is None or msgspec.structs.replace(
        selection, depths=vocab.selection.depths, theorems=vocab.selection.theorems, limit=None, seed=None
    ) != msgspec.structs.replace(vocab.selection, limit=None, seed=None):
        raise ValueError("vocabulary and representation candidates must describe the same corpus selection")
    selection = vocab.selection
    layout = compile_vocabulary(vocab)
    source = partial(read_candidates, candidates, names=selection.theorems)
    mapper = partial(
        _selection_results,
        candidates=candidates,
        names=selection.theorems,
        workers=workers,
        total=None if selection.theorems is None else len(selection.theorems),
    )
    heads, names = select_attributes(source, layout, label_policy, policy, mapper=mapper, stage=Phase)
    representation = Representation(
        policy,
        heads,
        names,
        selection,
        str(candidates.resolve()),
        stamp.st_size,
        stamp.st_mtime_ns,
        json.dumps(msgspec.to_builtins(label_policy), sort_keys=True, separators=(",", ":")).encode(),
    )
    return msgspec.structs.replace(vocab, representation=representation)


def _coverage_entries(shapes: PackedShapes, cols: np.ndarray) -> tuple[Entry, ...]:
    selected = (coverage_shape(shapes, int(col)) for col in cols)
    return tuple(Entry(shape.ident, shape.edges) for shape in selected)


def _complete_dimension_costs(shapes: PackedShapes, shortlist: np.ndarray) -> np.ndarray:
    # Decode only the bounded shortlist for actual fixed-attribute costs.
    return np.fromiter(
        (entry_dimensions(coverage_shape(shapes, int(col)).edges, attributes=True) for col in shortlist), dtype=np.int64
    )


def check_vocabulary_cfg(cfg: VocabularyBuildCfg) -> None:
    if not np.isfinite(cfg.name_share) or not 0 <= cfg.name_share <= 1:
        raise ValueError("name share must lie between zero and one")
    if min(cfg.workers, cfg.min_support, cfg.index_memory_mib, cfg.score_memory_mib, cfg.name_memory_mib) < 1:
        raise ValueError("workers, support and retained-memory guards must be positive")
    if cfg.hyp_slots < 0 or not cfg.depths or any(depth < 1 for depth in cfg.depths):
        raise ValueError("hypothesis slots must be nonnegative and discovery depths positive")
    stats_width = stat_width(len(NODE_NAMES), cfg.hyp_slots)
    if cfg.dims < stats_width:
        raise ValueError(
            f"total dimension budget {cfg.dims:,} cannot fit {stats_width:,} statistics/slot columns alone"
        )


def fit_vocabulary(
    candidates: Path,
    labels: Path,
    output: Path,
    cfg: VocabularyBuildCfg,
    *,
    coverage: Path,
    theorems: tuple[str, ...] | None = None,
    replace: bool = False,
) -> dict[str, object]:
    """Cached candidates -> training-only cover/scores/names -> frozen layout.

    Retain coverage for inspection and reuse its prepared index instead of decoding it
    again. The total cap includes fixed channels; it is not a memory/RSS limit.
    The selected heuristic baseline is preserved, not claimed globally minimal.
    """
    check_vocabulary_cfg(cfg)
    stats_width = stat_width(len(NODE_NAMES), cfg.hyp_slots)
    sources = (candidates, labels)
    targets = (coverage, output)
    # Reject every collision/existing output before running any expensive stage.
    for idx, path in enumerate(targets):
        for source in (*sources, *targets[:idx]):
            if path.resolve() == source.resolve() or (path.exists() and source.exists() and path.samefile(source)):
                raise ValueError("vocabulary build outputs must be distinct and cannot replace input files")
        if path.exists() and (not replace or not path.is_file()):
            raise FileExistsError(f"build output exists: {path}; use --replace for existing files")
    label_policy = read_label_policy(labels)
    stamps = [(path.stat().st_size, path.stat().st_mtime_ns) for path in sources]
    started = monotonic()
    index = prepare_coverage_archive(
        candidates,
        coverage,
        depths=cfg.depths,
        index_memory_mib=cfg.index_memory_mib,
        replace=replace,
        theorems=theorems,
    )
    intermediate_stamps = [(path.stat().st_size, path.stat().st_mtime_ns) for path in (candidates, coverage)]
    selection = msgspec.structs.replace(
        read_selection(candidates), depths=cfg.depths, theorems=index.names, limit=None, seed=None
    )
    theorem_count = len(index.names)

    with Phase("Select coverage baseline") as progress:
        selected = select_coverage(index)
        baseline_entries = _coverage_entries(index.shapes, selected.cols)
        baseline_width = fixed_dimensions(baseline_entries, cfg.hyp_slots)
        if baseline_width > cfg.dims:
            raise ValueError(
                f"total dimension budget {cfg.dims:,} cannot fit the selected coverage baseline "
                f"({baseline_width:,} columns, including {stats_width:,} statistics/slot columns); "
                "increase --dims; this heuristic cover is not a proof of the minimum feasible width"
            )
        progress.details = f"{len(baseline_entries):,} shapes; {baseline_width:,}/{cfg.dims:,} dimensions"
    remaining = cfg.dims - baseline_width
    shape_allowance = min(remaining, int(remaining * (1 - cfg.name_share)))
    cols, supervision = selected.cols, None
    vocab_entries = baseline_entries
    added_dims = 0
    if shape_allowance:
        shape_policy = SupervisedPolicy(
            dims=shape_allowance, min_support=cfg.min_support, memory_mib=cfg.score_memory_mib
        )
        source = partial(read_candidates, candidates, names=selection.theorems)
        mapper = partial(
            _selection_results,
            candidates=candidates,
            names=selection.theorems,
            workers=cfg.workers,
            total=len(index.names),
        )
        enriched = select_supervised(
            source,
            index,
            selected.cols,
            label_policy,
            shape_policy,
            depths=cfg.depths,
            dimension_costs=partial(_complete_dimension_costs, index.shapes),
            mapper=mapper,
            stage=Phase,
        )
        cols = np.concatenate((cols, enriched.cols))
        vocab_entries += _coverage_entries(index.shapes, enriched.cols)
        added_dims = enriched.dims
        stamp = candidates.stat()
        supervision = Supervision(
            shape_policy,
            json.dumps(msgspec.to_builtins(label_policy), sort_keys=True, separators=(",", ":")).encode(),
            str(candidates.resolve()),
            stamp.st_size,
            stamp.st_mtime_ns,
            label_policy.labels,
            tuple(map(int, enriched.evidence.totals)),
            enriched.evidence.labeled_theorems,
            enriched.shortlist_shapes,
            len(enriched.cols),
            added_dims,
            dimension_cost="complete",
            available_association=tuple(map(float, enriched.evidence.scores.max(axis=0, initial=0))),
            selected_association=enriched.selected_association,
            label_strength=enriched.label_strength,
        )
        del enriched
    require_coverage(index, cols)
    vocab = Vocabulary(
        selection.depths,
        vocab_entries,
        selection=selection,
        coverage=DEFAULT_COVER_POLICY,
        supervision=supervision,
        budget=DimensionBudget(cfg.dims, baseline_width, cfg.name_share),
    )
    # Release the corpus root/shape index before name evidence/scoring is built.
    del index, selected, baseline_entries
    attrs = AttributePolicy(
        hyp_slots=cfg.hyp_slots,
        name_dims=remaining - added_dims,
        min_support=cfg.min_support,
        memory_mib=cfg.name_memory_mib,
    )
    vocab = _fit_representation(candidates, vocab, label_policy, attrs, cfg.workers)
    layout = compile_vocabulary(vocab)  # independently enforce the complete cap
    if [(path.stat().st_size, path.stat().st_mtime_ns) for path in sources] != stamps:
        raise ValueError("database or label policy changed during vocabulary construction")
    if [(path.stat().st_size, path.stat().st_mtime_ns) for path in (candidates, coverage)] != intermediate_stamps:
        raise ValueError("candidate or coverage artifact changed during vocabulary construction")
    raw_bytes = publish(output, (encode_msgpack(vocab),), sources=(*sources, coverage), replace=replace)
    return {
        "theorems": theorem_count,
        "shapes": len(vocab.entries),
        "dimension_budget": cfg.dims,
        "baseline_dimensions": baseline_width,
        "shape_enrichment_dimensions": added_dims,
        "shape_associations": {
            label: {"available_r2": available, "selected_r2": retained, "balanced_strength": strength}
            for label, available, retained, strength in zip(
                supervision.labels,
                supervision.available_association,
                supervision.selected_association,
                supervision.label_strength,
                strict=True,
            )
        }
        if supervision is not None
        else {},
        "name_dimensions": layout.width - baseline_width - added_dims,
        "dimensions": layout.width,
        "unused_dimensions": cfg.dims - layout.width,
        "blocks": {name: hi - lo for name, (lo, hi) in feature_blocks(layout).items()},
        "candidates": str(candidates),
        "coverage": str(coverage),
        "vocab": str(output),
        "stored_bytes": output.stat().st_size,
        "uncompressed_stream_bytes": raw_bytes,
        "dur_sec": monotonic() - started,
    }


def convert_corpus(
    src: Path,
    vocab_path: Path,
    output: Path,
    stats_path: Path,
    *,
    candidates: bool = False,
    workers: int = 1,
    limit: int | None = None,
    seed: int | None = None,
    theorems: tuple[str, ...] | None = None,
    pair_limit: int = 128,
    labels: Path | None = None,
    replace: bool = False,
    split_id: str | None = None,
    expected_theorems: tuple[str, ...] | None = None,
) -> dict[str, int | float]:
    """Stream results, preserving all rows and writing no temporary SQLite data.

    Both sources bound outstanding jobs to 2*workers, not total memory. Each
    child retains its own compiled vocabulary and theorem scratch. Candidate
    frames are decoded in children and avoid discovery entirely. Sparse products
    can still grow with a single large theorem. No theorem or state is lost.
    """
    if workers < 1 or (limit is not None and limit < 1) or (seed is not None and limit is None):
        raise ValueError("workers/limit must be positive; seed requires limit")
    if theorems is not None and (not theorems or len(set(theorems)) != len(theorems) or limit is not None):
        raise ValueError("explicit theorem names must be nonempty, distinct and cannot accompany limit")
    if candidates and (limit is not None or theorems is not None):
        raise ValueError("candidate reuse streams its existing selection; sampling applies only to DB input")
    sources = (src, vocab_path) if labels is None else (src, vocab_path, labels)
    label_policy = read_label_policy(labels) if labels is not None else None
    frozen_labels = msgspec.json.encode(label_policy) if label_policy is not None else None
    if output.resolve() == stats_path.resolve():
        raise ValueError("feature and diagnostics outputs must be different paths")
    for path in (output, stats_path):
        if any(path.resolve() == source.resolve() or (path.exists() and path.samefile(source)) for source in sources):
            raise ValueError("output cannot replace a source artifact")
        if path.exists() and not replace:
            raise FileExistsError(f"output exists: {path}; use --replace")
    layout = compile_vocabulary(read_vocabulary(vocab_path))
    stats = prepare_stats(layout, pair_limit)
    stamp = src.stat()
    src_selection = (
        read_selection(src)
        if candidates
        else Selection(str(src.resolve()), stamp.st_size, stamp.st_mtime_ns, layout.vocab.depths, limit, theorems, seed)
    )
    if candidates and not set(layout.vocab.depths).issubset(src_selection.depths):
        raise ValueError("candidate archive lacks requested vocabulary discovery depths")
    started = monotonic()
    nnz = 0

    def frames(results: Iterable[FeatureRows | bytes], header: FeatureHeader) -> Iterator[bytes]:
        nonlocal nnz
        expected = None if expected_theorems is None else set(expected_theorems)
        seen: set[str] = set()
        yield encode_msgpack(header)
        with Phase("Convert proof states") as progress:
            for result in results:
                rows = (
                    unpack_rows(result, layout.width, real=layout.vocab.representation is not None)
                    if isinstance(result, bytes)
                    else result
                )
                if expected is not None:
                    if rows.name not in expected or rows.name in seen:
                        raise ValueError("conversion theorem population disagrees with split")
                    seen.add(rows.name)
                if label_policy is not None:
                    for tactic in rows.tactics:
                        label_policy.label(tactic)  # reject unmapped=error before publishing the archive
                add_stats(stats, rows)
                nnz += rows.matrix.nnz
                progress.details = f"{stats.theorems:,} theorems; {stats.states:,} states; {nnz:,} nonzeros"
                yield result if isinstance(result, bytes) else pack_rows(rows)
        if expected is not None and seen != expected:
            raise ValueError("candidate archive is missing development theorems")
        yield encode_msgpack(Footer(stats.theorems, stats.states))

    if candidates:
        header = FeatureHeader(
            layout.vocab, src_selection, str(src.resolve()), stamp.st_size, stamp.st_mtime_ns, frozen_labels, split_id
        )
        with closing(read_frames(src)) as observations:
            msgspec.msgpack.decode(first_frame(observations), type=Selection)
            with closing(_conversion_results(observations, layout, workers)) as results:
                raw_bytes = publish(output, frames(results, header), sources=sources, replace=replace)
    else:
        with closing(sqlite3.connect(src.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            if limit is not None:
                src_selection = msgspec.structs.replace(src_selection, theorems=_sample_theorems(db, limit, seed))
            header = FeatureHeader(
                layout.vocab,
                src_selection,
                str(src.resolve()),
                stamp.st_size,
                stamp.st_mtime_ns,
                frozen_labels,
                split_id,
            )
            with (
                closing(_src_rows(db, src_selection)) as observations,
                closing(_conversion_results(observations, layout, workers, stored_dicts(db))) as results,
            ):
                raw_bytes = publish(output, frames(results, header), sources=sources, replace=replace)
    # The rows artifact is independently complete. A failure publishing the
    # diagnostics leaves it usable; these two files are not an atomic pair.
    report = _report(stats, header, nnz)
    stat_bytes = publish(stats_path, (encode_msgpack(report),), sources=sources, replace=replace)
    return {
        "theorems": stats.theorems,
        "states": stats.states,
        "dimensions": layout.width,
        "shapes": len(layout.vocab.entries),
        "nontrivial_covered_theorems": stats.covered_theorems,
        "nontrivial_covered_states": stats.covered_states,
        "nnz": nnz,
        "cooccurrence_shapes": stats.pair_limit,
        "stored_bytes": output.stat().st_size,
        "uncompressed_stream_bytes": raw_bytes,
        "diagnostics_stored_bytes": stats_path.stat().st_size,
        "diagnostics_uncompressed_stream_bytes": stat_bytes,
        "dur_sec": monotonic() - started,
    }


# Public command entry points.


def candidates_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract rooted structural candidates, without selecting a vocabulary."
    )
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="compressed MessagePack candidate archive")
    parser.add_argument("--depths", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument("--workers", type=int, default=1)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--limit", type=int, help="uniform random sample of up to N theorems, without replacement")
    group.add_argument("--theorems-from", type=Path, help="one exact theorem name per line; # comments ignored")
    parser.add_argument("--seed", type=int, help="reproduce the random --limit selection on the same database")
    parser.add_argument("--replace", action="store_true")
    cfg = parser.parse_args(argv)
    if cfg.seed is not None and cfg.limit is None:
        parser.error("--seed requires --limit")
    theorems = _theorem_names(cfg.theorems_from)
    started = monotonic()
    counts = scan_candidates(
        cfg.db,
        cfg.output,
        depths=tuple(cfg.depths),
        workers=cfg.workers,
        limit=cfg.limit,
        seed=cfg.seed,
        theorems=theorems,
        replace=cfg.replace,
    )
    print(json.dumps({**msgspec.to_builtins(counts), "duration_s": round(monotonic() - started, 3)}))
    return 0


def inventory_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory canonical candidate shapes and exact observation counts.")
    parser.add_argument("--input", type=Path, required=True, help="candidate archive")
    parser.add_argument("--output", type=Path, required=True, help="compressed global inventory")
    parser.add_argument(
        "--index-memory-mib",
        type=int,
        default=DEFAULT_INVENTORY_MEMORY_MIB,
        help="estimated retained-index guard, not RSS",
    )
    parser.add_argument("--replace", action="store_true")
    cfg = parser.parse_args(argv)
    started = monotonic()
    summary = build_inventory(cfg.input, cfg.output, index_memory_mib=cfg.index_memory_mib, replace=cfg.replace)
    print(msgspec.json.encode(msgspec.to_builtins(summary) | {"dur_sec": monotonic() - started}).decode())
    return 0


def coverage_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare individual-expression coverage without graph rediscovery.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", help="subset of archived discovery depths; default all")
    parser.add_argument(
        "--index-memory-mib", type=int, default=DEFAULT_INDEX_MEMORY_MIB, help="estimated retained-index guard, not RSS"
    )
    parser.add_argument("--replace", action="store_true")
    cfg = parser.parse_args(argv)
    started = monotonic()
    index = prepare_coverage_archive(
        cfg.candidates,
        cfg.output,
        depths=None if cfg.depths is None else tuple(cfg.depths),
        index_memory_mib=cfg.index_memory_mib,
        replace=cfg.replace,
    )
    print(
        json.dumps(
            {
                "theorems": len(index.names),
                "roots": len(index.refs),
                "shapes": len(index.support),
                "memberships": index.matches.nnz,
                "stored_bytes": cfg.output.stat().st_size,
                "buffer_bytes": coverage_buffer_sizes(index),
                "dur_sec": monotonic() - started,
            }
        )
    )
    return 0


def select_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze an unsupervised structural baseline for exploration.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inventory", type=Path, help="support-ranked baseline (not guaranteed coverage)")
    source.add_argument("--coverage", type=Path, help="per-expression coverage archive; guaranteed observed coverage")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shapes", type=int, help="inventory only; default 10000; never truncate a cover")
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--min-nodes", type=int, default=2)
    parser.add_argument(
        "--objective", choices=("entries", "dims"), help=f"coverage only; default {DEFAULT_COVER_POLICY.objective}"
    )
    parser.add_argument(
        "--improvement-steps",
        type=int,
        help=f"coverage only; bounded local search; default {DEFAULT_COVER_POLICY.improvement_steps}",
    )
    parser.add_argument("--replace", action="store_true")
    cfg = parser.parse_args(argv)
    if cfg.coverage is not None:
        if cfg.max_shapes is not None:
            parser.error("--max-shapes cannot truncate a coverage-guaranteeing vocabulary")
        started = monotonic()
        report = select_coverage_archive(
            cfg.coverage,
            cfg.output,
            policy=CoverPolicy(
                DEFAULT_COVER_POLICY.objective if cfg.objective is None else cfg.objective,
                DEFAULT_COVER_POLICY.improvement_steps if cfg.improvement_steps is None else cfg.improvement_steps,
            ),
            min_support=cfg.min_support,
            min_nodes=cfg.min_nodes,
            replace=cfg.replace,
        )
        print(msgspec.json.encode(msgspec.to_builtins(report) | {"dur_sec": monotonic() - started}).decode())
        return 0
    if cfg.objective is not None or cfg.improvement_steps is not None:
        parser.error("--objective and --improvement-steps require --coverage")
    layout = select_archive(
        cfg.inventory,
        cfg.output,
        max_shapes=10_000 if cfg.max_shapes is None else cfg.max_shapes,
        min_support=cfg.min_support,
        min_nodes=cfg.min_nodes,
        replace=cfg.replace,
    )
    print(json.dumps({"shapes": len(layout.vocab.entries), "dimensions": 2 * layout.block_width}))
    return 0
