"""Atomic, framed MessagePack/Zstandard measurement archives; no database caches."""

from __future__ import annotations

import fcntl
import os
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from compression import zstd
from contextlib import closing, contextmanager
from dataclasses import dataclass, fields
from functools import cache, cached_property, partial
from itertools import batched
from pathlib import Path
from struct import Struct
from tempfile import NamedTemporaryFile, TemporaryDirectory
from types import GenericAlias
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import msgspec
import numpy as np

from trustmebro.artifact_sizes import report_file_sizes
from trustmebro.extraction.storage import Dicts, decode_nat_ext, encode_msgpack
from trustmebro.framing import read_frames as read_binary_frames
from trustmebro.framing import write_frame as write_binary_frame

from .diagnostics import TimingLog, checkpoint, timed, timed_batches
from .measurements import *

if TYPE_CHECKING:
    from .metrics import PatternCounts


# Archive types, paths, and framing defaults.


type Analysis = Literal["metrics", "topology", "patterns", "vocabulary"]
type MetricRow = StateRow | ExprRow | StralRow | ReuseRow | FreqRow | GraphSize
type StoredShape = tuple[bytes, int, int, int, str, int, tuple[tuple[int, int], ...]]
type PairBatch = tuple[ViewMode, ComparisonLvl, list[PairRow]]
type StoredPair = tuple[tuple[list[int], list[int]], int]
type StatsRecord = (
    tuple[Literal["theorem"], str, list[StateRow], list[ExprRow]]
    | tuple[Literal["stral"], list[StralRow]]
    | tuple[Literal["mdata"], Mdata]
    | tuple[Literal["freqs"], list[FreqRow]]
    | tuple[Literal["shape"], ShapeKey, str, int, bytes]
    | tuple[Literal["global_reuse"], str, list[ReuseRow], list[ReuseRow]]
    | tuple[Literal["stats"], StateSummary, FreqStats]
)
type StatsSink = Callable[[StatsRecord], None]


class SrcStamp(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    path: str
    size: int
    modified_ns: int
    limit: int | None

    @classmethod
    def from_path(cls, path: Path, limit: int | None) -> SrcStamp:
        stat = path.stat()
        return cls(str(path.resolve()), stat.st_size, stat.st_mtime_ns, limit)


@dataclass(frozen=True)
class AnalysisCfg:
    atlas_min_nodes: int
    depths: tuple[int, ...]
    embedding: bool = False


class Gen(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    dir: str
    src: SrcStamp
    cfg: AnalysisCfg
    artifacts: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisPaths:
    """One naming contract shared by producers, renderers, and CLI checks."""

    root: Path

    @cached_property
    def manifest(self) -> dict[Analysis, Gen]:
        path = self.root / MANIFEST
        return read_summary(path, dict[Analysis, Gen]) if path.is_file() else {}

    def dir(self, kind: Analysis) -> Path:
        if (gen := self.manifest.get(kind)) is None:
            return self.root
        if not gen.dir.startswith("run-") or Path(gen.dir).name != gen.dir:
            raise ValueError("invalid analysis-generation directory")
        return self.root / ".generations" / gen.dir

    def artifact(self, kind: Analysis, name: str) -> Path:
        if Path(name).name != name:
            raise ValueError("invalid analysis-artifact name")
        return self.dir(kind) / name

    def require(self, kinds: Iterable[Analysis], artifacts: Mapping[Analysis, Iterable[str]]) -> None:
        """Validate a pinned manifest and the selected consumers' dependencies."""
        srcs: set[SrcStamp] = set()
        for kind in kinds:
            gen = self.manifest.get(kind)
            if gen is None:
                raise ValueError(f"no published {kind} analysis; run trustmebro-analyze --replace")
            srcs.add(gen.src)
            for name in artifacts[kind]:
                if name not in gen.artifacts or not self.artifact(kind, name).is_file():
                    raise ValueError(f"incomplete {kind} analysis: missing {name}; rerun trustmebro-analyze --replace")
        if len(srcs) > 1:
            raise ValueError("selected analyses describe different source selections; regenerate them together")

    def analysis(self, kind: Analysis) -> Path:
        names = {
            "metrics": "metrics.msgpack.zst",
            "topology": "topology-counts.msgpack.zst",
            "patterns": "local-patterns.msgpack.zst",
            "vocabulary": "vocabulary-observations.msgpack.zst",
        }
        return self.artifact(kind, names[kind])

    def topo(self, mode: ViewMode = ViewMode.ORIGINAL) -> Path:
        suffix = "" if mode == "original" else f"-{mode}"
        return self.artifact("topology", f"topology-counts{suffix}.msgpack.zst")

    def patterns(self, mode: ViewMode, depth: int) -> Path:
        return self.artifact("patterns", f"local-patterns-{mode}-{depth}.msgpack.zst")

    def stats(self, kind: Literal["topology", "comparison", "head", "pattern"]) -> Path:
        return self.artifact("patterns" if kind == "pattern" else "topology", f"{kind}-statistics.msgpack.zst")

    def embedding(self, mode: str) -> Path:
        if mode not in ("size-aware", "shape-only"):
            raise ValueError(f"unknown embedding mode: {mode}")
        suffix = "" if mode == "size-aware" else "-shape-only"
        return self.artifact("metrics", f"structural-embedding{suffix}.npz")

    @property
    def embedding_info(self) -> Path:
        return self.artifact("metrics", "embedding.msgpack.zst")

    @property
    def comparisons(self) -> Path:
        return self.artifact("topology", "topology-comparisons.msgpack.zst")

    @property
    def examples(self) -> Path:
        return self.artifact("topology", "topology-examples.msgpack.zst")

    @property
    def heads(self) -> Path:
        return self.artifact("topology", "function-heads.msgpack.zst")


LENGTH = Struct("<Q")
MANIFEST = "analysis-manifest.msgpack.zst"
decode_record = partial(msgspec.msgpack.decode, ext_hook=decode_nat_ext)


# Record and array encoding.


@cache
def row_fields(cls: type[MetricRow]) -> tuple[str, ...]:
    return cls.__struct_fields__


def row_vals(row: MetricRow) -> tuple[object, ...]:
    return tuple(getattr(row, name) for name in row_fields(type(row)))


def decode_row[Row: MetricRow](vals: list[object], cls: type[Row]) -> Row:
    data = dict(zip(row_fields(cls), vals, strict=True))
    row = msgspec.convert(data, type=cls)
    if any(type(val) is int and val < 0 for val in row_vals(row)):
        raise ValueError("metric counts must be nonnegative")
    return row


def array_data(array: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    return array.dtype.str, array.shape, array.tobytes()


def decode_array(data: object) -> np.ndarray:
    dtype, shape, buffer = msgspec.convert(data, type=tuple[str, tuple[int, ...], bytes])
    if np.dtype(dtype) != np.dtype(np.int64) or len(shape) != 2 or any(dim < 0 for dim in shape):
        raise ValueError("invalid concentration-grid dtype or shape")
    if len(buffer) != shape[0] * shape[1] * np.dtype(dtype).itemsize:
        raise ValueError("concentration-grid byte length does not match its shape")
    array = np.frombuffer(buffer, dtype=dtype).reshape(shape)
    if np.any(array < 0):
        raise ValueError("concentration-grid counts must be nonnegative")
    return array


def _encode_mdata(data: Mdata) -> dict[str, object]:
    vals = {field.name: getattr(data, field.name) for field in fields(Mdata)}
    for name in ("conc", "rotated_conc"):
        vals[name] = array_data(getattr(data, name))
    return vals


def _decode_mdata(vals: dict) -> Mdata:
    vals = vals.copy()
    for name in ("conc", "rotated_conc"):
        vals[name] = decode_array(vals[name])
    if vals["complete"] is not True or not isinstance(vals["src_db"], str):
        raise ValueError("invalid analysis metadata")
    for name in ("theorem_limit", "empty_ctxts", "stral_roots", "stral_sample_size", "atlas_min_nodes", "expr_idents"):
        val = vals[name]
        if (name != "theorem_limit" or val is not None) and (type(val) is not int or val < 0):
            raise ValueError(f"invalid analysis metadata: {name}")
    return Mdata(**vals)


def _decode_stats(record: list) -> StatsRecord:
    match record:
        case ["theorem", name, states, exprs]:
            return (
                "theorem",
                name,
                [decode_row(row, StateRow) for row in states],
                [decode_row(row, ExprRow) for row in exprs],
            )
        case ["global_reuse", name, states, exprs]:
            return (
                "global_reuse",
                name,
                [decode_row(row, ReuseRow) for row in states],
                [decode_row(row, ReuseRow) for row in exprs],
            )
        case ["stral", rows]:
            return "stral", [decode_row(row, StralRow) for row in rows]
        case ["freqs", rows]:
            return "freqs", [decode_row(row, FreqRow) for row in rows]
        case ["mdata", vals]:
            return "mdata", _decode_mdata(vals)
        case ["shape", key, name, root, blob]:
            key, name, root, blob = msgspec.convert((key, name, root, blob), type=tuple[ShapeKey, str, int, bytes])
            return "shape", key, name, root, blob
        case ["stats", state, freqs]:
            return "stats", msgspec.convert(state, type=StateSummary), msgspec.convert(freqs, type=FreqStats)
        case _:
            raise ValueError("unknown metrics record; regenerate analysis artifacts")


# Framed I/O and atomic analysis publication.


@overload
def read_summary(path: Path) -> Any: ...


@overload
def read_summary[T](path: Path, cls: type[T]) -> T: ...


def read_summary(path: Path, cls: type[Any] = object) -> Any:
    with closing(records(path)) as stream:
        val = msgspec.convert(decode_record(next(stream)), type=cls)
        if next(stream, None) is not None:
            raise ValueError(f"unexpected records after archive summary: {path}")
        return val


def write_summary(path: Path, val: object) -> None:
    with writer(path) as write:
        write(val)


@contextmanager
def writer(
    path: Path, *, timings: TimingLog | None = None, report_sizes: bool = True
) -> Generator[Callable[[object], None]]:
    """Publish the archive only after its producer finishes successfully."""
    with NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as tmp:
        pending = Path(tmp.name)
        try:
            with zstd.open(tmp, "wb", level=3) as stream:

                def write(item: object) -> None:
                    encoded = timed(timings, path.name, "archive.encode", encode_msgpack, item)
                    with checkpoint(timings, path.name, "archive.compress_write"):
                        write_binary_frame(stream, encoded, LENGTH)

                yield write
                uncompressed_bytes = stream.tell()
            pending.replace(path)
            if report_sizes:
                report_file_sizes(path, uncompressed_bytes)
        finally:
            pending.unlink(missing_ok=True)


def records(path: Path) -> Generator[bytes]:
    """Stream records and reject truncation; frame lengths are not memory-capped."""
    with zstd.open(path, "rb") as stream:
        try:
            yield from read_binary_frames(stream, LENGTH)
        except ValueError as error:
            raise ValueError(f"{error}: {path}") from error


def vocabulary_theorems(path: Path) -> Iterator[VocabTheorem]:
    """Stream label-neutral observations; callers choose labels and feature predicates."""
    for frame in records(path):
        # Decode extensions before typed conversion, as in durable extraction
        # storage: typed integer fields reject an ext before invoking ext_hook.
        yield msgspec.convert(decode_record(frame), type=VocabTheorem)


def write_batches(write: Callable[[object], None], rows: Iterable[object], size: int = 50_000) -> None:
    for batch in batched(rows, size):
        write(batch)


def _products(paths: AnalysisPaths, kind: Analysis, cfg: AnalysisCfg) -> tuple[Path, ...]:
    from .measurements import PATTERN_MODES, VIEW_MODES

    match kind:
        case "metrics":
            required = [paths.analysis(kind)]
            if cfg.embedding:
                required.append(paths.embedding_info)
                required.extend(paths.embedding(mode) for mode in read_summary(paths.embedding_info, tuple[str, ...]))
            return tuple(required)
        case "topology":
            return (
                *(paths.topo(mode) for mode in VIEW_MODES),
                paths.comparisons,
                paths.examples,
                paths.heads,
                *(paths.stats(name) for name in ("topology", "comparison", "head")),
            )
        case "patterns":
            return (
                paths.analysis(kind),
                paths.stats("pattern"),
                *(paths.patterns(mode, depth) for mode in PATTERN_MODES for depth in cfg.depths),
            )
        case "vocabulary":
            return (paths.analysis(kind),)


@contextmanager
def analysis_set(
    root: Path,
    kinds: tuple[Analysis, ...],
    src: Path,
    *,
    limit: int | None = None,
    cfg: AnalysisCfg,
    replace: bool = False,
) -> Generator[AnalysisPaths]:
    """One manifest replacement publishes the entire request, including embeddings.

    Readers pin immutable generations. Failed producers leave the old manifest
    intact; unselected analyses retain their previous generation. The directory
    lock prevents concurrent publishers from losing each other's manifest edits.
    """
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root, os.O_RDONLY)
    gens = root / ".generations"
    published: Path | None = None
    committed = False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prev = AnalysisPaths(root)
        kinds = msgspec.convert(kinds, type=tuple[Analysis, ...])
        if not kinds:
            raise ValueError("at least one analysis is required")
        for kind in kinds:
            if not replace and (kind in prev.manifest or prev.analysis(kind).exists()):
                raise FileExistsError(f"{kind} measurements already exist; use --replace")
        stamp = SrcStamp.from_path(src, limit)
        gens.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix=".pending-", dir=gens) as tmp:
            pending = AnalysisPaths(Path(tmp))
            yield pending
            if SrcStamp.from_path(src, limit) != stamp:
                raise ValueError("source database changed during analysis")
            manifest = prev.manifest.copy()
            dir_name = "run-" + pending.root.name.removeprefix(".pending-")
            for kind in kinds:
                products = _products(pending, kind, cfg)
                if any(not path.is_file() for path in products):
                    raise ValueError(f"cannot publish incomplete {kind} analysis")
                manifest[kind] = Gen(dir_name, stamp, cfg, tuple(path.name for path in products))
            published = gens / dir_name
            pending.root.rename(published)
            write_summary(root / MANIFEST, manifest)
            committed = True
    finally:
        if published is not None and not committed:
            # A signal can arrive after the atomic swap but before `committed`
            # is assigned. Never remove a generation already referenced by it.
            try:
                active = AnalysisPaths(root).manifest.values()
                refd = any(gen.dir == published.name for gen in active)
            except OSError, ValueError, msgspec.DecodeError, msgspec.ValidationError:
                refd = True  # uncertain publication: retain recoverable data
            if not refd:
                shutil.rmtree(published)
        if gens.exists():
            try:
                gens.rmdir()  # only an empty, newly created container
            except OSError:
                pass
        os.close(descriptor)


# Metric archives and selected column readers.


@contextmanager
def stats_writer(path: Path) -> Generator[StatsSink]:
    """The completion marker distinguishes finished statistics from incomplete runs."""
    if path.exists():
        raise FileExistsError(f"statistics already exist: {path}; run trustmebro-analyze --replace")
    path.parent.mkdir(parents=True, exist_ok=True)

    with writer(path) as write:

        def write_stats(record: StatsRecord) -> None:
            match record:
                case ("theorem", name, states, exprs) | ("global_reuse", name, states, exprs):
                    write((record[0], name, [row_vals(row) for row in states], [row_vals(row) for row in exprs]))
                case ("stral" | "freqs", rows):
                    write((record[0], [row_vals(row) for row in rows]))
                case ("mdata", data):
                    write(("mdata", _encode_mdata(data)))
                case _:
                    write(record)

        yield write_stats
        write(("complete",))


def selected_records(path: Path, kinds: set[str] | None = None) -> Generator[tuple[str, list[msgspec.Raw]]]:
    """Select framed payloads before decoding their rows, including completion checks."""
    with closing(records(path)) as stream:
        for encoded in stream:
            parts = msgspec.msgpack.decode(encoded, type=list[msgspec.Raw])
            kind = msgspec.msgpack.decode(parts[0], type=str)
            if kind == "complete":
                if len(parts) != 1 or next(stream, None) is not None:
                    raise ValueError("unexpected data after completed statistics")
                return
            if kinds is None or kind in kinds:
                yield kind, parts
        raise ValueError("statistics scan is incomplete")


def _col(vals: list) -> np.ndarray:
    """Native numeric columns when representable, exact objects otherwise."""
    if vals and isinstance(vals[0], float):
        return np.asarray(vals, dtype=np.float64)
    max_int = np.iinfo(np.int64).max
    if vals and type(vals[0]) is int and all(0 <= val <= max_int for val in vals):
        return np.asarray(vals, dtype=np.int64)
    return np.asarray(vals, dtype=object)


def metric_cols(
    path: Path,
    kind: Literal["theorem", "global_reuse", "stral", "freqs"],
    scope: Literal["state", "expression"] | None,
    cols: tuple[str, ...],
    *,
    size: int = 10_000,
) -> Iterator[dict[str, np.ndarray]]:
    """Decode only requested columns, preserving arbitrary-precision raw counts.

    Raw slices leave unrelated expressions, strings, and constructor maps encoded.
    Each theorem is released before the next archive record is read.
    """
    if (kind in {"stral", "freqs"}) != (scope is None):
        raise ValueError("structural/frequency measurements have no state/expression scope")
    cls = (
        StralRow
        if kind == "stral"
        else FreqRow
        if kind == "freqs"
        else ReuseRow
        if kind == "global_reuse"
        else StateRow
        if scope == "state"
        else ExprRow
    )
    fields_ = row_fields(cls)
    types = tuple(field.type for field in msgspec.structs.fields(cls))
    slots = tuple(fields_.index(name) for name in cols)
    # Decode selected values together, but keep unrelated fields opaque. Any is
    # intentional: typed int decoding rejects our natural-number extension.
    # Batch conversion below applies the same schema checks to decoded values.
    row_type = GenericAlias(tuple, tuple(Any if slot in slots else msgspec.Raw for slot in range(len(fields_))))
    row_decoder = msgspec.msgpack.Decoder(type=row_type, ext_hook=decode_nat_ext)
    selections = tuple((name, slot, GenericAlias(list, types[slot])) for name, slot in zip(cols, slots, strict=True))
    with closing(selected_records(path, {kind})) as batches:
        for _, parts in batches:
            data = parts[1 if kind in {"stral", "freqs"} else 2 if scope == "state" else 3]
            rows = msgspec.msgpack.decode(data, type=list[msgspec.Raw])
            for chunk in batched(rows, size):
                decoded = [row_decoder.decode(encoded) for encoded in chunk]
                vals: dict[str, list] = {
                    name: msgspec.convert([row[slot] for row in decoded], type=col_type)
                    for name, slot, col_type in selections
                }
                for name, slot, _ in selections:
                    if types[slot] is int and any(val < 0 for val in vals[name]):
                        raise ValueError(f"metric counts must be nonnegative: {name}")
                # Raw integer columns remain exact even when different batches span
                # signed/unsigned ranges; concatenation must not promote them to float.
                yield {name: _col(col) for name, col in vals.items()}


def read_stats(path: Path, kinds: set[str] | None = None) -> Generator[StatsRecord]:
    with closing(selected_records(path, kinds)) as stream:
        for _, parts in stream:
            yield _decode_stats([decode_record(part) for part in parts])


def metric_summary(path: Path) -> StateSummary:
    result = None
    with closing(selected_records(path, {"stats"})) as batches:
        for _, parts in batches:
            result = msgspec.convert(decode_record(parts[1]), type=StateSummary)
    if result is None:
        raise ValueError("no statistics found")
    return result


def mdata_vals(path: Path, cols: Mapping[str, type]) -> dict[str, Any]:
    """One selected-mdata pass; unrelated grids and records stay encoded."""
    result = None
    with closing(selected_records(path, {"mdata"})) as batches:
        for _, parts in batches:
            fields_ = msgspec.msgpack.decode(parts[1], type=dict[str, msgspec.Raw])
            result = {
                name: decode_array(decode_record(fields_[name]))
                if cls is np.ndarray
                else msgspec.convert(decode_record(fields_[name]), type=cls)
                for name, cls in cols.items()
            }
    if result is None:
        raise ValueError("no mdata found")
    return result


def mdata_val[T](path: Path, field: str, cls: type[T]) -> T:
    return cast(T, mdata_vals(path, {field: cls})[field])


def freq_batches(path: Path, size: int = 10_000) -> Iterator[list[tuple[int, int, list[int]]]]:
    """Only the chosen aggregate field is decoded, in bounded plotting batches."""
    with closing(selected_records(path, {"stats"})) as batches:
        for _, parts in batches:
            fields_ = msgspec.msgpack.decode(parts[2], type=dict[str, msgspec.Raw])
            rows = msgspec.msgpack.decode(fields_["pairs"], type=list[msgspec.Raw])
            for chunk in batched(rows, size):
                yield [msgspec.convert(decode_record(row), type=tuple[int, int, list[int]]) for row in chunk]


def freq_coverage(path: Path) -> dict:
    result = None
    with closing(selected_records(path, {"stats"})) as batches:
        for _, parts in batches:
            fields_ = msgspec.msgpack.decode(parts[2], type=dict[str, msgspec.Raw])
            result = decode_record(fields_["coverage"])
    if result is None:
        raise ValueError("no frequency coverage found")
    return result


# Topology and comparison archives.


def write_shapes(path: Path, mdata: Mapping[str, str | int | None], counts: Mapping[bytes, ShapeCount]) -> None:
    with writer(path) as write:
        write(mdata)
        write_batches(
            write, ((digest, c.nodes, c.top, c.all, c.theorem, c.root, c.bands) for digest, c in counts.items())
        )


def shape_batches(path: Path) -> Iterator[list[tuple[bytes, ShapeCount]]]:
    stream = records(path)
    with closing(stream) as batches:
        msgspec.convert(decode_record(next(stream)), type=dict[str, str | int | None])
        for encoded in batches:
            rows = msgspec.convert(decode_record(encoded), type=list[StoredShape])
            yield [
                (digest, ShapeCount(size, top, all_, name, root, bands))
                for digest, size, top, all_, name, root, bands in rows
            ]


def shape_cols(path: Path, count: Literal["top", "all"]) -> Iterator[dict[str, np.ndarray]]:
    """Population plots need only node/occurrence columns, not names or bands."""
    stream = records(path)
    with closing(stream) as batches:
        msgspec.convert(decode_record(next(stream)), type=dict[str, str | int | None])
        for encoded in batches:
            rows = msgspec.msgpack.decode(encoded, type=list[list[msgspec.Raw]])
            nodes, counts = [], []
            for row in rows:
                occs = decode_record(row[2 if count == "top" else 3])
                if occs > 0:
                    nodes.append(decode_record(row[1]))
                    counts.append(occs)
            yield {"nodes": _col(nodes), "occurrences": _col(counts)}


def read_shapes(path: Path) -> tuple[dict[str, str | int | None], dict[bytes, ShapeCount]]:
    with closing(records(path)) as stream:
        mdata = msgspec.convert(decode_record(next(stream)), type=dict[str, str | int | None])
    return mdata, dict(item for batch in shape_batches(path) for item in batch)


def write_pairs(path: Path, pairs: Mapping[ViewMode, Mapping[ComparisonLvl, Counter[Pair]]]) -> None:
    with writer(path) as write:
        for mode, lvls in pairs.items():
            for lvl, counts in lvls.items():
                for batch in batched(counts.items(), 10_000):
                    write((mode, lvl, [((row_vals(a), row_vals(b)), weight) for (a, b), weight in batch]))


def _pair_batches(
    path: Path, *, modes: set[ViewMode] | None = None, lvls: set[ComparisonLvl] | None = None
) -> Generator[tuple[ViewMode, ComparisonLvl, list[StoredPair]]]:
    with closing(records(path)) as batches:
        for encoded in batches:
            parts = msgspec.msgpack.decode(encoded, type=list[msgspec.Raw])
            mode = msgspec.convert(decode_record(parts[0]), type=ViewMode)
            lvl = msgspec.convert(decode_record(parts[1]), type=ComparisonLvl)
            if (modes is not None and mode not in modes) or (lvls is not None and lvl not in lvls):
                continue
            rows = msgspec.convert(decode_record(parts[2]), type=list[StoredPair])
            for (a, b), weight in rows:
                if len(a) != len(SIZE_FIELDS) or len(b) != len(SIZE_FIELDS):
                    raise ValueError("invalid graph-size row length")
                if weight < 0 or any(val < 0 for val in (*a, *b)):
                    raise ValueError("comparison counts must be nonnegative")
            yield mode, lvl, rows


def pair_batches(
    path: Path, *, modes: set[ViewMode] | None = None, lvls: set[ComparisonLvl] | None = None
) -> Iterator[PairBatch]:
    with closing(_pair_batches(path, modes=modes, lvls=lvls)) as batches:
        for mode, lvl, rows in batches:
            batch: list[PairRow] = []
            for (a, b), weight in rows:
                batch.append(((GraphSize(*a), GraphSize(*b)), weight))
            yield mode, lvl, batch


def _size_cols(rows: list[StoredPair], idx: int) -> SizeCols:
    return SizeCols(
        *(
            np.asarray([pair[idx][col] for pair, _ in rows], dtype=object)
            if name == "expanded"
            else _col([pair[idx][col] for pair, _ in rows])
            for col, name in enumerate(SIZE_FIELDS)
        )
    )


def pair_col_batches(
    path: Path, *, modes: set[ViewMode], lvls: set[ComparisonLvl]
) -> Iterator[tuple[ViewMode, ComparisonLvl, PairCols]]:
    """Decode native plotting cols without allocating GraphSize records."""
    with closing(_pair_batches(path, modes=modes, lvls=lvls)) as batches:
        for mode, lvl, rows in batches:
            yield (
                mode,
                lvl,
                PairCols(
                    _size_cols(rows, 0),
                    _size_cols(rows, 1),
                    np.asarray([weight for _, weight in rows], dtype=np.float64),
                ),
            )


def read_pairs(path: Path) -> dict[ViewMode, dict[ComparisonLvl, Counter[Pair]]]:
    result: dict[ViewMode, dict[ComparisonLvl, Counter[Pair]]] = {}
    for mode, lvl, batch in pair_batches(path):
        result.setdefault(mode, {}).setdefault(lvl, Counter()).update(dict(batch))
    return result


# Pattern archives and column readers.


def write_patterns(
    path: Path,
    counts: PatternCounts,
    mdata: Mapping[str, object],
    timings: TimingLog | None = None,
    *,
    report_sizes: bool = True,
) -> None:
    write_pattern_batches(
        path, list(counts.heads), counts.batches(timings, path.name), mdata, timings=timings, report_sizes=report_sizes
    )


def write_pattern_batches(
    path: Path,
    heads: list[str],
    batches: Iterable[PatternBatch],
    mdata: Mapping[str, object],
    *,
    with_cols: bool = False,
    timings: TimingLog | None = None,
    report_sizes: bool = True,
) -> np.ndarray | None:
    """Persist native buffers directly; only exceptional weights are boxed.

    Keys use PATTERN_KEY_DTYPE and ordinary weights use little-endian uint64.
    The tuple's weight member is a binary buffer or arbitrary-natural rows.
    No tuple/integers per ordinary row are created for encoding or reading.
    """
    chunks: list[np.ndarray] = []
    with writer(path, timings=timings, report_sizes=report_sizes) as write:
        write({**mdata, "heads": heads})
        for batch in timed_batches(timings, path.name, "pattern.produce_batch", batches):
            weights = batch.weights
            data = weights.tolist() if weights.dtype == object else weights.astype("<u8", copy=False).tobytes()
            write((batch.keys.tobytes(), data))
            if with_cols:
                chunks.append(timed(timings, path.name, "pattern.cols", _pattern_cols, batch))
    return timed(timings, path.name, "pattern.concat_cols", _concat_pattern_cols, chunks) if with_cols else None


def pattern_heads(path: Path) -> list[str]:
    stream = records(path)
    try:
        return msgspec.convert(decode_record(next(stream))["heads"], type=list[str])
    finally:
        stream.close()


def packed_pattern_batches(path: Path, timings: TimingLog | None = None) -> Generator[tuple[list[str], PatternBatch]]:
    with closing(records(path)) as source:
        stream = timed_batches(timings, path.name, "archive.read", source)
        heads = msgspec.convert(decode_record(next(stream))["heads"], type=list[str])
        empty = PatternBatch(np.empty(0, dtype=PATTERN_KEY_DTYPE), np.empty((0, 4), dtype=np.uint64), tuple(heads))
        yield heads, empty  # Preserve the header even for an empty count stream.
        for encoded in stream:
            with checkpoint(timings, path.name, "archive.decode_validate"):
                data, vals = msgspec.convert(decode_record(encoded), type=tuple[bytes, bytes | list[PatternWeights]])
                if len(data) % PATTERN_KEY_DTYPE.itemsize:
                    raise ValueError("invalid local-pattern key buffer length")
                keys = np.frombuffer(data, dtype=PATTERN_KEY_DTYPE)
                if np.any(keys["flavour"] > 1) or np.any(keys["head"] >= len(heads)):
                    raise ValueError("invalid local-pattern key or head reference")
                if isinstance(vals, bytes):
                    if len(vals) != len(keys) * 4 * 8:
                        raise ValueError("invalid local-pattern weight buffer length")
                    weights = np.frombuffer(vals, dtype="<u8").reshape(-1, 4)
                else:
                    weights = np.asarray(vals, dtype=object).reshape(-1, 4)
                    if len(weights) != len(keys) or np.any(weights < 0):
                        raise ValueError("invalid local-pattern counts")
            yield heads, PatternBatch(keys, weights, tuple(heads))


def pattern_batches(path: Path) -> Generator[list[PatternRow]]:
    with closing(packed_pattern_batches(path)) as stream:
        for heads, batch in stream:
            yield [
                (
                    key[:PATTERN_FLAVOUR_OFFSET],
                    key[PATTERN_FLAVOUR_OFFSET],
                    heads[int.from_bytes(key[PATTERN_HEAD_OFFSET:], "little")],
                    weights,
                )
                for key, weights in zip(
                    batch.keys.view(PATTERN_BYTES_DTYPE).tolist(), map(tuple, batch.weights.tolist()), strict=True
                )
            ]


def _pattern_cols(batch: PatternBatch) -> np.ndarray:
    if not len(batch):
        return np.empty((0, len(PatternCol)), dtype=np.uint64)
    return np.column_stack((batch.keys["flavour"], batch.keys["head"], batch.weights))


def _concat_pattern_cols(chunks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(chunks) if chunks else np.empty((0, len(PatternCol)), dtype=np.uint64)


def read_pattern_cols(path: Path) -> tuple[list[str], np.ndarray]:
    chunks: list[np.ndarray] = []
    heads: list[str] = []
    with closing(packed_pattern_batches(path)) as batches:
        for heads, rows in batches:
            chunks.append(_pattern_cols(rows))
    return heads, _concat_pattern_cols(chunks)


def write_pattern_stats(path: Path, rows: Iterable[tuple[ViewMode, int, PatternStats]]) -> None:
    with writer(path) as write:
        for mode, depth, stats in rows:
            write((mode, depth, stats))


def read_pattern_stats(path: Path, *, include_points: bool = True) -> Iterator[tuple[ViewMode, int, PatternStats]]:
    with closing(records(path)) as batches:
        for encoded in batches:
            if include_points:
                yield msgspec.convert(decode_record(encoded), type=tuple[ViewMode, int, PatternStats])
                continue
            parts = msgspec.msgpack.decode(encoded, type=list[msgspec.Raw])
            fields_ = msgspec.msgpack.decode(parts[2], type=dict[str, msgspec.Raw])
            # Preserve the point-series keys for orchestration, not their populations.
            flavours = msgspec.msgpack.decode(fields_["points"], type=dict[int, msgspec.Raw])
            stats = {name: decode_record(val) for name, val in fields_.items() if name != "points"}
            stats["points"] = {flavour: [] for flavour in flavours}
            yield msgspec.convert(
                (decode_record(parts[0]), decode_record(parts[1]), stats), type=tuple[ViewMode, int, PatternStats]
            )


def pattern_points(path: Path, mode: ViewMode, depth: int, flavour: int) -> Iterator[np.ndarray]:
    with closing(records(path)) as batches:
        for encoded in batches:
            parts = msgspec.msgpack.decode(encoded, type=list[msgspec.Raw])
            if decode_record(parts[0]) != mode or decode_record(parts[1]) != depth:
                continue
            fields_ = msgspec.msgpack.decode(parts[2], type=dict[str, msgspec.Raw])
            points = msgspec.msgpack.decode(fields_["points"], type=dict[int, msgspec.Raw])
            if flavour not in points:
                continue
            rows = msgspec.msgpack.decode(points[flavour], type=list[msgspec.Raw])
            for chunk in batched(rows, 10_000):
                vals = [decode_record(row) for row in chunk]
                yield np.column_stack((_col([row[0] for row in vals]), _col([row[1] for row in vals])))


# Examples and human-facing summaries.


def write_examples(path: Path, dicts: Dicts | None, rows: Iterable[tuple[str, bytes]]) -> None:
    with writer(path) as write:
        write(dicts)
        for row in rows:
            write(row)


@contextmanager
def read_examples(path: Path) -> Generator[tuple[Dicts | None, Iterator[tuple[str, bytes]]]]:
    with closing(records(path)) as stream:
        dicts = msgspec.convert(decode_record(next(stream)), type=Dicts | None)
        yield dicts, (msgspec.convert(decode_record(row), type=tuple[str, bytes]) for row in stream)


def json_counts(val: object) -> object:
    """Presentation only: exceptional naturals use a lossless hexadecimal string.

    Binary archives always retain integers. Avoid changing the interpreter's
    decimal-conversion guard just to produce a human-facing JSON summary.
    """
    match val:
        case bool():
            return val
        case int() as number:
            limit = sys.get_int_max_str_digits()
            return hex(number) if limit and number.bit_length() > limit * 3 else number
        case Mapping():
            return {key: json_counts(item) for key, item in val.items()}
        case list() | tuple():
            return [json_counts(item) for item in val]
        case _:
            return val
