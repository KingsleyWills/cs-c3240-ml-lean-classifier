"""Compressed preprocessing archives: framing, codecs and atomic publication.

Readers do not query the source DB or start workers. All producers share the
same framing and source-preserving publication boundary; each archive keeps
its existing header, records and completion contract.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable, Iterator
from compression import zstd
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from struct import Struct
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Literal

import msgspec
import numpy as np
from scipy.sparse import csr_array

from trustmebro.artifact_sizes import report_file_sizes
from trustmebro.extraction import records as r
from trustmebro.extraction.storage import decode_nat_ext, encode_msgpack
from trustmebro.framing import read_frames as read_binary_frames
from trustmebro.framing import write_frame as write_binary_frame

from .candidates import Candidates, Selection
from .coverage import CoverageCandidates, CoverageIndex, IntArray, PackedShapes
from .features import FeatureRows, Vocabulary, compile_vocabulary
from .inventory import BASE_COLS, COUNT_DTYPE, DEPTH_COLS, MAX_COUNT, Inventory, ShapeCounts, Summary

_LENGTH = Struct("!Q")
ARRAY_DTYPE = np.dtype("<i8")
REAL_DTYPE = np.dtype("<f8")
BLOCK_ROWS = 4096
COVERAGE_CHUNK_BYTES = 2**20
COVERAGE_FRAME_BYTES = 64 * 2**20  # provenance can contain an explicit theorem selection
type CoverageCol = Literal[
    "idents", "shape_data", "shape_offsets", "nodes", "offsets", "refs", "goals", "hyps", "support", "indptr", "indices"
]
_COVERAGE_COLS: tuple[CoverageCol, ...] = (
    "idents",
    "shape_data",
    "shape_offsets",
    "nodes",
    "offsets",
    "refs",
    "goals",
    "hyps",
    "support",
    "indptr",
    "indices",
)
type PackedRows = tuple[str, bytes, tuple[r.Tactic, ...], bytes, bytes, bytes]


class Block(msgspec.Struct, frozen=True, array_like=True):
    shapes: tuple[tuple[bytes, bytes], ...]  # SHA-256 identity, packed canonical adjacency
    counts: bytes  # little-endian uint64, row-major
    large: tuple[tuple[int, int, int], ...]  # block-local row, column, exact count


class _CandidateName(msgspec.Struct, frozen=True, array_like=True):
    name: str  # trailing candidate fields are skipped, not reconstructed


class FeatureHeader(msgspec.Struct, frozen=True):
    vocab: Vocabulary
    selection: Selection
    src: str
    src_size: int
    src_modified_ns: int
    label_policy: bytes | None = None  # frozen JSON; no external policy needed for new training archives
    split_id: str | None = None  # binds a development archive to its logical split manifest


@dataclass(frozen=True, slots=True)
class FeatureStream:
    """Header and decoded rows owned by an open archive context."""

    header: FeatureHeader
    width: int
    rows: Iterator[FeatureRows]


class Footer(msgspec.Struct, frozen=True):
    theorems: r.Nat
    states: r.Nat


class Diagnostics(msgspec.Struct, frozen=True):
    header: FeatureHeader
    theorems: r.Nat
    covered_theorems: r.Nat
    states: r.Nat
    covered_states: r.Nat
    goal_covered: r.Nat
    hyp_covered: r.Nat
    nnz: r.Nat
    active_dims: tuple[tuple[r.Nat, r.PosNat], ...]
    active_patterns: tuple[tuple[r.Nat, r.PosNat], ...]
    support: bytes  # role-specific state support, int64
    pair_limit: r.Nat
    pairs: bytes  # role-unioned state co-occurrence, int64 square


class CoverageHeader(msgspec.Struct, frozen=True):
    selection: Selection
    depths: tuple[int, ...]
    src: str
    src_size: int
    src_modified_ns: int
    format: Literal["packed-coverage"]


class CoverageBlock(msgspec.Struct, frozen=True, array_like=True, tag="buffer"):
    col: CoverageCol
    data: bytes


class CoverageNames(msgspec.Struct, frozen=True, array_like=True, tag="names"):
    names: tuple[str, ...]


class CoverageEnd(msgspec.Struct, frozen=True, array_like=True, tag="end"):
    theorems: int
    roots: int
    shapes: int
    memberships: int


def write_frame(stream: BinaryIO, data: bytes) -> int:
    return write_binary_frame(stream, data, _LENGTH)


def read_frames(path: Path, *, max_frame_bytes: int | None = None) -> Iterator[bytes]:
    """Shared archive framing; each producer defines its own header/records."""
    with zstd.open(path, "rb") as stream:
        yield from read_binary_frames(stream, _LENGTH, max_frame_bytes=max_frame_bytes)


def publish(path: Path, frames: Iterable[bytes], *, sources: tuple[Path, ...], replace: bool) -> int:
    """An artifact commits only after its producer and immutable-source checks succeed."""
    for src in sources:
        if path.resolve() == src.resolve() or (path.exists() and src.exists() and path.samefile(src)):
            raise ValueError("output cannot replace a source artifact")
    if path.exists() and not replace:
        raise FileExistsError(f"output exists: {path}; use --replace")
    stamps = [(src.stat().st_size, src.stat().st_mtime_ns) for src in sources]
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".archive-", delete=False) as tmp:
        pending = Path(tmp.name)
    try:
        raw_bytes = 0
        with zstd.open(pending, "wb") as stream:
            for data in frames:
                raw_bytes += write_frame(stream, data)
        if [(src.stat().st_size, src.stat().st_mtime_ns) for src in sources] != stamps:
            raise ValueError("source changed during archive publication")
        pending.replace(path)
        report_file_sizes(path, raw_bytes)
        return raw_bytes
    finally:
        pending.unlink(missing_ok=True)


def first_frame(frames: Iterator[bytes]) -> bytes:
    data = next(frames, None)
    if data is None:
        raise ValueError("archive lacks its header")
    return data


def decode_candidates(data: bytes, decoder: msgspec.msgpack.Decoder) -> Candidates:
    """Keep extension-aware natural decoding identical for readers and workers."""
    return msgspec.convert(decoder.decode(data), type=Candidates, strict=True)


def candidate_frames(path: Path, names: tuple[str, ...] | None = None) -> Iterator[bytes]:
    """Select by name before payload decoding; retain archive order and scope checks."""
    wanted = None if names is None else set(names)
    if names is not None and (not wanted or len(wanted) != len(names)):
        raise ValueError("candidate selection must be nonempty and distinct")
    seen: set[str] = set()
    name_decoder = msgspec.msgpack.Decoder(type=_CandidateName)
    with closing(read_frames(path)) as frames:
        header = next(frames, None)
        if header is None:
            raise ValueError("candidate archive lacks its selection header")
        msgspec.convert(msgspec.msgpack.decode(header, ext_hook=decode_nat_ext), type=Selection, strict=True)
        for data in frames:
            if wanted is not None:
                name = name_decoder.decode(data).name
                if name not in wanted:
                    continue
                if name in seen:
                    raise ValueError(f"duplicate candidate theorem: {name!r}")
                seen.add(name)
            yield data
    if wanted is not None and seen != wanted:
        raise ValueError("candidate archive lacks requested training theorems")


def read_candidates(path: Path, *, names: tuple[str, ...] | None = None) -> Iterator[Candidates]:
    """Decode selected theorems fully, retaining arbitrary-size natural literals."""
    decoder = msgspec.msgpack.Decoder(ext_hook=decode_nat_ext)
    with closing(candidate_frames(path, names)) as frames:
        for data in frames:
            yield decode_candidates(data, decoder)


def read_coverage_candidates(path: Path, *, names: tuple[str, ...] | None = None) -> Iterator[CoverageCandidates]:
    """Typed native projection: references/memberships only, not feature validation."""
    decoder = msgspec.msgpack.Decoder(type=CoverageCandidates)
    with closing(candidate_frames(path, names)) as frames:
        for data in frames:
            yield decoder.decode(data)


def read_selection(path: Path) -> Selection:
    with closing(read_frames(path)) as frames:
        try:
            return msgspec.msgpack.decode(next(frames), type=Selection)
        except StopIteration as error:
            raise ValueError("candidate archive lacks its selection header") from error


def read_vocabulary(path: Path) -> Vocabulary:
    with closing(read_frames(path)) as frames:
        vocab = msgspec.msgpack.decode(first_frame(frames), type=Vocabulary)
        if next(frames, None) is not None:
            raise ValueError("unexpected trailing vocabulary frames")
    return vocab


def pack_rows(rows: FeatureRows) -> bytes:
    matrix = rows.matrix
    steps, indices, indptr = (
        array.astype(ARRAY_DTYPE, copy=False).tobytes() for array in (rows.steps, matrix.indices, matrix.indptr)
    )
    dtype = REAL_DTYPE if matrix.dtype.kind == "f" else ARRAY_DTYPE
    data = matrix.data.astype(dtype, copy=False).tobytes()
    return encode_msgpack((rows.name, steps, rows.tactics, data, indices, indptr))


def unpack_rows(data: bytes, width: int, *, real: bool = False) -> FeatureRows:
    name, steps, tactics, vals, indices, indptr = msgspec.msgpack.decode(data, type=PackedRows)
    step_array, col_array, row_array = (np.frombuffer(blob, dtype=ARRAY_DTYPE) for blob in (steps, indices, indptr))
    counts = np.frombuffer(vals, dtype=REAL_DTYPE if real else ARRAY_DTYPE)
    if (
        len(tactics) != len(step_array)
        or len(row_array) != len(step_array) + 1
        or len(counts) != len(col_array)
        or row_array[0] != 0
        or row_array[-1] != len(counts)
        or np.any(np.diff(row_array) < 0)
        or np.any(step_array < 0)
        or len(np.unique(step_array)) != len(step_array)
        or np.any(counts <= 0)
        or np.any(~np.isfinite(counts))
        or np.any(col_array < 0)
        or np.any(col_array >= width)
    ):
        raise ValueError("invalid sparse feature row buffers/provenance")
    matrix = csr_array((counts, col_array, row_array), shape=(len(step_array), width))
    if not matrix.has_canonical_format:
        raise ValueError("feature rows must have sorted, distinct column indices")
    return FeatureRows(name, step_array, tactics, matrix)


def read_feature_header(path: Path) -> FeatureHeader:
    with closing(read_frames(path)) as frames:
        return msgspec.msgpack.decode(first_frame(frames), type=FeatureHeader)


@contextmanager
def open_features(path: Path, *, max_frame_bytes: int | None = None) -> Generator[FeatureStream]:
    """Decode the header once; closing the context closes the compressed input.

    Frame limits apply before allocating encoded records. Consuming all rows
    also checks the footer; stopping early does not certify archive completion.
    """
    with closing(read_frames(path, max_frame_bytes=max_frame_bytes)) as frames:
        header = msgspec.msgpack.decode(first_frame(frames), type=FeatureHeader)
        width = compile_vocabulary(header.vocab).width
        with closing(_feature_records(frames, width, real=header.vocab.representation is not None)) as rows:
            yield FeatureStream(header, width, rows)


def _feature_records(frames: Iterator[bytes], width: int, *, real: bool) -> Iterator[FeatureRows]:
    states = 0
    for theorems, data in enumerate(frames):
        if not data:
            raise ValueError("empty feature row frame")
        # Arrays contain rows; maps contain the completion footer. Dispatch
        # does not generically decode/copy embedded CSR buffers twice.
        if data[0] & 0xF0 == 0x80 or data[0] in (0xDE, 0xDF):
            footer = msgspec.msgpack.decode(data, type=Footer)
            if (footer.theorems, footer.states) != (theorems, states) or next(frames, None) is not None:
                raise ValueError("feature completion counters/trailing frames disagree")
            return
        rows = unpack_rows(data, width, real=real)
        states += rows.matrix.shape[0]
        yield rows
    raise ValueError("feature archive lacks its completion footer")


def read_features(path: Path, *, max_frame_bytes: int | None = None) -> Iterator[FeatureRows]:
    """One theorem matrix at a time; validate the final completion counters."""
    with open_features(path, max_frame_bytes=max_frame_bytes) as source:
        yield from source.rows


def read_diagnostics(path: Path) -> Diagnostics:
    with closing(read_frames(path)) as frames:
        report = msgspec.msgpack.decode(first_frame(frames), type=Diagnostics)
        if next(frames, None) is not None:
            raise ValueError("unexpected trailing diagnostics frames")
    width = len(report.header.vocab.entries) * 2
    if (
        report.pair_limit > min(2048, width // 2)
        or len(report.support) != width * 8
        or len(report.pairs) != report.pair_limit**2 * 8
    ):
        raise ValueError("diagnostic array dimensions disagree with vocabulary/prefix")
    if (
        report.covered_theorems > report.theorems
        or report.covered_states > report.states
        or max(report.goal_covered, report.hyp_covered) > report.covered_states
        or any(
            sum(count for _, count in hgram) != report.states for hgram in (report.active_dims, report.active_patterns)
        )
        or sum(val * count for val, count in report.active_dims) != report.nnz
    ):
        raise ValueError("diagnostic coverage/histogram totals disagree")
    return report


def write_inventory(path: Path, summary: Summary, counts: ShapeCounts, *, src: Path, replace: bool) -> None:
    """Serialize coordinated count buffers with exact overflow corrections."""

    def frames() -> Iterator[bytes]:
        yield encode_msgpack(summary)
        large_rows: dict[int, list[tuple[int, int, int]]] = {}
        for (row, col), val in counts.large.items():
            large_rows.setdefault(row // BLOCK_ROWS, []).append((row % BLOCK_ROWS, col, val))
        for start in range(0, summary.shapes, BLOCK_ROWS):
            end = min(start + BLOCK_ROWS, summary.shapes)
            yield encode_msgpack(
                Block(
                    tuple(counts.shapes[start:end]),
                    counts.counts[start:end].tobytes(),
                    tuple(large_rows.get(start // BLOCK_ROWS, ())),
                )
            )

    publish(path, frames(), sources=(src,), replace=replace)


def read_inventory(path: Path) -> Inventory:
    decoder = msgspec.msgpack.Decoder(ext_hook=decode_nat_ext)
    with closing(read_frames(path)) as frames:
        try:
            summary = msgspec.convert(decoder.decode(next(frames)), type=Summary)
        except StopIteration as error:
            raise ValueError("inventory lacks its summary") from error
        width = len(BASE_COLS) + len(DEPTH_COLS) * len(summary.selection.depths)
        counts = np.empty((summary.shapes, width), dtype=COUNT_DTYPE)
        shapes: list[tuple[bytes, bytes]] = []
        large: dict[tuple[int, int], int] = {}
        for data in frames:
            block = msgspec.convert(decoder.decode(data), type=Block)
            start, end = len(shapes), len(shapes) + len(block.shapes)
            if end > summary.shapes or len(block.counts) != len(block.shapes) * width * COUNT_DTYPE.itemsize:
                raise ValueError("invalid inventory count block size")
            counts[start:end] = np.frombuffer(block.counts, dtype=COUNT_DTYPE).reshape(-1, width)
            for row, col, val in block.large:
                if not (0 <= row < len(block.shapes) and 0 <= col < width and val > MAX_COUNT):
                    raise ValueError("invalid inventory exact-count correction")
                large[start + row, col] = val
            shapes.extend(block.shapes)
        if len(shapes) != summary.shapes:
            raise ValueError("inventory shape count disagrees with its summary")
    return Inventory(summary, shapes, counts, large)


def write_coverage(path: Path, header: CoverageHeader, index: CoverageIndex, *, src: Path, replace: bool) -> int:
    """Encode packed shapes and numerical buffers in bounded frames, not one copy of the index."""

    def frames() -> Iterator[bytes]:
        yield encode_msgpack(header)
        for col, vals in _coverage_buffers(index).items():
            # Cast numerical slices, not a whole CSR index, if the source dtype
            # differs. Raw adjacency/identity buffers are sliced as bytes.
            chunk_size = COVERAGE_CHUNK_BYTES // 8 if isinstance(vals, np.ndarray) else COVERAGE_CHUNK_BYTES
            for start in range(0, max(1, len(vals)), chunk_size):
                chunk = vals[start : start + chunk_size]
                if isinstance(chunk, np.ndarray):
                    chunk = memoryview(chunk.astype(ARRAY_DTYPE, copy=False)).cast("B")
                yield encode_msgpack(CoverageBlock(col, chunk.tobytes()))
        start = 0
        while start < len(index.names):
            end = min(start + BLOCK_ROWS, len(index.names))
            while len(data := encode_msgpack(CoverageNames(index.names[start:end]))) > COVERAGE_CHUNK_BYTES:
                if end == start + 1:
                    raise ValueError("coverage theorem name exceeds the name-frame limit")
                end = start + (end - start) // 2
            yield data
            start = end
        yield encode_msgpack(CoverageEnd(len(index.names), len(index.refs), len(index.support), index.matches.nnz))

    def bounded_frames() -> Iterator[bytes]:
        for data in frames():
            if len(data) > COVERAGE_FRAME_BYTES:
                raise ValueError("coverage metadata exceeds the frame-size limit")
            yield data

    return publish(path, bounded_frames(), sources=(src,), replace=replace)


def _coverage_buffers(index: CoverageIndex) -> dict[CoverageCol, memoryview | IntArray]:
    buffers: dict[CoverageCol, memoryview | IntArray] = {
        "idents": memoryview(index.shapes.idents).cast("B"),
        "shape_data": index.shapes.data,
    }
    arrays: tuple[tuple[CoverageCol, IntArray], ...] = (
        ("shape_offsets", index.shapes.offsets),
        ("nodes", index.shapes.nodes),
        ("offsets", index.offsets),
        ("refs", index.refs),
        ("goals", index.goals),
        ("hyps", index.hyps),
        ("support", index.support),
        ("indptr", index.matches.indptr),
        ("indices", index.matches.indices),
    )
    for col, vals in arrays:
        buffers[col] = vals
    return buffers


def _check_shape_counts(shapes: PackedShapes) -> None:
    """Validate cost metadata from MessagePack array headers, without decoding trees.

    Arrays use fixarray, array16 or array32 lengths. NumPy gathers only those
    prefix bytes in batches; scratch does not scale with the whole shape table.
    Full adjacency/identity validation remains at selected-entry decoding.
    """
    raw = np.frombuffer(shapes.data, dtype=np.uint8)
    for start in range(0, len(shapes.nodes), 65536):
        stop = min(start + 65536, len(shapes.nodes))
        starts, ends = shapes.offsets[start:stop], shapes.offsets[start + 1 : stop + 1]
        markers = raw[starts]
        if np.any(~(((markers & 0xF0) == 0x90) | (markers == 0xDC) | (markers == 0xDD))):
            raise ValueError("packed coverage adjacency must be a MessagePack array")
        counts = (markers & 0x0F).astype(np.int64)
        for marker, width in ((0xDC, 2), (0xDD, 4)):
            rows = np.flatnonzero(markers == marker)
            if rows.size:
                if np.any(starts[rows] + width >= ends[rows]):
                    raise ValueError("truncated packed coverage array header")
                positions = starts[rows, None] + np.arange(1, width + 1)
                counts[rows] = raw[positions].astype(np.int64) @ (256 ** np.arange(width - 1, -1, -1))
        if not np.array_equal(counts, shapes.nodes[start:stop]):
            raise ValueError("invalid packed coverage node count")


def read_coverage(path: Path) -> tuple[CoverageHeader, CoverageIndex]:
    """Accumulate compact buffers; frame decoding never reconstructs all shape trees.

    The complete sparse index is still retained. Chunking bounds serialization
    scratch, not total RSS or the subsequent native solver's model copies.
    """
    buffers: dict[CoverageCol, bytearray] = {col: bytearray() for col in _COVERAGE_COLS}
    seen: set[CoverageCol] = set()
    names: list[str] = []
    end: CoverageEnd | None = None
    decoder = msgspec.msgpack.Decoder(type=CoverageBlock | CoverageNames | CoverageEnd)
    with closing(read_frames(path, max_frame_bytes=COVERAGE_FRAME_BYTES)) as frames:
        header = msgspec.msgpack.decode(first_frame(frames), type=CoverageHeader)
        for data in frames:
            record = decoder.decode(data)
            match record:
                case CoverageBlock(col=col, data=data):
                    if len(data) > COVERAGE_CHUNK_BYTES:
                        raise ValueError("coverage buffer chunk exceeds its size limit")
                    buffers[col].extend(data)
                    seen.add(col)
                case CoverageNames(names=batch):
                    names.extend(batch)
                case CoverageEnd():
                    end = record
                    break
        if end is None:
            raise ValueError("coverage archive lacks its completion record")
        if next(frames, None) is not None:
            raise ValueError("unexpected trailing coverage frames")
    if seen != set(_COVERAGE_COLS):
        raise ValueError("coverage archive lacks required buffers")
    idents = np.frombuffer(buffers.pop("idents"), dtype="V32")
    shape_data = memoryview(buffers.pop("shape_data"))
    arrays: dict[CoverageCol, IntArray] = {col: np.frombuffer(data, dtype=ARRAY_DTYPE) for col, data in buffers.items()}
    offsets, refs, goals, hyps, support, indptr, indices = (
        arrays[col] for col in ("offsets", "refs", "goals", "hyps", "support", "indptr", "indices")
    )
    shape_offsets, nodes = arrays["shape_offsets"], arrays["nodes"]
    if (
        not header.depths
        or any(depth < 1 for depth in header.depths)
        or not set(header.depths).issubset(header.selection.depths)
        or not names
        or len(set(names)) != len(names)
        or (end.theorems, end.roots, end.shapes, end.memberships) != (len(names), len(refs), len(idents), len(indices))
        or not len(idents)
        or len(shape_offsets) != len(idents) + 1
        or shape_offsets[0] != 0
        or shape_offsets[-1] != len(shape_data)
        or np.any(np.diff(shape_offsets) <= 0)
        or len(nodes) != len(idents)
        or np.any(nodes <= 0)
        or np.any(nodes > np.diff(shape_offsets))
        or len(offsets) != len(names) + 1
        or offsets[0] != 0
        or offsets[-1] != len(refs)
        or np.any(np.diff(offsets) <= 0)
        or len(goals) != len(refs)
        or len(hyps) != len(refs)
        or len(support) != len(idents)
        or np.any(refs < 0)
        or np.any(goals < 0)
        or np.any(hyps < 0)
        or np.any(support < 0)
        or np.any((goals == 0) & (hyps == 0))
        or len(indptr) != len(refs) + 1
        or indptr[0] != 0
        or indptr[-1] != len(indices)
        or np.any(np.diff(indptr) < 0)
        or (indices.size and (indices.min() < 0 or indices.max() >= len(idents)))
    ):
        raise ValueError("invalid coverage buffers or provenance")
    matches = csr_array(
        (np.ones(len(indices), dtype=bool), indices, indptr), shape=(len(refs), len(idents)), copy=False
    )
    if not matches.has_canonical_format:
        raise ValueError("coverage memberships must have sorted, distinct shape columns")
    shapes = PackedShapes(idents, shape_data, shape_offsets, nodes)
    _check_shape_counts(shapes)
    return header, CoverageIndex(shapes, matches, goals, hyps, tuple(names), offsets, refs, support)
