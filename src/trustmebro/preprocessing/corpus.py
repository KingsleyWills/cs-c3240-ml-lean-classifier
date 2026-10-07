"""Read-only corpus queries and training batches from precomputed feature archives."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import msgspec
import numpy as np
from scipy.sparse import csr_array, vstack

from trustmebro.extraction import records as r
from trustmebro.extraction.storage import (
    THEOREM_COLS,
    BlobCodec,
    check_extraction_schema,
    decode_theorem_row,
    decode_trns,
    load_theorem,
    stored_dicts,
)

from .archives import FeatureStream, open_features
from .features import FeatureRows, Layout, encode_theorem

if TYPE_CHECKING:
    from .partition import SplitManifest


class LabelPolicy(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Exact exported parser-kind mapping, independent of source spelling.

    Treat the mapping as immutable during a query/experiment. Unmapped actions
    are rejected, excluded from learning rows, or assigned an explicit label;
    none of these choices alters the stored theorem or source tactics.
    """

    kinds: dict[str, str]
    unmapped: Literal["error", "drop", "other"]
    other_label: str | None = None

    def __post_init__(self) -> None:
        if self.unmapped not in ("error", "drop", "other"):
            raise ValueError("unmapped must be error, drop or other")
        if any(not isinstance(val, str) or not val.strip() for pair in self.kinds.items() for val in pair):
            raise ValueError("tactic kinds and labels must be nonempty strings")
        if self.unmapped == "other":
            if not isinstance(self.other_label, str) or not self.other_label.strip():
                raise ValueError("unmapped=other requires a nonempty other_label")
        elif self.other_label is not None:
            raise ValueError("other_label is only used with unmapped=other")
        if not self.labels:
            raise ValueError("policy declares no labels")

    @property
    def labels(self) -> tuple[str, ...]:
        labels = set(self.kinds.values())
        if self.other_label is not None:
            labels.add(self.other_label)
        return tuple(sorted(labels))

    @property
    def label_ids(self) -> dict[str, int]:
        """Stable IDs for every declared class, including locally absent ones."""
        return {label: idx for idx, label in enumerate(self.labels)}

    def label(self, tactic: r.Tactic) -> str | None:
        if (label := self.kinds.get(tactic.kind)) is not None:
            return label
        if self.unmapped == "error":
            raise ValueError(f"unmapped tactic kind: {tactic.kind!r}")
        return self.other_label if self.unmapped == "other" else None


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, val in pairs:
        if key in result:
            raise ValueError(f"duplicate label-policy key: {key!r}")
        result[key] = val
    return result


def read_label_policy(path: Path) -> LabelPolicy:
    data = json.loads(path.read_bytes(), object_pairs_hook=_unique_keys)
    return msgspec.convert(data, type=LabelPolicy, strict=True)


class LabeledTrn(NamedTuple):
    theorem: str
    step: int
    label: str
    trn: r.Trn


@dataclass(frozen=True, slots=True)
class StateData:
    """One shared expression table and transitions aligned with feature rows."""

    exprs: tuple[r.Expr, ...]
    trns: tuple[r.Trn, ...]


@dataclass(frozen=True, slots=True)
class LabeledFeatures:
    rows: FeatureRows
    labels: tuple[str, ...]
    label_ids: np.ndarray
    classes: tuple[str, ...]  # label_ids index this policy-wide tuple
    states: StateData | None


class LabeledState(NamedTuple):
    theorem: str
    step: int
    label: str
    label_id: int
    trn: r.Trn
    exprs: tuple[r.Expr, ...]
    features: csr_array  # a single sparse row, never densified

    @property
    def state(self) -> r.ProofState:
        return self.trn.state


@dataclass(frozen=True, slots=True)
class TrainingCfg:
    """Bounds output batches and encoded theorem/header frames, not total RSS.

    float32 is an explicit approximate training representation. The canonical
    archive remains unchanged. Sparse indices use int32 when they fit.
    """

    rows: int = 8192
    nnz: int = 1_000_000
    frame_bytes: int = 64 * 2**20
    dtype: Literal["float32", "float64"] = "float64"


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    matrix: csr_array
    label_ids: np.ndarray
    classes: tuple[str, ...]
    theorems: tuple[str, ...]  # row-aligned; repeated names share their string object
    steps: np.ndarray  # original transition indices, not indices after filtering


DEFAULT_TRAINING_CFG = TrainingCfg()
DEFAULT_MAX_BYTES = 4 * 2**30


def _check_training_cfg(cfg: TrainingCfg) -> None:
    if min(cfg.rows, cfg.nnz, cfg.frame_bytes) < 1 or cfg.dtype not in ("float32", "float64"):
        raise ValueError("training batch/frame limits must be positive and dtype float32 or float64")


def _training_policy(source: FeatureStream, policy: LabelPolicy | None) -> LabelPolicy:
    if source.header.label_policy is not None:
        frozen = msgspec.json.decode(source.header.label_policy, type=LabelPolicy)
        if policy is not None and policy != frozen:
            raise ValueError("label policy differs from the frozen training archive")
        return frozen
    if policy is None:
        raise ValueError("archive has no frozen labels; supply a label policy or convert with --labels")
    return policy


def _index_dtype(width: int, rows: int, nnz: int) -> type[np.int32 | np.int64]:
    return np.int32 if max(width, rows, nnz) <= np.iinfo(np.int32).max else np.int64


def _join_training(parts: list[TrainingBatch]) -> TrainingBatch:
    if len(parts) == 1:
        return parts[0]
    return TrainingBatch(
        vstack([part.matrix for part in parts], format="csr"),
        np.concatenate([part.label_ids for part in parts]),
        parts[0].classes,
        tuple(chain.from_iterable(part.theorems for part in parts)),
        np.concatenate([part.steps for part in parts]),
    )


def _labeled_cached_rows(
    source: FeatureStream, policy: LabelPolicy, names: set[str] | None = None, population: set[str] | None = None
) -> Iterator[TrainingBatch]:
    classes, class_ids = policy.labels, policy.label_ids
    seen: set[str] = set()
    for rows in source.rows:
        if population is not None:
            if rows.name not in population or rows.name in seen:
                raise ValueError("feature theorem population disagrees with split manifest")
            seen.add(rows.name)
        if names is not None and rows.name not in names:
            continue
        ids = np.fromiter((class_ids.get(policy.label(tactic), -1) for tactic in rows.tactics), dtype=np.int64)
        keep = np.flatnonzero(ids >= 0)
        matrix, steps = rows.matrix, rows.steps
        if len(keep) != len(ids):
            matrix, steps, ids = matrix[keep], steps[keep], ids[keep]
        if len(ids):
            yield TrainingBatch(matrix, ids, classes, (rows.name,) * len(ids), steps)
    if population is not None and seen != population:
        raise ValueError("feature archive is missing split theorems")


def _training_batches(
    source: FeatureStream,
    policy: LabelPolicy,
    cfg: TrainingCfg,
    names: set[str] | None = None,
    population: set[str] | None = None,
) -> Iterator[TrainingBatch]:
    parts: list[TrainingBatch] = []
    used_rows = used_nnz = 0
    for rows in _labeled_cached_rows(source, policy, names, population):
        matrix, ids, steps = rows.matrix, rows.label_ids, rows.steps
        start = 0
        while start < len(ids):
            end = min(
                len(ids),
                start + cfg.rows - used_rows,
                int(np.searchsorted(matrix.indptr, matrix.indptr[start] + cfg.nnz - used_nnz, side="right")) - 1,
            )
            if end == start:
                if not parts:
                    raise MemoryError("one feature row exceeds the batch nonzero limit; increase TrainingCfg.nnz")
                yield _join_training(parts)
                parts.clear()
                used_rows = used_nnz = 0
                continue
            chunk = matrix if start == 0 and end == len(ids) else matrix[start:end]
            idx_dtype = _index_dtype(source.width, end - start, chunk.nnz)
            with np.errstate(over="raise", invalid="raise"):
                chunk = csr_array(
                    (
                        chunk.data.astype(cfg.dtype, copy=False),
                        chunk.indices.astype(idx_dtype, copy=False),
                        chunk.indptr.astype(idx_dtype, copy=False),
                    ),
                    shape=chunk.shape,
                )
            parts.append(TrainingBatch(chunk, ids[start:end], rows.classes, rows.theorems[start:end], steps[start:end]))
            used_rows += end - start
            used_nnz += chunk.nnz
            start = end
            if used_rows == cfg.rows or used_nnz == cfg.nnz:
                yield _join_training(parts)
                parts.clear()
                used_rows = used_nnz = 0
    if parts:
        yield _join_training(parts)


def read_training_batches(
    path: Path,
    policy: LabelPolicy | None = None,
    *,
    cfg: TrainingCfg = DEFAULT_TRAINING_CFG,
    split: SplitManifest | None = None,
    subset: Literal["train", "validation"] | None = None,
) -> Iterator[TrainingBatch]:
    """Decompress stored CSR rows; never query a DB or construct graph features.

    Limits bound each encoded frame, output rows and output nonzeros. A frame
    and decoded theorem coexist with accumulated chunks and assembly scratch;
    these are not process-RSS limits. Retaining batches is the caller's choice.
    Labels come from the frozen header unless an older archive needs a policy.
    Consume the iterator completely to verify footer counters; close it if you
    stop early. Batches can cross theorem boundaries; original provenance and
    input selection are preserved. A bound manifest selects a logical role;
    without one all labeled rows are returned. No split or vocabulary is fitted.
    """
    _check_training_cfg(cfg)
    with open_features(path, max_frame_bytes=cfg.frame_bytes) as source:
        names, population = _training_subset(source, split, subset)
        yield from _training_batches(source, _training_policy(source, policy), cfg, names, population)


def _training_subset(
    source: FeatureStream, split: SplitManifest | None, subset: Literal["train", "validation"] | None
) -> tuple[set[str] | None, set[str] | None]:
    if subset not in (None, "train", "validation") or (subset is None) != (split is None):
        raise ValueError("provide both a split manifest and train/validation subset, or neither")
    if split is None:
        return None, None
    from .partition import split_id

    header, fitted = source.header, source.header.vocab.selection
    if (
        header.split_id != split_id(split)
        or fitted is None
        or set(fitted.theorems or ()) != set(split.train)
        or _training_policy(source, None) != split.policy
        or (header.selection.db, header.selection.size, header.selection.modified_ns)
        != (split.selection.db, split.selection.size, split.selection.modified_ns)
    ):
        raise ValueError("split manifest differs from the frozen feature archive")
    return set(split.train if subset == "train" else split.validation), set(split.train) | set(split.validation)


def _training_bytes(width: int, rows: int, nnz: int, cfg: TrainingCfg) -> int:
    idx_size = np.dtype(_index_dtype(width, rows, nnz)).itemsize
    return nnz * (np.dtype(cfg.dtype).itemsize + idx_size) + (rows + 1) * idx_size + 16 * rows


def load_training_matrix(
    path: Path,
    policy: LabelPolicy | None = None,
    *,
    cfg: TrainingCfg = DEFAULT_TRAINING_CFG,
    max_bytes: int = DEFAULT_MAX_BYTES,
    split: SplitManifest | None = None,
    subset: Literal["train", "validation"] | None = None,
) -> TrainingBatch:
    """Two sequential decompression passes, one allocation of the full CSR.

    First count selected rows/nonzeros; then fill exactly allocated buffers.
    This avoids keeping all batch matrices plus a second full concatenation.
    max_bytes limits final numeric buffers (CSR, labels, steps), not total RSS:
    bounded decoder/batch scratch and theorem-name references also remain.
    Requires an immutable, completed archive; it does not rebuild features.
    """
    _check_training_cfg(cfg)
    if max_bytes < 1:
        raise ValueError("training numeric-buffer budget must be positive")
    stamp = path.stat().st_size, path.stat().st_mtime_ns
    rows = nnz = 0
    with open_features(path, max_frame_bytes=cfg.frame_bytes) as source:
        width = source.width
        resolved = _training_policy(source, policy)
        names, population = _training_subset(source, split, subset)
        # Counting needs no concatenation, precision conversion, or batch copies.
        for batch in _labeled_cached_rows(source, resolved, names, population):
            if np.diff(batch.matrix.indptr).max(initial=0) > cfg.nnz:
                raise MemoryError("one feature row exceeds the batch nonzero limit; increase TrainingCfg.nnz")
            rows += len(batch.label_ids)
            nnz += batch.matrix.nnz
            if _training_bytes(width, rows, nnz, cfg) > max_bytes:
                raise MemoryError("full training matrix exceeds its numeric-buffer budget; stream batches instead")
    if (path.stat().st_size, path.stat().st_mtime_ns) != stamp:
        raise ValueError("feature archive changed during counting")
    if _training_bytes(width, rows, nnz, cfg) > max_bytes:
        raise MemoryError("empty training matrix exceeds its numeric-buffer budget")
    idx_dtype = _index_dtype(width, rows, nnz)
    data, indices = np.empty(nnz, dtype=cfg.dtype), np.empty(nnz, dtype=idx_dtype)
    indptr = np.empty(rows + 1, dtype=idx_dtype)
    indptr[0] = 0
    ids, steps = np.empty(rows, dtype=np.int64), np.empty(rows, dtype=np.int64)
    names: list[str] = []
    row_pos = nnz_pos = 0
    with closing(read_training_batches(path, resolved, cfg=cfg, split=split, subset=subset)) as batches:
        for batch in batches:
            end_row, end_nnz = row_pos + len(batch.label_ids), nnz_pos + batch.matrix.nnz
            if end_row > rows or end_nnz > nnz:
                raise ValueError("feature archive counters changed between passes")
            data[nnz_pos:end_nnz], indices[nnz_pos:end_nnz] = batch.matrix.data, batch.matrix.indices
            indptr[row_pos + 1 : end_row + 1] = batch.matrix.indptr[1:] + nnz_pos
            ids[row_pos:end_row], steps[row_pos:end_row] = batch.label_ids, batch.steps
            names.extend(batch.theorems)
            row_pos, nnz_pos = end_row, end_nnz
    if (row_pos, nnz_pos) != (rows, nnz) or (path.stat().st_size, path.stat().st_mtime_ns) != stamp:
        raise ValueError("feature archive changed between loading passes")
    return TrainingBatch(
        csr_array((data, indices, indptr), shape=(rows, width)), ids, resolved.labels, tuple(names), steps
    )


def _feature_batch(
    theorem: r.Theorem,
    layout: Layout,
    policy: LabelPolicy,
    label_ids: dict[str, int],
    *,
    include_states: bool,
    validated: bool = False,
) -> LabeledFeatures:
    selected = [
        (step, label) for step, trn in enumerate(theorem.trns) if (label := policy.label(trn.tactic)) is not None
    ]
    steps = np.fromiter((step for step, _ in selected), dtype=np.int64)
    labels = tuple(label for _, label in selected)
    ids = np.fromiter((label_ids[label] for label in labels), dtype=np.int64)
    trns = tuple(theorem.trns[step] for step, _ in selected)
    if selected:
        retained = msgspec.structs.replace(theorem, trns=trns)
        converted = encode_theorem(retained, layout, validated=validated)
        rows = FeatureRows(converted.name, steps, converted.tactics, converted.matrix)
    else:
        dtype = np.float64 if layout.vocab.representation is not None else np.int64
        rows = FeatureRows(theorem.name, steps, (), csr_array((0, layout.width), dtype=dtype))
    states = StateData(theorem.exprs, trns) if include_states else None
    return LabeledFeatures(rows, labels, ids, tuple(label_ids), states)


def labeled_features(
    theorem: r.Theorem, layout: Layout, policy: LabelPolicy, *, include_states: bool = False
) -> LabeledFeatures:
    """Filter learning rows before preparing one shared theorem graph.

    The retained rows keep original transition indices, even across dropped
    actions. Optional states share the original theorem-local expression table.
    No vocabulary fitting or held-out-dependent column changes occur.
    """
    return _feature_batch(theorem, layout, policy, policy.label_ids, include_states=include_states)


def _state_row(batch: LabeledFeatures, idx: int) -> LabeledState:
    if batch.states is None:
        raise ValueError("individual state queries require include_states=True")
    return LabeledState(
        batch.rows.name,
        int(batch.rows.steps[idx]),
        batch.labels[idx],
        int(batch.label_ids[idx]),
        batch.states.trns[idx],
        batch.states.exprs,
        batch.rows.matrix[idx : idx + 1],
    )


def _check_names(names: Sequence[str] | None) -> None:
    if isinstance(names, str):
        # str satisfies Sequence[str]; the selection value, not its type, is invalid.
        raise ValueError("theorem selection must be a sequence of names, not a single string")  # noqa: TRY004
    if names is not None and len(set(names)) != len(names):
        raise ValueError("theorem selection contains duplicates")


def _transition_rows(db: sqlite3.Connection, names: Sequence[str] | None) -> Iterator[tuple[str, int, bytes]]:
    _check_names(names)
    query = "SELECT name,trn_count,trns FROM theorems"
    if names is None:
        yield from db.execute(query + " ORDER BY id")
    else:
        for name in names:
            row = db.execute(query + " WHERE name=?", (name,)).fetchone()
            if row is None:
                raise KeyError(name)
            yield row


@dataclass(frozen=True, slots=True)
class Corpus:
    """Own one read-only connection and dictionary decoder, not theorem caches.

    Use open_corpus as a context manager. Streaming methods decode at most one
    theorem at a time; a caller retaining their results owns that memory.
    """

    db: sqlite3.Connection
    codec: BlobCodec

    def theorem(self, name: str) -> r.Theorem:
        theorem = load_theorem(self.db, name, self.codec)
        if theorem is None:
            raise KeyError(name)
        return theorem

    def names(self) -> Iterator[str]:
        return (row[0] for row in self.db.execute("SELECT name FROM theorems ORDER BY id"))

    def theorems(self, names: Sequence[str] | None = None) -> Iterator[r.Theorem]:
        _check_names(names)
        if names is not None:
            for name in names:
                yield self.theorem(name)
        else:
            for row in self.db.execute(f"SELECT {THEOREM_COLS} FROM theorems ORDER BY id"):
                yield decode_theorem_row(row, self.codec)

    def labeled_transitions(self, policy: LabelPolicy, names: Sequence[str] | None = None) -> Iterator[LabeledTrn]:
        # Transition-only queries never decode expression blobs or build graphs.
        for name, count, blob in _transition_rows(self.db, names):
            trns = decode_trns(blob, self.codec)
            if len(trns) != count:
                raise r.SchemaError(f"transition count disagrees for {name!r}")
            for step, trn in enumerate(trns):
                if (label := policy.label(trn.tactic)) is not None:
                    yield LabeledTrn(name, step, label, trn)

    def features(
        self, layout: Layout, policy: LabelPolicy, names: Sequence[str] | None = None, *, include_states: bool = False
    ) -> Iterator[LabeledFeatures]:
        """Stream theorem batches; compile labels once and graphs once per theorem.

        Feature-only batches do not retain raw states or expression tables.
        This bounds the scanner to one theorem, not caller-retained results.
        """
        label_ids = policy.label_ids
        for theorem in self.theorems(names):
            # The storage boundary validated the complete theorem before label
            # filtering. Removing transitions cannot introduce invalid refs.
            yield _feature_batch(theorem, layout, policy, label_ids, include_states=include_states, validated=True)

    def proof_states(
        self, layout: Layout, policy: LabelPolicy, names: Sequence[str] | None = None
    ) -> Iterator[LabeledState]:
        """Inspect labeled states with sparse vectors, preparing each theorem once.

        Expression tables are shared between this theorem's results. Each row
        slice is a small sparse copy; use features() to avoid per-row allocations.
        """
        for batch in self.features(layout, policy, names, include_states=True):
            for idx in range(len(batch.labels)):
                yield _state_row(batch, idx)

    def proof_state(self, name: str, step: int, layout: Layout, policy: LabelPolicy) -> LabeledState:
        """Inspect an original transition index, not an index after label filtering.

        A single lookup prepares the retained theorem batch. Repeated lookups
        do not cache it; use proof_states(names=[name]) when inspecting many rows.
        """
        theorem = self.theorem(name)
        if not 0 <= step < len(theorem.trns):
            raise IndexError(f"transition index {step} outside theorem {name!r}")
        if policy.label(theorem.trns[step].tactic) is None:
            raise ValueError(f"transition {step} of {name!r} is excluded by the label policy")
        batch = labeled_features(theorem, layout, policy, include_states=True)
        idx = int(np.searchsorted(batch.rows.steps, step))
        return _state_row(batch, idx)


@contextmanager
def open_corpus(path: Path) -> Generator[Corpus]:
    """Open an existing, completed corpus without creating or modifying it."""
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        check_extraction_schema(db, path)
        yield Corpus(db, BlobCodec(stored_dicts(db)))
