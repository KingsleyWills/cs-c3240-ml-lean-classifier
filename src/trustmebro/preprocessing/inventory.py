"""Aggregate observed fragments, not arbitrary subgraphs or a selected vocabulary.

The source supplies ordered, canonical DAG fragments. Per-depth counts retain
every observation; union counts deduplicate only (theorem, anchor, exact shape).
Different anchors and nested/overlapping shapes are never collapsed. Node role
weights count reachable anchors once per observed top-level root occurrence.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import msgspec
import numpy as np

from .candidates import Candidates, Edges, Selection

# Column order is part of this archive's contract, not the extraction schema.
BASE_COLS = ("nodes", "edges", "frontiers", "anchors", "goals", "hyps", "theorems", "goal_theorems", "hyp_theorems")
DEPTH_COLS = ("anchors", "goals", "hyps", "theorems")
COUNT_DTYPE = np.dtype("<u8")
MAX_COUNT = np.iinfo(COUNT_DTYPE).max


class Summary(msgspec.Struct, frozen=True):
    src: str
    src_size: int
    src_modified_ns: int
    selection: Selection
    theorems: int
    shapes: int
    aggregation_sec: float
    index_bytes: int


@dataclass(frozen=True)
class Inventory:
    summary: Summary
    shapes: list[tuple[bytes, bytes]]
    counts: np.ndarray
    large: dict[tuple[int, int], int]

    def col(self, name: str, depth: int | None = None) -> np.ndarray:
        idx = BASE_COLS.index(name) if depth is None else depth_col(self.summary.selection.depths, depth, name)
        corrections = [(row, val) for (row, col), val in self.large.items() if col == idx]
        vals = self.counts[:, idx]
        if corrections:
            vals = vals.astype(object)
            for row, val in corrections:
                vals[row] = val
        return vals


def depth_col(depths: tuple[int, ...], depth: int, name: str) -> int:
    return len(BASE_COLS) + len(DEPTH_COLS) * depths.index(depth) + DEPTH_COLS.index(name)


class ShapeCounts:
    """Own one growing shape index, count buffer and exact overflow corrections.

    The guard estimates retained index memory, not process RSS or decoded-theorem
    scratch. It aborts instead of dropping shapes. Growth briefly retains both
    count buffers; archive blocks and decoded source frames have separate costs.
    """

    def __init__(self, depths: tuple[int, ...], budget: int) -> None:
        self.depths = depths
        self.budget = budget
        self.index: dict[bytes, int] = {}
        self.shapes: list[tuple[bytes, bytes]] = []
        self.counts = np.zeros((0, len(BASE_COLS) + len(DEPTH_COLS) * len(depths)), dtype=COUNT_DTYPE)
        self.large: dict[tuple[int, int], int] = {}
        self.item_bytes = 0

    def index_bytes(self) -> int:
        return (
            self.item_bytes
            + self.counts.nbytes
            + sys.getsizeof(self.index)
            + sys.getsizeof(self.shapes)
            + sys.getsizeof(self.large)
            + sum(sys.getsizeof(key) + sys.getsizeof(val) for key, val in self.large.items())
        )

    def check_budget(self) -> None:
        if self.index_bytes() > self.budget:
            raise MemoryError("candidate index exceeds --index-memory-mib; no inventory published")

    def register(self, theorem: Candidates) -> np.ndarray:
        refs = np.empty(len(theorem.shapes), dtype=np.intp)
        for local, shape in enumerate(theorem.shapes):
            packed = msgspec.msgpack.encode(shape.edges)
            idx = self.index.get(shape.ident)
            if idx is None:
                idx = len(self.shapes)
                if idx == len(self.counts):
                    capacity = max(256, 2 * idx)
                    projected = self.index_bytes() - self.counts.nbytes + capacity * self.counts.shape[1] * 8
                    if projected > self.budget:
                        raise MemoryError("candidate index exceeds --index-memory-mib; no inventory published")
                    counts = np.zeros((capacity, self.counts.shape[1]), dtype=COUNT_DTYPE)
                    counts[:idx] = self.counts
                    self.counts = counts
                self.index[shape.ident] = idx
                pair = (shape.ident, packed)
                self.shapes.append(pair)
                self.item_bytes += sum(map(sys.getsizeof, pair)) + sys.getsizeof(pair) + sys.getsizeof(idx)
                self.counts[idx, :3] = (
                    len(shape.edges),
                    sum(len(edges) for edges in shape.edges if edges is not None),
                    sum(edges is None for edges in shape.edges),
                )
            elif self.shapes[idx][1] != packed:
                raise ValueError("equal fragment identities have different canonical adjacency")
            refs[local] = idx
        self.check_budget()
        return refs

    def add(self, rows: np.ndarray, col: int, vals: np.ndarray) -> None:
        # Every supplied row is unique. Native additions cannot wrap silently;
        # only overflow/oversized inputs cross back into Python exact arithmetic.
        if vals.dtype == object:
            for row, val in zip(rows, vals, strict=True):
                key = (int(row), col)
                total = self.large.get(key, int(self.counts[row, col])) + int(val)
                self.counts[row, col] = min(total, MAX_COUNT)
                if total > MAX_COUNT:
                    self.large[key] = total
            return
        old = self.counts[rows, col]
        overflow = vals > MAX_COUNT - old
        safe = ~overflow
        self.counts[rows[safe], col] = old[safe] + vals[safe]
        for row, val in zip(rows[overflow], vals[overflow], strict=True):
            key = (int(row), col)
            total = self.large.get(key, int(self.counts[row, col])) + int(val)
            self.counts[row, col] = MAX_COUNT
            self.large[key] = total


def _role_weights(theorem: Candidates, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    refs = np.fromiter((node.ref for node in theorem.nodes), dtype=np.intp, count=len(theorem.nodes))
    order = np.argsort(refs)
    positions = order[np.searchsorted(refs[order], anchors)]
    # Retain Python integers only when a role count cannot fit the native array.
    roles: list[np.ndarray] = []
    for attr in ("goal_count", "hyp_count"):
        vals = [getattr(node, attr) for node in theorem.nodes]
        dtype = object if any(val > MAX_COUNT for val in vals) else COUNT_DTYPE
        roles.append(np.asarray(vals, dtype=dtype)[positions])
    return roles[0], roles[1]


def _add_observations(
    counts: ShapeCounts,
    refs: np.ndarray,
    shapes: np.ndarray,
    goals: np.ndarray,
    hyps: np.ndarray,
    cols: tuple[int, ...],
) -> None:
    """One sorted reduction handles anchor counts, role mass and theorem support."""
    if not len(shapes):
        return
    starts = np.flatnonzero(np.r_[True, shapes[1:] != shapes[:-1]])
    ends = np.r_[starts[1:], len(shapes)]
    rows = refs[shapes[starts]]
    totals: list[np.ndarray] = []
    for weights in (goals, hyps):
        # The maximum-times-length bound prevents overflow during reduceat too.
        if weights.dtype != object and int(weights.max(initial=0)) * len(weights) > MAX_COUNT:
            weights = weights.astype(object)
        totals.append(np.add.reduceat(weights, starts))
    vals = (ends - starts, *totals, np.ones(len(rows), dtype=COUNT_DTYPE))
    if len(cols) == 6:
        vals = (*vals, totals[0] > 0, totals[1] > 0)
    for col, val in zip(cols, vals, strict=True):
        counts.add(rows, col, val.astype(COUNT_DTYPE) if val.dtype.kind in "bi" else val)


def add_theorem(counts: ShapeCounts, theorem: Candidates) -> None:
    refs = counts.register(theorem)
    observations = np.fromiter(
        (val for occ in theorem.occurrences for val in (occ.shape, occ.nodes[0], occ.depth)), dtype=np.intp
    ).reshape(-1, 3)
    if not len(observations):
        return
    order = np.lexsort((observations[:, 2], observations[:, 1], observations[:, 0]))
    shapes, anchors, depths = observations[order].T
    goals, hyps = _role_weights(theorem, anchors)
    distinct = np.r_[True, (shapes[1:] != shapes[:-1]) | (anchors[1:] != anchors[:-1])]
    _add_observations(counts, refs, shapes[distinct], goals[distinct], hyps[distinct], tuple(range(3, 9)))
    for depth in counts.depths:
        selected = depths == depth
        start = depth_col(counts.depths, depth, "anchors")
        _add_observations(
            counts, refs, shapes[selected], goals[selected], hyps[selected], tuple(range(start, start + 4))
        )
    counts.check_budget()


def shape_edges(packed: bytes) -> Edges:
    return msgspec.msgpack.decode(packed, type=Edges)
