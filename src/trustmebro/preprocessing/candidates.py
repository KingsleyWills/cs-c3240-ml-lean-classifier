"""Discover ordered, radius-bounded DAG fragments without selecting a vocabulary.

graph_tool owns reachability and bounded discovery. Numeric adjacency supplies
all operand edges, including those absent from a BFS discovery tree. The JIT
kernel only canonicalizes the discovered region; it does not search the graph.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

import msgspec
import numpy as np
from graph_tool import Graph, VertexPropertyMap
from graph_tool.topology import label_out_component, shortest_distance
from numba import njit
from numpy.typing import NDArray

from trustmebro.extraction import records as r

type IntArray = NDArray[np.int64]
type CountArray = NDArray[np.int64] | NDArray[np.object_]
type Edges = tuple[tuple[int, ...] | None, ...]
DEFAULT_DEPTHS = (1, 2, 3)


# Passive archive records. Original references remain theorem-local.


class Selection(msgspec.Struct, frozen=True):
    db: str
    size: int
    modified_ns: int
    depths: tuple[int, ...]
    limit: int | None
    theorems: tuple[str, ...] | None
    seed: int | None = None


class Shape(msgspec.Struct, frozen=True, array_like=True):
    ident: bytes
    edges: Edges


class Occurrence(msgspec.Struct, frozen=True, array_like=True):
    shape: int
    depth: int
    nodes: tuple[int, ...]


class Node(msgspec.Struct, frozen=True, array_like=True):
    ref: int
    expr: r.Expr
    goal_count: int
    hyp_count: int


class Root(msgspec.Struct, frozen=True, array_like=True):
    ref: int
    anchors: tuple[int, ...]


class LocalInfo(msgspec.Struct, frozen=True, array_like=True):
    is_let: bool
    is_instance: bool


class State(msgspec.Struct, frozen=True, array_like=True):
    step: int
    tactic: r.Tactic
    goal: int
    hyps: tuple[int, ...]
    locals: tuple[LocalInfo, ...] | None = None  # absent in older candidate archives


class Candidates(msgspec.Struct, frozen=True, array_like=True):
    """One theorem's candidates, not fitted features or executable predictions.

    Root closures and states retain co-occurrence and repeated hypotheses without
    duplicating fragment records per state. Node counts use per-expression DAG
    anchors: one visit per goal or hypothesis occurrence, not per expanded path.
    Different query depths can share one shape; consumers must not count them as
    independent entries simply because both depths were requested.
    """

    name: str
    nodes: tuple[Node, ...]
    shapes: tuple[Shape, ...]
    occurrences: tuple[Occurrence, ...]
    roots: tuple[Root, ...]
    states: tuple[State, ...]


@dataclass(frozen=True)
class Adjacency:
    graph: Graph
    children: IntArray
    offsets: IntArray


@dataclass(frozen=True)
class SearchScratch:
    dist: VertexPropertyMap
    pred: VertexPropertyMap
    positions: IntArray


# Preparation occurs once per theorem, not once per state or fragment.


def prepare_adjacency(exprs: tuple[r.Expr, ...]) -> Adjacency:
    """Prepare validated post-order records; their references already ensure a DAG."""
    operands = [r.expr_refs(expr) for expr in exprs]
    counts = np.fromiter(map(len, operands), dtype=np.int64, count=len(exprs))
    offsets = np.empty(len(exprs) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    refs = np.fromiter((ref for items in operands for ref in items), dtype=np.int64, count=int(offsets[-1]))
    if refs.size and (refs.min() < 0 or refs.max() >= len(exprs)):
        raise ValueError("invalid expression operand reference")
    graph = Graph(directed=True)
    graph.add_vertex(len(exprs))
    if refs.size:
        graph.add_edge_list(np.column_stack((np.repeat(np.arange(len(exprs)), counts), refs)))
    return Adjacency(graph, refs, offsets)


def prepare_search(adjacency: Adjacency) -> SearchScratch:
    graph = adjacency.graph
    # Supply our own predecessor map to avoid the wrapper copying vertex_index
    # on every call. Both maps belong to this theorem and remain worker-local.
    return SearchScratch(
        graph.new_vp("int32_t"),
        graph.vertex_index.copy(value_type="int64_t"),
        np.full(graph.num_vertices(), -1, dtype=np.int64),
    )


def observe_roots(
    trns: tuple[r.Trn, ...], adjacency: Adjacency
) -> tuple[tuple[State, ...], tuple[Root, ...], CountArray]:
    states = tuple(
        State(
            step,
            trn.tactic,
            trn.state.target,
            tuple(local.type for local in trn.state.locals),
            tuple(LocalInfo(isinstance(local, r.LocalLet), local.is_instance) for local in trn.state.locals),
        )
        for step, trn in enumerate(trns)
    )
    goals = Counter(state.goal for state in states)
    hyps = Counter(ref for state in states for ref in state.hyps)
    # Each node is visited at most once per top-level occurrence. This bound
    # permits native batched sums without risking fixed-width overflow.
    upper_bound = max(sum(goals.values()), sum(hyps.values()))
    dtype = np.int64 if upper_bound <= np.iinfo(np.int64).max else object
    counts: CountArray = np.zeros((adjacency.graph.num_vertices(), 2), dtype=dtype)
    roots: list[Root] = []
    for ref in sorted(goals.keys() | hyps.keys()):
        mask = label_out_component(adjacency.graph, adjacency.graph.vertex(ref))
        anchors = np.flatnonzero(mask.a)
        roots.append(Root(ref, tuple(map(int, anchors))))
        counts[anchors] += (goals[ref], hyps[ref])
    return states, tuple(roots), counts


# Native bounded discovery, followed by compiled ordered canonicalization.


@njit(cache=True)
def _canonical_region(
    children: IntArray,
    offsets: IntArray,
    distances: NDArray[np.int32],
    positions: IntArray,
    root: int,
    radius: int,
    count: int,
) -> tuple[IntArray, IntArray, IntArray, IntArray]:
    """Serialize already discovered vertices in operand-ordered BFS numbering.

    Each distinct node has one local position; repeated edges retain repeated
    references. Scratch positions are reset only for vertices touched here.
    """
    nodes = np.empty(count, np.int64)
    depths = np.empty(count, np.int64)
    local_offsets = np.empty(count + 1, np.int64)
    nodes[0], depths[0], positions[root] = root, 0, 0
    end = 1
    edge_count = 0
    for idx in range(count):
        node = nodes[idx]
        local_offsets[idx] = edge_count
        if depths[idx] >= radius:
            continue
        for slot in range(offsets[node], offsets[node + 1]):
            child = children[slot]
            if positions[child] == -1:
                positions[child] = end
                nodes[end] = child
                depths[end] = distances[child]
                end += 1
            edge_count += 1
    local_offsets[count] = edge_count
    refs = np.empty(edge_count, np.int64)
    for idx in range(count):
        node = nodes[idx]
        if depths[idx] < radius:
            start = local_offsets[idx]
            for slot in range(offsets[node], offsets[node + 1]):
                refs[start + slot - offsets[node]] = positions[children[slot]]
    for node in nodes:
        positions[node] = -1
    return nodes, depths, refs, local_offsets


def extract_fragments(
    adjacency: Adjacency, scratch: SearchScratch, root: int, depths: tuple[int, ...]
) -> dict[int, tuple[Edges, tuple[int, ...]]]:
    """One native search to the largest radius; smaller radii reuse its prefix."""
    radius = max(depths)
    if adjacency.offsets[root] == adjacency.offsets[root + 1]:
        return dict.fromkeys(depths, (((),), (root,)))
    _, _, reached = shortest_distance(
        adjacency.graph, source=root, max_dist=radius, return_reached=True, dist_map=scratch.dist, pred_map=scratch.pred
    )
    # graph_tool's reached array omits the source. Its order is not our identity.
    nodes, distances, refs, offsets = _canonical_region(
        adjacency.children, adjacency.offsets, scratch.dist.a, scratch.positions, root, radius, len(reached) + 1
    )
    scratch.pred.a[reached] = reached
    scratch.pred.a[root] = root
    fragments: dict[int, tuple[Edges, tuple[int, ...]]] = {}
    for depth in depths:
        end = int(np.searchsorted(distances, depth, side="right"))
        edges: Edges = tuple(
            None
            if distances[idx] == depth and adjacency.offsets[node] != adjacency.offsets[node + 1]
            else tuple(map(int, refs[offsets[idx] : offsets[idx + 1]]))
            for idx, node in enumerate(nodes[:end])
        )
        fragments[depth] = edges, tuple(map(int, nodes[:end]))
    return fragments


def extract_candidates(
    theorem: r.Theorem, depths: tuple[int, ...] = DEFAULT_DEPTHS, *, validated: bool = False
) -> Candidates:
    """Extract every observed anchor; retain all candidates without filtering.

    validated=True is only for records already checked by validate_theorem;
    structural preparation still runs once here, irrespective of validation.
    """
    if not depths or any(depth < 1 for depth in depths):
        raise ValueError("candidate depths must be positive and nonempty")
    depths = tuple(sorted(set(depths)))
    if not validated:
        r.validate_theorem(theorem)
    adjacency = prepare_adjacency(theorem.exprs)
    scratch = prepare_search(adjacency)
    states, roots, counts = observe_roots(theorem.trns, adjacency)
    observed = np.flatnonzero(np.any(counts != 0, axis=1))
    nodes = tuple(Node(int(ref), theorem.exprs[ref], int(counts[ref, 0]), int(counts[ref, 1])) for ref in observed)
    shapes: list[Shape] = []
    shape_ids: dict[Edges, int] = {}
    ident_shapes: dict[bytes, Edges] = {}
    occurrences: list[Occurrence] = []
    for node in nodes:
        for depth, (edges, refs) in extract_fragments(adjacency, scratch, node.ref, depths).items():
            shape = shape_ids.get(edges)
            if shape is None:
                # Encode/hash each distinct theorem-local shape once, including
                # leaves and complete shapes repeated at several query depths.
                ident = hashlib.sha256(msgspec.msgpack.encode(edges)).digest()
                previous = ident_shapes.get(ident)
                if previous is not None and previous != edges:
                    raise ValueError("conflicting fragments for the same shape identity")
                shape = len(shapes)
                shape_ids[edges] = shape
                ident_shapes[ident] = edges
                shapes.append(Shape(ident, edges))
            occurrences.append(Occurrence(shape, depth, refs))
    return Candidates(theorem.name, nodes, tuple(shapes), tuple(occurrences), roots, states)
