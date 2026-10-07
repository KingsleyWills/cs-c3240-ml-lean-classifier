"""Measure prepared graphs and aggregate corpus statistics; no corpus/archive I/O."""

from __future__ import annotations

import hashlib
import heapq
import math
import sys
from collections import Counter
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as data_field
from functools import cache, lru_cache, partial
from itertools import batched, islice, repeat
from operator import attrgetter
from typing import Protocol

import msgspec
import numpy as np
from graph_tool import GraphView, edge_endpoint_property
from numba import njit
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from trustmebro.extraction import records as r

from .diagnostics import TimingLog, checkpoint
from .measurements import *
from .processing import (
    DEFAULT_REACH_BUDGET,
    ExprArrays,
    ExprGraph,
    GraphStats,
    Idents,
    Observations,
    PatternSigs,
    ReachBits,
    ReachCache,
    RootGraph,
    RootMeasure,
    RootPatterns,
    SignatureReuse,
    TopoView,
    bitmap_nodes,
    constructor_masks,
    data_digest,
    expr_arrays,
    extract_patterns,
    graph_stats,
    measure_roots,
    measure_view,
    reachable_from,
    reachable_nodes,
    resolve_root,
    root_bitmap,
    root_graph,
    topo_sig,
    view_arrays,
    view_heads,
)

# Analysis constants and result records.


STRAL_SAMPLE_SIZE = 200_000
PATTERN_WEIGHT_COUNT = 4
MOTIF_BATCH_WALKS = 262_144
PATTERN_CAPACITY_ROWS = 65_536
PATTERN_BUFFER_ROWS = 16_384
PATTERN_BUFFER_BATCHES = 64
CONCENTRATION_BATCH_CURVES = 64
PATTERN_WORK_DTYPE = np.dtype([("key", PATTERN_KEY_DTYPE), ("rad", "<u4")])
type FreqCounts = dict[bytes, FreqRow]


class ShapeCand(msgspec.Struct, frozen=True, order=True):
    prio: float
    root: int


class CorpusShapeCand(msgspec.Struct, frozen=True, order=True):
    prio: float
    theorem: str
    root: int


class RootRef(msgspec.Struct, frozen=True):
    digest: bytes
    root: int


class TheoremRoot(msgspec.Struct, frozen=True, order=True):
    theorem: str
    root: int


class StralCand(msgspec.Struct, frozen=True):
    prio: bytes
    root: int


class PatternIdx(Protocol):
    def find(self, digests: Iterable[bytes]) -> NDArray[np.bool_]: ...


@dataclass
class TheoremMetrics:
    name: str
    states: list[StateRow] = data_field(default_factory=list)
    exprs: list[ExprRow] = data_field(default_factory=list)
    curves: list[FloatArray] = data_field(default_factory=list)
    empty: int = 0
    freqs: list[FreqRow] = data_field(default_factory=list)
    shapes: dict[ShapeKey, ShapeCand] = data_field(default_factory=dict)
    stral: list[StralCand] = data_field(default_factory=list)
    common_roots: list[RootRef] = data_field(default_factory=list)


@dataclass
class ReuseMeasures:
    name: str
    states: list[ReuseRow]
    exprs: list[ReuseRow]
    stral: dict[int, StralRow]


@dataclass(slots=True)
class ReuseSizes:
    """Coordinated exact tree recurrences and native replacement costs."""

    expanded: list[int]
    reduced: list[int]
    costs: IntArray
    replaced: NDArray[np.bool_]
    depths: list[int] | None


@dataclass(order=True, slots=True)
class StralSample:
    prio: int
    name: str
    root: int
    row: StralRow | None = data_field(default=None, compare=False)


@dataclass
class StateCounts:
    """Exact percentile inputs plus the existing concentration measurement grids."""

    sizes: list[CountArray] = data_field(default_factory=list)
    conc: IntArray = data_field(default_factory=lambda: np.zeros((1000, 1000), dtype=np.int64))
    rotated_conc: IntArray = data_field(default_factory=lambda: np.zeros((1000, 1000), dtype=np.int64))
    empty: int = 0
    total: int = 0
    theorems: int = 0


@dataclass
class ExampleCounts:
    """Atlas representatives and bounded embedding candidates, not corpus counts."""

    min_nodes: int = 20
    common_roots: dict[bytes, TheoremRoot] = data_field(default_factory=dict)
    shapes: dict[ShapeKey, CorpusShapeCand] = data_field(default_factory=dict)
    stral: list[StralSample] = data_field(default_factory=list)
    stral_roots: int = 0


@dataclass
class MetricCounts:
    """Owned corpus aggregates; frequency and percentile retention are not bounded."""

    examples: ExampleCounts = data_field(default_factory=ExampleCounts)
    states: StateCounts = data_field(default_factory=StateCounts)
    freqs: FreqCounts = data_field(default_factory=dict)


@dataclass
class FreqTotals:
    pairs: dict[tuple[int, int], list[int]] = data_field(default_factory=dict)
    hgrams: list[Counter[int]] = data_field(default_factory=lambda: [Counter(), Counter(), Counter()])


@dataclass
class TopoCounts:
    shapes: dict[ViewMode, dict[bytes, ShapeCount]] = data_field(
        default_factory=lambda: {mode: {} for mode in VIEW_MODES}
    )
    pairs: dict[ViewMode, dict[ComparisonLvl, Counter[Pair]]] = data_field(
        default_factory=lambda: {mode: {} for mode in POLICY_MODES}
    )
    heads: dict[ViewMode, Counter[str]] = data_field(default_factory=lambda: {mode: Counter() for mode in VIEW_MODES})


@dataclass(slots=True)
class HeadDistrs:
    counts: IntArray
    totals: CountArray
    entropies: FloatArray


# Structural descriptor calculations.


@lru_cache(maxsize=8192)
def _descr_name(group: str, *parts: str | int) -> str:
    """Stable sparse column names, shared across theorem-specific histograms."""
    return "_".join((group, *(str(part) for part in parts)))


@njit(cache=True)
def _root_depths(
    order: IntArray, children: IntArray, offsets: IntArray, body_slots: IntArray, binder_counts: IntArray
) -> tuple[IntArray, IntArray]:
    """Longest-path levels and binder nesting over an already ordered DAG."""
    lvls = np.zeros(len(order), dtype=np.int64)
    nesting = np.zeros(len(order), dtype=np.int64)
    for node in order:
        for pos in range(offsets[node], offsets[node + 1]):
            child = children[pos]
            slot = pos - offsets[node]
            lvls[child] = max(lvls[child], lvls[node] + 1)
            nesting[child] = max(
                nesting[child], nesting[node] + (binder_counts[node] if slot == body_slots[node] else 0)
            )
    return lvls, nesting


@lru_cache(maxsize=8192)
def _motif_bin(descr: tuple[str, int, str, int, str]) -> int:
    digest = data_digest(descr)
    return int.from_bytes(digest[:2], "little") % 64


def _edge_motif_distrs(arrays: ExprArrays, root: RootGraph) -> dict[str, float]:
    """Operand-labelled edges and hashed two-edge motifs, preserving repeats."""
    children = root.children
    degrees = np.diff(root.offsets)
    parents = np.repeat(np.arange(len(root.nodes)), degrees)
    slots = np.arange(len(children)) - np.repeat(root.offsets[:-1], degrees)
    slot_count = max(3, int(degrees.max(initial=0)))
    kinds = len(arrays.names)
    codes = arrays.kinds[root.nodes]
    targets = children
    edges = (codes[parents] * slot_count + slots) * kinds + codes[targets]
    edge_codes, e_counts = np.unique(edges, return_counts=True)
    m_counts = _motif_counts(root, codes, edges, degrees, kinds, slot_count, arrays.names)
    descrs: dict[str, float] = {}
    e_total = max(1, e_counts.sum())
    m_total = max(1, m_counts.sum())
    for code, count in zip(edge_codes, e_counts, strict=True):
        parent_slot, child = divmod(int(code), kinds)
        parent, slot = divmod(parent_slot, slot_count)
        descrs[_descr_name("edge", arrays.names[parent], slot, arrays.names[child])] = float(count / e_total)
    descrs.update({_descr_name("motif", int(idx)): float(m_counts[idx] / m_total) for idx in np.flatnonzero(m_counts)})
    return descrs


def _motif_counts(
    root: RootGraph, codes: IntArray, edges: IntArray, degrees: IntArray, kinds: int, slots: int, names: tuple[str, ...]
) -> IntArray:
    """Count every ordered two-edge walk without retaining an E²-sized product.

    Each batch holds at most the walk threshold or one operand's fanout.
    Root arrays and the motif-signature cache are outside that working bound.
    """
    targets = root.children
    offsets = np.concatenate(([0], degrees[targets].cumsum()))
    counts = np.zeros(64, dtype=np.int64)
    start = 0
    while start < len(targets):
        stop = max(start + 1, int(offsets.searchsorted(offsets[start] + MOTIF_BATCH_WALKS, side="right")) - 1)
        stop = min(stop, len(targets))
        fanout = degrees[targets[start:stop]]
        first = np.repeat(np.arange(start, stop), fanout)
        second_slots = np.arange(len(first)) - np.repeat(offsets[start:stop] - offsets[start], fanout)
        grandchildren = root.children[root.offsets[targets[first]] + second_slots]
        motifs = (edges[first] * slots + second_slots) * kinds + codes[grandchildren]
        unique, weights = np.unique(motifs, return_counts=True)
        bins: list[int] = []
        for code in unique:
            code, grandchild = divmod(int(code), kinds)
            code, second = divmod(code, slots)
            code, child = divmod(code, kinds)
            parent, first_slot = divmod(code, slots)
            bins.append(_motif_bin((names[parent], first_slot, names[child], second, names[grandchild])))
        np.add.at(counts, np.asarray(bins, dtype=np.int64), weights)
        start = stop
    return counts


def _layer_distrs(names: tuple[str, ...], codes: IntArray, lvls: IntArray) -> dict[str, float]:
    depth = max(1, int(lvls.max()))
    layer = np.minimum((lvls / depth * 4).astype(np.int64), 3)
    counts = np.bincount(layer * len(names) + codes, minlength=4 * len(names))
    return {
        _descr_name("layer", int(code) // len(names), names[int(code) % len(names)]): float(counts[code] / len(codes))
        for code in np.flatnonzero(counts)
    }


def _sharing_distrs(incoming: IntArray, internal: NDArray[np.bool_]) -> dict[str, float]:
    bins = np.searchsorted([1, 2, 4, 8], incoming, side="right")
    counts = np.bincount(internal * 5 + bins, minlength=10) / len(incoming)
    return {
        _descr_name("sharing", kind, idx): float(counts[offset + idx])
        for kind, offset in (("leaf", 0), ("internal", 5))
        for idx in range(5)
    }


def _binder_spine_distrs(spines: IntArray, vars: IntArray, nesting: IntArray) -> dict[str, float]:
    pops = (
        ("spine", np.searchsorted([2, 4, 8, 16], spines[spines > 0], side="right")),
        ("binder_ref", vars[vars >= 0]),
        ("binder_nesting", np.minimum(nesting, 4)),
    )
    descrs: dict[str, float] = {}
    for prefix, bins in pops:
        counts = np.bincount(bins, minlength=5)
        total = max(1, counts.sum())
        descrs.update({_descr_name(prefix, idx): float(counts[idx] / total) for idx in range(5)})
    return descrs


def _shape_distrs(arrays: ExprArrays, root: RootGraph) -> dict[str, float]:
    """Root-local depth/binder traversal once, followed by independent histograms."""
    nodes = root.nodes
    lvls, nesting = _root_depths(
        root.order, root.children, root.offsets, arrays.body_slots[nodes], arrays.binder_counts[nodes]
    )
    return {
        **_edge_motif_distrs(arrays, root),
        **_layer_distrs(arrays.names, arrays.kinds[nodes], lvls),
        **_sharing_distrs(root.incoming, np.diff(root.offsets) > 0),
        **_binder_spine_distrs(arrays.spines[nodes], arrays.var_bins[nodes], nesting),
    }


def stral_features(rows: Sequence[StralRow]) -> tuple[FloatArray, list[str]]:
    """Scale-aware scalar descriptors and constructor fractions, not graph IDs."""
    kinds = sorted({kind for row in rows for kind in row.constrs})
    prefixes = ("edge_", "motif_", "layer_", "spine_", "sharing_", "binder_")
    topo = sorted({name for row in rows for name in row.topo if name.startswith(prefixes)})
    vals = np.array(
        [
            [
                math.log10(row.nodes),
                math.log10(row.depth),
                *(row.constrs.get(kind, 0) / row.nodes for kind in kinds),
                *(row.topo.get(name, 0) for name in topo),
            ]
            for row in rows
        ]
    )
    return vals, ["log10_nodes", "log10_depth", *[f"frac_{kind}" for kind in kinds], *topo]


# Original theorem and cross-theorem reuse measurements.


def count_exprs(
    graph: ExprGraph, stats: GraphStats, cache: ReachCache, trns: Sequence[r.Trn]
) -> tuple[list[FreqRow], list[RootRef]]:
    order = graph.order
    roots: Counter[int] = Counter()
    for trn in trns:
        roots.update([trn.state.target, *(local.type for local in trn.state.locals)])
    distinct_freqs = np.zeros(len(graph.exprs), dtype=np.int64)
    expanded_freqs = [roots[node] for node in range(len(graph.exprs))]
    for root, freq in roots.items():
        distinct_freqs[reachable_nodes(cache, (root,))] += freq
    # Propagate arbitrary-precision multiplicities down the DAG, not a tree.
    for node in order:
        for child in graph.edges[node]:
            expanded_freqs[child] += expanded_freqs[node]
    ids = Idents(graph)
    closures = [0] * len(graph.exprs)
    remaining = graph.graph.get_in_degrees(np.arange(len(graph.exprs)))
    recs: dict[tuple[bytes, int, int], list[int]] = {}
    for node in reversed(order):
        bits = 1 << node
        for child in graph.edges[node]:
            bits |= closures[child]
            remaining[child] -= 1
            if not remaining[child]:
                closures[child] = 0
        # Once every parent has consumed a closure, release its large bitmap.
        closures[node] = bits if remaining[node] else 0
        if not distinct_freqs[node]:
            continue
        digest = ids.sig(node)
        counts = recs.setdefault((digest, bits.bit_count(), stats.sizes[node]), [0, 0, 0])
        for idx, count in enumerate((roots[node], int(distinct_freqs[node]), expanded_freqs[node])):
            counts[idx] += count
    freqs = [
        FreqRow(digest, distinct, expanded, top_lvl, distinct_roots, expanded_roots)
        for (digest, distinct, expanded), (top_lvl, distinct_roots, expanded_roots) in recs.items()
    ]
    # Only the smallest root per identity can become the theorem's representative.
    representatives: dict[bytes, int] = {}
    for root in roots:
        digest = ids.sig(root)
        previous = representatives.get(digest)
        if previous is None or root < previous:
            representatives[digest] = root
    return freqs, [RootRef(digest, root) for digest, root in representatives.items()]


def _aux_roots(state: r.ProofState) -> list[int]:
    roots: list[int] = []
    for mvar in state.mvars:
        if mvar.decl is not None:
            roots.extend([mvar.decl.type, *(local.type for local in mvar.decl.locals)])
            roots.extend(lcl.val for lcl in mvar.decl.locals if isinstance(lcl, r.LocalLet))
        if mvar.assignment is not None:
            roots.append(mvar.assignment)
    return roots


def _conc_curve(sizes: list[int]) -> np.ndarray:
    total = sum(sizes)
    cum = np.cumsum([size / total for size in sizes])
    curve = np.interp((np.arange(1000) + 0.5) / 1000, np.linspace(0, 1, len(sizes) + 1), np.r_[0, cum])
    return curve


def _measure_state(
    result: TheoremMetrics, step: int, trn: r.Trn, stats: GraphStats, measure: Callable[[Sequence[int]], RootMeasure]
) -> None:
    state = trn.state
    roots = [local.type for local in state.locals]
    goal = measure([state.target])
    ctxt = measure(roots)
    combined = measure([state.target, *roots])
    sizes = sorted((stats.sizes[n] for n in roots), reverse=True)
    largest = sizes[0] / sum(sizes) if sizes else 0
    individual = goal.distinct + sum(measure([n]).distinct for n in roots)
    lets = [local.val for local in state.locals if isinstance(local, r.LocalLet)]
    # Expanded-only auxiliary measurements don't need expensive reachability.
    let_size = sum(stats.sizes[n] for n in lets)
    extra_size = sum(stats.sizes[n] for n in _aux_roots(state))
    result.states.append(
        StateRow(
            theorem=result.name,
            step=step,
            hyps=len(roots),
            goal_expanded=goal.expanded,
            goal_distinct=goal.distinct,
            goal_depth=goal.depth,
            ctxt_expanded=ctxt.expanded,
            ctxt_distinct=ctxt.distinct,
            ctxt_depth=ctxt.depth,
            state_expanded=combined.expanded,
            state_distinct=combined.distinct,
            largest_frac=largest,
            individual_distinct=individual,
            let_expanded=let_size,
            mvar_expanded=extra_size,
            goal_constrs=goal.constrs,
            ctxt_constrs=ctxt.constrs,
            state_constrs=combined.constrs,
        )
    )
    for role, nodes in (("goal", [state.target]), ("ctxt", roots)):
        result.exprs.extend(
            ExprRow(
                theorem=result.name,
                role=role,
                expanded=stats.sizes[n],
                distinct_nodes=measure([n]).distinct,
                depth=stats.depths[n],
            )
            for n in nodes
        )
    if sizes:
        result.curves.append(_conc_curve(sizes))
    else:
        result.empty += 1


def measure_theorem(
    name: str, graph: ExprGraph, stats: GraphStats, cache: ReachCache, trns: Sequence[r.Trn], *, min_nodes: int = 0
) -> TheoremMetrics:
    measure = partial(measure_roots, stats, constructor_masks(graph.exprs), cache)
    result = TheoremMetrics(name)
    for step, trn in enumerate(trns):
        _measure_state(result, step, trn, stats, measure)
    result.freqs, result.common_roots = count_exprs(graph, stats, cache, trns)
    eligible = {row.digest for row in result.freqs if row.distinct >= min_nodes and row.top}
    result.common_roots = [ref for ref in result.common_roots if ref.digest in eligible]
    # Sample distinct roots, not successive observations of unchanged states.
    roots = {trn.state.target for trn in trns}
    roots.update(local.type for trn in trns for local in trn.state.locals)
    for root in roots:
        measured = measure([root])
        size, distinct, depth = measured.expanded, measured.distinct, measured.depth
        # Only cheap selection data crosses the first pass. The global reuse
        # pass fills descriptors for selected roots, not every corpus root.
        prio = hashlib.sha256(f"{name}:{root}".encode()).digest()
        result.stral.append(StralCand(prio, root))
        # Include the largest/deepest/most-shared roots even if random sampling
        # misses them. Dedicated strata prevent them replacing ordinary examples.
        for idx, score in enumerate((distinct, depth, math.log10(size) - math.log10(distinct))):
            extreme = (4, idx, 0)
            cand = ShapeCand(-score, root)
            if extreme not in result.shapes or cand < result.shapes[extreme]:
                result.shapes[extreme] = cand
    return result


def _stral_row(
    name: str, stats: GraphStats, masks: Mapping[str, int], cache: ReachCache, arrays: ExprArrays, root: int
) -> StralRow:
    measured = measure_roots(stats, masks, cache, [root])
    graph = root_graph(cache, arrays, root)
    shared = graph.incoming > 1
    distinct, depth = measured.distinct, measured.depth
    return StralRow(
        name,
        root,
        distinct,
        depth,
        measured.expanded,
        float(np.count_nonzero(shared) / distinct),
        measured.constrs,
        _shape_distrs(arrays, graph),
    )


def _reuse_sizes(graph: ExprGraph, reused: NDArray[np.bool_], ids: Idents, *, depths: bool) -> ReuseSizes:
    """Compute both exact recurrences once; replacement eligibility needs raw size."""
    count = len(graph.exprs)
    expanded, reduced = [0] * count, [0] * count
    levels = [0] * count if depths else None
    costs = np.ones(count, dtype=np.int64)
    replaced = np.zeros(count, dtype=bool)
    for node in reversed(graph.order):
        refs = graph.edges[node]
        size = expanded[node] = 1 + sum(expanded[child] for child in refs)
        cost = 1 + len(ids.vars[node])
        if reused[node] and cost < size:
            costs[node] = reduced[node] = cost
            replaced[node] = True
        else:
            reduced[node] = 1 + sum(reduced[child] for child in refs)
        if levels is not None:
            levels[node] = 1 + max((levels[child] for child in refs), default=0)
    return ReuseSizes(expanded, reduced, costs, replaced, levels)


def measure_reuse(
    name: str,
    graph: ExprGraph,
    trns: Sequence[r.Trn],
    shared: PatternIdx,
    sampled_roots: Sequence[int],
    *,
    reach_budget: int = DEFAULT_REACH_BUDGET,
) -> ReuseMeasures:
    """Measure largest-first replacements without expanding implicit trees.

    An open pattern costs one reference plus one argument per distinct variable.
    The residual DAG retains children needed through any unreplaced path.
    """
    states_roots = [(trn.state.target, *(local.type for local in trn.state.locals)) for trn in trns]
    roots = tuple(dict.fromkeys(node for roots in states_roots for node in roots))
    observed = reachable_from(graph.graph, roots)
    order = np.asarray(graph.order[::-1], dtype=np.int64)
    nodes = order[observed[order]]
    ids = Idents(graph, observed)
    reused = np.zeros(len(graph.exprs), dtype=bool)
    reused[nodes] = shared.find(ids.sig(int(node)) for node in nodes)
    sizes = _reuse_sizes(graph, reused, ids, depths=bool(sampled_roots))
    novel = int.from_bytes(np.packbits(observed & ~reused, bitorder="little").tobytes(), "little")
    bitmaps = ReachBits(reach_budget)
    cache = ReachCache(graph, bitmaps)
    # A replacement retains its own vertex but cuts outgoing edges. Native
    # filtered reachability preserves descendants reached by an unreplaced path.
    retained = graph.graph.new_vertex_property("bool", val=True)
    retained.a[:] = ~sizes.replaced
    cut = GraphView(graph.graph, efilt=edge_endpoint_property(graph.graph, retained, "source"))

    memo: dict[tuple[int, ...], ReuseRow] = {}

    def measure(nodes: Sequence[int]) -> ReuseRow:
        key = tuple(nodes)
        if key in memo:
            return memo[key]
        original = 0
        remaining = 0
        for node in nodes:
            original |= root_bitmap(cache, node)
            remaining |= bitmaps.get(cut, node)
        dag_size = int(sizes.costs[bitmap_nodes(remaining)].sum())
        val = ReuseRow(
            expanded=sum(sizes.expanded[node] for node in nodes),
            local_dag=original.bit_count(),
            reduced_tree=sum(sizes.reduced[node] for node in nodes),
            reduced_dag=dag_size,
            novel=(original & novel).bit_count(),
            nodes=original.bit_count(),
        )
        memo[key] = val
        return val

    states: list[ReuseRow] = []
    exprs: list[ReuseRow] = []
    for nodes in states_roots:
        states.append(measure(nodes))
        exprs.extend(measure([node]) for node in nodes)
    stral: dict[int, StralRow] = {}
    if sampled_roots:
        assert sizes.depths is not None
        stats = GraphStats(sizes.expanded, sizes.depths)
        arrays, masks = expr_arrays(graph), constructor_masks(graph.exprs)
        stral = {root: _stral_row(name, stats, masks, cache, arrays, root) for root in sampled_roots}
    return ReuseMeasures(name, states, exprs, stral)


# Transformed graph and local-pattern measurements.


def _head_counts(view: TopoView, cache: ReachCache, states: Counter[tuple[int, ...]]) -> Counter[str]:
    """Complete application anchors, once per state DAG, weighted by states.

    Exported applications are already flattened; count each application anchor.
    Nonconstant heads and compact markers are reported separately by kind.
    """
    graph = view.graph
    occs = np.zeros(len(graph.exprs), dtype=np.int64)
    for roots, weight in states.items():
        bits = 0
        for root in roots:
            bits |= root_bitmap(cache, resolve_root(view, root))
        occs[bitmap_nodes(bits)] += weight
    counts: Counter[str] = Counter()
    heads = view_heads(view)
    for node in np.flatnonzero(occs):
        root = int(node)
        if not isinstance(graph.exprs[root], r.App):
            continue
        counts[heads[root]] += int(occs[root])
    return counts


def _size_band(size: int) -> int:
    return 0 if size < 8 else 1 if size < 32 else 2 if size < 128 else 3


def _view_measure(view: TopoView, stats: GraphStats, reach: ReachCache) -> Callable[[Sequence[int]], GraphSize]:
    """Memoize reductions only for this theorem/view; retain root multiplicity."""
    arrays = view_arrays(view)

    @cache
    def measure(roots: tuple[int, ...]) -> GraphSize:
        return measure_view(view, stats, reach, arrays, roots)

    def resolved_measure(roots: Sequence[int]) -> GraphSize:
        # Order cannot affect these union/sum/max measurements, but repetitions
        # must survive: two identical hypotheses count twice in expanded size.
        return measure(tuple(sorted(resolve_root(view, root) for root in roots)))

    return resolved_measure


def measure_topo(
    name: str,
    views: Mapping[ViewMode, TopoView],
    observations: Observations,
    raw_stats: GraphStats,
    raw_cache: ReachCache,
    timings: TimingLog | None = None,
    reuse: Mapping[ViewMode, SignatureReuse] | None = None,
) -> TopoMeasures:
    """Count roots per state; count each reachable node only once per state.

    Graph-tool supplies native reachability. Never unfold a DAG into its tree.
    The native bit-array conversion also avoids a Python visit per state node.
    """
    states, top, all_roots = observations.states, observations.top, observations.all
    caches = {mode: raw_cache if mode == ViewMode.ORIGINAL else ReachCache(view.graph) for mode, view in views.items()}
    measures: dict[ViewMode, Callable[[Sequence[int]], GraphSize]] = {}
    with checkpoint(timings, name, "topo.prepare_measurements"):
        for mode, view in views.items():
            stats = raw_stats if mode == ViewMode.ORIGINAL else graph_stats(view.graph)
            measures[mode] = _view_measure(view, stats, caches[mode])
    roots = tuple(map(int, np.flatnonzero(all_roots)))
    sigs_by_mode: dict[ViewMode, list[tuple[bytes, int]]] = {}
    previous_sigs: dict[int, tuple[bytes, int]] = {}
    previous_mode: ViewMode | None = None
    for mode, view in views.items():
        with checkpoint(timings, name, f"topo.signatures.{mode}"):
            resolved = tuple(resolve_root(view, root) for root in roots)
            inherited = reuse.get(mode) if reuse is not None else None
            src_sigs = previous_sigs if inherited is not None and inherited.src == previous_mode else {}
            sigs = {
                root: src_sigs[root]
                if inherited is not None and inherited.topo[root] and root in src_sigs
                else topo_sig(view.graph, root)
                for root in dict.fromkeys(resolved)
            }
            previous_sigs, previous_mode = sigs, mode
            sigs_by_mode[mode] = [sigs[root] for root in resolved]
    del previous_sigs
    shapes: dict[ViewMode, dict[bytes, ShapeCount]] = {mode: {} for mode in VIEW_MODES}
    pairs: dict[ViewMode, dict[ComparisonLvl, Counter[Pair]]] = {
        mode: {"expression": Counter[Pair](), "state": Counter[Pair]()} for mode in POLICY_MODES
    }
    with checkpoint(timings, name, "topo.shape_counts"):
        # Mutable scratch bands avoid a tuple/list round trip per observation.
        bands: dict[ViewMode, dict[bytes, list[list[int]]]] = {mode: {} for mode in VIEW_MODES}
        for idx, root in enumerate(roots):
            band = _size_band(sigs_by_mode[ViewMode.ORIGINAL][idx][1])
            top_weight, all_weight = int(top[root]), int(all_roots[root])
            for mode in VIEW_MODES:
                digest, size = sigs_by_mode[mode][idx]
                item = shapes[mode].get(digest)
                if item is None:
                    item = shapes[mode][digest] = ShapeCount(size, 0, 0, name, root)
                    bands[mode][digest] = [[0, 0] for _ in range(SHAPE_SIZE_BANDS)]
                elif item.nodes != size:
                    raise ValueError("same topology signature has conflicting sizes")
                item.top += top_weight
                item.all += all_weight
                weights = bands[mode][digest][band]
                weights[0] += top_weight
                weights[1] += all_weight
        for mode, counts in shapes.items():
            for digest, item in counts.items():
                item.bands = tuple((a, b) for a, b in bands[mode][digest])
    del bands, sigs_by_mode
    with checkpoint(timings, name, "topo.sizes"):
        for root in roots:
            if top[root]:
                sizes = [measures[mode]((root,)) for mode in VIEW_MODES]
                for mode, size in zip(POLICY_MODES, sizes[1:], strict=True):
                    pairs[mode]["expression"][(sizes[0], size)] += int(top[root])
        for roots, weight in states.items():
            sizes = [measures[mode](roots) for mode in VIEW_MODES]
            for mode, size in zip(POLICY_MODES, sizes[1:], strict=True):
                pairs[mode]["state"][(sizes[0], size)] += weight
    rows = {
        mode: [(digest, item.nodes, item.top, item.all, item.root, item.bands) for digest, item in counts.items()]
        for mode, counts in shapes.items()
    }
    with checkpoint(timings, name, "topo.heads"):
        heads = {mode: _head_counts(views[mode], caches[mode], states) for mode in VIEW_MODES}
    return TopoMeasures(name, rows, pairs, heads)


def pattern_observations(
    observations: Observations, affected: NDArray[np.bool_], heads: Mapping[int, str]
) -> PatternObs:
    """Prepare raw observation weights once, without resolving lossy-view aliases."""
    top, all_ = observations.top, observations.all
    roots = np.flatnonzero(all_)
    weights = np.column_stack((top[roots], all_[roots], top[roots] * affected[roots], all_[roots] * affected[roots]))
    names = {"": 0}
    head_ids = np.fromiter(
        (names.setdefault(heads.get(int(root), ""), len(names)) for root in roots), dtype=np.uint32, count=len(roots)
    )
    return PatternObs(roots, weights.astype(np.uint64), tuple(names), head_ids)


def measure_patterns(
    view: TopoView,
    radii: Sequence[int],
    observations: PatternObs,
    *,
    mode: ViewMode,
    timings: TimingLog | None = None,
    name: str = "",
    inherited: tuple[NDArray[np.bool_], PatternSigs] | None = None,
) -> tuple[dict[int, PatternBatch], PatternSigs]:
    """Return packed counts and scratch signatures for the immediately following view."""
    phase = f"patterns.{mode}"
    with checkpoint(timings, name, f"{phase}.signatures"):
        resolved = tuple(resolve_root(view, int(root)) for root in observations.roots)
        sigs: PatternSigs = {}
        for root in dict.fromkeys(resolved):
            if inherited is not None and inherited[0][root] and root in inherited[1]:
                sigs[root] = inherited[1][root]
            else:
                sigs[root] = extract_patterns(view, root, radii).sigs
    with checkpoint(timings, name, f"{phase}.counts_pack"):
        if not resolved or not radii:
            return {
                rad: PatternBatch(np.empty(0, PATTERN_KEY_DTYPE), np.empty((0, 4), np.uint64), observations.heads)
                for rad in radii
            }, sigs
        # One native grouping call covers all radii/flavours/head cohorts for
        # this view. Radius IDs exist only in scratch, not the archive keys.
        work = np.empty((len(radii), len(resolved), 2, 2), dtype=PATTERN_WORK_DTYPE)
        work["rad"] = np.arange(len(radii), dtype=np.uint32)[:, None, None, None]
        work["key"]["flavour"] = np.arange(2, dtype=np.uint8)[None, None, :, None]
        work["key"]["head"] = np.column_stack((np.zeros(len(resolved), np.uint32), observations.head_ids))[
            None, :, None, :
        ]
        for idx, rad in enumerate(radii):
            work["key"]["digest"][idx] = np.asarray([sigs[root][rad] for root in resolved], dtype="V32")[:, :, None]
        keep = np.ones(work.shape, dtype=bool)
        keep[..., 1] = observations.head_ids[None, :, None] != 0
        weights = np.broadcast_to(observations.weights[None, :, None, None, :], (*work.shape, 4))[keep]
        keys, totals = _coalesce_patterns(work[keep], weights)
        grouped = np.frombuffer(b"".join(keys), dtype=PATTERN_WORK_DTYPE)
        return {
            rad: PatternBatch(grouped["key"][grouped["rad"] == idx], totals[grouped["rad"] == idx], observations.heads)
            for idx, rad in enumerate(radii)
        }, sigs


def _vocab_node(ref: int, expr: r.Expr) -> VocabNode:
    name = expr.name if isinstance(expr, r.Const) else None
    match expr:
        case r.Fvar(id=id):
            var = ("free", id)
        case r.ExprMvar(id=id):
            var = ("meta", id)
        case r.Bvar(idx=idx):
            var = ("bound", idx)
        case _:
            var = None
    literal = expr.val if isinstance(expr, (r.NatLiteral, r.StringLiteral)) else None
    binders = len(expr.names) if isinstance(expr, (r.Lambda, r.Forall)) else int(isinstance(expr, r.Let))
    return VocabNode(ref, type(expr).__name__, name, var, literal, binders)


def measure_vocabulary(
    name: str, graph: ExprGraph, cache: ReachCache, trns: Sequence[r.Trn], prepared: Mapping[int, RootPatterns]
) -> VocabTheorem:
    """Retain occurrence correspondence and labels without choosing feature policies.

    Shapes are deduplicated locally; matches at different anchors still remain
    separate. A repeated shape at several radii remains a query observation,
    not several different vocabulary identities.
    """
    shapes: dict[bytes, VocabShape] = {}
    occurrences: list[VocabOccurrence] = []
    for patterns in prepared.values():
        for depth, fragment in patterns.fragments.items():
            digest = patterns.sigs[depth][0]
            shape = shapes.get(digest)
            if shape is None:
                shapes[digest] = VocabShape(digest, fragment.edges)
            elif shape.edges != fragment.edges:
                raise ValueError("conflicting canonical fragments for the same shape identity")
            occurrences.append(VocabOccurrence(digest, depth, fragment.nodes))
    states = tuple(
        VocabState(step, trn.tactic.kind, trn.tactic.src, trn.state.target, tuple(lcl.type for lcl in trn.state.locals))
        for step, trn in enumerate(trns)
    )
    roots = dict.fromkeys(root for state in states for root in (state.goal, *state.hyps))
    closures = tuple(VocabRoot(root, tuple(map(int, reachable_nodes(cache, (root,))))) for root in roots)
    return VocabTheorem(
        name,
        tuple(_vocab_node(ref, graph.exprs[ref]) for ref in prepared),
        tuple(shapes.values()),
        tuple(occurrences),
        closures,
        states,
    )


def pack_patterns(rows: Iterable[tuple[bytes, int, str, Sequence[int]]]) -> PatternBatch:
    """Pack theorem-local unique rows once, before crossing a process boundary."""
    heads = {"": 0}
    keys: list[bytes] = []
    weights: list[Sequence[int]] = []
    for digest, flavour, head, vals in rows:
        head_id = heads.setdefault(head, len(heads))
        keys.append(pattern_key(digest, flavour, head_id))
        weights.append(vals)
    return PatternBatch(np.frombuffer(b"".join(keys), dtype=PATTERN_KEY_DTYPE), exact_weights(weights), tuple(heads))


def exact_weights(rows: Sequence[Sequence[int]]) -> CountArray:
    """Keep the common path native without truncating exceptional naturals."""
    try:
        return np.asarray(rows, dtype=np.uint64).reshape(-1, PATTERN_WEIGHT_COUNT)
    except OverflowError:
        weights = np.asarray(rows, dtype=object).reshape(-1, PATTERN_WEIGHT_COUNT)
        if np.any(weights < 0):
            raise ValueError("local-pattern counts must be nonnegative")
        return weights


# Corpus aggregation and representative selection.


def add_freqs(counts: FreqCounts, records: Iterable[FreqRow]) -> None:
    """The aggregate owns its mutable counts, never a worker's record."""
    for row in records:
        entry = counts.get(row.digest)
        if entry is None:
            counts[row.digest] = FreqRow(
                row.digest, row.distinct, row.expanded, row.top, row.root_dag, row.expanded_tree, row.theorems
            )
        else:
            if entry.distinct != row.distinct or entry.expanded != row.expanded:
                raise ValueError("identical structural fingerprints have inconsistent sizes")
            entry.top += row.top
            entry.root_dag += row.root_dag
            entry.expanded_tree += row.expanded_tree
            entry.theorems += row.theorems


def add_samples(samples: list[StralSample], name: str, cands: Sequence[StralCand]) -> None:
    # Bottom-k hashes preserve selection across worker completion orders.
    fill = min(STRAL_SAMPLE_SIZE - len(samples), len(cands))
    if fill:
        samples.extend(StralSample(-int.from_bytes(data.prio, "big"), name, data.root) for data in cands[:fill])
        # Before the reservoir fills, no operation needs heap order.
        if len(samples) == STRAL_SAMPLE_SIZE:
            heapq.heapify(samples)
    for data in islice(cands, fill, None):
        prio = -int.from_bytes(data.prio, "big")
        if prio > samples[0].prio:
            heapq.heapreplace(samples, StralSample(prio, name, data.root))


def add_examples(examples: ExampleCounts, freqs: FreqCounts, result: TheoremMetrics) -> None:
    for ref in result.common_roots:
        if freqs[ref.digest].distinct < examples.min_nodes:
            continue
        previous = examples.common_roots.get(ref.digest)
        if previous is None or (result.name, ref.root) < (previous.theorem, previous.root):
            examples.common_roots[ref.digest] = TheoremRoot(result.name, ref.root)
    for key, cand in result.shapes.items():
        previous = examples.shapes.get(key)
        if previous is None or (cand.prio, result.name, cand.root) < (previous.prio, previous.theorem, previous.root):
            examples.shapes[key] = CorpusShapeCand(cand.prio, result.name, cand.root)
    examples.stral_roots += len(result.stral)
    add_samples(examples.stral, result.name, result.stral)


@njit(cache=True, nogil=True)
def _concentration_cells(conc: IntArray, rotated: IntArray, curves: FloatArray) -> None:
    """Fuse both rasters without temporary grids or racing shared-cell updates."""
    for curve in curves:
        for pos, val in enumerate(curve):
            x = (pos + 0.5) / 1000
            conc[pos, min(999, max(0, int(val * 1000)))] += 1
            # Keep the original floating operation order and truncation; fastmath
            # or parallel shared writes could change bins or occurrence weights.
            along = min(999, max(0, int((x + val) / 2 * 1000)))
            above = min(999, max(0, int((val - x) * 1000)))
            rotated[along, above] += 1


def add_concentration(conc: IntArray, rotated: IntArray, curves: Iterable[FloatArray]) -> None:
    for batch in batched(curves, CONCENTRATION_BATCH_CURVES):
        _concentration_cells(conc, rotated, np.asarray(batch))


def add_metrics(counts: MetricCounts, result: TheoremMetrics) -> None:
    add_freqs(counts.freqs, result.freqs)
    add_examples(counts.examples, counts.freqs, result)
    states = counts.states
    sizes = [(row.goal_expanded, row.ctxt_expanded, row.state_expanded) for row in result.states]
    try:
        values = np.asarray(sizes, dtype=np.uint64)
    except OverflowError:
        values = np.asarray(sizes, dtype=object)
    states.sizes.append(values.reshape(-1, 3))
    add_concentration(states.conc, states.rotated_conc, result.curves)
    states.empty += result.empty
    states.total += len(result.states)
    states.theorems += 1


def common_candidates(examples: ExampleCounts, freqs: FreqCounts) -> list[tuple[bytes, TheoremRoot]]:
    """References only; orchestration loads the selected expressions."""
    digests = heapq.nlargest(128, examples.common_roots, key=lambda digest: (freqs[digest].top, digest))
    return [(digest, examples.common_roots[digest]) for digest in digests]


def select_common(
    shapes: dict[ShapeKey, CorpusShapeCand], freqs: FreqCounts, cands: Iterable[tuple[bytes, TheoremRoot, bytes]]
) -> None:
    seen: set[bytes] = set()
    for digest, ref, sig in cands:
        if sig in seen:
            continue
        seen.add(sig)
        shapes[(-2, len(seen), freqs[digest].top)] = CorpusShapeCand(0, ref.theorem, ref.root)
        if len(seen) >= 12:
            break


def finish_samples(samples: Sequence[StralSample]) -> list[StralRow]:
    rows: list[StralRow] = []
    for sample in sorted(samples, key=attrgetter("prio", "name", "root"), reverse=True):
        if sample.row is None:
            raise RuntimeError("selected stral root was not measured in the global reuse pass")
        rows.append(sample.row)
    return rows


def add_freq_totals(totals: FreqTotals, rows: Iterable[FreqRow]) -> None:
    """Batch ordinary size groups; exceptional identities retain Python integers."""
    rows = tuple(rows)
    if not rows:
        return
    dtype = np.dtype([("distinct", "<u8"), ("expanded", "<u8"), ("weights", "<u8", (3,))])
    try:
        values = np.fromiter(
            ((row.distinct, row.expanded, (row.top, row.root_dag, row.expanded_tree)) for row in rows),
            dtype=dtype,
            count=len(rows),
        )
    except OverflowError:
        # One huge expression must not force every ordinary row in this batch
        # through boxed arithmetic. Both paths update the same exact aggregate.
        native: list[FreqRow] = []
        exceptional: list[FreqRow] = []
        max_nat = np.iinfo(np.uint64).max
        for row in rows:
            dst = (
                native
                if max(row.distinct, row.expanded, row.top, row.root_dag, row.expanded_tree) <= max_nat
                else exceptional
            )
            dst.append(row)
        _add_exact_freq_totals(totals, exceptional)
        if native:
            add_freq_totals(totals, native)
        return
    keys = values[["distinct", "expanded"]]
    order = np.lexsort((values["expanded"], values["distinct"]))
    keys, weights = keys[order], values["weights"][order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    sums = _group_totals(weights, starts)
    maxima = np.maximum.reduceat(weights, starts, axis=0)
    for key, sums_row, max_row in zip(keys[starts], sums, maxima, strict=True):
        pair = int(key["distinct"]), int(key["expanded"])
        entry = totals.pairs.get(pair)
        if entry is None:
            entry = totals.pairs[pair] = [0] * 6
        for pop in range(3):
            entry[2 * pop] += int(sums_row[pop])
            entry[2 * pop + 1] = max(entry[2 * pop + 1], int(max_row[pop]))
    for pop, col in enumerate(values["weights"].T):
        counts, multiplicities = np.unique(col[col > 0], return_counts=True)
        totals.hgrams[pop].update(dict(zip(map(int, counts), map(int, multiplicities), strict=True)))


def _add_exact_freq_totals(totals: FreqTotals, rows: Iterable[FreqRow]) -> None:
    for row in rows:
        pair = row.distinct, row.expanded
        entry = totals.pairs.get(pair)
        if entry is None:
            entry = totals.pairs[pair] = [0] * 6
        for pop, count in enumerate((row.top, row.root_dag, row.expanded_tree)):
            if count:
                entry[2 * pop] += count
                entry[2 * pop + 1] = max(entry[2 * pop + 1], count)
                totals.hgrams[pop][count] += 1


def add_topo(counts: TopoCounts, measured: TopoMeasures) -> None:
    name, shapes, pairs, heads = measured.name, measured.shapes, measured.pairs, measured.heads
    for mode, rows in shapes.items():
        entries = counts.shapes[mode]
        for digest, size, top, all_, root, bands in rows:
            item = entries.get(digest)
            if item is None:
                entries[digest] = ShapeCount(size, top, all_, name, root, bands)
            else:
                if item.nodes != size:
                    raise ValueError("same topology signature has conflicting sizes")
                item.top += top
                item.all += all_
                item.bands = tuple((a + c, b + d) for (a, b), (c, d) in zip(item.bands, bands, strict=True))
                if name < item.theorem or (name == item.theorem and root < item.root):
                    item.theorem, item.root = name, root
    for mode, lvls in pairs.items():
        for lvl, hgram in lvls.items():
            dst = counts.pairs[mode].get(lvl)
            if dst is None:
                dst = counts.pairs[mode][lvl] = Counter()
            dst.update(hgram)
    for mode, hgram in heads.items():
        counts.heads[mode].update(hgram)


class PatternCounts:
    """Own pending arrays/index/counts; use an append-only, possibly shared head table.

    Reconcile theorem-local head IDs immediately; coalesce pending native arrays
    before querying the global index. Neither graph references nor theorem data
    are retained. A large unique input bypasses buffering and regrouping.
    """

    def __init__(self, heads: dict[str, int] | None = None) -> None:
        self.idx: dict[bytes, int] = {}
        self.heads: dict[str, int] = {"": 0} if heads is None else heads
        self.weights: CountArray = np.empty((0, PATTERN_WEIGHT_COUNT), dtype=np.uint64)
        self.length = 0
        self.pending: list[tuple[NDArray[np.void], CountArray]] = []
        self.pending_rows = 0
        self.pending_bytes = 0

    def add(
        self,
        batch: PatternBatch,
        timings: TimingLog | None = None,
        label: str = "",
        remaps: dict[tuple[str, ...], NDArray[np.uint32]] | None = None,
    ) -> None:
        """Retain only packed pattern arrays; drain before exceeding the row limit."""
        if not len(batch):
            return
        with checkpoint(timings, label, "pattern.heads"):
            # The cache is scoped to this destination table and one theorem;
            # sibling indexes may share both without repeating reconciliation.
            remap = None if remaps is None else remaps.get(batch.heads)
            if remap is None:
                remap = np.fromiter(
                    (self.heads.setdefault(head, len(self.heads)) for head in batch.heads), dtype=np.uint32
                )
                if remaps is not None:
                    remaps[batch.heads] = remap
            identity = np.array_equal(remap, np.arange(len(remap), dtype=np.uint32))
            if identity:
                # Borrow without modifying; retained workers' arrays are never
                # mutated by aggregation, coalescing or publication.
                packed = batch.keys
            else:
                packed = batch.keys.copy()
                packed["head"] = remap[packed["head"]]
        if self.pending_rows + len(batch) > PATTERN_BUFFER_ROWS:
            self.flush(timings, label)
        if len(batch) >= PATTERN_BUFFER_ROWS:
            self._add_unique(packed.view(PATTERN_BYTES_DTYPE).tolist(), batch.weights, timings, label)
            return
        self.pending.append((packed, batch.weights))
        self.pending_rows += len(batch)
        self.pending_bytes += packed.nbytes + batch.weights.nbytes
        if batch.weights.dtype == object:
            self.pending_bytes += batch.weights.size * 32
        if len(self.pending) >= PATTERN_BUFFER_BATCHES:
            self.flush(timings, label)

    def flush(self, timings: TimingLog | None = None, label: str = "") -> None:
        """Consume pending arrays once; all published counts include the final tail."""
        if not self.pending:
            return
        with checkpoint(timings, label, "pattern.coalesce"):
            if len(self.pending) == 1:
                packed, weights = self.pending[0]
                keys = packed.view(PATTERN_BYTES_DTYPE).tolist()
            else:
                packed = np.concatenate([keys for keys, _ in self.pending])
                weights = np.concatenate([weights for _, weights in self.pending])
                keys, weights = _coalesce_patterns(packed, weights)
            self.pending.clear()
            self.pending_rows = self.pending_bytes = 0
        self._add_unique(keys, weights, timings, label)

    def _add_unique(self, keys: list[bytes], weights: CountArray, timings: TimingLog | None, label: str) -> None:
        """Query each coalesced key once and update its coordinated weight row."""
        with checkpoint(timings, label, "pattern.index"):
            idxs = np.fromiter(map(self.idx.get, keys, repeat(-1)), dtype=np.int64, count=len(keys))
            missing = np.flatnonzero(idxs < 0)
            new = np.arange(self.length, self.length + len(missing), dtype=np.int64)
            self.idx.update(zip(map(keys.__getitem__, missing.tolist()), new.tolist(), strict=True))
            idxs[missing] = new
            self.length += len(missing)
        with checkpoint(timings, label, "pattern.buffer_update"):
            capacity = len(self.weights)
            dtype = object if weights.dtype == object or self.weights.dtype == object else np.uint64
            if self.length > capacity or self.weights.dtype != dtype:
                size = max(self.length, capacity * 2, PATTERN_CAPACITY_ROWS)
                grown = np.empty((size, PATTERN_WEIGHT_COUNT), dtype=dtype)
                live = self.length - len(missing)
                grown[:live] = self.weights[:live]
                self.weights = grown
            self.weights[new] = 0
            if dtype == object:
                self.weights[idxs] += weights
            else:
                overflow = _update_pattern_totals(self.weights, idxs, weights)
                if np.any(overflow):
                    # Overflowing cells still contain their old value; other
                    # cells were updated once. Promote only after the native pass.
                    grown = np.empty(self.weights.shape, dtype=object)
                    grown[: self.length] = self.weights[: self.length]
                    self.weights = grown
                    rows, cols = np.nonzero(overflow)
                    self.weights[idxs[rows], cols] += weights[rows, cols].astype(object)

    @property
    def memory_bytes(self) -> int:
        boxed = self.weights.size * 32 if self.weights.dtype == object else 0
        return sys.getsizeof(self.idx) + len(self.idx) * 98 + self.weights.nbytes + boxed + self.pending_bytes

    def batches(self, timings: TimingLog | None = None, label: str = "") -> Generator[PatternBatch]:
        """Sort natively, retaining packed keys and gathering counters in bounded batches."""
        self.flush(timings, label)
        with checkpoint(timings, label, "pattern.sort"):
            keys = np.frombuffer(b"".join(self.idx), dtype=PATTERN_KEY_DTYPE)
            heads = tuple(self.heads)
            lexical = np.argsort(np.argsort(heads))
            order = np.lexsort((lexical[keys["head"]], keys["flavour"], keys["digest"]))
        for start in range(0, len(order), PATTERN_BATCH_ROWS):
            with checkpoint(timings, label, "pattern.gather"):
                idxs = order[start : start + PATTERN_BATCH_ROWS]
                batch = PatternBatch(keys[idxs], self.weights[idxs], heads)
            yield batch


@njit(cache=True, nogil=True)
def _update_pattern_totals(
    buffer: NDArray[np.uint64], idxs: IntArray, weights: NDArray[np.uint64]
) -> NDArray[np.bool_]:
    """Update unique indexed rows in place, leaving carry cells for exact repair."""
    overflow = np.zeros(weights.shape, dtype=np.bool_)
    for row, idx in enumerate(idxs):
        for col in range(weights.shape[1]):
            old = buffer[idx, col]
            val = old + weights[row, col]
            if val < old:
                overflow[row, col] = True
            else:
                buffer[idx, col] = val
    return overflow


@njit(cache=True, nogil=True)
def _packed_groups(raw: NDArray[np.uint8], words: NDArray[np.uint64]) -> tuple[IntArray, IntArray]:
    """Factor complete 37/41-byte keys through Numba's native typed hash table."""
    zero = np.uint64(0)
    idx = {(zero, zero, zero, zero, zero, zero): 0}
    idx.clear()
    codes = np.empty(len(raw), dtype=np.int64)
    first = np.empty(len(raw), dtype=np.int64)
    for row in range(len(raw)):
        tail = zero
        for col in range(32, min(40, raw.shape[1])):
            tail |= np.uint64(raw[row, col]) << np.uint64(8 * (col - 32))
        last = np.uint64(raw[row, 40]) if raw.shape[1] > 40 else zero
        key = (words[row, 0], words[row, 1], words[row, 2], words[row, 3], tail, last)
        if key in idx:
            code = idx[key]
        else:
            code = len(idx)
            idx[key] = code
            first[code] = row
        codes[row] = code
    return codes, first[: len(idx)]


@njit(cache=True, nogil=True)
def _indexed_totals(
    codes: IntArray, weights: NDArray[np.uint64], count: int
) -> tuple[NDArray[np.uint64], NDArray[np.bool_]]:
    """Accumulate all cohorts in one native pass and mark actual carry events."""
    totals = np.zeros((count, weights.shape[1]), dtype=np.uint64)
    overflow = np.zeros(totals.shape, dtype=np.bool_)
    for row, code in enumerate(codes):
        for col in range(weights.shape[1]):
            old = totals[code, col]
            val = old + weights[row, col]
            totals[code, col] = val
            overflow[code, col] |= val < old
    return totals, overflow


def _coalesce_patterns(packed: NDArray[np.void], weights: CountArray) -> tuple[list[bytes], CountArray]:
    """Box only distinct keys; ordinary grouping and accumulation stay native."""
    if not len(packed):
        return [], weights
    packed = np.ascontiguousarray(packed)
    width = packed.dtype.itemsize
    if width not in (PATTERN_KEY_DTYPE.itemsize, PATTERN_WORK_DTYPE.itemsize):
        raise ValueError("unsupported packed pattern key layout")
    # Four unaligned little-endian word views cover the digest without copying.
    # The tail includes flavour, head, and (for worker scratch) radius in full.
    words = np.ndarray((len(packed), 4), dtype="<u8", buffer=packed, strides=(width, 8))
    codes, first = _packed_groups(packed.view(np.uint8).reshape(-1, width), words)
    if weights.dtype == object:
        totals = np.zeros((len(first), weights.shape[1]), dtype=object)
        np.add.at(totals, codes, weights)
    else:
        totals, overflow = _indexed_totals(codes, weights, len(first))
        if np.any(overflow):
            totals = totals.astype(object)
            for group, col in zip(*np.nonzero(overflow), strict=True):
                totals[group, col] = sum(map(int, weights[codes == group, col]))
    return packed[first].view(np.dtype((np.void, width))).tolist(), totals


def remap_pattern_heads(
    src_heads: Sequence[str],
    dst_heads: Sequence[str],
    batches: Iterable[PatternBatch],
    timings: TimingLog | None = None,
    label: str = "",
) -> Generator[PatternBatch]:
    """Reconcile run-local head IDs in bulk without unpacking digests or weights."""
    with checkpoint(timings, label, "pattern.headmap"):
        dst_ids = {head: idx for idx, head in enumerate(dst_heads)}
        remap = np.fromiter((dst_ids[head] for head in src_heads), dtype=np.uint32, count=len(src_heads))
        heads = tuple(dst_heads)
    if np.array_equal(remap, np.arange(len(src_heads), dtype=np.uint32)):
        for batch in batches:
            yield PatternBatch(batch.keys, batch.weights, heads)
        return
    for batch in batches:
        with checkpoint(timings, label, "pattern.remap"):
            keys = batch.keys.copy()
            keys["head"] = remap[keys["head"]]
        yield PatternBatch(keys, batch.weights, heads)


@njit(cache=True, inline="always")
def _pattern_heap_key(
    words: NDArray[np.uint64], flavours: NDArray[np.uint8], heads: NDArray[np.uint32], row: int, run: int
) -> tuple[np.uint64, ...]:
    # Numeric big-endian digest words preserve byte ordering. Head IDs are
    # numeric, not ordered by their little-endian packed representation.
    return (
        words[row, 0],
        words[row, 1],
        words[row, 2],
        words[row, 3],
        np.uint64(flavours[row]),
        np.uint64(heads[row]),
        np.uint64(run),
    )


@njit(cache=True, nogil=True)
def _merge_pattern_order(
    words: NDArray[np.uint64], flavours: NDArray[np.uint8], heads: NDArray[np.uint32], offsets: IntArray
) -> IntArray:
    """Compile standard-library heap operations over already ordered prefixes."""
    pos = offsets[:-1].copy()
    heap = [_pattern_heap_key(words, flavours, heads, row, run) for run, row in enumerate(pos)]
    heapq.heapify(heap)
    order = np.empty(len(words), dtype=np.int64)
    for idx in range(len(words)):
        run = int(heap[0][-1])
        order[idx] = pos[run]
        pos[run] += 1
        if pos[run] == offsets[run + 1]:
            heapq.heappop(heap)
        else:
            heapq.heapreplace(heap, _pattern_heap_key(words, flavours, heads, pos[run], run))
    return order


def merge_patterns(
    streams: Iterable[Iterable[PatternBatch]], timings: TimingLog | None = None, label: str = ""
) -> Generator[PatternBatch]:
    """Merge sorted buffers in bounded prefixes, reducing equal keys natively.

    The smallest buffered last key bounds a prefix for which no unseen input
    can contribute. At most one chunk per run is retained, plus that prefix.
    Head IDs must already refer to a shared lexically ordered head table.
    """
    streams = [iter(stream) for stream in streams]
    if len(streams) == 1:
        yield from streams[0]
        return

    pending: list[PatternBatch | None] = [None] * len(streams)
    ends: list[tuple[bytes, int, int]] = [(b"", 0, 0)] * len(streams)
    while True:
        for idx, stream in enumerate(streams):
            batch = pending[idx]
            if batch is None or not len(batch):
                pending[idx] = next((batch for batch in stream if len(batch)), None)
                if (batch := pending[idx]) is not None:
                    last = batch.keys[-1]
                    ends[idx] = bytes(last["digest"]), int(last["flavour"]), int(last["head"])
        active = [batch for batch in pending if batch is not None]
        if not active:
            return
        if len(active) == 1:
            # The remaining run is already unique, sorted and rechunked.
            yield active[0]
            idx = next(idx for idx, batch in enumerate(pending) if batch is not None)
            yield from (batch for batch in streams[idx] if len(batch))
            return
        last_idx = min((idx for idx, batch in enumerate(pending) if batch is not None), key=ends.__getitem__)
        last_batch = pending[last_idx]
        assert last_batch is not None
        last = last_batch.keys[-1]
        key_parts: list[NDArray[np.void]] = []
        weight_parts: list[CountArray] = []
        for idx, batch in enumerate(pending):
            if batch is None:
                continue
            stop = int(np.searchsorted(batch.keys, last, side="right"))
            if not stop:
                continue
            key_parts.append(batch.keys[:stop])
            weight_parts.append(batch.weights[:stop])
            pending[idx] = PatternBatch(batch.keys[stop:], batch.weights[stop:], batch.heads)
        if len(key_parts) == 1:
            unique, totals = key_parts[0], weight_parts[0]
        else:
            keys = np.concatenate(key_parts)
            weights = np.concatenate(weight_parts)
            offsets = np.r_[0, np.cumsum([len(part) for part in key_parts])]
            with checkpoint(timings, label, "pattern.merge.order"):
                words = np.ndarray((len(keys), 4), dtype=">u8", buffer=keys, strides=(keys.dtype.itemsize, 8))
                order = _merge_pattern_order(words.astype(np.uint64), keys["flavour"], keys["head"], offsets)
                keys, weights = keys[order], weights[order]
            with checkpoint(timings, label, "pattern.merge.reduce"):
                starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
                totals = _group_totals(weights, starts)
                unique = keys[starts]
        # Rechunk every merge level. Otherwise fan-in multiplies frame sizes
        # again at each compaction round and defeats the decoded-buffer bound.
        for start in range(0, len(unique), PATTERN_BATCH_ROWS):
            stop = start + PATTERN_BATCH_ROWS
            yield PatternBatch(unique[start:stop], totals[start:stop], active[0].heads)


# Coverage, comparisons, and corpus summaries.


def _group_totals(weights: CountArray, starts: IntArray) -> CountArray:
    """Keep native inputs; only potentially overflowing output cells become exact."""
    if len(starts) == len(weights):
        return weights
    lengths = np.diff(np.r_[starts, len(weights)])
    totals = np.add.reduceat(weights, starts, axis=0)
    if weights.dtype == object or int(weights.max()) * int(lengths.max()) <= np.iinfo(np.uint64).max:
        return totals
    maxima = np.maximum.reduceat(weights, starts, axis=0)
    suspect = maxima > (np.iinfo(np.uint64).max // lengths.astype(np.uint64))[:, None]
    if np.any(suspect):
        # Native reduction may wrap only in these cells. Recompute them from
        # the original counts, never from a wrapped or floating intermediate.
        totals = totals.astype(object)
        for group, col in zip(*np.nonzero(suspect), strict=True):
            start = starts[group]
            totals[group, col] = sum(map(int, weights[start : start + lengths[group], col]))
    return totals


def exact_count_sum(vals: np.ndarray) -> int:
    """Use native addition when its conservative bound fits, exact integers otherwise."""
    if vals.dtype == object or (vals.size and int(vals.max()) * vals.size > np.iinfo(vals.dtype).max):
        return int(vals.sum(dtype=object))
    return int(vals.sum())


def coverage(weights: np.ndarray) -> dict[str, list[float]]:
    vals = weights[weights > 0]
    return _positive_coverage(vals, exact_count_sum(vals))


def _positive_coverage(vals: np.ndarray, total: int) -> dict[str, list[float]]:
    """Build coverage from an already filtered population and its exact total."""
    vals = np.sort(vals)[::-1]
    if not len(vals):
        return {"rank": [], "coverage": []}
    rank = np.unique(np.r_[0, np.geomspace(1, len(vals), min(1500, len(vals))).astype(int) - 1, len(vals) - 1])
    dtype = object if total > np.iinfo(np.int64).max else np.int64
    cum = np.cumsum(vals, dtype=dtype)
    return {"rank": (rank + 1).tolist(), "coverage": (cum[rank] / cum[-1]).tolist()}


def pair_cols(hgram: PairHgram) -> PairCols:
    if isinstance(hgram, PairCols):
        return hgram

    def cols(sizes: Sequence[GraphSize]) -> SizeCols:
        return SizeCols(
            *(
                np.asarray([getattr(size, name) for size in sizes], dtype=object if name == "expanded" else np.int64)
                for name in SIZE_FIELDS
            )
        )

    return PairCols(
        cols([original for original, _ in hgram]),
        cols([flat for _, flat in hgram]),
        np.asarray(list(hgram.values()), dtype=np.float64),
    )


def weighted_qtiles(vals: IntArray | FloatArray | NDArray[np.object_], weights: FloatArray) -> Qtiles | None:
    if not vals.size:
        return None
    order = np.argsort(vals)
    cum = np.cumsum(weights[order])
    thresholds = np.array((0.5, 0.95, 0.99, 1.0)) * cum[-1]
    idxs = np.minimum(np.searchsorted(cum, thresholds), len(order) - 1)
    return Qtiles(*(val.item() if isinstance(val, np.generic) else val for val in vals[order[idxs]]))


def comparison_summary(
    pairs: Mapping[ViewMode, Mapping[ComparisonLvl, PairHgram]],
) -> dict[ViewMode, dict[ComparisonLvl, LvlSummary]]:
    """Weighted percentiles describe observed states, rather than unique tuples."""

    def descr(cols: SizeCols, weights: FloatArray) -> dict[str, Qtiles | None]:
        return {name: weighted_qtiles(getattr(cols, name), weights) for name in SIZE_FIELDS}

    def summarize(hgram: PairHgram) -> LvlSummary:
        cols = pair_cols(hgram)
        return LvlSummary(
            int(cols.weights.sum()), descr(cols.original, cols.weights), descr(cols.transformed, cols.weights)
        )

    return {mode: {lvl: summarize(hgram) for lvl, hgram in lvls.items()} for mode, lvls in pairs.items()}


def _pattern_distr(weights: CountArray) -> tuple[CountArray, int, float]:
    """Return the positive population, exact total, and entropy for both consumers."""
    vals = weights[weights > 0]
    if not len(vals):
        return vals, 0, -0.0
    total = exact_count_sum(vals)
    if total > sys.float_info.max:
        probs = np.fromiter((int(val) / total for val in vals), dtype=float, count=len(vals))
    else:
        probs = vals.astype(float) / float(total)
    positive = probs[probs > 0]
    entropy = float(-np.sum(positive * np.log2(positive)))
    return vals, total, entropy


def _head_distrs(weights: CountArray, starts: IntArray) -> HeadDistrs:
    """Measure all sorted heads/cohorts together, not one small array at a time.

    Exceptional totals retain the scalar arbitrary-natural division path. Normal
    groups use native segmented reductions; entropy may vary in its last bits
    from the scalar sum order, while all population counts remain exact.
    """
    totals = _group_totals(weights, starts)
    counts = np.add.reduceat((weights > 0).astype(np.int64), starts, axis=0)
    entropies = np.empty(totals.shape, dtype=np.float64)
    lengths = np.diff(np.r_[starts, len(weights)])
    if totals.dtype == object:
        for idx, (start, length) in enumerate(zip(starts, lengths, strict=True)):
            for col in range(PATTERN_WEIGHT_COUNT):
                _, _, entropies[idx, col] = _pattern_distr(weights[start : start + length, col])
        return HeadDistrs(counts, totals, entropies)
    for col in range(PATTERN_WEIGHT_COUNT):
        probs = np.zeros(len(weights), dtype=np.float64)
        denominators = np.repeat(totals[:, col], lengths)
        np.divide(weights[:, col], denominators, out=probs, where=denominators > 0)
        terms = np.zeros_like(probs)
        np.log2(probs, out=terms, where=probs > 0)
        terms *= probs
        entropies[:, col] = -np.add.reduceat(terms, starts)
    return HeadDistrs(counts, totals, entropies)


def pattern_descr(weights: CountArray) -> PatternDescr:
    vals, total, entropy = _pattern_distr(weights)
    curve = _positive_coverage(vals, total)
    return {
        "patterns": len(vals),
        "occs": total,
        "entropy_bits": entropy,
        "effective_patterns": 2**entropy if total else 0,
        "singletons": int((vals == 1).sum()),
        "rank": curve["rank"],
        "coverage": curve["coverage"],
    }


def coverage_curve(hgram: Counter[int]) -> tuple[NDArray[np.integer], FloatArray]:
    """Equal-frequency groups yield an exact piecewise-linear coverage curve."""
    groups = sorted(hgram.items(), reverse=True)
    total = sum(freq * multiplicity for freq, multiplicity in groups)
    ranks = np.cumsum([multiplicity for _, multiplicity in groups])
    coverage = np.cumsum([freq * multiplicity / total for freq, multiplicity in groups])
    positions = np.unique(np.rint(np.geomspace(1, ranks[-1], min(1000, ranks[-1]))).astype(int))
    return positions, np.interp(positions, np.r_[0, ranks], np.r_[0, coverage])


def state_summary(counts: StateCounts, expr_idents: int) -> StateSummary:
    vals = np.concatenate(counts.sizes) if counts.sizes else np.empty((0, 3), dtype=object)
    qs = (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
    pcentiles: dict[str, dict[float, int | None]] = {}
    for col, role in enumerate(("goal", "ctxt", "state")):
        observed = vals[:, col]
        observed = np.sort(observed[observed > 0])
        # Nearest-rank quantiles retain exact observed integer counts.
        idxs = np.maximum(0, np.ceil(np.asarray(qs) * len(observed)).astype(int) - 1)
        qtiles = observed[idxs].tolist() if observed.size else [None] * len(qs)
        pcentiles[role] = dict(zip(qs, qtiles, strict=True))
    return {
        "states": counts.total,
        "theorems": counts.theorems,
        "expr_idents": expr_idents,
        "pcentiles_expanded": pcentiles,
    }


def freq_stats(summary: FreqTotals) -> FreqStats:
    curves: dict[Pop, CoverageCurve] = {}
    for name, hgram in zip(("top_level", "distinct_subexprs", "expanded_subexprs"), summary.hgrams, strict=True):
        if hgram:
            rank, covered = coverage_curve(hgram)
            curves[name] = {"retained": rank.tolist(), "coverage": covered.tolist()}
    return {
        "pairs": [(distinct, expanded, weights) for (distinct, expanded), weights in summary.pairs.items()],
        "coverage": curves,
    }


def topo_stats(shapes: Mapping[ViewMode, Mapping[bytes, ShapeCount]]) -> TopoStats:
    curves: dict[str, dict[ViewMode, dict[str, list[float]]]] = {
        f"{field}:{band}": {} for field in ("top", "all") for band in range(SHAPE_SIZE_BANDS + 1)
    }
    base: dict[str, dict[str, dict[str, list[float]]]] = {"top": {}, "all": {}}
    transformed: dict[ViewMode, int] = {}
    topos = top_occs = all_occs = 0
    for mode, mode_shapes in shapes.items():
        # One view at a time; no second collection of per-topology Python objects.
        rows = mode_shapes.values()
        array = np.fromiter(((r.nodes, r.top, r.all, r.bands) for r in rows), dtype=SHAPE_DTYPE, count=len(mode_shapes))
        if mode == "original":
            topos, top_occs, all_occs = len(array), int(array["top"].sum()), int(array["all"].sum())
        else:
            transformed[mode] = len(array)
        for scope, field in enumerate(("top", "all")):
            for band in range(SHAPE_SIZE_BANDS + 1):
                weights = array[field] if band == 0 else array["bands"][:, band - 1, scope]
                curves[f"{field}:{band}"][mode] = coverage(weights)
            if mode == "original":
                for label, lower, upper in (
                    ("All sizes", 1, np.inf),
                    ("1–7", 1, 8),
                    ("8–31", 8, 32),
                    ("32–127", 32, 128),
                    ("128+", 128, np.inf),
                ):
                    included = (array["nodes"] >= lower) & (array["nodes"] < upper)
                    curve = coverage(array[field][included])
                    if curve["rank"]:
                        base[field][label] = curve
    return TopoStats(TopoSummary(topos, top_occs, all_occs, transformed), curves, base)


def pattern_stats(heads: list[str], arr: CountArray) -> PatternStats:
    descrs: dict[str, PatternDescr] = {}
    rows: list[HeadPatternRow] = []
    points: dict[int, list[tuple[int, int]]] = {}
    cohorts = (
        (PatternCol.TOP, "top", "all"),
        (PatternCol.ALL, "all", "all"),
        (PatternCol.AFFECTED_TOP, "top", "affected"),
        (PatternCol.AFFECTED_ALL, "all", "affected"),
    )
    head_ids = arr[:, PatternCol.HEAD]
    unnamed, named = head_ids == 0, head_ids > 0
    for flavour, label in enumerate(("topology", "constructors")):
        included = arr[:, PatternCol.FLAVOUR] == flavour
        global_rows = included & unnamed
        for col, scope, cohort in cohorts:
            descrs[f"{label}:{scope}:{cohort}"] = pattern_descr(arr[:, col][global_rows])
        grouped = arr[included & named]
        grouped = grouped[np.argsort(grouped[:, PatternCol.HEAD], kind="stable")]
        points[flavour] = []
        if not len(grouped):
            continue
        head_ids = grouped[:, PatternCol.HEAD]
        starts = np.r_[0, np.flatnonzero(head_ids[1:] != head_ids[:-1]) + 1]
        distrs = _head_distrs(grouped[:, PatternCol.TOP :], starts)
        for head_id, counts, totals, entropies in zip(
            head_ids[starts], distrs.counts.tolist(), distrs.totals.tolist(), distrs.entropies.tolist(), strict=True
        ):
            head = heads[int(head_id)]
            for (_, scope, cohort), count, total, entropy in zip(cohorts, counts, totals, entropies, strict=True):
                rows.append((label, scope, cohort, head, count, total, entropy))
            if totals[-1]:
                points[flavour].append((totals[-1], counts[-1]))
    return PatternStats(descrs, rows, points)


def head_stats(heads: Mapping[ViewMode, Counter[str]]) -> dict[ViewMode, HeadStats]:
    result: dict[ViewMode, HeadStats] = {}
    for mode, hgram in heads.items():
        counts = np.sort(np.fromiter(hgram.values(), dtype=np.int64))[::-1]
        total = int(counts.sum())
        result[mode] = {
            "counts": counts.tolist(),
            "coverage": (np.cumsum(counts) / total).tolist() if total else [],
            "named_heads": sum(not name.startswith("<") for name in hgram),
            "app_anchors": total,
        }
    return result


# Embedding preparation and atlas selection.


def embedding_inputs(
    features: FloatArray, names: list[str], size_aware: bool
) -> tuple[NDArray[np.floating], list[str]]:
    """Square-root histogram geometry with equal total weight per feature group."""

    groups: dict[str, list[int]] = {}
    prefixes = ("frac", "edge", "motif", "layer", "spine", "sharing", "binder_ref", "binder_nesting")
    for idx, name in enumerate(names[2:], 2):
        group = next(prefix for prefix in prefixes if name.startswith(prefix + "_"))
        groups.setdefault(group, []).append(idx)
    blocks: list[NDArray[np.floating]] = []
    cols: list[int] = []
    for idxs in groups.values():
        block = np.sqrt(features[:, idxs]).astype(np.float32)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        blocks.append(block / np.maximum(norms, 1e-12))
        cols.extend(idxs)
    if size_aware:
        blocks.append(StandardScaler().fit_transform(features[:, :2]).astype(np.float32) / np.sqrt(2))
        cols.extend((0, 1))
    return np.concatenate(blocks, axis=1) / np.sqrt(len(blocks)), [names[idx] for idx in cols]


def embeddings(rows: list[StralRow], workers: int) -> dict[str, dict[str, NDArray]]:
    if len(rows) < 4:
        return {}
    from numba import get_num_threads
    from threadpoolctl import threadpool_limits
    from umap import UMAP

    features, names = stral_features(rows)
    threads = min(workers, get_num_threads())
    result: dict[str, dict[str, NDArray]] = {}
    for mode in ("size-aware", "shape-only"):
        inputs, input_names = embedding_inputs(features, names, mode == "size-aware")
        with threadpool_limits(limits=threads):
            coords = UMAP(
                n_components=2, n_neighbors=min(30, len(rows) - 1), min_dist=0.15, init="random", n_jobs=threads
            ).fit_transform(inputs)
        result[mode] = {
            "coordinates": coords,
            "features": features,
            "inputs": inputs,
            "feature_names": np.asarray(names),
            "input_names": np.asarray(input_names),
            "theorems": np.asarray([row.theorem for row in rows]),
            "roots": np.asarray([row.root for row in rows]),
            "algorithm": np.asarray("umap"),
        }
    return result


def atlas_examples(counts: Mapping[bytes, ShapeCount]) -> dict[bytes, ShapeCount]:
    """Retain common shapes in each size band plus large paired-atlas candidates.

    Selection is mergeable across batches; it never bounds corpus statistics.
    """
    selected: dict[bytes, ShapeCount] = {}
    for lower, upper in ((3, 8), (8, 32), (32, 128), (128, float("inf"))):
        for field in ("top", "all"):
            cands = (
                (key, item)
                for key, item in counts.items()
                if lower <= item.nodes < upper and (lower != 128 or item.top)
            )
            if lower == 128:
                score = lambda pair: (-pair[1].nodes, pair[1].theorem, pair[1].root)
            elif field == "top":
                score = lambda pair: (-pair[1].top, -pair[1].all, pair[1].theorem, pair[1].root)
            else:
                score = lambda pair: (-pair[1].all, pair[1].theorem, pair[1].root)
            selected.update(heapq.nsmallest(16 if lower == 128 else 4, cands, key=score))
    return selected
