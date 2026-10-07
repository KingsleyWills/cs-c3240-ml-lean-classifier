"""Prepare expression graphs and optional derived views; no corpus I/O or aggregation."""

from __future__ import annotations

import hashlib
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache
from itertools import pairwise
from types import MappingProxyType
from typing import Literal

import msgspec
import numpy as np
from graph_tool import Graph, GraphView
from graph_tool.search import bfs_iterator
from graph_tool.topology import label_out_component, topological_sort
from numpy.typing import NDArray

from trustmebro.extraction import records as r
from trustmebro.extraction.storage import encode_msgpack

from .measurements import GraphSize, IntArray, ViewMode

DEFAULT_REACH_BUDGET = 64 * 1024 * 1024

# Graph, view, and observation records.


class RootMeasure(msgspec.Struct, frozen=True):
    """Cached per-root measurements use compact, native-constructed records."""

    expanded: int
    distinct: int
    depth: int
    constrs: dict[str, int]


@dataclass(frozen=True)
class ExprGraph:
    """Ordered structural DAG; expression IDs remain those of the raw table."""

    exprs: tuple[r.Expr, ...]
    edges: list[tuple[int, ...]]
    graph: Graph
    order: tuple[int, ...]
    original: bool


class RootFragment(msgspec.Struct, frozen=True):
    """Canonical positions refer back to source nodes; None marks an open boundary."""

    nodes: tuple[int, ...]
    edges: tuple[tuple[int, ...] | None, ...]


@dataclass(slots=True)
class RootPatterns:
    """Signatures plus optional occurrence mappings from the same traversal."""

    sigs: dict[int, tuple[bytes, bytes]]
    fragments: dict[int, RootFragment]


@dataclass(frozen=True)
class GraphStats:
    """Exact implicit-tree sizes and depths, aligned to expression IDs."""

    sizes: list[int]
    depths: list[int]


@dataclass
class ReachBits:
    """LRU of retained integer bitmaps, shared only by simultaneously live graphs.

    The budget counts integer storage, not graphs, dictionary overhead, root
    measurements, temporary native masks, or total worker memory.
    """

    budget: int = DEFAULT_REACH_BUDGET
    used: int = 0
    entries: OrderedDict[tuple[int, int], int] = field(default_factory=OrderedDict)

    def get(self, graph: Graph, root: int) -> int:
        if self.budget < 0:
            raise ValueError("reachability bitmap budget must be nonnegative")
        key = (id(graph), root)
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        reachable = label_out_component(graph, graph.vertex(root))
        bits = int.from_bytes(np.packbits(reachable.a, bitorder="little").tobytes(), "little")
        size = sys.getsizeof(bits)
        if size <= self.budget:
            while self.used + size > self.budget:
                _, previous = self.entries.popitem(last=False)
                self.used -= sys.getsizeof(previous)
            self.entries[key] = bits
            self.used += size
        return bits


@dataclass
class ReachCache:
    """Scratch owned by one immutable graph, never shared between derived views."""

    graph: ExprGraph
    bitmaps: ReachBits = field(default_factory=ReachBits)
    measures: dict[tuple[int, ...], RootMeasure] = field(default_factory=dict)


@dataclass(frozen=True)
class ExprArrays:
    """Constructor-specific arrays for the exported expression graph."""

    children: IntArray
    offsets: IntArray
    names: tuple[str, ...]
    kinds: IntArray
    body_slots: IntArray
    binder_counts: IntArray
    var_bins: IntArray
    spines: IntArray
    ranks: IntArray


class AppHeads(msgspec.Struct, frozen=True):
    refs: IntArray
    arities: IntArray


class RootGraph(msgspec.Struct, frozen=True):
    """Root-local children/order/indegrees; nodes maps back to theorem IDs."""

    nodes: IntArray
    children: IntArray
    offsets: IntArray
    order: IntArray
    incoming: IntArray


@dataclass(frozen=True)
class TopoView:
    graph: ExprGraph
    # Source IDs retain each binder's original payload and scope.
    binders: tuple[tuple[int, ...], ...]
    aliases: tuple[int, ...] = ()
    markers: dict[int, Marker] = field(default_factory=dict)


class ViewArrays(msgspec.Struct, frozen=True):
    degrees: IntArray
    binders: IntArray


class SignatureReuse(msgspec.Struct, frozen=True):
    """Safe inheritance from the preceding view, indexed by resolved source IDs."""

    src: ViewMode
    topo: NDArray[np.bool_]
    labelled: NDArray[np.bool_]


type PatternSigs = dict[int, dict[int, tuple[bytes, bytes]]]


class Observations(msgspec.Struct, frozen=True):
    """Per-node uses: `top` counts root occurrences; `all` counts state DAGs.

    A shared node contributes once per state, unlike frequency `root_dag`,
    which counts once per top-level-root occurrence within that state.
    """

    states: Counter[tuple[int, ...]]
    top: IntArray
    all: IntArray


# Transformation rules and owned scratch data.


class Role(StrEnum):
    TYPE = "type"
    FAMILY = "type-family"
    INST = "instance"
    VAL = "value"


class Kind(StrEnum):
    OPERATOR = "operator"
    NUM = "numeral"
    FN_COE = "function-coercion"
    VAL_COE = "value-coercion"
    TYPE_COE = "type-coercion"
    SET_COE = "set-coercion"
    PROJ = "projection"
    INST_CONSTR = "instance-construction"


class OpName(StrEnum):
    TYPE = "type"
    INST = "instance"
    VAL = "value"
    LHS = "lhs"
    RHS = "rhs"
    SRC_TYPE = "source_type"
    DST_TYPE = "target_type"
    LHS_TYPE = "lhs_type"
    RHS_TYPE = "rhs_type"
    RES_TYPE = "result_type"
    ELEM_TYPE = "element_type"
    CONTAINER_TYPE = "container_type"
    CONTAINER = "container"
    ELEM = "element"
    NUM = "numeral"
    OBJ_TYPE = "object_type"
    DOMAIN = "domain"
    CODOMAIN = "codomain"
    OBJ = "object"
    PRED = "predicate"
    FN = "function"


class Policy(StrEnum):
    INSTS = "instances"
    COMPACT = "coercions-compact"
    ERASED = "coercions-erased"


class NodeDispo(StrEnum):
    RETAINED = "retained"
    ERASED = "erased"


@dataclass(frozen=True, slots=True)
class Slot:
    name: OpName
    role: Role


@dataclass(frozen=True, slots=True)
class FixedOp:
    name: OpName
    val: str


class OpRef(msgspec.Struct, frozen=True):
    name: OpName
    ref: int


@dataclass(frozen=True, slots=True)
class Rule:
    head: str
    kind: Kind
    slots: tuple[Slot, ...]
    # Fixed cast endpoints/numerals have no corresponding expression argument.
    fixed: tuple[FixedOp, ...] = ()


class Match(msgspec.Struct, frozen=True):
    rule: Rule
    root: int
    head: int
    args: tuple[int, ...]
    tail: tuple[int, ...]

    def arg(self, name: OpName) -> int:
        """Look up a semantic operand without depending on its numeric offset."""
        for slot, ref in zip(self.rule.slots, self.args, strict=True):
            if slot.name == name:
                return ref
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class RuleFamily:
    names: tuple[str, ...]
    kind: Kind
    slots: tuple[Slot, ...]
    fixed: tuple[FixedOp, ...] = ()

    def expand(self) -> tuple[Rule, ...]:
        return tuple(Rule(name, self.kind, self.slots, self.fixed) for name in self.names)


class Marker(msgspec.Struct, frozen=True):
    kind: Kind
    slots: tuple[OpRef, ...]
    fixed: tuple[FixedOp, ...]


@dataclass
class PlumbingState:
    """Mutable graph data owned by one view; transformations receive it explicitly."""

    edges: list[tuple[int, ...]]
    aliases: list[int]
    binders: list[tuple[int, ...]]
    markers: dict[int, Marker] = field(default_factory=dict)
    apps: set[int] = field(default_factory=set)


# Rule registry and processing defaults.


def _rules() -> Mapping[str, Rule]:
    """Declare shared layouts once, then expand aliases into immutable rules."""

    def family(*names: str, kind: Kind, slots: tuple[Slot, ...], fixed: tuple[FixedOp, ...] = ()) -> RuleFamily:
        return RuleFamily(names=names, kind=kind, slots=slots, fixed=fixed)

    type_ = Slot(OpName.TYPE, Role.TYPE)
    inst = Slot(OpName.INST, Role.INST)
    val = Slot(OpName.VAL, Role.VAL)
    binary = (Slot(OpName.LHS, Role.VAL), Slot(OpName.RHS, Role.VAL))
    coe = (Slot(OpName.SRC_TYPE, Role.TYPE), Slot(OpName.DST_TYPE, Role.TYPE), inst, val)
    het_binary = (
        Slot(OpName.LHS_TYPE, Role.TYPE),
        Slot(OpName.RHS_TYPE, Role.TYPE),
        Slot(OpName.RES_TYPE, Role.TYPE),
        inst,
        *binary,
    )
    # Elaborated membership places the container before the element
    membership = (
        Slot(OpName.ELEM_TYPE, Role.TYPE),
        Slot(OpName.CONTAINER_TYPE, Role.TYPE),
        inst,
        Slot(OpName.CONTAINER, Role.VAL),
        Slot(OpName.ELEM, Role.VAL),
    )
    # The fifth operand is the object; trailing operands are its call args
    dependent_fn_coe = (
        Slot(OpName.OBJ_TYPE, Role.TYPE),
        Slot(OpName.DOMAIN, Role.TYPE),
        Slot(OpName.CODOMAIN, Role.FAMILY),
        inst,
        Slot(OpName.OBJ, Role.VAL),
    )
    fn_coe = (type_, Slot(OpName.CODOMAIN, Role.FAMILY), inst, Slot(OpName.OBJ, Role.VAL))
    set_coe = (Slot(OpName.OBJ_TYPE, Role.TYPE), Slot(OpName.ELEM_TYPE, Role.TYPE), inst, Slot(OpName.OBJ, Role.VAL))
    subtype_proj = (type_, Slot(OpName.PRED, Role.FAMILY), Slot(OpName.OBJ, Role.VAL))
    cast = (Slot(OpName.DST_TYPE, Role.TYPE), inst, val)

    families: list[RuleFamily] = [
        # Operators
        family(
            "HAdd.hAdd",
            "HMul.hMul",
            "HSub.hSub",
            "HDiv.hDiv",
            "HPow.hPow",
            "HSMul.hSMul",
            kind=Kind.OPERATOR,
            slots=het_binary,
        ),
        family("Neg.neg", "Inv.inv", kind=Kind.OPERATOR, slots=(type_, inst, val)),
        family("LE.le", "LT.lt", kind=Kind.OPERATOR, slots=(type_, inst, *binary)),
        family("Membership.mem", kind=Kind.OPERATOR, slots=membership),
        # Numerals
        family("OfNat.ofNat", kind=Kind.NUM, slots=(type_, Slot(OpName.NUM, Role.VAL), inst)),
        family("Zero.zero", kind=Kind.NUM, slots=(type_, inst), fixed=(FixedOp(OpName.NUM, "0"),)),
        family("One.one", kind=Kind.NUM, slots=(type_, inst), fixed=(FixedOp(OpName.NUM, "1"),)),
        # Coercions and projections
        family("DFunLike.coe", kind=Kind.FN_COE, slots=dependent_fn_coe),
        family("CoeFun.coe", kind=Kind.FN_COE, slots=fn_coe),
        family("Coe.coe", "CoeTC.coe", "CoeHTCT.coe", kind=Kind.VAL_COE, slots=coe),
        family("CoeSort.coe", kind=Kind.TYPE_COE, slots=coe),
        family("SetLike.coe", kind=Kind.SET_COE, slots=set_coe),
        family("Subtype.val", kind=Kind.PROJ, slots=subtype_proj),
    ]
    families.extend(
        family(
            f"{src}.cast",
            f"{src}Cast.{src.lower()}Cast",
            kind=Kind.VAL_COE,
            slots=cast,
            fixed=(FixedOp(OpName.SRC_TYPE, src),),
        )
        for src in ("Nat", "Int")
    )
    families.extend(
        [
            family(
                "Int.ofNat",
                kind=Kind.VAL_COE,
                slots=(val,),
                fixed=(FixedOp(OpName.SRC_TYPE, "Nat"), FixedOp(OpName.DST_TYPE, "Int")),
            ),
            # Only these individually checked two-slot instance layouts are eligible
            family(
                "CommSemiring.toSemiring",
                "PartialOrder.toPreorder",
                "AddCommGroup.toAddCommMonoid",
                "Semiring.toNonAssocSemiring",
                "Preorder.toLE",
                "CommRing.toCommSemiring",
                "Zero.toOfNat0",
                "One.toOfNat1",
                "instHMul",
                "instHAdd",
                kind=Kind.INST_CONSTR,
                slots=(type_, inst),
            ),
        ]
    )
    expanded = tuple(rule for family in families for rule in family.expand())
    names = Counter(rule.head for rule in expanded)
    if dupes := [name for name, count in names.items() if count > 1]:
        raise ValueError(f"duplicate plumbing rules: {dupes}")
    return MappingProxyType({rule.head: rule for rule in expanded})


RULES = _rules()
POLICIES = tuple(Policy)


@dataclass(frozen=True)
class ProcessCfg:
    reg: Mapping[str, Rule] = field(default_factory=_rules)
    policies: tuple[Policy, ...] = POLICIES


DEFAULT_PROCESSING = ProcessCfg(RULES)


# Original graph preparation and root measurements.


def bitmap_nodes(bits: int) -> IntArray:
    """Decode only the occupied prefix, not a theorem-wide scratch array."""
    packed = bits.to_bytes((bits.bit_length() + 7) // 8, "little")
    return np.flatnonzero(np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little"))


def reachable_from(graph: Graph, roots: Sequence[int]) -> NDArray[np.bool_]:
    """Accumulate native root closures without a graph copy or Python bit shifts.

    A previously reached seed can be skipped because its descendants were
    included by the completed traversal that reached it. Return owned storage.
    """
    reached = graph.new_vertex_property("bool")
    for root in roots:
        if not reached[root]:
            label_out_component(graph, graph.vertex(root), label=reached)
    # graph_tool exposes Boolean properties as uint8; indexing needs bool dtype.
    return reached.a.astype(np.bool_, copy=True)


def build_graph(exprs: tuple[r.Expr, ...], *, edges: list[tuple[int, ...]] | None = None) -> ExprGraph:
    """Build and validate native topology, without preparing any measurements."""
    original = edges is None
    edges = [r.expr_refs(expr) for expr in exprs] if edges is None else edges
    graph = Graph(directed=True)
    graph.add_vertex(len(exprs))
    refs = np.array([(parent, child) for parent, nodes in enumerate(edges) for child in nodes], dtype=np.int64).reshape(
        -1, 2
    )
    if refs.size:
        if refs.min() < 0 or refs.max() >= len(exprs):
            raise ValueError("invalid expression reference")
        graph.add_edge_list(refs)
    return ExprGraph(exprs, edges, graph, tuple(int(node) for node in topological_sort(graph)), original)


def graph_stats(graph: ExprGraph) -> GraphStats:
    """One bottom-up pass; expanded sizes may exceed native integer widths."""
    sizes, depths = [0] * len(graph.exprs), [0] * len(graph.exprs)
    for node in reversed(graph.order):
        sizes[node] = 1 + sum(sizes[child] for child in graph.edges[node])
        depths[node] = 1 + max((depths[child] for child in graph.edges[node]), default=0)
    return GraphStats(sizes, depths)


def constructor_masks(exprs: Sequence[r.Expr]) -> dict[str, int]:
    names = np.asarray([type(expr).__name__ for expr in exprs])
    return {
        str(kind): int.from_bytes(np.packbits(names == kind, bitorder="little").tobytes(), "little")
        for kind in sorted(set(names))
    }


def expr_arrays(graph: ExprGraph) -> ExprArrays:
    """Prepare descriptors once, only for theorems with selected structural roots."""
    if not graph.original:
        raise ValueError("constructor-specific descriptors require the original expression graph")
    # Ordered CSR avoids a theorem-size × maximum-arity padded matrix.
    degrees = np.fromiter(map(len, graph.edges), dtype=np.int64, count=len(graph.exprs))
    offsets = np.concatenate(([0], degrees.cumsum()))
    children = np.fromiter((ref for refs in graph.edges for ref in refs), dtype=np.int64, count=int(offsets[-1]))
    body_slots = np.full(len(graph.exprs), -1, dtype=np.int64)
    binder_counts = np.zeros(len(graph.exprs), dtype=np.int64)
    var_bins = np.full(len(graph.exprs), -1, dtype=np.int64)
    spines = np.zeros(len(graph.exprs), dtype=np.int64)
    ranks = np.empty(len(graph.exprs), dtype=np.int64)
    ranks[np.asarray(graph.order, dtype=np.int64)] = np.arange(len(graph.exprs))
    for node in reversed(graph.order):
        expr = graph.exprs[node]
        if isinstance(expr, (r.Lambda, r.Forall)):
            body_slots[node] = 1
            binder_counts[node] = len(expr.names)
        elif isinstance(expr, r.Let):
            body_slots[node] = 2
            binder_counts[node] = 1
        elif isinstance(expr, r.Bvar):
            var_bins[node] = min(4, expr.idx.bit_length())
        elif isinstance(expr, r.App):
            spines[node] = len(expr.args) + spines[expr.fn]
    names, codes = np.unique([type(expr).__name__ for expr in graph.exprs], return_inverse=True)
    return ExprArrays(
        children, offsets, tuple(str(name) for name in names), codes, body_slots, binder_counts, var_bins, spines, ranks
    )


def root_bitmap(cache: ReachCache, root: int) -> int:
    return cache.bitmaps.get(cache.graph.graph, root)


def root_union(cache: ReachCache, roots: Sequence[int]) -> int:
    bits = 0
    for root in roots:
        bits |= root_bitmap(cache, root)
    return bits


def reachable_nodes(cache: ReachCache, roots: Sequence[int]) -> IntArray:
    return bitmap_nodes(root_union(cache, roots))


def root_graph(cache: ReachCache, arrays: ExprArrays, root: int) -> RootGraph:
    nodes = reachable_nodes(cache, (root,))
    degrees = np.diff(arrays.offsets)[nodes]
    offsets = np.concatenate(([0], degrees.cumsum()))
    positions = np.repeat(arrays.offsets[nodes] - offsets[:-1], degrees) + np.arange(offsets[-1])
    children = np.searchsorted(nodes, arrays.children[positions])
    incoming = np.bincount(children, minlength=len(nodes))
    return RootGraph(nodes, children, offsets, np.argsort(arrays.ranks[nodes]), incoming)


def measure_roots(stats: GraphStats, masks: Mapping[str, int], cache: ReachCache, roots: Sequence[int]) -> RootMeasure:
    key = tuple(roots)
    if key not in cache.measures:
        bits = root_union(cache, roots)
        cache.measures[key] = RootMeasure(
            sum(stats.sizes[node] for node in roots),
            bits.bit_count(),
            max((stats.depths[node] for node in roots), default=0),
            {kind: (bits & mask).bit_count() for kind, mask in masks.items() if bits & mask},
        )
    return cache.measures[key]


# Derived-view measurements.


def view_arrays(view: TopoView) -> ViewArrays:
    """Prepare only for topology-size consumers, not patterns or original-state counts."""
    return ViewArrays(
        view.graph.graph.get_out_degrees(np.arange(len(view.graph.exprs))),
        np.fromiter((len(group) for group in view.binders), dtype=np.int64),
    )


def resolve_root(view: TopoView, root: int) -> int:
    return view.aliases[root] if view.aliases else root


def view_heads(view: TopoView) -> tuple[str, ...]:
    """Resolve heads once, respecting aliases and conversion boundaries."""
    graph = view.graph
    names = [""] * len(graph.exprs)
    for node in reversed(graph.order):
        expr = graph.exprs[node]
        if (marker := view.markers.get(node)) is not None:
            names[node] = f"<{marker.kind}>"
        elif isinstance(expr, r.Metadata):
            names[node] = names[resolve_root(view, expr.expr)]
        elif isinstance(expr, r.App) and graph.edges[node]:
            names[node] = names[graph.edges[node][0]]
        else:
            names[node] = expr.name if isinstance(expr, r.Const) else f"<{type(expr).__name__}>"
    return tuple(names)


def measure_view(
    view: TopoView, stats: GraphStats, cache: ReachCache, arrays: ViewArrays, roots: Sequence[int]
) -> GraphSize:
    """Union nodes/refs, but recount each root for the implicit expanded tree."""
    if not roots:
        raise ValueError("a topology measurement needs at least one root")
    roots = [resolve_root(view, root) for root in roots]
    nodes = reachable_nodes(cache, roots)
    return GraphSize(
        len(nodes),
        int(arrays.degrees[nodes].sum()),
        int(arrays.binders[nodes].sum()),
        max(stats.depths[root] for root in roots),
        sum(stats.sizes[root] for root in roots),
        int(arrays.degrees[nodes].max()) if len(nodes) else 0,
    )


# Application lookup and view transformations.


def app_spine(exprs: Sequence[r.Expr], root: int) -> tuple[int, tuple[int, ...]]:
    """Read an ordered application spine, ignoring metadata only for lookup.

    References still point into the unchanged raw table. Metadata is not deleted
    from that table, and callers retain the original root for provenance.
    """
    groups: list[tuple[int, ...]] = []
    head = root
    while True:
        match exprs[head]:
            case r.App(fn=fn, args=args):
                groups.append(args)
                head = fn
            case r.Metadata(expr=ref):
                head = ref
            case _:
                return head, tuple(ref for group in reversed(groups) for ref in group)


def app_heads(graph: ExprGraph) -> AppHeads:
    """One DAG pass resolves heads/arity, without materializing every spine."""
    refs = np.arange(len(graph.exprs), dtype=np.int64)
    arities = np.zeros(len(refs), dtype=np.int64)
    for node in reversed(graph.order):
        match graph.exprs[node]:
            case r.App(fn=fn, args=args):
                refs[node], arities[node] = refs[fn], arities[fn] + len(args)
            case r.Metadata(expr=expr):
                refs[node], arities[node] = refs[expr], arities[expr]
    return AppHeads(refs, arities)


def match_rule(
    exprs: Sequence[r.Expr],
    root: int,
    *,
    reg: Mapping[str, Rule] = RULES,
    inst_ctxt: bool = False,
    spine: tuple[int, tuple[int, ...]] | None = None,
) -> Match | None:
    """Match exact named heads; leave unknown/partially applied heads alone.

    Oversaturated heads retain all trailing arguments. Instance construction
    rules are diagnostic and only eligible within a verified instance slot.
    """
    head, args = app_spine(exprs, root) if spine is None else spine
    expr = exprs[head]
    if not isinstance(expr, r.Const):
        return None
    rule = reg.get(expr.name)
    if rule is None or (rule.kind == Kind.INST_CONSTR and not inst_ctxt):
        return None
    arity = len(rule.slots)
    if len(args) < arity:
        return None
    return Match(rule, root, head, args[:arity], args[arity:])


def find_matches(exprs: tuple[r.Expr, ...], heads: AppHeads, reg: Mapping[str, Rule] = RULES) -> dict[int, Match]:
    matches: dict[int, Match] = {}
    for node, expr in enumerate(exprs):
        head = exprs[heads.refs[node]]
        rule = reg.get(head.name) if isinstance(head, r.Const) else None
        if (
            isinstance(expr, r.App)
            and rule is not None
            and rule.kind != Kind.INST_CONSTR
            and heads.arities[node] >= len(rule.slots)
        ):
            matched = match_rule(exprs, node, reg=reg)
            if matched is not None:
                matches[node] = matched
    return matches


def _initial_state(base: TopoView) -> PlumbingState:
    return PlumbingState(
        edges=base.graph.edges.copy(), aliases=list(range(len(base.graph.exprs))), binders=list(base.binders)
    )


def _apply_rule(state: PlumbingState, node: int, match: Match | None, policy: Policy) -> NodeDispo:
    if match is None:
        return NodeDispo.RETAINED
    conv = match.rule.kind in (Kind.FN_COE, Kind.VAL_COE, Kind.TYPE_COE, Kind.SET_COE, Kind.PROJ)
    if conv and policy == Policy.ERASED:
        name = OpName.OBJ if match.rule.kind in (Kind.FN_COE, Kind.SET_COE, Kind.PROJ) else OpName.VAL
        val = match.arg(name)
        if not match.tail:
            state.aliases[node] = state.aliases[val]
            state.edges[node] = ()  # No unary proxy remains reachable.
            state.binders[node] = ()
            return NodeDispo.ERASED
        state.edges[node] = (val, *match.tail)
        state.markers[node] = Marker(Kind.OPERATOR, (OpRef(OpName.FN, state.aliases[val]),), ())
        return NodeDispo.RETAINED

    retained = tuple(
        OpRef(slot.name, ref) for slot, ref in zip(match.rule.slots, match.args, strict=True) if slot.role != Role.INST
    )
    operands = tuple(op.ref for op in retained)
    if conv and policy == Policy.COMPACT:
        state.edges[node] = (*operands, *match.tail)
        state.markers[node] = Marker(
            match.rule.kind, tuple(OpRef(op.name, state.aliases[op.ref]) for op in retained), match.rule.fixed
        )
    else:
        state.edges[node] = (match.head, *operands, *match.tail)
    return NodeDispo.RETAINED


def _redirect_children(state: PlumbingState, node: int, exprs: Sequence[r.Expr], policy: Policy) -> None:
    # Children are already processed; binders/lets need redirects too.
    state.edges[node] = tuple(state.aliases[ref] for ref in state.edges[node])
    expr = exprs[node]
    if isinstance(expr, r.Metadata) and policy == Policy.ERASED and state.aliases[expr.expr] != expr.expr:
        state.aliases[node] = state.aliases[expr.expr]
        state.edges[node] = ()


def _flatten_app(state: PlumbingState, node: int, exprs: Sequence[r.Expr]) -> None:
    marker = state.markers.get(node)
    is_app = isinstance(exprs[node], r.App) and (marker is None or marker.kind == Kind.OPERATOR)
    if not is_app or not state.edges[node]:
        return
    fn, *args = state.edges[node]
    # Erasure can expose a new spine. Conversion markers are not applications:
    # their first edge describes a type/value role, not a function.
    if fn in state.apps:
        state.edges[node] = (*state.edges[fn], *args)
    state.apps.add(node)


def _finish_view(base: TopoView, state: PlumbingState) -> TopoView:
    return TopoView(
        build_graph(base.graph.exprs, edges=state.edges),
        tuple(state.binders),
        tuple(state.aliases),
        markers=state.markers,
    )


def _plumbing_view(base: TopoView, policy: Policy, matches: Mapping[int, Match]) -> TopoView:
    state = _initial_state(base)
    for node in reversed(base.graph.order):
        if _apply_rule(state, node, matches.get(node), policy) == NodeDispo.ERASED:
            continue
        _redirect_children(state, node, base.graph.exprs, policy)
        _flatten_app(state, node, base.graph.exprs)
    return _finish_view(base, state)


def prepare_views(
    graph: ExprGraph, modes: Sequence[ViewMode], *, matches: Mapping[int, Match], cfg: ProcessCfg = DEFAULT_PROCESSING
) -> dict[ViewMode, TopoView]:
    """Build selected views and shared prerequisites once; no measurements or observations."""
    if not modes:
        return {}
    binders = tuple(
        (node,) * len(expr.names) if isinstance(expr, (r.Forall, r.Lambda)) else ()
        for node, expr in enumerate(graph.exprs)
    )
    base = TopoView(graph, binders)
    views = {ViewMode.ORIGINAL: base}
    for mode in modes:
        if mode == ViewMode.ORIGINAL:
            continue
        policy = Policy(mode)
        if policy not in cfg.policies:
            raise ValueError(f"disabled plumbing policy: {policy}")
        views[mode] = _plumbing_view(base, policy, matches)
    return {mode: views[mode] for mode in modes}


def expr_views(graph: ExprGraph, *, cfg: ProcessCfg = DEFAULT_PROCESSING) -> tuple[TopoView, ...]:
    modes = (ViewMode.ORIGINAL, *(ViewMode(policy) for policy in cfg.policies))
    matches = find_matches(graph.exprs, app_heads(graph), cfg.reg) if cfg.policies else {}
    return tuple(prepare_views(graph, modes, matches=matches, cfg=cfg).values())


# Observed occurrences and affected nodes.


def observe_roots(cache: ReachCache, trns: Sequence[r.Trn]) -> Observations:
    states = Counter((trn.state.target, *(local.type for local in trn.state.locals)) for trn in trns)
    top = np.zeros(len(cache.graph.exprs), dtype=np.int64)
    all_ = np.zeros_like(top)
    for roots, weight in states.items():
        np.add.at(top, np.asarray(roots, dtype=np.int64), weight)
        all_[reachable_nodes(cache, roots)] += weight
    return Observations(states, top, all_)


def affected_nodes(graph: ExprGraph, matches: Mapping[int, Match]) -> NDArray[np.bool_]:
    conv_kinds = (Kind.FN_COE, Kind.VAL_COE, Kind.TYPE_COE, Kind.SET_COE, Kind.PROJ)
    roots = [node for node, matched in matches.items() if matched.rule.kind in conv_kinds]
    return reachable_from(GraphView(graph.graph, reversed=True), roots)


def observed_heads(exprs: Sequence[r.Expr], heads: AppHeads, observations: Observations) -> dict[int, str]:
    result: dict[int, str] = {}
    for root in np.flatnonzero(observations.all):
        if isinstance(exprs[root], r.App):
            expr = exprs[heads.refs[root]]
            result[int(root)] = expr.name if isinstance(expr, r.Const) else f"<{type(expr).__name__}>"
    return result


# Graph and expression identities.


def signature_reuse(views: Mapping[ViewMode, TopoView]) -> dict[ViewMode, SignatureReuse]:
    """Propagate changed edges/labels to ancestors once, not per signature root.

    Unchanged rooted regions retain identical ordered adjacency and node labels.
    Root aliases are resolved by consumers before lookup; measurement/binder
    caches are deliberately not shared by this identity-only preparation.
    """
    result: dict[ViewMode, SignatureReuse] = {}
    for (src, base), (mode, view) in pairwise(views.items()):
        changed = np.fromiter((a != b for a, b in zip(base.graph.edges, view.graph.edges, strict=True)), dtype=bool)
        reverse = GraphView(view.graph.graph, reversed=True)
        topo = ~reachable_from(reverse, np.flatnonzero(changed).tolist())
        # Constructors are shared by the source table. Only non-operator markers
        # change the labels used by pattern identities.
        labels = [
            node
            for node in base.markers.keys() | view.markers.keys()
            if _pattern_label(base, node) != _pattern_label(view, node)
        ]
        labelled = topo & ~reachable_from(reverse, labels) if labels else topo
        result[mode] = SignatureReuse(src, topo, labelled)
    return result


class Var(msgspec.Struct, frozen=True):
    kind: Literal["free", "meta", "level"]
    id: int


type Data = str | int | Var | tuple[Data, ...]


def data_digest(data: object) -> bytes:
    """Ordinary MessagePack identity; arbitrary natural numbers use encode_msgpack instead."""
    return hashlib.sha256(msgspec.msgpack.encode(data)).digest()


def _root_order(graph: ExprGraph, root: int) -> tuple[list[int], dict[int, int]]:
    """Native BFS discovers each vertex once; convert its destination column in C.

    Only numbering uses the discovery tree. Descriptors retain the full ordered
    adjacency, including parallel edges and references to already-seen nodes.
    """
    tree_edges = bfs_iterator(graph.graph, root, array=True)
    order = [root, *tree_edges[:, 1].tolist()]
    return order, dict(zip(order, range(len(order))))


def shape_sig(graph: ExprGraph, root: int) -> bytes:
    """Constructor-labelled DAG ident for the bounded atlas; scalar data is ignored."""
    order, nums = _root_order(graph, root)
    descr = [(type(graph.exprs[node]).__name__, [nums[child] for child in graph.edges[node]]) for node in order]
    return data_digest(descr)


def topo_sig(graph: ExprGraph, root: int) -> tuple[bytes, int]:
    """Canonical rooted, ordered DAG topology; preserve sharing but erase labels.

    A breadth-first numbering is stable because child slots are ordered. Two
    different leaves stay different vertices even though both are unlabelled.
    """
    if not graph.edges[root]:
        return _leaf_sig("")[0], 1
    order, nums = _root_order(graph, root)
    descr = [[nums[child] for child in graph.edges[node]] for node in order]
    return data_digest(descr), len(order)


def _lvl(lvl: r.Lvl) -> Data:
    match lvl:
        case r.LvlMvar(id=id):
            return Var("level", id)
        case r.LvlZero():
            return ("zero",)
        case r.LvlSucc(level=child):
            return ("succ", _lvl(child))
        case r.LvlMax(left=a, right=b) | r.LvlIMax(left=a, right=b):
            return (type(lvl).__name__, _lvl(a), _lvl(b))
        case r.LvlParam(name=name):
            return ("param", name)


def _data(expr: r.Expr) -> Data:
    """Keep scalar information; child references become structural hashes."""
    match expr:
        case r.Fvar(id=id):
            fields = (Var("free", id),)
        case r.ExprMvar(id=id):
            fields = (Var("meta", id),)
        case r.Sort(lvl=lvl):
            fields = (_lvl(lvl),)
        case r.Const(name=name, universes=universes):
            fields = (name, tuple(_lvl(lvl) for lvl in universes))
        case r.Bvar(idx=idx):
            fields = (idx,)
        case r.Lambda(names=names, binder_info=info) | r.Forall(names=names, binder_info=info):
            fields = (names[0] if len(names) == 1 else names, info.value)
        case r.Let(name=name, nondep=nondep):
            fields = (name, nondep)
        case r.NatLiteral(val=val):
            fields = (val,)
        case r.StringLiteral(val=val):
            fields = (val,)
        case r.Metadata(data=data):
            fields = (data,)
        case r.Proj(type_name=name, idx=idx):
            fields = (name, idx)
        case r.App():
            fields = ()
    return (type(expr).__name__, *fields)


def _vars(data: Data) -> Iterator[Var]:
    if isinstance(data, Var):
        yield data
    elif isinstance(data, tuple):
        for val in data:
            yield from _vars(val)


def _rename(data: Data, names: dict[Var, int]) -> Data:
    if isinstance(data, Var):
        return (data.kind, names[data])
    if isinstance(data, tuple):
        return tuple(_rename(val, names) for val in data)
    return data


class Idents:
    """Root-relative renaming, preserving equality and variable kinds.

    Memo keys include the variable numbering inherited from the parent: reusing
    a child's *standalone* hash would incorrectly equate `f x x` and `f x y`.
    Ground subexpressions have no numbering and are hashed only once.
    """

    def __init__(self, graph: ExprGraph, active: NDArray[np.bool_] | None = None):
        self.graph = graph
        if active is not None and (active.shape != (len(graph.exprs),) or active.dtype != np.bool_):
            raise ValueError("identity population must be a graph-aligned Boolean mask")
        # An optional descendant-closed population avoids preparing discarded
        # auxiliary expressions. Missing children are rejected during hashing.
        self.data = [_data(expr) if active is None or active[node] else None for node, expr in enumerate(graph.exprs)]
        self.vars: list[tuple[Var, ...]] = [()] * len(graph.exprs)
        for node in reversed(graph.order):
            data = self.data[node]
            if data is None:
                continue
            vars = dict.fromkeys(_vars(data))
            for child in graph.edges[node]:
                vars.update(dict.fromkeys(self.vars[child]))
            self.vars[node] = tuple(vars)
        self.cache: dict[tuple[int, tuple[int, ...]], bytes] = {}

    def sig(self, root: int) -> bytes:
        key = (root, tuple(range(len(self.vars[root]))))
        pending = [key]
        while pending:
            current = pending[-1]
            if current in self.cache:
                pending.pop()
                continue
            node, numbering = current
            data = self.data[node]
            if data is None:
                raise ValueError("expression outside the prepared identity population")
            names = dict(zip(self.vars[node], numbering, strict=True))
            children = [(child, tuple(names[var] for var in self.vars[child])) for child in self.graph.edges[node]]
            missing = [child for child in children if child not in self.cache]
            if missing:
                pending.extend(missing)
                continue
            digest = hashlib.sha256(encode_msgpack(_rename(data, names)))
            for child in children:
                digest.update(self.cache[child])
            self.cache[current] = digest.digest()
            pending.pop()
        return self.cache[key]


@cache
def _leaf_sig(label: str) -> tuple[bytes, bytes]:
    topo = [()]
    return _pattern_digest(topo, [label])


def _pattern_label(view: TopoView, node: int) -> str:
    marker = view.markers.get(node)
    return str(marker.kind) if marker and marker.kind != Kind.OPERATOR else type(view.graph.exprs[node]).__name__


def _pattern_digest(topo: object, labels: list[str]) -> tuple[bytes, bytes]:
    """Encode topology once; Raw embeds those exact bytes in the labelled identity."""
    encoded = msgspec.msgpack.encode(topo)
    return hashlib.sha256(encoded).digest(), data_digest((msgspec.Raw(encoded), labels))


def extract_patterns(view: TopoView, root: int, radii: Sequence[int], *, retain: bool = False) -> RootPatterns:
    """Canonical BFS numbering preserves sharing, without expanding a tree.

    Frontier vertices are distinct wildcard vertices: repeated edges to the
    same frontier still share it. A true leaf is not a wildcard boundary.
    Visit once to the largest requested radius. BFS prefixes retain exactly the
    same numbering as separate traversals; each smaller radius replaces its
    non-leaf frontier with wildcards before hashing the unchanged wire shape.
    These are fragment identities, not wildcard/VF2 matches. Retain source-node
    correspondence only for vocabulary consumers; histogram consumers need hashes.
    """
    root = resolve_root(view, root)
    if not radii:
        return RootPatterns({}, {})
    if not view.graph.edges[root]:
        sig = _leaf_sig(type(view.graph.exprs[root]).__name__)
        fragments = dict.fromkeys(radii, RootFragment((root,), ((),))) if retain else {}
        return RootPatterns(dict.fromkeys(radii, sig), fragments)
    max_rad = max(radii)
    ord = [root]
    nums = {root: 0}
    depths = [0]
    topo: list[tuple[int, ...] | None] = []
    labels: list[str] = []
    for idx, node in enumerate(ord):
        children = view.graph.edges[node]
        boundary = depths[idx] == max_rad and bool(children)
        if boundary:
            topo.append(None)
            labels.append("*")
            continue
        edges: list[int] = []
        for child in children:
            if child not in nums:
                nums[child] = len(ord)
                ord.append(child)
                depths.append(depths[idx] + 1)
            edges.append(nums[child])
        topo.append(tuple(edges))
        labels.append(_pattern_label(view, node))
    sigs: dict[int, tuple[bytes, bytes]] = {}
    fragments: dict[int, RootFragment] = {}
    for rad in radii:
        if rad == max_rad:
            local_topo, local_labels = topo, labels
        else:
            end = bisect_right(depths, rad)
            local_topo, local_labels = topo[:end], labels[:end]
            for idx in range(bisect_left(depths, rad), end):
                if view.graph.edges[ord[idx]]:
                    local_topo[idx], local_labels[idx] = None, "*"
        sigs[rad] = _pattern_digest(local_topo, local_labels)
        if retain:
            fragments[rad] = RootFragment(tuple(ord[: len(local_topo)]), tuple(local_topo))
    return RootPatterns(sigs, fragments)
