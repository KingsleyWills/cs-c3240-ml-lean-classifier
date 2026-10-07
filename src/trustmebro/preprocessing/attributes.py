"""Raw structural summaries and deliberately categorical expression attributes.

No theorem names, local identifiers, proof search, expression normalization or
fitted scaling enter these measurements. Numerical graph preparation is shared
with pattern projection; only the DAG-depth recurrence needs a small JIT loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.sparse import csr_array, hstack

from trustmebro.extraction import records as r

from .candidates import Candidates

LEAF_ATTRS = ("bvar_0", "bvar_1", "bvar_2", "bvar_3_plus", "nat_0", "nat_1", "nat_other", "sort_prop")
BINDER_ATTRS = tuple(
    attr for kind in ("lambda", "forall") for attr in (f"{kind}_binders", *(f"{kind}_{info}" for info in r.BinderInfo))
)
ROOT_FIELDS = ("distinct", "operand_refs", "depth", "shared_nodes", "shared_frac", "operands", "binders", "is_false")
ROOT_SCALARS = len(ROOT_FIELDS)
SHARED_FRAC = ROOT_FIELDS.index("shared_frac")
SLOT_FLAGS = ("present", "is_let", "is_instance")


@dataclass(frozen=True, slots=True)
class Projection:
    """Theorem-local coordinate maps and native root/state incidence matrices."""

    node_rows: dict[int, int]
    root_rows: dict[int, int]
    closures: csr_array
    goals: csr_array
    hyps: csr_array
    hyp_rows: np.ndarray
    hyp_roots: np.ndarray
    hyp_slots: np.ndarray


def incidence(rows: list[int] | np.ndarray, cols: list[int] | np.ndarray, shape: tuple[int, int]) -> csr_array:
    return csr_array((np.ones(len(rows), dtype=np.int64), (rows, cols)), shape=shape)


def prepare_projection(theorem: Candidates) -> Projection:
    node_rows = {node.ref: idx for idx, node in enumerate(theorem.nodes)}
    root_rows = {root.ref: idx for idx, root in enumerate(theorem.roots)}
    lengths = np.fromiter((len(root.anchors) for root in theorem.roots), dtype=np.int64)
    closures = incidence(
        np.repeat(np.arange(len(theorem.roots)), lengths),
        np.fromiter((node_rows[ref] for root in theorem.roots for ref in root.anchors), dtype=np.int64),
        (len(theorem.roots), len(theorem.nodes)),
    )
    counts = np.fromiter((len(state.hyps) for state in theorem.states), dtype=np.int64)
    hyp_rows = np.repeat(np.arange(len(theorem.states)), counts)
    hyp_roots = np.fromiter((root_rows[ref] for state in theorem.states for ref in state.hyps), dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    hyp_slots = np.arange(len(hyp_roots)) - np.repeat(offsets[:-1], counts)
    shape = len(theorem.states), len(theorem.roots)
    goals = incidence(np.arange(len(theorem.states)), [root_rows[state.goal] for state in theorem.states], shape)
    return Projection(
        node_rows, root_rows, closures, goals, incidence(hyp_rows, hyp_roots, shape), hyp_rows, hyp_roots, hyp_slots
    )


def position_attrs(refs: tuple[int, ...] | None) -> tuple[str, ...]:
    """Allocate only attributes compatible with a complete node's operand arity."""
    return (LEAF_ATTRS if refs is None or not refs else ()) + (BINDER_ATTRS if refs is None or len(refs) == 2 else ())


def expr_attrs(expr: r.Expr) -> tuple[tuple[str, int], ...]:
    match expr:
        case r.Bvar(idx=idx):
            return ((LEAF_ATTRS[min(idx, 3)], 1),)
        case r.NatLiteral(val=val):
            return (("nat_0" if val == 0 else "nat_1" if val == 1 else "nat_other", 1),)
        case r.Sort(lvl=r.LvlZero()):
            return (("sort_prop", 1),)
        case r.Lambda(names=names, binder_info=info) | r.Forall(names=names, binder_info=info):
            kind = "lambda" if isinstance(expr, r.Lambda) else "forall"
            return ((f"{kind}_binders", len(names)), (f"{kind}_{info}", 1))
        case _:
            return ()


def head_name(ref: int, exprs: dict[int, r.Expr]) -> str | None:
    expr = exprs[ref]
    if isinstance(expr, r.App):
        expr = exprs[expr.fn]
    return expr.name if isinstance(expr, r.Const) else None


@njit(cache=True)
def _depths(offsets: np.ndarray, refs: np.ndarray) -> np.ndarray:
    depths = np.ones(len(offsets) - 1, np.int64)
    for node in range(len(depths)):
        for idx in range(offsets[node], offsets[node + 1]):
            depths[node] = max(depths[node], depths[refs[idx]] + 1)
    return depths


def root_stats(theorem: Candidates, proj: Projection, kinds: np.ndarray, node_kind_count: int) -> np.ndarray:
    """One native adjacency product computes root-local incoming multiplicity.

    Repeated operands count as repeated edges. External parents do not affect
    shared-node counts; the compact nodes retain validated post-order numbering.
    Leaf depth is one. Binder counts refer to flattened binder groups only.
    """
    operands = [tuple(proj.node_rows[ref] for ref in r.expr_refs(node.expr)) for node in theorem.nodes]
    counts = np.fromiter(map(len, operands), dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    refs = np.fromiter((ref for items in operands for ref in items), dtype=np.int64)
    parents = np.repeat(np.arange(len(counts)), counts)
    if np.any(refs >= parents):
        raise ValueError("candidate nodes are not in expression post-order")
    graph = incidence(parents, refs, (len(counts), len(counts)))
    incoming = proj.closures @ graph
    root_nodes = np.fromiter((proj.node_rows[root.ref] for root in theorem.roots), dtype=np.int64)
    distinct = np.diff(proj.closures.indptr)
    shared = np.bincount(
        np.repeat(np.arange(len(theorem.roots)), np.diff(incoming.indptr))[incoming.data > 1],
        minlength=len(theorem.roots),
    )
    roots = [theorem.nodes[idx].expr for idx in root_nodes]
    scalars = np.column_stack(
        (
            distinct,
            np.asarray(incoming.sum(axis=1)).ravel(),
            _depths(offsets, refs)[root_nodes],
            shared,
            shared / np.maximum(distinct, 1),
            counts[root_nodes],
            [len(expr.names) if isinstance(expr, r.Lambda | r.Forall) else 0 for expr in roots],
            [isinstance(expr, r.Const) and expr.name == "False" for expr in roots],
        )
    )
    node_kinds = incidence(np.arange(len(kinds)), kinds, (len(kinds), node_kind_count))
    root_kinds = np.zeros((len(roots), node_kind_count), dtype=np.int64)
    root_kinds[np.arange(len(roots)), kinds[root_nodes]] = 1
    return np.column_stack((scalars, root_kinds, (proj.closures @ node_kinds).toarray()))


def _summary(proj: Projection, stats: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, roots = proj.hyp_rows[keep], proj.hyp_roots[keep]
    counts = np.bincount(rows, minlength=proj.goals.shape[0])
    totals = incidence(rows, roots, proj.hyps.shape) @ stats
    totals[:, SHARED_FRAC] /= np.maximum(counts, 1)
    maxima = np.zeros_like(totals)
    np.maximum.at(maxima, rows, stats[roots])
    return totals, maxima


def stat_width(node_kinds: int, hyp_slots: int) -> int:
    root_width = ROOT_SCALARS + 2 * node_kinds
    return 3 + 3 * root_width + hyp_slots * (3 + root_width) + 3 + 2 * root_width


def stat_fields(node_names: tuple[str, ...], hyp_slots: int) -> tuple[str, ...]:
    """Column names in state_stats order; operands includes an App's function.

    Shared fractions are root-local indegree>1 / distinct nodes. Context/overflow
    summaries average fractions but sum all counts; maxima remain separate.
    Local slots follow export order, including non-propositional declarations.
    """
    root = (*ROOT_FIELDS, *(f"root_{name}" for name in node_names), *(f"nodes_{name}" for name in node_names))
    flags = SLOT_FLAGS

    def summary(prefix: str) -> tuple[str, ...]:
        return tuple(f"{prefix}_{'mean' if name == 'shared_frac' else 'sum'}_{name}" for name in root)

    fields = ["locals", "lets", "instances", *(f"goal_{name}" for name in root)]
    fields.extend((*summary("ctxt"), *(f"ctxt_max_{name}" for name in root)))
    for slot in range(hyp_slots):
        fields.extend(f"hyp_{slot}_{name}" for name in (*flags, *root))
    fields.extend(
        (*(f"overflow_{name}" for name in flags), *summary("overflow"), *(f"overflow_max_{name}" for name in root))
    )
    return tuple(fields)


def state_stats(theorem: Candidates, proj: Projection, stats: np.ndarray, hyp_slots: int) -> csr_array:
    """Sparse slot assembly avoids a states-by-all-slots dense intermediate.

    Context/overflow summaries sum counts, average shared fractions, and retain
    maxima. Overflow never removes collective patterns or context summaries.
    """
    if any(state.locals is None or len(state.locals) != len(state.hyps) for state in theorem.states):
        raise ValueError("full features require local metadata; regenerate candidates from the database")
    info = [local for state in theorem.states for local in state.locals or ()]
    flags = np.asarray([(1, int(local.is_let), int(local.is_instance)) for local in info], dtype=np.int64).reshape(
        -1, 3
    )
    totals = np.zeros((len(theorem.states), 3), dtype=np.int64)
    np.add.at(totals, proj.hyp_rows, flags)
    all_hyps = np.ones(len(info), dtype=bool)
    ctxt_sum, ctxt_max = _summary(proj, stats, all_hyps)
    blocks = [csr_array(totals), csr_array(proj.goals @ stats), csr_array(ctxt_sum), csr_array(ctxt_max)]
    for slot in range(hyp_slots):
        keep = proj.hyp_slots == slot
        local = csr_array(np.column_stack((flags[keep], stats[proj.hyp_roots[keep]]))).tocoo()
        blocks.append(
            csr_array(
                (local.data, (proj.hyp_rows[keep][local.row], local.col)),
                shape=(len(theorem.states), 3 + stats.shape[1]),
            )
        )
    overflow = proj.hyp_slots >= hyp_slots
    over_flags = np.zeros_like(totals)
    np.add.at(over_flags, proj.hyp_rows[overflow], flags[overflow])
    over_sum, over_max = _summary(proj, stats, overflow)
    blocks.extend((csr_array(over_flags), csr_array(over_sum), csr_array(over_max)))
    return hstack(blocks, format="csr")
