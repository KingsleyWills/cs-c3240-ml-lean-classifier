"""Fixed-column, label-neutral structural features from observed DAG fragments.

Only exact canonical identities match. Frontier/leaf distinctions, operand
order and sharing remain those of candidate discovery. A column counts a node
kind at one canonical position across anchored occurrences, never query depths.
Goal and summed hypothesis-type blocks are separate. A frozen representation can
add structural statistics, local-declaration slots and selected node attributes.
No local identifiers, let values, proof search or fitted scaling enter the values.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import msgspec
import numpy as np
from scipy.sparse import csr_array, hstack

from trustmebro.extraction import records as r

from .attributes import (
    BINDER_ATTRS,
    LEAF_ATTRS,
    Projection,
    expr_attrs,
    head_name,
    incidence,
    position_attrs,
    prepare_projection,
    root_stats,
    stat_width,
    state_stats,
)
from .candidates import Candidates, Edges, Selection, extract_candidates
from .inventory import Inventory, shape_edges

NODE_KINDS = (
    r.Bvar,
    r.Fvar,
    r.ExprMvar,
    r.Sort,
    r.Const,
    r.App,
    r.Lambda,
    r.Forall,
    r.Let,
    r.NatLiteral,
    r.StringLiteral,
    r.Metadata,
    r.Proj,
)
NODE_NAMES = tuple(kind.__name__ for kind in NODE_KINDS)
KIND_COLS = {kind: col for col, kind in enumerate(NODE_KINDS)}
MAX_COUNT = np.iinfo(np.int64).max
ATTR_NAMES = LEAF_ATTRS + BINDER_ATTRS


class AttributePolicy(msgspec.Struct, frozen=True):
    hyp_slots: int = 32
    name_dims: int = 100_000
    max_names: int = 512
    max_heads: int = 128
    min_support: int = 3
    memory_mib: int = 512  # retained evidence/index estimate, not peak process memory


DEFAULT_ATTRIBUTE_POLICY = AttributePolicy()


class Representation(msgspec.Struct, frozen=True):
    policy: AttributePolicy
    heads: tuple[str, ...]
    names: tuple[tuple[bytes, int, str], ...]  # shape identity, canonical position, Const.name
    selection: Selection
    candidates: str
    size: int
    modified_ns: int
    label_policy_json: bytes


class Entry(msgspec.Struct, frozen=True, array_like=True):
    ident: bytes
    edges: Edges


class CoverPolicy(msgspec.Struct, frozen=True):
    objective: Literal["entries", "dims"] = "dims"
    improvement_steps: int = 0


DEFAULT_COVER_POLICY = CoverPolicy()

SUPERVISED_SCREENING = "role-count-correlation-balanced"


class SupervisedPolicy(msgspec.Struct, frozen=True):
    """Screening/enrichment settings, not classifier hyperparameters."""

    dims: int = 100_000  # additional dimensions; never charged against the cover
    shortlist: int = 256  # global association shortlist
    per_label: int = 32
    common: int = 64  # presence alone can miss informative positional node kinds
    min_support: int = 3  # distinct theorems with eligible labeled transitions
    redundancy: float = 0.5  # soft cosine penalty, not an exclusion threshold
    memory_mib: int = 1024  # retained score/pair buffers, not process-tree RSS


class Supervision(msgspec.Struct, frozen=True):
    policy: SupervisedPolicy
    label_policy_json: bytes  # canonical mapping contents, not just a mutable path
    candidates: str
    size: int
    modified_ns: int
    labels: tuple[str, ...]
    counts: tuple[int, ...]
    labeled_theorems: int
    shortlist_shapes: int
    added_shapes: int
    added_dims: int
    screening: Literal["role-presence-chi2", "role-count-correlation-balanced"] = SUPERVISED_SCREENING
    dimension_cost: Literal["structural", "complete"] = "structural"
    available_association: tuple[float, ...] = ()  # best supported squared count/label correlation, per family
    selected_association: tuple[float, ...] = ()  # best retained correlation, including the coverage backbone
    label_strength: tuple[float, ...] = ()  # accumulated normalized, redundancy-discounted entry utility


class DimensionBudget(msgspec.Struct, frozen=True):
    total: int
    baseline: int  # selected cover, fixed attributes, statistics and declaration slots
    name_share: float  # reserved fraction of the remaining budget; unused shape allocation transfers to names


class Vocabulary(msgspec.Struct, frozen=True):
    depths: tuple[int, ...]
    entries: tuple[Entry, ...]
    node_kinds: tuple[str, ...] = NODE_NAMES
    selection: Selection | None = None
    min_support: int = 1
    min_nodes: int = 2
    max_shapes: int | None = None
    coverage: CoverPolicy | None = None
    supervision: Supervision | None = None
    representation: Representation | None = None
    budget: DimensionBudget | None = None


@dataclass(frozen=True)
class Layout:
    """Explicitly compiled once, then reused across theorem conversions."""

    vocab: Vocabulary
    index: dict[bytes, int]
    offsets: np.ndarray  # entry boundaries within one role block
    entry_sizes: np.ndarray  # prepared once; selection/conversion validate canonical position counts
    block_width: int
    nontrivial: np.ndarray
    attr_cols: np.ndarray
    attr_width: int
    name_cols: csr_array
    name_index: dict[str, int]
    head_index: dict[str, int]
    width: int  # complete vector; block_width remains structural-only


@dataclass(frozen=True)
class FeatureRows:
    name: str
    steps: np.ndarray
    tactics: tuple[r.Tactic, ...]  # side information, not feature inputs
    matrix: csr_array


def select_vocabulary(
    inventory: Inventory, *, max_shapes: int = 10_000, min_support: int = 1, min_nodes: int = 2
) -> Vocabulary:
    """Provisional support-ranked prefix, not a coverage-optimized selection.

    Rank tie-breaks use canonical identity, independent of worker completion.
    The source selection remains attached; whole-corpus selection is exploratory.
    """
    if min(max_shapes, min_support, min_nodes) < 1:
        raise ValueError("vocabulary limits must be positive")
    support, nodes = inventory.col("theorems"), inventory.col("nodes")
    eligible = np.flatnonzero((support >= min_support) & (nodes >= min_nodes))
    ids = np.asarray([inventory.shapes[idx][0] for idx in eligible], dtype="S32")
    selected = eligible[np.lexsort((ids, support[eligible]))[::-1][:max_shapes]]
    if not len(selected):
        raise ValueError("no candidate shapes meet the selection thresholds")
    entries = tuple(Entry(ident, shape_edges(packed)) for ident, packed in (inventory.shapes[idx] for idx in selected))
    return Vocabulary(
        inventory.summary.selection.depths,
        entries,
        selection=inventory.summary.selection,
        min_support=min_support,
        min_nodes=min_nodes,
        max_shapes=max_shapes,
    )


def compile_vocabulary(vocab: Vocabulary) -> Layout:
    if not vocab.entries or not vocab.depths or any(depth < 1 for depth in vocab.depths):
        raise ValueError("vocabulary requires entries and positive discovery depths")
    if vocab.node_kinds != NODE_NAMES:
        raise ValueError("vocabulary node-kind order differs from this converter")
    index: dict[bytes, int] = {}
    boundaries = [0]
    for idx, entry in enumerate(vocab.entries):
        if entry.ident in index:
            raise ValueError("vocabulary contains a duplicate shape identity")
        if not entry.edges or hashlib.sha256(msgspec.msgpack.encode(entry.edges)).digest() != entry.ident:
            raise ValueError("vocabulary shape identity does not match canonical adjacency")
        if any(ref < 0 or ref >= len(entry.edges) for refs in entry.edges if refs is not None for ref in refs):
            raise ValueError("vocabulary contains an invalid fragment-local reference")
        index[entry.ident] = idx
        boundaries.append(boundaries[-1] + len(entry.edges) * len(NODE_NAMES))
    if 2 * boundaries[-1] > np.iinfo(np.intp).max:
        raise OverflowError("feature dimension count exceeds native sparse indexing")
    position_count = boundaries[-1] // len(NODE_NAMES)
    attr_cols = np.empty((0, len(ATTR_NAMES)), dtype=np.int64)
    attrs = 0
    name_rows: list[int] = []
    name_cols: list[int] = []
    name_index: dict[str, int] = {}
    head_index: dict[str, int] = {}
    cfg = vocab.representation
    extra_width = 0
    if cfg is not None:
        attr_cols = np.full((position_count, len(ATTR_NAMES)), -1, dtype=np.int64)
        check_attribute_policy(cfg.policy)
        if len(set(cfg.heads)) != len(cfg.heads) or len(set(cfg.names)) != len(cfg.names):
            raise ValueError("representation contains duplicate name channels")
        head_index = {name: idx for idx, name in enumerate(cfg.heads)}
        name_index = {name: idx for idx, name in enumerate(sorted({name for _, _, name in cfg.names}))}
        for entry, offset in zip(vocab.entries, boundaries[:-1], strict=True):
            for pos, refs in enumerate(entry.edges):
                for attr in position_attrs(refs):
                    attr_cols[offset // len(NODE_NAMES) + pos, ATTR_NAMES.index(attr)] = attrs
                    attrs += 1
        for ident, pos, name in cfg.names:
            entry = index.get(ident)
            if entry is None or not 0 <= pos < len(vocab.entries[entry].edges) or not name:
                raise ValueError("invalid representation name position")
            refs = vocab.entries[entry].edges[pos]
            if refs is not None and refs:
                raise ValueError("constant name channel occupies a non-leaf position")
            name_rows.append(boundaries[entry] // len(NODE_NAMES) + pos)
            name_cols.append(name_index[name])
        named_cost = 2 * len(cfg.names) + (2 + cfg.policy.hyp_slots) * len(cfg.heads)
        if named_cost > cfg.policy.name_dims or any(not name for name in cfg.heads):
            raise ValueError("representation exceeds its name budget or contains empty names")
        extra_width = (
            2 * (attrs + len(cfg.names) + len(cfg.heads))
            + stat_width(len(NODE_NAMES), cfg.policy.hyp_slots)
            + cfg.policy.hyp_slots * len(cfg.heads)
        )
    named_lookup = csr_array(
        (np.arange(1, len(name_rows) + 1, dtype=np.int64), (name_rows, name_cols)),
        shape=(position_count, len(name_index)),
    )
    width = 2 * boundaries[-1] + extra_width
    if vocab.budget is not None:
        budget = vocab.budget
        if budget.baseline < 0 or budget.total < budget.baseline or not 0 <= budget.name_share <= 1:
            raise ValueError("invalid complete dimension budget")
        if width > budget.total:
            raise ValueError(f"complete feature width {width:,} exceeds total budget {budget.total:,}")
    if width > np.iinfo(np.intp).max:
        raise OverflowError("complete feature dimension count exceeds native sparse indexing")
    return Layout(
        vocab,
        index,
        np.asarray(boundaries, dtype=np.int64),
        np.diff(boundaries) // len(NODE_NAMES),
        boundaries[-1],
        np.asarray([len(entry.edges) > 1 for entry in vocab.entries]),
        attr_cols,
        attrs,
        named_lookup,
        name_index,
        head_index,
        width,
    )


def entry_dimensions(edges: Edges, *, attributes: bool = False) -> int:
    """Both roles, including every fixed attribute allocated to these positions."""
    attrs = sum(len(position_attrs(refs)) for refs in edges) if attributes else 0
    return 2 * (len(edges) * len(NODE_NAMES) + attrs)


def fixed_dimensions(entries: tuple[Entry, ...], hyp_slots: int) -> int:
    return sum(entry_dimensions(entry.edges, attributes=True) for entry in entries) + stat_width(
        len(NODE_NAMES), hyp_slots
    )


def check_attribute_policy(cfg: AttributePolicy) -> None:
    if min(cfg.hyp_slots, cfg.name_dims, cfg.max_names, cfg.max_heads) < 0 or min(cfg.min_support, cfg.memory_mib) < 1:
        raise ValueError("attribute budgets/slots must be nonnegative and support/memory positive")


def feature_blocks(layout: Layout) -> dict[str, tuple[int, int]]:
    """Half-open column spans; structural coverage consumes only the first two."""
    sizes = [("goal_patterns", layout.block_width), ("hyp_patterns", layout.block_width)]
    cfg = layout.vocab.representation
    if cfg is not None:
        sizes.extend(
            (
                ("goal_attributes", layout.attr_width),
                ("hyp_attributes", layout.attr_width),
                ("goal_names", len(cfg.names)),
                ("hyp_names", len(cfg.names)),
                ("goal_heads", len(cfg.heads)),
                ("hyp_heads", len(cfg.heads)),
                ("statistics", stat_width(len(NODE_NAMES), cfg.policy.hyp_slots)),
                ("hyp_slot_heads", cfg.policy.hyp_slots * len(cfg.heads)),
            )
        )
    blocks: dict[str, tuple[int, int]] = {}
    offset = 0
    for name, width in sizes:
        blocks[name] = offset, offset + width
        offset += width
    return blocks


@dataclass(frozen=True, slots=True)
class Matches:
    anchors: np.ndarray  # one entry per matched canonical position
    entries: np.ndarray
    positions: np.ndarray
    nodes: np.ndarray  # compact theorem-node rows, not fragment-local IDs


def matched_positions(theorem: Candidates, layout: Layout, proj: Projection) -> Matches:
    entries = np.asarray([layout.index.get(shape.ident, -1) for shape in theorem.shapes], dtype=np.int64)
    for shape, entry in zip(theorem.shapes, entries, strict=True):
        if entry >= 0 and shape.edges != layout.vocab.entries[int(entry)].edges:
            raise ValueError("matching shape identities have different adjacency")
    return match_positions(theorem, proj, layout.vocab.depths, entries, layout.entry_sizes)


def match_positions(
    theorem: Candidates, proj: Projection, depths: tuple[int, ...], entries: np.ndarray, sizes: np.ndarray
) -> Matches:
    """Match stored occurrences once per anchor/entry, shared by selection and conversion.

    entries maps theorem-local shapes to a caller's column order; -1 skips a shape.
    All returned references are compact theorem-node rows, not vocabulary positions.
    """
    anchors: list[int] = []
    matched_entries: list[int] = []
    positions: list[tuple[int, ...]] = []
    matched: set[tuple[int, int]] = set()
    allowed_depths = set(depths)
    for occ in theorem.occurrences:
        if occ.depth not in allowed_depths:
            continue
        entry = int(entries[occ.shape])
        if entry < 0:
            continue
        anchor = proj.node_rows[occ.nodes[0]]
        if (anchor, entry) in matched:
            continue
        if len(occ.nodes) != sizes[entry]:
            raise ValueError("occurrence positions disagree with its shape")
        matched.add((anchor, entry))
        anchors.append(anchor)
        matched_entries.append(entry)
        positions.append(occ.nodes)
    lengths = np.fromiter(map(len, positions), dtype=np.int64)
    ends = np.r_[0, np.cumsum(lengths)]
    refs = np.fromiter((proj.node_rows[ref] for nodes in positions for ref in nodes), dtype=np.intp)
    local_positions = np.arange(len(refs)) - np.repeat(ends[:-1], lengths)
    return Matches(
        np.repeat(np.asarray(anchors, dtype=np.int64), lengths),
        np.repeat(np.asarray(matched_entries, dtype=np.int64), lengths),
        local_positions,
        refs,
    )


def _root_columns(
    proj: Projection, anchors: np.ndarray, cols: np.ndarray, vals: np.ndarray | None = None
) -> tuple[csr_array, np.ndarray]:
    # Products need only this theorem's active columns, not the full vocabulary
    # width. Restore global IDs after the goal/context products are complete.
    global_cols, local_cols = np.unique(cols, return_inverse=True)
    vals = np.ones(len(cols), dtype=np.int64) if vals is None else vals
    anchored = csr_array((vals, (anchors, local_cols)), shape=(proj.closures.shape[1], len(global_cols)))
    return proj.closures @ anchored, global_cols


def position_counts(
    proj: Projection, matches: Matches, offsets: np.ndarray, kinds: np.ndarray
) -> tuple[csr_array, np.ndarray]:
    """Count positional node kinds per root in compact active columns."""
    cols = offsets[matches.entries] + matches.positions * len(NODE_NAMES) + kinds[matches.nodes]
    return _root_columns(proj, matches.anchors, cols)


def _roles(proj: Projection, roots: csr_array, cols: np.ndarray, width: int) -> csr_array:
    local = hstack((proj.goals @ roots, proj.hyps @ roots), format="csr")
    indices = local.indices
    if len(cols):
        indices = cols[indices % len(cols)] + (indices // len(cols)) * width
    return csr_array((local.data, indices, local.indptr), shape=(proj.goals.shape[0], 2 * width))


def _node_attributes(theorem: Candidates, proj: Projection, matches: Matches, layout: Layout) -> csr_array:
    values = [
        (idx, ATTR_NAMES.index(attr), val)
        for idx, node in enumerate(theorem.nodes)
        for attr, val in expr_attrs(node.expr)
    ]
    attrs = csr_array(
        ([val for _, _, val in values], ([idx for idx, _, _ in values], [col for _, col, _ in values])),
        shape=(len(theorem.nodes), len(ATTR_NAMES)),
        dtype=np.int64,
    )[matches.nodes].tocoo()
    positions = layout.offsets[matches.entries] // len(NODE_NAMES) + matches.positions
    cols = layout.attr_cols[positions[attrs.row], attrs.col]
    if np.any(cols < 0):
        raise ValueError("node attributes conflict with matched shape arity")
    roots, active_cols = _root_columns(proj, matches.anchors[attrs.row], cols, attrs.data)
    return _roles(proj, roots, active_cols, layout.attr_width)


def _named_positions(theorem: Candidates, proj: Projection, matches: Matches, layout: Layout) -> csr_array:
    cfg = layout.vocab.representation
    if cfg is None:
        raise ValueError("named positions require a frozen representation")
    node_names = np.fromiter(
        (layout.name_index.get(node.expr.name, -1) if isinstance(node.expr, r.Const) else -1 for node in theorem.nodes),
        dtype=np.int64,
    )
    rows = np.flatnonzero(node_names[matches.nodes] >= 0)
    positions = layout.offsets[matches.entries[rows]] // len(NODE_NAMES) + matches.positions[rows]
    cols = (
        np.asarray(layout.name_cols[positions, node_names[matches.nodes[rows]]]).ravel() - 1
        if len(rows)
        else np.empty(0, dtype=np.int64)
    )
    keep = cols >= 0
    roots, active_cols = _root_columns(proj, matches.anchors[rows[keep]], cols[keep])
    return _roles(proj, roots, active_cols, len(cfg.names))


def root_heads(theorem: Candidates, proj: Projection, names: dict[str, int]) -> csr_array:
    exprs = {node.ref: node.expr for node in theorem.nodes}
    entries = [
        (idx, names[name])
        for idx, root in enumerate(theorem.roots)
        if (name := head_name(root.ref, exprs)) is not None and name in names
    ]
    return incidence([idx for idx, _ in entries], [col for _, col in entries], (len(theorem.roots), len(names)))


def _extra_features(
    theorem: Candidates, proj: Projection, matches: Matches, layout: Layout, kinds: np.ndarray
) -> csr_array:
    cfg = layout.vocab.representation
    if cfg is None:
        raise ValueError("extra features require a frozen representation")
    heads = root_heads(theorem, proj, layout.head_index)
    blocks = [
        _node_attributes(theorem, proj, matches, layout),
        _named_positions(theorem, proj, matches, layout),
        _roles(proj, heads, np.arange(len(layout.head_index)), len(layout.head_index)),
        state_stats(theorem, proj, root_stats(theorem, proj, kinds, len(NODE_NAMES)), cfg.policy.hyp_slots),
    ]
    for slot in range(cfg.policy.hyp_slots):
        keep = proj.hyp_slots == slot
        local = heads[proj.hyp_roots[keep]].tocoo()
        blocks.append(
            csr_array(
                (local.data, (proj.hyp_rows[keep][local.row], local.col)),
                shape=(len(theorem.states), len(layout.head_index)),
            )
        )
    return hstack(blocks, format="csr")


def encode_candidates(theorem: Candidates, layout: Layout) -> FeatureRows:
    """Prepare one projection and matched population for every requested block.

    Structural-only counts stay int64. Full vectors include sharing fractions
    and use float64; integer counts are checked before conversion, never silently
    rounded. Arbitrary-size source literals only enter categorical channels.
    """
    bound = (
        len(theorem.nodes)
        * len(layout.vocab.depths)
        * max((1 + len(state.hyps) for state in theorem.states), default=1)
    )
    if bound > MAX_COUNT:
        raise OverflowError("feature occurrence counts exceed exact int64 sparse arithmetic")
    if layout.vocab.representation is not None and bound > 2**53:
        raise OverflowError("full-vector count bound exceeds exact float64 integers")
    if layout.vocab.representation is not None:
        factor = max(
            (
                max(
                    len(r.expr_refs(node.expr)),
                    len(node.expr.names) if isinstance(node.expr, r.Lambda | r.Forall) else 1,
                )
                for node in theorem.nodes
            ),
            default=1,
        )
        if bound * max(factor, 1) > 2**53:
            raise OverflowError("full-vector attribute/statistic counts exceed exact float64 integers")
    proj = prepare_projection(theorem)
    matches = matched_positions(theorem, layout, proj)
    kinds = np.fromiter((KIND_COLS[type(node.expr)] for node in theorem.nodes), dtype=np.int64)
    roots, active_cols = position_counts(proj, matches, layout.offsets, kinds)
    matrix = _roles(proj, roots, active_cols, layout.block_width)
    if layout.vocab.representation is not None:
        matrix = hstack((matrix, _extra_features(theorem, proj, matches, layout, kinds)), format="csr")
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return FeatureRows(
        theorem.name,
        np.fromiter((state.step for state in theorem.states), dtype=np.int64),
        tuple(state.tactic for state in theorem.states),
        matrix,
    )


def encode_theorem(theorem: r.Theorem, layout: Layout, *, validated: bool = False) -> FeatureRows:
    """Prepare and encode an export entry; validate unless its caller already did."""
    return encode_candidates(extract_candidates(theorem, layout.vocab.depths, validated=validated), layout)


def pattern_presence(matrix: csr_array, layout: Layout, *, separate_roles: bool = True) -> csr_array:
    """Derive binary pattern presence from feature columns without graph searches."""
    matrix = matrix[:, : 2 * layout.block_width]
    entries = np.searchsorted(layout.offsets, matrix.indices % layout.block_width, side="right") - 1
    if separate_roles:
        entries += (matrix.indices // layout.block_width) * len(layout.vocab.entries)
    width = len(layout.vocab.entries) * (2 if separate_roles else 1)
    presence = csr_array(
        (np.ones(matrix.nnz, dtype=np.int64), entries, matrix.indptr.copy()), shape=(matrix.shape[0], width)
    )
    presence.sum_duplicates()
    presence.data.fill(1)
    return presence


@dataclass
class FeatureStats:
    """Own coordinated corpus totals and a bounded co-occurrence diagnostic."""

    layout: Layout
    pair_limit: int
    theorems: int
    covered_theorems: int
    states: int
    covered_states: int
    goal_covered: int
    hyp_covered: int
    active_dims: dict[int, int]
    active_patterns: dict[int, int]
    support: np.ndarray
    pairs: np.ndarray


def prepare_stats(layout: Layout, pair_limit: int = 128) -> FeatureStats:
    if not 0 <= pair_limit <= 2048:
        raise ValueError("co-occurrence prefix must be between 0 and 2048 shapes")
    count = min(pair_limit, len(layout.vocab.entries))
    return FeatureStats(
        layout,
        count,
        0,
        0,
        0,
        0,
        0,
        0,
        {},
        {},
        np.zeros(2 * len(layout.vocab.entries), dtype=np.int64),
        np.zeros((count, count), dtype=np.int64),
    )


def _add_hgram(dst: dict[int, int], vals: np.ndarray) -> None:
    keys, counts = np.unique(vals, return_counts=True)
    for key, count in zip(keys, counts, strict=True):
        dst[int(key)] = dst.get(int(key), 0) + int(count)


def add_stats(stats: FeatureStats, rows: FeatureRows) -> None:
    layout, matrix = stats.layout, rows.matrix
    count = matrix.shape[0]
    if stats.states + count > MAX_COUNT:
        raise OverflowError("diagnostic counts exceed exact native arithmetic")
    presence = pattern_presence(matrix, layout)
    vocab_size = len(layout.vocab.entries)
    state_rows = np.repeat(np.arange(count), np.diff(presence.indptr))
    nontrivial = layout.nontrivial[presence.indices % vocab_size]
    goal = np.bincount(state_rows[nontrivial & (presence.indices < vocab_size)], minlength=count) > 0
    hyp = np.bincount(state_rows[nontrivial & (presence.indices >= vocab_size)], minlength=count) > 0
    stats.theorems += 1
    stats.covered_theorems += int(np.any(goal | hyp))
    stats.states += count
    stats.covered_states += int(np.count_nonzero(goal | hyp))
    stats.goal_covered += int(np.count_nonzero(goal))
    stats.hyp_covered += int(np.count_nonzero(hyp))
    _add_hgram(stats.active_dims, np.diff(matrix.indptr))
    _add_hgram(stats.active_patterns, np.diff(presence.indptr))
    cols, support = np.unique(presence.indices, return_counts=True)
    stats.support[cols] += support
    if stats.pair_limit:
        # Union roles, then binarize again: goal+context counts as one state.
        selected = presence[:, : stats.pair_limit] + presence[:, vocab_size : vocab_size + stats.pair_limit]
        selected.data.fill(1)
        stats.pairs += (selected.T @ selected).toarray()
