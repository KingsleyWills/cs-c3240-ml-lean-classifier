"""Diagnostics of an observed inventory; no matching or vocabulary selection.

Growth/support use all shapes. Coverage is candidate-occurrence mass, NOT the
union of covered source nodes. Atlas bounds limit illustrations, not analysis.
The hypothetical dimension budget has separate goal/context blocks and one
channel per chosen node kind at every canonical position, including frontiers.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from graph_tool import Graph, PropertyMap
from graph_tool.draw import graph_draw

from trustmebro.preprocessing.archives import read_inventory
from trustmebro.preprocessing.inventory import Inventory, shape_edges

from .drawing import Axis, DensityPlot, count_ticks, density, log_counts, plot_style, save_fig, save_single

ATLAS_MAX_NODES = 128
ATLAS_MAX_EDGES = 512
ATLAS_PANELS = 18


def _occurrences(inventory: Inventory) -> np.ndarray:
    goals, hyps = inventory.col("goals"), inventory.col("hyps")
    if goals.dtype == object or hyps.dtype == object or np.any(hyps > np.iinfo(np.uint64).max - goals):
        return goals.astype(object) + hyps.astype(object)
    return goals + hyps


@plot_style
def _growth(inventory: Inventory, output: Path) -> None:
    max_support = inventory.summary.theorems
    thresholds = np.unique(np.r_[1, np.ceil(np.geomspace(1, max(1, max_support), 80)).astype(np.int64)])
    fig, ax = plt.subplots(figsize=(12, 8))
    for depth in inventory.summary.selection.depths:
        support = np.sort(inventory.col("theorems", depth))
        counts = len(support) - np.searchsorted(support, thresholds)
        ax.plot(thresholds, counts, label=f"Depth {depth}")
    support = np.sort(inventory.col("theorems"))
    ax.plot(thresholds, len(support) - np.searchsorted(support, thresholds), "w--", label="All depths (deduplicated)")
    ax.set(
        xscale="log",
        xlabel="Minimum number of supporting theorems",
        ylabel="Distinct canonical shapes",
        title="Observed vocabulary growth across depth and support",
    )
    count_ticks(ax.xaxis)
    ax.legend()
    fig.text(
        0.1,
        0.015,
        "Identical shapes across depths count once in the union; distinct nested shapes are retained.",
        fontsize=9,
    )
    save_single(fig, output / "candidate-growth.png")


def _support(inventory: Inventory, output: Path, dot_diam: int) -> None:
    nodes = log_counts(inventory.col("nodes"))
    caption = "One point per globally distinct ordered shape; node kinds/values omitted; all anchors and depths."
    for name, label, vals in (
        ("occurrences", "Goal + hypothesis root-occurrence weight", _occurrences(inventory)),
        ("support", "Number of supporting theorems", inventory.col("theorems")),
    ):
        plot = DensityPlot(
            f"candidate-size-{name}.png",
            "Fragment size and " + label.lower(),
            Axis("nodes", "Canonical positions (including frontier nodes)", "log10"),
            Axis("count", label, "log10"),
            caption,
        )
        frame = pd.DataFrame({"nodes": nodes, "count": log_counts(vals)})
        density(output, frame, plot, dot_diam=dot_diam)


def _exact_cumsum(vals: np.ndarray) -> np.ndarray:
    if vals.dtype != object and int(vals.max(initial=0)) * len(vals) > np.iinfo(np.uint64).max:
        vals = vals.astype(object)
    return np.cumsum(vals)


@plot_style
def _coverage(inventory: Inventory, output: Path, node_types: int) -> None:
    freq = _occurrences(inventory)
    ids = np.asarray([ident for ident, _ in inventory.shapes], dtype="S32")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for name, rank_col in (("Occurrence-ranked", freq), ("Theorem-support-ranked", inventory.col("theorems"))):
        # Identity tie-breaks keep illustrative prefix curves independent of
        # unordered worker completion. This is not a chosen vocabulary policy.
        rank = np.lexsort((ids, rank_col))[::-1]
        mass = _exact_cumsum(freq[rank])
        positions = _exact_cumsum(inventory.col("nodes")[rank])
        if positions.dtype != object and int(positions.max(initial=0)) * (2 * node_types) > np.iinfo(np.uint64).max:
            positions = positions.astype(object)
        dims = positions * (2 * node_types)
        if not len(rank) or int(mass[-1]) == 0:
            continue
        # Evaluate exact prefix totals on a logarithmic display grid, retaining
        # the final prefix. No candidate observations are sampled or discarded.
        points = np.unique(np.ceil(np.geomspace(1, len(rank), min(1200, len(rank)))).astype(np.intp)) - 1
        coverage = np.fromiter((int(val) / int(mass[-1]) for val in mass[points]), dtype=float)
        axes[0].plot(points + 1, coverage, label=name)
        axes[1].plot(dims[points], coverage, label=name)
    for ax in axes:
        ax.set(xscale="log", ylim=(0, 1.01))
        count_ticks(ax.xaxis)
        ax.legend()
    axes[0].set(xlabel="Number of distinct shapes", ylabel="Fraction of candidate-occurrence mass")
    axes[1].set_xlabel("Hypothetical feature dimensions")
    fig.suptitle("Candidate-observation coverage, not source-node coverage or predictive usefulness")
    fig.text(
        0.08,
        0.015,
        f"Dimension budget: 2 roles × {node_types} node-kind channels × canonical positions. "
        "No names or joint predicates.\nRepeated depth observations are deduplicated only at the same anchor and shape.",
        fontsize=9,
    )
    save_fig(fig, output / "candidate-coverage.png", dpi=110, tight=True)


def _dag_positions(graph: Graph, links: list[tuple[int, int, int]]) -> PropertyMap:
    """Use the installed Graphviz dot engine, not a custom layout algorithm.

    graph_tool 3.9 no longer exposes graphviz_draw. Only bounded atlas shapes
    cross this CLI boundary; ordering=out preserves the declared operand order.
    """
    nodes = (f"v{idx};" for idx in range(graph.num_vertices()))
    edges = (f"v{src} -> v{dst};" for src, dst, _ in links)
    data = "digraph { graph [ordering=out]; " + " ".join((*nodes, *edges)) + " }"
    result = subprocess.run(["dot", "-Tjson"], input=data, capture_output=True, text=True, check=True)
    positions = graph.new_vp("vector<double>")
    for node in json.loads(result.stdout)["objects"]:
        positions[int(node["name"][1:])] = tuple(map(float, node["pos"].split(",")))
    return positions


@plot_style
def _atlas(inventory: Inventory, output: Path, min_nodes: int) -> None:
    support, freq, nodes = inventory.col("theorems"), _occurrences(inventory), inventory.col("nodes")
    eligible = np.flatnonzero(
        (nodes >= min_nodes) & (nodes <= ATLAS_MAX_NODES) & (inventory.col("edges") <= ATLAS_MAX_EDGES)
    )
    ids = np.asarray([inventory.shapes[idx][0] for idx in eligible], dtype="S32")
    selected = eligible[np.lexsort((ids, freq[eligible], support[eligible]))[::-1][:ATLAS_PANELS]]
    if not len(selected):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"No observed shapes with {min_nodes}–{ATLAS_MAX_NODES} positions and ≤{ATLAS_MAX_EDGES} operand edges",
            ha="center",
        )
    else:
        rows = (len(selected) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, idx in zip(axes.flat, selected):
            edges = shape_edges(inventory.shapes[idx][1])
            graph = Graph(directed=True)
            graph.add_vertex(len(edges))
            links = [
                (src, dst, slot) for src, refs in enumerate(edges) if refs is not None for slot, dst in enumerate(refs)
            ]
            slots = graph.new_ep("int")
            if links:
                graph.add_edge_list(np.asarray(links, dtype=np.int64), eprops=[slots])
            labels = graph.new_vp("string", vals=[str(pos) for pos in range(len(edges))])
            colours = graph.new_vp("string", vals=["#ffaf45" if refs is None else "#57bde9" for refs in edges])
            positions = _dag_positions(graph, links)
            # Labels expose canonical positions; slot labels preserve ordered
            # operands and parallel edges, rather than merely a spanning tree.
            graph_draw(
                graph,
                pos=positions,
                mplfig=ax,
                vertex_text=labels,
                vertex_fill_color=colours,
                vertex_size=18,
                vertex_font_size=10,
                edge_text=slots,
                edge_font_size=8,
                edge_color="#a8bdc9",
                vertex_pen_width=0,
            )
            ax.set_title(
                f"{int(nodes[idx])} positions; {int(support[idx]):,} theorems\n{int(freq[idx]):,} weighted occurrences",
                fontsize=10,
            )
    fig.suptitle("Common nontrivial ordered shapes (node kinds and values omitted)")
    fig.text(
        0.08,
        0.01,
        f"Up to {ATLAS_PANELS} shapes with {min_nodes}–{ATLAS_MAX_NODES} positions and ≤{ATLAS_MAX_EDGES} edges, "
        "ranked by theorem support. "
        "Orange: wildcard frontier; blue terminal: true leaf.\nNode numbers are fragment-local IDs; "
        "edge numbers are operand slots; repeated IDs/edges retain sharing.",
        fontsize=10,
    )
    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.35)
    save_fig(fig, output / "candidate-atlas.png", tight=True)


def render_inventory(src: Path, output: Path, *, min_nodes: int = 20, dot_diam: int = 1, node_types: int = 13) -> None:
    """Read the compact inventory once; no source database or candidate scan."""
    if node_types < 1:
        raise ValueError("node-kind channel count must be positive")
    inventory = read_inventory(src)
    if not inventory.shapes:
        raise ValueError("candidate inventory contains no shapes to plot")
    output.mkdir(parents=True, exist_ok=True)
    _growth(inventory, output)
    _support(inventory, output, dot_diam)
    _coverage(inventory, output, node_types)
    _atlas(inventory, output, min_nodes)
