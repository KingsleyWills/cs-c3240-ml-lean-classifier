"""Draw actual feature activation/coverage; never rescan graphs or feature rows."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from trustmebro.preprocessing.archives import ARRAY_DTYPE, read_diagnostics

from .drawing import count_ticks, plot_style, save_single


@plot_style
def render_features(src: Path, output: Path) -> None:
    report = read_diagnostics(src)
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    counts = (report.covered_theorems, report.covered_states, report.goal_covered, report.hyp_covered)
    totals = (report.theorems, report.states, report.states, report.states)
    rates = [count / total if total else 0 for count, total in zip(counts, totals, strict=True)]
    labels = ("Theorems\n(any state)", "States\n(either role)", "States\ngoal", "States\nhypotheses")
    bars = ax.bar(labels, rates, color="#57bde9")
    ax.bar_label(bars, labels=[f"{count:,}/{total:,}" for count, total in zip(counts, totals, strict=True)])
    ax.set(
        ylim=(0, 1.12),
        ylabel="Fraction with at least one nontrivial selected pattern",
        title="Actual representation coverage",
    )
    fig.text(
        0.1,
        0.015,
        "Nontrivial means more than one canonical position; this does not measure predictive usefulness.",
        fontsize=9,
    )
    save_single(fig, output / "feature-coverage.png")

    fig, ax = plt.subplots(figsize=(12, 8))
    zeros: list[str] = []
    for name, hgram in (
        ("Active feature dimensions", report.active_dims),
        ("Active role/pattern pairs", report.active_patterns),
    ):
        vals = np.asarray([val for val, _ in hgram], dtype=np.int64)
        counts_array = np.asarray([count for _, count in hgram], dtype=np.int64)
        cumulative = np.cumsum(counts_array) / max(1, report.states)
        positive = vals > 0
        ax.step(vals[positive], cumulative[positive], where="post", label=name)
        zeros.append(f"{name}: {dict(hgram).get(0, 0):,} zero states")
    ax.set(
        xscale="log",
        ylim=(0, 1.01),
        xlabel="Nonzero columns per state (actual counts)",
        ylabel="Fraction of all states at or below this count",
        title="Actual sparse activation distributions",
    )
    count_ticks(ax.xaxis)
    ax.legend()
    fig.text(
        0.1, 0.015, "; ".join(zeros) + "\nZero states remain in the CDF denominator; no state sampling.", fontsize=9
    )
    save_single(fig, output / "feature-activation.png")

    fig, ax = plt.subplots(figsize=(12, 8))
    if report.pair_limit:
        pairs = np.frombuffer(report.pairs, dtype=ARRAY_DTYPE).reshape(report.pair_limit, report.pair_limit)
        support = np.diag(pairs).astype(float)
        union = support[:, None] + support[None, :] - pairs
        jaccard = np.divide(pairs, union, out=np.zeros_like(pairs, dtype=float), where=union > 0)
        img = ax.imshow(jaccard, vmin=0, vmax=1, cmap="turbo", interpolation="nearest", origin="lower")
        fig.colorbar(img, ax=ax, label="Fraction of union states containing both patterns (Jaccard)")
        ax.set(xlabel="Vocabulary entry index", ylabel="Vocabulary entry index")
    else:
        ax.text(0.5, 0.5, "Co-occurrence diagnostic disabled", ha="center", transform=ax.transAxes)
    ax.set_title("State-level pattern co-occurrence (goal/hypothesis roles unioned)")
    fig.text(
        0.1,
        0.015,
        f"First {report.pair_limit:,} entries only; all {report.states:,} converted states. "
        "All other entries remain in the features.\nRaw pair counts are retained; colour normalization happens here.",
        fontsize=9,
    )
    save_single(fig, output / "feature-cooccurrence.png")
