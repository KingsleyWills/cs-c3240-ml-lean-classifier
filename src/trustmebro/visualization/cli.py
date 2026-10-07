"""Analyze the corpus, render saved measurements, or run both phases."""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypedDict, assert_never, cast

from trustmebro.extraction.scheduler import Phase

from .archives import Analysis, AnalysisCfg, AnalysisPaths, analysis_set, read_summary
from .measurements import VIEW_MODES

# Command types and records.


type Cmd = Literal["analyze", "graphs", "pipeline"]
type RenderStage = Literal[
    "metrics", "topology", "comparisons", "heads", "topology-atlas", "local-patterns", "candidates", "features"
]


@dataclass(frozen=True)
class GraphFamily:
    analysis: Analysis | None
    descr: str
    artifacts: Callable[[AnalysisPaths], tuple[Path, ...]] = lambda paths: (paths.analysis("metrics"),)
    stage: RenderStage = "metrics"


class StageReport(TypedDict):
    dur_sec: float
    peak_rss_mib: float


# Supported analyses and parsed defaults.


ANALYSES: tuple[Analysis, ...] = ("metrics", "topology", "patterns")
ANALYSIS_CHOICES: tuple[Analysis, ...] = (*ANALYSES, "vocabulary")


@dataclass(frozen=True)
class Args:
    """Parsed arguments shared by the three commands and their isolated stages."""

    stats: Path = Path("data/analysis")
    output: Path = Path("data/graphs")
    db: Path = Path("data/mathlib.db")
    workers: int = 2
    atlas_min_nodes: int = 20
    dot_diam: int = 1
    limit: int | None = None
    replace: bool = False
    list_graphs: bool = False
    stage: str | None = None
    aggregation_memory_mib: int = 512
    timing_dir: Path | None = None
    skip_embedding: bool = False
    pattern_depths: tuple[int, ...] = (1, 2, 3)
    graphs: tuple[str, ...] = ("all",)
    analyses: tuple[Analysis, ...] = ANALYSES
    inventory: Path | None = None
    node_types: int = 13
    feature_stats: Path | None = None

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> Args:
        vals = vars(namespace)
        choices = {name: tuple(vals[name]) for name in ("pattern_depths", "graphs", "analyses") if name in vals}
        return cls(**(vals | choices))


DEFAULT_ARGS = Args()


# One registry governs CLI choices, dependencies, and render dispatch.
GRAPH_FAMILIES: Mapping[str, GraphFamily] = MappingProxyType(
    {
        "complexity": GraphFamily("metrics", "State size, expansion, and context density"),
        "constructors": GraphFamily("metrics", "Expression constructor composition"),
        "context": GraphFamily("metrics", "Context composition and concentration"),
        "reuse": GraphFamily("metrics", "Global reuse, novelty, and vocabulary coverage"),
        "frequencies": GraphFamily("metrics", "Expression occurrence distributions"),
        "embedding": GraphFamily(
            "metrics",
            "Structural features and UMAP embeddings",
            lambda paths: (paths.analysis("metrics"), paths.embedding_info),
        ),
        "expr-atlas": GraphFamily("metrics", "Expression DAGs, adjacency, and layer profiles"),
        "topology": GraphFamily(
            "topology",
            "Complete DAG topology frequencies and coverage",
            lambda paths: (paths.stats("topology"), *(paths.topo(mode) for mode in VIEW_MODES)),
            stage="topology",
        ),
        "comparisons": GraphFamily(
            "topology",
            "Coercion and instance transformations versus the exported baseline",
            lambda paths: (paths.comparisons, paths.stats("comparison")),
            stage="comparisons",
        ),
        "heads": GraphFamily(
            "topology", "Application-head frequencies", lambda paths: (paths.heads, paths.stats("head")), stage="heads"
        ),
        "topology-atlas": GraphFamily(
            "topology",
            "Common topologies and paired graph examples",
            lambda paths: (paths.topo(), paths.examples),
            stage="topology-atlas",
        ),
        "local-patterns": GraphFamily(
            "patterns",
            "Local patterns and head-conditioned coverage",
            lambda paths: (paths.analysis("patterns"), paths.stats("pattern")),
            stage="local-patterns",
        ),
        "candidates": GraphFamily(
            None,
            "Candidate growth, support, observation coverage and shape atlas",
            lambda paths: (),
            stage="candidates",
        ),
        "features": GraphFamily(
            None,
            "Actual feature coverage, activation and bounded pattern co-occurrence",
            lambda paths: (),
            stage="features",
        ),
    }
)


# Argument parsing and artifact requirements.


def _pos(val: str) -> int:
    num = int(val)
    if num < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return num


def _parser(cmd: Cmd) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description={
            "analyze": "Read the SQLite corpus and write compressed measurements; no plots.",
            "graphs": "Render saved measurements; no corpus analysis.",
            "pipeline": "Analyze the corpus, then render the selected graph families.",
        }[cmd]
    )
    parser.add_argument("--stats", type=Path, default=DEFAULT_ARGS.stats, help="measurements directory")
    parser.add_argument("--stage", help=argparse.SUPPRESS)
    parser.add_argument("--atlas-min-nodes", type=_pos, default=DEFAULT_ARGS.atlas_min_nodes)
    if cmd != "graphs":
        parser.add_argument(
            "--workers", type=_pos, default=DEFAULT_ARGS.workers, help="analysis processes and UMAP threads"
        )
        parser.add_argument("--db", type=Path, default=DEFAULT_ARGS.db)
        parser.add_argument("--limit", type=_pos, help="first N theorems for a smoke test")
        parser.add_argument("--replace", action="store_true", help="replace existing measurements")
        parser.add_argument(
            "--aggregation-memory-mib",
            type=_pos,
            default=DEFAULT_ARGS.aggregation_memory_mib,
            help="pattern counter budget before compressed count spills",
        )
        parser.add_argument("--skip-embedding", action="store_true", help="omit UMAP analysis")
        parser.add_argument("--timing-dir", type=Path, help="write per-process analysis timing CSVs")
        parser.add_argument("--pattern-depths", type=_pos, nargs="+", default=DEFAULT_ARGS.pattern_depths)
    if cmd == "analyze":
        parser.add_argument("--analyses", choices=ANALYSIS_CHOICES, nargs="+", default=DEFAULT_ARGS.analyses)
    else:
        parser.add_argument("--output", type=Path, default=DEFAULT_ARGS.output)
        parser.add_argument("--graphs", choices=("all", *GRAPH_FAMILIES), nargs="+", default=DEFAULT_ARGS.graphs)
        parser.add_argument("--list-graphs", action="store_true")
        parser.add_argument("--dot-diameter", dest="dot_diam", type=_pos, default=DEFAULT_ARGS.dot_diam)
        parser.add_argument(
            "--inventory", type=Path, help="global candidate inventory; enables candidates in --graphs all"
        )
        parser.add_argument(
            "--node-types",
            type=_pos,
            default=DEFAULT_ARGS.node_types,
            help="hypothetical node-kind channels per position in candidate coverage",
        )
        parser.add_argument(
            "--feature-stats", type=Path, help="feature diagnostics archive; enables features in --graphs all"
        )
    return parser


def _args(**cfg: Path | int | bool | list[int] | list[str] | tuple[int, ...] | tuple[str, ...] | None) -> list[str]:
    """Translate named options once, omitting absent values and false flags."""
    result: list[str] = []
    for name, val in cfg.items():
        if val is None or val is False:
            continue
        result.append("--" + name.replace("_", "-"))
        if val is not True:
            result.extend(map(str, val if isinstance(val, (list, tuple)) else (val,)))
    return result


def _require_graphs(
    paths: AnalysisPaths, graphs: Sequence[str], inventory: Path | None = None, feature_stats: Path | None = None
) -> None:
    requirements: dict[Analysis, set[str]] = {}
    for name in graphs:
        family = GRAPH_FAMILIES[name]
        if family.analysis is None:
            src = feature_stats if name == "features" else inventory
            flag = "--feature-stats" if name == "features" else "--inventory"
            if src is None or not src.is_file():
                raise ValueError(f"{name} graphs require {flag} pointing to their saved input")
            continue
        requirements.setdefault(family.analysis, set()).update(path.name for path in family.artifacts(paths))
    paths.require(requirements, requirements)
    if "embedding" in graphs:
        modes = read_summary(paths.embedding_info, tuple[str, ...])
        paths.require(("metrics",), {"metrics": (paths.embedding(mode).name for mode in modes)})


# Isolated stage dispatch and reporting.


def _analyze(stage: str, args: Args) -> dict[str, int]:
    from .scan import analyze_embeddings, collect_analysis

    if stage == "embedding":
        analyze_embeddings(args.stats, args.workers)
        return {}
    return collect_analysis(
        args.db,
        args.stats,
        analyses=tuple(args.analyses),
        limit=args.limit,
        workers=args.workers,
        atlas_min_nodes=args.atlas_min_nodes,
        depths=tuple(dict.fromkeys(args.pattern_depths)),
        aggr_mem=args.aggregation_memory_mib,
        timing_dir=args.timing_dir,
    )


def _render(stage: RenderStage, args: Args) -> None:
    # Heavy visualization imports remain in the isolated renderer process.
    import matplotlib

    matplotlib.use("Agg")
    match stage:
        case "metrics":
            from .state_render import render

            render(
                AnalysisPaths(args.stats).analysis("metrics"),
                args.output,
                atlas_min_nodes=args.atlas_min_nodes,
                dot_diam=args.dot_diam,
                families={name for name in args.graphs if GRAPH_FAMILIES[name].analysis == "metrics"},
            )
        case "topology":
            from . import stral_render

            stral_render.render_shapes(args.stats, args.output, args.dot_diam)
        case "comparisons":
            from . import stral_render

            stral_render.render_pairs(args.stats, args.output, args.dot_diam)
        case "heads":
            from . import stral_render

            stral_render.render_heads(args.stats, args.output)
        case "topology-atlas":
            from .atlas import render_atlas

            render_atlas(args.stats, args.output)
        case "local-patterns":
            from . import stral_render

            stral_render.render_patterns(args.stats, args.output, args.dot_diam)
        case "candidates":
            from .candidate_render import render_inventory

            if args.inventory is None:
                raise ValueError("candidate graphs require --inventory")
            render_inventory(
                args.inventory,
                args.output,
                min_nodes=args.atlas_min_nodes,
                dot_diam=args.dot_diam,
                node_types=args.node_types,
            )
        case "features":
            from .feature_render import render_features

            if args.feature_stats is None:
                raise ValueError("feature graphs require --feature-stats")
            render_features(args.feature_stats, args.output)
        case _:
            assert_never(stage)


def _stage_cmd(cmd: Cmd, stage: str, args: Args) -> list[str]:
    common = _args(
        stats=args.stats, workers=args.workers if cmd == "analyze" else None, atlas_min_nodes=args.atlas_min_nodes
    )
    cfg = (
        _args(
            db=args.db,
            limit=args.limit,
            replace=args.replace,
            pattern_depths=args.pattern_depths,
            analyses=args.analyses,
            aggregation_memory_mib=args.aggregation_memory_mib,
            timing_dir=args.timing_dir,
        )
        if cmd == "analyze"
        else _args(
            output=args.output,
            dot_diameter=args.dot_diam,
            graphs=args.graphs,
            inventory=args.inventory,
            node_types=args.node_types,
            feature_stats=args.feature_stats,
        )
    )
    return [sys.executable, "-m", "trustmebro.visualization.cli", cmd, "--stage", stage, *common, *cfg]


def _stage_report(cmd: list[str]) -> StageReport:
    """Own the stage's process group so cancellation also stops its workers."""
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, start_new_session=True) as process:
        try:
            stdout, _ = process.communicate()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, cmd, stdout)
    return cast(StageReport, json.loads(stdout))


def _run_stages(cmd: Cmd, stages: Sequence[str], args: Args) -> dict[str, StageReport]:
    reports: dict[str, StageReport] = {}
    paths = AnalysisPaths(args.stats)
    if cmd == "graphs":
        _require_graphs(paths, args.graphs, args.inventory, args.feature_stats)
    for stage in stages:
        # Pin an immutable generation before launching each renderer. Writers
        # may publish another manifest without changing this request's inputs.
        stage_args = args
        if cmd == "graphs":
            analysis = "metrics" if stage == "metrics" else GRAPH_FAMILIES[stage].analysis
            if analysis is not None:
                stage_args = replace(args, stats=paths.dir(analysis))
        # Shared analysis owns live progress in its subprocess; two refresh
        # threads on inherited stderr would repeatedly overwrite each other.
        with Phase(f"{cmd.capitalize()}: {stage}", refresh=not (cmd == "analyze" and stage == "shared")) as progress:
            report = _stage_report(_stage_cmd(cmd, stage, stage_args))
            reports[stage] = report
            progress.details = f"{report['dur_sec']:.2f}s; peak process RSS {report['peak_rss_mib']:,.0f} MiB"
    return reports


# Command entry points.


def _main(cmd: Cmd, argv: list[str] | None = None) -> int:
    parser = _parser(cmd)
    args = Args.from_namespace(parser.parse_args(argv))
    if args.list_graphs:
        for name, family in GRAPH_FAMILIES.items():
            print(f"{name:16} {family.descr}")
        return 0
    if args.atlas_min_nodes < 2:
        parser.error("--atlas-min-nodes must be at least 2")
    requested = args.graphs
    graphs = list(GRAPH_FAMILIES) if "all" in requested else list(dict.fromkeys(requested))
    if "all" in requested and args.inventory is None:
        graphs.remove("candidates")
    if "all" in requested and args.feature_stats is None:
        graphs.remove("features")
    if cmd == "pipeline" and args.skip_embedding:
        graphs = [graph for graph in graphs if graph != "embedding"]
        if not graphs:
            parser.error("--skip-embedding leaves no selected graph families")
    analyses = (
        list(dict.fromkeys(args.analyses))
        if cmd == "analyze"
        else list(
            dict.fromkeys(analysis for graph in graphs if (analysis := GRAPH_FAMILIES[graph].analysis) is not None)
        )
    )
    if args.stage:
        choices = (
            ("shared", "embedding")
            if cmd == "analyze"
            else tuple(dict.fromkeys(family.stage for family in GRAPH_FAMILIES.values()))
        )
        if args.stage not in choices or cmd == "pipeline":
            parser.error("invalid internal stage")
        started = time.perf_counter()
        report: dict[str, int] = {}
        if cmd == "analyze":
            report = _analyze(args.stage, args)
        else:
            _render(cast(RenderStage, args.stage), replace(args, graphs=tuple(graphs)))
        print(
            json.dumps(
                {
                    **report,
                    "dur_sec": time.perf_counter() - started,
                    "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                }
            )
        )
        return 0
    if cmd != "graphs" and analyses:
        if not args.db.is_file():
            parser.error(f"source database does not exist: {args.db}")
        paths = AnalysisPaths(args.stats)
        for analysis in analyses:
            archive = AnalysisPaths(args.stats).analysis(analysis)
            if archive.resolve() == args.db.resolve():
                parser.error("the source database cannot also be a measurements archive")
            if (archive.exists() or analysis in paths.manifest) and not args.replace:
                parser.error(f"measurements already exist: {archive}; use --replace or trustmebro-graphs")
        args.stats.mkdir(parents=True, exist_ok=True)
    else:
        try:
            _require_graphs(AnalysisPaths(args.stats), graphs, args.inventory, args.feature_stats)
        except ValueError as error:
            parser.error(str(error))
    started = time.perf_counter()
    reports: dict[str, object] = {}
    selected = replace(args, analyses=tuple(analyses), graphs=tuple(graphs))
    if cmd != "graphs" and analyses:
        stages = ["shared"]
        wants_embedding = cmd == "analyze" or "embedding" in graphs
        if "metrics" in analyses and wants_embedding and not args.skip_embedding:
            stages.append("embedding")
        cfg = AnalysisCfg(args.atlas_min_nodes, tuple(dict.fromkeys(args.pattern_depths)), "embedding" in stages)
        with analysis_set(
            args.stats, selected.analyses, args.db, limit=args.limit, cfg=cfg, replace=args.replace
        ) as pending:
            reports["analysis"] = _run_stages("analyze", stages, replace(selected, stats=pending.root))
    if cmd != "analyze":
        _require_graphs(AnalysisPaths(args.stats), graphs, args.inventory, args.feature_stats)
        args.output.mkdir(parents=True, exist_ok=True)
        stages = list(dict.fromkeys(GRAPH_FAMILIES[g].stage for g in graphs))
        reports["graphs"] = _run_stages("graphs", stages, selected)
    reports["dur_sec"] = time.perf_counter() - started
    print(json.dumps(reports))
    return 0


def analysis_main(argv: list[str] | None = None) -> int:
    return _main("analyze", argv)


def graphs_main(argv: list[str] | None = None) -> int:
    return _main("graphs", argv)


def main(argv: list[str] | None = None) -> int:
    return _main("pipeline", argv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("command", choices=("analyze", "graphs", "pipeline"))
    args, remaining = parser.parse_known_args()
    raise SystemExit(_main(args.command, remaining))
