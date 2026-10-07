"""Create a normalized tactic-transition corpus with LeanDojo v2.

The extractor traces one explicitly pinned Lean repository and emits one JSON
object per tactic retained by LeanDojo's ``atomic_only`` syntax filter.
Generated artifacts are written under ``data/`` by default and are not part of
the source repository.

LeanDojo v2 is intentionally imported only when tracing starts. This keeps
the project's ordinary scikit-learn environment independent of LeanDojo's
large training stack.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_REPOSITORY = "https://github.com/leanprover-community/mathlib4"
COMPILED_EXTRACTOR = (
    PROJECT_ROOT / ".lake" / "build" / "bin" / "trustmebro-extract-data"
)
DEFAULT_CONVERSION_JOBS = os.cpu_count() or 1

_STREAM_CONTEXT: tuple[Path, Any, Any] | None = None

# This pattern only extracts the leading tactic name from a source span already
# selected by LeanDojo. Its ``atomic_only`` filter is syntactic, not a guarantee
# that the span represents exactly one semantic proof action.
TACTIC_HEAD = re.compile(r"^([A-Za-z_][A-Za-z0-9_'.]*(?:[!?])?)")
COMMIT_HASH = re.compile(r"[0-9a-fA-F]{40}")

_FAMILY_HEADS = {
    "rule_application": [
        "apply",
        "apply_assumption",
        "assumption",
        "assumption'",
        "constructor",
        "constructorm",
        "exact",
        "exacts",
        "fapply",
        "fconstructor",
        "left",
        "right",
        "valid",
    ],
    "introduce_local": [
        "generalize",
        "have",
        "have'",
        "haveI",
        "let",
        "letI",
        "suffices",
    ],
    "refine": ["refine"],
    "binder_management": ["intro", "intros", "introv", "revert"],
    "destruct": ["injection", "injections", "rcases"],
    "case_split": [
        "by_cases",
        "by_cases!",
        "cases",
        "cases_type",
        "casesm",
        "fin_cases",
        "fun_cases",
        "interval_cases",
        "split",
        "split_ifs",
    ],
    "use": ["exists", "existsi", "use", "use!"],
    "induction": ["fun_induction", "hopf_tensor_induction", "induction"],
    "lift": ["lift"],
    "choose": ["choose", "choose!"],
    "rewrite": ["erw", "nth_rewrite", "nth_rw", "rewrite", "rw", "rw!"],
    "simplify": [
        "simp",
        "simp!",
        "simp_all",
        "simp_all!",
        "simp_rw",
        "simp_wf",
        "simpa",
        "simpa!",
    ],
    "definitional_transform": [
        "beta_reduce",
        "cbv",
        "change",
        "delta",
        "dsimp",
        "dsimp!",
        "eta_expand",
        "show",
        "unfold",
        "unfold_projs",
    ],
    "subst": ["subst", "subst_vars"],
    "convert": ["convert", "convert!", "convert_to", "convert_to!"],
    "conv": ["conv", "conv_lhs", "conv_rhs", "slice_lhs", "slice_rhs"],
    "extensionality": ["ext", "ext1", "funext"],
    "congr": ["congr", "congr!", "congrm", "rcongr"],
    "cast_normalization": [
        "apply_mod_cast",
        "assumption_mod_cast",
        "exact_mod_cast",
        "norm_cast",
        "norm_cast0",
        "push_cast",
        "qify",
        "rify",
        "rw_mod_cast",
        "zify",
    ],
    "relation_restructuring": ["symm", "trans", "transitivity"],
    "generalized_congruence": [
        "apply_rw",
        "gcongr",
        "gconvert",
        "grw",
        "nth_grw",
        "rel",
    ],
    "contradiction_reasoning": [
        "absurd",
        "by_contra",
        "by_contra!",
        "contrapose",
        "contrapose!",
        "exfalso",
    ],
    "computational_closure": ["decide", "infer_instance", "rfl"],
    "proof_search": [
        "aesop",
        "aesop_cat",
        "aesop_mat",
        "apply_rules",
        "cat_disch",
        "contradiction",
        "grind",
        "solve_by_elim",
        "tauto",
        "tauto_set",
        "trivial",
    ],
    "arithmetic_reasoning": [
        "arith_mult",
        "bound",
        "fin_omega",
        "lia",
        "linarith",
        "linarith!",
        "linear_combination",
        "nlinarith",
        "norm_num",
        "norm_num1",
        "omega",
        "order",
    ],
    "algebraic_normalization": [
        "abel",
        "abel1",
        "abel_nf",
        "ac_nf",
        "ac_rfl",
        "algebraize",
        "algebraize_only",
        "cancel_denoms",
        "compute_degree",
        "compute_degree!",
        "field",
        "field_simp",
        "grobner",
        "group",
        "init_ring",
        "match_scalars",
        "module",
        "module_nf",
        "monicity",
        "monicity!",
        "move_mul",
        "noncomm_ring",
        "ring",
        "ring!",
        "ring1",
        "ring_nf",
    ],
    "property_automation": [
        "continuity",
        "finiteness",
        "fun_prop",
        "measurability",
        "nontriviality",
        "positivity",
        "sz_positivity",
    ],
}
_HEAD_FAMILY = {
    head: family for family, heads in _FAMILY_HEADS.items() for head in heads
}
_RINTRO_DESTRUCTURING = re.compile(r"[⟨⟩]|\||\brfl\b", re.UNICODE)


def _top_level_list_items(tactic: str) -> int:
    """Count entries in the first bracketed filter_upwards argument list."""

    start = tactic.find("[")
    if start < 0:
        return 0
    depth = 0
    items = 1
    for character in tactic[start + 1 :]:
        if character == "[":
            depth += 1
        elif character == "]":
            if depth == 0:
                return items
            depth -= 1
        elif character == "," and depth == 0:
            items += 1
    return 0


def estimated_families(tactic: str, head: str | None) -> tuple[str, ...]:
    """Estimate post-decomposition families for one source record."""

    if head == "rwa":
        return "rewrite", "rule_application"
    if head == "rintro":
        if _RINTRO_DESTRUCTURING.search(tactic):
            return "binder_management", "destruct"
        return ("binder_management",)
    if head == "obtain":
        return "introduce_local", "destruct"
    if head in {"replace", "specialize"}:
        return ("introduce_local",)
    if head == "set":
        return "introduce_local", "rewrite"
    if head == "set!":
        return ("introduce_local",)
    if head == "rsuffices":
        return "introduce_local", "destruct"
    if head == "filter_upwards":
        result = ["definitional_transform", "rule_application"]
        result.extend("rule_application" for _ in range(_top_level_list_items(tactic)))
        if re.search(r"\bwith\b", tactic):
            result.append("binder_management")
        if re.search(r"\busing\b", tactic):
            result.append("rule_application")
        return tuple(result)
    return (_HEAD_FAMILY.get(head, "OTHER"),)


def tactic_head(tactic: str) -> str | None:
    """Return the leading identifier of a LeanDojo-selected tactic span."""

    match = TACTIC_HEAD.match(tactic.lstrip())
    return match.group(1) if match else None


@dataclass
class CorpusStats:
    """Streaming counts for one preliminary transition corpus."""

    transitions: int = 0
    tactic_heads: Counter[str] = field(default_factory=Counter)
    estimated_family_counts: Counter[str] = field(default_factory=Counter)
    theorem_lengths: Counter[tuple[str, str]] = field(default_factory=Counter)
    files: set[str] = field(default_factory=set)

    def add(self, record: Mapping[str, Any]) -> None:
        file_path = record["file_path"]
        theorem = record["theorem"]
        head = record["tactic_head"]
        self.transitions += 1
        self.theorem_lengths[(file_path, theorem)] += 1
        self.files.add(file_path)
        if head is not None:
            self.tactic_heads[head] += 1
        self.estimated_family_counts.update(estimated_families(record["tactic"], head))

    def update(self, other: CorpusStats) -> None:
        self.transitions += other.transitions
        self.tactic_heads.update(other.tactic_heads)
        self.estimated_family_counts.update(other.estimated_family_counts)
        self.theorem_lengths.update(other.theorem_lengths)
        self.files.update(other.files)


def _percentile(sorted_values: Sequence[int], fraction: float) -> int | float:
    """Return a linearly interpolated percentile without a NumPy dependency."""

    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    value = sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (
        position - lower
    )
    return int(value) if value.is_integer() else round(value, 4)


def _proof_length_summary(stats: CorpusStats) -> dict[str, Any]:
    """Summarize retained records per theorem represented in the corpus."""

    lengths = sorted(stats.theorem_lengths.values())
    if not lengths:
        return {"definition": "retained transition records per represented theorem"}
    return {
        "definition": "retained transition records per represented theorem",
        "minimum": lengths[0],
        "first_quartile": _percentile(lengths, 0.25),
        "median": _percentile(lengths, 0.5),
        "mean": round(sum(lengths) / len(lengths), 4),
        "third_quartile": _percentile(lengths, 0.75),
        "p90": _percentile(lengths, 0.9),
        "p95": _percentile(lengths, 0.95),
        "p99": _percentile(lengths, 0.99),
        "maximum": lengths[-1],
        "single_record_theorems": sum(length == 1 for length in lengths),
    }


def _configure_runtime(data_dir: Path) -> None:
    """Keep LeanDojo and its transitive libraries inside ignored data storage."""

    # LeanDojo probes its public precomputed-trace cache with an HTTP request
    # that has no timeout. Our recent pinned source commit (and local
    # compatibility fixture) will not be in that benchmark-oriented cache, so
    # trace locally instead of risking an indefinite network wait.
    os.environ.setdefault("DISABLE_REMOTE_CACHE", "1")

    runtime = data_dir / "runtime"
    paths = {
        "CACHE_DIR": data_dir / "lean-dojo-cache",
        "TMP_DIR": runtime / "tmp",
        "TRITON_CACHE_DIR": runtime / "triton",
        "MPLCONFIGDIR": runtime / "matplotlib",
        "HF_HOME": runtime / "huggingface",
        "TORCH_HOME": runtime / "torch",
    }
    for name, path in paths.items():
        os.environ.setdefault(name, str(path.resolve()))
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)


def _repository_key(url: str, commit: str) -> tuple[str, str]:
    normalized_url = url.rstrip("/").removesuffix(".git")
    return normalized_url, commit.lower()


def _local_lake_checkouts() -> dict[tuple[str, str], Path]:
    """Index exact dependency checkouts already installed by Lake."""

    manifest_path = PROJECT_ROOT / "lake-manifest.json"
    packages_dir = PROJECT_ROOT / ".lake" / "packages"
    if not manifest_path.is_file() or not packages_dir.is_dir():
        return {}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkouts: dict[tuple[str, str], Path] = {}
    for package in manifest.get("packages", []):
        if not all(key in package for key in ("name", "url", "rev")):
            continue
        checkout = packages_dir / package["name"]
        if not checkout.is_dir():
            continue
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=checkout,
            check=True,
            text=True,
            capture_output=True,
        )
        actual_commit = result.stdout.strip().lower()
        expected_commit = package["rev"].lower()
        if actual_commit != expected_commit:
            raise RuntimeError(
                f"Lake checkout {checkout} is at {actual_commit}, expected "
                f"{expected_commit}; run `lake update` before extraction"
            )
        checkouts[_repository_key(package["url"], expected_commit)] = checkout
    return checkouts


def _use_local_dependency_configs(LeanGitRepo: Any) -> None:
    """Avoid LeanDojo's unbounded raw-GitHub reads for pinned dependencies."""

    if getattr(LeanGitRepo, "_trustmebro_local_configs", False):
        return

    checkouts = _local_lake_checkouts()
    original_get_config = LeanGitRepo.get_config

    def get_config(self: Any, filename: str, num_retries: int = 2) -> dict[str, Any]:
        checkout = checkouts.get(_repository_key(self.url, self.commit))
        config_path = checkout / filename if checkout is not None else None
        if config_path is None or not config_path.is_file():
            return original_get_config(self, filename, num_retries)

        content = config_path.read_text(encoding="utf-8")
        if filename.endswith(".toml"):
            return tomllib.loads(content)
        if filename.endswith(".json"):
            return json.loads(content)
        return {"content": content}

    LeanGitRepo.get_config = get_config
    LeanGitRepo._trustmebro_local_configs = True


def _replace_once(text: str, old: str, new: str, description: str) -> str:
    """Apply one guarded patch to LeanDojo's version-specific Lean source."""

    occurrences = text.count(old)
    if occurrences != 1:
        raise RuntimeError(
            f"LeanDojo's extractor has {occurrences} matches for {description}; "
            "refusing an unsafe patch"
        )
    return text.replace(old, new)


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _use_compiled_lean_dojo_extractor(trace_module: Any, data_dir: Path) -> None:
    """Build LeanDojo's Lean extractor once and use it for per-file workers.

    LeanDojo normally launches ``lean --run ExtractData.lean FILE`` for every
    source file. Its outer scheduler still needs to run the script once, but a
    generated scheduler copy delegates each file to the native executable.
    The executable and scheduler both come from the exact ExtractData.lean
    shipped by the installed lean-dojo-v2 package, preserving its output.
    """

    source = Path(trace_module.LEAN4_DATA_EXTRACTOR_PATH).resolve()
    if not source.is_file():
        raise RuntimeError(f"LeanDojo extractor source does not exist: {source}")

    original = source.read_text(encoding="utf-8")
    old_main = "unsafe def main (args : List String) : IO Unit := do\n"
    scheduler_dir = data_dir / "runtime" / "compiled-extractor"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    compiled_source = scheduler_dir / "ExtractData.lean"
    compiled_text = _replace_once(
        original,
        old_main,
        old_main + "  initSearchPath (← findSysroot)\n",
        "the main definition",
    )
    old_lean_lib = "  let leanLib ← getLibDir (← getBuildDir)\n"
    compiled_text = _replace_once(
        compiled_text,
        old_lean_lib,
        "  let leanLib ← getLibDir (← findSysroot)\n",
        "the Lean library lookup",
    )
    old_path_assertion = "  assert! ← path.pathExists\n  return path\n"
    diagnostic_path_check = (
        "  unless ← path.pathExists do\n"
        '    throw <| IO.userError s!"Unable to map module {mod} from {olean} '
        'to source path {path} (cwd: {cwd})"\n'
        "  return path\n"
    )
    compiled_text = _replace_once(
        compiled_text,
        old_path_assertion,
        diagnostic_path_check,
        "the source-path assertion",
    )
    _atomic_write(compiled_source, compiled_text)

    try:
        subprocess.run(
            ("lake", "build", "trustmebro-extract-data"),
            cwd=PROJECT_ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Failed to build the compiled LeanDojo extractor") from error

    if not COMPILED_EXTRACTOR.is_file():
        raise RuntimeError(
            f"Lake reported success but did not create {COMPILED_EXTRACTOR}"
        )

    old_worker = (
        '{cmd := "lake", args := #["env", "lean", "--run", '
        '"ExtractData.lean", path.toString]}'
    )
    compiled_worker = (
        f"{{cmd := {json.dumps(str(COMPILED_EXTRACTOR.resolve()))}, "
        "args := #[path.toString]}"
    )
    # trace.py copies this file by basename and then unconditionally executes
    # ``ExtractData.lean``, so the generated scheduler must keep that name.
    scheduler = scheduler_dir / "scheduler" / "ExtractData.lean"
    scheduler.parent.mkdir(parents=True, exist_ok=True)
    generated = _replace_once(
        compiled_text, old_worker, compiled_worker, "the per-file worker command"
    )
    _atomic_write(scheduler, generated)
    trace_module.LEAN4_DATA_EXTRACTOR_PATH = scheduler


def _lean_dojo_api(data_dir: Path) -> tuple[Any, Any]:
    LeanGitRepo, trace_module, _, _ = _lean_dojo_components(data_dir)
    return LeanGitRepo, trace_module.trace


def _lean_dojo_components(data_dir: Path) -> tuple[Any, Any, Any, Any]:
    _configure_runtime(data_dir)
    if not os.environ.get("GITHUB_ACCESS_TOKEN"):
        raise SystemExit(
            "LeanDojo v2 requires GITHUB_ACCESS_TOKEN during import. Export a "
            "GitHub token before running extraction."
        )

    try:
        from lean_dojo_v2.lean_dojo.data_extraction import (  # pyright: ignore[reportMissingImports]
            ast as ast_module,  # pyright: ignore[reportMissingImports]
        )
        from lean_dojo_v2.lean_dojo.data_extraction import (  # pyright: ignore[reportMissingImports]
            trace as trace_module,  # pyright: ignore[reportMissingImports]
        )
        from lean_dojo_v2.lean_dojo.data_extraction.lean import (  # pyright: ignore[reportMissingImports]
            LeanGitRepo,  # pyright: ignore[reportMissingImports]
        )
        from lean_dojo_v2.lean_dojo.data_extraction.traced_data import (  # pyright: ignore[reportMissingImports]
            TracedFile,  # pyright: ignore[reportMissingImports]
        )
        from lean_dojo_v2.utils.filesystem import (  # pyright: ignore[reportMissingImports]
            working_directory,  # pyright: ignore[reportMissingImports]
        )
    except ModuleNotFoundError as error:
        raise SystemExit(
            "LeanDojo v2 is not installed. Run this command with "
            "`uv run --isolated --frozen --group extraction`."
        ) from error
    _use_local_dependency_configs(LeanGitRepo)
    _support_lean_434_ast(ast_module)
    _use_compiled_lean_dojo_extractor(trace_module, data_dir)
    return LeanGitRepo, trace_module, TracedFile, working_directory


def _support_lean_434_ast(ast_module: Any) -> None:
    """Teach LeanDojo v2's declaration wrapper about ``coinductive``.

    LeanDojo v2 1.0.9 predates Lean 4.34's parser node for coinductive
    declarations. The generic child tree is already parsed correctly, but the
    surrounding declaration wrapper rejects it using a hard-coded assertion.
    Coinductive declarations are not theorem nodes and therefore need no
    special extraction logic; retaining their generic subtree lets traversal
    continue to later declarations in the same file.
    """

    declaration = ast_module.CommandDeclarationNode
    if getattr(declaration, "_trustmebro_supports_coinductive", False):
        return

    original = declaration.from_data.__func__

    def from_data(cls: Any, node_data: dict[str, Any], lean_file: Any) -> Any:
        try:
            return original(cls, node_data, lean_file)
        except AssertionError:
            args = node_data.get("args", ())
            if len(args) < 2:
                raise
            child = args[1].get("node", {})
            if child.get("kind") != "Lean.Parser.Command.coinductive":
                raise
            children = ast_module._parse_children(node_data, lean_file)
            return cls(lean_file, None, None, children, None)

    declaration.from_data = classmethod(from_data)
    declaration._trustmebro_supports_coinductive = True


def _atomic_records(traced_repo: Any) -> Iterator[dict[str, Any]]:
    yield from _atomic_theorem_records(
        traced_repo.get_traced_theorems(),
        traced_repo.repo,
    )


def _atomic_theorem_records(
    traced_theorems: Iterable[Any],
    repository: Any,
) -> Iterator[dict[str, Any]]:
    for traced_theorem in traced_theorems:
        # build_deps=False should already enforce this. Keep the check explicit
        # so a future LeanDojo change cannot silently mix dependency proofs in.
        if traced_theorem.repo != repository:
            continue

        file_path = str(traced_theorem.file_path)
        theorem = traced_theorem.theorem.full_name
        for tactic_index, traced_tactic in enumerate(
            traced_theorem.get_traced_tactics(atomic_only=True)
        ):
            tactic = traced_tactic.tactic.strip()
            yield {
                "file_path": file_path,
                "theorem": theorem,
                "tactic_index": tactic_index,
                "state_before": traced_tactic.state_before,
                "tactic": tactic,
                "tactic_head": tactic_head(tactic),
            }


def _validate_trace_root(trace_root: Path, expected_commit: str) -> tuple[Path, ...]:
    """Validate a complete root-repository trace and return its AST paths."""

    trace_root = trace_root.resolve()
    if not (trace_root / ".git").exists():
        raise SystemExit(f"Trace root is not a Git checkout: {trace_root}")
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=trace_root,
        check=True,
        text=True,
        capture_output=True,
    )
    actual_commit = result.stdout.strip().lower()
    if actual_commit != expected_commit.lower():
        raise SystemExit(
            f"Trace root is at {actual_commit}, expected {expected_commit.lower()}"
        )

    build_dir = trace_root / ".lake" / "build"
    library_dir = build_dir / "lib" / "lean"
    ir_dir = build_dir / "ir"
    if not library_dir.is_dir() or not ir_dir.is_dir():
        raise SystemExit(f"Trace root has no completed Lake build: {trace_root}")

    oleans = {
        path.relative_to(library_dir).with_suffix("").as_posix()
        for path in library_dir.rglob("*.olean")
    }
    json_paths = tuple(sorted(ir_dir.rglob("*.ast.json")))
    jsons = {
        path.relative_to(ir_dir).as_posix().removesuffix(".ast.json")
        for path in json_paths
    }
    dependencies = {
        path.relative_to(ir_dir).as_posix().removesuffix(".dep_paths")
        for path in ir_dir.rglob("*.dep_paths")
    }
    missing_jsons = oleans - jsons
    missing_dependencies = oleans - dependencies
    unexpected_jsons = jsons - oleans
    if missing_jsons or missing_dependencies or unexpected_jsons:
        raise SystemExit(
            "Trace is incomplete: "
            f"{len(missing_jsons)} missing ASTs, "
            f"{len(missing_dependencies)} missing dependency files, and "
            f"{len(unexpected_jsons)} ASTs without compiled modules"
        )
    if not json_paths:
        raise SystemExit(f"Trace contains no AST files: {trace_root}")
    return json_paths


def _prepare_trace_root(
    trace_module: Any,
    working_directory: Any,
    repository: Any,
    data_dir: Path,
) -> Path:
    """Create or reuse a persistent raw trace without materializing all ASTs."""

    parent = data_dir / "raw-trace" / f"{repository.name}-{repository.commit[:12]}"
    trace_root = parent / repository.name
    if trace_root.exists():
        try:
            _validate_trace_root(trace_root, repository.commit)
        except SystemExit:
            print(f"Existing raw trace is incomplete; resuming under {trace_root}")
        else:
            print(f"Reusing complete raw trace under {trace_root}")
            return trace_root

    parent.mkdir(parents=True, exist_ok=True)
    with working_directory(parent):
        # LeanDojo's public trace() immediately loads every AST into memory.
        # Its lower-level extraction step stops after writing the raw per-file
        # artifacts, which we can safely consume one at a time.
        trace_module._trace(repository, build_deps=False)
    _validate_trace_root(trace_root, repository.commit)
    return trace_root


def _convert_trace_chunk(
    task: tuple[int, tuple[Path, ...], Path],
) -> tuple[int, int, CorpusStats]:
    """Convert one temporary chunk in a forked worker process."""

    if _STREAM_CONTEXT is None:
        raise RuntimeError("Streaming worker was not configured")
    trace_root, repository, TracedFile = _STREAM_CONTEXT
    chunk_index, json_paths, shard_path = task
    traced_repo = SimpleNamespace(repo=repository, dependencies={})
    stats = CorpusStats()
    with shard_path.open("w", encoding="utf-8") as output:
        for json_path in json_paths:
            try:
                traced_file = TracedFile.from_traced_file(
                    trace_root, json_path, repository
                )
                traced_file.traced_repo = traced_repo
                records = _atomic_theorem_records(
                    traced_file.get_traced_theorems(),
                    repository,
                )
                for record in records:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats.add(record)
            except BaseException as error:
                raise RuntimeError(f"Failed to convert AST file {json_path}") from error
    return chunk_index, len(json_paths), stats


def _write_trace_records(
    trace_root: Path,
    json_paths: Sequence[Path],
    repository: Any,
    TracedFile: Any,
    destination: Path,
    *,
    jobs: int,
    chunk_size: int = 10,
) -> CorpusStats:
    """Convert independent AST chunks in bounded processes and merge in order."""

    global _STREAM_CONTEXT

    from tqdm.auto import tqdm  # pyright: ignore[reportMissingModuleSource]

    if jobs < 1:
        raise ValueError("Conversion jobs must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    chunks = tuple(
        tuple(json_paths[start : start + chunk_size])
        for start in range(0, len(json_paths), chunk_size)
    )
    stats = CorpusStats()
    _STREAM_CONTEXT = (trace_root, repository, TracedFile)

    try:
        with tempfile.TemporaryDirectory(
            dir=destination.parent, prefix=".conversion-shards-"
        ) as shard_directory_name:
            shard_directory = Path(shard_directory_name)
            tasks = tuple(
                (index, chunk, shard_directory / f"{index:05d}.jsonl")
                for index, chunk in enumerate(chunks)
            )
            if jobs == 1:
                results = map(_convert_trace_chunk, tasks)
                pool = None
            else:
                context = multiprocessing.get_context("fork")
                pool = context.Pool(processes=jobs)
                results = pool.imap(_convert_trace_chunk, tasks)

            try:
                with (
                    temporary.open("wb") as output,
                    tqdm(
                        total=len(json_paths),
                        desc="Converting ASTs",
                        unit="file",
                        dynamic_ncols=True,
                    ) as progress,
                ):
                    for expected, (chunk_index, file_count, chunk_stats) in enumerate(
                        results
                    ):
                        if chunk_index != expected:
                            raise RuntimeError("AST chunks were returned out of order")
                        shard_path = tasks[chunk_index][2]
                        with shard_path.open("rb") as shard:
                            shutil.copyfileobj(shard, output)
                        shard_path.unlink()
                        stats.update(chunk_stats)
                        progress.update(file_count)
            except BaseException:
                if pool is not None:
                    pool.terminate()
                    pool.join()
                raise
            else:
                if pool is not None:
                    pool.close()
                    pool.join()

            temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        _STREAM_CONTEXT = None
    return stats


def _write_records(records: Iterator[dict[str, Any]], destination: Path) -> CorpusStats:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)

    stats = CorpusStats()
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats.add(record)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return stats


def _read_record_stats(source: Path) -> CorpusStats:
    """Recompute summary statistics from an existing transition JSONL file."""

    stats = CorpusStats()
    with source.open(encoding="utf-8") as records:
        for line_number, line in enumerate(records, 1):
            try:
                stats.add(json.loads(line))
            except (json.JSONDecodeError, KeyError) as error:
                raise SystemExit(f"Invalid record at {source}:{line_number}: {error}")
    return stats


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    )


def _summary_statistics(stats: CorpusStats, top: int) -> dict[str, Any]:
    return {
        "atomic_transitions": stats.transitions,
        "represented_files": len(stats.files),
        "represented_theorems": len(stats.theorem_lengths),
        "proof_length": _proof_length_summary(stats),
        "distinct_tactic_heads": len(stats.tactic_heads),
        "unparsed_tactic_heads": stats.transitions - sum(stats.tactic_heads.values()),
        "top_tactic_heads": stats.tactic_heads.most_common(top),
        "estimated_post_decomposition": {
            "definition": (
                "estimated tactic-family actions after the preliminary Stage 1 "
                "mapping and transparent decompositions, before adjacent-action merging"
            ),
            "validated_by_lean": False,
            "total_actions": stats.estimated_family_counts.total(),
            "family_counts": stats.estimated_family_counts.most_common(),
        },
    }


def refresh_summary(args: argparse.Namespace) -> None:
    """Refresh derived statistics without retracing or rewriting the corpus."""

    output_dir = args.data_dir.resolve() / "dataset"
    transitions_path = output_dir / "atomic-transitions.jsonl"
    summary_path = output_dir / "summary.json"
    if not transitions_path.is_file() or not summary_path.is_file():
        raise SystemExit(
            f"Existing dataset and summary are required under {output_dir}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stats = _read_record_stats(transitions_path)
    for key in tuple(summary):
        if key not in {"source", "extraction"}:
            del summary[key]
    summary.update(_summary_statistics(stats, args.top))
    _write_json(summary_path, summary)
    print(f"Refreshed summary from {stats.transitions:,} records in {transitions_path}")


def create_dataset(args: argparse.Namespace) -> None:
    if not COMMIT_HASH.fullmatch(args.commit):
        raise SystemExit("--commit must be a complete 40-character Git commit hash")

    data_dir = args.data_dir.resolve()
    output_dir = data_dir / "dataset"
    transitions_path = output_dir / "atomic-transitions.jsonl"
    summary_path = output_dir / "summary.json"
    if (transitions_path.exists() or summary_path.exists()) and not args.force:
        raise SystemExit(
            f"Output already exists under {output_dir}. Use --force to replace it."
        )

    LeanGitRepo, trace_module, TracedFile, working_directory = _lean_dojo_components(
        data_dir
    )
    repository = LeanGitRepo(args.repository, args.commit.lower())
    if args.trace_root is None:
        trace_root = _prepare_trace_root(
            trace_module, working_directory, repository, data_dir
        )
    else:
        trace_root = args.trace_root.resolve()
        print(f"Reusing explicitly supplied raw trace under {trace_root}")
    json_paths = _validate_trace_root(trace_root, repository.commit)
    print(f"Validated {len(json_paths):,} complete per-file AST exports")

    stats = _write_trace_records(
        trace_root,
        json_paths,
        repository,
        TracedFile,
        transitions_path,
        jobs=args.jobs,
    )
    summary = {
        "source": {
            "repository": repository.url,
            "commit": repository.commit,
            "lean_version": repository.lean_version,
            "build_dependencies": False,
            "raw_trace_root": str(trace_root),
            "traced_files": len(json_paths),
        },
        "extraction": {
            "atomic_only": True,
            "conversion_jobs": args.jobs,
            "record_fields": [
                "file_path",
                "theorem",
                "tactic_index",
                "state_before",
                "tactic",
                "tactic_head",
            ],
        },
    }
    summary.update(_summary_statistics(stats, args.top))
    _write_json(summary_path, summary)

    print(f"Wrote {stats.transitions:,} retained transitions to {transitions_path}")
    print(f"Wrote extraction summary to {summary_path}")


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument(
        "--commit",
        required=False,
        help="complete 40-character commit hash; tags and branches are rejected",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--trace-root",
        type=Path,
        help=(
            "reuse a complete raw LeanDojo checkout containing .ast.json and "
            ".dep_paths exports instead of tracing again"
        ),
    )
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--jobs",
        type=_positive_int,
        default=DEFAULT_CONVERSION_JOBS,
        help=f"parallel AST conversion processes (default: {DEFAULT_CONVERSION_JOBS})",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="refresh summary.json from the existing JSONL without tracing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.summarize_existing:
        refresh_summary(args)
    else:
        if args.commit is None:
            raise SystemExit("--commit is required unless --summarize-existing is used")
        create_dataset(args)


if __name__ == "__main__":
    main()
