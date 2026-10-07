"""Run a small Lean 4.34.0 and LeanDojo-v2 compatibility experiment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from trustmebro.stage1.extract import DEFAULT_DATA_DIR, PROJECT_ROOT, _lean_dojo_api

SOURCE_PATHS = (
    Path("lakefile.toml"),
    Path("lean-toolchain"),
    Path("lake-manifest.json"),
    Path("src/trustmebro/extraction/Fixture.lean"),
)


def _run(*command: str, cwd: Path, env: dict[str, str] | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as error:
        output = (error.stdout or "").strip()
        detail = f"\n{output}" if output else ""
        raise RuntimeError(
            f"Command {command!r} failed with exit code {error.returncode}{detail}"
        ) from error
    return result.stdout.strip()


def _copy_source(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for relative_path in SOURCE_PATHS:
        source = PROJECT_ROOT / relative_path
        if not source.exists():
            raise SystemExit(
                f"Missing {source}. Run `lake update` in the project root first."
            )
        target = destination / relative_path
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _commit_fixture(fixture: Path) -> str:
    if not (fixture / ".git").exists():
        _run("git", "init", "--quiet", cwd=fixture)
        _run("git", "config", "user.name", "trustmebro compatibility", cwd=fixture)
        _run(
            "git",
            "config",
            "user.email",
            "compatibility@invalid.example",
            cwd=fixture,
        )

    # The generated repository is only a local LeanDojo fixture. Do not inherit
    # a user's global commit-signing policy, which may require unavailable keys
    # or agents and would make the experiment machine-dependent.
    _run("git", "config", "commit.gpgsign", "false", cwd=fixture)
    (fixture / ".git" / "info" / "exclude").write_text(
        "/.lake/\n", encoding="utf-8"
    )

    _run("git", "add", "-A", cwd=fixture)
    staged = subprocess.run(
        ("git", "diff", "--cached", "--quiet"), cwd=fixture, check=False
    )
    if staged.returncode not in (0, 1):
        raise SystemExit("Unable to inspect the generated compatibility repository")
    if staged.returncode == 1:
        environment = os.environ.copy()
        environment.update(
            {
                "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
                "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
            }
        )
        _run(
            "git",
            "commit",
            "--quiet",
            "-m",
            "LeanDojo compatibility fixture",
            cwd=fixture,
            env=environment,
        )
    return _run("git", "rev-parse", "HEAD", cwd=fixture)


def _reuse_dependency_cache(fixture: Path) -> None:
    source_packages = PROJECT_ROOT / ".lake" / "packages"
    if not source_packages.is_dir():
        raise SystemExit("Missing .lake/packages. Run `lake update` first.")

    fixture_lake = fixture / ".lake"
    fixture_lake.mkdir(exist_ok=True)
    fixture_packages = fixture_lake / "packages"
    if fixture_packages.is_symlink() or fixture_packages.is_file():
        fixture_packages.unlink()
    elif fixture_packages.is_dir():
        shutil.rmtree(fixture_packages)
    fixture_packages.symlink_to(source_packages, target_is_directory=True)


def _remove_fixture_build(fixture: Path) -> None:
    """Remove generated Lake state before LeanDojo copies the local repo."""

    fixture_lake = fixture / ".lake"
    if fixture_lake.exists():
        shutil.rmtree(fixture_lake)


def prepare_fixture(data_dir: Path) -> tuple[Path, str, str]:
    fixture = data_dir / "compatibility" / "repository"
    _copy_source(fixture)
    commit = _commit_fixture(fixture)
    _reuse_dependency_cache(fixture)
    try:
        build_output = _run("lake", "build", "Fixture", cwd=fixture)
    finally:
        # LeanDojo's local-repository setup starts with shutil.copytree(), which
        # follows the temporary packages symlink. Leaving it in place would
        # duplicate the root project's multi-gigabyte dependency cache.
        _remove_fixture_build(fixture)
    lean_version = _run("lean", "--version", cwd=fixture).splitlines()[0]
    if build_output:
        print(build_output)
    print(f"Built compatibility fixture at {fixture}")
    print(f"Fixture commit: {commit}")
    print(lean_version)
    return fixture, commit, lean_version


def _tactic_record(tactic: Any) -> dict[str, Any]:
    return {
        "tactic": tactic.tactic,
        "state_before": tactic.state_before,
        "state_after": tactic.state_after,
        "start": list(tactic.start) if tactic.start is not None else None,
        "end": list(tactic.end) if tactic.end is not None else None,
        "ast_class": type(tactic.ast).__name__,
    }


def _trace_report(traced_repo: Any, lean_version: str) -> dict[str, Any]:
    theorems: list[dict[str, Any]] = []
    total_all = 0
    total_atomic = 0
    for theorem in traced_repo.get_traced_theorems():
        if theorem.repo != traced_repo.repo:
            continue
        all_tactics = theorem.get_traced_tactics(atomic_only=False)
        atomic_tactics = theorem.get_traced_tactics(atomic_only=True)
        total_all += len(all_tactics)
        total_atomic += len(atomic_tactics)
        theorems.append(
            {
                "name": theorem.theorem.full_name,
                "file_path": str(theorem.file_path),
                "all_tactics": [_tactic_record(tactic) for tactic in all_tactics],
                "atomic_tactics": [
                    _tactic_record(tactic) for tactic in atomic_tactics
                ],
            }
        )

    expected = {
        "Trustmebro.Extraction.Fixture.direct",
        "Trustmebro.Extraction.Fixture.bullets",
        "Trustmebro.Extraction.Fixture.namedCases",
        "Trustmebro.Extraction.Fixture.sequencing",
        "Trustmebro.Extraction.Fixture.allGoals",
        "Trustmebro.Extraction.Fixture.firstBranch",
        "Trustmebro.Extraction.Fixture.ringExample",
        "Trustmebro.Extraction.Fixture.omegaExample",
    }
    found = {theorem["name"] for theorem in theorems}
    missing = sorted(expected - found)
    if missing:
        raise RuntimeError(f"LeanDojo did not trace expected theorems: {missing}")
    if total_atomic == 0:
        raise RuntimeError("LeanDojo returned no atomic tactics")

    return {
        "compatible": True,
        "lean_version": lean_version,
        "repository": traced_repo.repo.url,
        "commit": traced_repo.repo.commit,
        "all_tactic_records": total_all,
        "atomic_tactic_records": total_atomic,
        "theorems": theorems,
    }


def run_experiment(data_dir: Path, prepare_only: bool) -> None:
    data_dir = data_dir.resolve()
    fixture, _, lean_version = prepare_fixture(data_dir)
    if prepare_only:
        return

    LeanGitRepo, trace = _lean_dojo_api(data_dir)
    repository = LeanGitRepo.from_path(fixture)
    traced_repo = trace(repository, build_deps=False)
    report = _trace_report(traced_repo, lean_version)

    report_path = data_dir / "compatibility" / "report.json"
    temporary = report_path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(report_path)
    print(f"LeanDojo compatibility check passed; wrote {report_path}")
    for theorem in report["theorems"]:
        all_text = [tactic["tactic"] for tactic in theorem["all_tactics"]]
        atomic_text = [tactic["tactic"] for tactic in theorem["atomic_tactics"]]
        print(f"{theorem['name']}: all={all_text!r}; atomic={atomic_text!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="build the Lean fixture without importing or running LeanDojo v2",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    run_experiment(arguments.data_dir, arguments.prepare_only)


if __name__ == "__main__":
    main()
