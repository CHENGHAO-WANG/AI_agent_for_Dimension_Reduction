"""The `drtools` command line — the whole of the agent's action space.

Every stage is a subcommand that reads artefacts from a run directory, writes its own
artefact back, and prints that artefact as JSON on stdout. The agent works exclusively
through this interface, which is also what lets the rule-based planner and the test
suite drive the identical pipeline with no model in the loop.

Failures are written to stderr as plain sentences rather than tracebacks, because the
agent is expected to read them and act — a contract violation from an adapter it wrote
should tell it what to fix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from drtools import jsonio
from drtools.contract import ContractError
from drtools.loaders import available, load
from drtools.profile import profile_dataset
from drtools.recon import reconnaissance
from drtools.runs import RunDir

EXIT_CONTRACT_ERROR = 2
EXIT_USAGE_ERROR = 3


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return EXIT_USAGE_ERROR

    try:
        payload = args.handler(args)
    except ContractError as error:
        print(f"contract error: {error}", file=sys.stderr)
        return EXIT_CONTRACT_ERROR
    except FileNotFoundError as error:
        print(f"missing artefact: {error}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    if payload is not None:
        print(jsonio.dumps(payload))
    return 0


# ------------------------------------------------------------------------ parser


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drtools",
        description="Deterministic toolbox behind the dr-agent analysis agent.",
    )
    subparsers = parser.add_subparsers(dest="command")

    datasets = subparsers.add_parser(
        "datasets", help="list the dataset specs that load without an adapter"
    )
    datasets.set_defaults(handler=_cmd_datasets)

    profile = subparsers.add_parser(
        "profile", help="measure a dataset and state what the measurements imply"
    )
    _add_data_arguments(profile)
    _add_run_arguments(profile)
    profile.set_defaults(handler=_cmd_profile)

    recon = subparsers.add_parser(
        "recon",
        help="probe structure: spectrum, intrinsic dimension, neighbourhood graph",
    )
    _add_data_arguments(recon)
    _add_run_arguments(recon)
    recon.add_argument(
        "--k", type=int, default=None, help="neighbours for the k-NN graph probe"
    )
    recon.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="cap on samples used by the neighbour probes",
    )
    recon.set_defaults(handler=_cmd_recon)

    return parser


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        required=True,
        help="built-in name (see `drtools datasets`) or a path to a data file",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="path to a module exposing load(path) -> (X, labels, meta), for formats "
        "with no built-in loader",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="column holding labels, for tabular and AnnData inputs",
    )


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir", default=None, help="existing run directory to write into"
    )
    parser.add_argument(
        "--runs-root", default="runs", help="where new run directories are created"
    )
    parser.add_argument("--run-id", default=None, help="name for a new run directory")
    parser.add_argument("--seed", type=int, default=0, help="random seed")


# ---------------------------------------------------------------------- handlers


def _cmd_datasets(_: argparse.Namespace) -> dict[str, Any]:
    return available()


def _cmd_profile(args: argparse.Namespace) -> dict[str, Any]:
    X, labels, meta = _load(args)
    run = _open_run(args, meta)
    run.write_manifest(dataset=meta.get("name"), spec=args.data, seed=args.seed)

    profile = profile_dataset(X, labels, meta)
    profile["run_id"] = run.id
    run.write_artifact("profile.json", profile)
    return profile


def _cmd_recon(args: argparse.Namespace) -> dict[str, Any]:
    X, labels, meta = _load(args)
    run = _open_run(args, meta)

    if run.profile_path.exists():
        profile = run.read_artifact("profile.json")
    else:
        profile = profile_dataset(X, labels, meta)
        run.write_manifest(dataset=meta.get("name"), spec=args.data, seed=args.seed)
        run.write_artifact("profile.json", profile)

    recon = reconnaissance(
        X, labels, profile, seed=args.seed, max_samples=args.max_samples, k=args.k
    )
    recon["run_id"] = run.id
    run.write_artifact("recon.json", recon)

    representation = recon["probe_representation"]
    run.log_decision(
        stage="recon",
        question="Which representation should the structural probes be measured on?",
        chosen=" -> ".join(representation["transform"]) or "raw values",
        rationale=representation["reason"],
        evidence=["profile.values.suspected_kind", "profile.features.std_ratio_p95_p05"],
        options_considered=["raw values", "normalise_total -> log1p", "standardise"],
        actor="rules",
    )
    return recon


# ----------------------------------------------------------------------- helpers


def _load(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    kwargs: dict[str, Any] = {}
    if getattr(args, "label_column", None):
        kwargs["label_column"] = args.label_column
    return load(args.data, adapter=args.adapter, **kwargs)


def _open_run(args: argparse.Namespace, meta: dict[str, Any]) -> RunDir:
    if args.run_dir:
        return RunDir(Path(args.run_dir))
    return RunDir.create(
        Path(args.runs_root), meta.get("name", args.data), run_id=args.run_id
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
