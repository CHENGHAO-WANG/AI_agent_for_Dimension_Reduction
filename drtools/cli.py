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
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from drtools import jsonio
from drtools.cache import ensure_cache, is_cached, read_cache
from drtools.contract import ContractError
from drtools.executors import ExecutionError
from drtools.heuristics import suggest
from drtools.isolation import run_candidate
from drtools.loaders import available, load
from drtools.metrics import evaluate_embedding
from drtools.pipeline import PipelineError, run_pipeline
from drtools.plan import Plan, validate_plan
from drtools.profile import profile_dataset
from drtools.recon import reconnaissance
from drtools.rank import RankingError, rank_candidates
from drtools.registry import RegistryError, load_registry
from drtools.runs import RunDir
from drtools.viz import (
    figure_class_facet,
    figure_comparison,
    figure_embedding,
    figure_metrics,
    figure_scree,
    figure_shepard,
)

EXIT_CONTRACT_ERROR = 2
EXIT_USAGE_ERROR = 3
EXIT_EXECUTION_ERROR = 4
EXIT_PLAN_ERROR = 5


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
    except (PipelineError, RegistryError, RankingError) as error:
        print(f"plan error: {error}", file=sys.stderr)
        return EXIT_PLAN_ERROR
    except ExecutionError as error:
        print(f"execution error: {error}", file=sys.stderr)
        return EXIT_EXECUTION_ERROR
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

    methods = subparsers.add_parser(
        "methods",
        help="capability records for every op: what it preserves, assumes and destroys",
    )
    methods.add_argument(
        "--op", default=None, help="show one op in full instead of the whole registry"
    )
    methods.add_argument(
        "--kind",
        default=None,
        choices=["preprocessing", "reduction"],
        help="restrict the listing to one kind of op",
    )
    methods.set_defaults(handler=_cmd_methods)

    embed = subparsers.add_parser(
        "embed", help="run one candidate pipeline and save its embedding"
    )
    _add_data_arguments(embed)
    _add_run_arguments(embed)
    embed.add_argument(
        "--stages",
        required=True,
        help='ordered stages as JSON, e.g. \'[{"op":"pca","params":{"n_components":2}}]\''
        ", or @path to read that JSON from a file",
    )
    embed.add_argument(
        "--id", default="candidate", help="name for this candidate's artefacts"
    )
    embed.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="wall-clock cap in seconds; the candidate is stopped and recorded as a "
        "timeout rather than left running",
    )
    embed.add_argument(
        "--in-process",
        action="store_true",
        help="run in this process instead of an isolated one. Faster, but a method that "
        "exhausts memory takes the whole command down with it and leaves no record",
    )
    embed.set_defaults(handler=_cmd_embed)

    reference = subparsers.add_parser(
        "prepare-reference",
        help="run the shared base preprocessing once and cache it as the common "
        "representation every candidate is measured against",
    )
    _add_run_arguments(reference)
    reference.add_argument("--stages", required=True, help="base stages as JSON, or @path")
    reference.set_defaults(handler=_cmd_prepare_reference)

    evaluate = subparsers.add_parser(
        "evaluate", help="score one candidate's embedding against the reference"
    )
    _add_run_arguments(evaluate)
    evaluate.add_argument("--id", required=True, help="candidate to score")
    evaluate.add_argument(
        "--k", type=int, default=15, help="neighbourhood size for the local metrics"
    )
    evaluate.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="cap for the quadratic metrics; recorded with the results",
    )
    evaluate.set_defaults(handler=_cmd_evaluate)

    rank = subparsers.add_parser(
        "rank",
        help="combine every candidate's metrics under the weights declared in the plan",
    )
    _add_run_arguments(rank)
    rank.set_defaults(handler=_cmd_rank)

    validate = subparsers.add_parser(
        "validate-plan",
        help="check a plan against the profile and recon before anything is computed",
    )
    _add_run_arguments(validate)
    validate.add_argument(
        "--plan",
        default=None,
        help="plan JSON, or @path; defaults to plan.json in the run directory",
    )
    validate.set_defaults(handler=_cmd_validate_plan)

    suggest = subparsers.add_parser(
        "suggest-params",
        help="profile-derived starting values for an op, with the reasoning behind them",
    )
    _add_run_arguments(suggest)
    suggest.add_argument("--op", required=True, help="the op to suggest parameters for")
    suggest.set_defaults(handler=_cmd_suggest_params)

    figures = subparsers.add_parser(
        "figures", help="draw the standard figure set for a run"
    )
    _add_run_arguments(figures)
    figures.add_argument(
        "--theme", default="light", choices=["light", "dark"], help="palette to render in"
    )
    figures.add_argument(
        "--facet-candidate",
        default=None,
        help="candidate to draw the per-class facet for; defaults to the ranking winner",
    )
    figures.set_defaults(handler=_cmd_figures)

    return parser


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        default=None,
        help="built-in name (see `drtools datasets`) or a path to a data file; "
        "optional when --run-dir names a run that already holds a cached dataset",
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
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random seed; fixed when the run is created and immutable afterwards",
    )


# ---------------------------------------------------------------------- handlers


def _cmd_datasets(_: argparse.Namespace) -> dict[str, Any]:
    return available()


def _cmd_profile(args: argparse.Namespace) -> dict[str, Any]:
    run, X, labels, meta = _resolve_run(args)
    # The seed is resolved — and a mismatched request refused — before anything is
    # written, the same discipline _resolve_run applies to the dataset digest.
    seed = run_seed(run, args.seed)
    # The manifest is written only now: a refused re-profile must not have overwritten
    # the command, spec and timestamp of the run it just declined to touch.
    # `spec` falls back to the cached source, so omitting --data records where the data
    # came from rather than recording null.
    run.write_manifest(
        dataset=meta.get("name"),
        spec=args.data or meta.get("source"),
        seed=seed,
    )

    profile = profile_dataset(X, labels, meta)
    profile["run_id"] = run.id
    # Which matrix this describes, so a loader repair that changed nothing is visible as
    # having changed nothing rather than looking like it was never read.
    profile["dataset_digest"] = meta.get("dataset_digest")
    run.write_artifact("profile.json", profile)
    return profile


def _cmd_recon(args: argparse.Namespace) -> dict[str, Any]:
    run, X, labels, meta = _resolve_run(args)
    seed = run_seed(run, args.seed)

    if run.profile_path.exists():
        profile = run.read_artifact("profile.json")
    else:
        profile = profile_dataset(X, labels, meta)
        run.write_manifest(
            dataset=meta.get("name"), spec=args.data or meta.get("source"), seed=seed
        )
        run.write_artifact("profile.json", profile)

    recon = reconnaissance(
        X,
        labels,
        profile,
        seed=seed,
        max_samples=args.max_samples,
        k=args.k,
        thumbnail_path=run.path / "figures" / "recon_thumbnail.png",
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


def _cmd_methods(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_registry()
    if args.op:
        return registry.describe(args.op)
    selected = (
        registry.ops
        if args.kind is None
        else {n: s for n, s in registry.ops.items() if s.kind == args.kind}
    )
    return {
        "version": registry.version,
        "ops": {name: registry.describe(name) for name in selected},
    }


def _cmd_embed(args: argparse.Namespace) -> dict[str, Any]:
    stages = _read_stages(args.stages)
    run, X, labels, meta = _resolve_run(args)
    seed = run_seed(run, args.seed)

    if not args.in_process:
        # A candidate that exhausts memory or never converges cannot be caught in
        # process, so by default it runs somewhere that can be killed.
        return run_candidate(
            run, args.id, stages, seed=seed, timeout_s=args.timeout
        )

    result = run_pipeline(X, labels, stages, seed=seed)

    embeddings = run.path / "embeddings"
    np.save(embeddings / f"{args.id}.npy", result.embedding)
    if result.labels is not None:
        np.save(embeddings / f"{args.id}.labels.npy", result.labels)
    if result.context.sample_index is not None:
        np.save(embeddings / f"{args.id}.index.npy", result.context.sample_index)

    record = {
        "id": args.id,
        "status": "ok",
        "run_id": run.id,
        "dataset": meta.get("name"),
        "stages_requested": stages,
        "seed": seed,
        "embedding_path": str(embeddings / f"{args.id}.npy"),
        **result.as_dict(),
    }
    jsonio.write(embeddings / f"{args.id}.json", record)
    return record


def _cmd_prepare_reference(args: argparse.Namespace) -> dict[str, Any]:
    """Compute the representation every candidate is scored against.

    Candidates differ in their own stages but share a base, and comparing each one
    against its *own* input would measure different things under the same name. The
    shared base output is the common ground that makes the comparison mean something.
    """
    run = RunDir(Path(args.run_dir)) if args.run_dir else _open_run(args, {})
    seed = run_seed(run, args.seed)
    stages = _read_stages(args.stages)
    X, labels, meta = read_cache(run)

    result = (
        run_pipeline(
            X, labels, stages, seed=seed, require_terminal_reduction=False
        )
        if stages
        else None
    )
    reference = result.embedding if result is not None else X

    directory = run.path / "data"
    np.save(directory / "reference.npy", np.asarray(reference))
    record = {
        "stages": stages,
        "shape": list(np.shape(reference)),
        "path": str(directory / "reference.npy"),
        "stage_records": result.as_dict()["stages"] if result is not None else [],
    }
    jsonio.write(directory / "reference.json", record)
    return record


def _cmd_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run = RunDir(Path(args.run_dir)) if args.run_dir else _open_run(args, {})
    embeddings = run.path / "embeddings"

    candidate = jsonio.read(embeddings / f"{args.id}.json")
    if candidate.get("status") != "ok":
        raise FileNotFoundError(
            f"candidate {args.id!r} has status {candidate.get('status')!r} and produced "
            "no embedding to evaluate"
        )
    embedding = np.load(embeddings / f"{args.id}.npy")

    reference, labels = _reference_for(run, args.id)
    metrics = evaluate_embedding(
        reference,
        embedding,
        labels,
        k=args.k,
        seed=run_seed(run, args.seed),
        max_samples=args.max_samples,
        runtime_s=candidate.get("total_duration_s"),
    )
    metrics["id"] = args.id
    metrics["reference"] = (
        "base preprocessing output"
        if (run.path / "data" / "reference.npy").exists()
        else "loaded dataset as cached"
    )
    jsonio.write(run.path / "metrics" / f"{args.id}.json", metrics)
    return metrics


def _cmd_rank(args: argparse.Namespace) -> dict[str, Any]:
    run = RunDir(Path(args.run_dir)) if args.run_dir else _open_run(args, {})
    plan = Plan.model_validate(run.read_artifact("plan.json"))

    metrics_by_id: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    for candidate in plan.candidates:
        record_path = run.path / "embeddings" / f"{candidate.id}.json"
        metrics_path = run.path / "metrics" / f"{candidate.id}.json"
        if metrics_path.exists():
            metrics_by_id[candidate.id] = jsonio.read(metrics_path)
        elif record_path.exists():
            failures[candidate.id] = jsonio.read(record_path).get("failure", {})

    ranking = rank_candidates(
        metrics_by_id,
        plan.evaluation.weights,
        justification=plan.evaluation.justification,
        failures=failures,
    )
    run.write_artifact("ranking.json", ranking)
    run.log_decision(
        stage="rank",
        question="Which candidate best serves the question this analysis is answering?",
        chosen=ranking["winner"],
        rationale=plan.evaluation.justification
        or "weights were declared in the plan before any embedding was computed",
        evidence=[f"metrics.{candidate}" for candidate in metrics_by_id],
        options_considered=sorted(metrics_by_id),
        weights_applied=ranking["weights_applied"],
    )
    return ranking


def _cmd_validate_plan(args: argparse.Namespace) -> dict[str, Any]:
    run = RunDir(Path(args.run_dir)) if args.run_dir else _open_run(args, {})
    plan = (
        _read_json_argument(args.plan)
        if args.plan
        else run.read_artifact("plan.json")
    )
    profile = run.read_artifact("profile.json")
    recon = (
        run.read_artifact("recon.json") if run.recon_path.exists() else None
    )

    report = validate_plan(plan, profile, recon)
    jsonio.write(run.path / "plan_validation.json", report)

    for finding in report["findings"]:
        if finding["severity"] == "error":
            run.log_decision(
                stage="validate_plan",
                question=f"Is the proposed plan runnable as written? ({finding['code']})",
                chosen="rejected",
                rationale=finding["message"],
                evidence=["profile.shape.n_samples"],
                actor="validator",
                candidate=finding["candidate"],
                op=finding["op"],
                fix=finding["fix"],
            )
    return report


def _cmd_suggest_params(args: argparse.Namespace) -> dict[str, Any]:
    run = RunDir(Path(args.run_dir)) if args.run_dir else _open_run(args, {})
    profile = run.read_artifact("profile.json")
    recon = run.read_artifact("recon.json") if run.recon_path.exists() else None
    return {"op": args.op, "suggested": suggest(args.op, profile, recon)}


def _cmd_figures(args: argparse.Namespace) -> dict[str, Any]:
    """Draw the standard set. The agent picks which of these to put in the report."""
    run = RunDir(Path(args.run_dir)) if args.run_dir else _open_run(args, {})
    figures_dir = run.path / "figures"
    embeddings_dir = run.path / "embeddings"
    drawn: dict[str, Any] = {}

    _, labels, meta = read_cache(run)
    names = meta.get("label_names")

    successful = {}
    for record_path in sorted(embeddings_dir.glob("*.json")):
        record = jsonio.read(record_path)
        if record.get("status") == "ok":
            successful[record["id"]] = np.load(embeddings_dir / f"{record['id']}.npy")
    # Panels read left to right, so the best candidate belongs first. Sorting by name
    # would put the winner wherever its id happened to fall in the alphabet.
    successful = _in_rank_order(run, successful)

    if successful:
        drawn["comparison"] = figure_comparison(
            successful,
            _labels_for(run, next(iter(successful)), labels),
            names,
            figures_dir / "comparison.png",
            theme_name=args.theme,
        )
        for candidate_id, embedding in successful.items():
            drawn[f"embedding_{candidate_id}"] = figure_embedding(
                embedding,
                _labels_for(run, candidate_id, labels),
                names,
                figures_dir / f"embedding_{candidate_id}.png",
                title=candidate_id,
                theme_name=args.theme,
            )

    winner = args.facet_candidate or _winner(run) or (
        next(iter(successful)) if successful else None
    )
    if winner in successful and labels is not None:
        drawn["class_facet"] = figure_class_facet(
            successful[winner],
            _labels_for(run, winner, labels),
            names,
            figures_dir / "class_facet.png",
            title=f"Classes in {winner}",
            theme_name=args.theme,
        )

    metrics_by_id = {
        path.stem: jsonio.read(path) for path in sorted((run.path / "metrics").glob("*.json"))
    }
    if metrics_by_id:
        first = next(iter(metrics_by_id.values()))
        drawn["metrics"] = figure_metrics(
            metrics_by_id,
            figures_dir / "metrics.png",
            reference_values=first.get("reference_values"),
            theme_name=args.theme,
        )

    if run.recon_path.exists():
        recon = run.read_artifact("recon.json")
        spectrum = recon["spectrum"]["probe"]
        drawn["scree"] = figure_scree(
            spectrum["explained_variance_ratio"],
            figures_dir / "scree.png",
            elbow=spectrum.get("elbow"),
            theme_name=args.theme,
        )
        if recon.get("thumbnail", {}).get("drawn"):
            drawn["recon_thumbnail"] = recon["thumbnail"]

    if winner in successful:
        drawn["shepard"] = _draw_shepard(
            run, winner, successful[winner], metrics_by_id.get(winner), args.theme
        )

    jsonio.write(figures_dir / "figures.json", drawn)
    return drawn


def _draw_shepard(run, candidate_id, embedding, metrics, theme_name):
    """Distances before against distances after, on the same capped subsample."""
    from sklearn.metrics import pairwise_distances

    reference, _ = _reference_for(run, candidate_id)
    n = min(reference.shape[0], 800)
    rng = np.random.default_rng(0)
    index = np.sort(rng.choice(reference.shape[0], size=n, replace=False))

    before = pairwise_distances(reference[index])
    after = pairwise_distances(np.asarray(embedding)[index])
    upper = np.triu_indices_from(before, k=1)

    return figure_shepard(
        before[upper],
        after[upper],
        run.path / "figures" / "shepard.png",
        correlation=(metrics or {}).get("values", {}).get("shepard_correlation"),
        title=f"Shepard diagram — {candidate_id}",
        theme_name=theme_name,
    )


def _labels_for(run: RunDir, candidate_id: str, labels):
    """Labels subset to the rows a candidate kept, so colours line up with points."""
    if labels is None:
        return None
    index_path = run.path / "embeddings" / f"{candidate_id}.index.npy"
    return labels[np.load(index_path)] if index_path.exists() else labels


def _in_rank_order(run: RunDir, embeddings: dict[str, Any]) -> dict[str, Any]:
    """Order candidates by the ranking when one exists, keeping any extras at the end."""
    path = run.path / "ranking.json"
    if not path.exists():
        return embeddings
    order = [row["id"] for row in jsonio.read(path).get("ranking", [])]
    ranked = {name: embeddings[name] for name in order if name in embeddings}
    ranked.update({name: xy for name, xy in embeddings.items() if name not in ranked})
    return ranked


def _winner(run: RunDir) -> str | None:
    path = run.path / "ranking.json"
    return jsonio.read(path).get("winner") if path.exists() else None


# ----------------------------------------------------------------------- helpers


def _reference_for(run: RunDir, candidate_id: str):
    """The representation to measure against, subset to the rows the candidate kept."""
    directory = run.path / "data"
    reference_path = directory / "reference.npy"
    if reference_path.exists():
        reference = np.load(reference_path, mmap_mode="r")
        _, labels, _ = read_cache(run)
    else:
        reference, labels, _ = read_cache(run)

    index_path = run.path / "embeddings" / f"{candidate_id}.index.npy"
    if index_path.exists():
        index = np.load(index_path)
        reference = reference[index]
        labels = None if labels is None else labels[index]
    return reference, labels


def _read_json_argument(argument: str) -> Any:
    text = (
        Path(argument[1:]).read_text(encoding="utf-8")
        if argument.startswith("@")
        else argument
    )
    return json.loads(text)


def _read_stages(argument: str) -> list[Any]:
    """Stages come as inline JSON, or as @path for anything long enough to want a file."""
    text = (
        Path(argument[1:]).read_text(encoding="utf-8")
        if argument.startswith("@")
        else argument
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise PipelineError(f"--stages is not valid JSON: {error}") from None


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


def _resolve_run(args: argparse.Namespace) -> tuple[RunDir, Any, Any, dict[str, Any]]:
    """The run and the matrix every handler should work from.

    A run holds one dataset. When it already has a cache that is the dataset, and
    `--data` becomes a verification rather than a second source of truth: it is loaded,
    digested, and refused if it disagrees. Nothing is written before that check.
    """
    spec = getattr(args, "data", None)

    if args.run_dir:
        path = Path(args.run_dir)
        if not path.exists():
            raise ContractError(
                f"no run at {path}. Omit --run-dir to create one, or correct the path."
            )
        run = RunDir(path)
        if is_cached(run):
            if spec is None:
                X, labels, meta = read_cache(run)
                return run, X, labels, meta
            X, labels, meta = _load(args)
            meta = ensure_cache(run, X, labels, meta)   # refuses before anything writes
            X, labels, _ = read_cache(run)
            return run, X, labels, meta
        if spec is None:
            raise ContractError(
                f"the run at {path} holds no cached dataset yet, so --data is needed "
                "to say which dataset this run is of."
            )
        X, labels, meta = _load(args)
        return run, X, labels, ensure_cache(run, X, labels, meta)

    if spec is None:
        raise ContractError(
            "--data is required when no --run-dir is given: there is no run to read a "
            "cached dataset from."
        )
    X, labels, meta = _load(args)
    run = _open_run(args, meta)
    return run, X, labels, ensure_cache(run, X, labels, meta)


def run_seed(run: RunDir, requested: int | None) -> int:
    """The run's seed, which is fixed when the run is created.

    Every metric subsample is drawn from this. Letting a re-profile move it would let
    two candidates be measured on different rows without the registered plan changing,
    which is the comparability guarantee defeated by a route that looks compliant.
    """
    if not run.manifest_path.exists():
        return 0 if requested is None else requested

    recorded = jsonio.read(run.manifest_path).get("seed")
    if recorded is None:
        return 0 if requested is None else requested
    if requested is not None and requested != recorded:
        raise ContractError(
            f"this run was created with seed {recorded} and every measurement in it "
            f"is drawn from that seed, so it cannot be re-run with seed {requested}. "
            "Use the recorded seed, or start a new run for the new one."
        )
    return recorded


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
