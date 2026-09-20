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
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from drtools import jsonio
from drtools.cache import ensure_cache, is_cached, read_cache
from drtools.contract import ContractError
from drtools.executors import ExecutionError
from drtools.heuristics import suggest, suggest_base
from drtools.isolation import BUDGET_MAX_CANDIDATES, budget_timeout, run_candidate
from drtools.loaders import available, load
from drtools.metrics import METRIC_SAMPLE_CAP, evaluate_embedding, neighbourhood_size
from drtools.pipeline import PipelineError, run_pipeline
from drtools.plan import Plan, validate_plan
from drtools.profile import profile_dataset
from drtools.recon import reconnaissance
from drtools.rank import RankingError, rank_candidates
from drtools.registry import RegistryError, load_registry
from drtools.runs import MISSING, RunDir, resolve_evidence
from drtools.status import MAX_ATTEMPTS, attempts as candidate_attempts, run_status
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
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        # A malformed invocation (an unknown flag, a missing required one) is refused by
        # argparse itself, before any handler runs. It already wrote its message to
        # stderr; the contract for `main` is that a refusal returns rather than raises,
        # so that holds here too instead of only for errors raised past this point.
        # argparse's own exit code for a usage error is 2, which collides with
        # EXIT_CONTRACT_ERROR — an agent reading 2 would go looking for a
        # "contract error:" sentence that was never printed. `--help` exits 0 and
        # should still mean success; every other argparse exit means a usage error.
        return EXIT_USAGE_ERROR if error.code else 0
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

    status = subparsers.add_parser(
        "status", help="where this run stands: what has run, what is outstanding"
    )
    _add_run_arguments(status)
    status.set_defaults(handler=_cmd_status)

    log_decision = subparsers.add_parser(
        "log-decision",
        help="append one decision, refusing any evidence key that does not resolve",
    )
    _add_run_arguments(log_decision)
    log_decision.add_argument(
        "--json",
        required=True,
        help="decision JSON, or @path. Requires stage, question, chosen and rationale; "
        "evidence is a list of dotted keys, and may be empty",
    )
    log_decision.set_defaults(handler=_cmd_log_decision)

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
        "--id", default="candidate", help="name for this candidate's artefacts"
    )
    embed.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="wall-clock cap in seconds, defaulting to the registered plan's budget. "
        "A value above that budget's cap is refused: a budget that can be exceeded on "
        "request is not a resource the run was planned under",
    )
    embed.add_argument(
        "--in-process",
        action="store_true",
        help="run in this process instead of an isolated one. Faster, but uncapped "
        "and unrecorded if a method exhausts memory, so it is reserved for the test "
        "harness and refused otherwise",
    )
    embed.set_defaults(handler=_cmd_embed)

    reference = subparsers.add_parser(
        "prepare-reference",
        help="run the shared base preprocessing once and cache it as the common "
        "representation every candidate is measured against",
    )
    _add_run_arguments(reference)
    reference.set_defaults(handler=_cmd_prepare_reference)

    evaluate = subparsers.add_parser(
        "evaluate", help="score one candidate's embedding against the reference"
    )
    _add_run_arguments(evaluate)
    evaluate.add_argument("--id", required=True, help="candidate to score")
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

    suggest_base_parser = subparsers.add_parser(
        "suggest-base",
        help="base preprocessing the planner may adopt, from reconnaissance's rule",
    )
    _add_run_arguments(suggest_base_parser)
    suggest_base_parser.set_defaults(handler=_cmd_suggest_base)

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


def _cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    """Read-only, so the agent may ask as often as it likes."""
    return run_status(_require_run(args, create=False))


RESERVED_DECISION_FIELDS = frozenset(
    {
        "stage",
        "question",
        "chosen",
        "rationale",
        "evidence",
        "options_considered",
        "evidence_resolved",
        "timestamp",
        "actor",
    }
)

LIFECYCLE_STAGES = frozenset(
    {"recon", "embed", "rank", "validate_plan", "register_plan"}
)
"""Stages only the toolbox may write.

The decision log is the authority the freeze anchors on: `rank` cross-checks
registration against it, attempts are counted from it, and `status` derives the whole
state machine from it. That holds only while these records are produced by the command
that did the thing. A generic write route that could emit them would let a plan be
"registered" without validation, an attempt be spent without running, or a ranking be
claimed without scoring -- which is the pre-registration guarantee defeated by the
mechanism built to record it.
"""

LIFECYCLE_FIELDS = frozenset({"plan_digest", "outcome", "weights", "candidates"})
"""Fields the freeze reads off lifecycle records. Refused for the same reason."""


def _cmd_log_decision(args: argparse.Namespace) -> dict[str, Any]:
    """The agent's only route into the decision log.

    Every key is resolved before anything is written, and what it resolved to is
    written beside it. Validating at write time and reading at report time are
    different guarantees: artefacts are rewritten in place, so a key can resolve to
    one value when a decision is logged and to another when the report reads it.
    Keeping the reading makes that divergence detectable instead of invisible.

    It does not make a rationale true. A real key with a false reading passes, and the
    report must not imply otherwise.
    """
    run = _require_run(args)
    document = _read_json_argument(args.json, flag="--json")
    if not isinstance(document, dict):
        raise ContractError(
            "--json must be a JSON object describing one decision, with stage, "
            "question, chosen and rationale."
        )

    missing = [
        field
        for field in ("stage", "question", "chosen", "rationale")
        if not str(document.get(field) or "").strip()
    ]
    if missing:
        raise ContractError(
            f"this decision is missing {', '.join(missing)}. A decision records what "
            "was asked, what was chosen and why; without them there is nothing for the "
            "report to be generated from."
        )

    stage = document["stage"]
    if stage in LIFECYCLE_STAGES:
        raise ContractError(
            f"'{stage}' records are written by the command that performs that step, "
            "and the freeze reads them as evidence it happened: registration, "
            "attempts and ranking are all checked against this log. Recording one "
            "here would assert something the run did not do. Use a stage of your own "
            f"-- profile, plan, execute, evaluate -- and run `drtools {stage.replace('_', '-')}` "
            "to produce the real record."
        )

    forbidden = sorted(LIFECYCLE_FIELDS & set(document))
    if forbidden:
        raise ContractError(
            f"{', '.join(forbidden)} belong to the records the toolbox writes for "
            "itself, and the freeze reads them to decide what this run has already "
            "done. Describe the decision in `chosen` and `rationale` instead."
        )

    evidence = list(document.get("evidence") or [])
    artifacts = _artifacts_for_evidence(run)
    resolved = resolve_evidence(evidence, artifacts)
    unresolved = [key for key, value in resolved.items() if value is MISSING]
    if unresolved:
        raise ContractError(_unresolved_message(unresolved, artifacts))

    extra = {
        key: value
        for key, value in document.items()
        if key not in RESERVED_DECISION_FIELDS
    }
    run.log_decision(
        stage=stage,
        question=document["question"],
        chosen=document["chosen"],
        rationale=document["rationale"],
        evidence=evidence,
        options_considered=document.get("options_considered") or [],
        evidence_resolved=jsonio.jsonable(resolved),
        **extra,
    )
    return {
        "logged": document["stage"],
        "evidence_resolved": jsonio.jsonable(resolved),
    }


def _artifacts_for_evidence(run: RunDir) -> dict[str, Any]:
    """Everything an evidence key may cite, keyed by its artefact-root name.

    Keys are artefact-rooted — `profile.shape.n_samples`, not `shape.n_samples` — so
    that a citation resolves identically from the decision log, the report or a test.
    """
    artifacts: dict[str, Any] = {}
    for name, path in (
        ("profile", run.profile_path),
        ("recon", run.recon_path),
        ("plan", run.plan_path),
        ("ranking", run.path / "ranking.json"),
    ):
        if path.exists():
            artifacts[name] = jsonio.read(path)

    registered = run.path / "plan.registered.json"
    if registered.exists():
        # `plan.registered.*` is the authority a rationale should cite, so it is
        # reachable whether or not an unregistered plan.json is also sitting there.
        plan = dict(artifacts.get("plan") or {})
        plan["registered"] = jsonio.read(registered)
        artifacts["plan"] = plan

    metrics_dir = run.path / "metrics"
    if metrics_dir.exists():
        metrics = {
            path.stem: jsonio.read(path) for path in sorted(metrics_dir.glob("*.json"))
        }
        if metrics:
            artifacts["metrics"] = metrics
    return artifacts


def _unresolved_message(unresolved: list[str], artifacts: dict[str, Any]) -> str:
    """Name the broken key and the keys that do exist beside it.

    An agent told only that something failed will guess again; told what is there, it
    corrects.
    """
    lines = []
    for key in unresolved:
        parent, _, _ = key.rpartition(".")
        if parent:
            neighbour = resolve_evidence([parent], artifacts)[parent]
        else:
            neighbour = artifacts
        if isinstance(neighbour, dict) and neighbour:
            available = ", ".join(sorted(str(k) for k in neighbour))
            # ASCII only. A console on a legacy codepage renders U+2014 as a literal
            # "?", and this message exists to be read and acted on by the agent.
            lines.append(f"  {key} -- {parent or 'the run'} holds: {available}")
        else:
            roots = ", ".join(sorted(artifacts)) or "nothing yet"
            lines.append(f"  {key} — no such path. This run holds: {roots}")
    return (
        "these evidence keys do not resolve against this run's artefacts:\n"
        + "\n".join(lines)
        + "\nCite a key that exists, or drop it. An empty evidence list is allowed and "
        "renders as unsupported, but a citation pointing at nothing is a broken "
        "rationale rather than a missing measurement."
    )


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
    run, X, labels, _ = _resolve_run(args)
    plan = _registered_plan(run)

    # Refused before anything is invalidated or attempted: a request the run cannot
    # honour must leave the run exactly as it found it.
    if args.in_process and not os.environ.get("DRTOOLS_ALLOW_IN_PROCESS"):
        raise ContractError(
            "--in-process runs with no wall-clock cap, so no budget can bind it and a "
            "method that exhausts memory takes the toolbox down with it, leaving no "
            "record of the attempt. It exists for the test harness and is not "
            "available here. Drop the flag to run this candidate in an isolated "
            "process under the budget this run registered."
        )

    cap = budget_timeout(plan.budget)
    if args.timeout is not None and args.timeout > cap:
        raise ContractError(
            f"this run registered the {plan.budget} budget, which caps a candidate at "
            f"{cap:g}s, and {args.timeout:g}s was requested. Lower the timeout, or "
            "start a new run under a larger budget: the budget is frozen at "
            "registration because every candidate was planned and rejected under it."
        )
    timeout_s = cap if args.timeout is None else args.timeout

    candidate = next((c for c in plan.candidates if c.id == args.id), None)
    if candidate is None:
        registered = ", ".join(c.id for c in plan.candidates)
        raise ContractError(
            f"{args.id} is not a candidate in the registered plan. The plan registered "
            f"[{registered}]; add the candidate by re-registering, or embed one of those."
        )

    outcome_so_far = _recorded_outcome(run, args.id)
    if outcome_so_far == "ok":
        raise ContractError(
            f"candidate {args.id} has already produced an embedding. Its stages are "
            "fixed once it has succeeded, and replacing the result would leave the "
            "embedding disagreeing with the record of how it was produced. Register a "
            "new candidate id to try something different."
        )

    tries = candidate_attempts(run, args.id)
    if len(tries) >= MAX_ATTEMPTS:
        raise ContractError(_exhausted_message(args.id, plan))

    # Resolving the seed and the stages are the last two steps that can refuse, so
    # they run before the invalidation rather than after it. A conflicting --seed
    # used to delete the previous attempt and then refuse to replace it, and the
    # decision log records only how an attempt ended: the op, the error type and the
    # message live in the embeddings artefact and nowhere else, so the diagnosis the
    # retry existed to act on was destroyed by the refusal to run it.
    seed = run_seed(run, args.seed)
    stages = plan.stages_for(candidate)

    _invalidate_candidate(run, args.id)

    if not args.in_process:
        # A candidate that exhausts memory or never converges cannot be caught in
        # process, so by default it runs somewhere that can be killed.
        outcome = run_candidate(run, args.id, stages, seed=seed, timeout_s=timeout_s)
    else:
        try:
            result = run_pipeline(X, labels, stages, seed=seed)
        except (ExecutionError, PipelineError) as error:
            # An in-process failure has to become a record like any other. Letting it
            # propagate skipped the decision below, so the attempt was never counted
            # and the same failing command could be repeated past the two-attempt
            # allowance while `status` still reported no attempts at all. The isolated
            # path has always turned a failure into an outcome; this one now agrees.
            outcome = {
                "id": args.id,
                "status": "failed",
                "stages_requested": stages,
                "failure": {
                    "op": getattr(error, "op", None),
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            }
            jsonio.write(run.path / "embeddings" / f"{args.id}.json", outcome)
            _log_embed_outcome(run, args.id, candidate, plan, outcome)
            return outcome

        embeddings = run.path / "embeddings"
        np.save(embeddings / f"{args.id}.npy", result.embedding)
        if result.labels is not None:
            np.save(embeddings / f"{args.id}.labels.npy", result.labels)
        if result.context.sample_index is not None:
            np.save(embeddings / f"{args.id}.index.npy", result.context.sample_index)
        outcome = {"id": args.id, "status": "ok", **result.as_dict()}
        jsonio.write(embeddings / f"{args.id}.json", outcome)

    _log_embed_outcome(run, args.id, candidate, plan, outcome)
    return outcome


def _log_embed_outcome(
    run: RunDir, candidate_id: str, candidate: Any, plan: Plan, outcome: dict[str, Any]
) -> None:
    """Record one attempt. Every path that runs a candidate ends here.

    Shared rather than duplicated because the attempt count is derived from these
    records: a route that runs a candidate without writing one is a route that does
    not count, which is how the in-process path escaped the retry allowance.
    """
    run.log_decision(
        stage="embed",
        question=f"What did candidate {candidate_id} produce?",
        chosen=candidate_id,
        rationale=candidate.rationale or "the candidate as registered",
        evidence=["plan.registered.candidates"],
        candidate=candidate_id,
        outcome=outcome.get("status"),
        # Recomputed from the registration rather than read off the manifest. The
        # manifest is rewritten by `profile`, so a re-profile after registering used to
        # drop the key and leave every later `embed` dying on a KeyError with no
        # message and no route. `plan.registered.json` is the authority for what was
        # registered; the manifest only ever held a copy of it.
        plan_digest=_plan_digest(plan.model_dump(mode="json")),
    )


def _recorded_outcome(run: RunDir, candidate_id: str) -> str | None:
    """How this candidate last ended according to the decision log.

    The freeze anchors here rather than on files, because files can be deleted: an
    agent that removes `embeddings/` would otherwise be free to re-register a different
    weighting and re-run. An attempt that began but recorded no outcome — the process
    died, the machine rebooted — reads as `crashed`, which is revisable, so an
    interrupted run is recoverable rather than wedged.
    """
    attempts = [
        record
        for record in run.decisions()
        if record.get("stage") == "embed" and record.get("candidate") == candidate_id
    ]
    if not attempts:
        return None
    return attempts[-1].get("outcome") or "crashed"


def _invalidate_candidate(run: RunDir, candidate_id: str) -> None:
    """Clear everything derived from a previous attempt at this candidate.

    The toolbox does this rather than the agent, and does it before *every* attempt
    rather than only when stages change: the commonest retry of all is the same stages
    with a bigger budget, and leaving the old metrics in place lets a candidate that has
    just timed out be ranked on the scores of the run before it.
    """
    for path in (run.path / "embeddings").glob(f"{candidate_id}.*"):
        path.unlink()
    metrics = run.path / "metrics" / f"{candidate_id}.json"
    metrics.unlink(missing_ok=True)


def _plan_digest(plan: dict[str, Any]) -> str:
    """A digest over the plan as registered, so divergence is detectable."""
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _plan_from(document: Any, source: str) -> Plan:
    """Parse a plan document, refusing a malformed one rather than raising out of `main`.

    `plan.json` is the artefact the agent edits most often, and pydantic's own
    `ValidationError` is neither a `ContractError` nor a `FileNotFoundError`, so a typo
    in it used to escape as a traceback with no exit code the agent could read and no
    route out. The detail pydantic produces already names the offending field, which is
    exactly what the agent needs, so it is carried through rather than summarised away.
    """
    try:
        return Plan.model_validate(document)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'plan'}: {item['msg']}"
            for item in error.errors()
        )
        raise ContractError(
            f"{source} is not a valid plan — {problems}. Correct the fields named "
            "above and re-run: a plan needs `dataset`, an `evaluation` with `weights`, "
            "and `candidates`, each with an `id` and at least one stage. "
            "`drtools methods` lists the ops a stage may name."
        ) from error


def _registered_plan(run: RunDir) -> Plan:
    path = run.path / "plan.registered.json"
    if not path.exists():
        raise ContractError(
            "this run has no registered plan. Run `drtools validate-plan` to register "
            "one before embedding or ranking: the weighting has to be fixed before any "
            "embedding exists for pre-registration to mean anything."
        )
    return _plan_from(jsonio.read(path), "plan.registered.json")


def _check_reregistration(run: RunDir, existing: Plan, proposed: Plan) -> None:
    """What a plan may still change once a run has already registered one.

    Re-registering an identical plan stays legal, and so does adding a new candidate
    id — an agent revising a failed candidate depends on that. What it cannot do is
    move the weighting, move the shared base every candidate is scored against, make
    a candidate that already ran disappear from the record, or revise the stages of a
    candidate that has already succeeded. Everything else — adding a candidate,
    revising one that failed, timed out or crashed — is the one diagnose-and-retry the
    design promises, and is the clearest evidence of agency the run can produce.
    """
    if proposed.budget != existing.budget:
        raise ContractError(
            f"this run registered the {existing.budget} budget and the plan just "
            f"submitted declares {proposed.budget}. Every candidate was planned, "
            "timed and rejected under the registered budget, so moving it after "
            "results exist would describe a resource discipline the run did not keep. "
            "Start a new run for the changed budget."
        )

    if dict(proposed.evaluation.weights) != dict(existing.evaluation.weights):
        raise ContractError(
            "this run already registered a plan, and the weighting cannot move once "
            "registered — that is the entire guarantee registration exists to make. "
            "The sanctioned route is an amendment, which this toolbox does not "
            "implement yet, so a changed weighting means starting a new run."
        )

    existing_base = [stage.model_dump() for stage in existing.base_preprocessing]
    proposed_base = [stage.model_dump() for stage in proposed.base_preprocessing]
    if proposed_base != existing_base:
        raise ContractError(
            "this run already registered a plan with different base preprocessing. "
            "Every candidate is scored against its output, so moving it would "
            "invalidate every metric already computed against the old one. Start a "
            "new run for the changed base."
        )

    existing_ids = {candidate.id for candidate in existing.candidates}
    proposed_ids = {candidate.id for candidate in proposed.candidates}
    dropped = sorted(existing_ids - proposed_ids)
    if dropped:
        raise ContractError(
            f"this run already registered candidate(s) {dropped}, which are absent "
            "from the plan just submitted. A candidate that ran and lost is part of "
            "the record and cannot be made to disappear by re-registering; add it "
            "back, or start a new run to drop it."
        )

    proposed_by_id = {candidate.id: candidate for candidate in proposed.candidates}
    for candidate in existing.candidates:
        if proposed_by_id[candidate.id].stages == candidate.stages:
            continue
        if _recorded_outcome(run, candidate.id) == "ok":
            raise ContractError(
                f"candidate {candidate.id} has already succeeded, so its stages cannot "
                "be revised. Register a new candidate id for the variant."
            )


def _exhausted_message(candidate_id: str, plan: Plan) -> str:
    """Why a third attempt is refused, and what is actually available instead.

    The advice has to be conditional. Registering a new candidate id is what mints a
    fresh allowance, so offering it unconditionally makes this refusal the instruction
    manual for the loop the ceiling exists to bound — true, helpful, and the exact
    sentence that lets an exhausted run keep spending.
    """
    ceiling = BUDGET_MAX_CANDIDATES[plan.budget]
    if plan.max_candidates is not None:
        ceiling = min(plan.max_candidates, ceiling)
    remaining = ceiling - len(plan.candidates)

    opening = (
        f"candidate {candidate_id} has already had two attempts, which is one run "
        "and the one diagnose-and-retry the design allows. "
    )
    if remaining > 0:
        return opening + (
            "Register a new candidate id for a further variant, so that what was "
            f"already tried stays on the record: {remaining} further "
            f"{'id is' if remaining == 1 else 'ids are'} available under this run's "
            f"ceiling of {ceiling}."
        )
    return opening + (
        f"This run has registered all {ceiling} candidates its budget allows, so "
        "there is no further variant to register and no route that would add one. "
        "What was tried is on the record: evaluate the candidates that succeeded "
        "and report the run."
    )

def _cmd_prepare_reference(args: argparse.Namespace) -> dict[str, Any]:
    """Compute the representation every candidate is scored against.

    Candidates differ in their own stages but share a base, and comparing each one
    against its *own* input would measure different things under the same name. The
    shared base output is the common ground that makes the comparison mean something.
    The base stages are read from the registered plan rather than taken as an argument,
    so the reference always reflects what was pre-registered rather than whatever was
    typed at the command line that day.
    """
    run = _require_run(args)
    plan = _registered_plan(run)
    stages = [stage.model_dump() for stage in plan.base_preprocessing]
    X, labels, _ = read_cache(run)
    seed = run_seed(run, args.seed)

    directory = run.path / "data"
    if stages:
        result = run_pipeline(
            X, labels, stages, seed=seed, require_terminal_reduction=False
        )
        reference = result.embedding
        np.save(directory / "reference.npy", np.asarray(reference))
        stage_records = result.as_dict()["stages"]
    else:
        # No base preprocessing means the reference *is* the cache. Writing a copy would
        # add a second artefact to keep honest, and np.asarray on a sparse matrix writes
        # an unloadable 0-d object array.
        reference = X
        stage_records = []

    n_rows = int(reference.shape[0])
    record = {
        "stages": stages,
        "shape": list(np.shape(reference)),
        "n_rows": n_rows,
        "is_cache": not stages,
        "settings": {
            "k": neighbourhood_size(min(n_rows, METRIC_SAMPLE_CAP)),
            "max_samples": METRIC_SAMPLE_CAP,
            "seed": seed,
        },
        "stage_records": stage_records,
    }
    jsonio.write(directory / "reference.json", record)
    return record


def _cmd_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run = _require_run(args)
    # A seed that conflicts with the run's recorded one is still refused here, the same
    # discipline every other command follows — even though the value actually used
    # below always comes from the recorded settings, not from this call's return.
    run_seed(run, args.seed)
    plan = _registered_plan(run)
    reference_record = run.path / "data" / "reference.json"

    if plan.base_preprocessing and not reference_record.exists():
        raise ContractError(
            "the registered plan declares base preprocessing, so candidates must be "
            "scored against its output rather than against the raw cache. Run "
            "`drtools prepare-reference` before evaluating."
        )

    if reference_record.exists():
        record = jsonio.read(reference_record)
        settings = record["settings"]
        reference_rows = int(record["n_rows"])
    else:
        # No base preprocessing: the reference is the cache, and the same rule applies
        # to its row count.
        reference_rows = int(
            jsonio.read(run.path / "data" / "meta.json")["cached_shape"][0]
        )
        settings = {
            "k": neighbourhood_size(min(reference_rows, METRIC_SAMPLE_CAP)),
            "max_samples": METRIC_SAMPLE_CAP,
            "seed": run_seed(run, None),
        }

    record_path = run.path / "embeddings" / f"{args.id}.json"
    if not record_path.exists():
        raise ContractError(
            f"candidate {args.id!r} has no embedding in this run — `embed` has not run "
            f"for it, so there is nothing to score. Run `drtools embed --run-dir "
            f"{run.path} --id {args.id}` first, or evaluate a candidate that has."
        )
    candidate = jsonio.read(record_path)
    if candidate.get("status") != "ok":
        raise ContractError(
            f"candidate {args.id!r} has status {candidate.get('status')!r} and produced "
            "no embedding to evaluate. Fix what made it fail and re-run `embed` for "
            "this id, or evaluate a different candidate."
        )
    embedding = np.load(run.path / "embeddings" / f"{args.id}.npy")
    reference, labels = _reference_for(run, args.id)
    _check_rows_support_k(run, args.id, reference.shape[0], reference_rows, settings)

    metrics = evaluate_embedding(
        reference,
        embedding,
        labels,
        k=settings["k"],
        seed=settings["seed"],
        max_samples=settings["max_samples"],
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


def _check_registration_is_recorded(run: RunDir, plan: Plan) -> None:
    """Check the registration against the append-only log that witnessed it.

    `plan.registered.json` is a file, and a file can be overwritten — one `cp` of
    `plan.json` over it makes the divergence check pass and lets a weighting chosen
    after the results were in look pre-registered. The decision log is what makes that
    more than a copy: `validate-plan` appends a `register_plan` record carrying the
    digest and the full weighting, so the registration has to agree with something that
    was written before any embedding existed. Tampering is thereby raised from editing
    one JSON file to forging a consistent log — which is the guarantee the design
    claimed and, until now, did not check anywhere.
    """
    digest = _plan_digest(plan.model_dump(mode="json"))
    weights = dict(plan.evaluation.weights)
    registrations = [
        record for record in run.decisions() if record.get("stage") == "register_plan"
    ]
    if not registrations:
        raise ContractError(
            "this run's decision log records no plan registration, so there is nothing "
            "showing the weighting was fixed before the results existed. Run `drtools "
            "validate-plan` to register the plan, or start a new run: a ranking whose "
            "pre-registration cannot be checked is not one the report can claim."
        )

    last = registrations[-1]
    if last.get("plan_digest") == digest and last.get("weights") == weights:
        return
    raise ContractError(
        "plan.registered.json does not match what this run's decision log recorded "
        f"being registered. The log's last registration is {last.get('weights')} under "
        f"digest {str(last.get('plan_digest'))[:12]}; the registration on disk is "
        f"{weights} under {digest[:12]}. The log is append-only and was written before "
        "any embedding existed, so it is the authority here and the file is not. "
        "Restore plan.registered.json from a backup if one exists, or start a new run "
        "and register the plan you actually mean to be judged by."
    )


def _check_rows_support_k(
    run: RunDir,
    candidate_id: str,
    candidate_rows: int,
    reference_rows: int,
    settings: dict[str, Any],
) -> None:
    """Refuse a candidate whose rows cannot carry the neighbourhood the run is fixed at.

    The battery's k is derived once, from the reference's row count, so that every
    candidate's trustworthiness is a measurement of the same thing. A candidate that
    subsampled has fewer rows than the reference, and scikit-learn needs `k < n / 2`;
    the tempting repair is to fall back to a smaller k for that one candidate, which is
    precisely how two candidates come to be scored at different neighbourhoods and
    ranked together with no caveat. So this refuses instead, and names the subsample as
    the reason rather than leaving the agent to infer it from an arithmetic error.
    """
    k = int(settings["k"])
    n_used = min(int(candidate_rows), int(settings["max_samples"]))
    if k < n_used / 2:
        return

    subsampled = (run.path / "embeddings" / f"{candidate_id}.index.npy").exists()
    kept = (
        f"candidate {candidate_id!r} subsampled to {candidate_rows} of the "
        f"reference's {reference_rows} rows"
        if subsampled
        else f"candidate {candidate_id!r} has {candidate_rows} rows against the "
        f"reference's {reference_rows}"
    )
    raise ContractError(
        f"{kept}, which cannot support this run's neighbourhood of k={k}: "
        f"scikit-learn needs k < n/2, so {n_used} rows admit at most "
        f"k={neighbourhood_size(n_used)}. k is fixed once from the reference so that "
        "every candidate is measured at the same neighbourhood, and scoring this one "
        "at a smaller k would make its numbers incomparable with the rest — which is "
        "the whole reason this refuses rather than adjusting. Drop the subsample stage "
        "and re-register this candidate under a new id, or start a new run whose "
        "dataset is the smaller sample so the reference is fixed to it."
    )


def _cmd_rank(args: argparse.Namespace) -> dict[str, Any]:
    run = _require_run(args)
    plan = _registered_plan(run)
    _check_registration_is_recorded(run, plan)

    live = run.path / "plan.json"
    if live.exists():
        current = _plan_from(run.read_artifact("plan.json"), "plan.json")
        if _plan_digest(current.model_dump(mode="json")) != _plan_digest(
            plan.model_dump(mode="json")
        ):
            raise ContractError(
                "plan.json no longer matches the plan this run registered, so ranking "
                "it would score results under a weighting chosen after they existed. "
                "Either copy plan.registered.json back over plan.json to restore what "
                "was registered, or start a new run for the changed plan. Copying the "
                "other way round is not a route: the registration is checked against "
                "the decision log, so overwriting it only makes `rank` refuse for a "
                "second reason. The sanctioned way to move a weighting is an "
                "amendment, which this toolbox does not yet implement."
            )

    metrics_by_id: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    for candidate in plan.candidates:
        record_path = run.path / "embeddings" / f"{candidate.id}.json"
        metrics_path = run.path / "metrics" / f"{candidate.id}.json"
        outcome = jsonio.read(record_path).get("status") if record_path.exists() else None
        if outcome == "ok" and metrics_path.exists():
            metrics_by_id[candidate.id] = jsonio.read(metrics_path)
        elif record_path.exists():
            failures[candidate.id] = jsonio.read(record_path).get("failure", {})

    ranking = rank_candidates(
        metrics_by_id,
        plan.evaluation.weights,
        justification=plan.evaluation.justification,
        failures=failures,
    )
    # Recomputed from the plan actually ranked, rather than read back off the
    # manifest: a hand-edited manifest would otherwise make the stamp lie, and the
    # manifest lookup has no guarantee the key is even there.
    ranking["plan_digest"] = _plan_digest(plan.model_dump(mode="json"))
    run.write_artifact("ranking.json", ranking)
    run.log_decision(
        stage="rank",
        question="Which candidate best serves the question this analysis is answering?",
        chosen=ranking["winner"],
        rationale=plan.evaluation.justification
        or "scored under the weighting registered for this run",
        evidence=[f"metrics.{candidate}" for candidate in metrics_by_id],
        options_considered=sorted(metrics_by_id),
        weights_applied=ranking["weights_applied"],
        plan_digest=ranking["plan_digest"],
    )
    return ranking


def _cmd_validate_plan(args: argparse.Namespace) -> dict[str, Any]:
    run = _require_run(args)
    source = args.plan if args.plan else "plan.json"
    document = (
        _read_json_argument(args.plan, flag="--plan")
        if args.plan
        else run.read_artifact("plan.json")
    )
    # Parsed before anything else reads it, so a malformed plan is one refusal naming
    # the bad field rather than a pydantic traceback out of the validator's own
    # `model_validate`.
    plan = _plan_from(document, source)

    # The freeze is checked before anything is written. A refusal that has already
    # overwritten `plan_validation.json` has left the run describing a plan it never
    # registered, which is the "refuse and write nothing" contract broken by the
    # command that exists to enforce contracts.
    registered_path = run.path / "plan.registered.json"
    if registered_path.exists():
        _check_reregistration(
            run,
            _plan_from(jsonio.read(registered_path), "plan.registered.json"),
            plan,
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

    errors = [f for f in report["findings"] if f["severity"] == "error"]
    if errors:
        return report

    # Registration is the moment the weighting becomes fixed: a plan that passes
    # validation is frozen here, before any embedding exists, so that ranking has
    # something pre-registered to hold itself to. A run that already registered a
    # plan may register again — but not to move what was already fixed.
    registered = plan
    digest = _plan_digest(registered.model_dump(mode="json"))
    jsonio.write(registered_path, registered.model_dump(mode="json"))
    run.update_manifest(plan_digest=digest)
    run.log_decision(
        stage="register_plan",
        question="What will this run compare, and how will the results be judged?",
        chosen=f"registered {len(registered.candidates)} candidates",
        rationale=registered.evaluation.justification
        or "the weighting was declared before this record was written",
        evidence=["profile.shape.n_samples"],
        plan_digest=digest,
        weights=dict(registered.evaluation.weights),
        candidates=[c.id for c in registered.candidates],
    )
    report["plan_digest"] = digest
    return report


def _cmd_suggest_params(args: argparse.Namespace) -> dict[str, Any]:
    run = _require_run(args)
    profile = run.read_artifact("profile.json")
    recon = run.read_artifact("recon.json") if run.recon_path.exists() else None
    return {"op": args.op, "suggested": suggest(args.op, profile, recon)}


def _cmd_suggest_base(args: argparse.Namespace) -> dict[str, Any]:
    """The default the planner accepts or overrides with a logged reason."""
    run = _require_run(args)
    profile = run.read_artifact("profile.json")
    recon = run.read_artifact("recon.json") if run.recon_path.exists() else None
    return suggest_base(profile, recon)


def _cmd_figures(args: argparse.Namespace) -> dict[str, Any]:
    """Draw the standard set. The agent picks which of these to put in the report."""
    run = _require_run(args)
    figures_dir = run.path / "figures"
    embeddings_dir = run.path / "embeddings"
    drawn: dict[str, Any] = {}

    if not is_cached(run):
        raise ContractError(
            f"the run at {run.path} holds no cached dataset, so there is nothing to "
            "draw labels or a reference from. Run `drtools profile --data ... "
            f"--run-dir {run.path}` first; that is the step that caches the dataset."
        )
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


def _require_run(args: argparse.Namespace, *, create: bool = True) -> RunDir:
    """The run a command must be given, rather than one it may create.

    `create=False` is for commands that only report: constructing a RunDir otherwise
    makes the run's subdirectories, which is a side effect a read-only command has no
    business having.
    """
    if not args.run_dir:
        raise ContractError("--run-dir is required: this command reads an existing run.")
    path = Path(args.run_dir)
    if not path.exists():
        raise ContractError(f"no run at {path}.")
    return RunDir(path, create=create)


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


def _read_json_argument(argument: str, *, flag: str) -> Any:
    """Parse a JSON argument, naming the source that did not parse.

    `_plan_from` already turned pydantic's ValidationError into a refusal. That is
    raised once the document has parsed; a document that never parses fails a step
    earlier, in `json.loads`, and `main` catches neither JSONDecodeError nor the
    ValueError it derives from. The agent read a traceback and exit 1, which is none
    of the four codes the toolbox documents and carries no route out.

    The flag and `@path` are two sources and the message says which was read, because
    `json.loads` reports only a line and a column.
    """
    from_file = argument.startswith("@")
    source = argument[1:] if from_file else flag
    text = Path(argument[1:]).read_text(encoding="utf-8") if from_file else argument
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ContractError(
            f"{source} is not valid JSON: {error}. Correct the document and run the "
            "command again."
        ) from error


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
