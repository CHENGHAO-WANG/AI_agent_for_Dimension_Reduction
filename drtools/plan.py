"""The plan schema and the gate that stands in front of execution.

A plan says what will be run, why each rejected method was rejected, and how the results
will be weighted — all before anything is computed. The validator's job is to catch the
plans that are incoherent given what the profile and reconnaissance already established,
so that the failure arrives as a sentence the planner can act on rather than as a
traceback forty minutes into a run.

The interesting part is that validation *simulates* the plan rather than pattern-matching
on it. Walking the stage list while tracking sample count, feature count, sparsity and
whether the values are still raw counts means the validator knows what each method will
actually receive. That is what lets it distinguish "Isomap on 100,000 points", which is
hopeless, from "subsample to 3,000 then Isomap", which is the correct way to do it —
where a rule keyed on the dataset's size alone would reject both.

Every rejection is logged. "The planner proposed X, the validator caught it, the agent
revised to Y" is a working agent loop with evidence, and it is worth more in the report
than a plan that happened to be right first time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from drtools.isolation import BUDGET_MAX_CANDIDATES
from drtools.pipeline import PipelineError, normalise_stages, validate_stages
from drtools.rank import RankingError, _check_weights
from drtools.registry import Registry, load_registry

Severity = Literal["error", "warning"]

# Methods whose geometry is Euclidean, and which therefore read raw counts as if
# sequencing depth were biology.
EUCLIDEAN_METHODS = frozenset(
    {"mds", "isomap", "lle", "laplacian_eigenmaps", "diffusion_maps", "tsne", "kernel_pca"}
)


# --------------------------------------------------------------------- the schema


class StageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: str
    params: dict[str, Any] = Field(default_factory=dict)


class CandidateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    stages: list[StageSpec]
    rationale: str = ""

    @field_validator("stages")
    @classmethod
    def _needs_at_least_one(cls, stages: list[StageSpec]) -> list[StageSpec]:
        if not stages:
            raise ValueError("a candidate needs at least one stage")
        return stages


class RejectionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    reason: str
    evidence: list[str] = Field(default_factory=list)


class EvaluationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: dict[str, float]
    justification: str = ""


class Plan(BaseModel):
    """What will be run, what was rejected, and how the results will be judged."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    instruction: str | None = None
    budget: Literal["fast", "standard", "thorough"] = "standard"
    max_candidates: int | None = Field(default=None, ge=1)
    base_preprocessing: list[StageSpec] = Field(default_factory=list)
    candidates: list[CandidateSpec]
    rejected: list[RejectionSpec] = Field(default_factory=list)
    evaluation: EvaluationSpec

    def stages_for(self, candidate: CandidateSpec) -> list[dict[str, Any]]:
        """The full stage list a candidate runs: shared base, then its own."""
        return [
            stage.model_dump()
            for stage in [*self.base_preprocessing, *candidate.stages]
        ]


# ------------------------------------------------------------------- plan state


@dataclass
class PlanState:
    """What the data looks like at a point in the plan, simulated rather than measured."""

    n_samples: int
    n_features: int
    is_sparse: bool
    is_raw_counts: bool
    normalised_per_sample: bool = False
    applied: list[str] = field(default_factory=list)

    def advance(self, op: str, params: dict[str, Any], registry: Registry) -> None:
        spec = registry[op]
        self.applied.append(op)

        if op == "subsample":
            self.n_samples = min(self.n_samples, int(params.get("n_samples") or 0) or self.n_samples)
        elif op == "select_variable_features":
            self.n_features = min(self.n_features, int(params.get("n_features") or self.n_features))
        elif op == "log1p":
            self.is_raw_counts = False
        elif op == "normalise_total":
            self.normalised_per_sample = True
        elif op == "l2_normalise":
            self.normalised_per_sample = True

        if spec.is_reduction:
            self.n_features = int(params.get("n_components") or 2)
            self.is_raw_counts = False
        if not spec.preserves_sparsity:
            self.is_sparse = False


# -------------------------------------------------------------------- validation


@dataclass
class Finding:
    code: str
    severity: Severity
    message: str
    fix: str
    candidate: str | None = None
    op: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "candidate": self.candidate,
            "op": self.op,
            "message": self.message,
            "fix": self.fix,
        }


def validate_plan(
    plan: Plan | dict[str, Any],
    profile: dict[str, Any],
    recon: dict[str, Any] | None = None,
    registry: Registry | None = None,
) -> dict[str, Any]:
    """Check a plan against what is already known about the data.

    Returns a report rather than raising, because the agent is expected to read every
    finding at once and revise, not to fix them one exception at a time.
    """
    registry = registry or load_registry()
    if isinstance(plan, dict):
        plan = Plan.model_validate(plan)

    findings: list[Finding] = []
    findings += _check_structure(plan, registry)
    findings += _check_weighting(plan)
    findings += _check_rejections(plan)
    for candidate in plan.candidates:
        findings += _check_candidate(plan, candidate, profile, recon, registry)

    errors = [f for f in findings if f.severity == "error"]
    return {
        "valid": not errors,
        "n_candidates": len(plan.candidates),
        "findings": [f.as_dict() for f in findings],
        "summary": _summary(errors, findings),
    }


def _check_ceiling(plan: Plan) -> list[Finding]:
    """How many Candidates this Run may register, and what the agent asked for.

    Attempts are capped per candidate id, and `_check_reregistration` refuses to let
    a registered id disappear, so the id set only grows. That makes its size the
    quantity that bounds a Run's total compute — and the quantity an agent mints to
    reset a spent per-candidate allowance. Capping it is what turns the
    exhaust-abandon-replace cycle from unbounded into `MAX_ATTEMPTS x ceiling`.

    A Plan may declare a tighter ceiling of its own. That buys no enforcement by
    itself — a bound the agent sets on itself is not a bound — but under the hard
    ceiling it is a checkable claim about self-restraint, which is what
    pre-registering the weighting already buys for the evaluation.
    """
    ceiling = BUDGET_MAX_CANDIDATES[plan.budget]
    declared = plan.max_candidates
    findings: list[Finding] = []

    if declared is not None and declared > ceiling:
        findings.append(
            Finding(
                code="declared_ceiling_above_budget",
                severity="error",
                message=f"the plan declares max_candidates={declared}, above the "
                f"{ceiling} the {plan.budget} budget allows. A ceiling the plan sets "
                "for itself can only tighten the budget's, never loosen it.",
                fix=f"declare max_candidates at {ceiling} or below, or omit it to "
                "accept the budget's ceiling",
            )
        )

    effective = min(declared, ceiling) if declared is not None else ceiling
    if len(plan.candidates) > effective:
        source = (
            f"declared max_candidates of {declared}"
            if declared is not None and declared < ceiling
            else f"{plan.budget} budget's ceiling of {ceiling}"
        )
        findings.append(
            Finding(
                code="candidates_exceed_ceiling",
                severity="error",
                message=f"this plan holds {len(plan.candidates)} candidates, and the "
                f"{source} allows {effective}. The ceiling is what bounds the run: "
                f"each candidate id may be attempted twice, so {effective} of them "
                f"caps the whole run at {effective * 2} executions.",
                fix=f"register at most {effective} candidates. A run cannot drop an "
                "id it has already registered, so one already at its ceiling has "
                "spent its scope of work: evaluate what succeeded and report it",
            )
        )
    return findings


def _check_structure(plan: Plan, registry: Registry) -> list[Finding]:
    findings: list[Finding] = []
    findings += _check_ceiling(plan)

    seen: set[str] = set()
    for candidate in plan.candidates:
        if candidate.id in seen:
            findings.append(
                Finding(
                    code="duplicate_candidate_id",
                    severity="error",
                    candidate=candidate.id,
                    message=f"two candidates share the id {candidate.id!r}, so their "
                    "artefacts would overwrite each other",
                    fix="give every candidate a distinct id",
                )
            )
        seen.add(candidate.id)

        try:
            validate_stages(plan.stages_for(candidate), registry)
        except PipelineError as error:
            findings.append(
                Finding(
                    code="malformed_candidate",
                    severity="error",
                    candidate=candidate.id,
                    message=str(error),
                    fix="correct the stage list; the message names the problem",
                )
            )

    terminals = {
        normalise_stages([s.model_dump() for s in c.stages])[-1]["op"]
        for c in plan.candidates
    }
    if "pca" not in terminals:
        findings.append(
            Finding(
                code="no_linear_baseline",
                severity="error",
                message="no candidate ends in plain PCA. Without a linear baseline "
                "there is nothing to measure the nonlinear methods against, and a "
                "claim that the data needs a manifold method cannot be supported: if "
                "PCA does as well, the extra machinery bought nothing.",
                fix="add a candidate whose terminal stage is pca",
            )
        )
    return findings


def _check_weighting(plan: Plan) -> list[Finding]:
    try:
        _check_weights(plan.evaluation.weights)
    except RankingError as error:
        return [
            Finding(
                code="invalid_weighting",
                severity="error",
                message=str(error),
                fix="declare weights over known metrics that sum to 1.0",
            )
        ]
    if not plan.evaluation.justification.strip():
        return [
            Finding(
                code="unjustified_weighting",
                severity="warning",
                message="the weighting has no justification. Pre-registering weights "
                "only constrains anything if the reasoning is recorded with them; "
                "otherwise the report cannot say why this emphasis was chosen.",
                fix="state what about the instruction or the data profile led to this "
                "emphasis",
            )
        ]
    return []


def _check_rejections(plan: Plan) -> list[Finding]:
    findings: list[Finding] = []
    for rejection in plan.rejected:
        if not rejection.evidence:
            findings.append(
                Finding(
                    code="unevidenced_rejection",
                    severity="warning",
                    op=rejection.method,
                    message=f"{rejection.method} was rejected without citing evidence. "
                    "The rejections are the clearest demonstration that the agent "
                    "selected rather than sprayed, and an uncited one carries no weight.",
                    fix="cite the profile or recon keys that make this method "
                    "unsuitable, for example recon.neighbourhood.n_connected_components",
                )
            )
    return findings


def _check_candidate(
    plan: Plan,
    candidate: CandidateSpec,
    profile: dict[str, Any],
    recon: dict[str, Any] | None,
    registry: Registry,
) -> list[Finding]:
    findings: list[Finding] = []
    shape = profile.get("shape", {})
    values = profile.get("values", {})

    state = PlanState(
        n_samples=int(shape.get("n_samples", 0)),
        n_features=int(shape.get("n_features", 0)),
        is_sparse=shape.get("storage") == "sparse_csr",
        is_raw_counts=values.get("suspected_kind") == "counts",
    )

    for stage in plan.stages_for(candidate):
        op = stage["op"]
        if op not in registry:
            continue  # already reported by the structural check
        spec = registry[op]
        try:
            params, _ = registry.resolve_params(op, stage.get("params", {}))
        except Exception:
            continue  # already reported by the structural check

        findings += _check_stage_against_state(
            candidate.id, op, params, spec, state, recon
        )
        state.advance(op, params, registry)

    return findings


def _check_stage_against_state(
    candidate_id: str,
    op: str,
    params: dict[str, Any],
    spec: Any,
    state: PlanState,
    recon: dict[str, Any] | None,
) -> list[Finding]:
    findings: list[Finding] = []
    n = state.n_samples

    if spec.is_reduction and spec.scales_to is not None and n > spec.scales_to:
        findings.append(
            Finding(
                code="exceeds_scale_limit",
                severity="error",
                candidate=candidate_id,
                op=op,
                message=f"{op} receives {n:,} samples but is documented as practical to "
                f"about {spec.scales_to:,} ({spec.raw.get('complexity', 'see registry')}).",
                fix=f"insert a subsample stage before {op}, and record that the metrics "
                "then describe the subsample; or choose a method that scales",
            )
        )

    if state.is_sparse and not spec.handles_sparse:
        findings.append(
            Finding(
                code="sparse_into_dense_method",
                severity="error",
                candidate=candidate_id,
                op=op,
                message=f"{op} requires dense input but the data is still sparse at "
                f"this point ({n:,} x {state.n_features:,}).",
                fix=f"add a densify stage, which will cost about "
                f"{n * state.n_features * 8 / 1e9:.2f} GB, or reduce the feature count "
                "first with select_variable_features",
            )
        )

    if state.is_raw_counts and op in EUCLIDEAN_METHODS:
        findings.append(
            Finding(
                code="raw_counts_into_euclidean_method",
                severity="error",
                candidate=candidate_id,
                op=op,
                message=f"{op} measures Euclidean distance, but the values reaching it "
                "are still raw counts. Counts are heteroscedastic and their sample "
                "totals vary, so the leading structure recovered would be sequencing "
                "depth rather than anything biological.",
                fix="add normalise_total and log1p before this method",
            )
        )

    if op == "tsne":
        perplexity = float(params.get("perplexity") or 30.0)
        if perplexity >= n / 3:
            findings.append(
                Finding(
                    code="perplexity_too_large",
                    severity="error",
                    candidate=candidate_id,
                    op=op,
                    message=f"perplexity {perplexity:g} is at or above n/3 = {n / 3:.0f}; "
                    "each point's neighbourhood would span most of the data and the "
                    "embedding would degenerate.",
                    fix=f"use a perplexity well below {n / 3:.0f}, around "
                    f"{max(5, round(n / 100) * 5)} for this sample count",
                )
            )

    neighbours = params.get("n_neighbors")
    if neighbours is not None and int(neighbours) >= n:
        findings.append(
            Finding(
                code="neighbours_exceed_samples",
                severity="error",
                candidate=candidate_id,
                op=op,
                message=f"n_neighbors={neighbours} is not smaller than the {n:,} samples "
                "reaching this stage.",
                fix=f"choose n_neighbors below {n}",
            )
        )

    if (
        neighbours is not None
        and recon is not None
        and op in {"isomap", "lle", "laplacian_eigenmaps"}
    ):
        graph = recon.get("neighbourhood", {})
        probe_k, parts = graph.get("k"), graph.get("n_connected_components")
        if parts and parts > 1 and probe_k and int(neighbours) <= probe_k:
            findings.append(
                Finding(
                    code="likely_disconnected_graph",
                    severity="warning",
                    candidate=candidate_id,
                    op=op,
                    message=f"reconnaissance found the k={probe_k} neighbourhood graph "
                    f"split into {parts} components, and this stage uses "
                    f"n_neighbors={neighbours}. {op} needs a connected graph and will "
                    "most likely fail.",
                    fix=f"raise n_neighbors well above {probe_k}, or drop this candidate "
                    "and record the disconnection as the reason",
                )
            )

    return findings


def _summary(errors: list[Finding], findings: list[Finding]) -> str:
    warnings = [f for f in findings if f.severity == "warning"]
    if errors:
        return (
            f"{len(errors)} error(s) and {len(warnings)} warning(s). The plan was not "
            "accepted; revise the candidates named and resubmit."
        )
    if warnings:
        return f"accepted with {len(warnings)} warning(s)."
    return "accepted."
