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

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from drtools.constraints import RULES
from drtools.decision import (
    DataDecision,
    base_rule,
    recorded_checkpoint,
    recorded_decision,
)
from drtools.isolation import BUDGET_MAX_CANDIDATES
from drtools.pipeline import PipelineError, normalise_stages, validate_stages
from drtools.metrics import METRIC_SPECS
from drtools.rank import (
    DEFAULT_MARGIN,
    MAX_MARGIN,
    WEIGHT_TOLERANCE,
    RankingError,
    _check_weights,
    matching_default,
)
from drtools.registry import Registry, load_registry
from drtools.runs import MISSING, resolve_evidence, unresolved_message
from drtools.tuning import (
    D_CAP,
    DEFAULT_D_GRID,
    DEFAULT_FALLBACK_SHARE,
    DEFAULT_FLATNESS,
    DEFAULT_MAX_CYCLES,
    DEFAULT_MULTIPLIERS,
    DEFAULT_T_GRID,
)

Severity = Literal["error", "warning"]


# --------------------------------------------------------------------- the schema


class OverrideSpec(BaseModel):
    """Why a stage sets a parameter to something other than the persisted suggestion."""

    model_config = ConfigDict(extra="forbid")

    reason: str
    evidence: list[str] = Field(default_factory=list)


class DepartureSpec(BaseModel):
    """Why the base preprocessing departs from section 3.10's rule, argued from evidence."""

    model_config = ConfigDict(extra="forbid")

    reason: str
    evidence: list[str] = Field(default_factory=list)


class StageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: str
    params: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, OverrideSpec] = Field(default_factory=dict)


class CandidateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    stages: list[StageSpec]
    rationale: str = ""
    # Nomination is held to the standard rejection is: the keys that argue for running
    # this candidate, resolved at registration. The linear baseline alone is exempt,
    # since a rule rather than the data puts it in the portfolio.
    evidence: list[str] = Field(default_factory=list)

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
    evidence: list[str] = Field(default_factory=list)
    #: The non-inferiority margin on the weighted score (section 3.7). The default is
    #: the rule; another value needs `margin_departure`, argued from evidence. Its own
    #: field, since a reason to move the weights is no argument for another margin.
    margin: float = DEFAULT_MARGIN
    margin_departure: DepartureSpec | None = None


class TuningSpec(BaseModel):
    """How every candidate is tuned (sections 3.5 and 3.7, settled on day 15).

    One block for the whole plan, frozen at registration. The defaults are the rule;
    anything else is a departure, argued from evidence as a weighting's is. Which
    criterion chooses each candidate's d follows from its method, so it is not here.
    """

    model_config = ConfigDict(extra="forbid")

    d_grid: list[int] = Field(default_factory=lambda: list(DEFAULT_D_GRID))
    multipliers: list[float] = Field(default_factory=lambda: list(DEFAULT_MULTIPLIERS))
    t_grid: list[int] = Field(default_factory=lambda: list(DEFAULT_T_GRID))
    #: Absolute: a curve rising at most this much above its value at the smallest d.
    flatness: float = DEFAULT_FLATNESS
    #: Relative: the fallback's smallest d reaching this share of the best.
    fallback_share: float = DEFAULT_FALLBACK_SHARE
    max_cycles: int = DEFAULT_MAX_CYCLES
    departure: DepartureSpec | None = None

    def is_default(self) -> bool:
        return self.model_dump(exclude={"departure"}) == TuningSpec().model_dump(
            exclude={"departure"}
        )


class Plan(BaseModel):
    """What will be run, what was rejected, and how the results will be judged."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    instruction: str | None = None
    budget: Literal["fast", "standard", "thorough"] = "standard"
    max_candidates: int | None = Field(default=None, ge=1)
    base_preprocessing: list[StageSpec] = Field(default_factory=list)
    # Required whenever the base differs from what the rule gives for the recorded data
    # decision. drop_constant stays first whatever it says.
    base_departure: DepartureSpec | None = None
    candidates: list[CandidateSpec]
    rejected: list[RejectionSpec] = Field(default_factory=list)
    evaluation: EvaluationSpec
    tuning: TuningSpec = Field(default_factory=TuningSpec)

    def stages_for(self, candidate: CandidateSpec) -> list[dict[str, Any]]:
        """The full stage list a candidate runs: shared base, then its own."""
        return [
            stage.model_dump(exclude={"overrides"})
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
    # For the memory estimate: bytes per stored value, and the share of entries stored
    # while the matrix is sparse, taken from the profile.
    itemsize: int = 8
    density: float = 1.0

    @property
    def nbytes(self) -> float:
        """What the matrix at this point occupies: CSR's value and index per entry."""
        if self.is_sparse:
            stored = self.n_samples * self.n_features * self.density
            return stored * (self.itemsize + 4) + (self.n_samples + 1) * 4
        return float(self.n_samples) * self.n_features * self.itemsize

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

        if spec.is_visualization:
            self.n_features = 2
            self.is_raw_counts = False
        elif spec.is_reduction:
            # Tuning chooses d inside the Attempt, so the estimate takes the largest it
            # could choose: section 3.7's cap, below the features the stage receives.
            # The resolved parameters always hold the registry's default of 2, and a plan
            # may not give another, so the value is not read.
            self.n_features = max(1, min(D_CAP, self.n_features - 1))
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
    *,
    artifacts: dict[str, Any] | None = None,
    frozen_provenance: dict[str, list[dict[str, Any]]] | None = None,
    memory_limit_bytes: int | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check a plan against what is already known about the data.

    Returns a report rather than raising, because the agent is expected to read every
    finding at once and revise, not to fix them one exception at a time.

    `artifacts` is everything the plan's evidence keys may cite, keyed by artefact
    root as `log-decision` resolves them; without it, the profile and reconnaissance
    passed in are what may be cited. `frozen_provenance` is what an earlier
    registration recorded for candidates it registered with the same stages.
    `memory_limit_bytes` is the ceiling each candidate's estimated peak is held to;
    without it, memory is not checked.
    """
    registry = registry or load_registry()
    if isinstance(plan, dict):
        plan = Plan.model_validate(plan)
    if artifacts is None:
        artifacts = {"profile": profile, **({"recon": recon} if recon is not None else {})}

    decision = recorded_decision(recon)
    answers = recorded_checkpoint(checkpoint)
    purpose = answers.purpose if answers else "representation"
    focus = answers.focus if answers else "balanced"
    weighting_findings, weighting = _check_weighting(plan, profile, focus)
    base_findings, base = _check_base(plan, decision)
    tuning_findings, tuning = _check_tuning(plan, registry)
    margin_findings, margin = _check_margin(plan)
    findings: list[Finding] = []
    if answers is None:
        findings.append(
            Finding(
                code="no_checkpoint",
                severity="error",
                message="no checkpoint is recorded for this run, so its purpose and "
                "focus are unknown. The purpose decides which methods are eligible and "
                "whether candidates are ranked, and the focus which default weighting "
                "applies; registering without them would judge the plan by rules "
                "nobody chose.",
                fix="run `drtools checkpoint` with the purpose and the focus, or their "
                "defaults -- representation and balanced -- when nothing was answered",
            )
        )
    findings += _check_structure(plan, registry)
    findings += tuning_findings
    findings += margin_findings
    findings += base_findings
    findings += weighting_findings
    findings += _check_rejections(plan)
    findings += _check_accounting(plan, registry, purpose)
    findings += _check_evidence(plan, artifacts)
    peaks: dict[str, float] = {}
    for candidate in plan.candidates:
        findings += _check_candidate(
            plan, candidate, profile, recon, registry, decision, memory_limit_bytes
        )
        findings += _check_selection(plan, candidate, profile, registry, decision)
        peaks[candidate.id] = estimate_peak_bytes(plan.stages_for(candidate), profile, registry, decision)
    provenance_findings, provenance = _check_provenance(
        plan, profile, recon, registry, artifacts.get("suggestions") or {},
        frozen_provenance or {},
    )
    findings += provenance_findings

    errors = [f for f in findings if f.severity == "error"]
    return {
        "valid": not errors,
        "n_candidates": len(plan.candidates),
        "purpose": purpose,
        "focus": focus,
        "weighting": weighting,
        "base": base,
        "tuning": tuning,
        "margin": margin,
        "memory": {
            "limit_bytes": memory_limit_bytes,
            "peak_bytes": {cid: int(value) for cid, value in peaks.items()},
        },
        "provenance": provenance,
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

    # The base's output is the Reference every candidate is scored against. A method
    # there would make the Reference an embedding, and every score would then measure
    # agreement with that embedding rather than with the data.
    for position, base_stage in enumerate(plan.base_preprocessing):
        op = base_stage.op
        if op in registry and (registry[op].is_reduction or registry[op].is_visualization):
            findings.append(
                Finding(
                    code="reduction_in_base",
                    severity="error",
                    op=op,
                    message=f"base preprocessing stage {position} is {op}, a "
                    f"{registry[op].kind} method. The base's output is the Reference "
                    "every candidate is scored against, so a method there would turn "
                    "the Reference into an embedding, and every score would measure "
                    "agreement with that embedding instead of with the data.",
                    fix=f"move {op} out of base_preprocessing and into the candidates "
                    "that should run it",
                )
            )

    if not any(is_linear_baseline(candidate) for candidate in plan.candidates):
        findings.append(
            Finding(
                code="no_linear_baseline",
                severity="error",
                message="no candidate is the linear baseline: the base preprocessing "
                "followed by a single pca stage and nothing else. Without it there is "
                "nothing to measure the nonlinear methods against, and a claim that "
                "the data needs a manifold method cannot be supported: if PCA does as "
                "well, the extra machinery bought nothing. A PCA behind a subsample or "
                "a candidate's own preprocessing is a different pipeline, and scores "
                "differently for reasons that have nothing to do with linearity.",
                fix="add a candidate whose stages are exactly one pca stage",
            )
        )
    return findings


def _check_base(
    plan: Plan, decision: DataDecision | None
) -> tuple[list[Finding], str | None]:
    """The base against section 3.10's rule for the recorded data decision.

    Returns the findings and what the base is: `rule`, `departure`, or None when there
    is no decision to hold it to. drop_constant first is not negotiable; any other
    difference from the rule needs `base_departure`, a reason with evidence.
    """
    findings: list[Finding] = []
    if decision is None:
        findings.append(
            Finding(
                code="no_data_decision",
                severity="error",
                message="no data decision is recorded, so nothing says whether the values "
                "are raw counts or whether the features are of one type -- the two facts "
                "the preprocessing rules read. Reconnaissance records it.",
                fix="run `drtools recon --decision ...` and re-register",
            )
        )
        return findings, None

    given = [stage.model_dump(exclude={"overrides"}) for stage in plan.base_preprocessing]
    if not given or given[0]["op"] != "drop_constant":
        findings.append(
            Finding(
                code="drop_constant_not_first",
                severity="error",
                op=given[0]["op"] if given else None,
                message="the base preprocessing does not begin with drop_constant. A "
                "constant feature carries nothing, and any later scaling divides by its "
                "zero spread, so it is dropped first in every run whatever else the base "
                "does.",
                fix="make drop_constant the first stage of base_preprocessing",
            )
        )

    rule = base_rule(decision)
    if normalise_stages(given) == normalise_stages(rule):
        return findings, "rule"
    departure = plan.base_departure
    if departure is not None and departure.reason.strip() and departure.evidence:
        return findings, "departure"
    shown = " -> ".join(stage["op"] for stage in rule)
    findings.append(
        Finding(
            code="unexplained_base_departure",
            severity="error",
            message=f"the recorded data decision ({decision.values}, {decision.features}) "
            f"gives the base {shown}, with every parameter at its default, and this plan's "
            "base differs from it. A departure is allowed, argued from the data, and has "
            "to show as one.",
            fix=f"use the rule's base, {shown}, or give base_departure a reason and the "
            "evidence keys it rests on",
        )
    )
    return findings, "departure"


def tuned_by_rule(op: str, spec: Any, params: dict[str, Any]) -> set[str]:
    """The parameters tuning sets outright on a method's stage, which a plan may not.

    `n_components` on a reduction, `width_multiplier`, and any parameter the registry
    says tuning sweeps -- Diffusion Maps' `t`. A visualization method's `n_components`
    is 2 by rule, not by tuning, so it is not among them.
    """
    names = {"width_multiplier"} & set(spec.params)
    if spec.is_reduction:
        names.add("n_components")
    if spec.tuning is not None and spec.tuning.swept:
        names.add(spec.tuning.swept)
    return names


def _check_tuning(plan: Plan, registry: Registry) -> tuple[list[Finding], str]:
    """The tuning block is usable and argued, and nothing tuning chooses is set by hand.

    Returns the findings and whether the block is the `default` or a `departure`, which
    the registration record keeps. Three things a stage may not set (day 15): a
    reduction's `n_components`, which tuning chooses by the method's criterion; a PCA
    pre-step's `n_components`, which PCA's own criterion chooses as it chooses a PCA's
    d; and `width_multiplier`, which is how tuning scales a width.
    """
    findings: list[Finding] = []
    block = plan.tuning

    def problem(message: str, fix: str) -> None:
        findings.append(
            Finding(code="invalid_tuning", severity="error", message=message, fix=fix)
        )

    if not block.d_grid or block.d_grid != sorted(set(block.d_grid)) or block.d_grid[0] < 1:
        problem("tuning.d_grid must be a strictly increasing list of dimensions from 1",
                "list the grid of d in increasing order, without repeats")
    if (not block.multipliers or any(m <= 0 for m in block.multipliers)
            or len(set(block.multipliers)) != len(block.multipliers)):
        problem("tuning.multipliers must be distinct positive numbers",
                "list the multipliers applied to each suggestion, such as [0.5, 1, 2]")
    if not block.t_grid or any(t < 1 for t in block.t_grid):
        problem("tuning.t_grid must be diffusion times of 1 or more",
                "list the diffusion times swept, such as [1, 2, 4]")
    if not 0 <= block.flatness < 1:
        problem("tuning.flatness is an absolute difference on a 0-1 curve, so it lies in "
                "[0, 1)", "set flatness between 0 and 1")
    if not 0 < block.fallback_share <= 1:
        problem("tuning.fallback_share is a share of the best value, so it lies in (0, 1]",
                "set fallback_share between 0 and 1")
    if block.max_cycles < 1:
        problem("tuning.max_cycles must be at least 1", "allow one cycle or more")

    state = "default"
    if not block.is_default():
        state = "departure"
        departure = block.departure
        if departure is None or not departure.reason.strip() or not departure.evidence:
            findings.append(
                Finding(
                    code="unexplained_tuning_departure",
                    severity="error",
                    message="the tuning block differs from the defaults -- the d grid "
                    f"{list(DEFAULT_D_GRID)}, multipliers {list(DEFAULT_MULTIPLIERS)}, "
                    f"t in {list(DEFAULT_T_GRID)}, flatness {DEFAULT_FLATNESS}, fallback "
                    f"share {DEFAULT_FALLBACK_SHARE} and {DEFAULT_MAX_CYCLES} cycles -- "
                    "without a departure. A different grid is allowed, argued from the "
                    "data, and has to show as one.",
                    fix="restore the defaults, or give tuning.departure a reason and the "
                    "evidence keys it rests on",
                )
            )

    for candidate in plan.candidates:
        for position, stage in enumerate(candidate.stages):
            if stage.op not in registry:
                continue
            spec = registry[stage.op]
            last = position == len(candidate.stages) - 1
            for name in sorted(tuned_by_rule(stage.op, spec, {}) - {"n_components"}):
                if name not in stage.params:
                    continue
                what = (
                    "how tuning scales the width"
                    if name == "width_multiplier"
                    else f"swept by tuning over the plan's t_grid {plan.tuning.t_grid}"
                )
                fix = (
                    f"remove width_multiplier; to move the width's centre, set "
                    f"{spec.tuning.param} itself with an override reason"
                    if name == "width_multiplier"
                    else "remove t; to sweep other diffusion times, change tuning.t_grid "
                    "with a departure"
                )
                findings.append(
                    Finding(
                        code="set_by_tuning",
                        severity="error",
                        candidate=candidate.id,
                        op=stage.op,
                        message=f"{stage.op}.{name} is {what}; a plan never sets it.",
                        fix=fix,
                    )
                )
            if "n_components" not in stage.params or not spec.is_reduction:
                continue
            if last:
                findings.append(
                    Finding(
                        code="d_is_tuned",
                        severity="error",
                        candidate=candidate.id,
                        op=stage.op,
                        message=f"{stage.op}.n_components is set to "
                        f"{stage.params['n_components']}, but a reduction's d is chosen "
                        f"by tuning, by {stage.op}'s criterion "
                        f"({spec.tuning.criterion}), from the grid the plan declares. A "
                        "d set by hand skips that rule and the comparison of d across "
                        "candidates, while the record would still read as tuned.",
                        fix="remove n_components from this stage",
                    )
                )
            else:
                findings.append(
                    Finding(
                        code="k_is_chosen",
                        severity="error",
                        candidate=candidate.id,
                        op=stage.op,
                        message=f"the {stage.op} before the method sets n_components to "
                        f"{stage.params['n_components']}, but a PCA pre-step's output "
                        "dimension is chosen by PCA's own criterion, as a PCA's d is.",
                        fix="remove n_components from this stage",
                    )
                )
    return findings, state


def _check_margin(plan: Plan) -> tuple[list[Finding], str]:
    """The non-inferiority margin is in range and, if not the default, argued.

    Returns the findings and whether the margin is the `default` or a `departure`,
    which the registration record keeps. The range is fixed whatever the argument:
    zero ranks by score alone, with d breaking exact ties, and above `MAX_MARGIN` the
    comparison between candidates would call negligible a larger difference than the
    flatness test does within one.
    """
    evaluation = plan.evaluation
    findings: list[Finding] = []
    if not 0 <= evaluation.margin <= MAX_MARGIN:
        findings.append(
            Finding(
                code="invalid_margin",
                severity="error",
                message=f"evaluation.margin is {evaluation.margin}, outside [0, "
                f"{MAX_MARGIN}]. It is an absolute difference on the weighted score, "
                f"and wider than {MAX_MARGIN} it would treat as negligible between "
                "candidates more than the flatness test treats as negligible within one.",
                fix=f"set margin between 0 and {MAX_MARGIN}, or leave it at the "
                f"default {DEFAULT_MARGIN}",
            )
        )
    if abs(evaluation.margin - DEFAULT_MARGIN) <= WEIGHT_TOLERANCE:
        return findings, "default"
    departure = evaluation.margin_departure
    if departure is None or not departure.reason.strip() or not departure.evidence:
        findings.append(
            Finding(
                code="unexplained_margin_departure",
                severity="error",
                message=f"evaluation.margin is {evaluation.margin}, not the default "
                f"{DEFAULT_MARGIN}, and has no departure. A different margin is "
                "allowed, argued from the data, and has to show as one.",
                fix="restore the default, or give evaluation.margin_departure a reason "
                "and the evidence keys it rests on",
            )
        )
    return findings, "departure"


def is_linear_baseline(candidate: CandidateSpec) -> bool:
    """The base preprocessing plus a single `pca`, which is the only thing the name means."""
    return len(candidate.stages) == 1 and candidate.stages[0].op == "pca"


def _check_weighting(
    plan: Plan, profile: dict[str, Any], focus: str = "balanced"
) -> tuple[list[Finding], str]:
    """Whether the weighting is usable, which default it is, and whether it is argued.

    Returns the findings and what the weighting is: `default`,
    `default_trusted_labels`, or `departure`, which the registration record keeps.
    """
    evaluation = plan.evaluation
    try:
        _check_weights(evaluation.weights)
    except RankingError as error:
        return [
            Finding(
                code="invalid_weighting",
                severity="error",
                message=str(error),
                fix="declare weights over known metrics that sum to 1.0",
            )
        ], "departure"

    findings: list[Finding] = []
    labelled = bool(profile.get("labels", {}).get("present"))
    needing_labels = sorted(
        name
        for name, value in evaluation.weights.items()
        if value > 0 and METRIC_SPECS[name].requires_labels
    )
    if needing_labels and not labelled:
        findings.append(
            Finding(
                code="label_metric_without_labels",
                severity="error",
                message=f"the weighting puts weight on {needing_labels}, which need "
                "labels, and this dataset has none (profile.labels.present is false). "
                "The weight would be dropped and redistributed at ranking, so the "
                "ranking would answer a different question from the one registered.",
                fix="move that weight onto trustworthiness, continuity and "
                "shepard_correlation, or use the default: 0.25, 0.25 and 0.5",
            )
        )

    weighting = matching_default(evaluation.weights, focus) or "departure"
    if not evaluation.evidence:
        if labelled:
            findings.append(
                Finding(
                    code="uncited_weighting",
                    severity="error",
                    message="this dataset has labels, and the weighting cites no "
                    "evidence. Whether the labels are trusted -- supplied with the "
                    "data rather than derived from it, and not in an analysis meant to "
                    "find new groups -- is a decision, and either default makes it: "
                    "the labelled default says they are trusted, the unlabelled one "
                    "that they are not.",
                    fix="cite the keys that settle whether the labels are trusted, "
                    "for example profile.labels.kind",
                )
            )
        elif weighting == "departure":
            findings.append(
                Finding(
                    code="uncited_weighting",
                    severity="error",
                    message="the weighting departs from the default of 0.25 "
                    "trustworthiness, 0.25 continuity and 0.5 shepard_correlation, and "
                    "cites no evidence. A departure is allowed, but every departure has "
                    "to show as one, argued from the data.",
                    fix="cite the profile or recon keys that justify this emphasis in "
                    "evaluation.evidence, or use the default",
                )
            )

    if not evaluation.justification.strip():
        findings.append(
            Finding(
                code="unjustified_weighting",
                severity="warning",
                message="the weighting has no justification. Pre-registering weights "
                "only constrains anything if the reasoning is recorded with them; "
                "otherwise the report cannot say why this emphasis was chosen.",
                fix="state what about the instruction or the data profile led to this "
                "emphasis",
            )
        )
    return findings, weighting


def _check_rejections(plan: Plan) -> list[Finding]:
    findings: list[Finding] = []
    for rejection in plan.rejected:
        if not rejection.evidence:
            findings.append(
                Finding(
                    code="unevidenced_rejection",
                    severity="error",
                    op=rejection.method,
                    message=f"{rejection.method} was rejected without citing evidence. "
                    "The rejections are the clearest demonstration that the agent "
                    "selected rather than sprayed, and an uncited one carries no weight.",
                    fix="cite the profile or recon keys that make this method "
                    "unsuitable, for example recon.neighbourhood.n_connected_components",
                )
            )
    for candidate in plan.candidates:
        if not candidate.evidence and not is_linear_baseline(candidate):
            findings.append(
                Finding(
                    code="unevidenced_candidate",
                    severity="error",
                    candidate=candidate.id,
                    message=f"candidate {candidate.id} cites no evidence. A candidate "
                    "is held to the standard a rejection is: running a method is a "
                    "judgment about the data as much as ruling one out.",
                    fix="cite the profile or recon keys that make this pipeline worth "
                    "running in the candidate's evidence",
                )
            )
    return findings


def _methods(registry: Registry, purpose: str = "visualization") -> dict[str, Any]:
    """The methods a run of this purpose may run, and so must account for.

    A representation run's deliverable is a representation, so only reductions are
    eligible; a visualization run's is a picture, which either class can draw.
    """
    if purpose == "representation":
        return dict(registry.reductions())
    return {**registry.reductions(), **registry.visualization_methods()}


def _check_accounting(
    plan: Plan, registry: Registry, purpose: str = "representation"
) -> list[Finding]:
    """Every method is nominated, rejected, or both; a rejection names a method.

    Both is allowed: a rejection's reason can rule out one configuration while a
    candidate runs another. What cannot stand is rejecting a method and running it
    alone, since nothing then separates the configuration ruled out from the one run.
    Since day 17 the methods are those the run's purpose makes eligible: a
    visualization method in a representation run is excluded by rule, so it is neither
    run nor rejected.
    """
    methods = _methods(registry, purpose)
    findings: list[Finding] = []
    visual = registry.visualization_methods() if purpose == "representation" else {}

    for candidate in plan.candidates:
        for stage in candidate.stages:
            if stage.op in visual:
                findings.append(
                    Finding(
                        code="visualization_method_in_representation_run",
                        severity="error",
                        candidate=candidate.id,
                        op=stage.op,
                        message=f"{stage.op} is a visualization method, and this run's "
                        "purpose is representation. A visualization method's output is "
                        "a picture, never a representation that downstream analysis "
                        "can use.",
                        fix=f"drop candidate {candidate.id}; if the deliverable is a "
                        "picture, record the purpose as visualization at the "
                        "checkpoint of a new run",
                    )
                )

    for rejection in plan.rejected:
        if rejection.method in visual:
            findings.append(
                Finding(
                    code="excluded_by_purpose",
                    severity="error",
                    op=rejection.method,
                    message=f"{rejection.method} is rejected, but in a representation "
                    "run a visualization method is excluded by rule. A rejection "
                    "records a judgment about the data where the rule has already "
                    "decided.",
                    fix=f"remove the rejection of {rejection.method}",
                )
            )
        elif rejection.method not in methods:
            findings.append(
                Finding(
                    code="unknown_rejected_method",
                    severity="error",
                    op=rejection.method,
                    message=f"{rejection.method!r} is rejected, and it is not a "
                    "reduction or visualization method in the registry. The report "
                    "prints rejections as the methods this run considered, so a name "
                    "the registry does not know is a record of nothing.",
                    fix="name one of: " + ", ".join(sorted(methods)),
                )
            )

    in_candidates = {stage.op for c in plan.candidates for stage in c.stages}
    rejected = {rejection.method for rejection in plan.rejected}
    unaccounted = sorted(set(methods) - in_candidates - rejected)
    if unaccounted:
        findings.append(
            Finding(
                code="method_unaccounted",
                severity="error",
                message=f"{', '.join(unaccounted)} appear neither in a candidate nor "
                "in `rejected`. Every method is either run or ruled out with a "
                "reason, so the selection is visible in the record.",
                fix="nominate each in a candidate, or add it to `rejected` with a "
                "reason and evidence",
            )
        )

    for candidate in plan.candidates:
        run = [stage.op for stage in candidate.stages if stage.op in methods]
        if len(run) != 1 or run[0] not in rejected:
            continue
        findings.append(
            Finding(
                code="rejected_method_run_alone",
                severity="error",
                candidate=candidate.id,
                op=run[0],
                message=f"{run[0]} is rejected, and candidate {candidate.id} runs it "
                "as its only method. A method may be rejected and still run only "
                "behind another method, so that the candidate is visibly a different "
                "configuration from the one the rejection rules out.",
                fix=f"remove the rejection of {run[0]}, or drop candidate "
                f"{candidate.id}"
                + (
                    "; pca cannot be rejected, since the linear baseline runs it alone"
                    if run[0] == "pca"
                    else ""
                ),
            )
        )
    return findings


def _check_evidence(plan: Plan, artifacts: dict[str, Any]) -> list[Finding]:
    """Every key the plan cites resolves, as `log-decision` already requires.

    A plan may not cite `plan.*`: a key into the plan being registered, or into an
    earlier registration of it, resolves because the plan says so and proves nothing.
    """
    citable = {root: value for root, value in artifacts.items() if root != "plan"}
    places: list[tuple[str, str | None, str | None, list[str]]] = [
        ("the weighting", None, None, plan.evaluation.evidence)
    ]
    for rejection in plan.rejected:
        places.append(
            (f"the rejection of {rejection.method}", None, rejection.method,
             rejection.evidence)
        )
    for candidate in plan.candidates:
        places.append((f"candidate {candidate.id}", candidate.id, None, candidate.evidence))
    if plan.base_departure is not None:
        places.append(("the base departure", None, None, plan.base_departure.evidence))
    if plan.tuning.departure is not None:
        places.append(("the tuning departure", None, None, plan.tuning.departure.evidence))
    if plan.evaluation.margin_departure is not None:
        places.append(
            ("the margin departure", None, None, plan.evaluation.margin_departure.evidence)
        )
    for where, candidate_id, stages in [
        ("base preprocessing", None, plan.base_preprocessing),
        *[(f"candidate {c.id}", c.id, c.stages) for c in plan.candidates],
    ]:
        for stage in stages:
            for param, override in stage.overrides.items():
                places.append(
                    (f"the override of {stage.op}.{param} in {where}", candidate_id,
                     stage.op, override.evidence)
                )

    findings: list[Finding] = []
    for where, candidate_id, op, keys in places:
        own = [key for key in keys if key.partition(".")[0] == "plan"]
        resolved = resolve_evidence([k for k in keys if k not in own], citable)
        broken = [key for key, value in resolved.items() if value is MISSING]
        if not own and not broken:
            continue
        parts = []
        if own:
            parts.append(
                f"{where} cites {', '.join(own)}. A plan cannot cite itself: a key "
                "into the plan resolves because the plan says so, and proves nothing."
            )
        if broken:
            parts.append(f"In {where}, " + unresolved_message(broken, citable))
        findings.append(
            Finding(
                code="unresolved_evidence",
                severity="error",
                candidate=candidate_id,
                op=op,
                message=" ".join(parts),
                fix="cite keys in the profile, recon, suggestions, metrics or ranking "
                "that exist, or drop them",
            )
        )
    return findings


def _same_value(param: Any, given: Any, suggested: Any) -> bool:
    try:
        suggested = param.coerce(suggested)
    except (TypeError, ValueError):
        return False
    if isinstance(given, float) or isinstance(suggested, float):
        return given is not None and suggested is not None and abs(given - suggested) <= 1e-9
    return given == suggested


def _check_provenance(
    plan: Plan,
    profile: dict[str, Any],
    recon: dict[str, Any] | None,
    registry: Registry,
    suggestions: dict[str, Any],
    frozen: dict[str, list[dict[str, Any]]],
) -> tuple[list[Finding], dict[str, list[dict[str, Any]]]]:
    """Where every parameter of every stage came from, against the persisted suggestions.

    Four states: `registry_default` (not given, and no different suggestion),
    `suggested` (given, equal to the persisted suggestion), `overridden` (differing from
    it, whether given or left at a default that differs -- refused without a reason in
    the stage's `overrides`), and `specified` (given, nothing suggested for it). `frozen` holds the provenance an earlier registration recorded for candidates
    whose stages have not changed since; those keep it, so a suggestion requested after
    registration can neither relabel them nor refuse them.
    """
    from drtools.heuristics import suggest

    findings: list[Finding] = []
    records: dict[str, list[dict[str, Any]]] = {}
    reported: set[tuple[str | None, int, str]] = set()
    warned: set[str] = set()
    n_base = len(plan.base_preprocessing)

    for candidate in plan.candidates:
        if candidate.id in frozen:
            records[candidate.id] = frozen[candidate.id]
            continue
        stages_out: list[dict[str, Any]] = []
        all_stages = [*plan.base_preprocessing, *candidate.stages]
        for position, stage in enumerate(all_stages):
            in_base = position < n_base
            terminal = position == len(all_stages) - 1
            owner = None if in_base else candidate.id
            if stage.op not in registry:
                continue  # already reported by the structural check
            spec = registry[stage.op]
            try:
                resolved, _ = registry.resolve_params(stage.op, stage.params)
            except Exception:
                continue  # already reported by the structural check

            persisted = {
                name: entry
                for name, entry in (
                    (suggestions.get(stage.op) or {}).get("suggested") or {}
                ).items()
                if not (terminal and entry.get("applies_to") == "intermediate")
            }
            params: dict[str, dict[str, Any]] = {}
            chosen_by_tuning = (
                set()
                if in_base or not (spec.is_reduction or spec.is_visualization)
                else tuned_by_rule(stage.op, spec, resolved)
            )
            tuned_param = None if in_base else spec.tuned_param(resolved)
            for name, param in spec.params.items():
                given = name in stage.params
                if name in chosen_by_tuning:
                    # Chosen inside the Attempt; registration refuses a value given.
                    params[name] = {"state": "tuned"}
                    continue
                if name == tuned_param and not given:
                    # Unset, the multiplier grid is centred on the suggestion or the
                    # rule, so leaving it out follows the data rather than a default.
                    params[name] = {"state": "tuned", "centre": spec.tuning.base}
                    continue
                if name not in persisted:
                    state = "specified" if given else "registry_default"
                    params[name] = {"state": state}
                    continue
                suggested = persisted[name].get("value")
                if _same_value(param, resolved[name], suggested):
                    # Left unset, a value equal to the suggestion follows it by default.
                    state = "suggested" if given else "registry_default"
                    if name == tuned_param:
                        params[name] = {"state": "tuned", "centre": "suggestion"}
                    else:
                        params[name] = {"state": state}
                    continue
                # Different from the suggestion, whether given or left at a default that
                # differs: leaving a parameter out is no way around giving the reason.
                entry: dict[str, Any] = {"state": "overridden", "suggested": suggested}
                override = stage.overrides.get(name)
                if override is not None and override.reason.strip():
                    entry["reason"] = override.reason
                elif (owner, position, name) not in reported:
                    reported.add((owner, position, name))
                    how = (
                        f"is {resolved[name]!r}"
                        if given
                        else f"is left unset, so it runs at the registry default of "
                        f"{resolved[name]!r}"
                    )
                    findings.append(
                        Finding(
                            code="unexplained_override",
                            severity="error",
                            candidate=owner,
                            op=stage.op,
                            message=f"{stage.op}.{name} {how}, and the suggestion "
                            f"persisted for this run is {suggested!r}. An override is "
                            "allowed, and needs its reason on the record, so the report "
                            "can say why the value was chosen rather than inherited.",
                            fix=f"give the stage overrides: {{{name!r}: {{\"reason\": "
                            "..., \"evidence\": [...]}}, or set the suggested value",
                        )
                    )
                if name == tuned_param:
                    entry["centre"] = "override"
                params[name] = entry
            stages_out.append({"op": stage.op, "params": params})

            if (
                stage.op not in suggestions
                and stage.op not in warned
                and suggest(stage.op, profile, recon, registry)
            ):
                warned.add(stage.op)
                findings.append(
                    Finding(
                        code="no_suggestion_requested",
                        severity="warning",
                        candidate=owner,
                        op=stage.op,
                        message=f"{stage.op} has profile-derived suggestions, and none "
                        "was requested for this run, so every value it is given is "
                        "recorded as specified and nothing shows whether it followed "
                        "the data or overrode it.",
                        fix=f"run `drtools suggest-params --op {stage.op}` and "
                        "re-register",
                    )
                )
        records[candidate.id] = stages_out

    return findings, records


def _initial_state(
    profile: dict[str, Any], decision: DataDecision | None = None
) -> PlanState:
    shape = profile.get("shape", {})
    values = profile.get("values", {})
    is_sparse = shape.get("storage") == "sparse_csr"
    raw_counts = (
        decision.values == "raw_counts"
        if decision is not None
        else values.get("suspected_kind") == "counts"
    )
    return PlanState(
        n_samples=int(shape.get("n_samples", 0)),
        n_features=int(shape.get("n_features", 0)),
        is_sparse=is_sparse,
        is_raw_counts=raw_counts,
        itemsize=int(np.dtype(shape.get("dtype") or "float64").itemsize),
        density=1.0 - float(values.get("sparsity") or 0.0) if is_sparse else 1.0,
    )


def _stage_peaks(
    stages: list[dict[str, Any]],
    profile: dict[str, Any],
    registry: Registry,
    decision: DataDecision | None = None,
) -> tuple[float, list[tuple[str, float]]]:
    """The loaded data's bytes, and each stage's input, output and copy held together.

    After a subsample, the matrix that entered it is held too: the rows it did not keep
    are projected from it once the later stages are fitted (section 3.12).
    """
    state = _initial_state(profile, decision)
    loaded = state.nbytes
    retained = 0.0
    peaks: list[tuple[str, float]] = []
    for position, stage in enumerate(stages):
        op = stage["op"]
        if op not in registry:
            continue
        try:
            params, _ = registry.resolve_params(op, stage.get("params", {}))
        except Exception:
            continue
        before, sparse_before = state.nbytes, state.is_sparse
        state.advance(op, params, registry)
        after = state.nbytes
        # PCA centres a dense input in a copy of it; the first stage's input is the
        # loaded data, which is already counted.
        copy = before if op == "pca" and not sparse_before else 0.0
        peaks.append((op, retained + (before if position else 0.0) + after + copy))
        if op == "subsample" and position:
            retained = before
    return loaded, peaks


def estimate_peak_bytes(
    stages: list[dict[str, Any]],
    profile: dict[str, Any],
    registry: Registry | None = None,
    decision: DataDecision | None = None,
) -> float:
    """The memory a candidate's pipeline needs at its heaviest stage.

    The loaded data stays in memory throughout; on top of it, each stage holds its
    input and its output at once, PCA on dense input a centred copy, and every stage
    after a subsample the matrix the subsample was taken from. The largest of those is
    the peak. A method's own working memory -- an n-by-n matrix -- is not
    counted: `scales_to` governs that.
    """
    loaded, peaks = _stage_peaks(stages, profile, registry or load_registry(), decision)
    return loaded + max((peak for _, peak in peaks), default=0.0)


def _without_subsample(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [stage for stage in stages if stage["op"] != "subsample"]


def _first_method(stages: list[dict[str, Any]], registry: Registry) -> Any:
    return next(
        (
            registry[s["op"]]
            for s in stages
            if s["op"] in registry
            and (registry[s["op"]].is_reduction or registry[s["op"]].is_visualization)
        ),
        None,
    )


def _gb(value: float) -> str:
    return f"{value / 1e9:.2f} GB"


def _check_candidate(
    plan: Plan,
    candidate: CandidateSpec,
    profile: dict[str, Any],
    recon: dict[str, Any] | None,
    registry: Registry,
    decision: DataDecision | None = None,
    memory_limit: int | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    state = _initial_state(profile, decision)

    stages = plan.stages_for(candidate)
    for position, stage in enumerate(stages):
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
        if op == "subsample":
            findings += _check_subsample_is_needed(
                candidate.id, stages, position, state, registry, profile, decision,
                memory_limit,
            )
        findings += _check_new_rows(
            candidate.id, op, spec, profile, stages, registry, decision, memory_limit
        )
        state.advance(op, params, registry)

    findings += _check_memory(
        candidate, stages, profile, registry, decision, memory_limit
    )
    return findings


def _check_memory(
    candidate: CandidateSpec,
    stages: list[dict[str, Any]],
    profile: dict[str, Any],
    registry: Registry,
    decision: DataDecision | None,
    memory_limit: int | None,
) -> list[Finding]:
    """Refuse a candidate whose estimated peak exceeds the run's memory limit.

    Memory is a reason to subsample, as `scales_to` is: a method that can place new rows
    is fitted on a subsample and the rest projected in chunks. A method that cannot is
    refused by `_check_new_rows` instead, whose fix is to reject it.
    """
    if memory_limit is None:
        return []
    loaded, peaks = _stage_peaks(stages, profile, registry, decision)
    if not peaks:
        return []
    op, heaviest = max(peaks, key=lambda item: item[1])
    total = loaded + heaviest
    method = _first_method(stages, registry)
    if total <= memory_limit or (method is not None and method.raw.get("new_rows") == "none"):
        return []
    if is_linear_baseline(candidate):
        fix = (
            "the Linear baseline cannot subsample, since it is the base followed by a "
            "single pca, so this plan cannot register within this machine's memory: "
            "run on a machine with more memory, or on a smaller dataset"
        )
    else:
        fix = (
            f"insert a subsample before {method.name if method else 'the first method'}"
            ": it is fitted on the rows kept, and the rest are projected through the "
            "fitted pipeline in chunks"
        )
    return [
        Finding(
            code="exceeds_memory_limit",
            severity="error",
            candidate=candidate.id,
            op=op,
            message=f"candidate {candidate.id} is estimated to need {_gb(total)} at its "
            f"heaviest stage, {op}: the loaded data ({_gb(loaded)}) plus that stage's "
            f"input, output and any copy ({_gb(heaviest)}). This run's limit is "
            f"{_gb(memory_limit)}, half the machine's memory.",
            fix=fix,
        )
    ]


def _check_subsample_is_needed(
    candidate_id: str,
    stages: list[dict[str, Any]],
    position: int,
    state: PlanState,
    registry: Registry,
    profile: dict[str, Any],
    decision: DataDecision | None,
    memory_limit: int | None,
) -> list[Finding]:
    """Section 3.12: subsample only when the method would otherwise exceed a limit.

    Either limit: the method's `scales_to` on rows, or the run's memory limit on the
    whole pipeline. A subsample neither calls for makes the candidate fit fewer rows for
    no gain, and `subsample(50) -> pca` is one route to a degenerate baseline.
    """
    method = _first_method(stages[position + 1:], registry)
    if method is None:
        return []
    limit = method.scales_to
    if limit is not None and state.n_samples > limit:
        return []
    without = stages[:position] + stages[position + 1:]
    if memory_limit is not None and estimate_peak_bytes(
        without, profile, registry, decision
    ) > memory_limit:
        return []
    within = (
        f"within {method.name}'s limit of {limit:,}"
        if limit is not None
        else f"and {method.name} declares no limit on rows"
    )
    memory = (
        f", and without it the pipeline fits in this run's {_gb(memory_limit)}"
        if memory_limit is not None
        else ""
    )
    return [
        Finding(
            code="subsample_not_needed",
            severity="error",
            candidate=candidate_id,
            op="subsample",
            message=f"this subsample feeds {method.name} with fewer rows than the "
            f"{state.n_samples:,} it would otherwise receive, {within}{memory}. A "
            "candidate may subsample only when its method would otherwise exceed its "
            "row limit or the memory limit: otherwise it fits fewer rows for no gain.",
            fix=f"remove the subsample stage and let {method.name} fit every row",
        )
    ]


def _check_new_rows(
    candidate_id: str,
    op: str,
    spec: Any,
    profile: dict[str, Any],
    stages: list[dict[str, Any]],
    registry: Registry,
    decision: DataDecision | None,
    memory_limit: int | None,
) -> list[Finding]:
    """Section 3.12: a method that cannot place new rows cannot be subsampled at all.

    A subsampled candidate is fitted on some rows and has the rest projected through
    its fitted pipeline. A method with no `transform` and no standard extension has no
    way to project, so when every row is too many -- for its `scales_to`, or for the
    memory limit -- there is no honest version of it to run.
    """
    if spec.raw.get("new_rows") != "none":
        return []
    n = int(profile.get("shape", {}).get("n_samples", 0))
    if spec.scales_to is not None and n > spec.scales_to:
        reason = f"the dataset has {n:,} samples, above {op}'s limit of {spec.scales_to:,}"
    elif memory_limit is not None and (
        needed := estimate_peak_bytes(_without_subsample(stages), profile, registry, decision)
    ) > memory_limit:
        reason = (
            f"on every row this pipeline is estimated to need {_gb(needed)}, above this "
            f"run's memory limit of {_gb(memory_limit)}"
        )
    else:
        return []
    return [
        Finding(
            code="cannot_place_new_rows",
            severity="error",
            candidate=candidate_id,
            op=op,
            message=f"{reason}, and {op} has no transform and no standard extension to "
            "place rows it was not fitted on. A subsample would leave it describing only "
            "the rows it kept, and every other candidate covers all of them.",
            fix=f"drop this candidate and add {op} to `rejected`, citing "
            "profile.shape.n_samples",
        )
    ]


#: Section 3.10's number, fixed so that selection cannot become a route to a
#: degenerate candidate.
SELECTED_FEATURES = 2000


def _check_selection(
    plan: Plan,
    candidate: CandidateSpec,
    profile: dict[str, Any],
    registry: Registry,
    decision: DataDecision | None,
) -> list[Finding]:
    """Section 3.10's candidate-specific stages, which are enforced, not advised.

    `select_variable_features(n_features=2000)` then `standardise`, before the first
    method, exactly when the features are of one type, more than 2,000 remain after the
    constants go, the first method works through Euclidean geometry at the parameters
    set, and the candidate is not the Linear baseline. Otherwise neither.
    """
    if decision is None:
        return []
    stages = [stage.model_dump(exclude={"overrides"}) for stage in candidate.stages]
    first = next(
        (
            position
            for position, stage in enumerate(stages)
            if stage["op"] in registry
            and (registry[stage["op"]].is_reduction or registry[stage["op"]].is_visualization)
        ),
        None,
    )
    if first is None:
        return []
    method = stages[first]
    governed = [
        stage for stage in stages[:first]
        if stage["op"] in {"select_variable_features", "standardise"}
    ]
    try:
        params, _ = registry.resolve_params(method["op"], method["params"])
    except Exception:
        return []

    features = profile.get("features", {})
    remaining = int(profile.get("shape", {}).get("n_features", 0)) - int(
        features.get("n_constant") or 0
    )
    reasons = []
    if is_linear_baseline(candidate):
        reasons.append("it is the Linear baseline, which never selects")
    if decision.features != "one_type":
        reasons.append("the features are of mixed types, and z-scored in the base instead")
    if remaining <= SELECTED_FEATURES:
        reasons.append(f"only {remaining:,} features remain, not more than 2,000")
    if not registry[method["op"]].holds("euclidean", params):
        reasons.append(
            f"{method['op']} at these parameters does not work through Euclidean "
            "geometry, so variance does not rank features by their effect on it"
        )

    wanted = [
        {"op": "select_variable_features", "params": {"n_features": SELECTED_FEATURES}},
        {"op": "standardise", "params": {}},
    ]
    if reasons:
        if not governed:
            return []
        return [
            Finding(
                code="selection_forbidden",
                severity="error",
                candidate=candidate.id,
                op=governed[0]["op"],
                message=f"candidate {candidate.id} carries "
                f"{' and '.join(s['op'] for s in governed)} before {method['op']}, and "
                "section 3.10's rule forbids it here: " + "; ".join(reasons) + ".",
                fix="remove select_variable_features and standardise from this candidate",
            )
        ]

    counts = [
        int(stage["params"].get("n_features", SELECTED_FEATURES))
        for stage in governed
        if stage["op"] == "select_variable_features"
    ]
    if any(count != SELECTED_FEATURES for count in counts):
        return [
            Finding(
                code="selection_count",
                severity="error",
                candidate=candidate.id,
                op="select_variable_features",
                message=f"candidate {candidate.id} selects {counts[0]:,} features. The "
                "number is fixed at 2,000 wherever the rule selects, so that selection "
                "cannot become a route to a degenerate candidate.",
                fix="select exactly 2000 features",
            )
        ]
    if normalise_stages(governed) != wanted:
        return [
            Finding(
                code="selection_required",
                severity="error",
                candidate=candidate.id,
                op=method["op"],
                message=f"candidate {candidate.id}'s first method, {method['op']}, works "
                f"through Euclidean geometry, the features are of one type, and "
                f"{remaining:,} of them remain: section 3.10's rule selects the 2,000 "
                "most variable and z-scores them before such a method, and this candidate "
                "does not do exactly that.",
                fix=f"put select_variable_features(n_features=2000), then standardise, "
                f"directly before {method['op']}",
            )
        ]
    return []


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

    # A method that cannot place new rows is refused above its limit by
    # `_check_new_rows`, whose fix is to reject it; offering a subsample here too would
    # give the agent two contradictory instructions for one problem.
    if (
        (spec.is_reduction or spec.is_visualization)
        and spec.scales_to is not None
        and n > spec.scales_to
        and spec.raw.get("new_rows") != "none"
    ):
        findings.append(
            Finding(
                code="exceeds_scale_limit",
                severity="error",
                candidate=candidate_id,
                op=op,
                message=f"{op} receives {n:,} samples but is documented as practical to "
                f"about {spec.scales_to:,} ({spec.raw.get('complexity', 'see registry')}).",
                fix=f"insert a subsample stage before {op}: {op} is fitted on the rows "
                "kept and every other row is projected through the fitted pipeline; or "
                "choose a method that scales",
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

    # Raw counts mislead whatever distance or kernel a method uses: sample totals vary,
    # and a count's variance grows with its mean. So the check covers every reduction
    # and visualization method, and does not read `euclidean`, which answers a different
    # question (section 3.10's selection rule).
    if state.is_raw_counts and (spec.is_reduction or spec.is_visualization):
        findings.append(
            Finding(
                code="raw_counts_not_normalised",
                severity="error",
                candidate=candidate_id,
                op=op,
                message=f"the values reaching {op} are still raw counts. Sample totals "
                "vary from sample to sample and a count's variance grows with its "
                "mean, so whatever distance or kernel the method uses, the leading "
                "structure it recovers would be sequencing depth and the most highly "
                "expressed features rather than the biology.",
                fix="add normalise_total and log1p before this method",
            )
        )

    # The executor calls the same rule, so what refuses at execution refuses here first,
    # before a mistake visible in the plan spends one of the candidate's two attempts.
    for rule in spec.d_limits:
        violation = RULES[rule].violation(params)
        if violation is not None:
            findings.append(
                Finding(
                    code="d_limit_violated",
                    severity="error",
                    candidate=candidate_id,
                    op=op,
                    message=violation,
                    fix=RULES[rule].sentence,
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
        and spec.holds("requires_connected_graph", params)
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
