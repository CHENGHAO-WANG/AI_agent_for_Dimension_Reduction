"""The two facts preprocessing rules take as input, the base those rules give, and the
checkpoint's purpose and focus.

Section 3.10 fixes both layers of preprocessing by rule, and the rule reads two facts
the matrix cannot settle alone: whether the values are raw counts, and whether the
features are all of one type. They are decided -- by the user, by the agent citing
evidence, or by the default -- and passed to `recon`, which probes under them and
records them. Everything that follows reads that record, so the decision changes only
by running reconnaissance again, and the probes can never describe a different
decision from the one the analysis uses.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DataDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: Literal["raw_counts", "not_counts"]
    features: Literal["one_type", "mixed"]
    # `user`: declared, needing no evidence. `agent`: the agent's call, citing evidence.
    # `default`: nothing declared and the evidence unclear, or a run under --auto.
    decided_by: Literal["user", "agent", "default"]
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)


def default_decision(profile: dict[str, Any]) -> DataDecision:
    """Raw counts when the profile suspects them, and mixed features.

    Mixed is the default because the two errors are not equally bad: z-scoring features
    of one type inflates noise features, while leaving mixed features unscaled lets the
    one in the largest units dominate every distance and every component.
    """
    counts = profile.get("values", {}).get("suspected_kind") == "counts"
    return DataDecision(
        values="raw_counts" if counts else "not_counts",
        features="mixed",
        decided_by="default",
        rationale=(
            "the default: raw counts because the profile suspects them, "
            if counts
            else "the default: not raw counts because the profile does not suspect them, "
        )
        + "and features of mixed types, the safer error when the evidence is unclear",
        evidence=["profile.values.suspected_kind"],
    )


def base_rule(decision: DataDecision) -> list[dict[str, Any]]:
    """The base preprocessing section 3.10's rule gives for this decision."""
    stages = [{"op": "drop_constant", "params": {}}]
    if decision.values == "raw_counts":
        stages += [{"op": "normalise_total", "params": {}}, {"op": "log1p", "params": {}}]
    if decision.features == "mixed":
        stages.append({"op": "standardise", "params": {}})
    return stages


def rule_reason(decision: DataDecision) -> str:
    """The rule's reasoning for this decision, in one sentence per fact."""
    parts = ["constant features are dropped first, so no later step sees one"]
    if decision.values == "raw_counts":
        parts.append(
            "the values are raw counts, so each sample is rescaled to the median total "
            "and log-transformed: sample totals vary, and a count's variance grows with "
            "its mean"
        )
    if decision.features == "mixed":
        parts.append(
            "the features are of mixed types, so every feature is z-scored, or the one "
            "in the largest units would dominate every distance"
        )
    else:
        parts.append("the features are of one type, so they keep their native scale")
    return "; ".join(parts) + "."


def recorded_decision(recon: dict[str, Any] | None) -> DataDecision | None:
    """The decision reconnaissance ran under, or None when none is recorded."""
    if not recon or "data_decision" not in recon:
        return None
    return DataDecision.model_validate(recon["data_decision"])


class Checkpoint(BaseModel):
    """The purpose and the focus, answered at the checkpoint (section 3.11, day 17).

    Recorded once by `drtools checkpoint`, after reconnaissance, and read by
    registration, so the plan never restates them. The purpose decides which methods are
    eligible and whether candidates are ranked; the focus decides which default
    weighting the plan is held to. Each answer says who gave it, since the user may
    answer one question and leave the other to the agent.
    """

    model_config = ConfigDict(extra="forbid")

    purpose: Literal["representation", "visualization"] = "representation"
    purpose_decided_by: Literal["user", "agent", "default"] = "default"
    # `local`: neighbourhoods -- clustering downstream, or a picture of clusters.
    # `global`: distances -- regression or a map downstream, or the overall layout.
    focus: Literal["local", "global", "balanced"] = "balanced"
    focus_decided_by: Literal["user", "agent", "default"] = "default"
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)


def recorded_checkpoint(checkpoint: dict[str, Any] | None) -> Checkpoint | None:
    """The checkpoint a run recorded, or None when it recorded none."""
    return None if checkpoint is None else Checkpoint.model_validate(checkpoint)
