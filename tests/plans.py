"""Completing a test's plan and run with what registration requires and the test is not about.

Since day 12 a plan must account for every reduction and visualization method, and
cite evidence for its candidates, its rejections and any weighting that departs from
the default. Since day 13 it registers only against a recorded data decision, and its
base preprocessing must follow the rule for that decision or say why not. Most tests
exercise something else and would otherwise each restate those parts. `complete` fills
in only what is absent, so a test about one of these rules builds its plan without it,
or sets the part it is testing explicitly; `reconnoitre` records the decision.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from drtools.registry import load_registry

CITED = ["profile.shape.n_samples"]
FILLER = "not the subject of this test"

#: The decision `reconnoitre` records unless a test says otherwise: not raw counts,
#: features of one type. Its base rule is drop_constant alone.
DECISION = {
    "values": "not_counts",
    "features": "one_type",
    "decided_by": "user",
    "rationale": "declared for the test",
}
RULE_BASE = [{"op": "drop_constant", "params": {}}]


def purpose_of(document: dict[str, Any]) -> str:
    """The purpose a test's plan implies: visualization if it runs a visualization method.

    Since day 17 a representation run may not run one, so a test that nominates t-SNE
    or UMAP to exercise some other rule is a visualization run's plan.
    """
    registry = load_registry()
    visual = registry.visualization_methods()
    runs_one = any(
        stage["op"] in visual
        for candidate in document.get("candidates", [])
        for stage in candidate.get("stages", [])
    )
    return "visualization" if runs_one else "representation"


def complete(
    document: dict[str, Any], *, base: bool = True, purpose: str | None = None
) -> dict[str, Any]:
    plan = copy.deepcopy(document)
    registry = load_registry()
    purpose = purpose or purpose_of(plan)

    for candidate in plan.get("candidates", []):
        candidate.setdefault("evidence", list(CITED))

    evaluation = plan.setdefault("evaluation", {})
    evaluation.setdefault("evidence", list(CITED))

    if base:
        # drop_constant first, as every base must; a base other than DECISION's rule
        # is given a departure, since the base is not what the test is about.
        stages = plan.setdefault("base_preprocessing", [])
        if not stages or stages[0].get("op") != "drop_constant":
            stages.insert(0, {"op": "drop_constant", "params": {}})
        if stages != RULE_BASE:
            plan.setdefault("base_departure", {"reason": FILLER, "evidence": list(CITED)})

    nominated = {
        stage["op"]
        for candidate in plan.get("candidates", [])
        for stage in candidate.get("stages", [])
    }
    # A rejection this helper added gives way once a test nominates the method, so a
    # test that adds a candidate to a completed plan and completes it again still reads
    # as the plan it means.
    plan["rejected"] = [
        rejection
        for rejection in plan.get("rejected", [])
        if not (rejection.get("reason") == FILLER and rejection["method"] in nominated)
    ]
    named = nominated | {rejection["method"] for rejection in plan["rejected"]}
    # Since day 17 a representation run accounts for the reductions alone.
    methods = dict(registry.reductions())
    if purpose == "visualization":
        methods.update(registry.visualization_methods())
    plan["rejected"] = [
        rejection
        for rejection in plan["rejected"]
        if not (rejection.get("reason") == FILLER and rejection["method"] not in methods)
    ]
    for method in sorted(set(methods) - named):
        plan["rejected"].append(
            {"method": method, "reason": FILLER, "evidence": list(CITED)}
        )
    return plan


def decision(**fields: Any) -> str:
    """A `--decision` argument: DECISION with `fields` changed."""
    return json.dumps({**DECISION, **fields})


def checkpoint(**fields: Any) -> dict[str, Any]:
    """A checkpoint with the defaults, `fields` changed."""
    return {**CHECKPOINT, **fields}


def reconnoitre(
    cli: Any, run: Any, *, purpose: str = "representation", focus: str = "balanced",
    **fields: Any,
) -> Any:
    """Run reconnaissance under a declared decision, then record the checkpoint.

    Registration requires both since day 17. The checkpoint defaults to a
    representation run with a balanced focus; a test about a visualization run says so.
    """
    result = cli("recon", "--run-dir", run, "--decision", decision(**fields))
    assert result.code == 0, result.stderr
    answered = cli("checkpoint", "--run-dir", run, "--answers",
                   json.dumps(checkpoint(purpose=purpose, focus=focus)))
    assert answered.code == 0, answered.stderr
    return result


#: What a test calling `validate_plan` directly passes as reconnaissance.
RECON = {"data_decision": {**DECISION, "evidence": []}}

#: The checkpoint's defaults: a representation run with a balanced focus.
CHECKPOINT = {
    "purpose": "representation",
    "purpose_decided_by": "default",
    "focus": "balanced",
    "focus_decided_by": "default",
}


def checkpoint_for(document: dict[str, Any]) -> dict[str, Any]:
    """The checkpoint a directly validated test plan is registered under."""
    return checkpoint(purpose=purpose_of(document))
