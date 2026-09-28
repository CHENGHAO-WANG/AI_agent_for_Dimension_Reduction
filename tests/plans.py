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


def complete(document: dict[str, Any], *, base: bool = True) -> dict[str, Any]:
    plan = copy.deepcopy(document)
    registry = load_registry()

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
    methods = {**registry.reductions(), **registry.visualization_methods()}
    for method in sorted(set(methods) - named):
        plan["rejected"].append(
            {"method": method, "reason": FILLER, "evidence": list(CITED)}
        )
    return plan


def decision(**fields: Any) -> str:
    """A `--decision` argument: DECISION with `fields` changed."""
    return json.dumps({**DECISION, **fields})


def reconnoitre(cli: Any, run: Any, **fields: Any) -> Any:
    """Run reconnaissance under a declared decision, as registration now requires."""
    result = cli("recon", "--run-dir", run, "--decision", decision(**fields))
    assert result.code == 0, result.stderr
    return result


#: What a test calling `validate_plan` directly passes as reconnaissance.
RECON = {"data_decision": {**DECISION, "evidence": []}}
