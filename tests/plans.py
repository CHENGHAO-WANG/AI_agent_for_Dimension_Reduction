"""Completing a test's plan with what registration requires and the test is not about.

Since day 12 a plan must account for every reduction and visualization method, and
cite evidence for its candidates, its rejections and any weighting that departs from
the default. Most tests exercise something else and would otherwise each restate
those parts. `complete` fills in only what is absent, so a test about one of these
rules builds its plan without it, or sets the part it is testing explicitly.
"""

from __future__ import annotations

import copy
from typing import Any

from drtools.registry import load_registry

CITED = ["profile.shape.n_samples"]
FILLER = "not the subject of this test"


def complete(document: dict[str, Any]) -> dict[str, Any]:
    plan = copy.deepcopy(document)
    registry = load_registry()

    for candidate in plan.get("candidates", []):
        candidate.setdefault("evidence", list(CITED))

    evaluation = plan.setdefault("evaluation", {})
    evaluation.setdefault("evidence", list(CITED))

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
