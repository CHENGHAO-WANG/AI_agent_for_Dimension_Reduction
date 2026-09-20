"""The ceiling on how many candidates one run may register.

Attempts are capped per candidate id, and the candidate id set only ever grows —
`_check_reregistration` refuses to let a registered id disappear. So the id set is the
one quantity the agent can spend that nothing declared, and it is exactly the quantity
it mints to reset the per-candidate allowance: exhaust an id's two attempts, abandon
it, register a replacement, get two more. Capping the set closes that loop with two
refusals that already exist plus one comparison, and bounds a whole run at
`MAX_ATTEMPTS x ceiling` executions.

The ceiling shrinks as the per-candidate time cap grows, which states the trade
honestly: a larger time allowance per candidate buys fewer of them.
"""

from __future__ import annotations

import pytest

from drtools.isolation import BUDGET_MAX_CANDIDATES
from drtools.plan import validate_plan

PROFILE = {
    "shape": {"n_samples": 300, "n_features": 10, "storage": "dense"},
    "values": {"suspected_kind": "continuous"},
}


def _plan(n_candidates: int, **overrides):
    """`n_candidates` PCA candidates, which satisfies the linear-baseline rule."""
    return {
        "dataset": "d",
        "candidates": [
            {"id": f"c{i}", "stages": [{"op": "pca", "params": {"n_components": 2}}]}
            for i in range(n_candidates)
        ],
        "evaluation": {"weights": {"trustworthiness": 1.0}, "justification": "up front"},
        **overrides,
    }


def _codes(report) -> set[str]:
    return {finding["code"] for finding in report["findings"]}


def test_a_plan_beyond_the_budget_ceiling_is_refused() -> None:
    ceiling = BUDGET_MAX_CANDIDATES["standard"]

    report = validate_plan(_plan(ceiling + 1, budget="standard"), PROFILE)

    assert report["valid"] is False
    assert "candidates_exceed_ceiling" in _codes(report)


def test_a_plan_at_the_budget_ceiling_is_accepted() -> None:
    """The refusal must bite one past the ceiling, not at it.

    Day 7 found that five of nine fixes reintroduced the class of defect they closed.
    An off-by-one here would refuse a legitimate portfolio rather than bound a loop.
    """
    ceiling = BUDGET_MAX_CANDIDATES["standard"]

    report = validate_plan(_plan(ceiling, budget="standard"), PROFILE)

    assert "candidates_exceed_ceiling" not in _codes(report)


@pytest.mark.parametrize("budget", sorted(BUDGET_MAX_CANDIDATES))
def test_every_budget_declares_a_ceiling(budget: str) -> None:
    report = validate_plan(
        _plan(BUDGET_MAX_CANDIDATES[budget] + 1, budget=budget), PROFILE
    )

    assert "candidates_exceed_ceiling" in _codes(report)


def test_the_ceiling_shrinks_as_the_per_candidate_time_cap_grows() -> None:
    """A thorough run buys longer candidates, so it may register fewer of them."""
    assert (
        BUDGET_MAX_CANDIDATES["thorough"]
        < BUDGET_MAX_CANDIDATES["standard"]
        < BUDGET_MAX_CANDIDATES["fast"]
    )

    count = BUDGET_MAX_CANDIDATES["standard"]
    assert "candidates_exceed_ceiling" not in _codes(
        validate_plan(_plan(count, budget="standard"), PROFILE)
    )
    assert "candidates_exceed_ceiling" in _codes(
        validate_plan(_plan(count, budget="thorough"), PROFILE)
    )


def test_the_ceiling_leaves_room_above_the_portfolio_guidance() -> None:
    """Section 3.4 asks for 3-5 candidates; the gap above 5 is the repair headroom.

    A ceiling of 5 would bound the loop by forbidding the sanctioned repair, which
    destroys the retry the design calls its best evidence of agency.
    """
    assert min(BUDGET_MAX_CANDIDATES.values()) > 5


def test_a_declared_ceiling_above_the_budget_is_refused() -> None:
    """A bound the agent sets on itself only means something under a hard one."""
    report = validate_plan(
        _plan(3, budget="standard", max_candidates=BUDGET_MAX_CANDIDATES["standard"] + 5),
        PROFILE,
    )

    assert report["valid"] is False
    assert "declared_ceiling_above_budget" in _codes(report)


def test_a_declared_ceiling_binds_tighter_than_the_budget() -> None:
    """Pre-registering a tighter bound is the point of declaring one at all."""
    report = validate_plan(_plan(4, budget="standard", max_candidates=3), PROFILE)

    assert report["valid"] is False
    assert "candidates_exceed_ceiling" in _codes(report)


def test_a_plan_within_its_declared_ceiling_is_accepted() -> None:
    report = validate_plan(_plan(3, budget="standard", max_candidates=3), PROFILE)

    assert "candidates_exceed_ceiling" not in _codes(report)
    assert "declared_ceiling_above_budget" not in _codes(report)
