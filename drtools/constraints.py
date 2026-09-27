"""Limits on the output dimension that a capability record cannot hold as a number.

A record can say `n_components: {min: 1}`. It cannot say "n_neighbors must reach a
minimum that grows with n_components": that limit is arithmetic in d and in another
parameter, and an arithmetic language written into the YAML would be a second, untested
copy of a rule that has to live in code anyway. So each such limit is a named rule here.
The registry names the rules an op obeys, `drtools methods` prints each rule's sentence
where the agent plans, and the executor and the plan validator call the same function,
so the rule that refuses at execution is the rule that refuses at registration.

This module imports nothing from drtools, so the registry can import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Rule:
    """A limit on d, with the sentence the agent reads and the check that enforces it."""

    name: str
    sentence: str
    violation: Callable[[dict[str, Any]], str | None]


def lle_neighbour_minimum(method: str, n_components: int) -> int:
    """The fewest neighbours an LLE variant needs to produce `n_components` coordinates.

    Hessian's minimum is scikit-learn's own: it refuses n_neighbors <= d(d + 3)/2. For
    the standard, modified and LTSA variants the library accepts one fewer than this
    returns, and the minimum is mathematical instead: d + 1 points are the fewest whose
    affine span is d-dimensional, so a smaller neighbourhood cannot describe a
    d-dimensional local patch.
    """
    if method == "hessian":
        return 1 + n_components * (n_components + 3) // 2
    return n_components + 1


def _lle_violation(params: dict[str, Any]) -> str | None:
    neighbours = params.get("n_neighbors")
    if neighbours is None:
        return None
    method = str(params.get("method") or "standard")
    d = int(params.get("n_components") or 2)
    minimum = lle_neighbour_minimum(method, d)
    if int(neighbours) >= minimum:
        return None
    return (
        f"lle(method={method!r}) needs at least {minimum} neighbours to produce {d} "
        f"components, and this stage gives it {int(neighbours)}. Raise n_neighbors to "
        f"{minimum} or more, or lower n_components."
    )


RULES: dict[str, Rule] = {
    "lle_neighbour_minimum": Rule(
        name="lle_neighbour_minimum",
        sentence=(
            "n_neighbors must reach a minimum that grows with n_components (d): d + 1 "
            "for the standard, modified and LTSA variants, the fewest points whose "
            "affine span is d-dimensional; and 1 + d(d + 3)/2 for Hessian, which is "
            "scikit-learn's own requirement -- 6 at d = 2, 45 at d = 8. Raising d "
            "therefore invalidates an n_neighbors chosen for a smaller d."
        ),
        violation=_lle_violation,
    ),
}
