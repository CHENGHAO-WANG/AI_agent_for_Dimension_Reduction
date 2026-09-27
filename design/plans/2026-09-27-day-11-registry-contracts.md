# Day 11 — registry and contracts: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every hardcoded set of method names with properties the registry declares, make the registry's text agree with its executors, and carry sample identifiers from loading into the cache and the dataset digest.

**Architecture:** The registry gains a third kind (`visualization`), conditional yes/no properties written in YAML (`{when: {param: [values]}}`), and named limits on d implemented in a new `drtools/constraints.py`. Readers that branched on op names switch to those declarations. Sample identifiers travel in `meta["sample_ids"]`, are checked by the loader contract, cached in `data/sample_ids.json` and hashed into the digest.

**Tech Stack:** Python 3.12 in `.venv`, PyYAML, pytest, scikit-learn, openTSNE, umap-learn, pandas, anndata.

**Spec:** `design/specs/2026-09-27-day-11-registry-contracts.md`. The settled items it names in its opening section (kernel PCA terminal-only; `intermediate` implies `stochastic: false`; `requires_connected_graph`; the `nested_in_d` comparison) come from `design/notes.md` sections 3.5 and 3.9 and are implemented here as those sections state them.

## Global Constraints

- Run Python as `.venv/Scripts/python.exe` from the repo root, in Git Bash. Tests: `.venv/Scripts/python.exe -m pytest -q`. Baseline on `day-11-registry` before Task 1: **416 passed in about 66 s**.
- The full suite passes at the end of every task; that is the daily smoke test (notes, section 9).
- Name concepts in `CONTEXT.md`'s terms: Op, Reduction, Visualization method, Terminal method, Stage, Candidate, Reference, Run. "Sample identifier" is new and goes to `CONTEXT.md` in Task 9.
- Every refusal says what was wrong and what to do, and names no route around the check.
- Commit only when the user has authorised commits for this execution; otherwise skip each task's commit step and leave the work uncommitted. Main takes changes by PR, never by direct push.
- Commit messages follow the repo's style (an imperative sentence saying what changed and why) and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Tasks run in order, one implementer. A later task may rely on names an earlier task defines; each task's **Interfaces** block lists them.

## Review Focus

1. A plan written before today that asks `tsne` for `n_components: 3` at small n was valid yesterday; it must now be refused with a message saying to set 2. Pinned in Task 3.
2. An adapter that returns `sample_ids` as a NumPy array or a pandas Index, the natural forms, must be refused with a message that says how to convert, not accepted half-way or failed deep in the cache. Pinned in Task 8.
3. A Run created before today, reopened with `--data`, must be told its cache predates the identity rule, not that it holds "a different dataset". Pinned in Task 8.
4. `--id-column` given for a format with no columns (`.npy`, a synthetic fixture) must be refused, not silently ignored. Pinned in Task 8.
5. A kernel added to `kernel_pca`'s `choices` without a matching `default_rule` branch must stop the registry loading, rather than leave the executor's label untested. Pinned in Task 2.

---

### Task 1: The named limits on d

**Files:**
- Create: `drtools/constraints.py`
- Modify: `drtools/executors/manifold.py:30-37` (delete `LLE_NEIGHBOUR_MINIMUM`), `:138` (read the new function)
- Test: `tests/test_constraints.py`

**Interfaces:**
- Produces: `drtools.constraints.Rule` (frozen dataclass: `name: str`, `sentence: str`, `violation: Callable[[dict[str, Any]], str | None]`); `drtools.constraints.RULES: dict[str, Rule]` holding `"lle_neighbour_minimum"`; `drtools.constraints.lle_neighbour_minimum(method: str, n_components: int) -> int`. `violation` takes a stage's resolved parameters and returns a refusal sentence, or `None` when the stage satisfies the rule. `constraints.py` imports nothing from `drtools`, so the registry can import it.

- [ ] **Step 1: Write the failing tests**

`tests/test_constraints.py`:

```python
"""Limits on d that a capability record cannot hold as a number.

Measured on day 11: scikit-learn refuses Hessian LLE below its minimum, and fits the
other three variants one below the toolbox's d + 1. So Hessian's minimum is the
library's and the others' is mathematical, and the tests say which is which.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

from drtools.constraints import RULES, lle_neighbour_minimum

VARIANTS = ["standard", "modified", "hessian", "ltsa"]


@pytest.fixture(scope="module")
def roll() -> np.ndarray:
    X, _ = make_swiss_roll(300, noise=0.05, random_state=0)
    return np.hstack([X, np.random.default_rng(0).normal(size=(300, 7))])


def _fit(X: np.ndarray, method: str, d: int, k: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        LocallyLinearEmbedding(
            n_components=d, n_neighbors=k, method=method,
            eigen_solver="dense", random_state=0,
        ).fit(X)


def test_the_minimum_grows_with_d_and_depends_on_the_variant() -> None:
    assert [lle_neighbour_minimum(m, 2) for m in VARIANTS] == [3, 3, 6, 3]
    assert lle_neighbour_minimum("hessian", 8) == 45


@pytest.mark.parametrize("d", [2, 4])
@pytest.mark.parametrize("method", VARIANTS)
def test_every_variant_fits_in_the_library_at_its_minimum(roll, method, d) -> None:
    """Our minimum is never below the library's, or the rule would pass a doomed stage."""
    _fit(roll, method, d, lle_neighbour_minimum(method, d))


@pytest.mark.parametrize("d", [2, 4])
def test_hessians_minimum_is_the_librarys_own(roll, d) -> None:
    with pytest.raises(ValueError, match="hessian"):
        _fit(roll, "hessian", d, lle_neighbour_minimum("hessian", d) - 1)


def test_the_rule_refuses_below_the_minimum_and_names_it() -> None:
    message = RULES["lle_neighbour_minimum"].violation(
        {"method": "hessian", "n_components": 4, "n_neighbors": 10}
    )
    assert message is not None
    assert "15" in message and "n_neighbors" in message


def test_the_rule_accepts_a_stage_at_its_minimum() -> None:
    rule = RULES["lle_neighbour_minimum"]
    assert rule.violation({"method": "hessian", "n_components": 4, "n_neighbors": 15}) is None
    assert rule.violation({"method": "standard", "n_components": 2, "n_neighbors": 3}) is None


def test_every_rule_has_a_sentence_for_the_agent() -> None:
    for name, rule in RULES.items():
        assert rule.name == name
        assert len(rule.sentence) > 40
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_constraints.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'drtools.constraints'`.

- [ ] **Step 3: Write `drtools/constraints.py`**

```python
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
```

- [ ] **Step 4: Make the LLE executor read it**

In `drtools/executors/manifold.py`, delete the comment and table at lines 30-37 (`# scikit-learn enforces these deep inside the fit ...` through the closing `}` of `LLE_NEIGHBOUR_MINIMUM`). Add to the imports:

```python
from drtools.constraints import lle_neighbour_minimum
```

and replace

```python
    minimum = LLE_NEIGHBOUR_MINIMUM[method](n_components)
```

with

```python
    # The same function day 12's validator refuses with, so a plan refused at
    # registration and a stage refused here are refused by one rule.
    minimum = lle_neighbour_minimum(method, n_components)
```

The executor's own refusal message below it stays as it is.

- [ ] **Step 5: Run the new tests, then the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_constraints.py -q` — Expected: all pass.
Run: `.venv/Scripts/python.exe -m pytest -q` — Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add drtools/constraints.py drtools/executors/manifold.py tests/test_constraints.py
git commit -m "Give LLE's neighbour minimum one home the validator can share"
```

---

### Task 2: The registry schema — kinds, conditions, and the new fields

**Files:**
- Modify: `drtools/registry.py` (whole file; replacement below)
- Test: `tests/test_registry_schema.py` (new)

**Interfaces:**
- Consumes: `drtools.constraints.RULES` (Task 1).
- Produces:
  - `KINDS = ("preprocessing", "reduction", "visualization")`; `CONDITIONAL_PROPERTIES = ("euclidean", "nested_in_d", "requires_connected_graph")`; `EMPHASES = ("local", "global", "balanced")`; `NEW_ROWS = ("transform", "nystrom", "none")`.
  - `Condition = bool | dict[str, tuple[Any, ...]]`.
  - `ParamSpec.default_rule: tuple[tuple[Condition, str], ...]` (empty when undeclared).
  - `OpSpec.conditions: dict[str, Condition]`, `OpSpec.d_limits: tuple[str, ...]`, `OpSpec.is_visualization: bool`, `OpSpec.holds(prop: str, params: dict | None = None) -> bool` (raises `RegistryError` for an undeclared property; a parameter missing from `params` takes its default), `OpSpec.default_rule(param_name: str, params: dict | None = None) -> str | None`.
  - `Registry.visualization_methods() -> dict[str, OpSpec]`; `Registry.describe(name)` adds `d_limit_rules: [{"name", "sentence"}]` when the op names any.
  - `load_registry` refuses: an unknown kind; a malformed condition, one naming an undeclared parameter, one on a parameter without `choices`, or one naming a value outside them; `emphasis` outside `EMPHASES`, or null without `emphasis_reason`; `new_rows` outside `NEW_ROWS`; an unknown name in `d_limits`; a `default_rule` whose entries do not each condition on the same single parameter, or that misses or repeats one of its choices.
  - Fields are validated **when present**. Which fields every method must declare is asserted by a test in Task 4, once the YAML carries them, so the real registry keeps loading between tasks.

- [ ] **Step 1: Write the failing tests**

`tests/test_registry_schema.py`:

```python
"""What the registry accepts, and what it refuses at load.

Each test writes a small registry of its own, so the rules are pinned independently of
what today's methods happen to declare.
"""

from __future__ import annotations

import textwrap

import pytest

from drtools.registry import RegistryError, load_registry

TOY_PARAMS = """\
    params:
      n_components: {type: int, default: 2, min: 1}
      kernel: {type: str, default: rbf, choices: [rbf, poly]}
"""


def _registry(tmp_path, body: str, name: str = "registry.yaml"):
    path = tmp_path / name
    path.write_text(
        "version: 1\nops:\n" + textwrap.indent(textwrap.dedent(body), "  "),
        encoding="utf-8",
    )
    return load_registry(path)


def _toy(extra: str = "", kind: str = "reduction") -> str:
    return (
        f"toy:\n  kind: {kind}\n  summary: a toy method\n"
        + textwrap.indent(textwrap.dedent(extra), "  ")
        + textwrap.indent(textwrap.dedent(TOY_PARAMS), "  ")
    )


def test_a_visualization_method_is_its_own_kind(tmp_path) -> None:
    registry = _registry(tmp_path, _toy(kind="visualization"))
    spec = registry["toy"]
    assert spec.is_visualization and not spec.is_reduction
    assert list(registry.visualization_methods()) == ["toy"]
    assert registry.reductions() == {}


def test_an_unknown_kind_is_refused_naming_the_three(tmp_path) -> None:
    with pytest.raises(RegistryError, match="preprocessing, reduction, visualization"):
        _registry(tmp_path, _toy(kind="embedding"))


def test_a_property_may_hold_always_never_or_under_a_condition(tmp_path) -> None:
    spec = _registry(tmp_path, _toy("""\
        euclidean: {when: {kernel: [rbf]}}
        nested_in_d: true
        requires_connected_graph: false
    """))["toy"]
    assert spec.holds("nested_in_d") is True
    assert spec.holds("requires_connected_graph") is False
    assert spec.holds("euclidean", {"kernel": "rbf"}) is True
    assert spec.holds("euclidean", {"kernel": "poly"}) is False


def test_a_condition_reads_the_default_when_the_stage_leaves_the_parameter_unset(tmp_path) -> None:
    spec = _registry(tmp_path, _toy("euclidean: {when: {kernel: [rbf]}}\n"))["toy"]
    assert spec.holds("euclidean", {}) is True


def test_asking_for_an_undeclared_property_is_an_error_not_false(tmp_path) -> None:
    spec = _registry(tmp_path, _toy())["toy"]
    with pytest.raises(RegistryError, match="does not declare 'euclidean'"):
        spec.holds("euclidean")


@pytest.mark.parametrize(
    "condition, message",
    [
        ("{when: {metric: [euclidean]}}", "does not declare"),
        ("{when: {n_components: [2]}}", "declares no choices"),
        ("{when: {kernel: [gaussian]}}", "not among"),
        ("{when: {kernel: []}}", "non-empty list"),
        ("{kernel: [rbf]}", "expected true, false"),
        ("sometimes", "expected true, false"),
    ],
)
def test_a_malformed_condition_stops_the_registry_loading(tmp_path, condition, message) -> None:
    with pytest.raises(RegistryError, match=message):
        _registry(tmp_path, _toy(f"euclidean: {condition}\n"))


def test_a_null_emphasis_needs_a_reason(tmp_path) -> None:
    with pytest.raises(RegistryError, match="emphasis_reason"):
        _registry(tmp_path, _toy("emphasis: null\n"))
    spec = _registry(
        tmp_path, _toy("emphasis: null\nemphasis_reason: depends on the width\n"), "b.yaml"
    )["toy"]
    assert spec.raw["emphasis"] is None


def test_an_unknown_emphasis_or_route_to_new_rows_is_refused(tmp_path) -> None:
    with pytest.raises(RegistryError, match="emphasis"):
        _registry(tmp_path, _toy("emphasis: medium\n"))
    with pytest.raises(RegistryError, match="new_rows"):
        _registry(tmp_path, _toy("new_rows: approximate\n"), "b.yaml")


def test_an_unknown_limit_on_d_is_refused_and_a_known_one_is_described(tmp_path) -> None:
    with pytest.raises(RegistryError, match="constraints.py"):
        _registry(tmp_path, _toy("d_limits: [no_such_rule]\n"))
    registry = _registry(tmp_path, _toy("d_limits: [lle_neighbour_minimum]\n"), "b.yaml")
    rules = registry.describe("toy")["d_limit_rules"]
    assert rules[0]["name"] == "lle_neighbour_minimum"
    assert "n_neighbors" in rules[0]["sentence"]


def _toy_with_gamma(rule: str) -> str:
    return (
        "toy:\n  kind: reduction\n  summary: a toy method\n  params:\n"
        "    n_components: {type: int, default: 2, min: 1}\n"
        "    kernel: {type: str, default: rbf, choices: [rbf, poly, cosine]}\n"
        "    gamma:\n      type: float\n      default: null\n"
        + textwrap.indent(textwrap.dedent(rule), "      ")
    )


def test_a_default_rule_names_the_label_each_branch_records(tmp_path) -> None:
    spec = _registry(tmp_path, _toy_with_gamma("""\
        default_rule:
          - {when: {kernel: [rbf]}, rule: median heuristic}
          - {when: {kernel: [poly]}, rule: library default}
          - {when: {kernel: [cosine]}, rule: not applicable}
    """))["toy"]
    assert spec.default_rule("gamma", {"kernel": "poly"}) == "library default"
    assert spec.default_rule("gamma", {}) == "median heuristic"


def test_a_plain_default_rule_applies_everywhere(tmp_path) -> None:
    spec = _registry(tmp_path, _toy_with_gamma("default_rule: median heuristic\n"))["toy"]
    assert spec.default_rule("gamma", {"kernel": "cosine"}) == "median heuristic"


def test_a_default_rule_that_misses_a_choice_stops_the_registry_loading(tmp_path) -> None:
    """Review focus 5: a kernel added to the choices without a branch is caught at load."""
    with pytest.raises(RegistryError, match="cosine"):
        _registry(tmp_path, _toy_with_gamma("""\
            default_rule:
              - {when: {kernel: [rbf]}, rule: median heuristic}
              - {when: {kernel: [poly]}, rule: library default}
        """))


def test_a_default_rule_that_names_a_choice_twice_is_refused(tmp_path) -> None:
    with pytest.raises(RegistryError, match="more than once"):
        _registry(tmp_path, _toy_with_gamma("""\
            default_rule:
              - {when: {kernel: [rbf, poly]}, rule: median heuristic}
              - {when: {kernel: [poly, cosine]}, rule: library default}
        """))
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_registry_schema.py -q`
Expected: failures — `kind must be 'preprocessing' or 'reduction'` for the visualization test, and `AttributeError` for `holds` / `default_rule` / `visualization_methods`.

- [ ] **Step 3: Replace `drtools/registry.py`**

```python
"""Reading the method capability records.

`registry.yaml` is what the agent knows about the methods; this module is how that
knowledge is queried, and it is the single place where a declared parameter is turned
into a concrete value. Resolution records provenance — whether a value was chosen by
the planner or fell through to the registry default — because "the agent set
n_neighbors to 30 because density varied 40-fold" and "the agent left n_neighbors at
the default" are very different claims to make in a report, and the difference should
not rest on anyone's memory.

Some properties are not properties of an op alone: kernel PCA works through Euclidean
geometry only with the RBF kernel, and LLE is nested in d only in its standard variant.
Those are written as conditions on the op's own parameters, `{when: {kernel: [rbf]}}`,
and read through `OpSpec.holds`, so every check that asks the question gets the answer
for the parameters the stage actually runs with. A condition is validated when the
registry loads, because a typo in one would otherwise make a property silently false.
Limits on d that are arithmetic in d are not conditions; they are named rules in
`drtools/constraints.py`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Union

import yaml

from drtools.constraints import RULES

REGISTRY_PATH = Path(__file__).with_name("registry.yaml")

PYTHON_TYPES: dict[str, type] = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
}

KINDS = ("preprocessing", "reduction", "visualization")
CONDITIONAL_PROPERTIES = ("euclidean", "nested_in_d", "requires_connected_graph")
EMPHASES = ("local", "global", "balanced")
NEW_ROWS = ("transform", "nystrom", "none")

# A property that holds always, never, or only while each named parameter takes one of
# the listed values.
Condition = Union[bool, dict[str, tuple[Any, ...]]]


class RegistryError(ValueError):
    """The registry file itself is malformed, or an op or parameter is unknown."""


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] | None = None
    describes: str = ""
    # What the executor does when the plan leaves this parameter unset: the label the
    # executor records in its `<name>_source` note, per branch of a condition.
    default_rule: tuple[tuple[Condition, str], ...] = ()

    def coerce(self, value: Any) -> Any:
        """Cast a supplied value to the declared type, leaving null alone.

        JSON has no integers distinct from floats, so a plan saying `n_components: 2.0`
        must not reach scikit-learn as a float and fail deep inside a fit.
        """
        if value is None:
            return None
        target = PYTHON_TYPES[self.type]
        if target is bool:
            return bool(value)
        if target is int and isinstance(value, float) and value != int(value):
            raise RegistryError(f"{self.name} must be a whole number, got {value}")
        return target(value)


@dataclass(frozen=True)
class OpSpec:
    name: str
    kind: str
    summary: str
    params: dict[str, ParamSpec] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    conditions: dict[str, Condition] = field(default_factory=dict)
    d_limits: tuple[str, ...] = ()

    @property
    def is_reduction(self) -> bool:
        return self.kind == "reduction"

    @property
    def is_visualization(self) -> bool:
        return self.kind == "visualization"

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(self.raw.get("roles", ()))

    @property
    def handles_sparse(self) -> bool:
        return bool(self.raw.get("handles_sparse", False))

    @property
    def preserves_sparsity(self) -> bool:
        return bool(self.raw.get("preserves_sparsity", False))

    @property
    def scales_to(self) -> int | None:
        value = self.raw.get("scales_to")
        return int(value) if value is not None else None

    @property
    def stochastic(self) -> bool:
        return bool(self.raw.get("stochastic", False))

    def can_be_intermediate(self) -> bool:
        return "intermediate" in self.roles or self.kind == "preprocessing"

    def holds(self, prop: str, params: dict[str, Any] | None = None) -> bool:
        """Whether a declared property holds for a stage running with `params`.

        An undeclared property is an error rather than `False`: a check that reads a
        property the op never declared has found a gap in the registry, and treating
        the gap as a "no" is how a new method escapes the check meant to govern it.
        """
        if prop not in self.conditions:
            raise RegistryError(
                f"{self.name} does not declare {prop!r}; declare it in registry.yaml "
                "as true, false or a condition on one of its parameters"
            )
        return _condition_holds(self.conditions[prop], params or {}, self.params)

    def default_rule(self, param_name: str, params: dict[str, Any] | None = None) -> str | None:
        """The label the executor records when `param_name` is left unset."""
        for condition, label in self.params[param_name].default_rule:
            if _condition_holds(condition, params or {}, self.params):
                return label
        return None


@dataclass(frozen=True)
class Registry:
    version: int
    ops: dict[str, OpSpec]

    def __contains__(self, name: object) -> bool:
        return name in self.ops

    def __getitem__(self, name: str) -> OpSpec:
        try:
            return self.ops[name]
        except KeyError:
            raise RegistryError(
                f"unknown op {name!r}; registered ops are "
                f"{', '.join(sorted(self.ops))}"
            ) from None

    def reductions(self) -> dict[str, OpSpec]:
        return {n: s for n, s in self.ops.items() if s.is_reduction}

    def visualization_methods(self) -> dict[str, OpSpec]:
        return {n: s for n, s in self.ops.items() if s.is_visualization}

    def preprocessing(self) -> dict[str, OpSpec]:
        return {n: s for n, s in self.ops.items() if s.kind == "preprocessing"}

    def resolve_params(
        self, name: str, given: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Merge supplied parameters over the declared defaults.

        Returns the resolved values and, for each one, where it came from. Unknown
        parameter names are an error rather than a silent no-op: a plan that sets
        `perplexity` on Isomap has misunderstood something, and swallowing it would
        leave the report describing a setting that never took effect.
        """
        spec = self[name]
        given = dict(given or {})

        unknown = set(given) - set(spec.params)
        if unknown:
            known = ", ".join(sorted(spec.params)) or "none"
            raise RegistryError(
                f"{name} has no parameter(s) {sorted(unknown)}; it accepts: {known}"
            )

        resolved: dict[str, Any] = {}
        provenance: dict[str, str] = {}
        for param_name, param in spec.params.items():
            if param_name in given:
                resolved[param_name] = _validated(param, param.coerce(given[param_name]))
                provenance[param_name] = "specified"
            else:
                resolved[param_name] = param.default
                provenance[param_name] = "registry_default"
        return resolved, provenance

    def describe(self, name: str) -> dict[str, Any]:
        """The full record for one op, as the agent and the report both read it."""
        spec = self[name]
        record = {"name": name, **spec.raw}
        if spec.d_limits:
            record["d_limit_rules"] = [
                {"name": rule, "sentence": RULES[rule].sentence} for rule in spec.d_limits
            ]
        return record


def _validated(param: ParamSpec, value: Any) -> Any:
    if value is None:
        return None
    if param.choices is not None and value not in param.choices:
        raise RegistryError(
            f"{param.name} must be one of {list(param.choices)}, got {value!r}"
        )
    if param.minimum is not None and value < param.minimum:
        raise RegistryError(f"{param.name} must be >= {param.minimum}, got {value}")
    if param.maximum is not None and value > param.maximum:
        raise RegistryError(f"{param.name} must be <= {param.maximum}, got {value}")
    return value


def _condition_holds(
    condition: Condition, params: dict[str, Any], specs: dict[str, ParamSpec]
) -> bool:
    if isinstance(condition, bool):
        return condition
    return all(
        params.get(name, specs[name].default) in values
        for name, values in condition.items()
    )


def _parse_condition(raw: Any, where: str, params: dict[str, ParamSpec]) -> Condition:
    if isinstance(raw, bool):
        return raw
    when = raw.get("when") if isinstance(raw, dict) and set(raw) == {"when"} else None
    if not isinstance(when, dict) or not when:
        raise RegistryError(
            f"{where}: expected true, false or {{when: {{parameter: [values]}}}}, "
            f"got {raw!r}"
        )
    parsed: dict[str, tuple[Any, ...]] = {}
    for param_name, values in when.items():
        param = params.get(param_name)
        if param is None:
            raise RegistryError(
                f"{where}: the condition names {param_name!r}, which this op does not "
                "declare"
            )
        if param.choices is None:
            raise RegistryError(
                f"{where}: the condition is on {param_name!r}, which declares no "
                "choices to match against"
            )
        if not isinstance(values, list) or not values:
            raise RegistryError(
                f"{where}: the values for {param_name!r} must be a non-empty list"
            )
        unknown = [v for v in values if v not in param.choices]
        if unknown:
            raise RegistryError(
                f"{where}: {unknown} are not among {param_name!r}'s choices "
                f"{list(param.choices)}"
            )
        parsed[param_name] = tuple(values)
    return parsed


def _parse_default_rule(
    raw: Any, where: str, params: dict[str, ParamSpec]
) -> tuple[tuple[Condition, str], ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return ((True, raw),)
    if not isinstance(raw, list) or not raw:
        raise RegistryError(
            f"{where}: default_rule must be a label, or a list of {{when, rule}} entries"
        )

    entries: list[tuple[Condition, str]] = []
    for entry in raw:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"when", "rule"}
            or not isinstance(entry["rule"], str)
        ):
            raise RegistryError(
                f"{where}: each default_rule entry needs exactly 'when' and a string 'rule'"
            )
        entries.append((_parse_condition({"when": entry["when"]}, where, params), entry["rule"]))

    names = {name for condition, _ in entries for name in condition}
    if len(names) != 1 or any(len(condition) != 1 for condition, _ in entries):
        raise RegistryError(
            f"{where}: every default_rule entry must condition on the same one parameter"
        )
    (name,) = names
    covered = [value for condition, _ in entries for value in condition[name]]
    repeated = sorted({v for v in covered if covered.count(v) > 1}, key=str)
    if repeated:
        raise RegistryError(
            f"{where}: default_rule names {repeated} more than once, so which label the "
            "executor records there is ambiguous"
        )
    missing = [v for v in params[name].choices if v not in covered]
    if missing:
        raise RegistryError(
            f"{where}: default_rule does not say what the executor records when "
            f"{name} is {missing}; add an entry for each"
        )
    return tuple(entries)


def _parse_param(name: str, raw: Any, op_name: str) -> ParamSpec:
    if not isinstance(raw, dict):
        raise RegistryError(f"{op_name}.{name}: parameter spec must be a mapping")
    declared = raw.get("type", "str")
    if declared not in PYTHON_TYPES:
        raise RegistryError(
            f"{op_name}.{name}: unknown type {declared!r}; "
            f"known types are {', '.join(sorted(PYTHON_TYPES))}"
        )
    choices = raw.get("choices")
    return ParamSpec(
        name=name,
        type=declared,
        default=raw.get("default"),
        minimum=raw.get("min"),
        maximum=raw.get("max"),
        choices=tuple(choices) if choices is not None else None,
        describes=str(raw.get("describes", "")).strip(),
    )


def _parse_op(name: str, raw: Any) -> OpSpec:
    if not isinstance(raw, dict):
        raise RegistryError(f"{name}: op record must be a mapping")
    kind = raw.get("kind")
    if kind not in KINDS:
        raise RegistryError(
            f"{name}: kind must be one of {', '.join(KINDS)}, got {kind!r}"
        )

    raw_params = raw.get("params") or {}
    params = {
        param_name: _parse_param(param_name, param_raw, name)
        for param_name, param_raw in raw_params.items()
    }
    params = {
        param_name: dataclasses.replace(
            param,
            default_rule=_parse_default_rule(
                raw_params[param_name].get("default_rule"),
                f"{name}.{param_name}.default_rule",
                params,
            ),
        )
        for param_name, param in params.items()
    }

    conditions = {
        prop: _parse_condition(raw[prop], f"{name}.{prop}", params)
        for prop in CONDITIONAL_PROPERTIES
        if prop in raw
    }

    if "emphasis" in raw:
        emphasis = raw["emphasis"]
        if emphasis is not None and emphasis not in EMPHASES:
            raise RegistryError(
                f"{name}: emphasis must be one of {', '.join(EMPHASES)} or null, "
                f"got {emphasis!r}"
            )
        if emphasis is None and not str(raw.get("emphasis_reason") or "").strip():
            raise RegistryError(
                f"{name}: emphasis is null, so emphasis_reason must say why no value is "
                "declared and when to revisit it"
            )

    if "new_rows" in raw and raw["new_rows"] not in NEW_ROWS:
        raise RegistryError(
            f"{name}: new_rows must be one of {', '.join(NEW_ROWS)}, "
            f"got {raw['new_rows']!r}"
        )

    d_limits = tuple(raw.get("d_limits") or ())
    unknown_rules = [rule for rule in d_limits if rule not in RULES]
    if unknown_rules:
        raise RegistryError(
            f"{name}: d_limits names {unknown_rules}, which drtools/constraints.py does "
            f"not define; defined rules are {sorted(RULES)}"
        )

    return OpSpec(
        name=name,
        kind=kind,
        summary=str(raw.get("summary", "")).strip(),
        params=params,
        raw=raw,
        conditions=conditions,
        d_limits=d_limits,
    )


# Sized well above one: the schema tests each load a small registry of their own, and
# a cache of four would evict the real one after every few of them.
@lru_cache(maxsize=32)
def load_registry(path: str | Path = REGISTRY_PATH) -> Registry:
    """Parse and validate the registry. Cached, since it is read on every command."""
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "ops" not in document:
        raise RegistryError(f"{path}: expected a mapping with an 'ops' key")

    ops = {name: _parse_op(name, raw) for name, raw in document["ops"].items()}
    return Registry(version=int(document.get("version", 1)), ops=ops)
```

- [ ] **Step 4: Run the new tests, then the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_registry_schema.py -q` — Expected: all pass.
Run: `.venv/Scripts/python.exe -m pytest -q` — Expected: all pass (the real registry declares none of the new fields yet, and `kind: reduction` is still valid).

- [ ] **Step 5: Commit**

```bash
git add drtools/registry.py tests/test_registry_schema.py
git commit -m "Let the registry declare a class, conditional properties and named limits"
```

---

### Task 3: The visualization class

**Files:**
- Modify: `drtools/registry.yaml` (`tsne` and `umap`: `kind`; `tsne.n_components`)
- Modify: `drtools/pipeline.py:133-178` (`validate_stages`)
- Modify: `drtools/plan.py:133` (`PlanState.advance`), `:392` (`exceeds_scale_limit`)
- Modify: `drtools/cli.py:179` (`--kind` choices)
- Test: `tests/test_pipeline.py`, `tests/test_registry.py`

**Interfaces:**
- Consumes: `OpSpec.is_visualization`, `Registry.visualization_methods()` (Task 2).
- Produces: `validate_stages` refuses a visualization method whose resolved `n_components` is not 2 (`PipelineError`, message contains `"visualization method"` and `"n_components = 2"`); a candidate may end in a reduction or a visualization method; the non-terminal refusal no longer claims the output lacks a metric. `drtools methods --kind visualization` lists the class.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
def test_a_visualization_method_runs_only_at_two_dimensions() -> None:
    """Review focus 1: a plan valid yesterday asking t-SNE for 3 is refused, saying 2."""
    for op in ("tsne", "umap"):
        with pytest.raises(PipelineError, match="n_components = 2"):
            validate_stages([{"op": op, "params": {"n_components": 3}}])


def test_the_refusal_says_what_a_visualization_method_is_for() -> None:
    with pytest.raises(PipelineError) as error:
        validate_stages([{"op": "tsne", "params": {"n_components": 3}}])
    assert "visualization method" in str(error.value)
    assert "reduction" in str(error.value)


def test_a_candidate_may_end_in_a_visualization_method() -> None:
    stages = validate_stages([{"op": "pca", "params": {"n_components": 10}},
                              {"op": "umap", "params": {"n_components": 2}}])
    assert stages[-1]["op"] == "umap"


def test_the_terminal_refusal_is_about_position_not_about_metric() -> None:
    """Section 3.9: Isomap's and Diffusion Maps' outputs do have a metric."""
    with pytest.raises(PipelineError) as error:
        validate_stages([{"op": "isomap"}, {"op": "umap"}])
    message = str(error.value)
    assert "last stage" in message and "pca" in message
    assert "no meaningful metric" not in message
```

In `tests/test_pipeline.py`, change the docstring of `test_a_terminal_method_cannot_feed_another_stage` from `"""t-SNE-like coordinates have no metric for a downstream method to consume."""` to `"""A terminal method may stand only in a candidate's last stage."""`.

In `tests/test_registry.py`, replace `test_reductions_declare_the_properties_selection_depends_on`'s loop header

```python
    for name, spec in registry.reductions().items():
```

with

```python
    for name, spec in {**registry.reductions(), **registry.visualization_methods()}.items():
```

and append:

```python
def test_t_sne_and_umap_are_visualization_methods(registry) -> None:
    """Section 3.11: their output is a picture, never a representation."""
    for name in ("tsne", "umap"):
        assert registry[name].is_visualization
        assert not registry[name].is_reduction


def test_methods_lists_the_visualization_class(cli) -> None:
    result = cli("methods", "--kind", "visualization")
    assert result.code == 0
    listed = result.payload["ops"]
    assert {"tsne", "umap"} <= set(listed)
    assert all(record["kind"] == "visualization" for record in listed.values())
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline.py tests/test_registry.py -q`
Expected: the new tests fail (t-SNE at 3 passes `validate_stages`; `is_visualization` is false; `--kind visualization` is an invalid choice).

- [ ] **Step 3: Declare the class in the YAML**

In `drtools/registry.yaml`, under `tsne:` and `umap:`, change `kind: reduction` to `kind: visualization`. Under `tsne:`, change

```yaml
      n_components: {type: int, default: 2, min: 1, max: 3}
```

to

```yaml
      n_components: {type: int, default: 2, min: 1}
```

(the class rule in `validate_stages` holds it at 2; a `max: 3` would be a second, looser copy).

- [ ] **Step 4: Rewrite the stage checks in `validate_stages`**

Replace the loop and the final check in `drtools/pipeline.py` (lines 151-178) with:

```python
    for position, stage in enumerate(stages):
        op = stage["op"]
        try:
            spec = registry[op]
            resolved, _ = registry.resolve_params(op, stage["params"])
        except RegistryError as error:
            raise PipelineError(f"stage {position} ({op}): {error}") from None

        is_last = position == len(stages) - 1
        if not is_last and not spec.can_be_intermediate():
            following = stages[position + 1]["op"]
            allowed = ", ".join(
                sorted(n for n, s in registry.ops.items() if "intermediate" in s.roles)
            )
            raise PipelineError(
                f"stage {position} ({op}) cannot be followed by {following!r}: {op} is "
                "a terminal method, which may stand only in a candidate's last stage. "
                f"Only {allowed} may come before a reduction or a visualization method."
            )

        # Section 3.11: a visualization method's output is a picture, drawn at two
        # dimensions. Keyed on the class, so a method added to it needs no entry here.
        if spec.is_visualization and resolved.get("n_components") != 2:
            raise PipelineError(
                f"stage {position} ({op}) asks for "
                f"n_components={resolved.get('n_components')}, but {op} is a "
                "visualization method, and a visualization method runs at "
                "n_components = 2: its output is a picture, never a representation of "
                "more dimensions. Set n_components to 2, or choose a reduction if the "
                "analysis needs more dimensions."
            )

    # Base preprocessing is the exception: it is a stage list by construction made only
    # of preprocessing, since its output is the common representation candidates are
    # measured against rather than an embedding.
    final = registry[stages[-1]["op"]]
    if require_terminal_reduction and not (final.is_reduction or final.is_visualization):
        raise PipelineError(
            f"a candidate must end in a reduction or a visualization method; this one "
            f"ends in {stages[-1]['op']!r}, which is preprocessing and leaves the data "
            "in its original dimensionality"
        )
    return stages
```

- [ ] **Step 5: Point the plan simulation and the scale check at both classes**

In `drtools/plan.py`, `PlanState.advance`, replace `if spec.is_reduction:` with `if spec.is_reduction or spec.is_visualization:`. In `_check_stage_against_state`, replace `if spec.is_reduction and spec.scales_to is not None and n > spec.scales_to:` with `if (spec.is_reduction or spec.is_visualization) and spec.scales_to is not None and n > spec.scales_to:`.

In `drtools/cli.py`, change the `--kind` argument's `choices=["preprocessing", "reduction"]` to `choices=["preprocessing", "reduction", "visualization"]`.

- [ ] **Step 6: Run the tests, then the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline.py tests/test_registry.py -q` — Expected: all pass.
Run: `.venv/Scripts/python.exe -m pytest -q` — Expected: all pass. A test that registers `tsne` or `umap` at `n_components` other than 2 would now fail; none exists on day 11 (checked by grep), so a failure here means a caller outside the tests builds such a stage — read it before changing anything.

- [ ] **Step 7: Commit**

```bash
git add drtools/registry.yaml drtools/pipeline.py drtools/plan.py drtools/cli.py tests/test_pipeline.py tests/test_registry.py
git commit -m "Class t-SNE and UMAP as visualization methods, drawn at two dimensions"
```

---

### Task 4: The declared properties

**Files:**
- Modify: `drtools/registry.yaml` (every reduction and visualization method)
- Test: `tests/test_registry.py`, `tests/test_nested_in_d.py` (new)

**Interfaces:**
- Consumes: `OpSpec.conditions`, `OpSpec.d_limits`, `Registry.reductions()`, `Registry.visualization_methods()` (Tasks 2-3); `drtools.constraints.RULES` (Task 1).
- Produces: every reduction and visualization method declares `euclidean`, `nested_in_d`, `requires_connected_graph`, `emphasis` and `new_rows`; `out_of_sample` no longer exists; `lle` declares `d_limits: [lle_neighbour_minimum]`. Tasks 6 and 7 read `requires_connected_graph` through `holds`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_registry.py`, add to the imports:

```python
import openTSNE
import umap
from sklearn.decomposition import PCA, KernelPCA, MiniBatchSparsePCA, TruncatedSVD
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding

from drtools.constraints import RULES
```

Replace `test_reductions_declare_the_properties_selection_depends_on` (as edited in Task 3) with:

```python
METHOD_FIELDS = {
    "preserves", "assumes", "scales_to", "handles_sparse", "roles",
    "euclidean", "nested_in_d", "requires_connected_graph", "emphasis", "new_rows",
}


def _methods(registry):
    return {**registry.reductions(), **registry.visualization_methods()}


def test_every_method_declares_the_properties_checks_read(registry) -> None:
    """A missing one is a question a check asks and the registry cannot answer."""
    for name, spec in _methods(registry).items():
        missing = METHOD_FIELDS - set(spec.raw)
        assert not missing, f"{name} does not declare {sorted(missing)}"
        assert "out_of_sample" not in spec.raw, f"{name} still declares out_of_sample"
```

Append:

```python
def test_only_a_deterministic_op_may_stand_before_another_method(registry) -> None:
    """Section 3.9. Vacuous today; it catches granting the role to LLE or Laplacian
    Eigenmaps, which are cheap and look like reasonable pre-steps."""
    for name, spec in registry.ops.items():
        if "intermediate" in spec.roles:
            assert not spec.stochastic, f"{name} is stochastic but may be intermediate"


def test_every_limit_on_d_is_named_by_some_method(registry) -> None:
    named = {rule for spec in registry.ops.values() for rule in spec.d_limits}
    assert named == set(RULES)


def test_lle_is_described_with_its_neighbour_minimum(registry) -> None:
    rules = registry.describe("lle")["d_limit_rules"]
    assert [r["name"] for r in rules] == ["lle_neighbour_minimum"]


# The library class each executor fits, for every method whose record says it can
# place new rows by `transform`. PCA's executor uses TruncatedSVD on sparse input.
TRANSFORM_CLASSES = {
    "pca": (PCA, TruncatedSVD),
    "kernel_pca": (KernelPCA,),
    "sparse_pca": (MiniBatchSparsePCA,),
    "isomap": (Isomap,),
    "lle": (LocallyLinearEmbedding,),
    "tsne": (openTSNE.TSNEEmbedding,),
    "umap": (umap.UMAP,),
}


def test_every_method_declaring_a_transform_has_one_in_its_library(registry) -> None:
    declared = {n for n, s in _methods(registry).items() if s.raw["new_rows"] == "transform"}
    assert declared == set(TRANSFORM_CLASSES)
    for name, classes in TRANSFORM_CLASSES.items():
        for cls in classes:
            assert hasattr(cls, "transform"), f"{name}: {cls.__name__} has no transform"


def test_the_methods_declared_without_a_transform_really_lack_one(registry) -> None:
    """Diffusion Maps is written in the toolbox and has none; its Nystrom extension is day 14's."""
    assert registry["laplacian_eigenmaps"].raw["new_rows"] == "nystrom"
    assert registry["diffusion_maps"].raw["new_rows"] == "nystrom"
    assert registry["mds"].raw["new_rows"] == "none"
    assert not hasattr(SpectralEmbedding, "transform")
    assert not hasattr(MDS, "transform")
```

`tests/test_nested_in_d.py`:

```python
"""Every method claiming nested_in_d is nested, checked rather than assumed.

Section 3.5: a method is nested in d when its d-dimensional embedding equals the first
d columns of its embedding at a larger d. Slicing a method that is not nested understates
its fidelity at the smaller d, so the claim is tested for every op that makes it,
including a claim that holds only under a condition (LLE's standard variant).
"""

from __future__ import annotations

import numpy as np
import pytest

from drtools.loaders import load
from drtools.pipeline import run_pipeline
from drtools.registry import load_registry


@pytest.fixture(scope="module")
def padded_roll():
    """A noisy Swiss roll with five low-variance columns added, so d = 5 is possible."""
    X, labels, _ = load("swiss_roll", n_samples=400, noise=0.3)
    padding = 0.1 * np.random.default_rng(0).normal(size=(X.shape[0], 5))
    return np.hstack([X, padding]), labels


def _claims() -> list[tuple[str, dict]]:
    registry = load_registry()
    methods = {**registry.reductions(), **registry.visualization_methods()}
    claims: list[tuple[str, dict]] = []
    for op, spec in methods.items():
        condition = spec.conditions["nested_in_d"]
        if condition is True:
            claims.append((op, {}))
        elif isinstance(condition, dict):
            claims += [(op, {p: v}) for p, values in condition.items() for v in values]
    return claims


def _leading_agreement(X, labels, op: str, params: dict) -> float:
    small = run_pipeline(X, labels, [{"op": op, "params": {"n_components": 2, **params}}], seed=0)
    large = run_pipeline(X, labels, [{"op": op, "params": {"n_components": 5, **params}}], seed=0)
    return min(
        abs(np.corrcoef(small.embedding[:, j], large.embedding[:, j])[0, 1]) for j in range(2)
    )


@pytest.mark.parametrize("op, params", _claims())
def test_a_method_claiming_nested_in_d_is_nested(padded_roll, op, params) -> None:
    assert _leading_agreement(*padded_roll, op, params) > 0.9999


def test_the_comparison_can_fail(padded_roll) -> None:
    """Negative control: modified LLE, declared not nested, scored 0.91 on day 11."""
    assert _leading_agreement(*padded_roll, "lle", {"method": "modified"}) < 0.99
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_registry.py tests/test_nested_in_d.py -q`
Expected: `test_every_method_declares...` fails naming the missing fields; `_claims()` raises `KeyError: 'nested_in_d'` at collection.

- [ ] **Step 3: Declare the properties**

In `drtools/registry.yaml`, delete every `out_of_sample:` line (ten of them). Then insert, directly after each method's `stochastic:` line, the block for that method:

`pca`:
```yaml
    euclidean: true
    nested_in_d: true
    requires_connected_graph: false
    emphasis: global
    new_rows: transform
```

`kernel_pca`:
```yaml
    euclidean: {when: {kernel: [rbf]}}
    nested_in_d: true
    requires_connected_graph: false
    emphasis: null
    emphasis_reason: >
      Which distances it keeps depends on the kernel and its width: close to PCA, and so
      global, with a wide RBF kernel; mainly neighbours with a narrow one. Reopen when
      tuning shows its chosen widths consistently favour one side.
    new_rows: transform
```

`sparse_pca`:
```yaml
    euclidean: true
    # Not measured. Declared false, which costs a refit at every d and never overstates
    # fidelity; a claim of true would have to pass tests/test_nested_in_d.py.
    nested_in_d: false
    requires_connected_graph: false
    emphasis: global
    new_rows: transform
```

`laplacian_eigenmaps`:
```yaml
    euclidean: true
    nested_in_d: true
    requires_connected_graph: true
    emphasis: local
    new_rows: nystrom
```

`diffusion_maps`:
```yaml
    euclidean: true
    nested_in_d: true
    requires_connected_graph: false
    emphasis: global
    new_rows: nystrom
```

`mds`:
```yaml
    euclidean: true
    nested_in_d: false
    requires_connected_graph: false
    emphasis: global
    new_rows: none
```

`isomap`:
```yaml
    euclidean: true
    nested_in_d: true
    requires_connected_graph: true
    emphasis: global
    new_rows: transform
```

`lle`:
```yaml
    euclidean: true
    nested_in_d: {when: {method: [standard]}}
    requires_connected_graph: true
    emphasis: local
    new_rows: transform
    d_limits: [lle_neighbour_minimum]
```

`tsne`:
```yaml
    euclidean: true
    nested_in_d: false
    requires_connected_graph: false
    emphasis: local
    new_rows: transform
```

`umap`:
```yaml
    euclidean: {when: {metric: [euclidean]}}
    # A smaller fit agrees with a larger one only through the spectral initialiser,
    # which is itself nested; under random initialisation it does not (section 3.5).
    nested_in_d: false
    requires_connected_graph: false
    emphasis: local
    new_rows: transform
```

- [ ] **Step 4: Run the tests, then the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_registry.py tests/test_nested_in_d.py -q` — Expected: all pass; the nested test runs six claims (pca, kernel_pca, laplacian_eigenmaps, diffusion_maps, isomap, lle standard) plus the control, in a few seconds.
Run: `.venv/Scripts/python.exe -m pytest -q` — Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add drtools/registry.yaml tests/test_registry.py tests/test_nested_in_d.py
git commit -m "Declare the properties checks read, and test the ones that can be measured"
```

---

### Task 5: Registry text kept true to its executor (defects 5 and 12)

**Files:**
- Modify: `drtools/registry.py` (`_parse_op`: refuse a null default with no rule)
- Modify: `drtools/registry.yaml` (`normalise_total.target`, `kernel_pca.gamma`, `diffusion_maps.epsilon`)
- Modify: `drtools/executors/linear.py:93-119` (`kernel_pca`'s gamma label)
- Test: `tests/test_registry.py`, `tests/test_registry_schema.py`

**Interfaces:**
- Consumes: `ParamSpec.default_rule`, `OpSpec.default_rule` (Task 2).
- Produces: every null-defaulted parameter declares a `default_rule`, and the registry refuses to load otherwise; `kernel_pca` records `gamma_source` as `"median pairwise distance heuristic"` (rbf, sigmoid), `"1/n_features (scikit-learn default)"` (poly) or `"not applicable"` (cosine), and records poly's effective gamma.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_registry_schema.py`:

```python
def test_a_null_default_without_a_rule_stops_the_registry_loading(tmp_path) -> None:
    body = (
        "toy:\n  kind: reduction\n  summary: a toy method\n  params:\n"
        "    width: {type: float, default: null}\n"
    )
    with pytest.raises(RegistryError, match="default_rule"):
        _registry(tmp_path, body)
```

Append to `tests/test_registry.py` (add `import numpy as np`, `from drtools.executors import Context, get_executor` and `from drtools.loaders import load` to the imports):

```python
def _default_rule_cases():
    registry = load_registry()
    cases = []
    for op, spec in registry.ops.items():
        for name, param in spec.params.items():
            for condition, label in param.default_rule:
                branches = (
                    [{}] if condition is True
                    else [{p: v} for p, values in condition.items() for v in values]
                )
                cases += [(op, name, branch, label) for branch in branches]
    return cases


@pytest.mark.parametrize("op, param, branch, label", _default_rule_cases())
def test_the_executor_records_the_default_rule_the_registry_declares(
    registry, op, param, branch, label
) -> None:
    """Two copies, checked against each other: the likely edit changes the rule and
    its label together in the executor, and this fails until the registry agrees."""
    X = np.abs(load("blobs", n_samples=120, n_features=6)[0])
    resolved, _ = registry.resolve_params(op, branch)
    _, notes = get_executor(op)(X, Context(seed=0), **resolved)
    assert notes[f"{param}_source"] == label


def test_every_null_default_is_covered_by_the_equality_test(registry) -> None:
    covered = {(op, param) for op, param, _, _ in _default_rule_cases()}
    nulls = {
        (op, name)
        for op, spec in registry.ops.items()
        for name, p in spec.params.items()
        if p.default is None
    }
    assert nulls == covered == {
        ("normalise_total", "target"), ("kernel_pca", "gamma"), ("diffusion_maps", "epsilon"),
    }
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_registry.py tests/test_registry_schema.py -q`
Expected: the null-default test fails (no refusal yet); the coverage test fails (`covered` is empty).

- [ ] **Step 3: Refuse a null default with no rule**

In `drtools/registry.py`, `_parse_op`, directly after the `params = {... dataclasses.replace ...}` block, add:

```python
    unruled = [p for p, spec in params.items() if spec.default is None and not spec.default_rule]
    if unruled:
        raise RegistryError(
            f"{name}: parameter(s) {unruled} default to null without a default_rule. "
            "Name the rule the executor applies when a plan leaves them unset, as the "
            "label it records in its <parameter>_source note."
        )
```

- [ ] **Step 4: Declare the three rules and rewrite their descriptions**

In `drtools/registry.yaml`, `normalise_total`:

```yaml
      target:
        type: float
        default: null
        default_rule: median sample total
        describes: common total to scale every sample to; left null, the median sample total
```

`kernel_pca`:

```yaml
      gamma:
        type: float
        default: null
        default_rule:
          - {when: {kernel: [rbf, sigmoid]}, rule: median pairwise distance heuristic}
          - {when: {kernel: [poly]}, rule: "1/n_features (scikit-learn default)"}
          - {when: {kernel: [cosine]}, rule: not applicable}
        describes: >
          kernel width, used by the rbf, sigmoid and poly kernels. Left null, rbf and
          sigmoid take the reciprocal of the median squared pairwise distance, measured
          on up to 1,000 sampled rows, which puts the width where the kernel varies;
          poly takes scikit-learn's 1/n_features; cosine has no width. The stage record's
          gamma_source says which rule ran.
```

`diffusion_maps`:

```yaml
      epsilon:
        type: float
        default: null
        default_rule: kernel-sum scaling criterion (Coifman and Singer)
        describes: >
          kernel bandwidth. Left null, it is chosen by Coifman and Singer's kernel-sum
          scaling criterion: the lower edge of the range of bandwidths over which the log
          kernel sum rises linearly with log bandwidth, where the kernel has just begun to
          resolve the manifold. The stage record's epsilon_source says so, and
          dimension_implied_by_bandwidth reports the intrinsic dimension the slope implies.
```

- [ ] **Step 5: Correct kernel PCA's poly label**

In `drtools/executors/linear.py`, replace lines 93-99 (the comment and the `gamma_source` / `if gamma is None ...` block) with:

```python
    # scikit-learn's default gamma of 1/n_features ignores the scale of the data
    # entirely. Tying it to the median pairwise distance at least puts the kernel
    # width in the range where the kernel actually varies. The polynomial kernel keeps
    # scikit-learn's default, and the cosine kernel has no width at all. Each label is
    # the registry's default_rule for that kernel, and a test holds the two together.
    recorded_gamma = gamma
    if kernel == "cosine":
        gamma_source = "not applicable"
    elif gamma is not None:
        gamma_source = "specified"
    elif kernel in {"rbf", "sigmoid"}:
        gamma = recorded_gamma = _median_heuristic_gamma(dense, ctx.seed)
        gamma_source = "median pairwise distance heuristic"
    else:
        recorded_gamma = 1.0 / dense.shape[1]
        gamma_source = "1/n_features (scikit-learn default)"
```

and replace the returned notes dict (lines 115-119) with:

```python
    return embedding, {
        "kernel": kernel,
        "gamma": float(recorded_gamma) if recorded_gamma is not None else None,
        "gamma_source": gamma_source,
    }
```

- [ ] **Step 6: Run the tests, then the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_registry.py tests/test_registry_schema.py -q` — Expected: all pass, with six default-rule cases (target; gamma at rbf, sigmoid, poly, cosine; epsilon).
Run: `.venv/Scripts/python.exe -m pytest -q` — Expected: all pass. A test asserting `gamma_source == "not applicable"` for poly would now fail; it asserted the defect, so correct it to the poly label.

- [ ] **Step 7: Commit**

```bash
git add drtools/registry.py drtools/registry.yaml drtools/executors/linear.py tests/test_registry.py tests/test_registry_schema.py
git commit -m "Hold each default's description to the rule its executor records"
```

---

### Task 6: The raw-counts check and the connected-graph check read declarations

**Files:**
- Modify: `drtools/plan.py` (delete `EUCLIDEAN_METHODS`, lines 35-39; rewrite the raw-counts finding, lines 421-434; the connected-graph condition, lines 467-471)
- Test: `tests/test_metrics_rank_plan.py`

**Interfaces:**
- Consumes: `OpSpec.is_reduction`, `OpSpec.is_visualization`, `OpSpec.holds` (Tasks 2-4).
- Produces: finding code `raw_counts_not_normalised` (was `raw_counts_into_euclidean_method`), raised for every reduction and visualization method reached while the values are raw counts; `likely_disconnected_graph` fires for any op whose `requires_connected_graph` holds.

- [ ] **Step 1: Write the failing tests**

In `tests/test_metrics_rank_plan.py`, replace `test_raw_counts_into_a_euclidean_method_are_rejected` with:

```python
COUNTS_PROFILE = {
    "shape": {"n_samples": 2700, "n_features": 32000, "storage": "sparse_csr"},
    "values": {"suspected_kind": "counts"},
}


def _raw_count_findings(candidates):
    report = validate_plan(make_plan(candidates=candidates), COUNTS_PROFILE)
    return [f for f in report["findings"] if f["code"] == "raw_counts_not_normalised"]


def test_raw_counts_into_any_method_are_rejected() -> None:
    findings = _raw_count_findings(
        [{"id": "t", "stages": [{"op": "densify"}, {"op": "tsne", "params": {"perplexity": 30}}]}]
    )
    assert [f["op"] for f in findings] == ["tsne"]
    assert "sequencing depth" in findings[0]["message"]
    assert "log1p" in findings[0]["fix"]


def test_pca_on_raw_counts_is_rejected() -> None:
    """Its leading component would track sample total (the registry's own sparse_note)."""
    findings = _raw_count_findings([{"id": "p", "stages": [{"op": "pca"}]}])
    assert [f["op"] for f in findings] == ["pca"]


def test_a_pca_first_stage_does_not_carry_raw_counts_past_the_check() -> None:
    """pca(50) -> tsne passed before day 11: pca was unchecked and cleared the flag."""
    findings = _raw_count_findings(
        [{"id": "c", "stages": [{"op": "pca", "params": {"n_components": 50}},
                                {"op": "tsne", "params": {"perplexity": 30}}]}]
    )
    assert [f["op"] for f in findings] == ["pca"]


def test_a_non_euclidean_kernel_on_raw_counts_is_still_rejected() -> None:
    """The objection is heteroscedasticity and depth, which no kernel removes."""
    findings = _raw_count_findings(
        [{"id": "k", "stages": [{"op": "densify"},
                                {"op": "kernel_pca", "params": {"kernel": "poly"}}]}]
    )
    assert [f["op"] for f in findings] == ["kernel_pca"]
```

In `test_normalising_first_clears_the_raw_counts_objection`, change `"raw_counts_into_euclidean_method"` to `"raw_counts_not_normalised"`.

Append:

```python
def test_the_connected_graph_warning_follows_the_declaration(blobs) -> None:
    """Laplacian Eigenmaps declares requires_connected_graph; UMAP does not."""
    _, _, profile = blobs
    recon = {"neighbourhood": {"k": 15, "n_connected_components": 5}}
    report = validate_plan(
        make_plan(candidates=[
            {"id": "pca2", "stages": [{"op": "pca"}]},
            {"id": "le", "stages": [{"op": "laplacian_eigenmaps", "params": {"n_neighbors": 10}}]},
            {"id": "u", "stages": [{"op": "umap", "params": {"n_neighbors": 10}}]},
        ]),
        profile,
        recon,
    )
    warned = {f["candidate"] for f in report["findings"] if f["code"] == "likely_disconnected_graph"}
    assert warned == {"le"}
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_metrics_rank_plan.py -q`
Expected: the new raw-count tests fail (no finding under the new code; `pca` not flagged).

- [ ] **Step 3: Rewrite the two checks**

In `drtools/plan.py`, delete the comment and `EUCLIDEAN_METHODS` (lines 35-39). Replace the raw-counts block (lines 421-434) with:

```python
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
```

Replace the connected-graph condition (lines 467-471)

```python
    if (
        neighbours is not None
        and recon is not None
        and op in {"isomap", "lle", "laplacian_eigenmaps"}
    ):
```

with

```python
    if (
        neighbours is not None
        and recon is not None
        and spec.holds("requires_connected_graph", params)
    ):
```

`_check_stage_against_state` only reaches this line for ops the registry knows, and every op with an `n_neighbors` parameter is a reduction or visualization method, which Task 4 requires to declare the property.

- [ ] **Step 4: Run the tests, then the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_metrics_rank_plan.py -q` — Expected: all pass.
Run: `.venv/Scripts/python.exe -m pytest -q` — Expected: all pass. A test registering a plan on count data with a PCA candidate and no `log1p` now sees it refused; correct the test's plan, not the check (spec, Testing).

- [ ] **Step 5: Commit**

```bash
git add drtools/plan.py tests/test_metrics_rank_plan.py
git commit -m "Refuse raw counts into any method, and read the graph requirement from the registry"
```

---

### Task 7: The extension point (defect 9)

**Files:**
- Modify: `drtools/heuristics.py:57-133` (`suggest`, `_neighbour_suggestion`)
- Modify: `drtools/registry.yaml:1-13` (header)
- Test: `tests/test_heuristics.py` (new), `tests/test_extension_point.py` (new)

**Interfaces:**
- Consumes: `load_registry`, `Registry.reductions()`, `Registry.visualization_methods()` (Task 2).
- Produces: `suggest(op, profile, recon=None, registry=None)`; the neighbour suggestion applies to any op declaring an `n_neighbors` parameter and starts from that parameter's registry default; an unknown op still gets `{}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_heuristics.py`:

```python
"""Suggested starting values, and which ops receive them."""

from __future__ import annotations

import textwrap

from drtools.heuristics import suggest
from drtools.registry import load_registry

LARGE = {"shape": {"n_samples": 5000}}


def test_any_op_with_n_neighbors_gets_the_neighbour_suggestion(tmp_path) -> None:
    """A new neighbour-graph method needs no edit here to be offered one."""
    path = tmp_path / "registry.yaml"
    path.write_text(textwrap.dedent("""\
        version: 1
        ops:
          toy_graph:
            kind: visualization
            summary: a toy neighbour-graph method
            params:
              n_components: {type: int, default: 2, min: 1}
              n_neighbors: {type: int, default: 20, min: 2}
    """), encoding="utf-8")
    suggested = suggest("toy_graph", LARGE, None, registry=load_registry(path))
    assert suggested["n_neighbors"]["value"] == 20


def test_lle_starts_from_its_registry_default_and_stays_capped() -> None:
    assert suggest("lle", LARGE)["n_neighbors"]["value"] == 12


def test_an_op_without_n_neighbors_gets_no_neighbour_suggestion() -> None:
    assert "n_neighbors" not in suggest("pca", LARGE)


def test_an_unknown_op_gets_nothing() -> None:
    assert suggest("not_an_op", LARGE) == {}
```

`tests/test_extension_point.py`:

```python
"""Adding a method: an entry in registry.yaml and one executor, plus what this lists.

A check that branches on a method's name escapes the registry: a new method missing
from the branch is silently ungoverned. So every place outside the executors that names
a method is listed below with the reason a new method can safely be absent from it. A
new branch fails the first test until it is listed; a listed branch that disappears
fails the second, so the list stays a true account of the extension point.
"""

from __future__ import annotations

import ast
from pathlib import Path

from drtools.registry import load_registry

DRTOOLS = Path(__file__).resolve().parents[1] / "drtools"

NAMED_ON_PURPOSE = {
    ("heuristics.py", "tsne"): "perplexity is t-SNE's own parameter; no other method has one",
    ("heuristics.py", "pca"): "the component count for a PCA first stage, read from the spectrum",
    ("heuristics.py", "diffusion_maps"): "alpha is density normalisation only here; sparse PCA's alpha is a sparsity penalty",
    ("heuristics.py", "lle"): "LLE's cap of 12 comes from its own measured sensitivity to n_neighbors",
    ("plan.py", "pca"): "section 3.4 names PCA as the linear baseline every plan includes",
    ("plan.py", "tsne"): "perplexity below n/3 is a check on t-SNE's own parameter",
    ("runs.py", "umap"): "the umap-learn library, whose version the run records; not the op",
}


def _named_methods() -> set[tuple[str, str]]:
    registry = load_registry()
    methods = set(registry.reductions()) | set(registry.visualization_methods())
    found: set[tuple[str, str]] = set()
    for path in DRTOOLS.rglob("*.py"):
        relative = path.relative_to(DRTOOLS)
        if relative.parts[0] == "executors":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in methods:
                found.add((relative.as_posix(), node.value))
    return found


def test_every_method_named_outside_the_executors_is_named_on_purpose() -> None:
    unexplained = sorted(_named_methods() - set(NAMED_ON_PURPOSE))
    assert not unexplained, (
        f"{unexplained} name a method outside the executors. Declare the property the "
        "branch is asking about in registry.yaml and read it instead, or, if the rule "
        "genuinely belongs to that one method's own parameter, list it here with why a "
        "new method can be absent from it."
    )


def test_the_list_is_current() -> None:
    stale = sorted(set(NAMED_ON_PURPOSE) - _named_methods())
    assert not stale, f"{stale} are listed but no longer named in the code; remove them"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_heuristics.py tests/test_extension_point.py -q`
Expected: `suggest()` rejects the `registry` keyword; LLE suggests 10; the extension test reports `('heuristics.py', 'isomap')`, `('heuristics.py', 'laplacian_eigenmaps')` and `('heuristics.py', 'umap')` as unexplained.

- [ ] **Step 3: Make the neighbour suggestion a predicate**

In `drtools/heuristics.py`, add `from drtools.registry import Registry, load_registry` to the imports. Change `suggest`'s signature and the neighbour branch:

```python
def suggest(
    op: str,
    profile: dict[str, Any],
    recon: dict[str, Any] | None = None,
    registry: Registry | None = None,
) -> dict[str, dict[str, Any]]:
    """Suggested parameters for `op` on this data, each with a rationale."""
    registry = registry or load_registry()
    spec = registry.ops.get(op)
    n = int(profile.get("shape", {}).get("n_samples", 0))
    suggestions: dict[str, dict[str, Any]] = {}
```

keep the `tsne` branch as it is, and replace

```python
    if op in {"umap", "laplacian_eigenmaps", "isomap", "lle"}:
        suggestions.update(_neighbour_suggestion(op, n, recon))
```

with

```python
    # Any op with a neighbourhood size gets the size-aware suggestion, starting from its
    # own registry default, so a new neighbour-graph method is offered one unedited.
    if spec is not None and "n_neighbors" in spec.params:
        base = int(spec.params["n_neighbors"].default)
        suggestions.update(_neighbour_suggestion(op, base, n, recon))
```

Change `_neighbour_suggestion`'s signature to `def _neighbour_suggestion(op: str, base: int, n: int, recon: dict[str, Any] | None) -> dict[str, dict[str, Any]]:` and delete its first line, `base = {"lle": 10, "isomap": 10, "laplacian_eigenmaps": 15, "umap": 15}[op]`. The rest of the function, including the `if op == "lle":` cap, stays.

- [ ] **Step 4: Narrow the registry header's claim**

In `drtools/registry.yaml`, replace line 13 (`# Adding a method means adding an entry here and one executor function. No skill edits.`) with:

```yaml
# Adding a method takes an entry here and one executor function, and no skill edits.
# Every property a check reads is declared in the entry: the class, whether the method
# works through Euclidean geometry, whether it is nested in d, whether it needs a
# connected graph, its emphasis, how it reaches new rows, and the named limits on d it
# obeys. A method may also bring a suggestion rule or a check on its own parameters;
# tests/test_extension_point.py lists every place outside the executors that names a
# method, and why a new method can be absent from each.
```

- [ ] **Step 5: Run the tests, then the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_heuristics.py tests/test_extension_point.py -q` — Expected: all pass.
Run: `.venv/Scripts/python.exe -m pytest -q` — Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add drtools/heuristics.py drtools/registry.yaml tests/test_heuristics.py tests/test_extension_point.py
git commit -m "List every method named outside the executors, and narrow the extension claim to it"
```

---

### Task 8: Sample identifiers (defect 21)

**Files:**
- Modify: `drtools/contract.py` (docstring; `check_dataset`; new `_check_sample_ids`, `resolve_sample_ids`)
- Modify: `drtools/loaders/__init__.py` (`load`, `_load_npz`, `_load_table`, `_load_h5ad`, `load_pbmc3k`)
- Modify: `drtools/digest.py` (`content_hash`)
- Modify: `drtools/cache.py` (`write_cache`, `ensure_cache`, new `read_sample_ids`)
- Modify: `drtools/cli.py:306-311` (`--id-column`), `:1580-1584` (`_load`)
- Test: `tests/test_contract.py`, `tests/test_loaders.py`, `tests/test_digest.py`, `tests/test_cache_identity.py`

**Interfaces:**
- Produces:
  - `drtools.contract.resolve_sample_ids(meta: dict, n_samples: int) -> tuple[list[str], str]` — the identifiers and their source; row positions `"0"`..`"n-1"` with source `"row_order"` when `meta` has none; otherwise `meta["sample_ids"]` with `meta.get("sample_ids_source", "loader")`.
  - `load()` always returns `meta["sample_ids"]` (list of `str`, length n) and `meta["sample_ids_source"]` (`"loader"` or `"row_order"`, assigned by the toolbox).
  - `content_hash(X, labels, sample_ids: Sequence[str] | None = None) -> str` — `None` hashes row order.
  - `drtools.cache.read_sample_ids(run: RunDir) -> list[str]`; `data/sample_ids.json`; `meta.json` gains `sample_ids_source` and `n_sample_ids` and never holds the list.
  - CLI `--id-column`, passed to `load` as `id_column`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_contract.py`:

```python
from drtools.contract import resolve_sample_ids


def test_accepts_one_string_identifier_per_sample() -> None:
    meta = {**VALID_META, "sample_ids": [f"s{i}" for i in range(10)]}
    check_dataset(np.zeros((10, 3)), None, meta)


def test_rejects_identifiers_that_are_not_a_list_of_strings_and_says_how_to_convert() -> None:
    """Review focus 2: an array or an Index is the natural thing for an adapter to return."""
    meta = {**VALID_META, "sample_ids": np.array([f"s{i}" for i in range(10)])}
    with pytest.raises(ContractError, match=r"list of strings.*\[str\(v\)"):
        check_dataset(np.zeros((10, 3)), None, meta)


def test_rejects_identifiers_misaligned_with_the_samples() -> None:
    meta = {**VALID_META, "sample_ids": ["a", "b"]}
    with pytest.raises(ContractError, match="misaligned"):
        check_dataset(np.zeros((10, 3)), None, meta)


def test_rejects_duplicated_identifiers_naming_one() -> None:
    meta = {**VALID_META, "sample_ids": ["a"] * 2 + [f"s{i}" for i in range(8)]}
    with pytest.raises(ContractError, match="'a'"):
        check_dataset(np.zeros((10, 3)), None, meta)


def test_rows_without_names_are_numbered_in_order() -> None:
    assert resolve_sample_ids({}, 3) == (["0", "1", "2"], "row_order")
```

Append to `tests/test_digest.py`:

```python
def test_sample_identifiers_are_part_of_identity():
    assert content_hash(BASE, None, ["a", "b"]) != content_hash(BASE, None, ["b", "a"])


def test_rows_without_names_hash_as_their_positions():
    n = BASE.shape[0]
    assert content_hash(BASE, None) == content_hash(BASE, None, [str(i) for i in range(n)])
```

(`BASE` is the module's existing two-row matrix.)

Append to `tests/test_cache_identity.py`:

```python
from drtools.cache import read_sample_ids
from drtools import jsonio


def test_identifiers_survive_the_cache_in_their_own_file(tmp_path):
    run = _run(tmp_path)
    ids = ["cell-a", "cell-b", "cell-c", "cell-d"]
    meta = write_cache(run, np.ones((4, 3)), None, {**META, "sample_ids": ids,
                                                   "sample_ids_source": "loader"})
    assert read_sample_ids(run) == ids
    assert "sample_ids" not in meta
    assert meta["sample_ids_source"] == "loader" and meta["n_sample_ids"] == 4


def test_ensure_cache_refuses_renamed_rows(tmp_path):
    run = _run(tmp_path)
    X = np.ones((2, 3))
    write_cache(run, X, None, {**META, "sample_ids": ["a", "b"]})
    with pytest.raises(ContractError, match="different dataset"):
        ensure_cache(run, X, None, {**META, "sample_ids": ["b", "a"]})


def test_a_cache_from_before_identifiers_is_told_so(tmp_path):
    """Review focus 3: not 'a different dataset' -- the rule changed, not the data."""
    run = _run(tmp_path)
    X = np.ones((4, 3))
    write_cache(run, X, None, META)
    path = run.path / "data" / "meta.json"
    legacy = jsonio.read(path)
    legacy.pop("sample_ids_source")
    jsonio.write(path, legacy)
    with pytest.raises(ContractError, match="before sample identifiers"):
        ensure_cache(run, X, None, META)
```

Append to `tests/test_loaders.py` (it already imports `load`, `ContractError`, `np` and `pytest`; add `import pandas as pd`):

```python
def test_a_csv_names_its_rows_from_the_id_column(tmp_path) -> None:
    path = tmp_path / "named.csv"
    pd.DataFrame({"sample": ["x", "y", "z"], "f0": [1.0, 2.0, 3.0],
                  "f1": [0.0, 1.0, 0.5]}).to_csv(path, index=False)
    X, _, meta = load(str(path), id_column="sample")
    assert X.shape == (3, 2)
    assert meta["sample_ids"] == ["x", "y", "z"]
    assert meta["sample_ids_source"] == "loader"


def test_an_unknown_id_column_is_refused_naming_the_columns(tmp_path) -> None:
    path = tmp_path / "named.csv"
    pd.DataFrame({"f0": [1.0, 2.0], "f1": [0.0, 1.0]}).to_csv(path, index=False)
    with pytest.raises(ContractError, match="no column named 'sample'"):
        load(str(path), id_column="sample")


def test_an_id_column_for_a_format_without_columns_is_refused() -> None:
    """Review focus 4: silently ignoring it would leave the rows numbered unasked."""
    with pytest.raises(ContractError, match="has none"):
        load("blobs", id_column="sample")


def test_rows_a_loader_cannot_name_are_numbered_and_say_so() -> None:
    X, _, meta = load("blobs", n_samples=30)
    assert meta["sample_ids"] == [str(i) for i in range(30)]
    assert meta["sample_ids_source"] == "row_order"


def test_a_loader_cannot_claim_the_source_for_itself(tmp_path) -> None:
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import numpy as np\n"
        "def load(path, **_):\n"
        "    return np.ones((3, 2)), None, {'name': 'a', 'source': 'a',\n"
        "                                   'sample_ids_source': 'loader'}\n",
        encoding="utf-8",
    )
    _, _, meta = load("anything", adapter=adapter)
    assert meta["sample_ids_source"] == "row_order"


def test_an_npz_may_carry_its_row_names(tmp_path) -> None:
    path = tmp_path / "d.npz"
    np.savez(path, X=np.ones((3, 2)), sample_ids=np.array(["p", "q", "r"]))
    _, _, meta = load(str(path))
    assert meta["sample_ids"] == ["p", "q", "r"]
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_contract.py tests/test_digest.py tests/test_cache_identity.py tests/test_loaders.py -q`
Expected: `ImportError` for `resolve_sample_ids` and `read_sample_ids`; `content_hash` rejects a third argument.

- [ ] **Step 3: The contract**

In `drtools/contract.py`, extend the module docstring's contract block, after the `meta` line:

```
                  meta may carry `sample_ids`: a list of distinct strings, one per sample,
                  in row order. A loader that has none leaves it out, and the toolbox
                  numbers the rows.
```

Change `check_dataset` to end with `_check_sample_ids(meta.get("sample_ids"), X.shape[0], origin)` after `_check_meta(meta, origin)`, and add:

```python
def _check_sample_ids(ids: Any, n_samples: int, origin: str) -> None:
    if ids is None:
        return
    if not isinstance(ids, list) or not all(isinstance(v, str) for v in ids):
        raise ContractError(
            f"{origin}: meta['sample_ids'] must be a list of strings, one per sample, "
            f"got {type(ids).__name__}. Convert in the loader, for example "
            "[str(v) for v in index]."
        )
    if len(ids) != n_samples:
        raise ContractError(
            f"{origin}: meta['sample_ids'] holds {len(ids)} identifiers for "
            f"{n_samples} samples — the two are misaligned"
        )
    seen: set[str] = set()
    repeated = [v for v in ids if v in seen or seen.add(v)]
    if repeated:
        raise ContractError(
            f"{origin}: meta['sample_ids'] repeats {len(set(repeated))} identifier(s), "
            f"for example {sorted(set(repeated))[:3]}. Each sample needs its own name, "
            "or an exported row cannot say which sample it is; make the names unique in "
            "the loader."
        )


def resolve_sample_ids(meta: dict[str, Any], n_samples: int) -> tuple[list[str], str]:
    """The identifiers a dataset's rows go by, and where they came from.

    Rows a loader did not name are numbered by position, recorded as `row_order`, so
    every reader can rely on identifiers existing and an export can say which kind it
    holds.
    """
    ids = meta.get("sample_ids")
    if ids is None:
        return [str(i) for i in range(n_samples)], "row_order"
    return list(ids), str(meta.get("sample_ids_source", "loader"))
```

- [ ] **Step 4: The loaders**

In `drtools/loaders/__init__.py`, import `resolve_sample_ids` from `drtools.contract` and add `COLUMN_SUFFIXES = (".csv", ".tsv", ".txt", ".h5ad")` beside `LABEL_COLUMN_CANDIDATES`. In `load`, add before the dispatch:

```python
    id_column = kwargs.get("id_column")
    has_columns = (
        spec not in GENERATORS
        and spec not in NAMED_DATASETS
        and Path(spec).suffix.lower() in COLUMN_SUFFIXES
    )
    if id_column is not None and adapter is None and not has_columns:
        raise ContractError(
            f"--id-column names a column, and {spec!r} has none: only .csv, .tsv, .txt "
            "and .h5ad inputs do. Its rows are named by its loader or numbered in "
            "order; drop the option."
        )
```

and replace its last three lines with:

```python
    check_dataset(X, labels, meta, origin=origin)
    meta.setdefault("spec", spec)
    # Assigned, never defaulted: whether the loader named the rows or the toolbox
    # numbered them is the toolbox's record, not a claim a loader makes for itself.
    meta["sample_ids_source"] = "loader" if meta.get("sample_ids") is not None else "row_order"
    meta["sample_ids"], _ = resolve_sample_ids(meta, X.shape[0])
    return X, labels, meta
```

In `_load_npz`, build `meta = {"name": path.stem, "source": str(path)}`, then before the return:

```python
    if "sample_ids" in archive.files:
        try:
            names = archive["sample_ids"]
        except ValueError:
            raise ContractError(
                f"{path.name}: sample_ids is stored as an object array, which cannot be "
                "read without unpickling. Store it as a string array, "
                "np.array(ids, dtype=str)."
            ) from None
        meta["sample_ids"] = [str(v) for v in names.ravel()]
    return _as_float(archive["X"]), labels, meta
```

In `_load_table`, change the signature to `(path: Path, *, label_column: str | None = None, id_column: str | None = None, **_: Any)` and add directly after `frame = pd.read_csv(path, sep=separator)`:

```python
    # read_csv builds no index, so row names arrive as an ordinary column. Named, it is
    # taken out of the features; not named, nothing guesses which column is not data.
    sample_ids = None
    if id_column is not None:
        if id_column not in frame.columns:
            raise ContractError(
                f"{path.name}: no column named {id_column!r} to take sample identifiers "
                f"from; its columns are {[str(c) for c in frame.columns][:20]}"
            )
        if id_column == label_column:
            raise ContractError(
                f"{path.name}: {id_column!r} is named as both the identifier and the "
                "label column; they must be different columns"
            )
        sample_ids = [str(v) for v in frame[id_column]]
        frame = frame.drop(columns=[id_column])
```

and before its return, `if sample_ids is not None: meta["sample_ids"] = sample_ids`.

In `_load_h5ad`, change the signature to add `id_column: str | None = None`, and after `meta` is built:

```python
    if id_column is not None and id_column not in adata.obs.columns:
        raise ContractError(
            f"{path.name}: no obs column named {id_column!r}; obs_names already name "
            "the rows, so omit the option to use them"
        )
    names = adata.obs[id_column] if id_column is not None else adata.obs_names
    meta["sample_ids"] = [str(v) for v in names]
```

In `load_pbmc3k`, add `"sample_ids": [str(v) for v in adata.obs_names],` to `meta`.

- [ ] **Step 5: The digest and the cache**

In `drtools/digest.py`, add `import json` and `from collections.abc import Sequence`; change the signature to `def content_hash(X: Matrix, labels: np.ndarray | None, sample_ids: Sequence[str] | None = None) -> str:` and its docstring to `"""A stable digest of the matrix, its label codes and its sample identifiers.

    Identifiers are part of identity because an export names its rows by them: the same
    numbers under different names would be a different claim about which sample is
    which. Rows with no names hash as their positions, so leaving them unnamed and
    numbering them 0..n-1 agree.
    """`. Before `return digest.hexdigest()`, add:

```python
    ids = sample_ids if sample_ids is not None else [str(i) for i in range(X.shape[0])]
    digest.update(b"sample_ids")
    digest.update(json.dumps(list(ids), ensure_ascii=False).encode("utf-8"))
```

In `drtools/cache.py`, import `resolve_sample_ids` from `drtools.contract`. In `write_cache`, after `sparse = sp.issparse(X)`:

```python
    ids, ids_source = resolve_sample_ids(meta, X.shape[0])
    # Their own file: at n = 107,000 the list is a megabyte or two, and meta.json is
    # read by the report and on every run lookup.
    jsonio.write(directory / "sample_ids.json", ids)
```

and replace the `descriptor` with:

```python
    descriptor = {
        **{key: value for key, value in meta.items() if key != "sample_ids"},
        "cached_storage": "sparse_csr" if sparse else "dense",
        "cached_shape": list(X.shape),
        "cached_dtype": str(X.dtype),
        "has_labels": labels is not None,
        "sample_ids_source": ids_source,
        "n_sample_ids": len(ids),
        "dataset_digest": content_hash(X, labels, ids),
    }
```

In `ensure_cache`, replace `incoming = content_hash(X, labels)` with:

```python
    if "sample_ids_source" not in cached:
        raise ContractError(_PREDATES_IDENTIFIERS)
    ids, _ = resolve_sample_ids(meta, X.shape[0])
    incoming = content_hash(X, labels, ids)
```

and add at module level:

```python
_PREDATES_IDENTIFIERS = (
    "this run's cache was written before sample identifiers became part of a dataset's "
    "identity, so its digest cannot be compared with the data just loaded. Analyse the "
    "data in a new run."
)


def read_sample_ids(run: RunDir) -> list[str]:
    """The cached rows' identifiers, in row order."""
    path = run.path / "data" / "sample_ids.json"
    if not path.exists():
        raise ContractError(_PREDATES_IDENTIFIERS)
    return jsonio.read(path)
```

- [ ] **Step 6: The CLI option**

In `drtools/cli.py`, after the `--label-column` argument:

```python
    parser.add_argument(
        "--id-column",
        default=None,
        help="column holding sample identifiers, for tabular and AnnData inputs",
    )
```

and in `_load`, after the `label_column` lines:

```python
    if getattr(args, "id_column", None):
        kwargs["id_column"] = args.id_column
```

- [ ] **Step 7: Run the tests, then the full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_contract.py tests/test_digest.py tests/test_cache_identity.py tests/test_loaders.py -q` — Expected: all pass.
Run: `.venv/Scripts/python.exe -m pytest -q` — Expected: all pass. `tests/test_skills.py` checks every flag the skills name exists; `--id-column` is new and named by no skill, so it is unaffected.

- [ ] **Step 8: Commit**

```bash
git add drtools/contract.py drtools/loaders/__init__.py drtools/digest.py drtools/cache.py drtools/cli.py tests/test_contract.py tests/test_digest.py tests/test_cache_identity.py tests/test_loaders.py
git commit -m "Carry sample identifiers through the loader contract, the cache and the digest"
```

---

### Task 9: The notes, the glossary, and closing the day

**Files:**
- Modify: `design/notes.md` (sections 3.2, 3.3, 3.4, 3.11, 9; the decision log)
- Modify: `CONTEXT.md` (via the `domain-modeling` skill)

**Interfaces:**
- Consumes: everything above; the final test count from the full suite.

- [ ] **Step 1: Correct the notes' descriptions**

In `design/notes.md`:

- Section 3.2, the sentence `The registry declares each method's role (\`can_be_intermediate\` for PCA/Kernel PCA, \`terminal_only\` for t-SNE/UMAP).` becomes `The registry declares each method's role: \`intermediate\` and \`terminal\` for PCA, the only op that may stand before another method, and \`terminal\` for every other (section 3.9).`
- Section 3.3, the sentence `The YAML is the extension point: adding a method is a registry entry plus one executor function, no skill edits.` becomes `The YAML is the extension point: adding a method is a registry entry plus one executor function, no skill edits, and every property a check reads is declared in the entry. A method may also bring a suggestion rule or a check on its own parameters; \`tests/test_extension_point.py\` lists every place outside the executors that names a method, and why a new method can be absent from it.`
- Section 3.4, the paragraph beginning `*Neither the count nor the spread is checked.*`: replace its last two sentences (from `Spread cannot be checked yet,` to the paragraph's end) with `The property it needs now exists: \`emphasis\`, declared on day 11, local, global or balanced, and null with a reason for kernel PCA. The check, a warning when no candidate other than the linear baseline is global, or none is local, is not yet scheduled.`
- Section 3.11, the reductions bullet `PCA, sparse PCA, kernel PCA, MDS, Isomap, Diffusion Maps, Laplacian Eigenmaps, LLE.` becomes `PCA, sparse PCA, kernel PCA, MDS, Isomap, Diffusion Maps, Laplacian Eigenmaps, LLE, and a GPLVM should one be added.`
- Section 9: in the day 11 row, change `(3.4, 3.9, 3.11, 3.12; defects 4, 5, 9, 12, 21)` to `(3.4, 3.9, 3.11, 3.12; defects 4, 5, 9, 10, 12, 21)`; in the day 12 row, change `defects 2, 3, 10, 11, 13-18, 20` to `defects 2, 3, 11, 13-18, 20`, and delete nothing else from it.

- [ ] **Step 2: Add the glossary term**

Invoke the `domain-modeling` skill and add to `CONTEXT.md`, under **Representations** or **The analysis** as the skill judges:

```markdown
**Sample identifier**:
The name a Run gives one row of the dataset: the loader's own row names where it has
them, or the row's position where it has none, recorded as which. Part of what the data
is, so a change to it alone makes a different dataset.
_Avoid_: row name, index, observation name, barcode
```

- [ ] **Step 3: Append the day 11 decision-log entry**

Append to the decision log in `design/notes.md`, keyed `- **Day 11** —`, in the voice of the existing entries: what was decided on each of the spec's eight sections and why, what was rejected, the two findings measured while planning (only Hessian LLE's neighbour minimum is scikit-learn's; the others' `d + 1` is mathematical; the nesting comparison through the pipeline gave 1.0 for every claim, 0.91 for modified LLE and 0.87 for MDS), the user's two corrections (t-SNE's d is fixed at 2 by its class, so no n-dependent rule; no combined `lowers_dimension` property), defect 10 closed a day early, and the test count and run time from Step 4.

- [ ] **Step 4: Run the full suite and record the count**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass. Put the count and time into the entry's last line, as `N tests, Ms.`

- [ ] **Step 5: Commit**

```bash
git add design/notes.md CONTEXT.md
git commit -m "Record day 11: the registry declares what checks read, and rows keep their names"
```

The pull request to main follows the project's route (squash merge with an explicit message) and is opened only when the user asks.
