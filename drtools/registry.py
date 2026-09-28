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

#: What chooses a method's d (sections 3.5 and 3.7). The first six are methods' own
#: criteria; `battery` reads the weighted battery score; `fixed` is a visualization
#: method's d = 2.
CRITERIA = (
    "explained_variance",
    "kernel_variance",
    "residual_variance",
    "stress",
    "diffusion_distance",
    "eigengap",
    "battery",
    "fixed",
)
#: What a tuned parameter's multiplier scales: the profile-derived suggestion, or the
#: executor's own rule for a width left unset.
TUNING_BASES = ("suggestion", "rule")

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
class TuningDecl:
    """How a method is tuned: the parameter scaled, what it scales, what picks d."""

    param: str | None
    base: str | None
    criterion: str
    applies: Condition = True
    #: A parameter swept over the plan's t_grid on every fit rather than refitted.
    swept: str | None = None


@dataclass(frozen=True)
class OpSpec:
    name: str
    kind: str
    summary: str
    params: dict[str, ParamSpec] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    conditions: dict[str, Condition] = field(default_factory=dict)
    d_limits: tuple[str, ...] = ()
    tuning: TuningDecl | None = None

    def tuned_param(self, params: dict[str, Any] | None = None) -> str | None:
        """The parameter tuning scales for a stage running with `params`, if any."""
        if self.tuning is None or self.tuning.param is None:
            return None
        if not _condition_holds(self.tuning.applies, params or {}, self.params):
            return None
        return self.tuning.param

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
    unruled = [p for p, spec in params.items() if spec.default is None and not spec.default_rule]
    if unruled:
        raise RegistryError(
            f"{name}: parameter(s) {unruled} default to null without a default_rule. "
            "Name the rule the executor applies when a plan leaves them unset, as the "
            "label it records in its <parameter>_source note."
        )

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
        tuning=_parse_tuning(name, kind, raw.get("tuning"), params),
    )


def _parse_tuning(
    name: str, kind: str, raw: Any, params: dict[str, ParamSpec]
) -> TuningDecl | None:
    """Validate a method's tuning declaration. Every method must carry one."""
    if kind == "preprocessing":
        if raw is not None:
            raise RegistryError(f"{name}: preprocessing is not tuned; remove `tuning`")
        return None
    if not isinstance(raw, dict):
        raise RegistryError(
            f"{name}: a {kind} must declare `tuning` -- the parameter it scales (or "
            "null), what that parameter's multiplier scales, and what chooses its d"
        )
    criterion = raw.get("criterion")
    if criterion not in CRITERIA:
        raise RegistryError(
            f"{name}: tuning.criterion must be one of {', '.join(CRITERIA)}, "
            f"got {criterion!r}"
        )
    if (criterion == "fixed") != (kind == "visualization"):
        raise RegistryError(
            f"{name}: tuning.criterion is `fixed` exactly for a visualization method, "
            "which runs at d = 2"
        )
    swept = raw.get("swept")
    if swept is not None and (swept not in params or params[swept].type != "int"):
        raise RegistryError(f"{name}: tuning.swept must name one of its integer parameters")
    param = raw.get("param")
    base = raw.get("base")
    if param is None:
        if base is not None or "applies" in raw:
            raise RegistryError(f"{name}: tuning with no param takes no base or applies")
        return TuningDecl(param=None, base=None, criterion=criterion)
    if param not in params:
        raise RegistryError(f"{name}: tuning.param {param!r} is not one of its parameters")
    if base not in TUNING_BASES:
        raise RegistryError(
            f"{name}: tuning.base must be one of {', '.join(TUNING_BASES)}, got {base!r}"
        )
    if base == "rule" and "width_multiplier" not in params:
        raise RegistryError(
            f"{name}: a width tuned from the executor's rule needs a width_multiplier "
            "parameter, since the rule is computed inside the executor"
        )
    applies = _parse_condition(raw.get("applies", True), f"{name}.tuning.applies", params)
    return TuningDecl(
        param=param, base=base, criterion=criterion, applies=applies, swept=swept
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
