"""Reading the method capability records.

`registry.yaml` is what the agent knows about the methods; this module is how that
knowledge is queried, and it is the single place where a declared parameter is turned
into a concrete value. Resolution records provenance — whether a value was chosen by
the planner or fell through to the registry default — because "the agent set
n_neighbors to 30 because density varied 40-fold" and "the agent left n_neighbors at
the default" are very different claims to make in a report, and the difference should
not rest on anyone's memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REGISTRY_PATH = Path(__file__).with_name("registry.yaml")

PYTHON_TYPES: dict[str, type] = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
}


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

    @property
    def is_reduction(self) -> bool:
        return self.kind == "reduction"

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
        return {"name": name, **self[name].raw}


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


@lru_cache(maxsize=4)
def load_registry(path: str | Path = REGISTRY_PATH) -> Registry:
    """Parse and validate the registry. Cached, since it is read on every command."""
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "ops" not in document:
        raise RegistryError(f"{path}: expected a mapping with an 'ops' key")

    ops: dict[str, OpSpec] = {}
    for name, raw in document["ops"].items():
        if not isinstance(raw, dict):
            raise RegistryError(f"{name}: op record must be a mapping")
        kind = raw.get("kind")
        if kind not in {"preprocessing", "reduction"}:
            raise RegistryError(
                f"{name}: kind must be 'preprocessing' or 'reduction', got {kind!r}"
            )
        params = {
            param_name: _parse_param(param_name, param_raw, name)
            for param_name, param_raw in (raw.get("params") or {}).items()
        }
        ops[name] = OpSpec(
            name=name,
            kind=kind,
            summary=str(raw.get("summary", "")).strip(),
            params=params,
            raw=raw,
        )

    return Registry(version=int(document.get("version", 1)), ops=ops)
