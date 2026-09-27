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
