"""The evaluation battery.

Every candidate is scored on the same metrics, whether or not they flatter it. The
agent chooses how to *weight* them and declares that weighting before any embedding is
computed; what it cannot do is choose which ones to look at after seeing the results.
A battery that the agent could prune would let a weak candidate win by quietly dropping
the measurement that exposed it.

Two ideas shape what is measured.

The first is that local and global fidelity are different questions and a single number
cannot answer both. Trustworthiness and continuity are a pair — one catches points
dragged together that were apart, the other points pulled apart that were together — and
an embedding can be excellent at one while destroying the other. The Shepard
correlation asks the global question separately.

The second is that a metric without a reference value is hard to read. If the original
data only achieves 0.62 label agreement among neighbours, an embedding reaching 0.58 has
lost almost nothing, and reporting 0.58 alone invites the opposite conclusion. So where
a ceiling exists, it is computed on the reference representation and reported alongside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors

from drtools.contract import Matrix

# Quadratic metrics are capped, since trustworthiness on a hundred thousand points is
# its own analysis project. The cap is recorded with the results.
METRIC_SAMPLE_CAP = 2000

#: Largest neighbourhood the battery ever uses.
MAX_K = 15

#: Below this many rows the local metrics are reported as unavailable rather than
#: computed. Trustworthiness over one or two neighbours is noise, and `rank` would
#: weight it at face value; below n=3 the rule's own guarantee (k < n/2) also stops
#: holding, since neighbourhood_size floors at 1.
LOCAL_METRIC_FLOOR = 20


def neighbourhood_size(n: int) -> int:
    """The battery's neighbourhood, chosen by rule rather than by the agent.

    scikit-learn requires `k < n / 2` for trustworthiness. The published rule is the
    largest k that satisfies it, capped at MAX_K: n=30 gives 14 against a limit of
    15.0, n=31 gives 15 against 15.5.
    """
    return max(1, min(MAX_K, math.ceil(n / 2) - 1))


@dataclass(frozen=True)
class MetricSpec:
    """How a metric is read, so that ranking never has to guess at direction or scale."""

    name: str
    higher_is_better: bool
    best: float
    worst: float
    requires_labels: bool
    measures: str
    describes: str

    def normalise(self, value: float | None) -> float | None:
        """Map a raw value onto [0, 1] using the metric's own scale, not the cohort's.

        Deliberately absolute rather than min-max across candidates. Min-max always
        awards 1.0 to the best candidate, so a field of uniformly poor embeddings would
        produce a winner that looks excellent. On an absolute scale a bad set scores
        badly, which is the honest outcome and the one the report should carry.
        """
        if value is None:
            return None
        span = self.best - self.worst
        if span == 0:
            return None
        return float(np.clip((value - self.worst) / span, 0.0, 1.0))


METRIC_SPECS: dict[str, MetricSpec] = {
    "trustworthiness": MetricSpec(
        name="trustworthiness",
        higher_is_better=True,
        best=1.0,
        worst=0.0,
        requires_labels=False,
        measures="local",
        describes="whether points that are neighbours in the embedding were also "
        "neighbours in the data; penalises false neighbours the embedding invents",
    ),
    "continuity": MetricSpec(
        name="continuity",
        higher_is_better=True,
        best=1.0,
        worst=0.0,
        requires_labels=False,
        measures="local",
        describes="whether points that were neighbours in the data are still "
        "neighbours in the embedding; penalises true neighbours the embedding tears apart",
    ),
    "shepard_correlation": MetricSpec(
        name="shepard_correlation",
        higher_is_better=True,
        best=1.0,
        worst=0.0,
        requires_labels=False,
        measures="global",
        describes="rank correlation between pairwise distances before and after; the "
        "clearest single statement of whether the layout can be read as a map",
    ),
    "knn_label_preservation": MetricSpec(
        name="knn_label_preservation",
        higher_is_better=True,
        best=1.0,
        worst=0.0,
        requires_labels=True,
        measures="local/supervised",
        describes="fraction of each point's embedded neighbours sharing its label; "
        "read against the reference value, which is the ceiling the data itself allows",
    ),
    "silhouette": MetricSpec(
        name="silhouette",
        higher_is_better=True,
        best=1.0,
        worst=-1.0,
        requires_labels=True,
        measures="global/supervised",
        describes="how separated the labelled groups are in the embedding; rewards "
        "visual separation, which is a display property rather than a fidelity one",
    ),
    "runtime_s": MetricSpec(
        name="runtime_s",
        higher_is_better=False,
        best=0.0,
        worst=0.0,  # filled in per cohort; runtime has no absolute scale
        requires_labels=False,
        measures="cost",
        describes="wall-clock seconds for the whole candidate pipeline",
    ),
}

LABEL_FREE_METRICS = tuple(
    name for name, spec in METRIC_SPECS.items() if not spec.requires_labels
)


def evaluate_embedding(
    reference: Matrix,
    embedding: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    k: int,
    seed: int = 0,
    max_samples: int = METRIC_SAMPLE_CAP,
    runtime_s: float | None = None,
) -> dict[str, Any]:
    """Score one embedding against the representation it was computed from.

    `reference` must be row-aligned with `embedding`. When a candidate subsampled, the
    caller is responsible for having subset the reference the same way — the sample
    index is recorded by the pipeline precisely so that this alignment is possible
    rather than assumed.

    `k` is required and has no default. It is derived once from the reference's row
    count — by `neighbourhood_size`, at the moment the reference is fixed — and passed
    in, rather than computed here from whatever rows this particular candidate kept.
    Deriving it per candidate is how two candidates that subsample differently end up
    measured at different neighbourhoods and ranked together anyway. A default would
    reopen that door for the next caller, and an optional override would leave it open
    on purpose, so there is neither: the caller must say which k this cohort is being
    measured at, and every member of the cohort is handed the same one.
    """
    # The reference is left in whatever storage it arrived in. Every metric below
    # reaches scikit-learn through pairwise_distances or NearestNeighbors, both of
    # which accept sparse input, and densifying a wide count matrix here purely to
    # measure it would cost hundreds of megabytes for no gain.
    embedding = np.asarray(embedding, dtype=np.float64)
    if reference.shape[0] != embedding.shape[0]:
        raise ValueError(
            f"reference has {reference.shape[0]} rows and the embedding has "
            f"{embedding.shape[0]}; they must be row-aligned for any of these metrics "
            "to mean anything. If the candidate subsampled, subset the reference by the "
            "recorded sample index first."
        )

    n_total = reference.shape[0]
    index = _subsample_index(n_total, max_samples, labels, seed)
    reference, embedding = reference[index], embedding[index]
    labels = None if labels is None else np.asarray(labels)[index]
    n_used = reference.shape[0]
    values: dict[str, float | None] = {}

    if n_used < LOCAL_METRIC_FLOOR:
        values["trustworthiness"] = None
        values["continuity"] = None
        values["shepard_correlation"] = None
    else:
        values["trustworthiness"] = float(
            trustworthiness(reference, embedding, n_neighbors=k)
        )
        # Continuity is trustworthiness with the two spaces exchanged: intrusions in one
        # direction are extrusions in the other.
        values["continuity"] = float(
            trustworthiness(embedding, reference, n_neighbors=k)
        )
        values["shepard_correlation"] = _shepard(reference, embedding)

    reference_values: dict[str, float | None] = {}
    n_classes = 0 if labels is None else int(np.unique(labels).size)

    # scikit-learn wants 1 < n_classes < n_used for silhouette, which is a separate
    # constraint from the neighbourhood: n=4 with 4 classes raises whatever k is.
    if labels is not None and 1 < n_classes < n_used:
        values["silhouette"] = float(silhouette_score(embedding, labels))
        reference_values["silhouette"] = float(silhouette_score(reference, labels))
    else:
        values["silhouette"] = None

    # The kNN agreement needs k + 1 rows to have k neighbours besides the point itself.
    if labels is not None and n_classes > 1 and k + 1 <= n_used:
        values["knn_label_preservation"] = _label_agreement(embedding, labels, k)
        reference_values["knn_label_preservation"] = _label_agreement(
            reference, labels, k
        )
    else:
        values["knn_label_preservation"] = None

    values["runtime_s"] = runtime_s

    return {
        "values": values,
        "reference_values": reference_values,
        "n_used": int(n_used),
        "n_total": int(n_total),
        "subsampled": bool(n_used < n_total),
        "seed": seed,
        "settings": {"k": k, "max_samples": max_samples, "seed": seed},
        "notes": _notes(values, reference_values, n_used, n_total),
    }


def _shepard(reference: Matrix, embedding: np.ndarray) -> float:
    """Spearman correlation of pairwise distances, over the upper triangle."""
    before = pairwise_distances(reference)
    after = pairwise_distances(embedding)
    upper = np.triu_indices_from(before, k=1)
    statistic = spearmanr(before[upper], after[upper]).statistic
    return float(statistic) if np.isfinite(statistic) else 0.0


def _label_agreement(data: Matrix, labels: np.ndarray, k: int) -> float:
    """Mean fraction of each point's k nearest neighbours that share its label."""
    _, neighbours = NearestNeighbors(n_neighbors=k + 1).fit(data).kneighbors(data)
    neighbour_labels = labels[neighbours[:, 1:]]
    return float((neighbour_labels == labels[:, None]).mean())


def _subsample_index(
    n_samples: int, cap: int, labels: np.ndarray | None, seed: int
) -> np.ndarray:
    if n_samples <= cap:
        return np.arange(n_samples)
    rng = np.random.default_rng(seed)
    if labels is None:
        return np.sort(rng.choice(n_samples, size=cap, replace=False))

    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    quota = np.maximum(1, np.floor(cap * counts / counts.sum()).astype(int))
    chosen = [
        rng.choice(np.flatnonzero(labels == cls), size=min(int(take), int(count)), replace=False)
        for cls, take, count in zip(classes, quota, counts)
    ]
    return np.sort(np.concatenate(chosen))


def _notes(
    values: dict[str, float | None],
    reference_values: dict[str, float | None],
    n_used: int,
    n_total: int,
) -> list[str]:
    """Short statements about how to read these particular numbers."""
    notes: list[str] = []

    trust, cont = values.get("trustworthiness"), values.get("continuity")
    if trust is not None and cont is not None:
        gap = trust - cont
        if gap > 0.05:
            notes.append(
                f"trustworthiness ({trust:.3f}) exceeds continuity ({cont:.3f}): the "
                "embedding keeps its neighbourhoods honest but tears apart points that "
                "were close in the data, which is the signature of a method fragmenting "
                "a connected structure."
            )
        elif gap < -0.05:
            notes.append(
                f"continuity ({cont:.3f}) exceeds trustworthiness ({trust:.3f}): the "
                "embedding keeps true neighbours together but also pulls in points that "
                "were far apart, so apparent clusters may merge distinct groups."
            )

    shepard = values.get("shepard_correlation")
    if shepard is not None and shepard < 0.5:
        notes.append(
            f"a Shepard correlation of {shepard:.3f} means distances on this plot do "
            "not track distances in the data; the layout should be read as a "
            "neighbourhood diagram, not as a map."
        )

    preserved = values.get("knn_label_preservation")
    ceiling = reference_values.get("knn_label_preservation")
    if preserved is not None and ceiling is not None:
        if ceiling > 0 and preserved <= ceiling:
            notes.append(
                f"label preservation is {preserved:.3f} against {ceiling:.3f} in the "
                f"reference representation, so the reduction kept "
                f"{preserved / ceiling:.0%} of the label structure that was there to "
                "keep. The reference value is the ceiling; read the absolute number "
                "against it rather than against 1.0."
            )
        elif ceiling > 0:
            notes.append(
                f"label preservation is {preserved:.3f}, *above* the reference value of "
                f"{ceiling:.3f}. An embedding can beat its own input here, and it is "
                "not an error: in high dimensions distances concentrate, so "
                "neighbourhoods in the reference are themselves noisy, and a reduction "
                "that discards the noisy directions can recover label structure the "
                "full-dimensional space obscured. It does mean the reference is a weak "
                "baseline for this dataset rather than a hard ceiling."
            )

    if n_used < n_total:
        notes.append(
            f"computed on {n_used:,} of {n_total:,} points, stratified where labels "
            "exist, because these metrics are quadratic in the sample count."
        )

    return notes
