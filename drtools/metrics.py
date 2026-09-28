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

import hashlib
import math
import zlib
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

#: The grouped jackknife's number of groups (section 3.7). Every candidate's scored
#: rows are split into the same groups, so the replicates are paired across candidates.
JACKKNIFE_GROUPS = 10


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
    # Section 2.4: runtime is measured and reported for every candidate, and never
    # weighted. By ranking time the cost is paid, it varies on replay, and it rewards
    # a candidate for looking at less data.
    weightable: bool = True

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
        worst=0.0,  # no absolute scale, which is one reason it carries no weight
        requires_labels=False,
        measures="cost",
        describes="wall-clock seconds for the whole candidate pipeline; reported, "
        "never weighted",
        weightable=False,
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

    `reference` must be row-aligned with `embedding`. Every candidate covers every row
    (section 3.12), so the scored rows are drawn from the same row count under the same
    seed and labels for every candidate, and the comparison is paired. `scored_rows`
    is a digest of the rows drawn, so that sameness can be checked rather than assumed.

    `k` is required and has no default. It is derived once from the reference's row
    count — by `neighbourhood_size`, at the moment the reference is fixed — and passed
    in, rather than computed here from whatever rows this particular candidate kept.
    Deriving it per candidate is how two candidates with different row counts ended up
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
            "to mean anything. Every candidate's embedding covers every row of the "
            "reference, in its order."
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
        "d": int(embedding.shape[1]),
        "reference_values": reference_values,
        "n_used": int(n_used),
        "n_total": int(n_total),
        "subsampled": bool(n_used < n_total),
        "scored_rows": scored_rows_digest(index),
        "seed": seed,
        "settings": {"k": k, "max_samples": max_samples, "seed": seed},
        "notes": _notes(values, reference_values, n_used, n_total),
        "jackknife": _jackknife(reference, embedding, labels, k=k, seed=seed),
    }


def derive_seed(seed: int, purpose: str) -> int:
    """A seed for one purpose, computed from the run's seed and never chosen.

    Keyed on the purpose rather than the candidate, so every candidate draws the same
    tuning rows, the same fitting stream and the same jackknife groups -- common random
    numbers -- while the run's own seed keeps its meaning for every refit and the final
    battery (section 3.5).
    """
    sequence = np.random.SeedSequence(
        entropy=int(seed), spawn_key=(zlib.crc32(purpose.encode("utf-8")),)
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def jackknife_groups(
    n: int, labels: np.ndarray | None, seed: int, n_groups: int = JACKKNIFE_GROUPS
) -> np.ndarray:
    """Each scored row's jackknife group, 0 to `n_groups - 1`.

    Drawn from a stream derived from the run's seed, so the assignment depends only on
    the seed, the number of scored rows and their labels, and every candidate gets the
    same groups. With labels, each class is shuffled and dealt round the groups in
    turn, the count carrying on from one class to the next, so every group holds each
    class in proportion and the group sizes differ by at most one.
    """
    rng = np.random.default_rng(derive_seed(seed, "jackknife"))
    if labels is None:
        order = rng.permutation(n)
    else:
        labels = np.asarray(labels)
        order = np.concatenate(
            [rng.permutation(np.flatnonzero(labels == cls)) for cls in np.unique(labels)]
        )
    groups = np.empty(n, dtype=np.int64)
    groups[order] = np.arange(n) % n_groups
    return groups


def _jackknife(
    reference: Matrix,
    embedding: np.ndarray,
    labels: np.ndarray | None,
    *,
    k: int,
    seed: int,
) -> dict[str, Any]:
    """The battery on the scored rows with one group left out, once per group.

    Each replicate recomputes every metric on the rows that remain, neighbourhoods
    included, at the cohort's k. The fitted Embedding is held fixed, so the replicates
    measure only which rows were scored. `rank` turns them into standard errors under
    the weights it applies, which is why the weighting is not needed here.
    """
    n = int(reference.shape[0])
    record: dict[str, Any] = {"n_groups": JACKKNIFE_GROUPS, "groups": None,
                              "replicates": None}
    if n - math.ceil(n / JACKKNIFE_GROUPS) < LOCAL_METRIC_FLOOR:
        record["unavailable"] = (
            f"{n} scored rows leave fewer than {LOCAL_METRIC_FLOOR} once a group of "
            f"{JACKKNIFE_GROUPS} is left out, below the battery's floor"
        )
        return record
    groups = jackknife_groups(n, labels, seed)
    wanted = {name for name, spec in METRIC_SPECS.items() if spec.weightable}
    replicates = []
    for group in range(JACKKNIFE_GROUPS):
        kept = np.flatnonzero(groups != group)
        scorer = BatteryScorer(
            reference[kept], None if labels is None else labels[kept], k=k, wanted=wanted
        )
        replicates.append(scorer.score(embedding[kept]))
    record["groups"] = scored_rows_digest(groups)
    record["replicates"] = replicates
    return record


def scored_rows_digest(index: np.ndarray) -> str:
    """A short digest of which rows were scored, equal across a paired comparison."""
    return hashlib.sha256(np.asarray(index, dtype=np.int64).tobytes()).hexdigest()[:16]


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


class BatteryScorer:
    """The battery on one fixed set of rows, for scoring many embeddings of them.

    Tuning scores every cell of its grid on the same tuning rows against the same
    Reference rows (section 3.5), so everything computed from the Reference alone is
    computed once: its distance ranks for trustworthiness, its neighbours for
    continuity, the ranks of its pairwise distances for the Shepard correlation. Each
    metric follows `evaluate_embedding`'s definition exactly, and a test holds the two
    equal; trustworthiness is scikit-learn's computation with its reference half kept.
    Only the metrics named in `wanted` are computed.
    """

    def __init__(
        self,
        reference: Matrix,
        labels: np.ndarray | None,
        *,
        k: int,
        wanted: set[str] | None = None,
    ) -> None:
        from scipy.stats import rankdata

        self.n = int(reference.shape[0])
        self.k = int(k)
        self.labels = None if labels is None else np.asarray(labels)
        self.wanted = set(wanted) if wanted is not None else set(METRIC_SPECS)
        self.local = self.n >= LOCAL_METRIC_FLOOR
        distances = pairwise_distances(reference)
        self.upper = np.triu_indices(self.n, k=1)
        if self.local and self.wanted & {"trustworthiness", "continuity"}:
            self.reference_ranks = _inverted_ranks(distances)
            self.reference_neighbours = _neighbour_indices(distances, self.k)
        if self.local and "shepard_correlation" in self.wanted:
            ranked = rankdata(distances[self.upper])
            self.reference_order = (ranked - ranked.mean()) / np.linalg.norm(
                ranked - ranked.mean()
            )

    def score(self, embedding: np.ndarray) -> dict[str, float | None]:
        from scipy.stats import rankdata

        embedding = np.asarray(embedding, dtype=np.float64)
        values: dict[str, float | None] = {}
        distances = None
        if self.local and self.wanted & {
            "trustworthiness", "continuity", "shepard_correlation"
        }:
            distances = pairwise_distances(embedding)
        if "trustworthiness" in self.wanted:
            values["trustworthiness"] = (
                _trust(self.reference_ranks, _neighbour_indices(distances, self.k), self.k)
                if self.local
                else None
            )
        if "continuity" in self.wanted:
            values["continuity"] = (
                _trust(_inverted_ranks(distances), self.reference_neighbours, self.k)
                if self.local
                else None
            )
        if "shepard_correlation" in self.wanted:
            if self.local:
                ranked = rankdata(distances[self.upper])
                centred = ranked - ranked.mean()
                norm = np.linalg.norm(centred)
                values["shepard_correlation"] = (
                    float(centred @ self.reference_order / norm) if norm > 0 else 0.0
                )
            else:
                values["shepard_correlation"] = None
        n_classes = 0 if self.labels is None else int(np.unique(self.labels).size)
        if "silhouette" in self.wanted:
            values["silhouette"] = (
                float(silhouette_score(embedding, self.labels))
                if self.labels is not None and 1 < n_classes < self.n
                else None
            )
        if "knn_label_preservation" in self.wanted:
            values["knn_label_preservation"] = (
                _label_agreement(embedding, self.labels, self.k)
                if self.labels is not None and n_classes > 1 and self.k + 1 <= self.n
                else None
            )
        return values


def _inverted_ranks(distances: np.ndarray) -> np.ndarray:
    """rank[i, j]: j's position among i's neighbours, 1 for the nearest, self excluded."""
    n = distances.shape[0]
    masked = distances.copy()
    np.fill_diagonal(masked, np.inf)
    order = np.argsort(masked, axis=1)
    ranks = np.zeros((n, n), dtype=np.int64)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(1, n + 1)[None, :]
    return ranks


def _neighbour_indices(distances: np.ndarray, k: int) -> np.ndarray:
    """Each row's k nearest other rows, as NearestNeighbors returns them."""
    masked = distances.copy()
    np.fill_diagonal(masked, np.inf)
    return np.argsort(masked, axis=1, kind="stable")[:, :k]


def _trust(ranks: np.ndarray, neighbours: np.ndarray, k: int) -> float:
    """scikit-learn's trustworthiness, given one space's ranks and the other's neighbours."""
    n = ranks.shape[0]
    excess = ranks[np.arange(n)[:, None], neighbours] - k
    penalty = float(excess[excess > 0].sum())
    return 1.0 - penalty * (2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0)))
