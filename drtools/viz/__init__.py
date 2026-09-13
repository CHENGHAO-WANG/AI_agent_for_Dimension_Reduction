"""The house style for every figure the agent produces.

The agent chooses *which* figures to draw. It never chooses how they look. One palette,
one set of rules, applied identically to both datasets, so that the same colour means
the same thing in every panel and two figures can be compared by eye.

Two decisions here are worth knowing before reading the code.

**Identity is carried by labels, not by hue, once there are more than three classes.**
The categorical palette was validated with the colour checker rather than by eye, and
for scatter-like forms — where every pair of colours can end up adjacent — only three
slots clear the separation floors. At eight slots the worst pair measures Delta E 7.1 to
normal vision and 3.2 under simulated protanopia: red and orange that nobody can reliably
tell apart. Both target datasets have eight or nine classes. So above three classes the
combined views draw points in a single hue and name each class at its centroid, and the
class facet — one panel per class, one series per panel — is the figure that answers
"which class is where". The convention in single-cell work is a nine-colour scatter; the
convention is not readable, and a label is.

**Rendering adapts to the sample count, because one mark style cannot span the range.**
A 28-point marker with a surface ring is right for six hundred points and is a solid
blob at a hundred thousand. Three regimes: markers with rings, then small translucent
rasterised points, then a density field with an optional thinned overlay. The mode is
chosen from n and recorded in the figure's metadata, so a reader knows whether they are
looking at points or at a histogram.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # figures are files, never windows

import matplotlib.patheffects as patheffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# Categorical slots in fixed order. Assigned by position, never cycled: the fourth
# class does not get slot 1 again, it goes to the facet.
CATEGORICAL_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a")
CATEGORICAL_DARK = ("#3987e5", "#d95926", "#199e70")

# Single hue, light to dark, for magnitude: density fields, Shepard diagrams.
SEQUENTIAL = (
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b",
)

#: Above this many classes, hue stops carrying identity and labels take over.
COLOUR_CLASS_CAP = 3

#: Sample-count thresholds between rendering regimes.
MARKER_LIMIT = 2_000
ALPHA_LIMIT = 20_000

#: Points drawn over a density field, at most.
OVERLAY_CAP = 4_000


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    primary_ink: str
    secondary_ink: str
    muted: str
    grid: str
    axis: str
    context: str
    categorical: tuple[str, ...]

    @property
    def sequential_cmap(self) -> LinearSegmentedColormap:
        steps = SEQUENTIAL if self.name == "light" else SEQUENTIAL[::-1]
        return LinearSegmentedColormap.from_list(f"drtools_{self.name}", steps)


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    primary_ink="#0b0b0b",
    secondary_ink="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    context="#d8d7d0",
    categorical=CATEGORICAL_LIGHT,
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    primary_ink="#ffffff",
    secondary_ink="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    context="#3a3a37",
    categorical=CATEGORICAL_DARK,
)

THEMES = {"light": LIGHT, "dark": DARK}


def theme(name: str = "light") -> Theme:
    try:
        return THEMES[name]
    except KeyError:
        raise ValueError(f"unknown theme {name!r}; choose from {sorted(THEMES)}") from None


def apply_style(active: Theme) -> None:
    """Chrome recedes; data does not. Applied once per figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": active.surface,
            "axes.facecolor": active.surface,
            "savefig.facecolor": active.surface,
            "text.color": active.primary_ink,
            "axes.labelcolor": active.secondary_ink,
            "axes.edgecolor": active.axis,
            "xtick.color": active.muted,
            "ytick.color": active.muted,
            "axes.titlecolor": active.primary_ink,
            "grid.color": active.grid,
            "grid.linewidth": 0.6,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "legend.frameon": False,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
        }
    )


# ------------------------------------------------------------------ rendering mode


def render_mode(n_samples: int) -> str:
    """Which mark style suits this many points.

    `markers` for a few hundred, where every point is individually visible and a ring
    keeps overlapping ones separable. `points` in the thousands, where rings merge into
    a smear and translucency carries density instead. `density` past twenty thousand,
    where individual marks are meaningless and a binned field is what the eye can read.
    """
    if n_samples <= MARKER_LIMIT:
        return "markers"
    if n_samples <= ALPHA_LIMIT:
        return "points"
    return "density"


def _scatter(ax, xy: np.ndarray, colour: str, active: Theme, mode: str, *, zorder=2):
    if mode == "markers":
        ax.scatter(
            xy[:, 0], xy[:, 1], s=26, c=colour, linewidths=0.6,
            edgecolors=active.surface, zorder=zorder,
        )
    else:
        ax.scatter(
            xy[:, 0], xy[:, 1], s=7, c=colour, alpha=0.5, linewidths=0,
            rasterized=True, zorder=zorder,
        )


def _density(ax, xy: np.ndarray, active: Theme, gridsize: int = 90):
    ax.hexbin(
        xy[:, 0], xy[:, 1], gridsize=gridsize, cmap=active.sequential_cmap,
        mincnt=1, linewidths=0, rasterized=True, zorder=1,
    )


def _thin(xy: np.ndarray, labels: np.ndarray | None, seed: int = 0):
    if xy.shape[0] <= OVERLAY_CAP:
        return xy, labels
    rng = np.random.default_rng(seed)
    index = rng.choice(xy.shape[0], size=OVERLAY_CAP, replace=False)
    return xy[index], (None if labels is None else labels[index])


def robust_limits(xy: np.ndarray, *, quantile: float = 0.5, pad: float = 0.06):
    """View bounds that a handful of far-flung points cannot destroy.

    Not cosmetic. Several methods place almost every point in a tight cluster and a
    couple of points decades away — a collapsed diffusion map is the usual culprit — and
    on full extent that renders as an empty panel with two dots, which reads as a broken
    figure rather than as the informative result it is. Clipping to a high quantile shows
    the structure, and the count of points left outside is reported on the panel so
    nothing is quietly cropped.
    """
    low = np.percentile(xy, quantile, axis=0)
    high = np.percentile(xy, 100 - quantile, axis=0)
    span = np.where(high - low > 0, high - low, 1.0)
    low, high = low - pad * span, high + pad * span
    outside = int(np.sum(np.any((xy < low) | (xy > high), axis=1)))
    return (low[0], high[0]), (low[1], high[1]), outside


def _separate(points: np.ndarray, span: np.ndarray, min_gap: float = 0.11) -> np.ndarray:
    """Nudge coincident labels apart, in units of the axis span.

    Class centroids collide whenever two classes overlap, and two labels printed on top
    of each other name nothing. A few rounds of pairwise repulsion suffice: a label only
    has to sit near its class, not on its exact centre.
    """
    scaled = points.astype(float) / span
    for _ in range(60):
        shifted = False
        for i in range(len(scaled)):
            for j in range(i + 1, len(scaled)):
                delta = scaled[i] - scaled[j]
                distance = float(np.hypot(*delta))
                if distance < min_gap:
                    direction = delta / distance if distance > 1e-9 else np.array([1.0, 0.0])
                    push = (min_gap - distance) / 2
                    scaled[i] += direction * push
                    scaled[j] -= direction * push
                    shifted = True
        if not shifted:
            break
    return scaled * span


def _label_centroids(ax, xy, labels, names, active: Theme, xlim, ylim) -> None:
    """Name each class where its points are. This is the identity channel.

    Positions are computed in the *clipped* view rather than in full data space, and
    clamped back inside it. Both matter: a class whose median sits among far-flung
    outliers would otherwise be labelled off-panel, and the repulsion step measured in
    full-span units would barely move labels apart when one stray point sets the scale.
    """
    classes = np.unique(labels)
    centres = np.array([np.median(xy[labels == code], axis=0) for code in classes])
    low = np.array([xlim[0], ylim[0]])
    high = np.array([xlim[1], ylim[1]])
    span = np.where(high - low > 0, high - low, 1.0)

    placed = _separate(np.clip(centres, low, high), span)
    # Keep a margin so a label sits fully within the panel rather than half over its edge.
    placed = np.clip(placed, low + 0.04 * span, high - 0.04 * span)

    for code, centre in zip(classes, placed):
        text = names[code] if names is not None and code < len(names) else str(code)
        ax.text(
            centre[0], centre[1], text,
            color=active.primary_ink, fontsize=8, fontweight="bold",
            ha="center", va="center", zorder=5, clip_on=True,
            path_effects=[patheffects.withStroke(linewidth=2.6, foreground=active.surface)],
        )


def draw_embedding(
    ax,
    xy: np.ndarray,
    labels: np.ndarray | None,
    names: Sequence[str] | None,
    active: Theme,
    *,
    mode: str | None = None,
    annotate: bool = True,
) -> dict[str, Any]:
    """Draw one embedding onto an axis, choosing marks and identity channel by size."""
    xy = np.asarray(xy, dtype=float)[:, :2]
    mode = mode or render_mode(xy.shape[0])
    n_classes = 0 if labels is None else int(np.unique(labels).size)
    used_colour = False

    if labels is not None and 0 < n_classes <= COLOUR_CLASS_CAP:
        used_colour = True
        for position, code in enumerate(np.unique(labels)):
            name = names[code] if names is not None and code < len(names) else str(code)
            member = xy[labels == code]
            if mode == "density":
                _density(ax, member, active)
            else:
                _scatter(ax, member, active.categorical[position], active, mode)
            ax.scatter([], [], s=26, c=active.categorical[position], label=name)
    elif mode == "density":
        _density(ax, xy, active)
        overlay, overlay_labels = _thin(xy, labels)
        ax.scatter(
            overlay[:, 0], overlay[:, 1], s=3, c=active.primary_ink, alpha=0.18,
            linewidths=0, rasterized=True, zorder=2,
        )
    else:
        _scatter(ax, xy, active.categorical[0], active, mode)

    # Limits before labels: the labels are positioned within the view, so the view has
    # to exist first.
    xlim, ylim, outside = robust_limits(xy)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    if labels is not None and annotate and not used_colour:
        _label_centroids(ax, xy, labels, names, active, xlim, ylim)

    if outside:
        ax.text(
            0.99, 0.01, f"{outside} outside view", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color=active.muted, zorder=6,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return {
        "render_mode": mode,
        "n_points": int(xy.shape[0]),
        "n_classes": n_classes,
        "identity_channel": "colour" if used_colour else ("labels" if labels is not None else "none"),
        "points_outside_view": outside,
    }


# ----------------------------------------------------------------------- figures


def _finish(fig: Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_embedding(
    xy: np.ndarray,
    labels: np.ndarray | None,
    names: Sequence[str] | None,
    path: Path,
    *,
    title: str = "",
    subtitle: str = "",
    theme_name: str = "light",
) -> dict[str, Any]:
    """One candidate, on its own."""
    active = theme(theme_name)
    apply_style(active)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    info = draw_embedding(ax, xy, labels, names, active)

    if title:
        ax.set_title(title, loc="left", pad=10)
    if subtitle:
        ax.text(
            0, 1.015, subtitle, transform=ax.transAxes,
            color=active.secondary_ink, fontsize=8, va="bottom",
        )
    if info["identity_channel"] == "colour":
        ax.legend(loc="upper right", fontsize=8, labelcolor=active.secondary_ink)
    return {**info, "path": str(_finish(fig, path))}


def figure_comparison(
    embeddings: dict[str, np.ndarray],
    labels: np.ndarray | None,
    names: Sequence[str] | None,
    path: Path,
    *,
    title: str = "Candidates on the same points",
    theme_name: str = "light",
) -> dict[str, Any]:
    """The headline: every candidate, same points, same rules, one row.

    Panels share nothing but the data — each embedding has its own arbitrary scale, so
    axes are not shared and no axis is drawn. What is comparable is the arrangement.
    """
    if not embeddings:
        raise ValueError("no embeddings to compare")

    active = theme(theme_name)
    apply_style(active)
    n = len(embeddings)
    columns = min(n, 4)
    rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(3.3 * columns, 3.5 * rows), squeeze=False
    )

    panels = {}
    for axis, (candidate_id, xy) in zip(axes.ravel(), embeddings.items()):
        panels[candidate_id] = draw_embedding(axis, xy, labels, names, active)
        axis.set_title(candidate_id, loc="left", pad=6, fontsize=9)
    for axis in axes.ravel()[n:]:
        axis.set_visible(False)

    fig.suptitle(title, x=0.008, ha="left", fontsize=11, color=active.primary_ink)
    first = next(iter(panels.values()))
    if first["identity_channel"] == "colour":
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, legend_labels, loc="lower center", ncol=min(len(legend_labels), 6),
            fontsize=8, labelcolor=active.secondary_ink, bbox_to_anchor=(0.5, -0.02),
        )
    fig.tight_layout()
    return {"panels": panels, "path": str(_finish(fig, path))}


def figure_class_facet(
    xy: np.ndarray,
    labels: np.ndarray,
    names: Sequence[str] | None,
    path: Path,
    *,
    title: str = "",
    theme_name: str = "light",
) -> dict[str, Any]:
    """One panel per class, each against the rest in grey.

    This is the figure that answers which class sits where. One series per panel means
    the colour question does not arise, and it stays readable at nine classes where a
    nine-colour scatter does not.
    """
    active = theme(theme_name)
    apply_style(active)
    labels = np.asarray(labels)
    xy = np.asarray(xy, dtype=float)[:, :2]
    classes = np.unique(labels)

    columns = min(len(classes), 4)
    rows = int(np.ceil(len(classes) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(2.6 * columns, 2.8 * rows), squeeze=False,
        sharex=True, sharey=True,
    )

    context, _ = _thin(xy, None)
    mode = render_mode(xy.shape[0])
    for axis, code in zip(axes.ravel(), classes):
        axis.scatter(
            context[:, 0], context[:, 1], s=4, c=active.context, linewidths=0,
            rasterized=True, zorder=1,
        )
        member = xy[labels == code]
        _scatter(axis, member, active.categorical[0], active, render_mode(member.shape[0]), zorder=3)
        name = names[code] if names is not None and code < len(names) else str(code)
        axis.set_title(f"{name}  ({member.shape[0]:,})", loc="left", fontsize=8, pad=4)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    for axis in axes.ravel()[len(classes):]:
        axis.set_visible(False)

    if title:
        fig.suptitle(title, x=0.008, ha="left", fontsize=11, color=active.primary_ink)
    fig.tight_layout()
    return {
        "n_classes": int(classes.size),
        "render_mode": mode,
        "path": str(_finish(fig, path)),
    }


def figure_metrics(
    metrics_by_id: dict[str, dict[str, Any]],
    path: Path,
    *,
    reference_values: dict[str, float] | None = None,
    title: str = "Evaluation battery",
    theme_name: str = "light",
) -> dict[str, Any]:
    """One panel per metric, candidates as bars.

    Small multiples rather than grouped bars on one axis: the metrics do not share a
    scale, and putting a bounded correlation beside a wall-clock time on one axis would
    be a lie about their comparability.
    """
    active = theme(theme_name)
    apply_style(active)

    names = [
        name
        for name in next(iter(metrics_by_id.values()))["values"]
        if any(m["values"].get(name) is not None for m in metrics_by_id.values())
    ]
    candidates = list(metrics_by_id)

    columns = min(len(names), 3)
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 2.5 * rows), squeeze=False)

    for axis, metric in zip(axes.ravel(), names):
        values = [metrics_by_id[c]["values"].get(metric) for c in candidates]
        drawn = [0.0 if v is None else float(v) for v in values]
        positions = np.arange(len(candidates))
        axis.barh(positions, drawn, height=0.62, color=active.categorical[0], zorder=2)
        axis.set_yticks(positions)
        axis.set_yticklabels(candidates, fontsize=8)
        axis.invert_yaxis()
        axis.set_title(metric.replace("_", " "), loc="left", fontsize=9, pad=6)
        axis.grid(axis="x", zorder=0)
        axis.set_axisbelow(True)

        for position, (value, shown) in enumerate(zip(values, drawn)):
            text = "n/a" if value is None else f"{value:,.3f}"
            # A label to the right of a negative bar lands on top of the bar.
            negative = shown < 0
            axis.text(
                shown, position, f"{text}  " if negative else f"  {text}",
                va="center", ha="right" if negative else "left",
                fontsize=7.5, color=active.secondary_ink,
            )

        ceiling = (reference_values or {}).get(metric)
        if ceiling is not None:
            axis.axvline(
                ceiling, color=active.muted, linewidth=1.2, linestyle=(0, (4, 3)), zorder=3
            )
            axis.text(
                ceiling, len(candidates) - 0.35, " reference", fontsize=7,
                color=active.muted, va="top",
            )
        axis.margins(x=0.22)

    for axis in axes.ravel()[len(names):]:
        axis.set_visible(False)

    fig.suptitle(title, x=0.008, ha="left", fontsize=11, color=active.primary_ink)
    fig.tight_layout()
    return {"metrics": names, "path": str(_finish(fig, path))}


def figure_shepard(
    reference_distances: np.ndarray,
    embedding_distances: np.ndarray,
    path: Path,
    *,
    correlation: float | None = None,
    title: str = "Shepard diagram",
    theme_name: str = "light",
) -> dict[str, Any]:
    """Distance before against distance after, as a density.

    Pair counts are quadratic, so this is always a binned field rather than points. A
    tight band along a rising line means the layout can be read as a map; a cloud means
    only neighbourhood membership survived.
    """
    active = theme(theme_name)
    apply_style(active)
    fig, ax = plt.subplots(figsize=(4.4, 4.2))

    ax.hexbin(
        reference_distances, embedding_distances, gridsize=70,
        cmap=active.sequential_cmap, mincnt=1, linewidths=0, rasterized=True, zorder=2,
    )
    ax.set_xlabel("distance in the reference")
    ax.set_ylabel("distance in the embedding")
    ax.set_title(title, loc="left", pad=10)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)

    if correlation is not None:
        ax.text(
            0.02, 0.97, f"Spearman {correlation:.3f}", transform=ax.transAxes,
            fontsize=9, va="top", color=active.secondary_ink,
        )
    return {"path": str(_finish(fig, path))}


def figure_scree(
    explained: Sequence[float],
    path: Path,
    *,
    elbow: int | None = None,
    title: str = "Explained variance",
    theme_name: str = "light",
) -> dict[str, Any]:
    """Per-component and cumulative variance on one axis.

    Both are fractions of the same total, so they belong on the same scale. A second
    y-axis here would invent a comparison that does not exist.
    """
    active = theme(theme_name)
    apply_style(active)
    explained = np.asarray(explained, dtype=float)
    components = np.arange(1, explained.size + 1)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(components, explained, color=active.categorical[0], width=0.72, zorder=2,
           label="per component")
    ax.plot(components, np.cumsum(explained), color=active.categorical[1], linewidth=2,
            zorder=3, label="cumulative")

    if elbow is not None:
        ax.axvline(elbow, color=active.muted, linewidth=1.2, linestyle=(0, (4, 3)), zorder=4)
        ax.text(elbow, 1.02, f" elbow at {elbow}", fontsize=8, color=active.muted,
                transform=ax.get_xaxis_transform(), va="bottom")

    ax.set_xlabel("component")
    ax.set_ylabel("fraction of variance")
    ax.set_title(title, loc="left", pad=10)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, labelcolor=active.secondary_ink)
    return {"path": str(_finish(fig, path))}


def figure_thumbnail(
    xy: np.ndarray,
    labels: np.ndarray | None,
    path: Path,
    *,
    theme_name: str = "light",
) -> dict[str, Any]:
    """A small, plain view of the reconnaissance embedding.

    Produced so the agent can *look* at the data before planning, not only read
    statistics about it. Deliberately unadorned: no title, no legend, nothing that would
    tell the agent what to conclude.
    """
    active = theme(theme_name)
    apply_style(active)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    info = draw_embedding(ax, xy, labels, None, active, annotate=False)
    fig.tight_layout(pad=0.4)
    return {**info, "path": str(_finish(fig, path))}
