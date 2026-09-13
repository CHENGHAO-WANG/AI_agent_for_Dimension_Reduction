"""The house style.

Figures are checked by their decisions rather than their pixels: which rendering regime
was chosen, which channel carries class identity, how many points the view clipped. A
pixel comparison would break on every matplotlib release and tell us nothing about
whether the figure is readable.

The two properties that matter most are the ones that were got wrong first: a view that
a couple of outliers can destroy, and labels placed outside it.
"""

from __future__ import annotations

import numpy as np
import pytest

from drtools.viz import (
    ALPHA_LIMIT,
    COLOUR_CLASS_CAP,
    MARKER_LIMIT,
    figure_class_facet,
    figure_comparison,
    figure_embedding,
    figure_metrics,
    figure_scree,
    figure_shepard,
    figure_thumbnail,
    render_mode,
    robust_limits,
    theme,
)


def blobs(n: int, k: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    centres = rng.standard_normal((k, 2)) * 8
    return centres[labels] + rng.standard_normal((n, 2)), labels


# ------------------------------------------------------------------ rendering mode


@pytest.mark.parametrize(
    "n, expected",
    [
        (200, "markers"),
        (MARKER_LIMIT, "markers"),
        (MARKER_LIMIT + 1, "points"),
        (ALPHA_LIMIT, "points"),
        (ALPHA_LIMIT + 1, "density"),
        (200_000, "density"),
    ],
)
def test_rendering_regime_follows_the_sample_count(n: int, expected: str) -> None:
    assert render_mode(n) == expected


@pytest.mark.parametrize("n", [400, 6_000, 40_000])
def test_every_regime_produces_a_figure(tmp_path, n: int) -> None:
    xy, labels = blobs(n)

    info = figure_embedding(xy, labels, None, tmp_path / f"e{n}.png")

    assert info["render_mode"] == render_mode(n)
    assert (tmp_path / f"e{n}.png").stat().st_size > 0


# -------------------------------------------------------------------- robust view


def test_a_few_extreme_points_do_not_destroy_the_view() -> None:
    """The bug this exists for: a collapsed embedding rendering as a blank panel."""
    xy = np.random.default_rng(0).standard_normal((500, 2))
    xy[:3] = [[1e6, 1e6], [-1e6, 5e5], [3e5, -1e6]]

    (x_low, x_high), _, outside = robust_limits(xy)

    assert x_high - x_low < 100, "the view is still being set by the outliers"
    assert outside >= 3


def test_clipped_points_are_counted_rather_than_hidden(tmp_path) -> None:
    xy, labels = blobs(400)
    xy[0] = [1e5, 1e5]

    info = figure_embedding(xy, labels, None, tmp_path / "clipped.png")

    assert info["points_outside_view"] >= 1


def test_a_degenerate_embedding_still_renders(tmp_path) -> None:
    """Every point identical: the span is zero and nothing may divide by it."""
    xy = np.zeros((300, 2))

    info = figure_embedding(xy, None, None, tmp_path / "degenerate.png")

    assert (tmp_path / "degenerate.png").stat().st_size > 0
    assert info["n_points"] == 300


# ---------------------------------------------------------------- identity channel


def test_three_classes_or_fewer_are_carried_by_colour(tmp_path) -> None:
    xy, labels = blobs(300, k=COLOUR_CLASS_CAP)

    info = figure_embedding(xy, labels, ["a", "b", "c"], tmp_path / "few.png")

    assert info["identity_channel"] == "colour"


def test_more_classes_than_the_palette_can_separate_switch_to_labels(tmp_path) -> None:
    """Eight hues fail the all-pairs separation floors; a printed name does not."""
    xy, labels = blobs(600, k=9)

    info = figure_embedding(xy, labels, None, tmp_path / "many.png")

    assert info["identity_channel"] == "labels"
    assert info["n_classes"] == 9


def test_unlabelled_data_gets_no_identity_channel(tmp_path) -> None:
    xy, _ = blobs(300)

    info = figure_embedding(xy, None, None, tmp_path / "bare.png")

    assert info["identity_channel"] == "none"


def test_labels_are_placed_inside_the_clipped_view(tmp_path) -> None:
    """Labels outside the axes expand the saved bounding box and wreck the layout.

    This is the regression for the day-6 mistake: positioning labels in full data space
    while the axes showed a clipped view produced a figure 4.8x too wide, with class
    names stranded in the corners of an otherwise empty canvas.
    """
    xy, labels = blobs(500, k=6)
    xy[0] = [1e6, 1e6]

    figure_embedding(xy, labels, None, tmp_path / "tight.png")

    import matplotlib.image as mpimg

    height, width = mpimg.imread(tmp_path / "tight.png").shape[:2]
    assert width < 1200 and height < 1200, f"figure blew up to {width}x{height}"


# ------------------------------------------------------------------------ figures


def test_the_comparison_panel_draws_every_candidate(tmp_path) -> None:
    xy, labels = blobs(400)
    candidates = {"pca": xy, "umap": xy[:, ::-1], "tsne": xy * 2}

    info = figure_comparison(candidates, labels, None, tmp_path / "cmp.png")

    assert set(info["panels"]) == set(candidates)


def test_comparing_nothing_is_an_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="no embeddings"):
        figure_comparison({}, None, None, tmp_path / "cmp.png")


def test_the_class_facet_gives_each_class_its_own_panel(tmp_path) -> None:
    """One series per panel is what keeps nine classes readable."""
    xy, labels = blobs(600, k=9)

    info = figure_class_facet(xy, labels, None, tmp_path / "facet.png")

    assert info["n_classes"] == 9


def test_the_metrics_figure_handles_absent_values(tmp_path) -> None:
    metrics = {
        "a": {"values": {"trustworthiness": 0.9, "silhouette": None, "runtime_s": 1.0}},
        "b": {"values": {"trustworthiness": 0.7, "silhouette": 0.3, "runtime_s": 12.0}},
    }

    info = figure_metrics(metrics, tmp_path / "metrics.png", reference_values={"trustworthiness": 0.8})

    assert "trustworthiness" in info["metrics"]


def test_negative_metric_values_render(tmp_path) -> None:
    """Silhouette is the metric that goes negative, and it did overlap its own bar."""
    metrics = {"a": {"values": {"silhouette": -0.4}}, "b": {"values": {"silhouette": 0.3}}}

    figure_metrics(metrics, tmp_path / "negative.png")

    assert (tmp_path / "negative.png").stat().st_size > 0


def test_the_scree_plot_marks_the_elbow(tmp_path) -> None:
    explained = [0.4, 0.25, 0.15, 0.08, 0.05, 0.03, 0.02, 0.01]

    figure_scree(explained, tmp_path / "scree.png", elbow=4)

    assert (tmp_path / "scree.png").stat().st_size > 0


def test_the_shepard_diagram_renders_from_distance_pairs(tmp_path) -> None:
    rng = np.random.default_rng(0)
    before = rng.random(5000)

    figure_shepard(before, before * 2 + rng.random(5000) * 0.1, tmp_path / "shep.png",
                   correlation=0.92)

    assert (tmp_path / "shep.png").stat().st_size > 0


def test_the_reconnaissance_thumbnail_is_deliberately_plain(tmp_path) -> None:
    """It exists for the agent to look at, not to tell it what to conclude."""
    xy, labels = blobs(800, k=5)

    info = figure_thumbnail(xy, labels, tmp_path / "thumb.png")

    assert info["identity_channel"] == "labels"
    assert (tmp_path / "thumb.png").stat().st_size > 0


# ------------------------------------------------------------------------- themes


@pytest.mark.parametrize("name", ["light", "dark"])
def test_both_themes_render(tmp_path, name: str) -> None:
    """Dark is a selected palette stepped for its own surface, not an inverted light one."""
    xy, labels = blobs(300, k=3)

    figure_embedding(xy, labels, None, tmp_path / f"{name}.png", theme_name=name)

    assert (tmp_path / f"{name}.png").stat().st_size > 0


def test_the_two_themes_use_different_categorical_steps() -> None:
    assert theme("light").categorical != theme("dark").categorical


def test_an_unknown_theme_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown theme"):
        theme("solarized")
