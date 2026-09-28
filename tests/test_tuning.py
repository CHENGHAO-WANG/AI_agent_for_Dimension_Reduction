"""Choosing d and tuning (sections 3.5 and 3.7, settled on day 15).

The rules are tested on curves whose answer is known, including the LLE curve section
3.7 worked through by hand. The search is tested on data whose d is known: three strong
directions and five of noise. The rest pins what the day settled: the refit recomputes a
suggestion at the rows the method is fitted on, derived seeds keep the run's own seed for
the refit, multipliers that round to one value merge, and registration refuses what
tuning chooses.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.datasets import make_swiss_roll

from drtools.executors import Context, get_executor
from drtools.metrics import BatteryScorer, evaluate_embedding
from drtools.plan import validate_plan as _validate_plan
from drtools.tuning import (
    TUNING_ROWS,
    Tuner,
    TuningSettings,
    choose_d_by_curve,
    choose_d_by_eigengap,
    derive_seed,
    diffusion_curve,
)
from plans import RECON, checkpoint_for, complete, reconnoitre

WEIGHTS = {"trustworthiness": 0.25, "continuity": 0.25, "shepard_correlation": 0.5}
RULE = {"flatness": 0.10, "fallback_share": 0.90}


@pytest.fixture(scope="module")
def three_directions():
    """Three directions carry the structure; five more carry noise a tenth their size."""
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(600, 3)) * np.array([5.0, 4.0, 3.0])
    return np.hstack([signal, rng.normal(scale=0.3, size=(600, 5))])


def tune(X, stages, *, suggest=None, settings=None, labels=None, **kwargs):
    suggest = suggest or (lambda op, n, params: {})
    tuner = Tuner(
        X, labels, stages, seed=0, weights=WEIGHTS, k=15,
        settings=settings or TuningSettings(), suggest=suggest, **kwargs,
    )
    return tuner.tune()


# -------------------------------------------------------------------- the rules


def test_section_3_7s_lle_curve_chooses_d_8_and_reports_it_capped() -> None:
    """Plain Kneedle called d = 2 the elbow; the running maximum and fallback fix it."""
    curve = dict(zip([2, 3, 4, 5, 6, 8, 10],
                     [0.630, 0.609, 0.621, 0.627, 0.629, 0.706, 0.758]))
    choice = choose_d_by_curve(curve, **RULE)
    assert choice["rule"] == "fallback"
    assert choice["d"] == 8
    assert choice["capped"] is True


def test_flatness_is_an_absolute_difference() -> None:
    """Best 0.50 and 0.42 at d = 2: flat, though 0.42 is below 90% of the best."""
    choice = choose_d_by_curve({2: 0.42, 5: 0.46, 10: 0.50}, **RULE)
    assert (choice["d"], choice["rule"]) == (2, "flat")


def test_the_fallback_is_a_share_of_the_best() -> None:
    """A curve rising steadily in log d has no interior point above its chord."""
    curve = {d: 0.2 + 0.7 * np.log(d / 2) / np.log(50) for d in (2, 5, 10, 20, 50, 100)}
    choice = choose_d_by_curve(curve, **RULE)
    assert choice["rule"] == "fallback"
    best = max(curve.values())
    assert choice["d"] == min(d for d, v in curve.items() if v >= 0.9 * best)


def test_a_curve_that_bends_has_its_elbow_chosen_and_never_at_an_end() -> None:
    curve = {2: 0.50, 3: 0.80, 4: 0.90, 5: 0.93, 6: 0.95, 8: 0.96, 10: 0.97}
    choice = choose_d_by_curve(curve, **RULE)
    assert choice["rule"] == "elbow"
    assert choice["d"] not in (2, 10)


def test_the_eigengap_takes_the_largest_gap_and_ties_go_down() -> None:
    eigenvalues = [0.99, 0.98, 0.50, 0.49, 0.30]
    assert choose_d_by_eigengap(eigenvalues, [2, 3, 4])["d"] == 2
    tied = [0.9, 0.8, 0.7, 0.6, 0.5]
    assert choose_d_by_eigengap(tied, [2, 3, 4])["d"] == 2


def test_diffusion_distance_retained_rises_to_one_and_faster_at_larger_t() -> None:
    eigenvalues = [0.9, 0.7, 0.5, 0.3, 0.1]
    at_1, at_4 = diffusion_curve(eigenvalues, 1, 5), diffusion_curve(eigenvalues, 4, 5)
    assert at_1[5] == pytest.approx(1.0) and at_4[5] == pytest.approx(1.0)
    assert at_4[1] > at_1[1]


def test_derived_seeds_are_computed_keyed_on_purpose_and_never_the_run_s() -> None:
    assert derive_seed(7, "tuning-rows") == derive_seed(7, "tuning-rows")
    assert derive_seed(7, "tuning-rows") != derive_seed(7, "tuning-fits")
    assert derive_seed(7, "tuning-fits") != 7


# ------------------------------------------------------------------- the search


def test_pca_chooses_the_three_directions_that_carry_the_structure(three_directions) -> None:
    stages, record = tune(three_directions, [{"op": "pca"}])
    assert record["chosen"]["d"] == 3
    assert stages[-1]["params"]["n_components"] == 3
    assert record["method"]["criterion"] == "explained_variance"
    assert record["rows"] == {"n_rows": 600, "of": 600, "stratified": False}


def test_a_pca_pre_step_has_its_k_chosen_by_the_same_criterion(three_directions) -> None:
    stages, record = tune(
        three_directions, [{"op": "pca"}, {"op": "umap"}],
        suggest=lambda op, n, params: {"n_neighbors": 15},
    )
    assert record["pca_k"]["k"] == 3
    assert stages[0]["params"]["n_components"] == 3
    assert stages[1]["params"]["n_components"] == 2  # a visualization method's d


def test_a_visualization_method_tunes_only_its_multiplier(three_directions) -> None:
    stages, record = tune(
        three_directions[:300], [{"op": "umap"}],
        suggest=lambda op, n, params: {"n_neighbors": 10},
    )
    assert {cell["d"] for cell in record["cells"]} == {2}
    assert sorted(cell["value"] for cell in record["cells"]) == [5, 10, 20]
    assert stages[-1]["params"]["n_neighbors"] == record["chosen"]["value"]


def test_multipliers_that_round_to_one_value_become_one_cell(three_directions) -> None:
    """A suggestion of 3 neighbours: 0.5 gives 1.5, rounding to 2, the minimum."""
    _, record = tune(
        three_directions[:300], [{"op": "isomap"}],
        suggest=lambda op, n, params: {"n_neighbors": 3},
    )
    values = [cell["value"] for cell in record["cells"]]
    assert len(values) == len(set(values))


def test_the_refit_recomputes_the_suggestion_at_the_rows_it_fits_on() -> None:
    """Tuned on 2,000 rows, refitted on the 2,500 its subsample keeps -- not on 3,000."""
    X, _ = make_swiss_roll(3000, noise=0.05, random_state=0)
    stages, record = tune(
        X, [{"op": "subsample", "params": {"n_samples": 2500}}, {"op": "isomap"}],
        suggest=lambda op, n, params: {"n_neighbors": n // 100},
    )
    assert record["method"]["n_fit"] == TUNING_ROWS
    assert record["method"]["n_fit_refit"] == 2500
    multiplier = record["chosen"]["multiplier"]
    assert stages[-1]["params"]["n_neighbors"] == round(multiplier * 25)


def test_diffusion_maps_sweeps_t_on_every_fit(three_directions) -> None:
    _, record = tune(three_directions[:300], [{"op": "diffusion_maps"}])
    assert record["chosen"]["t"] in (1, 2, 4)
    swept = {(c["multiplier"], c["t"]) for c in record["cells"] if c["feasible"]}
    assert {t for _, t in swept} == {1, 2, 4}
    assert "t_at_grid_edge" in record["chosen"]


def test_a_method_not_nested_in_d_alternates_from_reconnaissance_s_d() -> None:
    X, _ = make_swiss_roll(500, noise=0.05, random_state=0)
    _, record = tune(
        X, [{"op": "lle", "params": {"method": "modified"}}],
        suggest=lambda op, n, params: {"n_neighbors": 10},
        settings=TuningSettings(max_cycles=1), intrinsic_dimension=2.1,
    )
    steps = record["alternating"]
    assert steps["start_d"] == 2
    assert steps["outcome"] in ("converged", "stopped at the cap")
    if steps["outcome"] == "stopped at the cap":
        assert "neighbours" in steps


def test_mds_chooses_d_by_its_stress(three_directions) -> None:
    _, record = tune(
        three_directions[:200], [{"op": "mds"}],
        settings=TuningSettings(d_grid=(2, 3, 4, 5, 6)),
    )
    assert record["method"]["criterion"] == "stress"
    assert record["chosen"]["d"] == 3


def test_an_infeasible_cell_is_skipped_and_recorded_not_scored_as_zero() -> None:
    """Standard LLE needs more neighbours than d, so 3 neighbours reach only d = 2."""
    X, _ = make_swiss_roll(400, noise=0.05, random_state=0)
    _, record = tune(
        X, [{"op": "lle"}],
        suggest=lambda op, n, params: {"n_neighbors": 6},
        settings=TuningSettings(d_grid=(2, 3, 4, 5, 6)),
    )
    skipped = [c for c in record["cells"] if not c["feasible"]]
    assert skipped and all(c["score"] is None and c["reason"] for c in skipped)


def test_a_choice_at_either_end_of_the_grid_is_flagged(three_directions) -> None:
    _, record = tune(
        three_directions[:300], [{"op": "umap"}],
        suggest=lambda op, n, params: {"n_neighbors": 10},
    )
    at_end = record["chosen"]["multiplier"] in (0.5, 2.0)
    assert record["chosen"]["at_grid_edge"] is at_end


# ---------------------------------------------------------------- the pieces


def test_the_cached_scorer_equals_the_battery() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 12))
    labels = rng.integers(0, 3, 400)
    embedding = X[:, :3] + 0.1 * rng.normal(size=(400, 3))
    expected = evaluate_embedding(X, embedding, labels, k=15)["values"]
    got = BatteryScorer(X, labels, k=15).score(embedding)
    for name, value in got.items():
        assert value == pytest.approx(expected[name], abs=1e-12), name


@pytest.mark.parametrize(
    "op, kind",
    [("pca", "explained_variance"), ("kernel_pca", "kernel_variance"),
     ("isomap", "residual_variance")],
)
def test_a_curve_criterion_rises_on_a_zero_to_one_scale(three_directions, op, kind) -> None:
    context = Context(seed=0, measure_criterion=True)
    get_executor(op)(three_directions[:300], context, n_components=6)
    assert context.criterion["kind"] == kind
    curve = np.asarray(context.criterion["curve"])
    assert curve.min() >= -1e-9 and curve.max() <= 1 + 1e-9
    if op != "isomap":
        assert np.all(np.diff(curve) >= -1e-12)


def test_the_width_multiplier_scales_the_rule_s_width(three_directions) -> None:
    _, plain = get_executor("diffusion_maps")(three_directions[:200], Context(seed=0))
    _, doubled = get_executor("diffusion_maps")(
        three_directions[:200], Context(seed=0), width_multiplier=2.0
    )
    assert doubled["epsilon"] == pytest.approx(2 * plain["epsilon"])


# ---------------------------------------------------------------- registration


def _profile() -> dict:
    return {"shape": {"n_samples": 300, "n_features": 10, "storage": "dense"},
            "values": {}, "labels": {"present": False}}


def _validate(candidates, **plan_fields) -> dict:
    document = complete({
        "dataset": "d",
        "candidates": [{"id": f"c{i}", "stages": s} for i, s in enumerate(candidates)],
        "evaluation": {"weights": WEIGHTS, "justification": "declared up front"},
        **plan_fields,
    })
    return _validate_plan(document, _profile(), RECON, checkpoint=checkpoint_for(document))


def _codes(report) -> set[str]:
    return {f["code"] for f in report["findings"] if f["severity"] == "error"}


def test_a_reduction_s_d_given_by_hand_is_refused() -> None:
    report = _validate([[{"op": "pca", "params": {"n_components": 5}}]])
    assert "d_is_tuned" in _codes(report)


def test_a_pca_pre_step_s_k_given_by_hand_is_refused() -> None:
    report = _validate([[{"op": "pca"}],
                        [{"op": "pca", "params": {"n_components": 10}}, {"op": "umap"}]])
    assert "k_is_chosen" in _codes(report)


@pytest.mark.parametrize("params", [{"width_multiplier": 2.0}, {"t": 2}])
def test_what_tuning_sweeps_is_never_set_by_a_plan(params) -> None:
    report = _validate([[{"op": "pca"}], [{"op": "diffusion_maps", "params": params}]])
    assert "set_by_tuning" in _codes(report)


def test_the_default_tuning_block_registers_as_the_default() -> None:
    report = _validate([[{"op": "pca"}]])
    assert report["valid"], report["findings"]
    assert report["tuning"] == "default"


def test_a_different_grid_needs_a_departure_argued_from_evidence() -> None:
    unargued = _validate([[{"op": "pca"}]], tuning={"multipliers": [0.5, 0.71, 1, 1.41, 2]})
    assert "unexplained_tuning_departure" in _codes(unargued)
    argued = _validate([[{"op": "pca"}]], tuning={
        "multipliers": [0.5, 0.71, 1, 1.41, 2],
        "departure": {"reason": "the score was still moving at the grid's ends",
                      "evidence": ["profile.shape.n_samples"]},
    })
    assert argued["valid"], argued["findings"]
    assert argued["tuning"] == "departure"


def test_a_malformed_grid_is_refused() -> None:
    report = _validate([[{"op": "pca"}]], tuning={
        "d_grid": [5, 2], "departure": {"reason": "x", "evidence": ["profile.shape.n_samples"]},
    })
    assert "invalid_tuning" in _codes(report)


# ---------------------------------------------------------------------- the CLI


@pytest.fixture
def run_with_isomap(cli, tmp_path):
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(400, 3)) * np.array([5.0, 4.0, 3.0])
    frame = np.hstack([signal, rng.normal(scale=0.3, size=(400, 5))])
    import pandas as pd

    path = tmp_path / "data.csv"
    pd.DataFrame(frame, columns=[f"f{i}" for i in range(8)]).to_csv(path, index=False)
    runs = tmp_path / "runs"
    cli("profile", "--data", path, "--runs-root", runs, "--run-id", "r1")
    reconnoitre(cli, runs / "r1")
    document = complete({
        "dataset": "d",
        "candidates": [{"id": "pca", "stages": [{"op": "pca"}]},
                       {"id": "iso", "stages": [{"op": "isomap"}]}],
        "evaluation": {"weights": WEIGHTS, "justification": "declared up front"},
    })
    (runs / "r1" / "plan.json").write_text(json.dumps(document), encoding="utf-8")
    assert cli("validate-plan", "--run-dir", runs / "r1").code == 0
    return runs / "r1"


def test_embed_tunes_inside_the_attempt_and_records_how(cli, run_with_isomap) -> None:
    run = run_with_isomap
    result = cli("embed", "--run-dir", run, "--id", "pca", "--in-process")
    assert result.code == 0, result.stderr
    record = json.loads((run / "embeddings" / "pca.json").read_text(encoding="utf-8"))
    assert record["tuning"]["chosen"]["d"] == 3
    assert record["output_shape"] == [400, 3]
    pca = next(s for s in record["stages"] if s["op"] == "pca")
    assert pca["param_provenance"]["n_components"] == "tuned"
    # The tuning fits drew their own stream; the refit ran under the run's seed.
    seeds = record["tuning"]["seeds"]
    assert seeds["tuning_fits"] != seeds["run"]


def test_the_tuning_block_is_frozen_at_registration(cli, run_with_isomap) -> None:
    run = run_with_isomap
    document = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    document["tuning"] = {
        "multipliers": [0.5, 1, 2, 4],
        "departure": {"reason": "wider", "evidence": ["profile.shape.n_samples"]},
    }
    (run / "plan.json").write_text(json.dumps(document), encoding="utf-8")
    result = cli("validate-plan", "--run-dir", run)
    assert result.code == 2
    assert "tuning block" in result.stderr
