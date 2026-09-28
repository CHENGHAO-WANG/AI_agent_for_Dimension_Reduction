"""Day 16: comparing candidates (sections 3.7 and 3.8; defects 6, 7 and 8).

The non-inferiority margin and the fewest dimensions inside it; close competitors in
the data model and the decision log; the path of winners and its sentences; the
jackknife standard error, reported and never used to choose; ranking notes that carry
a kind.
"""

import json

import numpy as np
import pytest
from plans import complete, reconnoitre

from drtools.metrics import (
    JACKKNIFE_GROUPS,
    METRIC_SPECS,
    evaluate_embedding,
    jackknife_groups,
)
from drtools.rank import (
    COMPETITOR_CAP,
    NOTE_KINDS,
    _jackknife_se,
    rank_candidates,
)
from drtools.report import LIMITATION_KINDS

ONE = {"trustworthiness": 1.0}


def scored(d, score, replicates=None):
    record = {"d": d, "values": {**{k: None for k in METRIC_SPECS},
                                 "trustworthiness": score}}
    if replicates is not None:
        record["jackknife"] = {
            "n_groups": len(replicates), "groups": "same",
            "replicates": [{"trustworthiness": value} for value in replicates],
        }
    return record


#: Question 1's example: A leads, and C has the fewest dimensions within 0.02 of it.
EXAMPLE = {
    "A": scored(10, 0.850), "D": scored(6, 0.845), "B": scored(4, 0.838),
    "C": scored(2, 0.832), "E": scored(3, 0.800),
}


def ids(result):
    return [row["id"] for row in result["ranking"]]


# ------------------------------------------------------------------ the margin


def test_the_fewest_dimensions_within_the_margin_win():
    result = rank_candidates(EXAMPLE, ONE)

    assert result["leader"] == "A"
    assert result["winner"] == "C"


def test_the_close_set_is_ordered_by_the_rule_and_the_rest_by_score():
    """Rank k is who the rule would choose were ranks 1 to k-1 withdrawn."""
    result = rank_candidates(EXAMPLE, ONE)

    assert ids(result) == ["C", "B", "D", "A", "E"]
    assert [c["id"] for c in result["close_competitors"]] == ["B", "D", "A"]
    flags = {row["id"]: row["close_competitor"] for row in result["ranking"]}
    assert flags == {"C": False, "B": True, "D": True, "A": True, "E": False}


def test_each_competitor_carries_its_signed_difference_from_the_winner():
    result = rank_candidates(EXAMPLE, ONE)

    differences = {c["id"]: c["difference_from_winner"] for c in result["close_competitors"]}
    assert differences == {"B": 0.006, "D": 0.013, "A": 0.018}


def test_a_candidate_exactly_at_the_margin_is_inside_it():
    result = rank_candidates({"A": scored(10, 0.85), "B": scored(3, 0.83)}, ONE)

    assert result["winner"] == "B"


def test_a_candidate_just_outside_the_margin_is_not():
    result = rank_candidates({"A": scored(10, 0.85), "B": scored(3, 0.8299)}, ONE)

    assert result["winner"] == "A"
    assert result["close_competitors"] == []


def test_a_d_tie_inside_the_margin_goes_to_the_higher_score():
    result = rank_candidates(
        {"A": scored(10, 0.85), "B": scored(3, 0.84), "C": scored(3, 0.835)}, ONE
    )

    assert ids(result)[:2] == ["B", "C"]


def test_a_zero_margin_ranks_by_score_and_breaks_only_exact_ties_by_d():
    result = rank_candidates(
        {"A": scored(10, 0.85), "B": scored(3, 0.85), "C": scored(2, 0.849)},
        ONE, margin=0.0,
    )

    assert ids(result) == ["B", "A", "C"]


def test_membership_walks_the_whole_list_not_the_top_two():
    """Defect 6: with 0.800, 0.790 and 0.785 the third sits within 0.02 as well."""
    result = rank_candidates(
        {"a": scored(2, 0.800), "b": scored(2, 0.790), "c": scored(2, 0.785)}, ONE
    )

    assert [c["id"] for c in result["close_competitors"]] == ["b", "c"]


def test_above_the_cap_the_set_is_recorded_whole_and_the_note_changes():
    cohort = {f"m{d}": scored(d, 0.850 - 0.001 * d) for d in (2, 3, 4, 5, 6)}

    result = rank_candidates(cohort, ONE)

    assert len(result["close_competitors"]) == 4 > COMPETITOR_CAP
    assert result["discriminated"] is False
    kinds = [note["kind"] for note in result["notes"]]
    assert "not_discriminated" in kinds and "close_competitors" not in kinds


# ---------------------------------------------------------- the path of winners


def test_the_path_follows_the_upper_hull():
    result = rank_candidates(EXAMPLE, ONE)

    path = [(step["id"], step["from_rate"], step["to_rate"]) for step in result["path"]]
    assert path == [("A", 0.0, 0.00125), ("D", 0.00125, 0.00325), ("C", 0.00325, None)]
    assert result["wins_at_no_rate"] == ["B", "E"]


def test_the_path_sentences_name_each_breakpoint_and_the_margin_rules_rate():
    sentences = rank_candidates(EXAMPLE, ONE)["path_sentences"]

    assert sentences[0] == "With no value placed on a dimension, A (d = 10) scores highest."
    assert "worth more than 0.0013 of score, D (d = 6) would win" in sentences[1]
    assert sentences[2] == "B and E win at no rate."
    assert sentences[3].startswith("The margin rule chose C, which wins whenever")


def test_a_margin_winner_off_the_hull_is_said_to_win_at_no_rate():
    """C misses the margin by 0.0010, so the rule picks B, which no single rate picks."""
    result = rank_candidates(
        {"A": scored(10, 0.850), "B": scored(4, 0.8305), "C": scored(2, 0.829)}, ONE
    )

    assert result["winner"] == "B"
    assert "B" in result["wins_at_no_rate"]
    assert result["path_sentences"][-1] == (
        "The margin rule chose B, which wins at no rate: at every value of a "
        "dimension, A or C scores higher net of its dimensions."
    )


def test_a_leader_with_the_fewest_dimensions_wins_at_every_rate():
    result = rank_candidates({"A": scored(2, 0.85), "B": scored(4, 0.80)}, ONE)

    assert [step["id"] for step in result["path"]] == ["A"]
    assert "wins at every value of a dimension" in result["path_sentences"][0]


def test_collinear_candidates_between_two_corners_win_at_no_rate():
    """At the one rate where they tie, the fewer dimensions take it."""
    cohort = {"A": scored(10, 0.850), "B": scored(6, 0.848), "C": scored(2, 0.846)}

    result = rank_candidates(cohort, ONE)

    assert result["wins_at_no_rate"] == ["B"]


# ----------------------------------------------------------- the standard error


def test_the_jackknife_standard_error_follows_the_formula():
    values = [0.80, 0.82, 0.81, 0.79]
    mean = sum(values) / 4
    expected = (3 / 4 * sum((v - mean) ** 2 for v in values)) ** 0.5

    assert _jackknife_se(values) == round(expected, 4)


def test_ranking_reports_each_score_and_each_paired_difference_its_own_error():
    result = rank_candidates(
        {
            "a": scored(2, 0.80, [0.80, 0.82, 0.81, 0.79]),
            # Moves with a in every replicate: the paired error is zero.
            "b": scored(4, 0.81, [0.81, 0.83, 0.82, 0.80]),
        },
        ONE,
    )

    rows = {row["id"]: row for row in result["ranking"]}
    assert rows["a"]["se"] == rows["b"]["se"] > 0
    assert rows["b"]["se_difference"] == 0.0
    assert result["close_competitors"][0]["se_difference"] == 0.0
    assert any(note["kind"] == "standard_error_scope" for note in result["notes"])


def test_the_standard_error_does_not_enter_the_choice():
    noisy = {"a": scored(2, 0.80, [0.70, 0.90, 0.75, 0.85]),
             "b": scored(4, 0.81, [0.81, 0.81, 0.81, 0.81])}
    quiet = {"a": scored(2, 0.80), "b": scored(4, 0.81)}

    assert ids(rank_candidates(noisy, ONE)) == ids(rank_candidates(quiet, ONE))


def test_a_missing_standard_error_is_said_once_per_reason():
    result = rank_candidates({"a": scored(2, 0.80), "b": scored(4, 0.81)}, ONE)

    unavailable = [n for n in result["notes"] if n["kind"] == "standard_error_unavailable"]
    assert len(unavailable) == 1 and "a and b" in unavailable[0]["text"]


def test_evaluate_records_replicates_on_groups_shared_by_every_candidate():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    labels = rng.integers(0, 3, size=200)

    first = evaluate_embedding(X, X[:, :2], labels, k=10, seed=5)
    second = evaluate_embedding(X, X[:, :4], labels, k=10, seed=5)

    assert first["d"] == 2 and second["d"] == 4
    assert first["jackknife"]["groups"] == second["jackknife"]["groups"]
    assert len(first["jackknife"]["replicates"]) == JACKKNIFE_GROUPS


def test_jackknife_groups_follow_the_seed_and_balance_each_class():
    labels = np.repeat([0, 1, 2], [50, 30, 20])

    groups = jackknife_groups(100, labels, seed=3)

    assert np.array_equal(groups, jackknife_groups(100, labels, seed=3))
    assert not np.array_equal(groups, jackknife_groups(100, labels, seed=4))
    for cls in (0, 1, 2):
        counts = np.bincount(groups[labels == cls], minlength=JACKKNIFE_GROUPS)
        assert counts.max() - counts.min() <= 1


def test_too_few_rows_for_the_jackknife_is_recorded_not_raised():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(21, 4))

    record = evaluate_embedding(X, X[:, :2], None, k=5, seed=0)["jackknife"]

    assert record["replicates"] is None and "floor" in record["unavailable"]


# ------------------------------------------------------------------- the notes


def test_every_note_carries_a_kind_rank_declares():
    result = rank_candidates(
        {"a": scored(2, 0.80), "b": scored(4, 0.81)},
        {"trustworthiness": 0.5, "silhouette": 0.5},
        failures={"c": {"op": "isomap"}},
    )

    kinds = {note["kind"] for note in result["notes"]}
    assert kinds <= set(NOTE_KINDS)
    assert {"weights_dropped", "failed_candidates", "close_competitors"} <= kinds


def test_the_report_classifies_every_kind_rank_can_write():
    """Defect 8: a kind nobody classified would drop out of section 9 silently."""
    assert set(LIMITATION_KINDS) == set(NOTE_KINDS)


# ------------------------------------------------------------ the registration


def _plan(**evaluation):
    return {
        "dataset": "d",
        "candidates": [{"id": "a", "stages": [{"op": "pca", "params": {}}]}],
        "evaluation": {"weights": {"trustworthiness": 1.0},
                       "justification": "declared up front", **evaluation},
    }


def _register(cli, csv_dataset, tmp_path, plan):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    reconnoitre(cli, runs / "r1")
    (runs / "r1" / "plan.json").write_text(json.dumps(complete(plan)), encoding="utf-8")
    return runs / "r1", cli("validate-plan", "--run-dir", runs / "r1")


def _codes(result):
    return {f["code"] for f in result.payload["findings"] if f["severity"] == "error"}


def test_the_default_margin_registers_and_is_recorded(cli, csv_dataset, tmp_path):
    run, result = _register(cli, csv_dataset, tmp_path, _plan())

    assert result.payload["valid"] and result.payload["margin"] == "default"
    records = [json.loads(line) for line in
               (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    registration = next(r for r in records if r["stage"] == "register_plan")
    assert registration["margin"] == 0.02 and registration["margin_state"] == "default"


@pytest.mark.parametrize("margin", [-0.01, 0.11])
def test_a_margin_outside_its_range_is_refused(cli, csv_dataset, tmp_path, margin):
    _, result = _register(cli, csv_dataset, tmp_path, _plan(
        margin=margin,
        margin_departure={"reason": "argued", "evidence": ["profile.shape.n_samples"]},
    ))

    assert "invalid_margin" in _codes(result)


def test_another_margin_without_a_departure_is_refused(cli, csv_dataset, tmp_path):
    _, result = _register(cli, csv_dataset, tmp_path, _plan(margin=0.05))

    assert "unexplained_margin_departure" in _codes(result)


def test_another_margin_argued_from_evidence_registers(cli, csv_dataset, tmp_path):
    _, result = _register(cli, csv_dataset, tmp_path, _plan(
        margin=0.05,
        margin_departure={"reason": "few rows", "evidence": ["profile.shape.n_samples"]},
    ))

    assert result.payload["valid"] and result.payload["margin"] == "departure"


def test_a_margin_departure_citing_nothing_real_is_refused(cli, csv_dataset, tmp_path):
    _, result = _register(cli, csv_dataset, tmp_path, _plan(
        margin=0.05,
        margin_departure={"reason": "few rows", "evidence": ["profile.not.a.key"]},
    ))

    assert not result.payload["valid"]


# ------------------------------------------------------------------- the CLI


def _ranked_run(cli, csv_dataset, tmp_path):
    plan = _plan()
    plan["candidates"].append({"id": "b", "stages": [{"op": "pca", "params": {"whiten": True}}]})
    run, _ = _register(cli, csv_dataset, tmp_path, plan)
    cli("prepare-reference", "--run-dir", run)
    for candidate in ("a", "b"):
        cli("embed", "--run-dir", run, "--id", candidate, "--in-process")
        cli("evaluate", "--run-dir", run, "--id", candidate)
    return run


def test_the_decision_log_qualifies_the_winner(cli, csv_dataset, tmp_path):
    run = _ranked_run(cli, csv_dataset, tmp_path)

    result = cli("rank", "--run-dir", run)

    assert result.code == 0
    records = [json.loads(line) for line in
               (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    entry = next(r for r in records if r["stage"] == "rank")
    assert entry["chosen"] == result.payload["winner"]
    assert entry["close_competitors"] == [c["id"] for c in result.payload["close_competitors"]]
    assert entry["margin"] == 0.02 and "leader" in entry and "discriminated" in entry


def test_rank_refuses_candidates_jackknifed_on_different_groups(cli, csv_dataset, tmp_path):
    run = _ranked_run(cli, csv_dataset, tmp_path)
    path = run / "metrics" / "b.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["jackknife"]["groups"] = "0000000000000000"
    path.write_text(json.dumps(record), encoding="utf-8")

    result = cli("rank", "--run-dir", run)

    assert result.code == 2 and not (run / "ranking.json").exists()
