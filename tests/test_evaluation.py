import json
from dataclasses import asdict

import pytest

from shadowbox.boxing import Parameters
from shadowbox.evaluation import evaluate, search


def test_evaluation_repeats_and_swaps_robot_assignments():
    baseline = asdict(Parameters())
    first = evaluate(baseline, [baseline], [7], seconds=4)
    assert first == evaluate(baseline, [baseline], [7], seconds=4)
    assert [m["candidate_side"] for m in first["matches"]] == ["a", "b"]
    assert all(m["finished"] for m in first["matches"])
    assert first["mean_reward"] == 0  # Identical policies cannot gain from side assignment.


def test_search_keeps_validation_separate_and_writes_loadable_policy(tmp_path):
    output = tmp_path / "candidate.json"
    report = search(iterations=1, seeds=1, seconds=3, output=output)
    saved = json.loads(output.read_text())
    Parameters(**saved["parameters"])
    assert not set(report["training_seeds"]) & set(report["validation_seeds"])
    assert len(report["validation"]["candidate"]["matches"]) == 6
    assert (tmp_path / "candidate_snapshots" / "baseline.json").is_file()
    assert (tmp_path / "candidate_snapshots" / "iteration_001.json").is_file()
    if not report["validation"]["improved"]:
        assert saved["parameters"] == report["baseline"]
    assert output.with_name("candidate_report.json").is_file()


@pytest.mark.parametrize("arguments", [{"seeds": 0}, {"seconds": 0}, {"iterations": -1}])
def test_search_rejects_empty_evaluation(arguments):
    with pytest.raises(ValueError):
        search(**arguments)
