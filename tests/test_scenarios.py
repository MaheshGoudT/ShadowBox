"""Behavior-level tests for the deterministic scenario report."""

from pathlib import Path
import sys


SOURCE = Path("/home/mahesh/Desktop/ShadowBox/src")
sys.path.insert(0, str(SOURCE))

# The implementation is staged separately because the supplied source tree is read-only.
SCENARIO_FILE = Path(__file__).parents[1] / "src/shadowbox/evaluation/scenarios.py"
import shadowbox.evaluation as evaluation_package
evaluation_package.__path__.insert(0, str(SCENARIO_FILE.parent))
from shadowbox.evaluation.scenarios import run_scenarios


def test_all_scenarios_pass_and_measure_expected_behaviors():
    report = run_scenarios(seed=23)
    assert report["passed"]
    assert {scenario["name"] for scenario in report["scenarios"]} == {
        "closing_opponent", "boundary_defense", "asymmetric_position_start",
        "emergency_stop_reset", "offense_and_guard",
    }
    boundary = next(s for s in report["scenarios"] if s["name"] == "boundary_defense")
    assert boundary["evidence"]["unsafe_action_reason"] == "arena boundary"
    assert "step_backward" not in boundary["evidence"]["defender_actions"]
    assert boundary["evidence"]["defender_punches"] == 0
    stopped = next(s for s in report["scenarios"] if s["name"] == "emergency_stop_reset")
    assert stopped["evidence"]["no_progress"] and stopped["evidence"]["reset_recovered"]


def test_scenario_output_is_deterministic():
    assert run_scenarios(seed=31) == run_scenarios(seed=31)
