# Legacy capability retention and removal gates

The Playground task is the primary new architecture. Retention below is based
on a live caller and a capability that the current tracker task does not replace.
It is not a declaration that the old supported-root physics qualifies free-base
boxing. Remove a group after the replacement gate passes and its callers migrate.

| Retained group | Concrete current value and caller | Executable checks | Removal gate |
| --- | --- | --- | --- |
| `boxing/`, `safety/`, arena-facing `motion/` | Match rounds, scoring, legal actions, safety and authored movement used by `boxing/match.py`, `app.py` and `rl/environment.py` | `test_boxing.py`, `test_active_defense.py`, `test_body_defense.py`, `test_guard_geometry.py`, `test_stance.py` | Playground round/referee/movement behavior passes equivalent cases with free bases |
| `simulation/` | Existing visible CPU arena, G1 asset loading and diagnostics used by `boxing/match.py` and motion replay | `test_arena.py`, `test_unitree.py`, `test_timing.py` | New task supports the visible match and replay workflows with measured stability |
| `ui/viewer.py`, arena branch of `app.py` | User-facing keyboard controls and captures; Playground currently has saved-model replay plus upstream viewer transport | `test_viewer_controls.py` | Equivalent interactive controls are tested in the new task |
| `rl/` | Existing tactical observation/mask/reward, training and held-out opponent workflow; new categorical path currently has only guard/body guard | `test_rl_environment.py`, `test_tactical_learning.py` | New action set, scoring, movement and evaluation suite qualify a replacement |
| `evaluation/` | Scenario audits, optimization, tabular baselines and native encoder/IMU motion diagnostics called by project entry points | `test_evaluation.py`, `test_scenarios.py`, `test_qlearning.py`, `test_motion_replay.py` | Each still-used entry point migrates or is explicitly retired |
| `motion/supermimic.py`, `clip.py`, `morphology.py` and external integration launchers | Existing clip/replay/robot-profile experiments; these are not silently used as Playground motion policies | `test_supermimic.py`, `test_morphmimic.py`, `test_holoboxing.py`, `test_isaaclab.py` | A replacement supports the relevant recorded-clip/profile/launcher contract or that experiment is retired |

Already removed: disconnected `policy/lfm3_policy.py` and `v2/flora.py` scaffolds.
Their presence added no executable capability. The shared Primitive enumeration
is used by both runtimes; importing it no longer imports the legacy controller.
Generic backend lifecycle/provenance is shared rather than duplicated inside the
Isaac Lab adapter. Retaining an experimental launcher is not evidence that its
simulator or model has passed qualification.
