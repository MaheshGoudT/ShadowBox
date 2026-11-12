# Primary simulation and RL backend: MuJoCo Playground

ShadowBox uses the official [MuJoCo Playground](https://playground.mujoco.org/)
codebase as its primary scalable simulation/training integration. The Git
submodule `external/mujoco_playground` is pinned to
`66640404324ddfefd0223359a7b15974c22d6d63` (Playground 1.0.1). Its Apache 2.0
license remains upstream. This revision uses PyTorch, RSL-RL, reaf-sim and
MuJoCo Warp. The older JAX/MJX research environments live in upstream `rss2025/`;
they are not the API targeted by this adapter.

The local CPU arena remains the reproducible two-boxer tactical baseline.
It provides scoring, observations, action legality and controllers that the
upstream tracker does not replace. Retain legacy code only while it supplies a concrete capability still used by
the new architecture or a reproducible comparison. Remove superseded code,
unused scaffolds and duplicate execution paths as replacement gates pass. The shared backend detection
was moved out of the Isaac Lab adapter into the generic backend module.

## Setup

From the ShadowBox root:

```bash
git submodule update --init external/mujoco_playground
uv venv --python 3.12 .venv-playground
uv pip install --python .venv-playground/bin/python --index-strategy unsafe-best-match -r requirements/playground-cpu.txt
uv pip check --python .venv-playground/bin/python
# CUDA 13.0 runtime used for Thor validation:
uv venv --python 3.12 .venv-playground-gpu
uv pip install --python .venv-playground-gpu/bin/python --index-strategy unsafe-best-match -r requirements/playground-gpu.txt
uv pip check --python .venv-playground-gpu/bin/python
.venv/bin/python -m pip install --no-deps -e .
```

Use upstream's [installation instructions](../external/mujoco_playground/README.md#installation)
for the driver-specific Torch/Warp wheels (including CUDA 12). Python 3.12 is
also supported. The source checkout has additional upstream source mappings;
retain the resolved dependency inventory with experiment results. Do not
install this stack into ShadowBox's `.venv`: the pinned
source requires MuJoCo >=4 and Torch <2.14. ARM64/Jetson wheel availability,
driver access and throughput must be validated on the intended host. The
adapter fails on unavailable CUDA instead of silently switching to CPU.

The dependency resolver was checked on this ARM64 Python 3.12 host with the
MuJoCo package index above. Resolving with only PyPI failed because the matching
`mujoco-studio` package was missing. A successful dependency preview is not a
completed installation or a GPU runtime check.

CPU task runtime is validated separately from GPU training; current architecture
boundaries are summarized in [architecture.md](architecture.md), and detailed
run evidence is in the local ignored `docs/planning.md`. `doctor` reports missing interpreters, import failures, wrong source,
wrong commit, missing reference motions and unavailable configured devices.
First use can also fetch upstream robot assets, which may fail independently.

## Commands

```bash
shadowbox-playground doctor
shadowbox-playground smoke --dry-run
shadowbox-playground train --dry-run
shadowbox --backend playground --backend-dry-run
shadowbox --config configs/playground.yaml --headless
shadowbox-playground train
shadowbox-playground play
```

Until editable entry points are refreshed, use
`.venv/bin/python -m shadowbox.integrations.playground`.
`shadowbox --backend playground` selects the local `ShadowBoxG1Defense` smoke
task through Playground/reaf. It is a low-level defensive training environment. No-config `shadowbox --demo` continues to open the CPU arena.
`configs/playground.yaml` owns the task, world count, runtime, device, seed,
reference trajectory, checkpoint and timeout. Relative paths resolve from the
ShadowBox root. Change `checkpoint` to a trained run directory before `play`.
Logs, command/source provenance and exit status go to a unique
`tmp/playground/runs/run-*/` directory. Smoke writes finite-observation/reward
checks and throughput to `smoke.json`; zero actions validate plumbing rather
than learned performance. GPU smoke throughput includes checks and resets and
is not a training throughput claim.

For a small installation check, copy the config and set `task: CartpoleBalance`,
`device: cpu`, `num_envs: 1`, `iterations: 1`, `trajectory: null`. This avoids
requiring a G1 motion corpus and selects upstream's `cpp` physics backend.
The adapter uses the pinned internal registry and wrapper behind one module;
upstream API changes require a deliberate pin update and integration checks.

## Motion and defensive task boundary

G1WholeBodyTracking needs a retargeted `.npz` reference. Follow upstream's
[motion conversion and training guide](../external/mujoco_playground/mujoco_playground/tasks/whole_body/tracking/README.md).
SOMA/Frax can supply retargeted motions; validate native G1 joint order, units,
base quaternion order and the upstream trajectory fields before conversion.
ShadowBox clip JSON is not an interchangeable trajectory format. Do not invent
a standing trajectory to imply that defensive motion training has completed.

`playground/task.py` implements the initial two-free-base-G1 scene with physical
floor/glove/robot contacts, bounded 29-joint learner actions, alternating
independent opponent arm targets, 162-value paired observations, contact-force
penalties, uprightness/control penalties, fall/boundary termination and masked
resets. `playground/registry.py` registers it locally; `playground/launch.py`
registers before upstream train/play scripts, leaving the submodule untouched.
The G1 asset is pinned to Menagerie commit
`98a70c569c4c1b6da467a20012f766ee1f8dc521`. Arm fixture angles were fitted against
that model's glove geometry; these are authored fixtures, not motion capture.
Learned balance, qualified defensive actions, event-level scoring and held-out
evaluation remain gates. The scripted opponent can fall; no balance capability
is inferred from bounded PD targets. Use Playground's scene,
task and device configuration layers rather than writing another physics/RL
engine. Tracking single-robot references does not implement opponent contacts
or self-play by itself. The separate `ShadowBoxG1SelfPlay` task adds both.
Validate those gates before replacing the stabilized baseline.
Synthetic interaction trajectories can then feed GAWM predictive planning;
GAWM and SOMA/Frax are architectural boundaries, not new runtime dependencies.
Hardware rollout remains a separate qualification step.

The boxing task uses the pinned articulated G1 hand model and resets both sets
of fingers to a closed fist. The tracker still controls the original 29 body
joints; finger targets are held at the fixed boxing pose. Transparent glove
hitboxes remain for contact measurement, with no visible fist-sized spheres.
MuJoCo Studio's browser viewer needs WebGL 2 and still depends on browser frame
rate. The headless GIF path uses offscreen EGL for repeatable review while the
physics rollout stays independent of browser rendering speed.

## Iterative policy self-play prototype

`ShadowBoxG1SelfPlay` exposes nine authored goals (guard, turn, four movement
primitives, jab, cross and body guard) plus a `CONTINUE` action to one learned
boxer. `CONTINUE` keeps the active goal running through its timed phase; the
player observation includes that boxer's active goal and normalized phase. Once
a jab or cross starts, its authored motion completes. A follow-up punch request
is latched until the active phase ends; `CONTINUE` and target-turn requests do
not erase it, while a guard or footwork request clears it. Guard, turn and
movement do not preempt an already active punch. It uses the same two-G1 tracker scene and a symmetric
sparse reward from per-glove contact episodes, block contacts, posture, falls
and ring exits. Head/body contact rewards and penalties are configurable task
options. Contact events are training signals, not an official boxing referee
or qualified score.
The opponent starts as the built-in timed jab/cross schedule. After each PPO
round, the runner exports a frozen actor and samples that schedule (or an
explicit `opponent_policy_script`) alongside prior snapshots once per episode.
The starting pool index rotates with the global round number, so a short round
does not restart on the same first snapshot each time.
This is one learner against a snapshot pool, not two simultaneously updated
actors or a scalable league. The tracker
currently restricts it to one paired world; no policy competence is claimed.
The learner observation expresses relative position, orientation and linear
velocity in the controlled boxer's heading frame, so swapping world-facing
directions does not change the same physical state representation. Free-joint
angular velocity remains local, matching MuJoCo's qvel convention. Quaternion
signs use a canonical representative, including at the 180-degree boundary.

Configure `configs/playground-motion-selfplay.yaml` with an exported Metatron
bundle, then run:

```bash
.venv/bin/python -m shadowbox.integrations.playground doctor --config configs/playground-motion-selfplay.yaml
.venv/bin/python -m shadowbox.integrations.playground selfplay --config configs/playground-motion-selfplay.yaml --dry-run
.venv/bin/python -m shadowbox.integrations.playground selfplay --config configs/playground-motion-selfplay.yaml
```

The default is four rounds of ten PPO updates with 64 transitions per update.
Training alternates the controlled boxer between sides A and B by round. After
each round, the candidate is checked against the scripted opponent from both
sides at the current guard width. The curriculum narrows from 0.45 m toward
0.10 m only when both sides record at least one legal head/body touch in their
300-step check; otherwise the next round retries the same width. Guard widths
follow the controlled and opposing roles when sides swap. These settings are a
configurable contact-learning experiment, not a calibrated opponent model.
The self-play config targets the lower ribs at `body_target_z: -0.05`: fixed
controls produced three legal body contacts per side through a 0.11 m guard at
that height. At 0.11 m, the 24-case head-defense candidate scored 81 head/18
body contacts against 10 head/44 body from opponents, with 318 blocks. Raising
body reward from 4 to 5 while keeping head-hit penalty 6 and block reward 0.5
produced 63 head/23 body against 1 head/48 body, with 313 blocks. Its net
contact margin was 37 versus 45 for the head-defense candidate: it improved
against the newest snapshot (12/1 vs 0/9) while giving up some total offense.
Action 7 was sampled 589 times, but the cross goal never executed; traces show
the queued request was canceled after target-turn or footwork choices. A
same-checkpoint ablation preserved queued punches through target turns but
still canceled them on guard or footwork. It produced 224 cross-goal steps
against frozen snapshots and increased body contacts from 11 to 20 in the
12-case seed-101 subset, with candidate total contacts unchanged at 38. A
six-round continuation passed its curriculum gates but then scored only 25
head/3 body against 2 head/30 body from opponents in the full 24-case matrix;
it chose target-turn in all 2,400 steps against the older snapshot and executed
no crosses, so that checkpoint is not promoted. The next iteration resumes
from the prior head-defense checkpoint and its earlier opponent pool lineage.
These are held-out diagnostics, not a qualified boxing score.

Continue a previous league from its latest two-side-qualified checkpoint,
restore the frozen snapshots available through that round, and start at the
guard width recorded by its last passing gate:

```bash
.venv/bin/python -m shadowbox.integrations.playground selfplay \
  --config configs/playground-motion-selfplay.yaml \
  --resume-run tmp/playground/runs/run-<id>
```

Use `--resume-guard-width 0.15` to override the recorded next width when a
held-out evaluation supports a more conservative restart. The trainer verifies
the checkpoint hash and snapshot files before continuing. Within a run, failed
stages continue updating the current candidate at the same guard width; when a
new run resumes, it selects the last two-side-qualified checkpoint and starts
with the opposite controlled role. The trainer
carries prior round provenance into the new `league.json`. Configure
`playground.self_play.opponent_guard_width_step` when retries should hold a fixed
width increment instead of recomputing it from the remaining round budget.
Each round writes a candidate checkpoint, deterministic TorchScript snapshot,
and atomically updated `league.json` under its `tmp/playground/runs/` directory.
The league records lineage and both-side curriculum measurements, so completed
rounds remain resumable if a later round stops early. The runner does not perform
held-out promotion. Evaluate candidates against fixed seeded opponents, both
starting orientations and separate opponent snapshots before any promotion. The test
task still supports only one world because the qualified frozen tracker bundle
is batch-two for exactly the two robots.

Run the diagnostic held-out matrix with an exported candidate checkpoint; pass
one or more frozen opponent actors with repeated `--opponent` flags:

```bash
.venv/bin/python -m shadowbox.playground.selfplay_eval \
  --config configs/playground-motion-selfplay.yaml \
  --candidate tmp/playground/runs/run-<id>/round-003/candidate.pt \
  --opponent tmp/playground/runs/run-<id>/round-002/snapshot/opponent.pt \
  --seeds 101 103 --steps 500 --opponent-guard-width 0.12
```

It evaluates both controlled sides and both start orientations, records contact
counts, completed/censored episode lengths and transition rate under `tmp/`,
and exits nonzero if a case fails. The report includes `summary_by_opponent`
totals for contacts, sampled actions and executed skill goals, including net
head/body contact counts; the original per-case rows retain seed, controlled
side and swapped-start fields. Set `--opponent-guard-width` to the measured
curriculum stage; without it, evaluation uses the config's target width.
`evaluation_seed` allows a config to request an exact evaluation seed; otherwise
the adapter offsets its training seed.
These measurements are diagnostic only. They do not implement promotion or an
official match score.

The trainer currently requires a local, qualified motion bundle and CUDA
motion inference. New trainable policies are high-level primitive selectors;
the tracker remains frozen. The learned actor receives player-centric
proprioception and relative opponent state, not camera images or human video.
No real-robot system identification or deployment is wired into this stage.

## Generic plugin boundary

`integrations/backend.py` defines a small `Backend` protocol: `probe`, `command`,
`provenance`, output directory and timeout. Its runner owns subprocess isolation,
logs and lifecycle metadata; `load_backend` is the lazy factory seam for new
adapters. Only the Playground adapter knows the upstream task/CLI details.
The CPU arena, motion retargeting, learned planning, and policy transport keep
their existing separate observation/action contracts. Isaac Lab and the other
existing launchers remain optional independent integrations.

## Optional ManiFlow-style policy plumbing benchmark

No ManiFlow package, checkpoint or submodule is required. Its Franka manipulation
checkpoint cannot control G1, so it is not loaded, adapted or retrained.
`policy_chunks.py` supplies a generic callable policy protocol, optional HTTP
transport, chunk validation, per-step legality checks and episode queue cleanup.

```bash
shadowbox-policy-benchmark --steps 8 --horizon 4
shadowbox-policy-benchmark --endpoint http://localhost:8000/predict --steps 8
```

The default uses synthetic guard chunks through the real CPU BoxingEnv.
The HTTP option requires an explicit server adapter that returns the following
ShadowBox schema; a raw ManiFlow endpoint is not assumed compatible:

```json
{"schema": "shadowbox.tactical.v1", "actions": [0, 3, 4, 0]}
```

Requests include `schema`, `observation` (12 current tactical features),
`action_mask` (10 booleans), and `max_actions`. Only defensive indices are
accepted: guard=0, turn=1, retreat=3, sidestep left/right=4/5, body guard=8.
Every queued action is checked against the current mask; an invalidated action
clears the chunk and selects a legal defensive fallback. Continuous Franka
vectors and attack indices fail validation. The report records inference calls,
latencies, mask interventions and simulation steps. This checks serving,
decoding and closed-loop chunks; it does not validate vision encoding, a
flow-matching model, physical balance or sim-to-real performance. Add a visual
observation contract only if a dedicated vision-policy experiment needs it.

## CPU validation commands

```bash
.venv/bin/python -m shadowbox.integrations.playground doctor --config configs/playground-cpu.yaml
.venv/bin/python -m shadowbox.integrations.playground smoke --config configs/playground-cpu.yaml
.venv-playground/bin/python -m pytest -q tests/test_playground_runtime.py
```

The C++ physics backend supports one CPU world. Batched validation requires the
Warp backend and is measured separately. Smoke writes `transitions.npz`; its
`qpos_next` is post-reset on done rows, so exclude those rows when fitting
transition dynamics. Run manifests hash local task sources as well as recording
the pinned upstream revision. A passing smoke checks execution and finite
values; it does not establish learned stability or boxing skill.

## Viewer and recorded-state captures

```bash
# Headless 10-second play by default; writes a GIF under tmp/playground/gifs:
shadowbox-playground play --config configs/playground-motion-endurance.yaml --zero-actions
# Open the hardware-backed Chrome viewer only when explicitly requested:
shadowbox-playground play --config configs/playground-motion-endurance.yaml --viewer --zero-actions
# With the isolated interpreter and source import path, render a recorded run:
PYTHONPATH=src .venv-playground/bin/python -m shadowbox.playground.replay tmp/playground/runs/RUN_ID --frame 0 --output tmp/playground/replay.png
```

All Playground configs default to `viewer: false` and `play_steps: 500` (10 s at
50 Hz). The `gif` config block controls preview output: `enabled: true` saves
headless `play`, `smoke`, and `evaluate` previews in the configured directory
(default `tmp/playground/gifs`); set it to `false` to skip GIF rendering. Each
filename includes its unique run ID and stage, and the run manifest records the
path. The GIF spans the complete configured rollout, sampled to at most 200
frames. Training stays headless; use a headless play/evaluate run afterward to
produce a GIF. `--viewer` opts into interactive Chrome; the Python launcher
selects the tested NVIDIA GLX/EGL provider and dedicated profile automatically.
The replay renderer uses NVIDIA offscreen EGL at 640x480 and hides glove
hitboxes while preserving them in the saved physics model.

For a learned actor, configure its checkpoint and omit `--zero-actions`. Recorded
runs include `model.mjb`, tying the capture to the compiled physics model; use
the matching MuJoCo runtime to read it. All captures remain under `tmp/`.

## Tracker-backed boxing experiments

The motion task additionally needs `requirements/playground-motion.txt`, the
pinned Metatron G1 package, its exported pipeline/sidecar and pelvis FK bundle.
The qualification/export scripts live under `scripts/metatron_*`; artifacts are
local under `tmp/playground/metatron/compiled` and are intentionally not source
files. Keep the downloaded native package's license, notice and model card.
ProtoMotions export/probing uses the separate `requirements/protomotions-*`
environments; do not merge those dependencies into the Playground environment.

```bash
.venv/bin/python -m shadowbox.integrations.playground evaluate --config configs/playground-motion-defense.yaml
.venv/bin/python -m shadowbox.integrations.playground evaluate --config configs/playground-motion-reactive.yaml
.venv/bin/python -m shadowbox.integrations.playground evaluate --config configs/playground-motion-graph.yaml
.venv-playground-gpu/bin/python scripts/playground_rollout_metrics.py tmp/playground/runs/RUN_ID
.venv-playground-gpu/bin/python scripts/playground_contact_controls.py --steps 210
.venv-playground-gpu/bin/python scripts/playground_motion_controls.py --seed 17034
.venv-playground-gpu/bin/python scripts/playground_contact_controls.py --sides a b --targets head --distances 0.70 --delays 0 --approach-secs 0.5 --reset-yaw-radians 0.15 --reset-lateral-m 0.05 --swap-start-sides --steps 210
```

The graph config requires a matching qualified fixed-batch-two derivative and
its JSON qualification. It is an optional performance experiment. Current
end-to-end timing still misses real time. The motion registry uses a categorical guard/body-guard actor (IDs 0/8);
Gaussian sampling is confined to the separate joint-control diagnostic task.
`configs/playground-motion-train.yaml` runs a two-update transport pilot:

```bash
.venv/bin/python -m shadowbox.integrations.playground train --config configs/playground-motion-train.yaml
```

Set `playground.checkpoint` to the saved model path for `evaluate` or `play`.
The short pilot establishes transport, not learned defensive competence.
`configs/playground-motion-exposed.yaml` is a wide-hand diagnostic control.
A bounded fixed/reactive distance sweep is available through
`scripts/playground_defense_gauntlet.py`; `scripts/playground_contact_controls.py`
checks scripted head/body positive contacts on each side across distances and
attack delays. Its `--approach-secs` option inserts a direct step-forward phase
before the jab/cross sequence; `--reset-yaw-radians`, `--reset-lateral-m` and
`--swap-start-sides` vary the starting state. Set `--opponent-guard-width` to
measure whether a fixed punch can reach through a narrower guard; the default
0.45 m is the exposed-target control. `--body-attack-arc-lateral` bends the
cross trajectory around the guard along the opponent-relative left axis for
measured angled body-shot controls; the equivalent config option is
`task_options.body_attack_arc_lateral` and defaults to zero. Summaries mark
incomplete runs explicitly.
`playground_motion_controls.py` measures fixed
turn, forward/backward, and sidestep commands on both boxers, then runs posture,
foot-contact and drift metrics. Both tools use diagnostic scripted actions;
they do not expand the learned actor beyond IDs 0/8 or measure learned boxing
skill. Their outputs remain under `tmp/`. Current paired movement and 0.70 m
approach/contact evidence is summarized in the ignored local `docs/planning.md`.
The motion configs expose validated `task_options.approach_speed_a/b` and
`retreat_speed_a/b` baseline values. The self-play config uses 0.4 m/s forward
approach, a 0.38 m jab reach cap, a 0.5 m approach target, a 0.6-second queued
approach and progress reward 12 per meter. A fixed-control test at 0.4 m/s
moved both sides about 0.64 m in 1.5 s, with no falls and 1–2 cm settling drift.
These are simulator baselines, not values inferred from real motion. A 0.9 m
fixed-command test at 0.27 m/s moved each boxer about 0.27 m in one second.
Jab debugging found two causes: the tracker does not
condition elbow pose, and the hidden glove proxy was 4 cm behind the tracker's
0.12 m wrist-to-palm point. The proxy now aligns to that point and the active
jab blends a calibrated four-joint target with the tracked wrist path. A
fixed-control test from 0.9 m with a 1.5-second approach recorded two head and
two body contact episodes per side, without a fall; elbow bend reached about
14 degrees at extension. This is a scripted open-target positive control, not
a learned jab or a closed-guard result. Retargeted boxing examples and an
elbow-aware tracker remain the preferred motion source.
The six-round policy candidate scored 87 head/10 body against 30 opponent
contacts over 24 cases, with no head contacts conceded; body attack coverage
remains uneven, so it is not promoted. Its 0.9 m continuation initially
produced no legal contacts. The earlier qualified checkpoint passed 0.65 m and
0.70 m, then `run-longrange-approach04-20261112` resumed at 0.75 m with the
updated 0.4 m/s gait and progress reward 12. It passed 0.75 m on candidate
three and 0.80 m on candidate eight, then failed at 0.85 m. The 0.80 m
checkpoint scored zero head/body contacts in eight 0.9 m held-out cases against
a 0.10 m guard; all 31 recorded candidate attacks were blocked. Against an
exposed 0.45 m guard, the same checkpoint scored 21 head and 43 body contacts
over four 0.9 m cases, while conceding 8 body contacts. The straight jab is
executing; the narrow guard stops it. A 10-second headless preview is saved at
`tmp/playground/gifs/run-4bbbee954f67-play.gif`. Neither result qualifies
closed-guard boxing or promotion; held-out defense and broader footwork checks
remain necessary.
The runner automates PPO iterations against frozen opponent snapshots.
Held-out evaluation, candidate promotion, skill extraction and robot execution
remain manual and unqualified.

Pinned motion setup and export commands: [Metatron G1 artifact path](metatron-playground.md).
