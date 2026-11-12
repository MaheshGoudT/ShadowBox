# Configuration by purpose

| File | Purpose |
| --- | --- |
| `simulation.yaml` | Shared simulator and rule defaults. |
| `unitree.yaml` | Live G1 demo overrides; preserves the existing motion setup. |
| `simulation-diagnostics.yaml` | Repeatable G1 motion, foot slip, and safety trend check. |
| `tactical-audit.yaml` | Strict-scoring action sweep, quality gates, and example collection. |
| `tactical-train.yaml` | PPO run length, seed, and checkpoint path. |
| `tactical-eval.yaml` | Held-out opponents, seeds, and spacings. |

The tactical runner reads all three tactical files. Change one stage's settings in its own file. Generated logs, demonstrations, checkpoints, and reports go under `tmp/` and are ignored by Git.

## HoloBoxing benchmark

`holoboxing.yaml` selects the pinned upstream checkout, separate Python environments, CUDA device, evaluation image and asset/controller/data/output paths. See `requirements/README.md` for setup and launcher commands. It does not change the local simulation or tactical configs.

## MorphMimic shared motion tracker

`morphmimic.yaml` selects the pinned upstream checkout, separate training/runtime
interpreters, robot profile and data/export/simulation paths under `tmp/`.
`shadowbox-morphmimic` uses it for upstream stages; `shadowbox-motion-eval
--morphmimic` uses it for canonical encoder/IMU diagnostic exports. See
`docs/morphmimic.md` and the updated multi-robot layer in `docs/architecture.md`.

## Playground

`playground.yaml` configures the primary external simulation/RL backend. It is
not merged into the CPU arena configuration. Use `shadowbox-playground` or
`shadowbox --config configs/playground.yaml --headless` for the upstream smoke
task. See [setup and task boundaries](../docs/playground.md).

Tracker fixtures: `playground-motion.yaml` (standing),
`playground-motion-defense.yaml` (active guard),
`playground-motion-reactive.yaml` (geometry baseline),
`playground-motion-exposed.yaml` (wide-hand threat control),
`playground-motion-graph.yaml` (qualified fixed-batch CUDA graph experiment), and
`playground-motion-train.yaml` (two-update categorical guard/body-guard pilot),
`playground-motion-varied.yaml` (independent initial yaw/lateral variation), and
`playground-motion-endurance.yaml` (30-second episode).
`playground-motion-selfplay.yaml` iterates nine-goal plus CONTINUE PPO candidates against
the timed scripted attack schedule and frozen policy snapshots in a one-pair
prototype. It alternates the controlled side and advances the guard-width
curriculum only after both roles record a legal touch. This is not a qualified
or batched boxing learner. Evaluate a
candidate across sides/orientations with
`python -m shadowbox.playground.selfplay_eval --config configs/playground-motion-selfplay.yaml --candidate /path/to/candidate.pt --opponent /path/to/snapshot.pt --seeds 101 103`.
The self-play fixture targets the lower ribs at -0.05 m based on fixed-contact
controls through a 0.11 m guard. Defaults use equal head/body contact rewards
and penalties with categorical exploration coefficient 0.02. The latest
24-case body-reward-5 candidate, trained with head-hit penalty 6 and block
reward 0.5, scored 63 head/23 body contacts against 1 head/48 body from
opponents, with 313 blocks. It improved against the newest snapshot but had a
lower aggregate contact margin than the previous head-defense candidate.
Action 7 was sampled 589 times but no cross goal executed, exposing queued-punch
cancellation by target-turn/footwork choices. The queue now preserves a pending
punch through target turns while guard and footwork still cancel it. A
12-case same-checkpoint ablation executed 224 cross-goal steps against frozen
snapshots. Its six-round continuation, `tmp/playground/runs/run-24edf1fcaeec`,
passed curriculum gates but failed the full held-out matrix (25 head/3 body vs
2 head/30 body, no cross goals), so it was not promoted. The next iteration
resumes from the stronger head-defense checkpoint with its earlier snapshot
pool. See the ignored local run plan for per-style results and lineage.
Use `task_options.body_attack_arc_lateral` to test a cross trajectory that bends
around a closed guard; keep it at zero until fixed-punch controls show the arc
reaches the torso.
These do not establish learned boxing or real-time capability. All Playground
configs run headless by default. The `playground.gif.enabled` switch controls
recorded preview rendering, and `playground.gif.directory` selects its output
directory (default `tmp/playground/gifs`).
Motion fixtures expose `task_options.approach_speed_a/b` and
`retreat_speed_a/b` as explicit baseline gait commands. The self-play config
targets 0.9 m and uses 0.4 m/s approach speed per side; its 0.38 m punch
reach cap matches measured near-full extension in the pinned G1 model. The
distance curriculum starts at 0.6 m. These are simulator baselines for later
real-to-sim calibration. The tracker leaves elbow pose out of its goals, so
the task now blends a configurable four-joint near-straight jab pose during the
strike. The hidden glove proxy sits at the tracker's 0.12 m wrist-to-hand point;
previously it was 4 cm behind. In an open-target control from 0.9 m, with a
1.5-second approach, both sides made two head and two body contact episodes
without falling, while the elbow reached about 14 degrees of bend at extension.
This is a fixed-control positive test, not a learned or held-out result.
Self-play now rewards forward approach progress at weight 12 until 0.5 m; the
approach skill is held for 0.6 s so sampled footwork survives the current jab.
The prior curriculum passed 0.65 m and 0.70 m. A 10-round continuation passed
0.75 m on candidate three and 0.80 m on candidate eight, then failed at 0.85 m.
Its qualified 0.80 m checkpoint scored zero head/body contacts across eight
0.9 m held-out cases against a 0.10 m guard; 31 attacks were blocked. Against
an exposed 0.45 m guard it scored 21 head and 43 body contacts in four cases,
while conceding 8 body contacts. The jab is executing; it does not yet get
through the tight guard. A 10-second headless preview is saved at
`tmp/playground/gifs/run-4bbbee954f67-play.gif`. The checkpoint is diagnostic
and not promoted.
