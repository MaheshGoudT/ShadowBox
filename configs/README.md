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
These do not establish learned boxing or real-time capability. All Playground
configs run headless by default. The `playground.gif.enabled` switch controls
recorded preview rendering, and `playground.gif.directory` selects its output
directory (default `tmp/playground/gifs`).
Motion fixtures set `task_options.retreat_speed_a/b` as explicit baseline gait
commands; the current diagnostic values are 0.27/0.18 m/s and should be
replaced when real-to-sim calibration supplies qualified speeds.
