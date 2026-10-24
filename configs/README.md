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
