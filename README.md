# ShadowBox

Two simple articulated humanoids spar in a MuJoCo boxing ring. Both have independent reactive offense/defense policies. The visible demo includes guard, jab, cross, approach, retreat, sidestep, round timing, virtual-touch scoring, and a HUD.

## Run

From this folder:

```bash
source .venv/bin/activate
shadowbox --demo
```

The window opens by default. Space pauses, R resets, E stops, Escape closes. Drag to orbit and scroll to zoom. The final HUD stays open for inspection and reset. Mesa software rendering is the default on this machine; use `--hardware` to try the system GPU.

```bash
shadowbox --defense-only                     # B never initiates a punch
shadowbox --headless --round-seconds 60 --rounds 1
shadowbox --seconds 15 --capture tmp/captures/arena.png
shadowbox-train                              # small self-play parameter search
shadowbox --policy-b tmp/policies/tabular.json
shadowbox-learn # tabular reinforcement learning experiment
shadowbox --learned-b tmp/policies/tabular.json
shadowbox --supermimic # enable SuperMimic motion imitation
python -m shadowbox.evaluation.scenarios
pytest
shadowbox-learn                             # tabular reinforcement learning experiment
shadowbox --learned-b tmp/policies/tabular.json
python -m shadowbox.evaluation.scenarios
pytest
```

`shadowbox-train` optimizes reactive policy settings against a pool of opponents, plays both starting sides, and compares the final candidate with baseline on separate evaluation seeds and additional opponents. It records whether promotion was justified; this is parameter search, not neural reinforcement learning or imitation learning.

`shadowbox-learn` learns a small state/action value table from simulated matches, including frozen self-play opponents. Its result is evaluated separately before live promotion; an unpromoted artifact loads the reactive baseline. This learns tactics over the existing stabilized movements, with no neural network or balance training.

## Real Unitree G1 model

```bash
shadowbox --config configs/unitree.yaml --demo
shadowbox --config configs/unitree.yaml --joint-demo
```

The G1 profile uses the actual licensed 29-joint model and meshes. Each robot retains legs, waist, shoulders, elbows, and wrists, with three stabilized root controls. The joint demo moves 23 native joints. Boxing distances and safety spacing are scaled to G1 geometry. Full-joint articulation uses position servos; free-base balance and physical impacts remain future work.

The model assets are pinned and carry their original license under `assets/unitree_g1/`. If assets are absent:

```bash
python -m shadowbox.simulation.fetch_assets --revision f3b6013629c7b94f8ee93fd977b161fa8350f9bc
```

## Configuration and outputs

Edit `configs/simulation.yaml` for physics timestep, ring size, motion limits, rounds, policies, rendering, output paths, and search settings. CLI flags override selected settings. `configs/noisy.yaml` provides a seeded observation-noise/delay experiment; exact simulation state still drives safety and scoring. A partial override file can change just the settings needed for an experiment:

```yaml
# Example: slower, defensive B.
policies:
  b: {aggression: 0.35, reaction: 0.3, desired_distance: 0.8}
```

```bash
shadowbox --config configs/my-experiment.yaml
shadowbox-train --config configs/my-experiment.yaml
```

Relative generated paths resolve inside ShadowBox even when launched from another directory. Generated files are ignored by Git:

- `tmp/logs/shadowbox-*.log`: separate rotating event logs for each run.
- `tmp/results/latest.json`: match settings, actions, touches, blocks, misses, and simplified scorecards.
- `tmp/captures/`: requested screenshots.
- `tmp/policies/`: candidate parameters, full search reports, and baseline/iteration snapshots.

## Fresh setup

Python 3.10+ is required; the verified environment uses Python 3.12 on ARM64 Linux.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/dev.txt
python -m pip install --no-deps -e .
```

If `venv` reports missing `ensurepip`, install your OS Python venv support. The existing project `.venv` is already repaired and runnable. Optional Isaac Lab, SuperMimic, FLORA, and Cosmos work will use separate environments; see `requirements/README.md` and `docs/research.md`.

## Code layout

```text
src/shadowbox/
  app.py                 # CLI and live simulation loop
  config.py              # shared YAML settings and logging
  simulation/            # model construction and world state
  motion/                # action vocabulary and smooth trajectories
  boxing/                # independent policies, match, touch geometry
  safety/                # root paths, destination reservations, overrides
  ui/                    # synchronous viewer and HUD
  evaluation/            # seeded evaluation and policy parameter search
```

## Current limits

The bases stay upright. G1 footwork uses planted/swinging foot targets and IK,
but does not solve physical walking or balance. Gravity compensation is active
and collision response is disabled. Hits are virtual tags from simulated glove
trajectories. The simplified 10–9/10–10 scorecards use touch counts and do not
reproduce official boxing judging. Motion limits constrain commanded setpoint
trajectories; physical joint-speed, contact-force and balance validation remain
future work.

The 29-joint Unitree G1 and reference-motion replay are implemented. Contact
dynamics and free-base balance are later gates; voice control is deferred.
See [the documentation index](docs/README.md) and
[architecture](docs/architecture.md) for current implementation and next gates.

## Optional HoloBoxing benchmark

`external/HoloMotion` pins the upstream HoloMotion/HoloBoxing implementation.
`shadowbox-holoboxing` provides config-driven separate-environment stages,
readiness diagnostics, command previews, and official Docker validation/matches.
See [setup and commands](requirements/README.md#holoboxing--holomotion-benchmark)
and `configs/holoboxing.yaml`. Reference training requires a Linux x86-64
CUDA 12.8 workstation; no training stack is installed on this ARM64 Jetson.
The upstream pose-target policy is a separate benchmark and is not wired into
the local discrete-action match controller.

## Multi-robot motion tracking with MorphMimic

`external/MorphMimic` pins the shared G1/X2 Ultra/T2 tracker. The
`shadowbox-morphmimic` command provides isolated upstream training/export and CPU
simulation stages, and `shadowbox-motion-eval --morphmimic --transitions ...`
exports canonical measured-state diagnostics from the existing G1 simulator.
See [setup and contracts](docs/morphmimic.md) and
[the evolving architecture](docs/architecture.md#multi-robot-motion-layer-implemented-boundary-and-adoption-plan).
Trained policy bundles and per-robot free-base/contact validation are required
before adopting the shared tracker for boxing or the later sim-to-real phase.
