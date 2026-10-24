# MorphMimic integration

## Why it is useful

[MorphMimic](https://lzyang2000.github.io/morphmimic/) addresses the intended
multi-robot motion layer: a shared tracker/world model conditions on a robot's
actuation, geometry and mass, while reference motions, normalization statistics
and replay buffers stay specific to each robot. Its published scope is three
humanoids: Unitree G1, AgiBot X2 Ultra and Booster T2. It is an independent
SuperMimic implementation. Its G1-only transfer experiments fail on other bodies,
so the result does not justify loading a G1 policy onto an arbitrary new robot.

The [technical report](https://lzyang2000.github.io/morphmimic/static/morphmimic.pdf)
reports shared tracking in simulation and full-motion G1 hardware demonstrations.
X2/T2 hardware is pending. Neither those results nor ShadowBox's virtual touches
establish boxing contact or balance readiness on hardware.

## Implemented in ShadowBox

- `external/MorphMimic` is a Git submodule pinned at
  `aa40fe46a7fa995ba7a7354ad065a2c8ad64a302`.
- `shadowbox-morphmimic` launches upstream conversion, subset preparation,
  training, export, asset export and CPU MuJoCo simulation. Separate interpreter
  paths and all artifact roots live in `configs/morphmimic.yaml`.
- `motion/morphology.py` reuses upstream's NumPy-only `obs.py` and official
  robot constants for native/canonical action mapping and measured encoder/IMU
  capture. No mjlab, Torch or ONNX package is imported for this bridge.
- `Arena.full_velocity()` and `shadowbox-motion-eval --morphmimic` add canonical
  measurements to the existing transition-export path. The G1 elbow coordinate
  offset and local IMU gyro frame are handled explicitly.

The current two-player Arena has a G1 model. X2/T2 profiles can be inspected and
used with the upstream free-base sim2sim path; they cannot be selected as Arena
boxers yet. The existing gait and `Primitive` match controller remain the local
baseline. A trained ONNX export and matching reference/scene assets are required
for an actual shared-tracker simulation. They are not manufactured by cloning
the source or by generating a ShadowBox trace.

## Source and environments

From the ShadowBox root:

```bash
git submodule update --init external/MorphMimic
.venv/bin/python -m shadowbox.integrations.morphmimic doctor
.venv/bin/python -m shadowbox.integrations.morphmimic robots
```

`doctor` returning zero verifies the pinned observation source only. It reports
whether interpreter and export files exist, without claiming CUDA or model
readiness. `robots` shows canonical/native joints, IMU body and control period.
Missing or locally modified upstream observation code fails clearly.

Training/export follows the pinned [official README](../external/MorphMimic/README.md)
and `uv.lock`: Python 3.11, mjlab 1.6.0, CUDA Torch cu128, Warp below 1.18, and an
NVIDIA GPU. The reference three-robot recipe uses three GPUs; the README reports
about 28 GB per GPU even for its 1 h subset at 4096 environments. Start with a
small reproduction on suitable hardware. No training stack is installed on this
ARM64 Jetson by this integration.

On the training workstation, initialize the workspace dependency and create an
isolated environment using upstream's lock:

```bash
git -C external/MorphMimic submodule update --init third_party/GMR
UV_PROJECT_ENVIRONMENT="$PWD/.venv-morphmimic" uv sync --project external/MorphMimic --frozen --group export --no-dev
```

CPU simulation uses its own smaller environment; binary availability must match
the host architecture:

```bash
python3.11 -m venv .venv-morphmimic-runtime
.venv-morphmimic-runtime/bin/python -m pip install -r requirements/morphmimic-runtime.txt
.venv-morphmimic-runtime/bin/python -m pip check
```

The runtime requirements pin the direct packages in upstream's lock; they are
not a separate transitive lock. Upstream `deploy` modules are run with the
submodule as their working directory, so the namespace resolves without adding
the training project to ShadowBox's core environment. Refresh ShadowBox's
editable install to get `shadowbox-morphmimic`, or use
`.venv/bin/python -m shadowbox.integrations.morphmimic` for every command below.

## Commands and artifact paths

`--dry-run` goes before the stage. `{data}`, `{output}`, `{assets}` and
`{deployment}` expand from config, including path suffixes. Paths are absolute
after expansion; ordinary relative upstream flags resolve inside its checkout.
Generated data, checkpoints, exports and reports belong under ShadowBox `tmp/`.
The launcher redirects upstream log/asset/output-directory defaults there.
Specify export's output directory explicitly as its final positional argument.

```bash
shadowbox-morphmimic run --dry-run convert -- --robot g1 --csv-dir '{data}/seed/g1/csv' --metadata '{data}/seed/seed_metadata_v003.csv'
shadowbox-morphmimic run subsets -- --manifest '{data}/g1/manifest.csv'
shadowbox-morphmimic run --dry-run train -- --robots g1 x2 t2 --subset-yamls '{data}/g1/subsets/1h.yaml' '{data}/x2/subsets/1h.yaml' '{data}/t2/subsets/1h.yaml' --share all --gpu-ids '[0,1,2]' --kp-interface --kp-masking full --run-name shadowbox-unified
shadowbox-morphmimic run export -- '{output}/training/shadowbox-unified' '{deployment}' --iteration 40000
shadowbox-morphmimic run assets -- --robot g1 --subset-yaml '{data}/g1/subsets/1h.yaml' --clips YOUR_CLIP
shadowbox-morphmimic simulate --robot g1 --clip YOUR_CLIP --interface motion --delay-ms 20 --output tmp/morphmimic/results/g1-motion-delay20.json
```

Use X2/T2-specific retargeted subsets and export their assets before selecting
them. A shared bundle contains one `policy.onnx`, one JSON per trained robot,
`MANIFEST.txt` and `SHA256SUMS`. Keep each JSON with the model: it contains the
robot's normalization inputs, morphology and action/sensor constants. Exporter
parity and matching source/data/resource identities are prerequisites to the
simulator benchmark. A previously generated summary is not overwritten.

Training data is BONES-SEED, with its own access and license. X2/T2 use
robot-specific retargeting through upstream's SOMA scripts. Code is Apache-2.0;
vendored models, datasets and nested repositories retain their licenses.

## Canonical diagnostic traces in the main evaluator

```bash
shadowbox-motion-eval assets/motions/boxing-jab-example.json --morphmimic --transitions tmp/morphmimic/traces/g1-jab.npz --output tmp/morphmimic/results/g1-jab.json
```

The NPZ retains `qpos`, `qvel`, `ctrl` and their next states, and adds
`canonical_q[N,29]`, `canonical_dq[N,29]`, `canonical_action[N,29]`,
`proprioception[N,93]` and `next_proprioception[N,93]`. Observation precedes the
current command: its last-action field reconstructs the previously held native
targets, and the next observation sees the current command. This is a diagnostic
trace sampled at ShadowBox's physics timestep, **not** upstream's 240-value
world-model training state or a 50 Hz policy history. The metadata records both
the sample timestep and profile control period. Do not train a tracker by
passing this NPZ directly to the upstream data loader.

Only an actual exported bundle supplies fitted descriptor inputs. Add
`--morphmimic-deployment /absolute/export/folder` to record its 125-value
morphology descriptor when present; constants-only capture marks it unavailable.
No descriptors or normalization statistics are invented from the modified,
supported-root boxing model.

## Next adoption gates

1. Reproduce full-motion and five-point tracking per robot in upstream's
   contact-enabled/free-base simulator, using matching bundles and clips.
2. Compare joint/body tracking, sole slip, falls, drift, latency and actuator
   limits at zero and delayed control. Preserve reports and source identities.
3. Build a boxing-skill compiler from measured reference motions to the tracker's
   goals. Keep the tactical `Primitive` interface above it; compare active boxing,
   defense and safety against the current local baseline before replacing it.
4. Add each new humanoid's semantic joint/body map, IMU frame, actuator descriptor,
   retargeted motions and simulator tests. The 29-slot/14-body assumption must
   fit; unsupported geometries need a different representation or adaptation.
5. Later, calibrate and qualify sim-to-real per robot. Begin from validated
   standing/tracking before boxing; physical balance, contact and actuator
   evidence are separate from the tactical promotion gate.

HoloBoxing remains a complementary high-level boxing benchmark. Its
`obs[1097] -> target_pose[225]` contract differs from MorphMimic's ten-frame
450-value five-point goal and canonical joint-action output. Point order,
orientation convention, normalization and timing need an explicit tested
conversion; shared use of five points does not make the interfaces compatible.
