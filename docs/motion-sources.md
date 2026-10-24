# G1 motion and policy sources

**Current selection:** [MorphMimic](morphmimic.md) is integrated for the shared
multi-humanoid motion layer; HoloBoxing supplies a complementary boxing benchmark.
See [architecture](architecture.md) for current gates. The catalog below is an
October 24 assessment of alternatives; its ordering is historical and is not
the active plan. Recheck artifacts, runtimes and licenses before adopting one.

Checked 2026-10-24 against the official Hugging Face dataset/model cards and official NVIDIA/Unitree project repositories. This research is for a Unitree G1 29 DoF MJCF. No files were downloaded; shell network access could not resolve GitHub in the prior asset-fetch attempt. No dependencies were installed.

## Historical integration candidates

### 1. Immediate gameplay: GEAR-SONIC kinematic planner (boxing is explicitly present)

**Best match for a playable humanoid with authored Kung Fu/boxing behavior.** NVIDIA’s keyboard planner exposes a `Boxing` set with `Idle Boxing`, `Walk Boxing`, `Left Jab`, `Right Jab`, `Random Punches`, `Left Hook`, and `Right Hook`. This is genuine released functionality, not an inference from a general motion tracker. The controller can be steered with keyboard/gamepad. NVIDIA's docs give a MuJoCo sim2sim launch path and explicitly list these boxing modes.

- Model hub: https://huggingface.co/nvidia/GEAR-SONIC
- Official code/runtime: https://github.com/NVlabs/GR00T-WholeBodyControl
- Keyboard and MuJoCo sim instructions: https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/keyboard.html
- Kinematic planner reference: https://nvlabs.github.io/GR00T-WholeBodyControl/reference/kinematic_planner.html (linked by the keyboard guide; verify current docs path)
- Planner artifact: `planner_sonic.onnx` in the `nvidia/GEAR-SONIC` model repository; official download code supports default policy + planner (`python download_from_hf.py`). Exact direct file URL: `https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/planner_sonic.onnx` (repository may require LFS resolve redirect).
- Policy artifacts: `model_encoder.onnx`, `model_decoder.onnx`, `observation_config.yaml`; use a matched set. Other variants live under `low_latency/` and `sonic_v1_1/`.
- Joint/action interface: 29 DoF action decoder, 64-D universal motion tokens; 50 Hz controller. Planner supports named modes and controls. Full policy input layout is defined in variant `observation_config.yaml` and deployment runtime; this is not a drop-in generic MuJoCo actuator policy.
- License: Hugging Face lists `nvidia-open-model-license` for model weights. Review that license before use. NVIDIA's repo includes the deployment and MuJoCo sim loop, which is the practical integration route.
- Runtime: TensorRT/C++ deployment; documented for desktop NVIDIA GPU and Jetson. Do not assume CPU-only or browser/WASM support. A CPU port would need a separate implementation and is not provided by the official deployment path.
- Limit: it is a whole-body motion behavior planner, not a combat/gameplay system with hitboxes, opponent reaction, or combat decision-making. The named boxing modes are confirmed; the behavior quality/interactive combat semantics are not implied.

This is the recommended path when the immediate goal is responsive gameplay. For an existing browser game, the upstream stack is not a small asset import; either run it as a separate native controller and bridge 29 joint targets, or keep gameplay responsive using the app's current procedural controls and import poses/clips only after the runtime boundary is designed.

### 2. Later offline imitation: ProtoMotions G1 BONES-SEED deployment tracker

**Strongest open MuJoCo inference example for tracking a reference motion on the same G1.** NVlabs provides a trained G1 29-action tracker, a unified ONNX export, input/output metadata, a small sample motion package, and a standalone MuJoCo test runner. The deployment card says it is a motion tracker, not a generator. It requires the target to preserve the G1 joint order, observation preprocessing, 50 Hz control rate, PD gains, and action scaling.

- Tracker card/checkpoint directory: https://github.com/NVlabs/ProtoMotions/tree/main/data/pretrained_models/motion_tracker/g1-bones-deploy
- ONNX metadata: https://github.com/NVlabs/ProtoMotions/blob/main/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.yaml
- Standalone MuJoCo ONNX runner: https://github.com/NVlabs/ProtoMotions/blob/main/deployment/test_tracker_mujoco.py
- Quickstart with miniature BONES-SEED and sample retargeted AMASS motion files: https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/quickstart.rst
- Checkpoint: `data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt`; ONNX: `compiled_models/unified_pipeline.onnx`; example reference: `data/motion_for_trackers/g1_bones_seed_mini.pt`.
- ONNX contract from shipped metadata (batch size 1): current torso anchor quaternion `[1,4]` in xyzw; joint position `[1,29]`; joint velocity `[1,29]`; pelvis local angular velocity `[1,3]`; processed-action history `[1,1,29]`; four future torso orientation frames `[1,4,4]`; future joint position `[1,4,29]`; future joint velocity `[1,4,29]`. Output includes `joint_pos_targets [1,29]` plus `actions`, stiffness and damping vectors. Reference future times 0.02, 0.04, 0.08, 0.16 s. Control dt 0.02 s; physics dt 0.001 s. Joint names are present in the YAML alias `*id001`; the G1 order matches the 29 revolute joints in the MJCF.
- CPU/GPU: the official MuJoCo runner explicitly creates ONNX Runtime `CPUExecutionProvider`; this is the most practical CPU-capable pretrained tracker found. Its stated dependencies are `mujoco`, `onnxruntime`, `numpy`, `pyyaml`, `torch` (for `torch.load` only); caching a motion to 50 fps removes ProtoMotions imports from later runs. This still requires an ONNX inference runtime and a compatible MuJoCo model/control setup.
- License/data provenance: repository/model card describes training data as BONES-SEED retargeted to G1. The checkpoint card does not establish a separate weight license in the inspected page; verify repository/model license before redistribution. The BONES-SEED dataset itself requires acceptance of its own license and is gated; it is not the same as a permissive public CSV library.
- Limit: it needs a reference motion in the tracker’s format and sensor-like observations. It does not autonomously invent boxing or produce a sequence without motion reference.

### 3. Optional small reference clip: LAFAN1 G1 sprint sample

A specific publicly viewable G1 CSV is small enough for a later fetch: `g1/sprint1_subject4.csv`, listed at 2.79 MB. It is an ordinary locomotion sample, not a boxing or combat motion. This is only a candidate for locomotion/playback or format validation.

- Dataset page: https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
- Exact file page: https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/blob/main/g1/sprint1_subject4.csv
- Exact download URL: https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1/sprint1_subject4.csv
- Reported file size: 2.79 MB.
- Format: CSV, G1 at 30 fps; root XYZ and quaternion components in `root_joint(XYZQXQYQZQW)`, then the 29 joints in G1 order. The listed order is 6 left leg, 6 right leg, 3 waist, 7 left arm, 7 right arm; the exact names are in the dataset page. The retargeting documents kinematic constraints but not dynamics or actuator limits, so its raw trajectories are not guaranteed dynamically feasible.
- License caution: HF card says the source LAFAN1 motion is CC BY-NC-ND 4.0. The inspected card does not declare a distinct license for the retargeted G1 trajectory file. Retargeting is a derivative transformation; CC BY-NC-ND is not a safe permission basis for modified/redistributed clips. Do not stage/use this file in a shipped product without confirming the dataset's applicable rights with the maintainer/rightsholder. No clip was downloaded.

### 4. BONES-SEED G1 reference corpus (later, with eligibility)

BONES-SEED has G1 MuJoCo-compatible CSVs and a metadata catalog; HF card describes 142,220 motions and includes a 20-motion `Martial Arts` category. This confirms there are martial-arts-tagged samples in the corpus, but does not establish that any particular one is a boxing guard/jab, so do not present it as a verified boxing clip. Its full G1 export is a tar archive and the dataset is gated, therefore it is not a small immediately fetchable CSV option.

- Dataset: https://huggingface.co/datasets/bones-studio/seed
- License: https://bones.studio/info/seed-license
- Download page requires account-information sharing and explicit acceptance. The agreement says access is limited to (a) academic users performing non-commercial publicly available research or (b) qualifying startups under $1M annual gross revenue; other users need a separate commercial license. It prohibits redistributing raw data. Do not download/accept without confirming eligibility.
- G1 CSV layout per NVlabs guide: fields include `Frame`, root translation (cm), root extrinsic XYZ Euler rotation (deg), then `<joint>_dof` (degrees) matching G1 MJCF order. Dataset main README lists 120 fps source motion; NVlabs' ingestion scripts convert/sample G1 CSV to model-specific 30 fps motion files. Check the exact file/readme version when integrating.
- Dataset tags: 142,220 motions, 120 fps source rate, G1 CSV retarget, Motion categories include 20 Martial Arts. Official NVIDIA/ProtoMotions examples know how to convert the corpus to `.motion`/MotionLib `.pt` for tracker inference.

## Compatibility and work strategy

The G1 MJCF used previously has 29 actuated joints ordered by the standard G1 schema. Preserve exact left leg → right leg → waist → left arm → right arm ordering when mapping targets. The fixed rubber hand meshes are visual only; there is no articulated hand joint.

For immediate gameplay, the best researched behavior is GEAR-SONIC's Kinematic Planner (boxing styles included), but its runtime brings a native C++/TensorRT and ONNX planner/decoder stack. For later offline imitation, ProtoMotions' standalone MuJoCo ONNX runner is a more compact reference: it runs ONNX Runtime CPU and includes an established MuJoCo loop, but follows input motion rather than generating a combat behavior. G1 sample CSVs make a plausible animation/format-validation asset, subject to the rights issues above.

No checkpoint or dataset was downloaded here. Large weight archives were not fetched; no installs were run.
