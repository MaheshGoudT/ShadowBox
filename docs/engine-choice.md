# ShadowBox engine choice: MuJoCo now, Isaac Lab experiment later

**Current decision:** CPU MuJoCo and small tactical PPO rollouts/training are
working. MorphMimic's exported-policy CPU replay is wired but still requires a
trained bundle and runtime verification. MorphMimic/HoloBoxing reference training
uses separate GPU stacks; see [setup](../requirements/README.md). The detailed
backend/version comparison below is an October 24 research snapshot, not a
current installation or hardware-compatibility guarantee. Recheck upstream
requirements when a measured need justifies a new backend.

**Checked:** 2026-10-24. **Machine context:** Linux aarch64 / Jetson Thor, Python 3.12; MuJoCo 3.17 CPU headless and Mesa viewer work. `nvidia-smi` in the current tools failed with `NvRmMemMgr` / driver-access errors; that result does not establish whether the host GPU is physically absent or unavailable to this environment.

## Recommendation

Keep the working MuJoCo CPU setup for the current lightweight/tabular RL and boxing-task iteration. Do not install Isaac Sim now. When the policy or experiment needs many parallel rollouts, run a small, isolated Isaac Lab 3.0 kit-less/Newton + MJWarp experiment, after read-only checks confirm the host’s CUDA driver/runtime is exposed and supported. This tests GPU batching without committing the project to a new environment or full Isaac Sim install.

Try Isaac Sim only if you later need its Kit/PhysX-specific workflow, high-fidelity scene/sensor pipeline, or a relevant Isaac Sim asset path that the kit-less backend cannot support. A more numerous Unitree joint set alone is not a sufficient trigger: Isaac Lab includes H1 and G1 assets, but the actual boxing setup needs its own articulation, actuator, contact, and task validation. Keep a MuJoCo control baseline to compare contact outcomes and training throughput.

## Decision table

| Need | Current MuJoCo 3.17 CPU | Isaac Lab 3.0 kit-less / Newton MJWarp | Isaac Sim + Isaac Lab |
|---|---|---|---|
| CPU baseline / small tabular RL | Best fit today: already working, low setup cost, good single-step latency. MuJoCo also has a CPU thread-pool batch API. | Possible, but GPU-oriented Warp path adds setup and overhead without a measured batch need. | Poor fit for current need; substantially larger stack. |
| GPU-batched neural RL / imitation later | Standard MuJoCo CPU remains useful as baseline; MuJoCo also documents MJWarp for NVIDIA CUDA parallelism. | Best first GPU experiment: Isaac Lab 3.0 defaults to Newton and supports MJWarp/CUDA graphs; kit-less install avoids Isaac Sim. Isaac Lab offers RL integrations and learning-from-demonstration workflows. | Viable GPU training, but brings Kit/PhysX and rendering dependencies that are not needed for the stated no-photorealism goal. |
| Unitree joint coverage | Supports custom models; existing ShadowBox asset/task work likely transfers conceptually, not automatically. | H1 and G1 are listed as included humanoid assets; custom USD/URDF/MJCF assets can be brought in subject to backend support. Confirm exact G1 variant, actuated joint list, limits, and hands before choosing. | Also includes Isaac Lab’s H1/G1 ecosystem; use if the needed import/asset/render workflow requires Isaac Sim. |
| Balance, impacts, contact | MuJoCo is the known baseline; preserve it for reproducible contact comparison. | MJWarp uses MuJoCo solver through Newton, but Isaac Lab/Newton has backend-specific support and tuning. Validate gloves/forearms, opponent collisions, friction, solver stability, and impact observables in the actual task. | PhysX is a distinct solver path; contact behavior may differ. Requires task-specific validation and likely retuning. |
| Visuals | CPU/Mesa viewer already adequate for iteration. | Kit-less Newton visualizers include local GL and browser/remote options; Isaac Lab docs distinguish lightweight Newton Warp visuals from RTX paths. | Appropriate when Kit/RTX scene or sensor fidelity is required. No stated need currently. |
| ARM64 and driver | Working on this machine now. | Isaac Lab 3.0 docs list Linux aarch64 (including DGX Spark), Python 3.12 and CUDA-capable NVIDIA GPU for GPU training; default CUDA 13 stack says Turing+ / compute capability 7.5+. Check Thor’s exact driver/CUDA/PyTorch compatibility and package/image architecture before setup. | Current Isaac Sim docs publish an aarch64 build, but Isaac Sim has its own driver, GPU, memory, runtime and graphics requirements. Confirm Thor is explicitly supported for the specific Isaac Sim release; don’t infer this from the kit-less Isaac Lab support. |

## Trigger and validation plan

Move to a kit-less experiment when either (a) neural RL/imitation rollouts are limited by CPU simulation throughput, or (b) a concrete requirement demands Isaac Lab’s H1/G1 asset/configuration ecosystem. Before investing, verify GPU visibility and supported architecture, then run a minimal Newton/MJWarp cartpole or locomotion smoke workflow in an isolated checkout/container. Next, port only the robot articulation and a small boxing contact scene; compare fixed-seed single-step contacts and episode outcomes against the MuJoCo baseline. Continue only if the target joint set loads and contact behavior is acceptable, and batched throughput improves enough to matter.

Consider full Isaac Sim only when a measured requirement for Kit/PhysX or its renderer/import/sensor stack cannot be met in kit-less mode. Photorealism is not a current justification.

## Read-only capability checks

These commands inspect system state; they do not install packages or alter system configuration.

```bash
uname -m
cat /etc/os-release
python3 --version
command -v nvidia-smi && nvidia-smi
command -v nvcc && nvcc --version
ls -l /dev/nvidia* 2>/dev/null
python3 - <<'PY'
import importlib.util, platform
print("arch:", platform.machine())
for name in ("torch", "warp", "mujoco", "mujoco_warp"):
    spec = importlib.util.find_spec(name)
    print(name, spec.origin if spec else "not installed")
try:
    import torch
    print("torch:", torch.__version__, "compiled CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
except Exception as e:
    print("torch probe error:", type(e).__name__, str(e))
PY
```

`nvidia-smi` failing in this container/shell is inconclusive about physical GPU presence. For the published Isaac Lab kit-less Docker image, the official install guide specifically recommends checking the tag’s architectures before selecting it:

```bash
docker buildx imagetools inspect nvcr.io/nvidia/isaac-lab:3.0.0-kitless
```

Also inspect Isaac Lab’s aarch64 support matrix and the exact JetPack/Thor driver compatibility before choosing a CUDA/PyTorch build. Do not use the regular Isaac Sim image merely because the kit-less image publishes for the architecture.

## Official sources

- [Isaac Lab 3.0 installation and system requirements](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation) — kit-less/Newton default, Python 3.12, Linux aarch64, CUDA/GPU and driver requirements; Docker image architecture inspection.
- [Isaac Lab 3.0 overview and included robots](https://docs.nvidia.com/isaaclab/3.0/isaac-lab) — RL/imitation scope and Unitree H1/G1 assets.
- [Isaac Lab 3.0 Newton backend](https://docs.nvidia.com/isaaclab/3.0/concepts/newton) — Newton’s kit-less role and backend/solver configuration.
- [Isaac Lab 3.0 renderers](https://docs.nvidia.com/isaaclab/3.0/concepts/renderers) — lightweight Newton Warp versus RTX rendering choices.
- [Isaac Lab 3.0 release notes](https://docs.nvidia.com/isaaclab/3.0/references/release-notes) — Newton powered by MuJoCo-Warp, solver options and CUDA graph acceleration; note 3.0 was identified as beta in the release notes consulted.
- [MuJoCo Warp documentation](https://mujoco.readthedocs.io/en/latest/mjwarp/) — CUDA parallel simulation, throughput tradeoffs, and complexity caveat (performance degradation beyond 60 DoFs is documented).
- [MuJoCo batch API](https://mujoco.readthedocs.io/en/stable/python.html#batch) — CPU thread-pool batch option for multiple simulations.
- [Isaac Sim 6.2 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html) and [download platforms](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/download.html) — separate Isaac Sim workstation requirements and aarch64 package availability. Isaac Sim availability on aarch64 does not by itself certify Jetson Thor as a supported workstation target.
