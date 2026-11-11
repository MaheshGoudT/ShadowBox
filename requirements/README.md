# Python environments

The primary scalable simulation/RL integration is now
[MuJoCo Playground](../docs/playground.md), pinned under
`external/mujoco_playground`. Install it in `.venv-playground` as documented
there; the CPU baseline setup below remains independent. ManiFlow adds no
required dependency.

Run from the ShadowBox repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/dev.txt
python -m pip install --no-deps -e .
```

Use `requirements/base.txt` instead of `dev.txt` when tests are not needed. These files pin directly used packages from the working Python 3.12 ARM64 environment. They are not a cross-platform lockfile with hashes. `pyproject.toml` declares the package and broader supported ranges. Regenerate and verify pins when upgrading.

For tactical PPO on CPU, install `requirements/learning.txt` in the same environment. It adds PyTorch; simulation and the rule-based demo do not require it.

If Debian/Ubuntu reports missing `ensurepip`, its OS `python3-venv` support is missing. The current `.venv` has already been repaired; activation and pip work. Do not recreate it just to follow fresh-install instructions.

## Optional projects

| Experiment | Environment | Setup authority | Status |
| --- | --- | --- | --- |
| MuJoCo Playground (primary scalable simulation/RL) | Separate `.venv-playground` CPU / `.venv-playground-gpu` CUDA, Python 3.12/3.13 | [Setup and integration](../docs/playground.md) | Two-G1 CPU/GPU smoke, masked reset and PPO plumbing pass; sustained balance/defense pending |
| MorphMimic shared humanoid tracker | Separate Python 3.11 training/export and CPU runtime environments | [Setup and integration](../docs/morphmimic.md) | Canonical bridge and launcher integrated; trained-bundle reproduction pending |
| HoloBoxing / HoloMotion | Separate training and retarget environments | [Pinned upstream README](../external/HoloMotion/holoboxing/README.md) | Launcher verified; reference GPU training pending |
| Isaac Lab GPU training | Separate `tmp/isaaclab/venv` | [Setup and integration](../docs/isaaclab.md) | Launcher integrated; GPU simulation not yet validated here |
| SuperMimic tracking | Chosen reproduction's checkout/environment | [Research notes](../docs/research.md#supermimic-later-motion-layer) | Integration candidate |
| FLORA annotation | Separate annotation environment/service | [FLORA project](https://research.nvidia.com/labs/gear/flora/) | Optional offline experiment |
| Cosmos | Separate upstream environment | [Cosmos](https://www.nvidia.com/en-us/ai/cosmos/) | Deferred |

Add a named requirements file and reproducible setup notes when an integration actually begins. For experiments without a named requirements file, no speculative package pins are supplied: accelerator builds, upstream revisions, assets, and licenses remain to be selected. Do not mix these stacks into the working MuJoCo environment or assume a similarly named PyPI package is the intended research implementation.

## HoloBoxing / HoloMotion benchmark

The pinned submodule is `external/HoloMotion` at
`f52dd81ae935a828b0296f5ee0f833beb7378bf0`. Follow its
[official HoloBoxing README](https://github.com/HorizonRobotics/HoloMotion/blob/f52dd81ae935a828b0296f5ee0f833beb7378bf0/holoboxing/README.md)
for resource hashes, data licenses, training stages, and evaluation rules.
The reference trainer needs Linux x86-64, Python 3.11, an NVIDIA GPU and a
CUDA 12.8-compatible driver. This ARM64 Jetson has source/command-preview support;
reference training and Docker GPU evaluation have not been validated here.
Do not install this stack into ShadowBox's `.venv`.

On a supported workstation, from the ShadowBox root:

```bash
git submodule update --init external/HoloMotion
python3.11 -m venv .venv-holoboxing
.venv-holoboxing/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv-holoboxing/bin/python -m pip install -r requirements/holoboxing-training.txt
.venv-holoboxing/bin/python -m pip check
python3.11 -m venv .venv-holoboxing-retarget
.venv-holoboxing-retarget/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv-holoboxing-retarget/bin/python -m pip install -r requirements/holoboxing-retarget.txt
.venv-holoboxing-retarget/bin/python -m pip check
```

Training pins Warp 1.11; HoloRetarget/Newton needs Warp 1.12. Keep both
interpreters separate. The wrapper requirements reuse the pinned upstream files;
they are not independent locks. Configure interpreters, device, resource paths,
and the evaluation image in `configs/holoboxing.yaml`. Pin the published image by
digest for a reproducible pilot and retain resource inventories, data splits,
checkpoint hashes and reports. The tag alone is not an immutable image identity.

Use `.venv/bin/python -m shadowbox.integrations.holoboxing` in place of
`shadowbox-holoboxing` until the editable package entry points are refreshed.
`doctor` exits 1 if the checkout, host, or either runtime is not ready; absent
training environments are expected here. It probes only the selected interpreters.

```bash
shadowbox-holoboxing doctor
shadowbox-holoboxing run --dry-run supervised -- --assets '{assets}' --lower '{lower}' --data '{data}' --output '{output}' --device '{device}'
shadowbox-holoboxing run export -- --checkpoint /absolute/checkpoint.pt --output /absolute/ShadowBox/tmp/holoboxing/results/boxing.onnx
shadowbox-holoboxing validate tmp/holoboxing/results/boxing.onnx --benchmark
shadowbox-holoboxing match tmp/holoboxing/results/boxing.onnx tmp/holoboxing/results/opponent.onnx --output tmp/holoboxing/results/match-001
shadowbox-holoboxing view tmp/holoboxing/results/match-001/replay.hdf5
```

`run` allowlists assets, download, retarget, prepare, verify, prepare-prior, prior,
supervised, rl and export. Pass official upstream flags after `--`; each exact
`{assets}`, `{lower}`, `{data}`, `{output}`, or `{device}` argument expands from
config. Other arguments pass unchanged. Relative upstream argument paths resolve
in `external/HoloMotion`; use absolute paths or the config placeholders for
outputs under ShadowBox `tmp/`. No shell expansion is performed. The config
provides paths, not stage hyperparameters: retain the complete invocation with
results. Use distinct output paths for stages/runs; upstream refuses overwrites.
Dry runs print commands without requiring the separate environments or CUDA.

Acquire the asset submodule and matching controller using the upstream resource
instructions before preparation. Keep raw/converted/prepared data and downloaded
controller files under ShadowBox `tmp/holoboxing/`; substitute these absolute
paths for upstream tutorial defaults. Resource checkout files remain upstream
assets. InterHuman and SMPL have separate access/license conditions; do not
redistribute them. The smoke data checks plumbing, not boxing quality.

Validation/matches call the official `holoboxing/run.sh` Docker evaluator with
the configured image. Docker, NVIDIA Container Toolkit and GPU access are
separate prerequisites. Benchmark latency on the actual GPU; a result from a
non-reference GPU does not establish official RTX 5090 qualification. Inspect
`validation.json`, `match.json`, and replay before comparing with ShadowBox.

## User-wide Hugging Face CLI

The `hf` command is installed under `~/.local/bin/hf` using an isolated uv tool
with `huggingface_hub==2.4.0`; the version is recorded in
`requirements/huggingface-cli.txt`. Reproduce with:

```bash
uv tool install 'huggingface_hub==2.4.0'
hf --version
hf cache ls --cache-dir /home/mahesh/Desktop/huggingfacehub/hub
```

Use `/home/mahesh/Desktop/huggingfacehub` for shared model storage, as configured
in `configs/simulation.yaml`. Supply `--cache-dir` on CLI downloads or use
`HF_HOME=/home/mahesh/Desktop/huggingfacehub` and
`HF_HUB_CACHE=/home/mahesh/Desktop/huggingfacehub/hub` for the specific process.
Global shell settings and credentials were not changed. Existing Qwen bundles
are language-model resources, not validated G1 motion-control weights. Any
motion bundle still needs its native observation/action contract, source,
license, revision and tracking checks.

Official CLI install reference:
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

The separate Metatron G1 export/runtime requirements and pinned artifact commands
are documented in [the motion setup guide](../docs/metatron-playground.md).
