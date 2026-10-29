# Isaac Lab environment

ShadowBox keeps MuJoCo as its default boxing simulator. The optional Isaac Lab
3.0.0 integration launches NVIDIA's upstream G1 zero-action task as a bounded
gravity/contact smoke test. It does not yet implement a ShadowBox boxing match
or measure scoring contacts. No upstream source checkout is needed for the
published wheel. Use a supported Linux GPU system and consult the
[official installation guide](https://docs.nvidia.com/isaaclab/isaac-lab/installation)
for current platform and driver requirements. ARM64 support does not by itself
confirm a particular Jetson is supported.

From the repository root, with `uv` and Python 3.12 installed:

```bash
uv venv --python 3.12 tmp/isaaclab/venv
UV_CACHE_DIR="$PWD/tmp/uv-cache" uv pip install --python tmp/isaaclab/venv/bin/python --index https://pypi.nvidia.com --torch-backend cu130 --index-strategy unsafe-best-match -r requirements/isaaclab.txt
.venv/bin/python -m shadowbox.integrations.isaaclab doctor
.venv/bin/python -m shadowbox.app --config configs/isaaclab.yaml --backend-dry-run
.venv/bin/python -m shadowbox.app --config configs/isaaclab.yaml --headless
```

The doctor checks the isolated interpreter, Isaac Lab version, CUDA Torch, and
GPU visibility. It does not certify device compatibility. Runtime logs and a
run manifest are written in unique directories under `tmp/isaaclab/`. Edit
`configs/isaaclab.yaml` for task, environment count, step count, physics preset,
and timeout. The CLI deliberately rejects ShadowBox boxing flags until a real
boxing task is wired. A dry run only previews the command; it is not a simulation.

The integration and installation have not been validated on this host: package
installation was blocked by sandbox DNS and the GPU device was not visible in
the sandbox. These observations do not establish the host's GPU capability.
