# Qualified G1 motion artifact path

The current ShadowBox motion task uses the G1 package from NVIDIA Metatron v1.
This is an optional, separately licensed model resource below the tactical
policy. It does not replace boxing rules, movement qualification or evaluation.
All commands below run from the ShadowBox repository root. Scripts currently
use the explicitly recorded local bundle/source paths; they are qualification
harnesses for this pinned profile, not a general model installer.

## Pinned inputs

| Input | Identity |
| --- | --- |
| ProtoMotions source | `13de0cccd06f53ce5c9cf9ec7d1932c7da73610f` |
| Hugging Face repository | `nvidia/Metatron-v1` |
| Model revision | `ada816723d62c194c82ffd8fab97a3d8a001f94d` |
| G1 checkpoint SHA256 | `e5fc2e47a8f9ee9b96a674cabbf236b3c510f887427645a14bc1779b10f0b43a` |
| Native source path | `tmp/playground/protomotions-source` |
| Exported bundle | `tmp/playground/metatron/compiled` |

Keep the G1 package's license, notice, model card, config and checkpoint together.
The existing download is under
`/home/mahesh/Desktop/huggingfacehub/models/nvidia/Metatron-v1/ada816723d62c194c82ffd8fab97a3d8a001f94d/G1`.
`hf` is installed user-wide in `~/.local/bin`. A fresh pinned download can use:

```bash
hf download nvidia/Metatron-v1 --revision ada816723d62c194c82ffd8fab97a3d8a001f94d --include 'G1/*' --local-dir /home/mahesh/Desktop/huggingfacehub/models/nvidia/Metatron-v1/ada816723d62c194c82ffd8fab97a3d8a001f94d
```

The already downloaded files were checked against the official native package
manifest. Checkpoint export additionally rejects an unexpected checkpoint hash.
Do not change ownership of the shared Hub cache to work around lock permissions;
the writable `models/` directory holds this native package.

## Separate environments and artifact generation

The native MuJoCo/ProtoMotions stack and the current Playground stack have
incompatible dependency versions. Install them separately:

```bash
uv venv --python 3.12 .venv-protomotions
uv pip install --python .venv-protomotions/bin/python --index-strategy unsafe-best-match -r requirements/protomotions-inference.txt
uv venv --python 3.12 .venv-protomotions-gpu
uv pip install --python .venv-protomotions-gpu/bin/python --index-strategy unsafe-best-match -r requirements/protomotions-inference-gpu.txt
uv pip install --python .venv-playground-gpu/bin/python -r requirements/playground-motion.txt
```

Check out the pinned ProtoMotions source at the source path above before running
these harnesses. The requirements file also pins that revision for dependencies.
The mesh helper copies assets into `tmp/playground/protomotions-assets`, retrieves
pinned Git LFS blobs and verifies their pointer sizes/SHA256; it preserves source.
The graph path requires the CUDA Torch environment and actual supported hardware.

```bash
.venv-protomotions/bin/python scripts/fetch_protomotions_g1_assets.py
.venv-protomotions/bin/python scripts/metatron_export.py
.venv-protomotions/bin/python scripts/metatron_export_frames.py
.venv-protomotions/bin/python scripts/metatron_probe.py --eager --onnx tmp/playground/metatron/compiled/unified_pipeline.onnx --steps 32 --compare-native --record-all-feeds --output tmp/playground/metatron/native-onnx-feed-trace.json
.venv-protomotions-gpu/bin/python scripts/metatron_export_torchscript.py
.venv-playground-gpu/bin/python scripts/metatron_graphsafe_torchscript.py --batch-size 2
.venv-playground-gpu/bin/python scripts/metatron_cuda_graph.py --graphsafe --batch-size 2
```

The final command now checks parent/candidate/sidecar/ONNX lineage, finite output
shapes, replay error and CUDA graph capture, then emits the matching runtime JSON
qualification automatically. It does not qualify physical boxing. Failed parity
or unsupported CUDA stops the harness. The original TorchScript file is preserved.
The derivative is specialized to batch two and must not be used for other sizes.
Standard ONNX Runtime CUDA failed on this Thor device; the CPU ONNX path is used
for FK, and qualified TorchScript supplies the frozen controller.

The CUDA dependency check currently flags NVIDIA cuSPARSELt's vendor wheel tag;
this known metadata issue remains visible. Actual Torch CUDA execution passes.
Do not report the full GPU dependency check as clean or edit installed metadata.

## Physical checks and replay

```bash
.venv/bin/python -m shadowbox.integrations.playground evaluate --config configs/playground-motion.yaml
.venv/bin/python -m shadowbox.integrations.playground evaluate --config configs/playground-motion-exposed.yaml
.venv/bin/python scripts/playground_defense_gauntlet.py --config configs/playground-motion-varied.yaml --distances .55 .6 .7
.venv-playground-gpu/bin/python src/shadowbox/playground/replay.py tmp/playground/runs/RUN_ID --frames 100 --stride 5 --output tmp/playground/replay-10s.gif
```

Replays use each run's exact saved compiled model and recorded positions. GIF
stride five displays one frame per 100 ms; no reconstructed current scene is
substituted. See [architecture](architecture.md) for the current task boundary;
measured standing/contact results and failed candidates are recorded in the local
ignored `docs/planning.md`. The current authored movement fixture uses world
velocity, heading, height and timed alternating foot goals, following the
[published Metatron steering pattern](https://research.nvidia.com/labs/par/projects/metatron/).
At a 0.70 m start, a 0.5 s forward approach produces two to three classified
head-jab contacts per side across sampled yaw/lateral and swapped-start
controls; an immediate standstill jab still misses. One paired
movement sweep remained upright and showed directional steps. The backward
fixture reads its 0.27/0.18 m/s side speeds from validated motion-config options;
four paired reset seeds and a swapped-start pair produce nearly equal retreat
distances without falls.
This is a fixed-layout diagnostic calibration; facing drift and foot travel
still need qualification. The learned actor still selects guard/body guard
only. Current host execution is below real time;
learned defense, diverse-opponent, referee, movement-policy, and sim-to-real
gates remain open.
