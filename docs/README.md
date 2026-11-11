# ShadowBox documentation

- [MuJoCo Playground](playground.md): primary scalable simulation/RL backend,
  pinned source, isolated runtime, task boundaries and optional chunk benchmark.
- [Architecture](architecture.md): authoritative current implementation,
  tactical/motion boundaries, multi-robot adoption and remaining gates.
- [Environment setup](../requirements/README.md) and
  [configuration](../configs/README.md): runnable commands and config ownership.
- [Research decisions](research.md), [motion sources](motion-sources.md) and
  [reference target](reference-target.md): research rationale, candidate assets
  and movement goals; current implementation status stays in the architecture.
- Model setup: [Metatron](metatron-playground.md),
  [MorphMimic](morphmimic.md), and the optional [Isaac Lab](isaaclab.md) path.
- [Legacy capability removal gates](legacy-retention.md): callers, value-add,
  checks and replacement requirements.

`planning.md` is the active local execution plan and is intentionally ignored
by Git. The tracked architecture and setup guides describe the shared system;
the local plan contains detailed run evidence and next implementation steps.
Generated logs, checkpoints, reports and captures stay under ignored `tmp/`.
Historical test counts and experiment results do not override current code or
its source/checkpoint identity checks.
