# ShadowBox documentation

- [Architecture](architecture.md): authoritative current implementation,
  tactical/motion boundaries, multi-robot adoption and remaining gates.
- [MorphMimic](morphmimic.md): pinned integration, environments, commands,
  canonical diagnostics and reproduction limits.
- [Environment setup](../requirements/README.md) and
  [configuration](../configs/README.md): runnable commands and config ownership.
- [Engine choice](engine-choice.md): CPU/GPU roles and the dated backend review.
- [Research decisions](research.md), [motion sources](motion-sources.md) and
  [reference target](reference-target.md): dated source assessments, with current
  decisions owned by the architecture.
- [Experiments](experiments.md): remaining measured questions.

`planning.md` is a local execution plan, ignored by Git. Generated logs,
checkpoints, reports and captures stay under the ignored `tmp/` directory.
Historical test counts and experiment results do not override current code or
its source/checkpoint identity checks.
