# Research decisions

Current implementation is recorded in [architecture](architecture.md). This
page holds current technology choices, research rationale and deferred
experiments; dated source assessments remain historical and do not override the
active gates in [architecture](architecture.md). MorphMimic's
canonical bridge and source launcher, and the HoloBoxing launcher, are integrated.
Trained upstream policy behavior has not been reproduced here.

## Engine and tool choices

| System | Role in ShadowBox | Decision |
| --- | --- | --- |
| MuJoCo Playground / Warp | Articulated physics and batched training | Primary scalable backend. The paired self-play task and qualified boxing reward are still future work; see [setup and task boundaries](playground.md). |
| Local CPU arena | Boxing rules, seeded matches, tactical baseline | Keep until the free-base task replaces its measured scoring, movement, safety and interaction capabilities. |
| MorphMimic | Shared motion tracker across supported humanoids | Integrated as a separate candidate; a trained bundle and per-robot reproduction remain gates. |
| HoloBoxing | Contact-scored boxing benchmark | Isolated launcher integrated; its upstream policy is not the local boxing policy. |
| Isaac Lab | Alternative GPU robotics stack | Optional isolated smoke path; no current requirement justifies replacing Playground. |
| SuperMimic | Whole-body tracking of reference motion | Research candidate for later imitation and balance; tactics remain a separate policy. |
| Cosmos / FLORA | Video and world-model tooling | Defer until a specific data-labeling or visual-prediction need is demonstrated. |

Playground uses MuJoCo physics with a Warp-backed batched path on supported NVIDIA devices. Use its headless simulation for training and render separate evaluation runs. Keep other engines as isolated experiments unless a measured workload or asset requirement shows a concrete advantage. [MuJoCo overview](https://mujoco.readthedocs.io/en/stable/overview.html), [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)

Cosmos provides world models for physical AI. A correct numerical control/referee loop is more immediately useful here; revisit Cosmos for visual augmentation or learned video reasoning. [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/)

## Boxing rules and the first referee

World Boxing's current competition index lists March 2026 v5.0. Its PDF could not be retrieved during this review. The accessible approved November 2024 rules provide the baseline below; we do not claim compliance with the current edition. [Current rule index](https://worldboxing.org/competitions/)

The approved rules use ten-point-must scoring and consider scoring blows, technical/tactical superiority, and competitiveness. Legal targets cover the front and sides of the head and torso above the belt. Scoring blows must land cleanly with the glove's knuckle surface. The neck, back, and areas below the belt are excluded. Real judging goes beyond raw punch totals. [World Boxing November 2024, Rules 7.1–7.3](https://worldboxing.org/wp-content/uploads/2024/11/World-Boxing-Competition-Rules-Nov-2024-Approved.pdf)

The first referee is a **virtual touch experiment**:

- Use actual glove/target geometry over the motion, with at most one result per punch.
- Restrict targets to configured front head/torso regions; report blocks and misses separately.
- End attacks at round boundaries; disable scoring during breaks.
- Keep touch statistics separate from round scorecards.
- A touch-based 10–9 round assignment is an engineering proxy, not official judging. Report exact ties openly rather than breaking them arbitrarily.
- Defense-only mode never initiates punches.

Body-weight transfer, knuckle orientation, judging viewpoints, fouls, knockdowns, counts, and full competition adjudication need further modeling. Higher impact force is not the optimization target.

## SuperMimic: later motion layer

The paper trains a goal-conditioned whole-body motion tracker through a learned differentiable dynamics model. Simulator transitions train the world model; short imagined rollouts improve the policy. Experiments use MuJoCo Warp and H100 80 GB GPUs. This could support later joint control, footwork imitation, and balance, but it does not establish boxing tactics or robustness to opponent contact. Flat-ground training and no terrain/object interaction are stated limitations. The headline three-hour result is not a promise for our machine or task. [SuperMimic paper](https://arxiv.org/pdf/2610.28480)

The official project links the paper and a browser policy demonstration. An official training repository was not verified in this review. Runnable community code exists:

| Repository | Verified description | Integration consideration |
| --- | --- | --- |
| [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) | Independent paper reproduction, MIT code; Isaac Lab 3 with Newton/MuJoCo Warp | Separate GPU environment and external G1/motion assets; README labels the code unreviewed. |
| [Motphys/Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic) | Independent UniLab reproduction, Apache-2.0 code; MuJoCo, mjwarp, and Motrix backends | Test a supplied tracker independently before adapting it to ShadowBox. |

These are independent implementations. Code licenses do not automatically cover downloaded datasets or robot assets. [Official project](https://supermimic.github.io/)

Integration order: pin a repository revision → reproduce one supplied motion → verify joints, gains, observation order, frequency, and asset licenses → adapt behind the motion interface → compare against the preserved deterministic controller.

## FLORA: offline labels

FLORA is a robotics data engine for timestamped captions and outcome/process reward labels. Possible uses include segmenting future clips into attacks, guards, and footwork, or finding failed episodes for review. These are project hypotheses: the published page does not validate boxing-specific annotation or judging. Use exact simulator geometry/events for today's referee and validate any learned labels against a manually checked sample. [NVIDIA FLORA](https://research.nvidia.com/labs/gear/flora/)

## Recommended learning sequence

1. Qualify scoring and controls first: repeatable legal head/body touch and miss
   controls at multiple ranges and timings, on both sides; then qualify footwork,
   balance and falls. Keep touch events, punch episodes and round outcomes
   distinct.
2. Build a small real-video motion set. Estimate pose and punch/foot-contact
   phases, retarget the motion to G1, and reject clips that fail joint tracking,
   foot-slide or contact-timing checks. Use the clips to initialize a compact
   tactical planner; do not treat monocular video as ground-truth depth or force.
3. Train one planner against qualified scripted styles and saved opponents.
   Start from the existing frozen low-level tracker and expose meaningful
   high-level boxing goals; the current learned action set only chooses guard
   and body guard. Measure attack activity, defense, balance and repetitive
   behavior on held-out seeds and opponents.
4. Add symmetric two-agent self-play only after both sides can execute and score
   safely. Keep a pool of frozen historical opponents (or an average-policy
   method) so the newest policies do not only learn to beat each other. Promote
   on held-out opponent snapshots and seeds, with anti-repetition and balance
   gates.
5. Scale qualified episodes with MuJoCo Warp / Playground vectorization, where
   each environment contains a pair of G1s. The current batched defense task
   uses a scripted opponent; the transferred motion task is not yet a scalable
   self-play trainer. Profile throughput after the two-agent task and policy
   contract work.
6. Only then begin sim-to-real qualification: measure the real robot's latency,
   actuator response, contact and sensing; randomize measured uncertainty in
   simulation; then test progressively with low-energy targets and a safety
   supervisor. Feed robot logs back into the model and repeat the loop.

This ordering follows recent hierarchical boxing work: HoloBoxing reports a
frozen whole-body execution layer with imitation-initialized target planning,
then task RL; RoboStriker reports latent-space self-play over bounded motion
actions rather than unconstrained joint actions. These are research precedents,
not results reproduced by ShadowBox. [HoloBoxing](https://arxiv.org/abs/2610.32909),
[RoboStriker](https://arxiv.org/abs/2601.22517). Real-human motion can also be
retargeted and filtered through a simulated humanoid controller before
sim-to-real transfer, as demonstrated by [H2O](https://arxiv.org/abs/2403.04436);
video pose alone still does not establish robot contact forces or judging.

## Deferred experiments

These are research questions after the active controls, referee and movement
gates in [the architecture](architecture.md), not current blockers:

| Question | Comparison | Evidence |
| --- | --- | --- |
| Does a non-scoring feint create a useful opening? | Feint then attack versus direct attack across held-out opponents | Follow-up legal-touch rate, time/action cost, misses and false scores |
| How sensitive is defense to observation delay/noise? | Clean observations versus seeded delay/noise with exact referee geometry | Incoming attempts, blocks, conceded touches, action latency, rejections and safety |
