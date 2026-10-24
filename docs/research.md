# Research decisions

Current implementation is recorded in [architecture](architecture.md).
MorphMimic's canonical bridge and source launcher, and the HoloBoxing launcher,
are integrated. Trained upstream policy behavior has not been reproduced here.
The detailed external assessments below were checked October 24 and are dated
research notes rather than the current execution order.

## Engine and tool choices

| System | Role in ShadowBox | Decision |
| --- | --- | --- |
| MuJoCo | Articulated physics, seeded matches, simple rendering | Keep as the working engine. |
| MorphMimic | Shared motion tracker across supported humanoids | Pinned source and canonical bridge integrated; exported-policy reproduction is the next motion gate. |
| HoloBoxing | Contact-scored boxing benchmark | Pinned source and isolated launcher integrated; supported GPU workstation pilot pending. |
| Isaac Lab | Batched GPU training and robotics task tooling | Evaluate after match/reward definitions stabilize; separate environment. |
| SuperMimic | Whole-body tracking of reference motion | Candidate for later imitation and balance; tactics remain a separate policy. |
| Cosmos | World models and visual data generation/reasoning | Optional research; photorealism is not a goal. |
| FLORA | Video captions and outcome/process reward annotations | Possible offline data preparation and episode review. |

MuJoCo provides articulated rigid-body simulation and contact dynamics, fitting the current need for repeatable motion and measurements. [MuJoCo overview](https://mujoco.readthedocs.io/en/stable/overview.html)

Isaac Lab Release 3 uses Newton by default and runs without Isaac Sim. The documentation lists Linux aarch64 and Python 3.12. Its default CUDA 13 PyTorch training stack requires a suitable GPU and Linux driver 580.65.06 or newer. Full Isaac Sim and RTX rendering add separate requirements. The GPU driver was not accessible from the initial runtime, so local GPU training is unverified. [Isaac Lab installation](https://docs.nvidia.com/isaaclab/isaac-lab/installation)

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

## Original learning sequence (partly implemented)

1. Establish independent observable opponents and a deterministic referee.
2. Evaluate across seeds, opponent styles, and starting sides.
3. Search a few policy parameters against a saved opponent pool; preserve candidates and held-out results. Call this parameter optimization.
4. Add self-play RL after checking reward exploitation, repetitive loops, and forgetting earlier opponents.
5. Add compatible reference motions and imitation, then fine-tune tactics in self-play.
6. Increase degrees of freedom and introduce free-base balance/contact progressively. Stabilized-base wins do not demonstrate dynamic humanoid boxing.
