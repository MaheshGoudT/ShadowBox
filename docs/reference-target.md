# Unitree reference target

Reviewed 24 October 2026. This reference informs the movement target; it is not evidence that ShadowBox matches a physical Unitree robot.

## Source qualification

The [user's Reddit post](https://www.reddit.com/r/interestingasfuck/comments/1xc6jo5/chinas_robotics_company_unitree_has_unveiled_the/) labels its video H2-X “Kung Fu Legend: Guardian” and makes height/endurance claims. Reddit is a reference-sharing source, not a technical specification or validation report.

The linked manufacturer announcement title resolves to [Unitree Introducing: H2-X Kung Fu Legend Guardian](https://www.youtube.com/watch?v=Y3zb4o47yqA). Automated access returned page metadata rather than playable footage; the [official X announcement](https://x.com/UnitreeRobotics/status/2112819774573179175) returned HTTP 403. This review did not independently verify the complete video, autonomous control, active-motion endurance, repeatability, failure rates, an H2-X-specific specification, or a released H2-X boxing controller. Treat those as unknown. An edited promotional scenario does not by itself establish robust autonomous behavior.

A directly relevant primary source is [Unitree's G1-Boxing page](https://www.unitree.com/boxing/). It describes motion-capture training, straight punches/hooks, balance and recovery, and motion/controller/voice inputs. These are manufacturer descriptions, not reproducible performance measurements or a verified software release. They motivate our staged movement/control goals.

For later model selection, the [official standard H2 page](https://www.unitree.com/H2/) lists 31 joints: six per leg, seven per arm, three at the waist, and two in the head. These are **standard H2 specifications**; do not assign them to H2-X without an H2-X-specific source. The same page notes that some sample functions remain under development/testing.

## Boxing target for ShadowBox

Use simple visible humanoid geometry, clear glove paths, and independent agents. Photorealism is unnecessary. Keep this a boxing task: punches, guard, footwork, observation, scoring, and recovery. Kicks, acrobatics, vehicle demonstrations, weapons, rescue storylines, and physical deployment are outside this iteration.

| Stage | Desired behavior | Measurable acceptance |
| --- | --- | --- |
| Now: stance | Both robots face one another and maintain a recognizable guard | Facing error and guard joint error are logged; reset restores the same initial stance. |
| Now: attacks | Distinct jab/cross with explicit extension and recovery; bounded combinations | Each command has start/completion/cancellation events; combo order matches its definition; each punch yields at most one referee result. |
| Now: footwork | Approach, retreat, sidestep, and turn to face the opponent | Commanded displacements and resulting root paths are recorded; trajectories remain inside configured arena limits. |
| Now: defense | Guard or evade from actual observed opponent motion | Defense-only agent records zero initiated punches over seeded episodes; tests include close range and limited retreat space. |
| Now: visible match | Two agents move concurrently in an actual MuJoCo window | Demo visibly animates attacks and responses; pause/reset/stop work; headless mode reproduces the same seeded action/referee events. |
| Next: richer execution | Torso rotation, shoulder coordination, stance/weight-shift references | Added joints preserve actuator/observation mappings; trajectory limits and end-effector errors are measured against references. |
| Later: physical fidelity | Free base, stance stability, contact response, perturbation recovery | Report fall rate, torso tilt, foot slip, recovery time, and actuator-limit violations on held-out scenarios before claiming balanced boxing. |

The current stabilized model can validate policy and referee logic. It cannot establish physically realizable balance, impact handling, or hardware motion performance. Realistic weight transfer is a later dynamic-control objective; an animated torso lean alone is not a balance result.

## Evidence required before claiming improvement

- Preserve the working deterministic controller and the policy/model revision for every evaluation.
- Compare both starting sides and held-out seeds/opponent styles.
- Report clean virtual touches, attempted punches, blocks, misses, boundary/separation interventions, and completion rate.
- Keep touch statistics separate from simplified round scores and official judging claims.
- Label parameter search, imitation learning, and neural self-play accurately; show the actual before/after evaluation.
- For a later reference-motion controller, verify asset licenses, joint order, actuator gains, control frequency, and state/action normalization before integration.

## First reference-driven improvement

Make guard → jab → guard and guard → cross → guard unmistakable in the running viewer. Then add a bounded six-punch sequence and evasion/return-to-guard behavior. Improve tactical choices against measured opponent motion before increasing degrees of freedom or adding balance/contact complexity.
