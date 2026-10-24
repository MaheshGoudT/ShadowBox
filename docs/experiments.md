# Remaining experiment questions

The current system and adoption order are in [architecture](architecture.md).
Both-side evaluation, frozen opponents, head/body scoring and qualified imitation
are already implemented; they are regression requirements rather than new ideas.

| Question | Smallest useful experiment | Evidence to keep |
| --- | --- | --- |
| Does the learned boxer defend actual attacks? | Legal repeated head/body attackers on both sides and several close ranges | Incoming attempts, guard blocks, conceded rate, evasion misses, activity and safety per threat |
| Can demonstrations fix weak head defense? | Qualified responses to observed incoming glove motion, split by complete episode | Both sides/ranges, accepted actions and measured outcomes; fresh confirmation seeds |
| Is timing robust? | Seeded observation delay/noise while referee geometry stays exact | Clean and delayed results separately; action latency, rejections and safety |
| Does MorphMimic track boxing references? | Matching exported policy/robot bundles and licensed clips in upstream free-base sim2sim | Joint/body error, drift, sole slip, falls, latency and actuator limits per robot |
| Do feints create useful openings? | One non-scoring feint followed by a measured attack | Follow-up touch rate versus direct attack, action/time cost, no false scoring |
| Does a new engine solve a measured bottleneck? | Fixed workload on an isolated GPU backend | Throughput, setup cost, contact consistency and stability against CPU MuJoCo |

Preserve deterministic closing-opponent, boundary, asymmetric-start and emergency
stop checks. Compare both sides, log incomplete bouts, and keep development seeds
separate from final confirmation. Sim-to-real remains a later per-robot gate.
