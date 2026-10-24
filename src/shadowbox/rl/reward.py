"""Outcome-based tactical reward, independent of the simulator loop."""


DEFAULT_WEIGHTS = dict(touch=1.0, conceded_touch=-1.0, block=.3,
                       opponent_block=-.3, miss=-.1, rejection=-.1,
                       second=-.01, safety_stop=-5.0,
                       accepted_attempt=.08, idle_guard=-.12,
                       unprovoked_retreat=-.10)


def tactical_reward(own, opponent, seconds, rejected, stopped, weights=None,
                    action=None, accepted_attempts=0, unprovoked_retreat=False):
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    # A blocked or missed punch costs more than the attempt bonus. The bonus
    # still gives PPO a learning signal before its first scored touch.
    # Guarding without a measured block should not be the cheapest perpetual
    # action; a successful block retains its positive reward.
    idle_guard = action in ('guard', 'body_guard') and own['blocks'] == 0
    return (w['touch'] * own['touches']
            + w['conceded_touch'] * opponent['touches']
            + w['block'] * own['blocks']
            + w['opponent_block'] * opponent['blocks']
            + w['miss'] * own['misses']
            + w['rejection'] * int(rejected)
            + w['second'] * seconds
            + w['safety_stop'] * int(stopped)
            + w['accepted_attempt'] * accepted_attempts
            + w['idle_guard'] * int(idle_guard)
            + w['unprovoked_retreat'] * int(unprovoked_retreat))
