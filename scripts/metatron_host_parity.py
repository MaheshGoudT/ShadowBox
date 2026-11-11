"""Validate the independent history/graph host against recorded native inputs."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import numpy as np
    from shadowbox.playground.motion_policy import MotionPolicy
    base=ROOT/'tmp/playground/metatron'
    policy=MotionPolicy(base/'compiled/unified_pipeline.yaml')
    samples=np.load(base/'native-onnx-feed-trace-feeds.npz')
    states=np.load(base/'native-onnx-feed-trace.npz')
    specs=policy.metadata['policy_inputs']
    worst_history=worst_action=0.
    for tick in range(len(states['action'])):
        current={s['key']:samples[f'step_{tick}_{s["name"]}'] for s in specs}
        if tick==0:policy.reset(current)
        for key,value in policy.history.inputs().items():
            worst_history=max(worst_history,float(np.max(np.abs(value-current[key]))))
        output=policy.predict(current,{})
        worst_action=max(worst_action,float(abs(output['actions']-states['action'][tick]).max()))
        # Feed back the command actually used in the recorded physics rollout.
        policy.accept(current,{'actions':states['action'][tick], 'joint_pos_targets':output['joint_pos_targets']})
    report=dict(samples=len(states['action']),max_history_error=worst_history,
        max_action_error=worst_action,passed=worst_history<1e-6 and worst_action<1e-4,
        policy=policy.provenance(),scope='native-feed playback; not a Playground transfer result')
    (base/'independent-host-parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise ValueError('Independent host parity failed')


if __name__=='__main__':main()
