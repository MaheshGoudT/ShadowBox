import numpy as np
import pytest
from shadowbox.playground.motion_policy import MotionHistory


def metadata():
    return {'control':{'pd_action_offset':[0.]}, 'policy_inputs':[
        dict(name='last',key='current_action',shape=[1,1],dtype='float32',output_key='actions',
             steps_back=[1],fill_before_available='zeros'),
        dict(name='past',key='old_actions',shape=[1,2,1],dtype='float32',output_key='actions',
             steps_back=[2,3],fill_before_available='zeros'),
        dict(name='readings',key='old_pos',shape=[1,2,1],dtype='float32',
             observation_history=dict(current_key='joint_pos',steps_back=[1,2]))]}


def test_history_aligns_accepted_actions_with_previous_measurements():
    history=MotionHistory(metadata(),1)
    history.reset({'joint_pos':np.array([[1.]])})
    for tick in range(4):
        values=history.inputs()
        assert values['current_action'][0,0] == (10*(tick-1) if tick else 0)
        assert values['old_actions'][0,0,0] == (10*(tick-2) if tick>=2 else 0)
        assert values['old_actions'][0,1,0] == (10*(tick-3) if tick>=3 else 0)
        assert values['old_pos'][0,0,0] == max(1,tick)
        history.accept({'joint_pos':np.array([[tick+1.]])},{'actions':np.array([[10*tick]])})


def test_partial_reset_clears_one_robots_commands_and_repeats_its_new_measurement():
    history=MotionHistory(metadata(),2)
    history.reset({'joint_pos':np.array([[1.],[3.]])})
    history.accept({'joint_pos':np.array([[1.],[3.]])},{'actions':np.array([[10.],[20.]])})
    history.reset({'joint_pos':np.array([[2.],[4.]])},mask=[True,False])
    values=history.inputs()
    np.testing.assert_array_equal(values['current_action'],[[0.],[20.]])
    np.testing.assert_array_equal(values['old_pos'],[[[2.],[2.]],[[3.],[3.]]])
    history.accept({'joint_pos':np.array([[2.],[4.]])},{'actions':np.array([[30.],[40.]])})
    np.testing.assert_array_equal(history.inputs()['old_actions'][:,0],[[0.],[20.]])


def test_invalid_commit_is_atomic_and_predictions_cannot_mutate_history():
    history=MotionHistory(metadata(),1)
    history.reset({'joint_pos':np.array([[1.]])})
    value=history.inputs();value['current_action'][:]=999
    assert not history.inputs()['current_action'].any()
    with pytest.raises(ValueError):
        history.accept({'joint_pos':np.array([[float('nan')]])},{'actions':np.array([[5.]])})
    assert not history.inputs()['current_action'].any()
