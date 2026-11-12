"""Two free-base G1 defensive boxing task using Playground/reaf components.

This is a low-level joint-control training task, independent of the supported-root
legacy arena. No pretrained tracker or reference corpus is required.
"""
import math
from functools import partial
from numbers import Real

import mujoco
from mujoco_playground.robots.g1.sim import device as g1
from mujoco_playground.robots.g1.sim import scene
from reaf.simulation import torch as devices
from reaf.simulation.torch import action_history, flat_scene as cfg, functional_reset
from reaf.core import commands_processor
from gdm_robotics.interfaces import types as command_types
import numpy as np
import torch
from mujoco_playground import asset
from .skill_goals import STRAIGHT_JAB_ARM_TARGET

TASK_ID = 'ShadowBoxG1Defense'
DT = 0.02
MENAGERIE_REVISION = '98a70c569c4c1b6da467a20012f766ee1f8dc521'
G1_XML_URI = g1.G1_XML_URI + '?commit=' + MENAGERIE_REVISION
G1_HANDS_XML_URI = ('github:google-deepmind/mujoco_menagerie/unitree_g1/'
                    'g1_with_hands.xml?commit=' + MENAGERIE_REVISION)

# The 29-DoF motion tracker does not control fingers. Hold the Menagerie hand
# joints in a boxing fist at reset while leaving the learned arm controller
# unchanged. Invisible glove proxies remain the physical contact surfaces.
FIST_JOINT_POSITIONS = {
    'left_hand_thumb_0_joint': 0.6,
    'left_hand_thumb_1_joint': 0.95,
    'left_hand_thumb_2_joint': 1.6,
    'left_hand_middle_0_joint': -1.45,
    'left_hand_middle_1_joint': -1.6,
    'left_hand_index_0_joint': -1.45,
    'left_hand_index_1_joint': -1.6,
    'right_hand_thumb_0_joint': 0.6,
    'right_hand_thumb_1_joint': -0.95,
    'right_hand_thumb_2_joint': -1.6,
    'right_hand_middle_0_joint': 1.45,
    'right_hand_middle_1_joint': 1.6,
    'right_hand_index_0_joint': 1.45,
    'right_hand_index_1_joint': 1.6,
}


def _g1_robot_spec_with_fists():
    """Build pinned G1 hand geometry with the 29-DoF controller interface."""
    robot = asset.get(G1_HANDS_XML_URI).spec()
    # The Menagerie XML defines all actuators. Keep its finger actuators and
    # let the pinned Playground scene helper add the calibrated 29 arm/leg
    # actuators used by the motion tracker.
    base_actuators = set(g1.JOINT_NAMES)
    present = {actuator.name for actuator in robot.actuators}
    if not base_actuators <= present:
        raise ValueError('Pinned articulated G1 model is missing base actuators')
    for actuator in list(robot.actuators):
        if actuator.name in base_actuators:
            robot.delete(actuator)
        elif actuator.name in FIST_JOINT_POSITIONS:
            # Menagerie finger servos are torque-limited to 1.4 Nm. Raise
            # their hold limit for this fixed visual fist pose so impacts at
            # the wrist do not unfold the thumb; the hand meshes do not collide.
            actuator.forcerange[0] = -5.0
            actuator.forcerange[1] = 5.0
    # Task resets establish the two-robot stance directly; discard the
    # single-robot stand keyframe from the upstream file.
    for key in list(robot.keys):
        robot.delete(key)
    scene._add_actuators(robot)
    # Contact sensors address these target geoms by name. The articulated
    # Menagerie file leaves collision mesh names blank, so give the head and
    # torso collision meshes the stable names expected by the boxing referee.
    contact_targets = {'head_link': 'head_collision', 'torso_link': 'torso_collision'}
    found_targets = set()
    for geom in robot.geoms:
        parent_name = getattr(geom.parent, 'name', '')
        if (parent_name.startswith(('left_hand_', 'right_hand_'))
                or parent_name in ('left_wrist_yaw_link', 'right_wrist_yaw_link')) and geom.contype:
            # The hidden gloves below are the measured contact surfaces. Hand
            # links and wrist endcaps stay render-only so collisions cannot
            # force the fingers open.
            geom.contype = 0
            geom.conaffinity = 0
        target = contact_targets.get(geom.meshname)
        if target and geom.contype:
            geom.name = target
            found_targets.add(target)
    if found_targets != set(contact_targets.values()):
        raise ValueError('Pinned articulated G1 model is missing target collision meshes')
    return robot


def guard_pose():
    pose = list(g1.KNEES_BENT_JOINT_POS)
    values = {'left_shoulder_pitch_joint': -0.41387, 'right_shoulder_pitch_joint': -0.41387,
              'left_shoulder_roll_joint': 0.08201, 'right_shoulder_roll_joint': -0.08201,
              'left_shoulder_yaw_joint': 0.11347, 'right_shoulder_yaw_joint': -0.11347,
              'left_elbow_joint': -0.70133, 'right_elbow_joint': -0.70133}
    return [values.get(name, value) for name, value in zip(g1.JOINT_NAMES, pose)]


def boxing_scene(xml_uri=None):
    """Attach two articulated G1 specs; preserve contacts and free roots."""
    arena = mujoco.MjSpec()
    arena.option.timestep = 0.005
    arena.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    arena.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                             size=[5, 5, 0.1], friction=[0.8, 0.005, 0.0001])
    arena.worldbody.add_light(pos=[0, 0, 3])
    for side in ('a', 'b'):
        robot = (scene.g1_robot_spec(xml_uri) if xml_uri else _g1_robot_spec_with_fists())
        while robot.keys:
            robot.delete(robot.keys[0])
        # Hidden glove hitboxes preserve measured hand contacts without drawing
        # colored balls over the G1's visible hand meshes. The tracker FK hand
        # point is 0.12 m along the local wrist x axis; keep the contact proxy
        # at that same point so physical hits match the conditioned hand target.
        for hand in ('left', 'right'):
            robot.body(f'{hand}_wrist_yaw_link').add_geom(
                name=f'{hand}_glove', type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.065, 0, 0], pos=[0.12, 0, 0], mass=0.2,
                rgba=[0, 0, 0, 0],
                contype=1, conaffinity=1)
        arena.attach(robot, prefix=f'{side}/', frame=arena.worldbody.add_frame())
    return arena


class BoxingCommands(commands_processor.CommandsProcessor):
    """Bounded learner targets and independent alternating opponent jabs."""
    def __init__(self, name='', nworld=1, xpu_device='cpu', lower=(), upper=(), opponent_enabled=True):
        self._name = name
        self.phase = torch.zeros(nworld, device=xpu_device)
        self.home = torch.tensor(guard_pose(), device=xpu_device)
        self.scale = torch.tensor(g1.ACTION_SCALE_VECTOR, device=xpu_device)
        self.lower = torch.tensor(lower, device=xpu_device)
        self.upper = torch.tensor(upper, device=xpu_device)
        self.opponent_enabled = opponent_enabled

    @property
    def name(self):
        return self._name

    def consumed_commands_spec(self):
        return {'action': command_types.UnboundedArraySpec(shape=(29,), dtype=np.float32)}

    def produced_commands_keys(self):
        return {'a/ctrl', 'b/ctrl'}

    def process_commands(self, consumed_commands):
        action = consumed_commands['action'].clamp(-1, 1)
        own = (self.home + action * self.scale).clamp(self.lower, self.upper)
        other = self.home.expand(action.shape[0], -1).clone()
        # One punch per 1.6s; left/right alternate. No access to learner actions.
        cycle = torch.remainder(self.phase, 1.6) / 1.6
        extension = torch.sin((cycle / 0.45).clamp(0, 1) * math.pi).clamp_min(0)
        if not self.opponent_enabled:
            extension = torch.zeros_like(extension)
        body_attack = ((self.phase / 3.2).floor().long() % 2 == 1)
        for hand, parity in (('left', 0), ('right', 1)):
            active = ((self.phase / 1.6).floor().long() % 2 == parity)
            amount = extension * active
            mirror = 1 if hand == 'left' else -1
            # Native-model IK fixtures: targets relative to pelvis, meters.
            # Guard=(.22,+/-.15,.36), body=(.42,0,.16), head=(.42,0,.40).
            for suffix, body_value, head_value in (
                    ('shoulder_pitch', -1.33024, -1.54810),
                    ('shoulder_roll', -0.22750 * mirror, -0.01048 * mirror),
                    ('shoulder_yaw', 1.44903 * mirror, 0.04930 * mirror),
                    ('elbow', 1.41364, 1.03735)):
                j = g1.JOINT_NAMES.index(f'{hand}_{suffix}_joint')
                target = torch.where(body_attack, body_value, head_value)
                other[:, j] += amount * (target - self.home[j])
        other[:, g1.JOINT_NAMES.index('waist_pitch_joint')] += 0.15 * extension
        self.phase += DT
        return {'a/ctrl': own, 'b/ctrl': other.clamp(self.lower, self.upper)}

    def reset(self, env_mask=None):
        if env_mask is None:
            self.phase.zero_()
        else:
            self.phase[env_mask] = 0


def reset_boxers(dc, env_mask, qpos, ctrl, root_addresses, reset_yaw_radians=0., reset_lateral_m=0.):
    data = dc.sim.data
    mask = env_mask[:, None]
    home = torch.tensor(qpos, device=data.qpos.device, dtype=data.qpos.dtype)
    # Randomize both robots together in y, keeping their facing and distance.
    sampled = home.expand(data.qpos.shape[0], -1).clone()
    shift = (torch.rand(data.qpos.shape[0], device=data.qpos.device) - 0.5) * 0.2
    for address in root_addresses:
        sampled[:, address + 1] += shift
        if reset_lateral_m:
            sampled[:,address+1]+=(torch.rand_like(shift)*2-1)*reset_lateral_m
        if reset_yaw_radians:
            half=(torch.rand_like(shift)*2-1)*reset_yaw_radians/2
            c,s=half.cos(),half.sin()
            q=sampled[:,address+3:address+7].clone()
            sampled[:,address+3:address+7]=torch.stack((c*q[:,0]-s*q[:,3],
                c*q[:,1]-s*q[:,2], c*q[:,2]+s*q[:,1], c*q[:,3]+s*q[:,0]),dim=-1)
    data.qpos[:] = torch.where(mask, sampled, data.qpos)
    data.qvel[:] = torch.where(mask, torch.zeros_like(data.qvel), data.qvel)
    data.ctrl[:] = torch.where(mask, torch.tensor(ctrl, device=data.ctrl.device), data.ctrl)


def paired_observation(f):
    a, b = f['a_root/xpos'], f['b_root/xpos']
    return torch.cat((f['a/joint_pos'], f['a/joint_vel'], f['a_root/xquat'],
                      f['a_root/joint_vel'], b-a, f['b_root/xquat'],
                      f['b/joint_pos'], f['b/joint_vel'], f['action']), dim=-1)


def player_observation(f, side='a', active_primitive=None, active_phase=None):
    """Return yaw-invariant state, including this boxer's active skill phase."""
    if side not in ('a', 'b'):
        raise ValueError('Player observation side must be a or b')
    other = 'b' if side == 'a' else 'a'

    def yaw(quaternion):
        w, x, y, z = quaternion.unbind(-1)
        return torch.atan2(2 * (w * z + x * y),
                           1 - 2 * (y.square() + z.square()))

    def remove_yaw(quaternion, heading):
        """Express a world quaternion after rotating the view to heading zero."""
        w, x, y, z = quaternion.unbind(-1)
        cosine = torch.cos(heading / 2)
        sine = torch.sin(heading / 2)
        local = torch.stack((cosine * w + sine * z,
                             cosine * x + sine * y,
                             cosine * y - sine * x,
                             cosine * z - sine * w), dim=-1)
        # q and -q encode the same rotation. Choose one stable representative,
        # including the exactly 180-degree case where w is zero.
        vector = local[..., 1:]
        first_nonzero = (vector.abs() > 1e-7).to(torch.int64).argmax(dim=-1)
        axis_value = vector.gather(-1, first_nonzero.unsqueeze(-1)).squeeze(-1)
        flip = (local[..., 0] < -1e-7) | (
            (local[..., 0].abs() <= 1e-7) & (axis_value < 0))
        return torch.where(flip.unsqueeze(-1), -local, local)

    def rotate_world_vector(vector, heading):
        x, y, z = vector[..., :3].unbind(-1)
        cosine, sine = torch.cos(heading), torch.sin(heading)
        return torch.stack((cosine * x + sine * y,
                            -sine * x + cosine * y, z), dim=-1)

    own_quaternion = f[f'{side}_root/xquat']
    other_quaternion = f[f'{other}_root/xquat']
    heading = yaw(own_quaternion)
    relative = f[f'{other}_root/xpos'] - f[f'{side}_root/xpos']
    relative = rotate_world_vector(relative, heading)
    own_root_velocity = f[f'{side}_root/joint_vel']
    other_root_velocity = f[f'{other}_root/joint_vel']
    # Free-joint linear velocity is world-frame; angular velocity is local.
    own_root_velocity = torch.cat((rotate_world_vector(own_root_velocity, heading),
                                   own_root_velocity[:, 3:]), dim=-1)
    other_root_velocity = torch.cat((rotate_world_vector(other_root_velocity, heading),
                                     other_root_velocity[:, 3:]), dim=-1)

    def batch_value(value, key, default):
        if value is None:
            value = f.get(key)
        if value is None:
            value = own_quaternion.new_full((own_quaternion.shape[0],), default)
        value = torch.as_tensor(value, device=own_quaternion.device)
        if value.ndim == 0:
            value = value.expand(own_quaternion.shape[0])
        return value.reshape(own_quaternion.shape[0])

    active_primitive = batch_value(active_primitive,
                                   f'motion/{side}_primitive', 0).long().clamp(0, 8)
    active_phase = batch_value(active_phase,
                               f'motion/{side}_phase', 0).to(
                                   own_quaternion.dtype).clamp(0, 2) / 2
    active_skill = torch.nn.functional.one_hot(active_primitive, num_classes=9).to(
        own_quaternion.dtype)
    return torch.cat((f[f'{side}/joint_pos'],
                      f[f'{side}/joint_vel'], remove_yaw(own_quaternion, heading),
                      own_root_velocity, relative,
                      remove_yaw(other_quaternion, heading), other_root_velocity,
                      f[f'{other}/joint_pos'],
                      f[f'{other}/joint_vel'], active_skill,
                      active_phase.unsqueeze(-1)), dim=-1)


def selfplay_observation_a(f):
    return player_observation(f, 'a')


def selfplay_observation_b(f):
    return player_observation(f, 'b')


def upright(f, side='a'):
    q = f[f'{side}_root/xquat']
    return 1 - 2 * (q[:, 1].square() + q[:, 2].square())


def fallen(f, terminate_opponent=True):
    own = (f['a_root/xpos'][:, 2] < 0.45) | (upright(f) < 0.4)
    if terminate_opponent:
        own = own | (f['b_root/xpos'][:, 2] < 0.45) | (upright(f, 'b') < 0.4)
    return own


def out_of_bounds(f, terminate_opponent=True):
    own = (f['a_root/xpos'][:, :2].abs() > 3).any(-1)
    if terminate_opponent:
        own = own | (f['b_root/xpos'][:, :2].abs() > 3).any(-1)
    return own


def contact_force(f, key):
    return f[f'{key}/force_history'].norm(dim=-1).amax(dim=(1, 2))


def defensive_reward(f, control_cost=True):
    # Penalize measured glove-to-torso contacts. Guard contacts are separate.
    speed_cost = f['a/joint_vel'].square().mean(-1)
    own_fall = (f['a_root/xpos'][:, 2] < 0.45) | (upright(f) < 0.4)
    return (1.0 + 0.5 * upright(f).clamp(0, 1)
            + 0.5 * torch.exp(-((f['a_root/xpos'][:, 2] - 0.76) / 0.1).square())
            + 0.2 * torch.exp(-0.1 * speed_cost)
            - (0.01 * f['action'].clamp(-1, 1).square().mean(-1) if control_cost else 0)
            - 0.02 * contact_force(f, 'incoming').clamp(max=50)
            - 50.0 * own_fall.float())


def competitive_reward(f, controlled_side='a', head_contact_reward=4.,
                       body_contact_reward=2., received_head_penalty=4.,
                       received_body_penalty=2., block_reward=.25,
                       approach_progress_reward=0.):
    """Per-player sparse reward with separately tunable contact values."""
    other = 'b' if controlled_side == 'a' else 'a'
    events = f['boxing/events']
    if controlled_side == 'a':
        own_head, own_body = events[:, 3], events[:, 4]
        received_head, received_body = events[:, 0], events[:, 1]
        blocks_for, blocks_against = events[:, 2], events[:, 5]
    else:
        own_head, own_body = events[:, 0], events[:, 1]
        received_head, received_body = events[:, 3], events[:, 4]
        blocks_for, blocks_against = events[:, 5], events[:, 2]
    own_fall = ((f[f'{controlled_side}_root/xpos'][:, 2] < 0.45)
                | (upright(f, controlled_side) < 0.4))
    opponent_fall = ((f[f'{other}_root/xpos'][:, 2] < 0.45)
                     | (upright(f, other) < 0.4))
    own_out = (f[f'{controlled_side}_root/xpos'][:, :2].abs() > 3).any(-1)
    opponent_out = (f[f'{other}_root/xpos'][:, :2].abs() > 3).any(-1)
    progress = f.get(f'motion/{controlled_side}_approach_progress')
    approach_reward = (torch.zeros_like(own_head.sum(-1)) if progress is None else
                       approach_progress_reward * progress.clamp(min=0))
    return (head_contact_reward * own_head.sum(-1)
            + body_contact_reward * own_body.sum(-1)
            - received_head_penalty * received_head.sum(-1)
            - received_body_penalty * received_body.sum(-1)
            + block_reward * (blocks_for.sum(-1) - blocks_against.sum(-1))
            + 0.2 * (upright(f, controlled_side) - upright(f, other))
            + approach_reward
            - 10.0 * own_fall.float() + 10.0 * opponent_fall.float()
            - 5.0 * own_out.float() + 5.0 * opponent_out.float())



def boxing_env(episode_length_secs=10.0, xml_uri=None, start_distance=0.6,
               opponent_enabled=True, terminate_opponent=True, motion_bundle=None,
               motion_script=None, motion_runtime='torchscript', motion_device='cuda:0', defender_guard_width=.10,
               opponent_guard_width=.10, head_target_z=.385, body_target_z=.15,
               head_overshoot=.16, body_overshoot=.08,
               body_attack_arc_lateral=0.,
               retreat_speed_a=.18, retreat_speed_b=.18,
               approach_speed_a=.27, approach_speed_b=.27,
               max_punch_reach_m=.48, approach_target_range_m=.65,
               approach_duration_secs=.4,
               jab_arm_target=STRAIGHT_JAB_ARM_TARGET,
               opponent_attack_target='alternating', opponent_attack_delay_secs=0.,
               controlled_side='a', self_play=False, opponent_policy_script=None,
               reset_yaw_radians=0., reset_lateral_m=0.,swap_start_sides=False,
               head_contact_reward=4., body_contact_reward=2.,
               received_head_penalty=4., received_body_penalty=2., block_reward=.25,
               approach_progress_reward=0.):
    if not 0<=reset_yaw_radians<=.15 or not 0<=reset_lateral_m<=.05:
        raise ValueError('Reset variation requires yaw in [0,.15] and lateral in [0,.05]')
    if type(swap_start_sides) is not bool:
        raise ValueError('swap_start_sides must be boolean')
    if not 0.5 <= start_distance <= 2.0:
        raise ValueError('start_distance must be between 0.5 and 2 meters')
    if opponent_attack_target not in ('alternating','head','body'):
        raise ValueError('opponent_attack_target must be alternating, head or body')
    if not np.isfinite(opponent_attack_delay_secs) or not 0.<=opponent_attack_delay_secs<=1.5:
        raise ValueError('opponent_attack_delay_secs must be in [0, 1.5]')
    retreat_speeds=np.asarray((retreat_speed_a,retreat_speed_b),dtype=np.float32)
    if not np.isfinite(retreat_speeds).all() or np.any((retreat_speeds<0)|(retreat_speeds>.4)):
        raise ValueError('retreat speeds must be finite and in [0, .4] m/s')
    approach_speeds=np.asarray((approach_speed_a,approach_speed_b),dtype=np.float32)
    if not np.isfinite(approach_speeds).all() or np.any((approach_speeds<0)|(approach_speeds>.4)):
        raise ValueError('approach speeds must be finite and in [0, .4] m/s')
    if not np.isfinite(max_punch_reach_m) or not .3<=max_punch_reach_m<=.7:
        raise ValueError('maximum punch reach must be finite and in [.3, .7] m')
    if not np.isfinite(approach_target_range_m) or not .5<=approach_target_range_m<=1.5:
        raise ValueError('approach target range must be finite and in [.5, 1.5] m')
    if not np.isfinite(approach_duration_secs) or not .1<=approach_duration_secs<=1.5:
        raise ValueError('approach duration must be finite and in [.1, 1.5] seconds')
    jab_arm_target=np.asarray(jab_arm_target,dtype=np.float32)
    if (jab_arm_target.shape!=(4,) or not np.isfinite(jab_arm_target).all()
            or np.any(np.abs(jab_arm_target)>3.2)):
        raise ValueError('jab_arm_target must contain four finite joint angles within +/-3.2 rad')
    if controlled_side not in ('a','b'):
        raise ValueError('controlled_side must be a or b')
    reward_weights=(head_contact_reward,body_contact_reward,received_head_penalty,
                    received_body_penalty,block_reward,approach_progress_reward)
    if any(isinstance(value,bool) or not isinstance(value,Real)
           or not math.isfinite(value) or not 0.<=value<=20.
           for value in reward_weights):
        raise ValueError('Boxing contact reward weights must be finite values in [0, 20]')
    if self_play and not motion_bundle:
        raise ValueError('Self-play requires the tracker-backed primitive task')
    if opponent_policy_script is not None and not self_play:
        raise ValueError('opponent_policy_script requires the self-play task')
    if opponent_policy_script is not None and not opponent_enabled:
        raise ValueError('A learned opponent requires opponent_enabled: true')
    spec = boxing_scene(xml_uri)
    motion_enabled = motion_bundle is not None
    physics_dt = 0.001 if motion_enabled else 0.005
    history_depth = round(DT/physics_dt)
    model = spec.compile()
    home = model.qpos0.copy()
    pose = guard_pose()
    limits = []
    roots = []
    starts={'a':(-start_distance/2,[1,0,0,0]),'b':(start_distance/2,[0,0,0,1])}
    if swap_start_sides:
        starts={'a':(start_distance/2,[0,0,0,1]),'b':(-start_distance/2,[1,0,0,0])}
    for side in ('a','b'):
        x,quat=starts[side]
        free = next(j for j in spec.joints if j.name.startswith(f'{side}/') and
                    j.type == mujoco.mjtJoint.mjJNT_FREE)
        address = model.jnt_qposadr[model.joint(free.name).id]
        roots.append(int(address))
        home[address:address+7] = [x, 0, g1.KNEES_BENT_ROOT_POS[2], *quat]
        for name, value in zip(g1.JOINT_NAMES, pose):
            joint = model.joint(f'{side}/{name}')
            home[joint.qposadr[0]] = value
            if side == 'a':
                limits.append(joint.range.copy())
        if xml_uri is None:
            for name, value in FIST_JOINT_POSITIONS.items():
                joint = model.joint(f'{side}/{name}')
                home[joint.qposadr[0]] = value
    initial_ctrl = [home[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
                    for i in range(model.nu)]
    device_map = {}
    for side in ('a', 'b'):
        free = next(j.name for j in spec.joints if j.name.startswith(f'{side}/') and
                    j.type == mujoco.mjtJoint.mjJNT_FREE)
        device_map[side] = cfg.Component(devices.ActuatorDevice,
            targets=[f'{side}/{n}' for n in g1.JOINT_NAMES], measurements=['joint_pos', 'joint_vel'])
        device_map[f'{side}_root'] = cfg.Component(devices.FreeDevice,
            body_name=f'{side}/pelvis', joint_name=free,
            measurements=['xpos', 'xquat', 'joint_vel'])
    for key, targets, body in (
            ('incoming', ('b/left_glove', 'b/right_glove'), 'a/torso_link'),
            ('outgoing', ('a/left_glove', 'a/right_glove'), 'b/torso_link')):
        device_map[key] = cfg.Component(devices.ContactSensorDevice,
            fields=('found', 'force'), history_length=history_depth).configure_spec(
                objtype='geom', objname=targets, reftype='body', refname=body, reduce='netforce')
    from .referee import CHANNELS, ContactEpisodes, contact_count, sensor_components
    device_map.update(sensor_components(history_depth))
    contact_episodes = cfg.Component(ContactEpisodes)
    action_width=1 if motion_enabled else 29
    history = cfg.Component(action_history.ActionHistory, key='action', feature_shape=(action_width,), history_length=1)
    commands = cfg.Component(BoxingCommands, lower=[v[0] for v in limits], upper=[v[1] for v in limits],
                             opponent_enabled=opponent_enabled)
    shared={'history':history}
    separate={'boxing':commands}
    if motion_enabled:
        from .motion_commands import TrackedBoxingCommands
        ids=[model.actuator(f'a/{name}').id for name in g1.JOINT_NAMES]
        shared['motion']=cfg.Component(TrackedBoxingCommands,bundle=str(motion_bundle),script=motion_script,
            runtime=motion_runtime,policy_device=motion_device,opponent_enabled=opponent_enabled,
            defender_guard_width=defender_guard_width,opponent_guard_width=opponent_guard_width,
            head_target_z=head_target_z,body_target_z=body_target_z,
            head_overshoot=head_overshoot,body_overshoot=body_overshoot,
            body_attack_arc_lateral=body_attack_arc_lateral,
            retreat_speed_a=retreat_speed_a,retreat_speed_b=retreat_speed_b,
            approach_speed_a=approach_speed_a,approach_speed_b=approach_speed_b,
            max_punch_reach_m=max_punch_reach_m,
            approach_target_range_m=approach_target_range_m,
            approach_duration_secs=approach_duration_secs,
            jab_arm_target=jab_arm_target.tolist(),
            opponent_attack_target=opponent_attack_target,
            opponent_attack_delay_secs=opponent_attack_delay_secs,
            controlled_side=controlled_side,
            opponent_policy_script=opponent_policy_script,
            joint_names=list(g1.JOINT_NAMES),stiffness=model.actuator_gainprm[ids,0].tolist(),
            damping=(-model.actuator_biasprm[ids,2]).tolist(),effort=model.actuator_forcerange[ids,1].tolist())
        separate={}
    obs_fn = (selfplay_observation_a if controlled_side == 'a' else selfplay_observation_b) if self_play else paired_observation
    obs_width = 149 if self_play else 133 + action_width
    observations = {'paired': cfg.Obs(obs_fn).configure(output_shape=(obs_width,))}
    rewards = ({'boxing': cfg.Reward(partial(
                   competitive_reward, controlled_side=controlled_side,
                   head_contact_reward=float(head_contact_reward),
                   body_contact_reward=float(body_contact_reward),
                   received_head_penalty=float(received_head_penalty),
                   received_body_penalty=float(received_body_penalty),
                   block_reward=float(block_reward),
                   approach_progress_reward=float(approach_progress_reward)), weight=1.0)}
               if self_play else
               {'defense': cfg.Reward(defensive_reward, weight=1.0, control_cost=not motion_enabled)})
    return cfg.Environment(base_scene=spec, device_layer=cfg.DeviceLayer(devices=device_map),
        task_layer=cfg.TaskLayer(
            command_feature_producers=shared,
            command_processors=separate,
            feature_producers={'contact_episodes': contact_episodes},
            rewards=rewards,
            terminations={'fall': cfg.Term(fallen, terminate_opponent=terminate_opponent),
                          'bounds': cfg.Term(out_of_bounds, terminate_opponent=terminate_opponent)},
            metrics={'incoming_force': cfg.Metric(contact_force, key='incoming', reduce='sum'),
                     'outgoing_force': cfg.Metric(contact_force, key='outgoing', reduce='sum'),
                     **{name: cfg.Metric(contact_count, channel=i, reduce='last')
                        for i, name in enumerate(CHANNELS)}}),
        observation_adapter={'policy': cfg.ObsGroup(terms=observations), 'critic': cfg.ObsGroup(terms=observations)},
        resetters={'boxers': cfg.Component(functional_reset.FunctionalEnvironmentReset,
            terms=[cfg.Term(reset_boxers, type='reset', qpos=home.tolist(), ctrl=initial_ctrl, root_addresses=roots,
                reset_yaw_radians=reset_yaw_radians,reset_lateral_m=reset_lateral_m)])},
        sim=cfg.Sim(control_timestep=DT, simulation_timestep=physics_dt),
        episode_length_secs=episode_length_secs, nan_termination=True)
