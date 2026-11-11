"""Two free-base G1 defensive boxing task using Playground/reaf components.

This is a low-level joint-control training task, independent of the supported-root
legacy arena. No pretrained tracker or reference corpus is required.
"""
import math

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
        # colored balls over the G1's visible hand meshes.
        for hand in ('left', 'right'):
            robot.body(f'{hand}_wrist_yaw_link').add_geom(
                name=f'{hand}_glove', type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.065, 0, 0], pos=[0.08, 0, 0], mass=0.2,
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



def boxing_env(episode_length_secs=10.0, xml_uri=None, start_distance=0.6,
               opponent_enabled=True, terminate_opponent=True, motion_bundle=None,
               motion_script=None, motion_runtime='torchscript', motion_device='cuda:0', defender_guard_width=.10,
               opponent_guard_width=.10, head_target_z=.385, body_target_z=.15,
               head_overshoot=.16, body_overshoot=.08,
               retreat_speed_a=.18, retreat_speed_b=.18,
               opponent_attack_target='alternating', opponent_attack_delay_secs=0.,
               controlled_side='a',
               reset_yaw_radians=0., reset_lateral_m=0.,swap_start_sides=False):
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
    if controlled_side not in ('a','b'):
        raise ValueError('controlled_side must be a or b')
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
            retreat_speed_a=retreat_speed_a,retreat_speed_b=retreat_speed_b,
            opponent_attack_target=opponent_attack_target,
            opponent_attack_delay_secs=opponent_attack_delay_secs,
            controlled_side=controlled_side,
            joint_names=list(g1.JOINT_NAMES),stiffness=model.actuator_gainprm[ids,0].tolist(),
            damping=(-model.actuator_biasprm[ids,2]).tolist(),effort=model.actuator_forcerange[ids,1].tolist())
        separate={}
    observations = {'paired': cfg.Obs(paired_observation).configure(output_shape=(133+action_width,))}
    return cfg.Environment(base_scene=spec, device_layer=cfg.DeviceLayer(devices=device_map),
        task_layer=cfg.TaskLayer(
            command_feature_producers=shared,
            command_processors=separate,
            feature_producers={'contact_episodes': contact_episodes},
            rewards={'defense': cfg.Reward(defensive_reward, weight=1.0, control_cost=not motion_enabled)},
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
