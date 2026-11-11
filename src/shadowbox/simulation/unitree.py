"""Two licensed Unitree G1 models with stabilized planar roots for virtual sparring.

The original 29 hinge joints, meshes, inertias and force limits are retained.
Elbows use a shifted coordinate so the existing boxing vocabulary stays usable.
"""

from copy import deepcopy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ..config import PROJECT_ROOT

CONTROL_ALIASES = {
    'left_shoulder_pitch_joint': 'left_shoulder',
    'left_elbow_joint': 'left_elbow',
    'right_shoulder_pitch_joint': 'right_shoulder',
    'right_elbow_joint': 'right_elbow',
}


def _robot(source, name, color, config):
    body = deepcopy(source.find('worldbody/body'))
    body.set('name', name)
    if 'pelvis_height' in config:
        position = body.get('pos', '0 0 .793').split()
        position[2] = str(config['pelvis_height'])
        body.set('pos', ' '.join(position))
    body.remove(body.find('freejoint'))
    bound = config['arena_half_size'] - 0.2
    for index, (joint, axis) in enumerate((('x', '1 0 0'), ('y', '0 1 0'), ('yaw', '0 0 1'))):
        attributes = dict(name=f'{name}_{joint}', type='slide' if joint != 'yaw' else 'hinge',
                          axis=axis, damping=str(config['translation_damping'] if joint != 'yaw' else 12))
        attributes.update(limited='false') if joint == 'yaw' else attributes.update(range=f'-{bound} {bound}')
        body.insert(index, ET.Element('joint', attributes))
    # Prefix the original robot names so the two articulated trees are independent.
    for element in body.iter():
        old = element.get('name')
        if old and element is not body and not old.startswith(name + '_'):
            element.set('name', f'{name}_{CONTROL_ALIASES.get(old, old)}')
        if element.tag == 'geom' and element.get('class') == 'visual' and element.get('mesh') in ('head_link', 'torso_link'):
            kind = 'head' if element.get('mesh') == 'head_link' else 'torso'
            element.set('name', f'{name}_{kind}_mesh')
        if element.tag == 'geom' and element.get('material') == 'metal':
            element.set('material', f'{name}_metal')
    for side in ('left', 'right'):
        foot = body.find(f'.//site[@name="{name}_{side}_foot"]')
        foot.set('pos', '.035 0 -.035')
        elbow = body.find(f'.//joint[@name="{name}_{side}_elbow"]')
        lo, hi = map(float, elbow.get('range').split())
        elbow.set('ref', str(-math.pi / 2))
        elbow.set('range', f'{lo-math.pi/2} {hi-math.pi/2}')
        hand = body.find(f'.//body[@name="{name}_{side}_wrist_yaw_link"]')
        ET.SubElement(hand, 'geom', name=f'{name}_{side}_glove', type='sphere',
                      pos='0.10 0 0', size='0.09', mass='0', rgba=color,
                      contype='0', conaffinity='0', group='1')
    torso = body.find(f'.//body[@name="{name}_torso_link"]')
    ET.SubElement(torso, 'geom', name=f'{name}_torso', type='capsule',
                  fromto='0 0 0.03 0 0 0.25', size='0.13', mass='0',
                  rgba='0 0 0 0', contype='0', conaffinity='0', group='5')
    ET.SubElement(torso, 'geom', name=f'{name}_head', type='sphere', pos='0 0 0.43',
                  size='0.105', mass='0', rgba='0 0 0 0', contype='0', conaffinity='0', group='5')
    return body


def make_unitree_model(timestep, config, directory=None):
    """Compose two G1 trees; a free-base balance controller is deliberately deferred."""
    directory = Path(directory or PROJECT_ROOT / 'assets/unitree_g1')
    path = directory / 'g1.xml'
    if not path.exists():
        raise ValueError(f'Unitree assets missing: {path}; run the documented asset fetch command')
    source = ET.parse(path).getroot()
    root = ET.Element('mujoco', model='ShadowBox-Unitree-G1')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true', meshdir=str((directory/'assets').resolve()))
    ET.SubElement(root, 'option', timestep=str(timestep), integrator='implicitfast')
    root.append(deepcopy(source.find('default')))
    asset = deepcopy(source.find('asset'))
    ET.SubElement(asset, 'material', name='a_metal', rgba='0.60 0.69 0.84 1')
    ET.SubElement(asset, 'material', name='b_metal', rgba='0.84 0.64 0.59 1')
    root.append(asset)
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -2 6', diffuse='0.9 0.9 0.9')
    half = config['arena_half_size']
    ET.SubElement(world, 'geom', name='floor', type='plane', size=f'{half+.1} {half+.1} .1',
                  rgba='.13 .18 .24 1', contype='0', conaffinity='0')
    for z in (.45, .8, 1.15):
        for sign in (-1, 1):
            for coordinates in (f'-{half} {sign*half} {z} {half} {sign*half} {z}',
                                f'{sign*half} -{half} {z} {sign*half} {half} {z}'):
                ET.SubElement(world, 'geom', type='capsule', fromto=coordinates, size='.025',
                              rgba='.7 .72 .8 1', contype='0', conaffinity='0')
    actuators = ET.SubElement(root, 'actuator')
    for name, color in (('a', '.18 .5 .95 1'), ('b', '.95 .26 .18 1')):
        world.append(_robot(source, name, color, config))
        for joint in ('x', 'y', 'yaw'):
            values = dict(name=f'{name}_{joint}', joint=f'{name}_{joint}',
                          kp=str(config['translation_gain'] if joint != 'yaw' else config['joint_gain']))
            if joint == 'yaw':
                values['ctrllimited'] = 'false'
            else:
                values['ctrlrange'] = f'{-half+.2} {half-.2}'
            ET.SubElement(actuators, 'position', values)
        for original in source.find('actuator'):
            actuator = deepcopy(original)
            original_joint = original.get('joint')
            alias = f'{name}_{CONTROL_ALIASES.get(original_joint, original_joint)}'
            actuator.set('name', alias)
            actuator.set('joint', alias)
            actuators.append(actuator)
    ET.SubElement(ET.SubElement(root, 'keyframe'), 'key', name='neutral')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    model.body_gravcomp[:] = 1
    model.geom_contype[:] = model.geom_conaffinity[:] = 0
    # Keep the non-boxing joints actively supported in a mild standing stance.
    for name in ('a', 'b'):
        for side, sign in (('left', 1), ('right', -1)):
            for joint, value in (('hip_pitch', -.10), ('knee', .20), ('ankle_pitch', -.10),
                                 ('shoulder_roll', .12*sign)):
                key = f'{name}_{side}_{joint}_joint'
                joint_id = model.joint(key).id
                model.key_qpos[0, model.jnt_qposadr[joint_id]] = value
        for actuator in range(model.nu):
            joint_id = model.actuator_trnid[actuator, 0]
            model.key_ctrl[0, actuator] = model.key_qpos[0, model.jnt_qposadr[joint_id]]
    # Fit virtual scoring targets to the real visual mesh bounds in torso coordinates.
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name in ('a', 'b'):
        torso = data.body(f'{name}_torso_link')
        for kind in ('head', 'torso'):
            visual = model.geom(f'{name}_{kind}_mesh').id
            mesh = model.geom_dataid[visual]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            points = model.mesh_vert[start:start+count] @ data.geom_xmat[visual].reshape(3, 3).T
            points = (points + data.geom_xpos[visual] - torso.xpos) @ torso.xmat.reshape(3, 3)
            center = (points.min(axis=0) + points.max(axis=0)) / 2
            half = (points.max(axis=0) - points.min(axis=0)) / 2
            target = model.geom(f'{name}_{kind}').id
            model.geom_pos[target] = center
            radius = max(half) if kind == 'head' else max(half[:2])
            model.geom_size[target, 0] = radius
            if kind == 'torso':
                model.geom_size[target, 1] = max(0.01, half[2] - radius)
    return model
