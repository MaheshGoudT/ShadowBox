"""Simple articulated humanoids in a MuJoCo ring; roots stay upright for now."""

import mujoco
import numpy as np

from ..config import load_config

JOINTS = ("x", "y", "yaw", "left_shoulder", "left_elbow",
          "right_shoulder", "right_elbow")
LIMITS = ((-2.6, 2.6), (-2.6, 2.6), (-6.4, 6.4),
          (-2.1, 0.4), (-2.2, 0.0), (-2.1, 0.4), (-2.2, 0.0))
GUARD = (-1.30, -2.15, -1.30, -2.15)


def robot_xml(name: str, color: str, config: dict) -> str:
    bound = config["arena_half_size"] - 0.2
    damping = config["translation_damping"]
    arms = []
    for side, y in (("left", 0.24), ("right", -0.24)):
        arms.append(f"""
        <body name="{name}_{side}_upper" pos="0 {y} 1.35">
          <joint name="{name}_{side}_shoulder" axis="0 1 0" range="-2.1 0.4" damping="4"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.32" size="0.065" mass="1"/>
          <body name="{name}_{side}_lower" pos="0 0 -0.32">
            <joint name="{name}_{side}_elbow" axis="0 1 0" range="-2.2 0" damping="3"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.30" size="0.055" mass="0.6"/>
            <geom name="{name}_{side}_glove" type="sphere" pos="0 0 -0.30"
                  size="0.11" mass="0.25" rgba="{color}"/>
          </body>
        </body>""")
    return f"""
    <body name="{name}">
      <joint name="{name}_x" type="slide" axis="1 0 0" range="-{bound} {bound}" damping="{damping}"/>
      <joint name="{name}_y" type="slide" axis="0 1 0" range="-{bound} {bound}" damping="{damping}"/>
      <joint name="{name}_yaw" axis="0 0 1" limited="false" damping="8"/>
      <geom name="{name}_torso" type="capsule" fromto="0 0 0.82 0 0 1.3"
            size="0.19" mass="8" rgba="{color}"/>
      <geom name="{name}_head" type="sphere" pos="0 0 1.60" size="0.15" mass="1"
            rgba="0.85 0.77 0.62 1"/>
      <geom type="box" pos="0 0 0.73" size="0.16 0.20 0.12" mass="2" rgba="{color}"/>
      <geom type="capsule" fromto="0 0.13 0.67 0 0.13 0.12" size="0.08" mass="2"/>
      <geom type="capsule" fromto="0 -0.13 0.67 0 -0.13 0.12" size="0.08" mass="2"/>
      <geom type="box" pos="0.06 0.13 0.06" size="0.16 0.09 0.06" mass="0.5"/>
      <geom type="box" pos="0.06 -0.13 0.06" size="0.16 0.09 0.06" mass="0.5"/>
      {''.join(arms)}
    </body>"""


def make_model(timestep: float = 0.01, config=None) -> mujoco.MjModel:
    config = config or load_config()["simulation"]
    half = config["arena_half_size"]
    actuators = []
    for name in ("a", "b"):
        for joint, limits in zip(JOINTS, LIMITS):
            lo, hi = limits
            if joint in ("x", "y"):
                lo, hi = -half + 0.2, half - 0.2
            gain = config["translation_gain"] if joint in ("x", "y") else config["joint_gain"]
            control_limit = 'ctrllimited="false"' if joint == "yaw" else f'ctrlrange="{lo} {hi}"'
            actuators.append(f'<position name="{name}_{joint}" joint="{name}_{joint}" '
                             f'kp="{gain}" {control_limit}/>')
    rails = []
    for z in (0.45, 0.8, 1.15):
        for axis in (-1, 1):
            rails.append(f'<geom type="capsule" fromto="-{half} {axis*half} {z} {half} {axis*half} {z}" size="0.025" rgba="0.7 0.72 0.8 1"/>')
            rails.append(f'<geom type="capsule" fromto="{axis*half} -{half} {z} {axis*half} {half} {z}" size="0.025" rgba="0.7 0.72 0.8 1"/>')
    xml = f"""
    <mujoco model="ShadowBox">
      <compiler angle="radian" autolimits="true"/>
      <option timestep="{timestep}" integrator="implicitfast"/>
      <default><geom rgba="0.33 0.37 0.44 1" contype="0" conaffinity="0"/></default>
      <visual><global offwidth="1000" offheight="750"/></visual>
      <worldbody>
        <light pos="0 -2 6" diffuse="0.9 0.9 0.9"/>
        <geom name="floor" type="plane" size="{half+0.1} {half+0.1} 0.1" rgba="0.13 0.18 0.24 1"/>
        {''.join(rails)}
        {robot_xml('a', '0.18 0.5 0.95 1', config)}
        {robot_xml('b', '0.95 0.26 0.18 1', config)}
      </worldbody>
      <actuator>{''.join(actuators)}</actuator>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    model.body_gravcomp[:] = 1  # Stabilized motion baseline; free-base balance comes later.
    return model
