"""Export the bundle's deterministic pelvis-to-body kinematics for a new host."""
import os
from pathlib import Path
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tmp/playground/protomotions-source'
BUNDLE=Path('/home/mahesh/Desktop/huggingfacehub/models/nvidia/Metatron-v1/ada816723d62c194c82ffd8fab97a3d8a001f94d/G1')


def main():
    revision=subprocess.run(['git','-C',str(SOURCE),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if revision!='13de0cccd06f53ce5c9cf9ec7d1932c7da73610f':
        raise ValueError('ProtoMotions source differs from qualification pin')
    os.environ['TORCH_COMPILE_DISABLE']='1'
    sys.path.insert(0,str(SOURCE))
    os.chdir(SOURCE)
    import torch
    from protomotions.utils.experiment_configs import load_experiment_configs_from_checkpoint
    from protomotions.utils.baked_fk import BakedAnchorFK
    from deployment.export_fk_to_onnx import FkSidecar, write_fk_graph
    checkpoint=BUNDLE/'inference_last.ckpt'
    resolved=load_experiment_configs_from_checkpoint(checkpoint)
    resolved.robot.asset.asset_root=str(ROOT/'tmp/playground/protomotions-assets')
    info=resolved.robot.kinematic_info
    bodies=tuple(info.body_names)
    joints=tuple(info.dof_names)
    fk=BakedAnchorFK(info,0,torch.arange(info.num_bodies))
    sidecar=FkSidecar(body_names=bodies,joint_names=joints,anchor_body_name=bodies[0],anchor_body_index=0,
        inputs=(('joint_pos','physical.dof_pos',(1,len(joints))),
                ('pelvis_pos','physical.anchor_pos',(1,3)),('pelvis_rot','physical.anchor_rot',(1,4))),
        outputs=(('body_pos',(1,len(bodies),3)),('body_rot',(1,len(bodies),4))),checkpoint=checkpoint.name)
    write_fk_graph(fk,info,sidecar,ROOT/'tmp/playground/metatron/compiled/pelvis_fk',validate=True,samples=16)


if __name__=='__main__':
    main()
