"""Render recorded Playground physics states with their saved compiled model."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--world', type=int, default=0)
    parser.add_argument('--frames', type=int, default=1, help='1 for PNG; 2..200 for animated GIF')
    parser.add_argument('--stride', type=int, default=5, help='Recorded 20 ms control steps per animation frame')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--output', type=Path, default=Path('tmp/playground/replay.png'))
    args = parser.parse_args()
    if not 1<=args.frames<=200 or not 1<=args.stride<=50:
        parser.error('Require 1..200 frames and stride 1..50')
    if not 160<=args.width<=1920 or not 120<=args.height<=1080:
        parser.error('Require width 160..1920 and height 120..1080')
    if args.frames>1 and args.output.suffix.lower()!='.gif':
        parser.error('Animated replay output requires .gif')
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if not output.is_relative_to(root / 'tmp'):
        parser.error('Generated captures must be inside ShadowBox tmp/')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    import numpy as np
    from PIL import Image
    model = mujoco.MjModel.from_binary_path(str(args.run_dir / 'model.mjb'))
    with np.load(args.run_dir / 'transitions.npz') as trace:
        qpos = trace['qpos']
        if not 0 <= args.frame < qpos.shape[0] or not 0 <= args.world < qpos.shape[1]:
            parser.error('Requested frame/world is outside the recorded trace')
        indices=list(range(args.frame,min(qpos.shape[0],args.frame+args.frames*args.stride),args.stride))
        states=qpos[indices,args.world].copy()
        state = states[0]
    if state.shape != (model.nq,):
        parser.error('Saved model and recorded state dimensions differ')
    # Keep measured glove hitboxes in physics but show the G1 hand meshes.
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ''
        if name.endswith('_glove'):
            model.geom_rgba[geom_id, 3] = 0
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    data = mujoco.MjData(model)
    data.qpos[:] = state
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, .7]
    camera.distance = 3
    camera.azimuth = 100
    camera.elevation = -15
    output.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        captures=[]
        for state in states:
            data.qpos[:]=state
            mujoco.mj_forward(model,data)
            renderer.update_scene(data, camera=camera)
            capture=Image.fromarray(renderer.render())
            captures.append(capture.convert('P',palette=Image.Palette.ADAPTIVE,colors=128) if args.frames>1 else capture)
        if args.frames>1:
            captures[0].save(output,save_all=True,append_images=captures[1:],duration=20*args.stride,loop=0,disposal=2)
        else:
            captures[0].save(output)
    print(output)


if __name__ == '__main__':
    main()
