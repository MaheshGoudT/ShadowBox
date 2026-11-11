"""Export the verified native G1 bundle through pinned upstream export/parity."""
import hashlib
import os
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'tmp/playground/protomotions-source'
BUNDLE = Path('/home/mahesh/Desktop/huggingfacehub/models/nvidia/Metatron-v1/ada816723d62c194c82ffd8fab97a3d8a001f94d/G1')


def main():
    revision=subprocess.run(['git','-C',str(SOURCE),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if revision!='13de0cccd06f53ce5c9cf9ec7d1932c7da73610f':
        raise ValueError('ProtoMotions source differs from qualification pin')
    os.environ['TORCH_COMPILE_DISABLE'] = '1'
    sys.path.insert(0, str(SOURCE))
    os.chdir(SOURCE)
    from protomotions.utils.experiment_configs import load_experiment_configs_from_checkpoint
    from deployment.export_to_onnx import export_from_configs
    checkpoint = BUNDLE/'inference_last.ckpt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != 'e5fc2e47a8f9ee9b96a674cabbf236b3c510f887427645a14bc1779b10f0b43a':
        raise ValueError('Checkpoint differs from verified native release')
    resolved = load_experiment_configs_from_checkpoint(checkpoint)
    resolved.robot.asset.asset_root = str(ROOT/'tmp/playground/protomotions-assets')
    output = ROOT/'tmp/playground/metatron/compiled'
    export_from_configs(resolved, str(checkpoint), str(output), validate=True, host='robot')


if __name__ == '__main__':
    main()
