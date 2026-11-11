"""Capture the pinned upstream graph boundary and qualify a CUDA TorchScript copy.

The upstream exporter constructs observations, frozen model and action conversion.
Only its final ONNX serialization call is intercepted; this harness does not
reimplement policy construction. Validation uses recorded native inputs.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tmp/playground/protomotions-source'
BUNDLE=Path('/home/mahesh/Desktop/huggingfacehub/models/nvidia/Metatron-v1/ada816723d62c194c82ffd8fab97a3d8a001f94d/G1')


def main():
    revision=subprocess.run(['git','-C',str(SOURCE),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if revision!='13de0cccd06f53ce5c9cf9ec7d1932c7da73610f':
        raise ValueError('ProtoMotions source differs from qualification pin')
    os.environ['TORCH_COMPILE_DISABLE']='1'
    sys.path.insert(0,str(SOURCE));os.chdir(SOURCE)
    import torch
    import numpy as np
    import onnxruntime as ort
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable; no fallback')
    from protomotions.utils.experiment_configs import load_experiment_configs_from_checkpoint
    from deployment.export_to_onnx import export_from_configs
    resolved=load_experiment_configs_from_checkpoint(BUNDLE/'inference_last.ckpt')
    resolved.robot.asset.asset_root=str(ROOT/'tmp/playground/protomotions-assets')
    captured={}
    class Captured(Exception):pass
    def capture(module,inputs,*args,**kwargs):
        captured.update(module=module,inputs=inputs,names=kwargs['input_names'],outputs=kwargs['output_names'])
        raise Captured()
    folder=ROOT/'tmp/playground/metatron/compiled'
    try:
        with patch('torch.onnx.export',side_effect=capture):
            export_from_configs(resolved,str(BUNDLE/'inference_last.ckpt'),str(folder/'script-build'),validate=False)
    except Captured:pass
    if not captured:raise ValueError('Upstream export boundary was not reached')
    module=captured['module'].eval().to('cuda:0')
    inputs=tuple(value.to('cuda:0') for value in captured['inputs'])
    with torch.inference_mode():
        module(*inputs)  # Warm static FK constants on the actual target device.
        script=torch.jit.trace(module,inputs,strict=False,check_trace=False)
        script=torch.jit.freeze(script.eval())
    target=folder/'unified_pipeline.cuda.ts'
    script.save(str(target))
    replay=torch.jit.load(str(target),map_location='cuda:0').eval()
    options=ort.SessionOptions();options.intra_op_num_threads=8;options.inter_op_num_threads=1
    oracle=ort.InferenceSession(str(folder/'unified_pipeline.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    data=np.load(ROOT/'tmp/playground/metatron/native-onnx-feed-trace-feeds.npz')
    cases=[{name:data[f'step_{step}_{name}'] for name in captured['names']} for step in (0,1,8,16,31)]
    cases.append({name:np.concatenate([cases[0][name],cases[-1][name]],axis=0) for name in captured['names']})
    errors={name:0. for name in captured['outputs']}
    times={}
    with torch.inference_mode():
        for feed in cases:
            expected=oracle.run(captured['outputs'],feed)
            tensors=tuple(torch.from_numpy(feed[name]).to('cuda:0') for name in captured['names'])
            actual=replay(*tensors)
            for name,value,reference in zip(captured['outputs'],actual,expected):
                difference=float(np.max(np.abs(value.cpu().numpy()-reference)))
                errors[name]=max(errors[name],difference)
            size=next(iter(feed.values())).shape[0]
            for _ in range(3):replay(*tensors)
            torch.cuda.synchronize();samples=[]
            for _ in range(20):
                tick=time.monotonic();replay(*tensors);torch.cuda.synchronize();samples.append(1000*(time.monotonic()-tick))
            times[size]=dict(mean_ms=float(np.mean(samples)),p95_ms=float(np.quantile(samples,.95)))
    report=dict(device='cuda:0',hardware=torch.cuda.get_device_name(0),torch=torch.__version__,
        script_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        onnx_sha256=hashlib.sha256((folder/'unified_pipeline.onnx').read_bytes()).hexdigest(),
        sidecar_sha256=hashlib.sha256((folder/'unified_pipeline.yaml').read_bytes()).hexdigest(),
        qualified_batch_sizes=[1,2],max_errors=errors,timings=times,passed=max(errors.values())<=1e-4,
        scope='five native input frames plus batch-two parity; inference only')
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise ValueError('CUDA TorchScript parity failed')


if __name__=='__main__':main()
