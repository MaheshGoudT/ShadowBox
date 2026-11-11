"""Measure exported Metatron on real recorded inputs, including batch parity."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', type=Path, default=root/'tmp/playground/metatron/compiled/unified_pipeline.onnx')
    parser.add_argument('--feeds', type=Path, default=root/'tmp/playground/metatron/native-onnx-feed-trace-feeds.npz')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--output', type=Path, default=root/'tmp/playground/metatron/graph-benchmark.json')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(root/'tmp'):
        parser.error('Output must be under ShadowBox tmp/')
    import numpy as np
    if args.device == 'cuda':
        import torch  # Loads the selected CUDA/cuDNN libraries before ORT's provider.
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; no CPU fallback')
    import onnxruntime as ort
    providers = ['CPUExecutionProvider'] if args.device == 'cpu' else ['CUDAExecutionProvider']
    if providers[0] not in ort.get_available_providers():
        raise ValueError(f'Required provider absent: {providers[0]}')
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=providers)
    if providers[0] not in session.get_providers():
        raise ValueError('Requested provider failed initialization')
    names = [item.name for item in session.get_inputs()]
    with np.load(args.feeds) as archive:
        samples = [{name:archive[f'step_{step}_{name}'].copy() for name in names} for step in (0,16)]
    separate = [session.run(None, feed) for feed in samples]
    batch = {name:np.concatenate([s[name] for s in samples],axis=0) for name in names}
    together = session.run(None,batch)
    errors = {out.name:float(np.max(np.abs(value-np.concatenate([s[i] for s in separate],axis=0))))
              for i,(out,value) in enumerate(zip(session.get_outputs(),together))}
    if any(not np.isfinite(v).all() for v in together) or max(errors.values()) > 1e-4:
        raise ValueError(f'Batch parity failed: {errors}')
    timings = {}
    for size,feed in ((1,samples[0]),(2,batch)):
        for _ in range(3):session.run(None,feed)
        elapsed=[]
        for _ in range(20):
            start=time.monotonic();session.run(None,feed);elapsed.append(1000*(time.monotonic()-start))
        timings[size]=dict(mean_ms=float(np.mean(elapsed)),p95_ms=float(np.quantile(elapsed,.95)))
    report=dict(onnx_sha256=hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
        provider=session.get_providers(),device=args.device,threads=args.threads,batch_parity_errors=errors,
        timings=timings,scope='recorded native feeds; inference only, not boxing performance')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
