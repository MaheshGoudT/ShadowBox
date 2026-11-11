"""Qualify static CUDA graph replay against the saved TorchScript policy."""
import hashlib
import argparse
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    import numpy as np
    import torch
    import yaml
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable; no fallback')
    torch.set_float32_matmul_precision('highest')
    folder=ROOT/'tmp/playground/metatron/compiled'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphsafe',action='store_true')
    parser.add_argument('--batch-size',type=int,choices=(1,2))
    args=parser.parse_args()
    artifact=folder/(f'unified_pipeline.cuda.graphsafe.b{args.batch_size}.ts' if args.graphsafe and args.batch_size
                     else ('unified_pipeline.cuda.graphsafe.ts' if args.graphsafe else 'unified_pipeline.cuda.ts'))
    candidate=None
    if args.graphsafe:
        candidate=json.loads(artifact.with_suffix('.json').read_text())
        parent=folder/'unified_pipeline.cuda.ts'
        parent_qualification=json.loads(parent.with_suffix('.json').read_text())
        if (not parent_qualification.get('passed')
                or parent_qualification['script_sha256']!=hashlib.sha256(parent.read_bytes()).hexdigest()
                or candidate['source_sha256']!=parent_qualification['script_sha256']
                or candidate['candidate_sha256']!=hashlib.sha256(artifact.read_bytes()).hexdigest()
                or parent_qualification['sidecar_sha256']!=hashlib.sha256((folder/'unified_pipeline.yaml').read_bytes()).hexdigest()
                or parent_qualification['onnx_sha256']!=hashlib.sha256((folder/'unified_pipeline.onnx').read_bytes()).hexdigest()):
            raise ValueError('Candidate lineage or parent qualification does not match current artifacts')
        if candidate.get('specialized_batch_size') not in (None,args.batch_size):
            raise ValueError('Replay batch differs from shape specialization')
    model=torch.jit.load(str(artifact),map_location='cuda:0').eval()
    metadata=yaml.safe_load((folder/'unified_pipeline.yaml').read_text())
    names=[s['name'] for s in metadata['policy_inputs']]
    archive=np.load(ROOT/'tmp/playground/metatron/native-onnx-feed-trace-feeds.npz')
    feeds=[{name:archive[f'step_{step}_{name}'] for name in names} for step in range(32)]
    original=torch.jit.load(str(folder/'unified_pipeline.cuda.ts'),map_location='cuda:0').eval()
    report=dict(feed_trace_sha256=hashlib.sha256((ROOT/'tmp/playground/metatron/native-onnx-feed-trace-feeds.npz').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),device=torch.cuda.get_device_name(0),results={})
    with torch.inference_mode():
        for size in ((args.batch_size,) if args.batch_size else (1,2)):
            sample={name:np.concatenate([feeds[i][name] for i in range(size)],axis=0) for name in names}
            buffers=[torch.from_numpy(sample[name]).to('cuda:0').clone() for name in names]
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(5):model(*buffers)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):outputs=model(*buffers)
            worst=0.
            for tick in range(0,32,size):
                feed={name:np.concatenate([feeds[(tick+i)%32][name] for i in range(size)],axis=0) for name in names}
                tensors=[torch.from_numpy(feed[name]).to('cuda:0') for name in names]
                reference=original(*tensors)
                for buffer,value in zip(buffers,tensors):buffer.copy_(value)
                graph.replay()
                if len(outputs)!=4 or len(reference)!=4:raise ValueError('Incomplete motion command')
                for a,b in zip(outputs,reference):
                    if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                        raise ValueError('Invalid replay output shape or nonfinite values')
                    worst=max(worst,float((a-b).abs().max().cpu()))
            timings=[]
            for _ in range(30):
                tick=time.monotonic();graph.replay();torch.cuda.synchronize();timings.append(1000*(time.monotonic()-tick))
            report['results'][size]=dict(max_replay_error=worst,mean_ms=float(np.mean(timings)),p95_ms=float(np.quantile(timings,.95)))
    report['passed']=all(v['max_replay_error']<=1e-5 for v in report['results'].values())
    (artifact.with_suffix('.qualification.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if args.graphsafe:
        candidate.update(passed=report['passed'],script_sha256=report['script_sha256'],
            sidecar_sha256=parent_qualification['sidecar_sha256'],onnx_sha256=parent_qualification['onnx_sha256'],
            qualified_batch_sizes=list(report['results']),cuda_graph_qualification=report,
            scope='fixed shape/allocation derivative; recorded native numerical replay only, physical transfer separately evaluated')
        artifact.with_suffix('.json').write_text(json.dumps(candidate,indent=2)+'\n')
    if not report['passed']:raise ValueError('CUDA replay parity failed')


if __name__=='__main__':main()
