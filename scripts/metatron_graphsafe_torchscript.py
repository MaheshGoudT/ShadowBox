"""Replace scalar host copies with equivalent device scalar creation in a trace.

This targets Torch 2.13's traced ``torch.tensor(scalar, device=...)`` lowering.
It changes allocation, not values. The original artifact is preserved; a separate
full replay parity gate must pass before this derivative can be used.
"""
import hashlib
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    import torch
    folder=ROOT/'tmp/playground/metatron/compiled'
    source=folder/'unified_pipeline.cuda.ts'
    model=torch.jit.load(str(source),map_location='cuda:0').eval()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size',type=int,choices=(1,2))
    args=parser.parse_args()
    graph=model.graph
    torch._C._jit_pass_inline(graph)
    if args.batch_size:
        import numpy as np,yaml
        meta=yaml.safe_load((folder/'unified_pipeline.yaml').read_text())
        archive=np.load(ROOT/'tmp/playground/metatron/native-onnx-feed-trace-feeds.npz')
        tensors=[torch.from_numpy(np.repeat(archive[f'step_0_{s["name"]}'],args.batch_size,axis=0)).to('cuda:0')
                 for s in meta['policy_inputs']]
        inputs=list(graph.inputs())[1:]
        if len(inputs)!=len(tensors):raise ValueError('Graph input/sidecar mismatch')
        for value,tensor in zip(inputs,tensors):value.setType(torch._C.TensorType.create_from_tensor(tensor))
        torch._C._jit_pass_propagate_shapes_on_graph(graph)
        torch._C._jit_pass_peephole(graph,False)
        torch._C._jit_pass_constant_propagation(graph)
    specialized_ints=0
    if args.batch_size:
        allowed={'prim::Constant','aten::size','aten::dim','prim::dtype','prim::device'}
        arithmetic={'prim::NumToTensor','aten::div','aten::floor_divide','aten::mul','aten::add','aten::sub',
                    'aten::Int','aten::Float','aten::to','aten::scalar_tensor','aten::tensor'}
        def shape_only(value,seen=None):
            node=value.node()
            if node.kind() in allowed:return True
            if node.kind() not in arithmetic:return False
            return all(shape_only(v) for v in node.inputs())
        nodes=[node for node in graph.findAllNodes('aten::Int',recurse=True)
               if node.owningBlock()==graph.block() and shape_only(node.output())]
        if nodes:
            probe=graph.copy()
            original_nodes=list(graph.findAllNodes('aten::Int',recurse=True))
            copied_nodes=list(probe.findAllNodes('aten::Int',recurse=True))
            if len(original_nodes)!=len(copied_nodes):raise ValueError('Shape-probe topology changed')
            indices={n.output().debugName():i for i,n in enumerate(original_nodes)}
            outputs=[copied_nodes[indices[n.output().debugName()]].output() for n in nodes]
            probe.eraseOutput(0)
            if list(probe.inputs())[0].uses():raise ValueError('Frozen graph still reads module state')
            probe.eraseInput(0)
            construct=probe.create('prim::TupleConstruct',outputs,1)
            construct.output().setType(torch._C.TupleType([value.type() for value in outputs]))
            construct.insertBefore(probe.return_node())
            probe.registerOutput(construct.output())
            torch._C._jit_pass_dce(probe)
            evaluate=torch._C._create_function_from_graph('fixed_shape_values',probe)
            with torch.inference_mode():values=evaluate(*tensors)
            for node,value in zip(nodes,values):
                graph.setInsertPoint(node)
                constant=graph.insertConstant(int(value))
                node.output().replaceAllUsesWith(constant)
                node.destroy();specialized_ints+=1
            graph.setInsertPoint(graph.return_node())
            torch._C._jit_pass_constant_propagation(graph)
    changed=0
    for node in list(graph.findAllNodes('aten::tensor',recurse=True)):
        inputs=list(node.inputs())
        if str(inputs[0].type()) not in ('int','float','bool'):
            continue
        if inputs[3].toIValue() is not False:
            raise ValueError('Only constant non-gradient scalar creation is qualified')
        graph.setInsertPoint(node)
        none=graph.insertConstant(None)
        replacement=graph.create('aten::scalar_tensor',[inputs[0],inputs[1],none,inputs[2],none],1)
        replacement.insertBefore(node)
        replacement.output().setType(node.output().type())
        node.output().replaceAllUsesWith(replacement.output())
        node.destroy()
        changed+=1
    graph.setInsertPoint(graph.return_node())
    if not changed:raise ValueError('No qualified scalar allocations were rewritten')
    torch._C._jit_pass_constant_propagation(graph)
    moved_indices=0
    for node in list(graph.findAllNodes('aten::index',recurse=True)):
        index_list=list(node.inputs())[1]
        items=index_list.toIValue()
        if items is None and index_list.node().kind()=='prim::ListConstruct':
            items=[value.toIValue() for value in index_list.node().inputs()]
            if any(value is None and original.node().kind()!='prim::Constant'
                   for value,original in zip(items,index_list.node().inputs())):
                continue
        if not isinstance(items,list) or not any(isinstance(v,torch.Tensor) and v.device.type=='cpu' for v in items):
            continue
        graph.setInsertPoint(node)
        indices=[None if value is None else value.to('cuda:0') for value in items]
        replacement=graph.insertConstant(indices)
        replacement.setType(index_list.type())
        node.replaceInput(1,replacement)
        moved_indices+=1
    graph.setInsertPoint(graph.return_node())
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)
    target=folder/(f'unified_pipeline.cuda.graphsafe.b{args.batch_size}.ts' if args.batch_size else 'unified_pipeline.cuda.graphsafe.ts')
    model.save(str(target))
    report=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                candidate_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                torch=torch.__version__,specialized_batch_size=args.batch_size,specialized_shape_integers=specialized_ints,rewritten_scalar_allocations=changed,moved_constant_index_lists=moved_indices,scope='allocation-only candidate; not qualified until replay parity passes')
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
