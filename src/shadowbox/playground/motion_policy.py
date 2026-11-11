"""Dependency-light exported motion policy transport and explicit feedback history.

The exported sidecar owns coordinates, history lags and action semantics.
A host must supply measured current fields and validated goals; this module
never invents a missing sensor or interprets a tactical action as joint control.
"""
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


class MotionHistory:
    """Policy-rate measurement and *accepted-command* rings from the sidecar."""
    def __init__(self, metadata, batch_size):
        self.specs = [s for s in metadata['policy_inputs']
                      if 'output_key' in s or 'observation_history' in s]
        self.batch_size = batch_size
        self.offset = np.asarray(metadata['control']['pd_action_offset'], dtype=np.float32)
        self.buffers = {}
        self.ready = False

    def _definition(self, spec):
        if 'observation_history' in spec:
            history = spec['observation_history']
            return history['current_key'], history['steps_back'], True
        return spec['output_key'], spec['steps_back'], False

    def reset(self, current: Mapping, mask=None):
        initial = {}
        depths = {}
        for spec in self.specs:
            key, steps, measured = self._definition(spec)
            if not steps or min(steps) < 1:
                raise ValueError('History must refer to earlier policy calls')
            width = spec['shape'][-1]
            if measured:
                value = np.asarray(current[key], dtype=spec['dtype'])
            else:
                fill = spec.get('fill_before_available')
                if fill == 'zeros':
                    value = np.zeros((self.batch_size, width), dtype=spec['dtype'])
                elif fill == 'pd_action_offset':
                    value = np.broadcast_to(self.offset, (self.batch_size, width)).copy()
                else:
                    raise ValueError(f'Unsupported history fill: {fill}')
            if value.shape != (self.batch_size, width) or not np.isfinite(value).all():
                raise ValueError(f'Invalid initial history field: {key}')
            initial[key] = value
            depths[key] = max(depths.get(key, 0), max(steps))
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (self.batch_size,) or not self.ready:
                raise ValueError('Partial reset requires an initialized history and a batch mask')
        for key, value in initial.items():
            if mask is None:
                self.buffers[key] = np.repeat(value[None], depths[key], axis=0)
            else:
                self.buffers[key][:, mask] = value[mask]
        self.ready = True

    def inputs(self):
        if not self.ready:
            raise ValueError('Reset motion history from initial measurements before inference')
        result = {}
        for spec in self.specs:
            key, steps, _ = self._definition(spec)
            selected = [self.buffers[key][step-1] for step in steps]
            value = selected[0] if len(spec['shape']) == 2 else np.stack(selected, axis=1)
            result[spec['key']] = value.copy()
        return result

    def accept(self, before_step: Mapping, applied_outputs: Mapping):
        """Commit once after acceptance; a rejected prediction never enters the ring."""
        if not self.ready:
            raise ValueError('Motion history is not initialized')
        values = {}
        for key, buffer in self.buffers.items():
            source = applied_outputs if key in ('actions', 'joint_pos_targets') else before_step
            value = np.asarray(source[key], dtype=buffer.dtype)
            if value.shape != buffer.shape[1:] or not np.isfinite(value).all():
                raise ValueError(f'Invalid accepted history field: {key}')
            values[key] = value
        for key, value in values.items():
            buffer = self.buffers[key]
            buffer[1:] = buffer[:-1].copy()
            buffer[0] = value


class MotionPolicy:
    """Stateless exported graph; the host explicitly resets and accepts history."""
    def __init__(self, sidecar, batch_size=1, runtime='onnx', device='cpu', script=None):
        self.sidecar = Path(sidecar).resolve()
        self.metadata = yaml.safe_load(self.sidecar.read_text())
        m = self.metadata
        if m.get('schema') != 'unified_pipeline/1' or m.get('dt') != 0.02:
            raise ValueError('Unqualified motion schema or control period')
        if batch_size < 1:
            raise ValueError('Motion batch size must be positive')
        self.batch_size = batch_size
        self.history = MotionHistory(m, batch_size)
        self.runtime = runtime
        self.device = device
        self.output_names = [s['name'] for s in m['policy_outputs']]
        if self.output_names != ['actions', 'joint_pos_targets', 'stiffness_targets', 'damping_targets']:
            raise ValueError('Unqualified motion output contract')
        if runtime == 'onnx':
            if device != 'cpu':
                raise ValueError('Only the qualified CPU ONNX path is supported')
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 8
            options.inter_op_num_threads = 1
            self.graph = ort.InferenceSession(str(self.sidecar.with_suffix('.onnx')),
                sess_options=options, providers=['CPUExecutionProvider'])
            actual = [s.name for s in self.graph.get_inputs()]
            expected = [s['name'] for s in m['policy_inputs']]
            if set(actual) != set(expected):
                raise ValueError('Graph inputs differ from the sidecar')
            self.script = None
        elif runtime in ('torchscript','torchscript_graph'):
            import torch
            if device != 'cuda:0' or not torch.cuda.is_available() or script is None:
                raise ValueError('Qualified TorchScript requires an explicit CUDA:0 artifact/runtime')
            self.script = Path(script).resolve()
            qualification = json.loads(self.script.with_suffix('.json').read_text())
            if (not qualification.get('passed') or batch_size not in qualification.get('qualified_batch_sizes', [])
                    or qualification.get('script_sha256') != hashlib.sha256(self.script.read_bytes()).hexdigest()
                    or qualification.get('sidecar_sha256') != hashlib.sha256(self.sidecar.read_bytes()).hexdigest()):
                raise ValueError('TorchScript artifact has no matching successful qualification')
            if runtime=='torchscript_graph' and not qualification.get('cuda_graph_qualification',{}).get('passed'):
                raise ValueError('CUDA graph replay has no successful parity qualification')
            self.graph = torch.jit.load(str(self.script), map_location=device).eval()
            self._capture=None
        else:
            raise ValueError('Motion runtime must be onnx, torchscript or torchscript_graph')

    def provenance(self):
        artifact = self.script if self.script else self.sidecar.with_suffix('.onnx')
        return dict(runtime=self.runtime, device=self.device, artifact=str(artifact),
                    sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    sidecar_sha256=hashlib.sha256(self.sidecar.read_bytes()).hexdigest())

    def reset(self, current, mask=None):
        self.history.reset(current, mask)

    def predict(self, current: Mapping, goals: Mapping, *, empty_scene=True):
        values = {**current, **goals, **self.history.inputs()}
        feed = {}
        for spec in self.metadata['policy_inputs']:
            shape = (self.batch_size, *spec['shape'][1:])
            key = spec['key']
            if key in values:
                value = np.asarray(values[key], dtype=spec['dtype'])
            elif spec['provider'] == 'scene' and empty_scene:
                value = np.zeros(shape, dtype=spec['dtype'])
                if spec['kind'] == 'scene_object_rot':
                    value[..., 3] = 1
            else:
                raise ValueError(f'Missing declared motion input: {key}')
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'Invalid motion input {key}: expected {shape}, got {value.shape}')
            feed[spec['name']] = np.ascontiguousarray(value)
        if self.runtime == 'onnx':
            arrays = self.graph.run(self.output_names, feed)
        else:
            import torch
            inputs = [torch.from_numpy(feed[s['name']]).to(self.device) for s in self.metadata['policy_inputs']]
            with torch.inference_mode():
                if self.runtime=='torchscript_graph':
                    if self._capture is None:
                        self._buffers=[value.clone() for value in inputs]
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(5):self.graph(*self._buffers)
                        torch.cuda.current_stream().wait_stream(stream)
                        self._capture=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(self._capture):
                            self._outputs=self.graph(*self._buffers)
                    for buffer,value in zip(self._buffers,inputs):buffer.copy_(value)
                    self._capture.replay()
                    outputs=self._outputs
                else:
                    outputs=self.graph(*inputs)
                arrays = [value.cpu().numpy() for value in outputs]
        result = dict(zip(self.output_names, arrays))
        if len(result) != 4:
            raise ValueError('Motion graph returned an incomplete command')
        for name, value in result.items():
            if value.shape != (self.batch_size, len(self.metadata['joint_names'])) or not np.isfinite(value).all():
                raise ValueError(f'Invalid motion output: {name}')
        for name, field in (('stiffness_targets','stiffness'),('damping_targets','damping')):
            if not np.allclose(result[name], self.metadata['control'][field], atol=1e-5, rtol=1e-6):
                raise ValueError(f'Motion gains differ from qualified robot profile: {name}')
        return result

    def accept(self, before_step, applied_outputs):
        self.history.accept(before_step, applied_outputs)


class MotionKinematics:
    """Derive the trained anchor/body frames from encoders and a pelvis IMU.

    The root-anchored FK graph comes from the same training bundle. It does not
    consume privileged body poses from the simulation or an opponent's plan.
    """
    def __init__(self, folder, metadata):
        import onnxruntime as ort
        folder=Path(folder).resolve()
        sidecar=yaml.safe_load((folder/'fk.yaml').read_text())
        if (sidecar.get('anchor_body_name') != metadata['robot']['root_body_name']
                or sidecar['body_names'] != metadata['body_names']
                or sidecar['joint_names'] != metadata['joint_names']):
            raise ValueError('Kinematics and policy robot profiles differ')
        sensors=metadata['sensors']
        if (sensors['imu']['body_name'] != 'pelvis'
                or sensors['imu']['mount_sensor_to_body_xyzw'] != [0.,0.,0.,1.]
                or sensors['localization']['body_name'] != 'pelvis'
                or sensors['localization']['offset_body_xyz'] != [0.,0.,0.]
                or not sensors['w_last']):
            raise ValueError('Unqualified motion IMU/localization profile')
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(folder/'fk.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        self.anchor=metadata['robot']['anchor_body_index']
        self.joints=len(metadata['joint_names'])

    def measure(self, joint_pos, joint_vel, pelvis_pos, pelvis_rot, pelvis_local_gyro):
        values=[np.asarray(value,dtype=np.float32) for value in
                (joint_pos,joint_vel,pelvis_pos,pelvis_rot,pelvis_local_gyro)]
        count=values[0].shape[0]
        shapes=((count,self.joints),(count,self.joints),(count,3),(count,4),(count,3))
        if any(value.shape!=shape or not np.isfinite(value).all() for value,shape in zip(values,shapes)):
            raise ValueError('Invalid physical motion measurements')
        if not np.allclose(np.linalg.norm(values[3],axis=-1),1.,atol=1e-3):
            raise ValueError('IMU quaternion must be unit length (xyzw)')
        positions=[];rotations=[]
        for index in range(count):
            pos,rot=self.session.run(['body_pos','body_rot'],dict(joint_pos=values[0][index:index+1],
                pelvis_pos=values[2][index:index+1],pelvis_rot=values[3][index:index+1]))
            positions.append(pos[0]);rotations.append(rot[0])
        positions=np.stack(positions);rotations=np.stack(rotations)
        current={'noisy.dof_pos':values[0], 'noisy.dof_vel':values[1],
                 'noisy.anchor_pos':positions[:,self.anchor],
                 'noisy.anchor_rot':rotations[:,self.anchor],
                 'noisy.root_local_ang_vel':values[4]}
        return current,positions,rotations
