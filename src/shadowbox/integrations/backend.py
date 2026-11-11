"""Small, dependency-free contract for external simulation/training plugins."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Protocol
import uuid

import yaml


def detect_backend(path):
    if path is None:
        return None
    try:
        value = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f'Could not read configuration {path}') from error
    if isinstance(value, dict) and 'backend' in value:
        if value['backend'] not in ('mujoco', 'isaaclab', 'playground'):
            raise ValueError('backend must be mujoco, isaaclab or playground')
        return value['backend']
    return None


class Backend(Protocol):
    name: str
    output_dir: Path
    timeout_seconds: int

    def probe(self) -> dict: ...
    def command(self, stage: str, run_dir: Path) -> list[str]: ...
    def provenance(self) -> dict: ...


def run_backend(backend: Backend, stage: str, dry_run=False):
    """Run a plugin without shell expansion; retain command, logs and outcome."""
    run_dir = backend.output_dir / ('run-' + uuid.uuid4().hex[:12])
    command = backend.command(stage, run_dir)
    if dry_run:
        print(shlex.join(command))
        return 0
    readiness = backend.probe()
    if not readiness.get('ready'):
        raise ValueError(readiness.get('error', 'Backend is not ready'))
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = dict(backend=backend.name, stage=stage, command=command,
                    provenance=backend.provenance(), readiness=readiness,
                    started_utc=datetime.now(timezone.utc).isoformat(), status='running')
    capture = getattr(backend, 'capture_sources', None)
    if capture is not None:
        metadata['source_archive'] = capture(run_dir)
    manifest = run_dir / 'run.json'
    manifest.write_text(json.dumps(metadata, indent=2) + '\n')
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1')
    print(f'{backend.name} {stage}; logs: {run_dir}', flush=True)
    try:
        with (run_dir / 'runtime.log').open('w') as log:
            result = subprocess.run(command, cwd=run_dir, env=env, stdout=log,
                                    stderr=subprocess.STDOUT,
                                    timeout=backend.timeout_seconds, check=False)
        code = result.returncode
        if code == 0:
            finalize = getattr(backend, 'finalize', None)
            if finalize is not None:
                artifacts = finalize(stage, run_dir)
                if artifacts:
                    metadata.update(artifacts)
                    for name, value in artifacts.items():
                        print(f'{name}: {value}', flush=True)
        metadata.update(status='completed' if code == 0 else 'failed', exit_code=code)
        return code
    except subprocess.TimeoutExpired:
        metadata.update(status='timeout', exit_code=124)
        return 124
    except subprocess.SubprocessError as error:
        metadata.update(status='failed', exit_code=1, error=str(error))
        return 1
    except (OSError, KeyboardInterrupt) as error:
        metadata.update(status='interrupted' if isinstance(error, KeyboardInterrupt)
                        else 'failed', error=str(error))
        raise
    finally:
        metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
        manifest.write_text(json.dumps(metadata, indent=2) + '\n')


def load_backend(name, config=None):
    """Lazy registry: simulator packages never enter the arena's interpreter."""
    if name == 'playground':
        from .playground import PlaygroundBackend
        return PlaygroundBackend(config)
    raise ValueError(f'Unknown external backend: {name}')
