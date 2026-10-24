"""Load and validate the shared YAML defaults and optional overrides."""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import math

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = PROJECT_ROOT / 'configs' / 'simulation.yaml'
DEFAULT_CONFIG = PROJECT_ROOT / 'configs' / 'unitree.yaml'


def _read_yaml(path):
    try:
        value = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f'Could not read configuration {path}: {exc}') from exc
    return {} if value is None else value


def load_config(path=None) -> dict:
    defaults = _read_yaml(BASE_CONFIG)
    if not isinstance(defaults, dict):
        raise ValueError('Default configuration must be a mapping')

    def merge(base, values, prefix=''):
        if not isinstance(values, dict):
            raise ValueError(f'{prefix or "Configuration"} must be a mapping')
        for key, value in values.items():
            label = f'{prefix}.{key}' if prefix else str(key)
            if key not in base:
                raise ValueError(f'Unknown configuration key: {label}')
            if isinstance(base[key], dict):
                if not isinstance(value, dict):
                    raise ValueError(f'{label} must be a mapping')
                merge(base[key], value, label)
            else:
                expected = base[key]
                if isinstance(expected, bool):
                    valid = isinstance(value, bool)
                elif isinstance(expected, (int, float)):
                    valid = isinstance(value, (int, float)) and not isinstance(value, bool)
                else:
                    valid = isinstance(value, type(expected))
                if not valid:
                    raise ValueError(f'Invalid value type for {label}')
                base[key] = value

    selected = Path(path).resolve() if path is not None else DEFAULT_CONFIG
    if selected != BASE_CONFIG:
        merge(defaults, _read_yaml(DEFAULT_CONFIG))
        if selected != DEFAULT_CONFIG:
            merge(defaults, _read_yaml(selected))

    def finite(value, label='Configuration'):
        if isinstance(value, dict):
            for key, item in value.items():
                finite(item, f'{label}.{key}')
        elif isinstance(value, list):
            for index, item in enumerate(value):
                finite(item, f'{label}[{index}]')
        elif isinstance(value, bool) or value is None or isinstance(value, str):
            return
        elif isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise ValueError(f'{label} must be finite')
        else:
            raise ValueError(f'{label} has an unsupported value')

    finite(defaults)
    required = ('simulation', 'motion', 'match', 'policies', 'safety', 'scoring', 'viewer', 'logging', 'training')
    for section in required:
        if not isinstance(defaults.get(section), dict):
            raise ValueError(f'{section} must be a mapping')
    simulation = defaults['simulation']
    if not 0 < simulation['timestep'] <= 0.02:
        raise ValueError('simulation.timestep must be in (0, 0.02]')
    if not 1 < simulation['arena_half_size'] <= 20:
        raise ValueError('simulation.arena_half_size must be in (1, 20]')
    if not defaults['safety']['min_separation'] + .05 < simulation['robot_spacing'] < 2 * (simulation['arena_half_size'] - 0.5):
        raise ValueError('Initial robot spacing must fit inside the ring')
    for key in ('velocity_limits', 'acceleration_limits'):
        values = defaults['motion'][key]
        if not isinstance(values, list) or len(values) != 7 or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f'motion.{key} requires seven positive finite values')
    for key in ('rounds',):
        if not isinstance(defaults['match'][key], int) or isinstance(defaults['match'][key], bool) or defaults['match'][key] < 1:
            raise ValueError(f'match.{key} must be a positive integer')
    if defaults['match']['round_seconds'] <= 0 or defaults['match']['break_seconds'] < 0:
        raise ValueError('Round duration must be positive; break duration nonnegative')
    if not 0 < defaults['safety']['monitor_separation'] <= defaults['safety']['min_separation']:
        raise ValueError('Separation thresholds must be positive and ordered')
    if not isinstance(defaults['safety']['path_samples'], int) or isinstance(defaults['safety']['path_samples'], bool) or defaults['safety']['path_samples'] < 2:
        raise ValueError('safety.path_samples must be an integer >= 2')
    viewer = defaults['viewer']
    for key in ('width', 'height'):
        if not isinstance(viewer[key], int) or isinstance(viewer[key], bool) or viewer[key] < 1:
            raise ValueError(f'viewer.{key} must be a positive integer')
    if viewer['frames_per_second'] <= 0:
        raise ValueError('viewer.frames_per_second must be positive')
    log = defaults['logging']
    if not isinstance(log['directory'], str) or not log['directory']:
        raise ValueError('logging.directory must be a non-empty path')
    if not isinstance(log['level'], str) or log['level'].upper() not in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'NOTSET'):
        raise ValueError('logging.level must be a valid logging level name')
    if 'max_bytes' in log and (not isinstance(log['max_bytes'], int) or isinstance(log['max_bytes'], bool) or log['max_bytes'] < 1):
        raise ValueError('logging.max_bytes must be a positive integer')
    if 'backup_count' in log and (not isinstance(log['backup_count'], int) or isinstance(log['backup_count'], bool) or log['backup_count'] < 0):
        raise ValueError('logging.backup_count must be a nonnegative integer')
    return deepcopy(defaults)


def configure_logging(config: dict) -> Path:
    import logging
    from logging.handlers import RotatingFileHandler
    from uuid import uuid4

    settings = config['logging']
    directory = artifact_path(settings['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    logfile = directory / f"shadowbox-{uuid4().hex}.log"
    handler = RotatingFileHandler(logfile, maxBytes=settings.get('max_bytes', 5_000_000),
                                  backupCount=settings.get('backup_count', 3), encoding='utf-8')
    handler.setLevel(getattr(logging, settings['level'].upper()))
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    root = logging.getLogger()
    for old in root.handlers[:]:
        root.removeHandler(old)
        old.close()
    root.setLevel(handler.level)
    root.addHandler(handler)
    return logfile


def artifact_path(value) -> Path:
    """Keep relative generated paths inside this project, regardless of shell cwd."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def source_fingerprint() -> str:
    """Identify the Python sources used for a run, including uncommitted edits."""
    import hashlib
    digest = hashlib.sha256()
    package = Path(__file__).parent
    for path in sorted(package.rglob('*.py')):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]
