"""Focused regression tests for ShadowBox configuration loading and use."""

from pathlib import Path

import pytest
import yaml

from shadowbox.config import configure_logging, load_config
from shadowbox.simulation import Arena


def write_override(tmp_path: Path, values: dict) -> Path:
    path = tmp_path / "override.yaml"
    path.write_text(yaml.safe_dump(values))
    return path


def test_partial_yaml_override_retains_defaults(tmp_path):
    path = write_override(tmp_path, {"simulation": {"timestep": 0.015}})

    config = load_config(path)
    defaults = load_config()

    assert config["simulation"]["timestep"] == 0.015
    assert config["simulation"]["arena_half_size"] == defaults["simulation"]["arena_half_size"]
    assert config["motion"] == defaults["motion"]
    assert config["match"] == defaults["match"]


def test_unknown_key_is_rejected(tmp_path):
    path = write_override(tmp_path, {"simulation": {"typo_timestep": 0.01}})

    with pytest.raises(ValueError, match="Unknown configuration key: simulation.typo_timestep"):
        load_config(path)


@pytest.mark.parametrize("timestep", [0, -0.01, 0.0201])
def test_invalid_timestep_is_rejected(tmp_path, timestep):
    path = write_override(tmp_path, {"simulation": {"timestep": timestep}})

    with pytest.raises(ValueError, match=r"simulation\.timestep"):
        load_config(path)


def test_invalid_motion_limit_length_is_rejected(tmp_path):
    path = write_override(tmp_path, {"motion": {"velocity_limits": [1, 1, 1]}})

    with pytest.raises(ValueError, match=r"motion\.velocity_limits"):
        load_config(path)


def test_robot_spacing_override_updates_arena_root_positions(tmp_path):
    spacing = 3.2
    path = write_override(tmp_path, {"simulation": {"robot_spacing": spacing}})
    arena = Arena(config=load_config(path))

    assert arena.position("a")[0] == pytest.approx(-spacing / 2)
    assert arena.position("b")[0] == pytest.approx(spacing / 2)


@pytest.mark.parametrize('text', ['simulation: slow', 'simulation: {timestep: fast}', 'false'])
def test_malformed_config_sections_fail_clearly(tmp_path, text):
    path = tmp_path / 'bad.yaml'
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


def test_logging_creates_a_distinct_per_run_file(tmp_path):
    import logging

    config = load_config()
    config['logging']['directory'] = str(tmp_path)
    first = configure_logging(config)
    logging.getLogger('test').warning('first run')
    second = configure_logging(config)
    logging.getLogger('test').warning('second run')

    assert first != second
    assert first.parent == second.parent == tmp_path
    assert 'first run' in first.read_text()
    assert 'second run' in second.read_text()
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
        handler.close()


@pytest.mark.parametrize('text', [
    'logging: []',
    'viewer: {width: 0}',
    'viewer: {height: 2.5}',
    'viewer: {frames_per_second: .nan}',
    'simulation: {timestep: .inf}',
    'motion: {velocity_limits: [1, 1, 1, 1, 1, 1, .nan]}',
])
def test_malformed_or_unsafe_values_rejected(tmp_path, text):
    path = tmp_path / 'bad.yaml'
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)
