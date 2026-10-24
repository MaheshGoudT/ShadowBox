"""Validated G1 motion references, including future video-derived joint tracks."""

from dataclasses import dataclass
import json
import math

from ..config import artifact_path


@dataclass(frozen=True)
class MotionClip:
    source: dict
    fps: float
    frames: list[dict]

    @property
    def duration(self):
        return (len(self.frames) - 1) / self.fps

    def sample(self, seconds):
        """Linearly interpolate joint and root targets; contact labels are discrete."""
        position = min(max(seconds * self.fps, 0.0), len(self.frames) - 1)
        first = min(int(position), len(self.frames) - 2)
        fraction = position - first
        a, b = self.frames[first:first + 2]
        joints = {name: a['joints'][name] + fraction * (b['joints'][name] - a['joints'][name])
                  for name in a['joints']}
        root = [x + fraction * (y - x) for x, y in zip(a['root'], b['root'])]
        return joints, root, a['planted'] if fraction < .5 else b['planted']


def load_motion_clip(path, arena):
    """Reject ambiguous, out-of-range references before simulation playback."""
    try:
        value = json.loads(artifact_path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'Could not read motion clip {path}') from error
    if not isinstance(value, dict) or value.get('schema_version') != 1:
        raise ValueError('Motion clip requires schema_version 1')
    source = value.get('source')
    if not isinstance(source, dict) or source.get('kind') not in ('hand_authored', 'video_estimate', 'mocap'):
        raise ValueError('Motion clip requires a source.kind')
    if source['kind'] != 'hand_authored' and not source.get('uri'):
        raise ValueError('Video and MoCap clips require a source uri')
    fps, frames = value.get('fps'), value.get('frames')
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 0 < fps <= 240:
        raise ValueError('Motion clip fps must be in (0, 240]')
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError('Motion clip requires at least two frames')
    limits = arena.full_limits('a')
    names = None
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError('Motion clip frames must be mappings')
        joints = frame.get('joints')
        if not isinstance(joints, dict) or not joints or set(joints) - set(limits):
            raise ValueError('Motion clip has missing or unknown joints')
        if names is None:
            names = set(joints)
        elif set(joints) != names:
            raise ValueError('Every motion clip frame must name the same joints')
        if any(isinstance(angle, bool) or not isinstance(angle, (int, float))
               or not math.isfinite(angle) or not limits[name][0] <= angle <= limits[name][1]
               for name, angle in joints.items()):
            raise ValueError('Motion clip joint target exceeds G1 limits')
        root = frame.setdefault('root', [0.0, 0.0, 0.0])
        if not isinstance(root, list) or len(root) != 3 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in root):
            raise ValueError('Motion clip root must be finite [forward, left, yaw]')
        planted = frame.setdefault('planted', ['left', 'right'])
        if not isinstance(planted, list) or len(planted) != len(set(planted)) or set(planted) - {'left', 'right'}:
            raise ValueError('Motion clip planted feet must be left and/or right')
    return MotionClip(source, float(fps), frames)
