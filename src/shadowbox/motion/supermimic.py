"""Optional named G1 reference poses; trained SuperMimic inference is not implemented."""

import json
import math

from ..config import artifact_path


class SuperMimic:
    def __init__(self, config):
        settings = config.get("supermimic", {})
        self.enabled = settings.get("enabled", False)
        self.model_path = settings.get("model_path", "")
        self.motion_library = settings.get("motion_library", "")
        self.policy = None
        self.motions = None

        if not self.enabled:
            return
        if config["simulation"]["model"] != "unitree_g1":
            raise ValueError("SuperMimic reference poses require the Unitree G1 model")
        if self.model_path:
            raise ValueError("Trained SuperMimic model inference is not implemented")
        if not self.motion_library:
            raise ValueError("supermimic.motion_library is required when enabled")

        path = artifact_path(self.motion_library)
        try:
            motions = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not load SuperMimic reference poses from {path}") from error
        if not isinstance(motions, dict) or not motions:
            raise ValueError("SuperMimic reference poses must be a nonempty mapping")
        for name, joints in motions.items():
            if not isinstance(joints, dict) or not joints:
                raise ValueError(f"SuperMimic motion {name} has no joint targets")
            for joint, values in joints.items():
                samples = values if isinstance(values, list) else [values]
                if not samples or any(isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) for value in samples):
                    raise ValueError(f"SuperMimic motion {name}.{joint} needs finite angles")
        self.motions = motions

    def get_motion_targets(self, robot, motion_name, frame=0):
        if not self.enabled or motion_name not in self.motions:
            return None
        return {joint: values[frame % len(values)] if isinstance(values, list) else values
                for joint, values in self.motions[motion_name].items()}

    def track_motion(self, robot, motion_sequence):
        if not self.enabled:
            return []
        return [self.get_motion_targets(robot, name) for name in motion_sequence]
