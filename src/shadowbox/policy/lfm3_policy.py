"""LFM-3 decision model integration for ShadowBox."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# llama.cpp bindings for GGUF inference
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("Warning: llama_cpp not available. Install with: pip install llama-cpp-python")


class LFM3Policy:
    """Decision model policy using LFM-3 (LiquidAI d1-3B)."""

    def __init__(
        self,
        model_path: str = "tmp/models/d1-3B-Q4_K_M.gguf",
        device: str = "cpu",
        n_ctx: int = 4096,
        n_batch: int = 512,
    ):
        """Initialize LFM-3 policy.

        Args:
            model_path: Path to GGUF model file
            device: "cpu" or "gpu"
            n_ctx: Context size
            n_batch: Batch size for inference
        """
        self.model_path = Path(model_path)
        self.device = device
        self.llama = None

        if LLAMA_AVAILABLE and self.model_path.exists():
            try:
                self.llama = Llama(
                    model=str(self.model_path),
                    n_ctx=n_ctx,
                    n_batch=n_batch,
                    n_threads=4,
                    verbose=False,
                )
                print(f"✓ LFM-3 model loaded: {self.model_path}")
            except Exception as e:
                print(f"✗ Failed to load LFM-3 model: {e}")
                self.llama = None
        else:
            if not LLAMA_AVAILABLE:
                print("✗ llama_cpp not installed")
            if not self.model_path.exists():
                print(f"✗ Model file not found: {self.model_path}")

    def choose(
        self,
        observation: Dict[str, Any],
        supermimic=None
    ) -> str:
        """Choose an action based on observation using LFM-3.

        Args:
            observation: Dict with keys: distance, relative_position,
                        relative_velocity, opponent_motion, boundary_space
            supermimic: Optional SuperMimic instance for motion generation

        Returns:
            Action primitive name (str)
        """
        if self.llama is None:
            # Fallback to rule-based policy
            return self._fallback_choose(observation)

        # Format observation as JSON for the model
        prompt = self._format_prompt(observation)

        # Run inference
        try:
            response = self.llama(
                prompt,
                max_tokens=10,
                temperature=0.7,
                top_p=0.9,
                stop=["</s>", "\n"]
            )

            # Parse response to extract action
            action = self._parse_action(str(response))
            return action if action else self._fallback_choose(observation)

        except Exception as e:
            print(f"LFM-3 inference failed: {e}")
            return self._fallback_choose(observation)

    def _format_prompt(self, observation: Dict[str, Any]) -> str:
        """Format observation as a prompt for LFM-3."""
        obs_json = json.dumps({
            "distance": observation.get("distance", 0),
            "relative_x": observation.get("relative_position", [0, 0])[0],
            "relative_y": observation.get("relative_position", [0, 0])[1],
            "rel_vel_x": observation.get("relative_velocity", [0, 0])[0],
            "rel_vel_y": observation.get("relative_velocity", [0, 0])[1],
            "opponent_moving": bool(observation.get("opponent_motion", False)),
            "boundary_space": observation.get("boundary_space", 1.0),
        })

        prompt = f"""You are a boxing AI. Choose an action based on the observation.
Observation: {obs_json}

Available actions: guard, turn_to_target, step_forward, step_backward, sidestep_left, sidestep_right, jab, cross

Output only the action name:"""

        return prompt

    def _parse_action(self, response: str) -> Optional[str]:
        """Parse the model response to extract the action."""
        actions = [
            "guard", "turn_to_target", "step_forward", "step_backward",
            "sidestep_left", "sidestep_right", "jab", "cross"
        ]

        response_lower = response.lower()
        for action in actions:
            if action in response_lower:
                return action

        return None

    def _fallback_choose(self, observation: Dict[str, Any]) -> str:
        """Fallback rule-based policy."""
        distance = observation.get("distance", 1.0)
        opponent_moving = bool(observation.get("opponent_motion", False))
        boundary_space = observation.get("boundary_space", 1.0)

        # Simple rule-based logic
        if distance > 1.0:
            return "step_forward"
        elif distance < 0.5:
            return "step_backward"
        elif opponent_moving and boundary_space < 0.3:
            return "sidestep_right"
        elif distance < 0.7:
            return "jab"
        else:
            return "guard"


# Alternative: Simple wrapper for integration
def create_lfm3_policy(model_path="tmp/models/d1-3B-Q4_K_M.gguf"):
    """Factory function to create LFM-3 policy."""
    return LFM3Policy(model_path=model_path)
