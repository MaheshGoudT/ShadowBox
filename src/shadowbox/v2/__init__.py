"""ShadowBox v2: GPU-accelerated VLA/WAM/FLORA integration."""

from .flora import (
    check_cuda,
    load_flora_captioner,
    load_flora_rewarder,
    annotate_video,
)

__all__ = [
    "check_cuda",
    "load_flora_captioner",
    "load_flora_rewarder",
    "annotate_video",
]