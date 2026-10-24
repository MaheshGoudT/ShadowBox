"""Swept segment distances used by the virtual-touch referee."""

import numpy as np


def _segment_distance(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
    delta = end - start
    fraction = np.clip(np.dot(point - start, delta) / max(np.dot(delta, delta), 1e-12), 0, 1)
    return float(np.linalg.norm(start + fraction * delta - point))



def _segment_segment_distance(p0, p1, q0, q1) -> float:
    """Exact distance between two finite line segments, including parallel cases."""
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c, d, e = np.dot(u, u), np.dot(u, v), np.dot(v, v), np.dot(u, w), np.dot(v, w)
    distances = [_segment_distance(p0, p1, q0), _segment_distance(p0, p1, q1),
                 _segment_distance(q0, q1, p0), _segment_distance(q0, q1, p1)]
    denominator = a * c - b * b
    if denominator > 1e-12:
        s, t = (b * e - c * d) / denominator, (a * e - b * d) / denominator
        if 0 <= s <= 1 and 0 <= t <= 1:
            distances.append(float(np.linalg.norm(w + s * u - t * v)))
    return min(distances)



def sphere_entry(start, end, center, radius):
    """First normalized time a swept point enters a sphere, or None."""
    direction, offset = end - start, start - center
    c = np.dot(offset, offset) - radius * radius
    if c <= 0:
        return 0.0
    a, b = np.dot(direction, direction), np.dot(offset, direction)
    discriminant = b * b - a * c
    if a < 1e-12 or discriminant < 0:
        return None
    fraction = (-b - np.sqrt(discriminant)) / a
    return float(fraction) if 0 <= fraction <= 1 else None


def capsule_entry(start, end, axis_start, axis_end, radius):
    """First normalized intersection with a finite capsule's cylinder or caps."""
    if _segment_distance(axis_start, axis_end, start) <= radius:
        return 0.0
    candidates = [sphere_entry(start, end, center, radius) for center in (axis_start, axis_end)]
    axis = axis_end - axis_start
    length = np.linalg.norm(axis)
    if length > 1e-12:
        axis = axis / length
        offset, direction = start - axis_start, end - start
        perpendicular = offset - axis * np.dot(offset, axis)
        velocity = direction - axis * np.dot(direction, axis)
        a, b = np.dot(velocity, velocity), np.dot(perpendicular, velocity)
        c = np.dot(perpendicular, perpendicular) - radius * radius
        discriminant = b * b - a * c
        if a > 1e-12 and discriminant >= 0:
            fraction = (-b - np.sqrt(discriminant)) / a
            height = np.dot(offset + fraction * direction, axis)
            if 0 <= fraction <= 1 and 0 <= height <= length:
                candidates.append(float(fraction))
    return min((value for value in candidates if value is not None), default=None)
