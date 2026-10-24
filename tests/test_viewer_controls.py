from types import SimpleNamespace

import glfw
import mujoco

from shadowbox.config import load_config
from shadowbox.simulation import Arena
from shadowbox.ui.viewer import Viewer


def camera_viewer():
    viewer = Viewer(Arena())
    viewer.camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(viewer.camera)
    viewer.camera.distance = 5
    return viewer


def test_scroll_zooms_using_current_mujoco_api():
    viewer = camera_viewer()
    before = viewer.camera.distance
    viewer._scroll(None, 0, 1)
    assert viewer.camera.distance < before


def test_drag_orbits_using_current_mujoco_api(monkeypatch):
    viewer = camera_viewer()
    monkeypatch.setattr(glfw, 'get_window_size', lambda _: (1000, 700))
    monkeypatch.setattr(glfw, 'get_mouse_button', lambda _, button: button == glfw.MOUSE_BUTTON_LEFT)
    before = (viewer.camera.azimuth, viewer.camera.elevation)
    viewer._mouse(None, 30, 20)
    assert (viewer.camera.azimuth, viewer.camera.elevation) != before
