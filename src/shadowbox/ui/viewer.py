"""A small synchronous MuJoCo viewer with keyboard and camera controls."""

from pathlib import Path
import struct
import zlib

import glfw
import mujoco
import numpy as np


class Viewer:
    def __init__(self, arena, title: str = "ShadowBox"):
        self.arena = arena
        self.paused = False
        self.stopped = False
        self.reset_requested = False
        self.window = None
        self.context = None
        self.lines = []
        self.cursor = (0.0, 0.0)
        self.title = title

    def __enter__(self):
        if not glfw.init():
            raise RuntimeError("Cannot start GLFW. Try --software in a desktop session.")
        glfw.window_hint(glfw.SAMPLES, self.arena.config["viewer"]["samples"])
        self.window = glfw.create_window(self.arena.config["viewer"]["width"], self.arena.config["viewer"]["height"], self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Cannot open the 3D window. Try --software.")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.scene = mujoco.MjvScene(self.arena.model, maxgeom=3000)
        self.context = mujoco.MjrContext(self.arena.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:] = (0, 0, 0.8)
        self.camera.distance = self.arena.config["viewer"]["camera_distance"]
        self.camera.azimuth = self.arena.config["viewer"]["camera_azimuth"]
        self.camera.elevation = self.arena.config["viewer"]["camera_elevation"]
        self.options = mujoco.MjvOption()
        self.options.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = self.arena.config['viewer']['convex_hulls']
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.arena.config['viewer']['shadows']
        self.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        glfw.set_key_callback(self.window, self._key)
        glfw.set_cursor_pos_callback(self.window, self._mouse)
        glfw.set_scroll_callback(self.window, self._scroll)
        return self

    @property
    def running(self) -> bool:
        return self.window is not None and not glfw.window_should_close(self.window)

    def _key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            self.paused = not self.paused
        elif key == glfw.KEY_E:
            self.stopped = True
        elif key == glfw.KEY_R:
            self.reset_requested = True
            self.stopped = self.paused = False

    def _mouse(self, window, x, y):
        old_x, old_y = self.cursor
        self.cursor = (x, y)
        width, height = glfw.get_window_size(window)
        if height < 1:
            return
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT):
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT):
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            return
        mujoco.mjv_moveCamera(self.arena.model, action, (x - old_x) / height,
                             (y - old_y) / height, self.camera)

    def _scroll(self, window, x_offset, y_offset):
        mujoco.mjv_moveCamera(self.arena.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                             0, 0.05 * y_offset, self.camera)

    def _draw(self):
        width, height = glfw.get_framebuffer_size(self.window)
        if width < 1 or height < 1:
            return None
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(self.arena.model, self.arena.data, self.options,
                              None, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)
        status = "STOPPED - press R to reset" if self.stopped else "PAUSED" if self.paused else ""
        text = "\n".join([*self.lines, status])
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           viewport, text, "", self.context)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                           viewport, "Space: pause | R: reset | E: stop | Esc: close\n"
                           "Left drag: orbit | Right drag: pan | Scroll: zoom", "", self.context)
        return width, height

    def render(self, lines: list[str]):
        self.lines = lines
        self._draw()
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def capture(self, path):
        """Write the actual rendered framebuffer as a PNG, without extra packages."""
        size = self._draw()
        if size is None:
            raise RuntimeError("Cannot capture a minimized window.")
        width, height = size
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, width, height), self.context)
        rows = b"".join(b"\0" + row.tobytes() for row in rgb[::-1])

        def chunk(kind, data):
            return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(png)

    def __exit__(self, *args):
        if self.context is not None:
            self.context.free()
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()
