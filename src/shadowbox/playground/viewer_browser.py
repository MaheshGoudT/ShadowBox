"""Select an NVIDIA-backed Chrome profile for the optional web viewer."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import webbrowser


def _chrome_binary():
    bundled = Path('/opt/google/chrome/chrome')
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return str(bundled)
    for name in ('google-chrome-stable', 'google-chrome', 'chromium', 'chromium-browser'):
        path = shutil.which(name)
        if path:
            return path
    return None


class _NvidiaChrome(webbrowser.BaseBrowser):
    def __init__(self, executable, profile):
        super().__init__(name='ShadowBox NVIDIA Chrome')
        self.executable = executable
        self.profile = profile

    def open(self, url, new=0, autoraise=True):
        env = os.environ.copy()
        env['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
        egl_vendor = Path('/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
        if egl_vendor.is_file():
            env['__EGL_VENDOR_LIBRARY_FILENAMES'] = str(egl_vendor)
        args = [self.executable, f'--user-data-dir={self.profile}', '--no-first-run',
                '--disable-default-apps', '--enable-webgl', '--ignore-gpu-blocklist', url]
        try:
            subprocess.Popen(args, env=env, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            print(f'Could not launch NVIDIA Chrome: {exc}', file=sys.stderr)
            return False
        return True


def prefer_nvidia_chrome():
    """Register the tested Thor Chrome path; leave other hosts unchanged."""
    if os.environ.get('BROWSER'):
        return False
    egl_vendor = Path('/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
    executable = _chrome_binary()
    if not egl_vendor.is_file() or not executable:
        return False
    cache_root = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    profile = cache_root / 'shadowbox' / 'chrome-nvidia-webgl'
    webbrowser.register('shadowbox-nvidia-chrome', None,
                        _NvidiaChrome(executable, profile), preferred=True)
    return True
