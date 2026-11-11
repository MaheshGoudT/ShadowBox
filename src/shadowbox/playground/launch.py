"""Isolated entry point: register ShadowBox then call upstream RL scripts."""
import argparse
from pathlib import Path
import runpy
import sys

# Run by absolute path in a separate environment, without installing core deps.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('train', 'play'))
    args, remaining = parser.parse_known_args()
    from shadowbox.playground.registry import register_tasks
    register_tasks()
    if args.stage == 'play':
        from shadowbox.playground.viewer_browser import prefer_nvidia_chrome
        prefer_nvidia_chrome()
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_module(f'mujoco_playground.rl.rsl_rl.{args.stage}', run_name='__main__')


if __name__ == '__main__':
    main()
