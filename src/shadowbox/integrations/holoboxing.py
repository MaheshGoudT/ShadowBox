"""Run the upstream HoloBoxing workflow in its own environment."""

import argparse
import json
import platform
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "holoboxing.yaml"
STAGES = {
    "assets": ("training_python", "holoboxing"),
    "download": ("training_python", "holoboxing.data"),
    "prepare-prior": ("training_python", "holoboxing.training.prepare_prior"),
    "retarget": ("retarget_python", "holoboxing.data"),
    "prepare": ("training_python", "holoboxing.data"),
    "verify": ("training_python", "holoboxing.data"),
    "prior": ("training_python", "holoboxing.training.train_prior"),
    "supervised": ("training_python", "holoboxing.training.train_sl"),
    "rl": ("training_python", "holoboxing.training.train_rl"),
    "export": ("training_python", "holoboxing.training.export"),
}
DATA_COMMANDS = {"download", "retarget", "prepare", "verify", "assets"}


def _path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_settings(path):
    settings = yaml.safe_load(Path(path).read_text())
    if not isinstance(settings, dict):
        raise ValueError("HoloBoxing config must be a mapping")
    required = {"repository", "revision", "training_python", "retarget_python", "evaluation_image", "assets", "lower", "data", "output", "device"}
    missing = required - settings.keys()
    if missing:
        raise ValueError(f"Missing HoloBoxing settings: {', '.join(sorted(missing))}")
    if any(not isinstance(settings[key], str) or not settings[key].strip() for key in required):
        raise ValueError("HoloBoxing settings must be nonempty strings")
    settings["repository"] = _path(settings["repository"])
    for name in ("training_python", "retarget_python"):
        settings[name] = _path(settings[name])
    for key in ("assets", "lower", "data", "output"):
        settings[key] = _path(settings[key])
    if not isinstance(settings.get("device"), str) or not settings["device"].startswith("cuda:"):
        raise ValueError("device must name a CUDA device, e.g. cuda:0")
    return settings


def revision(repository):
    if not (repository / "holoboxing" / "README.md").is_file():
        return None
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def probe(interpreter):
    code = "import json,sys,torch; print(json.dumps(dict(python=list(sys.version_info[:2]),cuda=torch.cuda.is_available(),torch_cuda=torch.version.cuda)))"
    try:
        result = subprocess.run([str(interpreter), "-c", code], capture_output=True, text=True, timeout=30)
        if result.returncode:
            return {"ready": False, "error": result.stderr.strip()[-2000:]}
        info = json.loads(result.stdout)
        info["ready"] = info["python"] == [3, 11] and info["cuda"] and info["torch_cuda"] == "12.8"
        return info
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        return {"ready": False, "error": str(error)}


def report(settings):
    found = revision(settings["repository"])
    machine = platform.machine().lower()
    return {
        "repository": str(settings["repository"]),
        "revision": found,
        "pinned_revision": settings["revision"],
        "revision_matches": found == settings["revision"],
        "host": f"{platform.system()} {machine}",
        "reference_training_host": platform.system() == "Linux" and machine in {"x86_64", "amd64"},
        "training_python": str(settings["training_python"]),
        "training_python_exists": settings["training_python"].is_file(),
        "retarget_python": str(settings["retarget_python"]),
        "retarget_python_exists": settings["retarget_python"].is_file(),
        "evaluation_image": settings["evaluation_image"],
        "training_runtime": probe(settings["training_python"]),
        "retarget_runtime": probe(settings["retarget_python"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Use HoloBoxing as a separate G1 boxing benchmark")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="show pinned checkout and host readiness")
    run = commands.add_parser("run", help="run an allowlisted upstream stage")
    run.add_argument("stage", choices=STAGES)
    run.add_argument("stage_args", nargs=argparse.REMAINDER)
    run.add_argument("--dry-run", action="store_true", help="place before stage; print command without execution")
    for name, count in (("validate", 1), ("match", 2), ("view", 1)):
        command = commands.add_parser(name)
        command.add_argument("inputs", nargs=count, type=Path)
        command.add_argument("--output", type=Path)
        command.add_argument("--benchmark", action="store_true")
        command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
        if args.command == "doctor":
            state = report(settings)
            state["ready"] = all((state["revision_matches"], state["reference_training_host"], state["training_runtime"]["ready"], state["retarget_runtime"]["ready"]))
            print(json.dumps(state, indent=2))
            return 0 if state["ready"] else 1
        if revision(settings["repository"]) != settings["revision"]:
            parser.error("HoloMotion checkout is missing or does not match the pinned revision")
        env = os.environ.copy()
        env.update({"HOLOBOXING_" + key.upper(): str(settings[key]) for key in ("assets", "lower", "data")})
        env["HOLOBOXING_IMAGE"] = settings["evaluation_image"]
        if args.command == "run":
            python_key, module = STAGES[args.stage]
            extra = args.stage_args
            if extra and extra[0] == "--":
                extra = extra[1:]
            # Expand config paths explicitly, without shell interpolation.
            substitutions = {"{" + key + "}": str(settings[key]) for key in ("assets", "lower", "data", "output", "device")}
            extra = [substitutions.get(value, value) for value in extra]
            if args.stage in DATA_COMMANDS:
                extra = [args.stage, *extra]
            command = [str(settings[python_key]), "-m", module, *extra]
            if not args.dry_run:
                if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
                    parser.error("Reference workflow requires Linux x86-64; use --dry-run on this host")
                if not probe(settings[python_key])["ready"]:
                    parser.error("Selected environment requires Python 3.11 and CUDA 12.8 Torch with CUDA available; run doctor")
        else:
            if args.benchmark and args.command != "validate":
                parser.error("--benchmark is only valid for validate")
            if args.output and args.command == "view":
                parser.error("view does not accept --output")
            command = ["bash", str(settings["repository"] / "holoboxing/run.sh"), "--" + args.command,
                       *[str(_path(str(value))) for value in args.inputs], "--image", settings["evaluation_image"]]
            if args.command != "view":
                output = _path(str(args.output)) if args.output else settings["output"] / ("validation.json" if args.command == "validate" else "match")
                command += ["--output", str(output)]
            if args.benchmark:
                command.append("--benchmark")
        if args.dry_run:
            print(shlex.join(command))
            return 0
        return subprocess.run(command, cwd=settings["repository"], env=env, check=False).returncode
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
