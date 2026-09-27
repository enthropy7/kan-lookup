import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import torch


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cpu():
    try:
        with open("/proc/cpuinfo") as f:
            return next((l.split(":", 1)[1].strip() for l in f if l.startswith("model name")), None)
    except OSError:
        return None


def snapshot():
    dirty = _run(["git", "status", "--porcelain", "--", ".", ":(exclude)results"])
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "host": platform.node(),
        "kernel": platform.release(),
        "cpu": _cpu(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "nixpkgs_rev": os.environ.get("KANTAB_NIXPKGS_REV"),
        "env": {k: os.environ.get(k) for k in ("PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES")},
        "git": {"commit": _run(["git", "rev-parse", "HEAD"]), "dirty_files": None if dirty is None else dirty.splitlines()},
    }


def require_clean(snapshot):
    git = snapshot["git"]
    if git["commit"] is None or git["dirty_files"]:
        sys.exit(f"refusing to run on uncommitted code: {git}")
