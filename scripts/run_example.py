#!/usr/bin/env python3
"""Run a repo example after applying modern CUDA build defaults."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

from common import configure_build_environment, repo_root


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a LeetCUDA example with modern GPU defaults."
    )
    parser.add_argument("script", help="Path to a Python example inside the repo.")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    root = repo_root()
    script_path = (root / args.script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Example not found: {script_path}")

    env_info = configure_build_environment(verbose=True)
    print(
        "Running example: "
        f"{script_path.relative_to(root)} "
        f"on {env_info['device_name']} (capability {env_info['capability']})"
    )

    script_dir = script_path.parent
    # Match "python path/to/script.py" semantics after switching into the
    # example directory so relative CUDA source paths still resolve.
    os.chdir(script_dir)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(script_dir))
    sys.argv = [str(script_path)] + args.script_args
    runpy.run_path(str(script_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
