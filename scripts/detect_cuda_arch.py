#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import configure_build_environment, gencode_flags_for_capability


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gencode", action="store_true")
    parser.add_argument("--max-jobs", action="store_true")
    parser.add_argument("--fallback", type=str, default="89")
    args = parser.parse_args()

    info = configure_build_environment(verbose=False)
    capability = info["capability"]
    if args.max_jobs:
        print(info["max_jobs"])
        return 0

    if not capability:
        fallback = args.fallback.replace("sm_", "").replace(".", "")
        capability = f"{fallback[0]}.{fallback[1:]}" if len(fallback) >= 2 else "8.9"

    if args.gencode:
        print(gencode_flags_for_capability(str(capability)))
    else:
        print(capability)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
