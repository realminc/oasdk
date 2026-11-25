#!/usr/bin/env python3
"""Build and stage one oasdk profile."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(*arguments: str) -> None:
	print("$ " + " ".join(arguments), flush=True)
	subprocess.run(arguments, cwd=ROOT, check=True)


def build(profile: str, clean: bool) -> None:
	if clean:
		run("cargo", "clean", "--profile", "dev" if profile == "debug" else "release")
		run(sys.executable, "tool/build/stage.py", "--clean", "--profile", profile)
	command = ["cargo", "build", "--workspace", "--all-targets", "--all-features"]
	if profile == "release":
		command.insert(2, "--release")
	run(*command)
	if not clean:
		run(sys.executable, "tool/build/stage.py", "--clean", "--profile", profile)
	run(sys.executable, "tool/build/stage.py", "--profile", profile)
	run(sys.executable, "tool/build/stage.py", "--profile", profile, "--tests")


def clean_all() -> None:
	run("cargo", "clean")
	run(sys.executable, "tool/build/stage.py", "--clean")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--profile", choices=("debug", "release"))
	parser.add_argument("--clean", action="store_true")
	parser.add_argument("--clean-all", action="store_true", help="remove both build profiles")
	args = parser.parse_args()
	if args.clean_all:
		if args.profile or args.clean:
			parser.error("--clean-all cannot be combined with other options")
		clean_all()
		return
	if args.profile is None:
		parser.error("--profile is required for a build")
	build(args.profile, args.clean)


if __name__ == "__main__":
	main()
