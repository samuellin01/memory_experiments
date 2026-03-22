#!/usr/bin/env python3
"""
repo_manager.py — CLI tool for managing the memory_experiments repository.

Run from the repo root:
    python repo_manager.py <command> [options]

Commands:
    list-instances   List all experiment instance directories in sbp/.
    file-counts      Show file counts per variant for each instance.
"""

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_instances(experiment_dir: Path) -> list[Path]:
    """Return sorted list of instance directories under experiment_dir."""
    if not experiment_dir.exists():
        print(f"Error: experiment directory '{experiment_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)
    instances = sorted(
        p for p in experiment_dir.iterdir()
        if p.is_dir() and p.name.startswith("instance_")
    )
    return instances


def get_variants(instance_dir: Path) -> list[Path]:
    """Return sorted list of variant subdirectories within an instance directory."""
    return sorted(p for p in instance_dir.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list_instances(args: argparse.Namespace) -> None:
    """List all experiment instance directories."""
    experiment_dir = Path(args.experiment_dir)
    instances = get_instances(experiment_dir)
    if not instances:
        print(f"No instances found in '{experiment_dir}'.")
        return
    for instance in instances:
        print(instance.name)


def cmd_file_counts(args: argparse.Namespace) -> None:
    """Show file counts per variant for each instance."""
    experiment_dir = Path(args.experiment_dir)
    instances = get_instances(experiment_dir)
    if not instances:
        print(f"No instances found in '{experiment_dir}'.")
        return
    for instance in instances:
        print(instance.name)
        variants = get_variants(instance)
        if not variants:
            print("  (no variant subdirectories found)")
        for variant in variants:
            file_count = sum(1 for f in variant.iterdir() if f.is_file())
            print(f"  {variant.name}: {file_count} files")


# ---------------------------------------------------------------------------
# CLI setup
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repo_manager.py",
        description="CLI tool for managing the memory_experiments repository.",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # -- list-instances --
    p_list = subparsers.add_parser(
        "list-instances",
        help="List all experiment instance directories.",
        description="List all experiment instance directories in the experiment directory.",
    )
    p_list.add_argument(
        "--experiment-dir",
        default="sbp",
        metavar="DIR",
        help="Root directory containing instance subdirectories (default: sbp).",
    )
    p_list.set_defaults(func=cmd_list_instances)

    # -- file-counts --
    p_counts = subparsers.add_parser(
        "file-counts",
        help="Show file counts per variant for each instance.",
        description=(
            "For each instance directory, list its variant subdirectories and "
            "the number of files each one contains."
        ),
    )
    p_counts.add_argument(
        "--experiment-dir",
        default="sbp",
        metavar="DIR",
        help="Root directory containing instance subdirectories (default: sbp).",
    )
    p_counts.set_defaults(func=cmd_file_counts)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
