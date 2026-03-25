#!/usr/bin/env python3
"""
detect_context_editing.py — Detect whether context editing actually happened
in smart context experiments.

Context editing is detected by checking if the input_tokens_per_turn array is
NOT monotonically increasing.  If input tokens ever decrease from one turn to
the next, the context was edited (trimmed/compacted).

Run from the repo root:
    python detect_context_editing.py [--experiment-dir {sbp,osworld}]

When --experiment-dir is omitted both sbp/ and osworld/ are scanned.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent


def is_uuid_dir(path: Path) -> bool:
    """Return True if the directory name looks like a UUID."""
    return bool(re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        path.name,
        re.IGNORECASE,
    ))


def get_instances(experiment_dir: Path) -> list[Path]:
    """Return sorted instance directories for the given experiment dir."""
    if not experiment_dir.exists():
        print(
            f"Warning: experiment directory '{experiment_dir}' does not exist.",
            file=sys.stderr,
        )
        return []

    exp_name = experiment_dir.name
    if exp_name == "sbp":
        return sorted(
            p for p in experiment_dir.iterdir()
            if p.is_dir() and p.name.startswith("instance_")
        )
    elif exp_name == "osworld":
        return sorted(p for p in experiment_dir.iterdir() if p.is_dir() and is_uuid_dir(p))
    else:
        # Generic fallback: any subdirectory
        return sorted(p for p in experiment_dir.iterdir() if p.is_dir())


def load_input_tokens(variant_dir: Path) -> list[int] | None:
    """
    Load the input_tokens_per_turn array from variant_dir.

    Tries token_usage.json first; falls back to stats.json.
    Returns None if neither file exists or the required key is missing.
    """
    token_usage_path = variant_dir / "token_usage.json"
    stats_path = variant_dir / "stats.json"

    if token_usage_path.exists():
        with token_usage_path.open() as f:
            data = json.load(f)
        tokens = data.get("input_tokens_per_turn")
        if tokens is not None:
            return tokens

    if stats_path.exists():
        with stats_path.open() as f:
            data = json.load(f)
        llm_calls = data.get("llm_calls")
        if llm_calls is not None:
            return [call["input_tokens"] for call in llm_calls]

    return None


def find_decreasing_turns(tokens: list[int]) -> list[tuple[int, int, int]]:
    """
    Return a list of (turn_index, previous_value, current_value) for every
    position where input tokens decreased.

    turn_index is 1-based (the turn at which the decrease was observed).
    """
    decreases = []
    for i in range(1, len(tokens)):
        if tokens[i] < tokens[i - 1]:
            decreases.append((i + 1, tokens[i - 1], tokens[i]))
    return decreases


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_experiment_dir(experiment_dir: Path) -> list[dict]:
    """
    Scan one experiment directory and return a list of result dicts for every
    smart_context variant where context editing was detected.

    Each dict has keys:
        experiment  – experiment dir name (e.g. "sbp")
        instance    – instance dir name
        variant     – variant dir name (e.g. "smart_context_50000_20000")
        decreases   – list of (turn, prev_tokens, cur_tokens)
    """
    results = []
    instances = get_instances(experiment_dir)

    for instance_dir in instances:
        smart_variants = sorted(
            p for p in instance_dir.iterdir()
            if p.is_dir() and p.name.startswith("smart_context_")
        )
        for variant_dir in smart_variants:
            tokens = load_input_tokens(variant_dir)
            if tokens is None:
                continue
            decreases = find_decreasing_turns(tokens)
            if decreases:
                results.append({
                    "experiment": experiment_dir.name,
                    "instance": instance_dir.name,
                    "variant": variant_dir.name,
                    "decreases": decreases,
                })

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(all_results: list[dict]) -> None:
    """Print a human-readable report grouped by experiment directory."""
    if not all_results:
        print("No context editing detected in any smart context experiment.")
        return

    # Group by experiment
    by_experiment: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        by_experiment[r["experiment"]].append(r)

    for exp_name, results in sorted(by_experiment.items()):
        print(f"\n=== {exp_name}/ ===")
        for r in results:
            print(f"  Instance : {r['instance']}")
            print(f"  Variant  : {r['variant']}")
            for turn, prev, cur in r["decreases"]:
                print(
                    f"    Turn {turn}: {prev} → {cur} tokens "
                    f"(decrease of {prev - cur})"
                )
            print()


def print_summary(all_results: list[dict], total_smart_context: int) -> None:
    """Print a one-line summary count."""
    print(
        f"Found context editing in {len(all_results)} out of "
        f"{total_smart_context} smart context experiments"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detect_context_editing.py",
        description=(
            "Detect context editing in smart context experiments by checking "
            "whether input_tokens_per_turn is not monotonically increasing."
        ),
    )
    parser.add_argument(
        "--experiment-dir",
        dest="experiment_dirs",
        metavar="DIR",
        action="append",
        help=(
            "Experiment directory to scan (e.g. sbp or osworld). "
            "Can be specified multiple times. "
            "Defaults to both sbp and osworld when omitted."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    experiment_dir_names: list[str] = args.experiment_dirs or ["sbp", "osworld"]
    experiment_dirs = [REPO_ROOT / name for name in experiment_dir_names]

    all_results: list[dict] = []
    total_smart_context = 0

    for exp_dir in experiment_dirs:
        instances = get_instances(exp_dir)
        for instance_dir in instances:
            smart_variants = [
                p for p in instance_dir.iterdir()
                if p.is_dir() and p.name.startswith("smart_context_")
            ]
            total_smart_context += len(smart_variants)

        results = analyze_experiment_dir(exp_dir)
        all_results.extend(results)

    print_report(all_results)
    print_summary(all_results, total_smart_context)

    sys.exit(0)


if __name__ == "__main__":
    main()
