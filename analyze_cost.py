#!/usr/bin/env python3
"""
analyze_cost.py — Compare monetary cost and resolve rate across experiment
configurations (no_compression vs smart_context), filtered to only those
instances where context editing actually occurred.

Also analyzes cost/resolve rate in relation to the effective compression ratio.

Run from the repo root:
    python analyze_cost.py [--output-dir plots]

Output:
    plots/sbp_cost_comparison.png
    plots/osworld_cost_comparison.png
    plots/sbp_resolve_rate.png
    plots/osworld_resolve_rate.png
    plots/cost_vs_compression_ratio.png
    plots/resolve_rate_vs_compression_ratio.png
    plots/aggregate_summary.png
    plots/sbp_domain_test_time_compute.png
"""

import argparse
import glob as glob_module
import json
import re
import sys
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Reuse helpers from the existing detect_context_editing script.
from detect_context_editing import (
    analyze_experiment_dir,
    get_instances,
    load_input_tokens,
    find_decreasing_turns,
    REPO_ROOT,
)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_cost(variant_dir: Path) -> float | None:
    """
    Return total_cost_usd for a variant directory.

    Tries token_usage.json first (field: total_cost_usd), then stats.json
    (fields: total_cost_usd, total_cost, cost).  Returns None when no cost
    information can be found.
    """
    token_usage_path = variant_dir / "token_usage.json"
    stats_path = variant_dir / "stats.json"

    if token_usage_path.exists():
        with token_usage_path.open() as f:
            data = json.load(f)
        cost = data.get("total_cost_usd")
        if cost is not None:
            return float(cost)

    if stats_path.exists():
        with stats_path.open() as f:
            data = json.load(f)
        for field in ("total_cost_usd", "total_cost", "cost"):
            cost = data.get(field)
            if cost is not None:
                return float(cost)

    return None


def load_sbp_resolve_rates(sbp_dir: Path) -> dict[str, dict[str, bool]]:
    """
    Return a mapping  {config_name: {instance_name: resolved}}.

    Discovers all eval_scores_*.json files under sbp_dir.
    Config name is derived by stripping the ``eval_scores_`` prefix and
    ``.json`` suffix, e.g. ``eval_scores_no_compression.json`` →
    ``no_compression``.
    """
    pattern = str(sbp_dir / "eval_scores_*.json")
    result: dict[str, dict[str, bool]] = {}
    for path_str in glob_module.glob(pattern):
        path = Path(path_str)
        config = path.stem.removeprefix("eval_scores_")
        with path.open() as f:
            data = json.load(f)
        result[config] = {k: bool(v) for k, v in data.get("instance_results", {}).items()}
    return result


def load_osworld_resolve_rate(variant_dir: Path) -> float | None:
    """
    Read resolve rate from result.txt in variant_dir.
    The file contains a single float in [0, 1].
    """
    result_path = variant_dir / "result.txt"
    if not result_path.exists():
        return None
    try:
        return float(result_path.read_text().strip())
    except (ValueError, OSError):
        return None


# ---------------------------------------------------------------------------
# Effective compression ratio
# ---------------------------------------------------------------------------

def effective_compression_ratio(decreases: list[tuple[int, int, int]]) -> float:
    """
    Given the list of (turn, prev_tokens, cur_tokens) compression events,
    return the average per-event ratio prev/cur.
    """
    ratios = [prev / cur for _, prev, cur in decreases if cur > 0]
    if not ratios:
        return 1.0
    return mean(ratios)


# ---------------------------------------------------------------------------
# Core data collection
# ---------------------------------------------------------------------------

def collect_data(repo_root: Path) -> dict:
    """
    Run context-editing detection and collect cost + resolve-rate data for
    every instance/variant where context editing was detected.

    Returns a dict with keys "sbp" and "osworld", each being a list of record
    dicts:
        instance        – instance dir name
        variant         – smart_context variant name
        no_comp_cost    – cost for no_compression variant (float or None)
        sc_cost         – cost for smart_context variant (float or None)
        no_comp_resolve – resolve rate for no_compression variant (float or None)
        sc_resolve      – resolve rate for smart_context variant (float or None)
        compression_ratio – effective compression ratio (float)
        decreases       – raw decreases list
    """
    sbp_dir = repo_root / "sbp"
    osworld_dir = repo_root / "osworld"

    sbp_eval = load_sbp_resolve_rates(sbp_dir)

    results: dict[str, list[dict]] = {"sbp": [], "osworld": []}

    for exp_name, exp_dir in [("sbp", sbp_dir), ("osworld", osworld_dir)]:
        editing_results = analyze_experiment_dir(exp_dir)

        for r in editing_results:
            instance_name = r["instance"]
            variant_name = r["variant"]
            decreases = r["decreases"]

            instance_dir = exp_dir / instance_name
            sc_dir = instance_dir / variant_name
            nc_dir = instance_dir / "no_compression"

            sc_cost = load_cost(sc_dir)
            nc_cost = load_cost(nc_dir)

            if exp_name == "sbp":
                # Determine config key used in eval scores (variant name maps
                # directly, e.g. "smart_context_50000_20000").
                sc_config = variant_name
                nc_config = "no_compression"
                sc_resolve: float | None = None
                nc_resolve: float | None = None
                if sc_config in sbp_eval:
                    v = sbp_eval[sc_config].get(instance_name)
                    sc_resolve = float(v) if v is not None else None
                if nc_config in sbp_eval:
                    v = sbp_eval[nc_config].get(instance_name)
                    nc_resolve = float(v) if v is not None else None
            else:  # osworld
                sc_resolve = load_osworld_resolve_rate(sc_dir)
                nc_resolve = load_osworld_resolve_rate(nc_dir)

            ratio = effective_compression_ratio(decreases)

            results[exp_name].append({
                "instance": instance_name,
                "variant": variant_name,
                "no_comp_cost": nc_cost,
                "sc_cost": sc_cost,
                "no_comp_resolve": nc_resolve,
                "sc_resolve": sc_resolve,
                "compression_ratio": ratio,
                "decreases": decreases,
            })

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _short_name(instance: str, max_len: int = 24) -> str:
    """Shorten an instance name for axis labels."""
    if len(instance) <= max_len:
        return instance
    return instance[:max_len] + "…"


def _extract_domain(instance_name: str) -> str:
    """
    Extract the ``owner/repo`` domain from an SBP instance name.

    Instance names follow the pattern
    ``instance_<owner>__<repo>-<40-char-hash>-v<version>``.
    Returns ``<owner>/<repo>``, e.g. ``element-hq/element-web``.
    """
    # Strip leading "instance_" prefix if present
    name = instance_name
    if name.startswith("instance_"):
        name = name[len("instance_"):]
    # Split on double-underscore to separate owner from the rest
    if "__" not in name:
        return name
    owner, rest = name.split("__", 1)
    # Find the 40-hex-char hash segment preceded by a hyphen to locate where
    # the repo name ends.  Everything before "-<hash>" is the repo name.
    m = re.search(r"-[0-9a-f]{40}(?:-|$)", rest)
    if m:
        repo = rest[: m.start()]
    else:
        # Fallback: take the part before the first "-"
        repo = rest.split("-")[0]
    return f"{owner}/{repo}"


def _bar_positions(n: int, width: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(n)
    return x - width / 2, x + width / 2


def plot_cost_comparison(records: list[dict], title: str, output_path: Path) -> None:
    """Grouped bar chart: no_compression cost vs smart_context cost per instance."""
    valid = [r for r in records if r["no_comp_cost"] is not None or r["sc_cost"] is not None]
    if not valid:
        print(f"  No cost data available for {title}, skipping.")
        return

    instances = [_short_name(r["instance"]) for r in valid]
    nc_costs = [r["no_comp_cost"] if r["no_comp_cost"] is not None else 0 for r in valid]
    sc_costs = [r["sc_cost"] if r["sc_cost"] is not None else 0 for r in valid]

    n = len(instances)
    pos_nc, pos_sc = _bar_positions(n)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.7), 6))
    ax.bar(pos_nc, nc_costs, width=0.35, label="no_compression", color="steelblue", alpha=0.85)
    ax.bar(pos_sc, sc_costs, width=0.35, label="smart_context", color="darkorange", alpha=0.85)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(instances, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cost (USD)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_resolve_rate(records: list[dict], title: str, output_path: Path) -> None:
    """Grouped bar chart: no_compression vs smart_context resolve rate per instance."""
    valid = [
        r for r in records
        if r["no_comp_resolve"] is not None or r["sc_resolve"] is not None
    ]
    if not valid:
        print(f"  No resolve rate data available for {title}, skipping.")
        return

    instances = [_short_name(r["instance"]) for r in valid]
    nc_rr = [r["no_comp_resolve"] if r["no_comp_resolve"] is not None else 0 for r in valid]
    sc_rr = [r["sc_resolve"] if r["sc_resolve"] is not None else 0 for r in valid]

    n = len(instances)
    pos_nc, pos_sc = _bar_positions(n)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.7), 6))
    ax.bar(pos_nc, nc_rr, width=0.35, label="no_compression", color="steelblue", alpha=0.85)
    ax.bar(pos_sc, sc_rr, width=0.35, label="smart_context", color="darkorange", alpha=0.85)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(instances, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Resolve rate")
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_scatter(
    all_records: dict[str, list[dict]],
    y_key: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    """
    Scatter plot of a metric (cost or resolve rate) vs effective compression
    ratio for all instances across both benchmarks.

    Each point is coloured by benchmark (SBP vs OSWorld) and shaped by config.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {"sbp": "steelblue", "osworld": "darkorange"}
    default_marker = "D"
    markers = {"no_compression": "o", "smart_context": "s"}

    plotted_labels: set[str] = set()

    for exp_name, records in all_records.items():
        for r in records:
            ratio = r["compression_ratio"]
            # Both no_compression and smart_context values
            for config_key, val in [
                ("no_compression", r.get(f"no_comp_{y_key}")),
                ("smart_context", r.get(f"sc_{y_key}")),
            ]:
                if val is None:
                    continue
                color = colors.get(exp_name, "gray")
                marker = markers.get(config_key, default_marker)
                label = f"{exp_name} / {config_key}"
                handle = ax.scatter(
                    ratio,
                    val,
                    c=color,
                    marker=marker,
                    alpha=0.75,
                    s=60,
                    label=label if label not in plotted_labels else "_nolegend_",
                )
                plotted_labels.add(label)

    ax.set_xlabel("Effective compression ratio (prev / cur tokens, avg over events)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_aggregate_summary(
    all_records: dict[str, list[dict]],
    output_path: Path,
) -> None:
    """
    Summary bar chart with aggregate (average) cost and resolve rate
    by benchmark and config.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel, title_suffix in [
        (axes[0], "cost", "Avg cost (USD)", "Average Cost"),
        (axes[1], "resolve", "Avg resolve rate", "Average Resolve Rate"),
    ]:
        groups: dict[str, list[float]] = {}
        for exp_name, records in all_records.items():
            for config_key in ("no_compression", "smart_context"):
                if config_key == "no_compression":
                    vals = [r.get(f"no_comp_{metric}") for r in records if r.get(f"no_comp_{metric}") is not None]
                else:
                    vals = [r.get(f"sc_{metric}") for r in records if r.get(f"sc_{metric}") is not None]
                group_label = f"{exp_name}\n{config_key}"
                if vals:
                    groups[group_label] = vals

        labels = list(groups.keys())
        avgs = [mean(v) for v in groups.values()]

        x = np.arange(len(labels))
        bar_colors = []
        for lbl in labels:
            if "sbp" in lbl and "no_compression" in lbl:
                bar_colors.append("steelblue")
            elif "sbp" in lbl and "smart_context" in lbl:
                bar_colors.append("royalblue")
            elif "osworld" in lbl and "no_compression" in lbl:
                bar_colors.append("darkorange")
            else:
                bar_colors.append("orange")

        ax.bar(x, avgs, color=bar_colors, alpha=0.85, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title_suffix)
        if metric == "resolve":
            ax.set_ylim(0, 1.1)

    fig.suptitle("Aggregate Summary (instances with context editing only)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_domain_test_time_compute(records: list[dict], output_path: Path) -> None:
    """
    Scatter plot of cost (X) vs resolve rate (Y) for SBP records with context
    editing, grouped by domain.

    For each domain two points are drawn (no_compression and smart_context) and
    connected by an arrow showing the direction of change.
    """
    # Group records by domain
    domain_records: dict[str, list[dict]] = {}
    for r in records:
        domain = _extract_domain(r["instance"])
        domain_records.setdefault(domain, []).append(r)

    if not domain_records:
        print(f"  No SBP domain data available, skipping {output_path}.")
        return

    domains = sorted(domain_records.keys())
    n_domains = len(domains)
    cmap = plt.get_cmap("tab20")
    colors = {d: cmap(i / n_domains if n_domains > 1 else 0.5) for i, d in enumerate(domains)}

    fig, ax = plt.subplots(figsize=(10, 7))

    for domain in domains:
        recs = domain_records[domain]
        color = colors[domain]

        nc_costs = [r["no_comp_cost"] for r in recs if r["no_comp_cost"] is not None]
        nc_resolves = [r["no_comp_resolve"] for r in recs if r["no_comp_resolve"] is not None]
        sc_costs = [r["sc_cost"] for r in recs if r["sc_cost"] is not None]
        sc_resolves = [r["sc_resolve"] for r in recs if r["sc_resolve"] is not None]

        if not nc_costs or not nc_resolves or not sc_costs or not sc_resolves:
            continue

        nc_x = mean(nc_costs)
        nc_y = mean(nc_resolves)
        sc_x = mean(sc_costs)
        sc_y = mean(sc_resolves)

        ax.scatter(nc_x, nc_y, color=color, marker="o", s=80, zorder=3)
        ax.scatter(sc_x, sc_y, color=color, marker="s", s=80, zorder=3)

        ax.annotate(
            "",
            xy=(sc_x, sc_y),
            xytext=(nc_x, nc_y),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )

    # Build legend: one colored patch per domain, plus marker-shape entries
    domain_handles = [mpatches.Patch(color=colors[d], label=d) for d in domains]
    shape_handles = [
        ax.scatter([], [], color="gray", marker="o", s=80, label="no_compression"),
        ax.scatter([], [], color="gray", marker="s", s=80, label="smart_context"),
    ]
    ax.legend(handles=domain_handles + shape_handles, loc="best", fontsize=7, ncol=2)

    ax.set_xlabel("Cost (USD)")
    ax.set_ylabel("Resolve rate")
    ax.set_title(
        "SBP — Test-time Compute: no_compression → smart_context by Domain"
        " (context editing instances)"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(all_records: dict[str, list[dict]]) -> None:
    """Print a text summary table to stdout."""
    print("\n" + "=" * 72)
    print("SUMMARY TABLE (instances where context editing occurred)")
    print("=" * 72)

    for exp_name, records in all_records.items():
        print(f"\n--- {exp_name.upper()} ---")
        print(f"  Instances with context editing detected: {len(records)}")

        nc_costs = [r["no_comp_cost"] for r in records if r["no_comp_cost"] is not None]
        sc_costs = [r["sc_cost"] for r in records if r["sc_cost"] is not None]
        nc_resolve = [r["no_comp_resolve"] for r in records if r["no_comp_resolve"] is not None]
        sc_resolve = [r["sc_resolve"] for r in records if r["sc_resolve"] is not None]
        ratios = [r["compression_ratio"] for r in records]

        def _fmt(vals: list[float]) -> str:
            if not vals:
                return "N/A"
            return f"{mean(vals):.4f} (n={len(vals)})"

        print(f"  Avg cost  no_compression : {_fmt(nc_costs)}")
        print(f"  Avg cost  smart_context  : {_fmt(sc_costs)}")
        print(f"  Resolve   no_compression : {_fmt(nc_resolve)}")
        print(f"  Resolve   smart_context  : {_fmt(sc_resolve)}")
        print(f"  Avg effective compression ratio: {_fmt(ratios)}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_cost.py",
        description=(
            "Analyze monetary cost and resolve rate across experiment "
            "configurations, filtered to instances with context editing."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        metavar="DIR",
        help="Directory where plot PNGs are saved (default: plots/).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting data …")
    all_records = collect_data(REPO_ROOT)

    sbp_records = all_records["sbp"]
    osworld_records = all_records["osworld"]

    print(f"  SBP instances with context editing   : {len(sbp_records)}")
    print(f"  OSWorld instances with context editing: {len(osworld_records)}")

    print("\nGenerating plots …")

    plot_cost_comparison(
        sbp_records,
        "SBP — Per-instance cost comparison (context editing instances only)",
        output_dir / "sbp_cost_comparison.png",
    )
    plot_cost_comparison(
        osworld_records,
        "OSWorld — Per-instance cost comparison (context editing instances only)",
        output_dir / "osworld_cost_comparison.png",
    )
    plot_resolve_rate(
        sbp_records,
        "SBP — Per-instance resolve rate comparison (context editing instances only)",
        output_dir / "sbp_resolve_rate.png",
    )
    plot_resolve_rate(
        osworld_records,
        "OSWorld — Per-instance resolve rate comparison (context editing instances only)",
        output_dir / "osworld_resolve_rate.png",
    )
    plot_scatter(
        all_records,
        y_key="cost",
        y_label="Cost (USD)",
        title="Cost vs Effective Compression Ratio",
        output_path=output_dir / "cost_vs_compression_ratio.png",
    )
    plot_scatter(
        all_records,
        y_key="resolve",
        y_label="Resolve rate",
        title="Resolve Rate vs Effective Compression Ratio",
        output_path=output_dir / "resolve_rate_vs_compression_ratio.png",
    )
    plot_aggregate_summary(all_records, output_dir / "aggregate_summary.png")
    plot_domain_test_time_compute(sbp_records, output_dir / "sbp_domain_test_time_compute.png")

    print_summary_table(all_records)
    sys.exit(0)


if __name__ == "__main__":
    main()
