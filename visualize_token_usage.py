#!/usr/bin/env python3
"""
Visualize token usage across memory compaction experiment variants.

Run from repo root:
    python visualize_token_usage.py

Outputs are saved to output_graphs/.
"""

import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANTS = ["no_compression", "smart_context_30000_12000", "smart_context_50000_20000"]
VARIANT_LABELS = {
    "no_compression": "No Compression",
    "smart_context_30000_12000": "Smart 30K/12K",
    "smart_context_50000_20000": "Smart 50K/20K",
}
COLORS = {
    "no_compression": "#d62728",          # red
    "smart_context_30000_12000": "#1f77b4",  # blue
    "smart_context_50000_20000": "#2ca02c",  # green
}


def token_formatter(v, _):
    """Format large token counts as K or M for axis tick labels."""
    if v >= 1e6:
        return f"{v/1e6:.1f}M"
    if v >= 1e3:
        return f"{v/1e3:.0f}K"
    return str(int(v))

# Anthropic Claude API pricing (per token)
PRICE_UNCACHED_INPUT = 3.00 / 1_000_000      # $3/MTok
PRICE_CACHE_WRITE = 3.75 / 1_000_000         # $3.75/MTok
PRICE_CACHE_READ = 0.30 / 1_000_000          # $0.30/MTok
PRICE_OUTPUT = 15.00 / 1_000_000             # $15/MTok

OUTPUT_DIR = Path("output_graphs")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_instance_name(instance_dir: str) -> tuple[str, str]:
    """Return (project, short_id) from an instance directory name like
    'instance_ansible__ansible-abc123-v0foo'."""
    name = re.sub(r"^instance_", "", instance_dir)
    # project is everything up to the first '-' after the repo slug
    # repo slug pattern: owner__repo  or  owner  (single-name repos like ansible, navidrome)
    parts = name.split("-")
    repo_slug = parts[0]  # e.g. "ansible__ansible" or "future-architect__vuls"
    project = repo_slug.split("__")[0]  # e.g. "ansible", "future-architect"
    # short id: first 8 chars of the commit hash portion
    hash_part = parts[1] if len(parts) > 1 else "unknown"
    short_id = hash_part[:8]
    return project, f"{project}-{short_id}"


def load_data(sbp_root: Path = Path("sbp")) -> pd.DataFrame:
    """Recursively discover all token_usage.json files and return a DataFrame."""
    records = []
    for path in sbp_root.rglob("token_usage.json"):
        variant = path.parent.name
        instance_dir = path.parent.parent.name
        project, instance_short = parse_instance_name(instance_dir)
        with open(path) as f:
            data = json.load(f)
        records.append({
            "instance_dir": instance_dir,
            "instance": instance_short,
            "project": project,
            "variant": variant,
            "num_turns": data.get("num_turns", 0),
            "total_input": data.get("total_input", 0),
            "total_output": data.get("total_output", 0),
            "total_uncached_input": data.get("total_uncached_input", 0),
            "total_cache_creation_input": data.get("total_cache_creation_input", 0),
            "total_cache_read_input": data.get("total_cache_read_input", 0),
            "input_tokens_per_turn": data.get("input_tokens_per_turn", []),
            "output_tokens_per_turn": data.get("output_tokens_per_turn", []),
        })
    df = pd.DataFrame(records)
    # Keep only known variants
    df = df[df["variant"].isin(VARIANTS)].copy()
    df["variant_label"] = df["variant"].map(VARIANT_LABELS)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


def save_fig(fig, name: str):
    path = OUTPUT_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path.name


def grouped_bar(ax, instances, data_by_variant, ylabel, title):
    """Draw a grouped bar chart on ax."""
    x = np.arange(len(instances))
    n = len(VARIANTS)
    width = 0.25
    for i, variant in enumerate(VARIANTS):
        vals = [data_by_variant[variant].get(inst, 0) for inst in instances]
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=VARIANT_LABELS[variant], color=COLORS[variant], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(token_formatter))


# ---------------------------------------------------------------------------
# Graph 1 – Total Input Tokens by Variant
# ---------------------------------------------------------------------------

def graph1_total_input(df: pd.DataFrame) -> str:
    instances = sorted(df["instance"].unique())
    data_by_variant = {}
    for variant in VARIANTS:
        sub = df[df["variant"] == variant].set_index("instance")["total_input"]
        data_by_variant[variant] = sub.to_dict()

    fig, ax = plt.subplots(figsize=(max(12, len(instances) * 0.4), 6))
    grouped_bar(ax, instances, data_by_variant, "Total Input Tokens", "Graph 1: Total Input Tokens by Variant")
    fig.tight_layout()
    return save_fig(fig, "graph1_total_input.png")


# ---------------------------------------------------------------------------
# Graph 2 – Total Output Tokens by Variant
# ---------------------------------------------------------------------------

def graph2_total_output(df: pd.DataFrame) -> str:
    instances = sorted(df["instance"].unique())
    data_by_variant = {}
    for variant in VARIANTS:
        sub = df[df["variant"] == variant].set_index("instance")["total_output"]
        data_by_variant[variant] = sub.to_dict()

    fig, ax = plt.subplots(figsize=(max(12, len(instances) * 0.4), 6))
    grouped_bar(ax, instances, data_by_variant, "Total Output Tokens", "Graph 2: Total Output Tokens by Variant")
    fig.tight_layout()
    return save_fig(fig, "graph2_total_output.png")


# ---------------------------------------------------------------------------
# Graph 3 – Number of Turns by Variant
# ---------------------------------------------------------------------------

def graph3_num_turns(df: pd.DataFrame) -> str:
    instances = sorted(df["instance"].unique())
    data_by_variant = {}
    for variant in VARIANTS:
        sub = df[df["variant"] == variant].set_index("instance")["num_turns"]
        data_by_variant[variant] = sub.to_dict()

    def plain_formatter(v, _):
        return str(int(v))

    fig, ax = plt.subplots(figsize=(max(12, len(instances) * 0.4), 6))
    grouped_bar(ax, instances, data_by_variant, "Number of Turns", "Graph 3: Number of Turns by Variant")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(plain_formatter))
    fig.tight_layout()
    return save_fig(fig, "graph3_num_turns.png")


# ---------------------------------------------------------------------------
# Graph 4 – Input Tokens Per Turn (line chart, faceted by project)
# ---------------------------------------------------------------------------

def graph4_tokens_per_turn(df: pd.DataFrame) -> str:
    projects = sorted(df["project"].unique())
    ncols = 2
    nrows = (len(projects) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4), squeeze=False)
    fig.suptitle("Graph 4: Input Tokens Per Turn Over Conversation (by Project)", fontsize=13, y=1.01)

    for idx, project in enumerate(projects):
        ax = axes[idx // ncols][idx % ncols]
        proj_df = df[df["project"] == project]
        # pick one representative instance per project (the one with most turns in no_compression)
        nc_df = proj_df[proj_df["variant"] == "no_compression"]
        if nc_df.empty:
            nc_df = proj_df
        rep_instance_dir = (
            nc_df.sort_values("num_turns", ascending=False).iloc[0]["instance_dir"]
        )
        inst_df = proj_df[proj_df["instance_dir"] == rep_instance_dir]
        inst_label = inst_df.iloc[0]["instance"]

        for _, row in inst_df.iterrows():
            turns = row["input_tokens_per_turn"]
            if not turns:
                continue
            ax.plot(
                range(1, len(turns) + 1),
                [t / 1000 for t in turns],
                label=VARIANT_LABELS[row["variant"]],
                color=COLORS[row["variant"]],
                linewidth=1.5,
                marker=".",
                markersize=3,
            )
        ax.set_title(f"{project}\n({inst_label})", fontsize=9)
        ax.set_xlabel("Turn #", fontsize=8)
        ax.set_ylabel("Input Tokens (K)", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    # hide unused subplots
    for idx in range(len(projects), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    return save_fig(fig, "graph4_tokens_per_turn.png")


# ---------------------------------------------------------------------------
# Graph 5 – Aggregate Summary Statistics (Box/Violin)
# ---------------------------------------------------------------------------

def graph5_aggregate_stats(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Graph 5: Aggregate Summary Statistics by Variant", fontsize=13)

    metrics = [
        ("total_input", "Total Input Tokens"),
        ("total_output", "Total Output Tokens"),
        ("num_turns", "Number of Turns"),
    ]

    order = VARIANTS
    palette = {v: COLORS[v] for v in VARIANTS}
    label_map = VARIANT_LABELS

    for ax, (col, title) in zip(axes, metrics):
        plot_df = df[df["variant"].isin(order)].copy()
        plot_df["variant_label"] = plot_df["variant"].map(label_map)
        label_order = [label_map[v] for v in order]
        sns.violinplot(
            data=plot_df,
            x="variant_label",
            y=col,
            hue="variant_label",
            order=label_order,
            hue_order=label_order,
            palette={label_map[v]: COLORS[v] for v in order},
            ax=ax,
            inner="box",
            alpha=0.8,
            legend=False,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel(col.replace("_", " ").title(), fontsize=9)
        ax.tick_params(axis="x", rotation=15, labelsize=8)
        if col in ("total_input",):
            ax.yaxis.set_major_formatter(plt.FuncFormatter(token_formatter))

    fig.tight_layout()
    return save_fig(fig, "graph5_aggregate_stats.png")


# ---------------------------------------------------------------------------
# Graph 6 – Token Savings Percentage
# ---------------------------------------------------------------------------

def graph6_token_savings(df: pd.DataFrame) -> str:
    baseline = df[df["variant"] == "no_compression"].set_index("instance_dir")["total_input"]
    savings_rows = []
    for variant in ["smart_context_30000_12000", "smart_context_50000_20000"]:
        sub = df[df["variant"] == variant].set_index("instance_dir")["total_input"]
        for inst_dir in sub.index:
            if inst_dir in baseline.index and baseline[inst_dir] > 0:
                pct = (1 - sub[inst_dir] / baseline[inst_dir]) * 100
                instance_label = df[df["instance_dir"] == inst_dir]["instance"].iloc[0]
                savings_rows.append({
                    "instance": instance_label,
                    "variant": variant,
                    "savings_pct": pct,
                })

    if not savings_rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No savings data available", ha="center", va="center")
        return save_fig(fig, "graph6_token_savings.png")

    sdf = pd.DataFrame(savings_rows)
    # Sort ascending so the largest savings appear at the top of the horizontal chart
    avg_savings = sdf.groupby("instance")["savings_pct"].mean().sort_values(ascending=True)
    sorted_instances = avg_savings.index.tolist()

    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_instances) * 0.35)))
    y = np.arange(len(sorted_instances))
    height = 0.35

    for i, variant in enumerate(["smart_context_30000_12000", "smart_context_50000_20000"]):
        pivot = sdf[sdf["variant"] == variant].groupby("instance")["savings_pct"].mean()
        vals = [float(pivot.get(inst, 0)) for inst in sorted_instances]
        offset = (i - 0.5) * height
        ax.barh(y + offset, vals, height, label=VARIANT_LABELS[variant], color=COLORS[variant], alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_instances, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Token Savings (%)")
    ax.set_title("Graph 6: Token Savings % vs No Compression (by Instance)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return save_fig(fig, "graph6_token_savings.png")


# ---------------------------------------------------------------------------
# Graph 7 – Cache Efficiency (Stacked Bar)
# ---------------------------------------------------------------------------

def graph7_cache_efficiency(df: pd.DataFrame) -> str:
    agg = df.groupby("variant")[
        ["total_uncached_input", "total_cache_creation_input", "total_cache_read_input"]
    ].sum()

    labels = [VARIANT_LABELS[v] for v in VARIANTS if v in agg.index]
    variants_present = [v for v in VARIANTS if v in agg.index]
    uncached = [agg.loc[v, "total_uncached_input"] for v in variants_present]
    cache_write = [agg.loc[v, "total_cache_creation_input"] for v in variants_present]
    cache_read = [agg.loc[v, "total_cache_read_input"] for v in variants_present]

    totals = [u + w + r for u, w, r in zip(uncached, cache_write, cache_read)]
    uncached_pct = [u / t * 100 if t else 0 for u, t in zip(uncached, totals)]
    write_pct = [w / t * 100 if t else 0 for w, t in zip(cache_write, totals)]
    read_pct = [r / t * 100 if t else 0 for r, t in zip(cache_read, totals)]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x, uncached_pct, label="Uncached Input", color="#e377c2")
    b2 = ax.bar(x, write_pct, bottom=uncached_pct, label="Cache Write Input", color="#ff7f0e")
    b3 = ax.bar(x, read_pct, bottom=[u + w for u, w in zip(uncached_pct, write_pct)],
                label="Cache Read Input", color="#17becf")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Graph 7: Cache Efficiency by Variant (Proportional Breakdown)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)

    # annotate totals
    for i, t in enumerate(totals):
        ax.text(i, 102, f"{t/1e6:.1f}M", ha="center", fontsize=8)

    fig.tight_layout()
    return save_fig(fig, "graph7_cache_efficiency.png")


# ---------------------------------------------------------------------------
# Graph 8 – Cost Estimation
# ---------------------------------------------------------------------------

def graph8_cost_estimation(df: pd.DataFrame) -> str:
    def compute_cost(row):
        return (
            row["total_uncached_input"] * PRICE_UNCACHED_INPUT
            + row["total_cache_creation_input"] * PRICE_CACHE_WRITE
            + row["total_cache_read_input"] * PRICE_CACHE_READ
            + row["total_output"] * PRICE_OUTPUT
        )

    df = df.copy()
    df["estimated_cost"] = df.apply(compute_cost, axis=1)
    instances = sorted(df["instance"].unique())

    data_by_variant = {}
    for variant in VARIANTS:
        sub = df[df["variant"] == variant].set_index("instance")["estimated_cost"]
        data_by_variant[variant] = sub.to_dict()

    x = np.arange(len(instances))
    n = len(VARIANTS)
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(12, len(instances) * 0.4), 6))

    for i, variant in enumerate(VARIANTS):
        vals = [data_by_variant[variant].get(inst, 0) for inst in instances]
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=VARIANT_LABELS[variant], color=COLORS[variant], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Estimated Cost (USD)")
    ax.set_title("Graph 8: Estimated API Cost by Variant\n(Uncached $3/MTok, Cache Write $3.75/MTok, Cache Read $0.30/MTok, Output $15/MTok)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.2f}"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_fig(fig, "graph8_cost_estimation.png")


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def build_summary_table(df: pd.DataFrame) -> str:
    agg = df.groupby("variant").agg(
        instances=("instance", "nunique"),
        avg_turns=("num_turns", "mean"),
        avg_total_input=("total_input", "mean"),
        avg_total_output=("total_output", "mean"),
        total_input_sum=("total_input", "sum"),
        total_output_sum=("total_output", "sum"),
    ).reset_index()

    def compute_cost_row(row):
        sub = df[df["variant"] == row["variant"]]
        cost = (
            sub["total_uncached_input"].sum() * PRICE_UNCACHED_INPUT
            + sub["total_cache_creation_input"].sum() * PRICE_CACHE_WRITE
            + sub["total_cache_read_input"].sum() * PRICE_CACHE_READ
            + sub["total_output"].sum() * PRICE_OUTPUT
        )
        return cost

    agg["total_cost"] = agg.apply(compute_cost_row, axis=1)
    agg["variant_label"] = agg["variant"].map(VARIANT_LABELS)

    rows = ""
    for _, r in agg.iterrows():
        rows += (
            f"<tr>"
            f"<td>{r['variant_label']}</td>"
            f"<td>{int(r['instances'])}</td>"
            f"<td>{r['avg_turns']:.1f}</td>"
            f"<td>{r['avg_total_input']/1e3:.1f}K</td>"
            f"<td>{r['avg_total_output']/1e3:.1f}K</td>"
            f"<td>{r['total_input_sum']/1e6:.1f}M</td>"
            f"<td>${r['total_cost']:.2f}</td>"
            f"</tr>\n"
        )
    return f"""
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;font-family:monospace">
  <thead>
    <tr style="background:#eee">
      <th>Variant</th><th>Instances</th><th>Avg Turns</th>
      <th>Avg Input</th><th>Avg Output</th><th>Total Input</th><th>Total Cost</th>
    </tr>
  </thead>
  <tbody>
{rows}  </tbody>
</table>"""


def generate_html_report(df: pd.DataFrame, graph_files: list[str]):
    table_html = build_summary_table(df)
    imgs = "\n".join(
        f'<div style="margin:20px 0"><img src="{f}" style="max-width:100%;border:1px solid #ccc"></div>'
        for f in graph_files
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Memory Compaction – Token Usage Report</title>
  <style>
    body {{ font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
    h1 {{ color: #333; }}
    h2 {{ color: #555; margin-top: 40px; }}
  </style>
</head>
<body>
  <h1>Memory Compaction – Token Usage Report</h1>
  <p>Generated from <code>sbp/</code> experiment data.
     Variants: No Compression (red), Smart 30K/12K (blue), Smart 50K/20K (green).</p>

  <h2>Summary Statistics</h2>
  {table_html}

  <h2>Graphs</h2>
  {imgs}
</body>
</html>"""
    report_path = OUTPUT_DIR / "report.html"
    with open(report_path, "w") as f:
        f.write(html)
    print(f"HTML report saved: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_output_dir()
    print("Loading data from sbp/ ...")
    df = load_data()
    print(f"  Loaded {len(df)} records from {df['instance_dir'].nunique()} instances.")

    graphs = []
    steps = [
        ("Graph 1: Total Input Tokens", graph1_total_input),
        ("Graph 2: Total Output Tokens", graph2_total_output),
        ("Graph 3: Number of Turns", graph3_num_turns),
        ("Graph 4: Tokens Per Turn (line)", graph4_tokens_per_turn),
        ("Graph 5: Aggregate Stats (violin)", graph5_aggregate_stats),
        ("Graph 6: Token Savings %", graph6_token_savings),
        ("Graph 7: Cache Efficiency", graph7_cache_efficiency),
        ("Graph 8: Cost Estimation", graph8_cost_estimation),
    ]

    for label, fn in steps:
        print(f"  Generating {label} ...")
        fname = fn(df)
        graphs.append(fname)

    print("Generating HTML report ...")
    generate_html_report(df, graphs)
    print(f"\nDone! All outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
