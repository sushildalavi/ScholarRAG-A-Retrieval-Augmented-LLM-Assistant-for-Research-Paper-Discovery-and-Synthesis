"""
plot_benchmarks.py — Advanced Version

Generates 6 presentation-friendly plot types from benchmark_metric.csv.

Input CSV columns (required):
    model, recall_at_5, recall_at_10, ndcg_at_10, mrr,
    latency_p50_ms, latency_p95_ms, token_cost

Usage:
    python evaluation/plot_benchmarks.py --csv evaluation/benchmark_metric.csv --outdir plots

Outputs:
    recall.png          (grouped bar for Recall@5/10)
    radar.png           (radar chart for Recall@5/10, nDCG@10, MRR)
    line.png            (line plot across ranking metrics)
    latency.png         (horizontal bar plot for p50/p95 latency)
    heatmap.png         (heatmap for ranking metrics)
    token_cost.png      (horizontal bar for token cost)
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "model",
        "recall_at_5",
        "recall_at_10",
        "ndcg_at_10",
        "mrr",
        "latency_p50_ms",
        "latency_p95_ms",
        "token_cost",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


# ============================================================
# 1. Grouped Bar Plot (Recall@5, Recall@10)
# ============================================================
def grouped_recall(df: pd.DataFrame, out: str):
    melted = df.melt(
        id_vars="model",
        value_vars=["recall_at_5", "recall_at_10"],
        var_name="metric",
        value_name="value",
    )
    melted["metric"] = melted["metric"].map(
        {"recall_at_5": "Recall@5", "recall_at_10": "Recall@10"}
    )

    plt.figure(figsize=(7, 5))
    sns.barplot(data=melted, x="metric", y="value", hue="model")
    plt.title("Recall Comparison")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# ============================================================
# 2. Radar Chart (Recall5, Recall10, nDCG, MRR)
# ============================================================
def radar_chart(df: pd.DataFrame, out: str):
    metrics = ["recall_at_5", "recall_at_10", "ndcg_at_10", "mrr"]
    labels = ["Recall@5", "Recall@10", "nDCG@10", "MRR"]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    for _, row in df.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, label=row["model"])
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Overall Ranking Quality (Radar)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# ============================================================
# 3. Line Plot (Ranking metrics)
# ============================================================
def line_plot(df: pd.DataFrame, out: str):
    melted = df.melt(
        id_vars="model",
        value_vars=["recall_at_5", "recall_at_10", "ndcg_at_10", "mrr"],
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=(7, 5))
    sns.lineplot(data=melted, x="metric", y="value", hue="model", marker="o")
    plt.title("Ranking Metrics Comparison (Line)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# ============================================================
# 4. Horizontal Bar Plot (Latency p50/p95)
# ============================================================
def latency_horizontal(df: pd.DataFrame, out: str):
    melted = df.melt(
        id_vars="model",
        value_vars=["latency_p50_ms", "latency_p95_ms"],
        var_name="metric",
        value_name="value",
    )
    melted["metric"] = melted["metric"].map(
        {"latency_p50_ms": "Latency p50 (ms)", "latency_p95_ms": "Latency p95 (ms)"}
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=melted, y="model", x="value", hue="metric", orient="h")
    plt.title("Latency Comparison (p50/p95)")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# ============================================================
# 5. Heatmap (Model × Ranking Metrics)
# ============================================================
def metric_heatmap(df: pd.DataFrame, out: str):
    hm = df.set_index("model")[["recall_at_5", "recall_at_10", "ndcg_at_10", "mrr"]]
    plt.figure(figsize=(6, 4))
    sns.heatmap(hm, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Metric Heatmap (Models × Metrics)")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# ============================================================
# 6. Horizontal Bar (Token Cost)
# ============================================================
def token_cost_plot(df: pd.DataFrame, out: str):
    plt.figure(figsize=(6, 3))
    sns.barplot(data=df, y="model", x="token_cost", orient="h")
    plt.title("Token Cost per Query")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", default="plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_data(args.csv)

    grouped_recall(df, f"{args.outdir}/recall.png")
    radar_chart(df, f"{args.outdir}/radar.png")
    line_plot(df, f"{args.outdir}/line.png")
    latency_horizontal(df, f"{args.outdir}/latency.png")
    metric_heatmap(df, f"{args.outdir}/heatmap.png")
    token_cost_plot(df, f"{args.outdir}/token_cost.png")


if __name__ == "__main__":
    main()
