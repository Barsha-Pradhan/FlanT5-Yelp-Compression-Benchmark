"""
Generate comparison plots from the combined compression benchmark results.

Usage:
    python generate_plots.py

Reads results/combined_results.csv (built from the per-method CSVs in this
folder) and writes PNG charts to results/plots/.
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODEL_ORDER = ["Flan-T5-Small", "Flan-T5-Base", "Flan-T5-Large", "Flan-T5-XL"]
METHOD_COLORS = {
    "GAMC": "#4C72B0",
    "HAWQ": "#DD8452",
    "Magnitude": "#55A868",
    "Movement": "#C44E52",
    "SparseGPT": "#8172B2",
    "ZeroQuant": "#937860",
}


def load_combined():
    combined_path = os.path.join(RESULTS_DIR, "combined_results.csv")
    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path)
    else:
        # Rebuild from the individual per-method CSVs if combined file is missing.
        csvs = [f for f in glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
                if not f.endswith("combined_results.csv")]
        df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    present_order = [m for m in MODEL_ORDER if m in df["Model Name"].unique()]
    df["Model Name"] = pd.Categorical(df["Model Name"], categories=present_order, ordered=True)
    return df.sort_values(["Method", "Model Name"])


def plot_accuracy_by_model(df):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    models = [m for m in MODEL_ORDER if m in df["Model Name"].unique()]
    methods = sorted(df["Method"].unique())
    width = 0.8 / len(methods)
    x = range(len(models))
    for i, method in enumerate(methods):
        sub = df[df["Method"] == method].set_index("Model Name").reindex(models)
        offsets = [xi + i * width - 0.4 + width / 2 for xi in x]
        ax.bar(offsets, sub["Accuracy (%)"], width=width,
               label=method, color=METHOD_COLORS.get(method))
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Exact-Match Accuracy by Compression Method and Model Size")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "accuracy_by_model.png"), dpi=150)
    plt.close(fig)


def plot_accuracy_vs_size(df):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for method in sorted(df["Method"].unique()):
        sub = df[df["Method"] == method].sort_values("Size (MB)")
        ax.plot(sub["Size (MB)"], sub["Accuracy (%)"], marker="o",
                label=method, color=METHOD_COLORS.get(method))
    ax.set_xlabel("Model Size (MB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Model Size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "accuracy_vs_size.png"), dpi=150)
    plt.close(fig)


def plot_accuracy_vs_latency(df):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for method in sorted(df["Method"].unique()):
        sub = df[df["Method"] == method].sort_values("Latency(ms)")
        ax.plot(sub["Latency(ms)"], sub["Accuracy (%)"], marker="o",
                label=method, color=METHOD_COLORS.get(method))
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Inference Latency")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "accuracy_vs_latency.png"), dpi=150)
    plt.close(fig)


def plot_macro_f1_by_model(df):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    models = [m for m in MODEL_ORDER if m in df["Model Name"].unique()]
    methods = sorted(df["Method"].unique())
    width = 0.8 / len(methods)
    x = range(len(models))
    for i, method in enumerate(methods):
        sub = df[df["Method"] == method].set_index("Model Name").reindex(models)
        offsets = [xi + i * width - 0.4 + width / 2 for xi in x]
        ax.bar(offsets, sub["Macro-F1(%)"], width=width,
               label=method, color=METHOD_COLORS.get(method))
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Macro-F1 by Compression Method and Model Size")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "macro_f1_by_model.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = load_combined()
    plot_accuracy_by_model(df)
    plot_accuracy_vs_size(df)
    plot_accuracy_vs_latency(df)
    plot_macro_f1_by_model(df)
    print(f"Saved 4 plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
