from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "results" / "summary"
DEFAULT_OUTPUT = ROOT / "figures" / "generated"

BENCHMARK_ORDER = ["hc3", "hc3_zh", "mage", "raid"]
BENCHMARK_LABELS = {
    "hc3": "HC3-English",
    "hc3_zh": "HC3-Chinese",
    "mage": "MAGE",
    "raid": "RAID",
}
METHOD_ORDER = [
    "Word TF-IDF",
    "Char TF-IDF",
    "Style-LR",
    "Hybrid-LR",
    "Encoder-only",
    "Encoder + Style",
]
METHOD_LABELS = {
    "word_tfidf": "Word TF-IDF",
    "char_tfidf": "Char TF-IDF",
    "style": "Style-LR",
    "hybrid": "Hybrid-LR",
    "roberta_encoder": "Encoder-only",
    "xlmr_encoder": "Encoder-only",
    "roberta_style": "Encoder + Style",
    "xlmr_style": "Encoder + Style",
}
METHOD_SHORT = {
    "Word TF-IDF": "W",
    "Char TF-IDF": "C",
    "Style-LR": "S",
    "Hybrid-LR": "H",
    "Encoder-only": "E",
    "Encoder + Style": "E+S",
}
ATTACK_ORDER = ["whitespace", "connector_swap", "sentence_shuffle", "punct_drop"]
ATTACK_LABELS = {
    "whitespace": "Whitespace",
    "connector_swap": "Connector swap",
    "sentence_shuffle": "Sentence shuffle",
    "punct_drop": "Punctuation drop",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate report figures from experiment summaries.")
    parser.add_argument("--main-results", default=str(DEFAULT_SUMMARY / "main_results.csv"))
    parser.add_argument("--robustness-results", default=str(DEFAULT_SUMMARY / "robustness_results.csv"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--formats",
        default="png,svg,pdf",
        help="comma-separated output formats (default: png,svg,pdf)",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def load_inputs(main_path: Path, robustness_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = pd.read_csv(main_path)
    robustness = pd.read_csv(robustness_path)
    require_columns(
        main,
        {
            "benchmark",
            "method",
            "seed",
            "f1",
            "auroc",
            "human_fpr",
            "inference_ms_per_sample",
            "model_size_mb",
        },
        main_path,
    )
    require_columns(
        robustness,
        {"benchmark", "method", "attack", "seed", "f1"},
        robustness_path,
    )
    if main.empty or robustness.empty:
        raise ValueError("summary inputs must not be empty")
    seeds = sorted(set(main["seed"].dropna()) | set(robustness["seed"].dropna()))
    if len(seeds) != 1:
        raise ValueError(f"expected one shared seed, found {seeds}")
    main = main.copy()
    robustness = robustness.copy()
    main["benchmark_label"] = main["benchmark"].map(BENCHMARK_LABELS)
    main["method_label"] = main["method"].map(METHOD_LABELS)
    robustness["benchmark_label"] = robustness["benchmark"].map(BENCHMARK_LABELS)
    robustness["method_label"] = robustness["method"].map(METHOD_LABELS)
    if main[["benchmark_label", "method_label"]].isna().any().any():
        raise ValueError("main results contain an unknown benchmark or method")
    if robustness[["benchmark_label", "method_label"]].isna().any().any():
        raise ValueError("robustness results contain an unknown benchmark or method")
    return main, robustness


def save_figure(fig, output_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    saved = []
    for extension in formats:
        target = output_dir / f"{stem}.{extension}"
        fig.savefig(target, dpi=dpi, bbox_inches="tight", facecolor="white")
        saved.append(target)
    plt.close(fig)
    return saved


def plot_main_f1_heatmap(main: pd.DataFrame):
    matrix = main.pivot(index="benchmark_label", columns="method_label", values="f1")
    matrix = matrix.reindex(
        index=[BENCHMARK_LABELS[item] for item in BENCHMARK_ORDER], columns=METHOD_ORDER
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "F1 score"},
        ax=ax,
    )
    ax.set_title("Main Results Across Benchmarks", pad=12, weight="bold")
    ax.set_xlabel("Method")
    ax.set_ylabel("Benchmark")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig


def robustness_delta_matrix(robustness: pd.DataFrame) -> pd.DataFrame:
    originals = robustness.loc[
        robustness["attack"] == "original", ["benchmark", "method", "f1"]
    ].rename(columns={"f1": "original_f1"})
    attacked = robustness.loc[robustness["attack"] != "original"].merge(
        originals, on=["benchmark", "method"], how="left", validate="many_to_one"
    )
    if attacked["original_f1"].isna().any():
        raise ValueError("each robustness group must contain an original row")
    attacked["delta_f1"] = attacked["f1"] - attacked["original_f1"]
    attacked["benchmark_label"] = attacked["benchmark"].map(BENCHMARK_LABELS)
    attacked["method_label"] = attacked["method"].map(METHOD_LABELS)
    attacked["row_label"] = attacked["benchmark_label"] + " / " + attacked["method_label"]
    row_order = [
        f"{BENCHMARK_LABELS[benchmark]} / {method}"
        for benchmark in BENCHMARK_ORDER
        for method in METHOD_ORDER
    ]
    matrix = attacked.pivot(index="row_label", columns="attack", values="delta_f1")
    return matrix.reindex(index=row_order, columns=ATTACK_ORDER).rename(columns=ATTACK_LABELS)


def plot_robustness_heatmap(robustness: pd.DataFrame):
    matrix = robustness_delta_matrix(robustness)
    limit = max(0.15, float(np.nanmax(np.abs(matrix.to_numpy()))))
    limit = np.ceil(limit * 20) / 20
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    axes = axes.ravel()
    colorbar_ax = fig.add_axes([0.92, 0.16, 0.018, 0.68])
    for index, (ax, benchmark) in enumerate(zip(axes, BENCHMARK_ORDER)):
        benchmark_label = BENCHMARK_LABELS[benchmark]
        prefix = f"{benchmark_label} / "
        rows = [item for item in matrix.index if item.startswith(prefix)]
        part = matrix.loc[rows].copy()
        part.index = [item.removeprefix(prefix) for item in part.index]
        sns.heatmap(
            part,
            annot=True,
            fmt="+.3f",
            cmap="RdBu",
            center=0,
            vmin=-limit,
            vmax=limit,
            linewidths=0.6,
            linecolor="white",
            cbar=index == 0,
            cbar_ax=colorbar_ax if index == 0 else None,
            cbar_kws={"label": "Delta F1 relative to Original"},
            ax=ax,
        )
        ax.set_title(benchmark_label, weight="bold", pad=8)
        ax.set_xlabel("Perturbation" if index >= 2 else "")
        ax.set_ylabel("Method" if index % 2 == 0 else "")
        ax.tick_params(axis="x", rotation=18, labelsize=8.5)
        ax.tick_params(axis="y", rotation=0, labelsize=8.5)
    fig.suptitle("Robustness Under Text Perturbations", y=0.98, weight="bold")
    fig.subplots_adjust(left=0.12, right=0.89, bottom=0.10, top=0.91, wspace=0.22, hspace=0.28)
    return fig


def fusion_pairs(main: pd.DataFrame) -> pd.DataFrame:
    deep = main.loc[
        main["method_label"].isin(["Encoder-only", "Encoder + Style"]),
        ["benchmark", "benchmark_label", "method_label", "f1", "auroc", "human_fpr"],
    ]
    if len(deep) != len(BENCHMARK_ORDER) * 2:
        raise ValueError("expected one encoder and one fusion result per benchmark")
    return deep


def plot_fusion_dumbbell(main: pd.DataFrame):
    deep = fusion_pairs(main)
    metrics = [("f1", "F1"), ("auroc", "AUROC"), ("human_fpr", "Human FPR")]
    benchmark_labels = [BENCHMARK_LABELS[item] for item in BENCHMARK_ORDER]
    y_positions = np.arange(len(benchmark_labels))
    colors = {"Encoder-only": "#0072B2", "Encoder + Style": "#D55E00"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        pivot = deep.pivot(index="benchmark_label", columns="method_label", values=metric)
        pivot = pivot.reindex(benchmark_labels)
        for y, benchmark in zip(y_positions, benchmark_labels):
            encoder = pivot.loc[benchmark, "Encoder-only"]
            fusion = pivot.loc[benchmark, "Encoder + Style"]
            ax.plot([encoder, fusion], [y, y], color="#8c8c8c", linewidth=2, zorder=1)
            ax.scatter(encoder, y, s=75, color=colors["Encoder-only"], zorder=2)
            ax.scatter(fusion, y, s=75, color=colors["Encoder + Style"], zorder=2)
            ax.text(
                max(encoder, fusion) + 0.012,
                y,
                f"{fusion - encoder:+.3f}",
                va="center",
                fontsize=8,
            )
        ax.set_xlim(0, 1.08)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.grid(axis="x", color="#dddddd", linewidth=0.8)
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Score")
        ax.invert_yaxis()
    axes[0].set_yticks(y_positions, benchmark_labels)
    axes[0].set_ylabel("Benchmark")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=8, label=label)
        for label, color in colors.items()
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False)
    fig.suptitle("Effect of Explicit Style Fusion", y=0.99, weight="bold")
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.14, top=0.78, wspace=0.08)
    return fig


def bubble_sizes(model_size: pd.Series, maximum_size: float) -> np.ndarray:
    values = np.log1p(pd.to_numeric(model_size, errors="coerce").fillna(0).to_numpy())
    maximum = np.log1p(maximum_size) or 1.0
    return 70 + 260 * values / maximum


def plot_performance_efficiency(main: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    norm = plt.Normalize(0, max(0.35, float(main["human_fpr"].max())))
    cmap = plt.get_cmap("plasma")
    maximum_size = float(main["model_size_mb"].max())
    label_offsets = {
        "W": (-9, 11),
        "C": (-9, -13),
        "S": (9, 10),
        "H": (9, -13),
        "E": (-11, 12),
        "E+S": (13, -13),
    }
    for ax, benchmark in zip(axes, BENCHMARK_ORDER):
        part = main.loc[main["benchmark"] == benchmark].copy()
        sizes = bubble_sizes(part["model_size_mb"], maximum_size)
        scatter = ax.scatter(
            part["inference_ms_per_sample"],
            part["f1"],
            s=sizes,
            c=part["human_fpr"],
            cmap=cmap,
            norm=norm,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.8,
        )
        for _, row in part.iterrows():
            short = METHOD_SHORT[row["method_label"]]
            ax.annotate(
                short,
                (row["inference_ms_per_sample"], row["f1"]),
                xytext=label_offsets[short],
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=7.5,
                color="black",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
            )
        ax.set_xscale("log")
        ax.set_xlim(0.01, 30)
        ax.set_ylim(0, 1.03)
        ax.set_title(BENCHMARK_LABELS[benchmark], weight="bold")
        ax.grid(True, which="both", color="#e5e5e5", linewidth=0.7)
    reference_sizes = [0.002, 3, 20, 391]
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#999999",
            markeredgecolor="white",
            markersize=float(np.sqrt(bubble_sizes(pd.Series([value]), maximum_size)[0])),
        )
        for value in reference_sizes
    ]
    axes[0].legend(
        size_handles,
        ["0.002", "3", "20", "391"],
        title="Model size (MB)",
        loc="lower left",
        ncol=2,
        fontsize=7,
        title_fontsize=8,
        frameon=True,
    )
    fig.supxlabel("Inference time per sample (ms, log scale)", y=0.065)
    fig.supylabel("F1 score")
    colorbar = fig.colorbar(scatter, ax=axes.tolist(), fraction=0.025, pad=0.03)
    colorbar.set_label("Human FPR")
    legend_text = "  ".join(f"{short}={label}" for label, short in METHOD_SHORT.items())
    fig.text(0.5, 0.018, legend_text, ha="center", fontsize=8.5)
    fig.suptitle("Performance-Efficiency Trade-off", y=0.99, weight="bold")
    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.13, top=0.93, wspace=0.14, hspace=0.20)
    return fig


def main():
    args = parse_args()
    formats = [item.strip().lower().lstrip(".") for item in args.formats.split(",") if item.strip()]
    print(f"selected formats: {formats}")
    invalid_formats = set(formats) - {"png", "svg", "pdf"}
    if not formats or invalid_formats:
        raise SystemExit(f"formats must be selected from png,svg,pdf; invalid={sorted(invalid_formats)}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_frame, robustness_frame = load_inputs(
        Path(args.main_results), Path(args.robustness_results)
    )
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    figures = [
        ("01_main_f1_heatmap", plot_main_f1_heatmap(main_frame)),
        ("02_robustness_delta_f1_heatmap", plot_robustness_heatmap(robustness_frame)),
        ("03_fusion_ablation_dumbbell", plot_fusion_dumbbell(main_frame)),
        ("04_performance_efficiency_bubble", plot_performance_efficiency(main_frame)),
    ]
    saved = []
    for stem, figure in figures:
        saved.extend(save_figure(figure, output_dir, stem, formats, args.dpi))
    print(f"seed={int(main_frame['seed'].iloc[0])} figures={len(figures)} files={len(saved)}")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
