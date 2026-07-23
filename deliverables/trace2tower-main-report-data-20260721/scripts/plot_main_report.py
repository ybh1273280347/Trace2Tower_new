from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "No-Skill": "#999999",
    "Expert-Crafted": "#EE7733",
    "Trace2Skill": "#AA4499",
    "SkillX": "#0077BB",
    "ExpeL": "#009988",
    "Trace2Tower": "#CC3311",
    "Full": "#CC3311",
    "High-only": "#33BBEE",
    "Semantic-only": "#BBBBBB",
    "No transition": "#0077BB",
    "No outcome": "#EE7733",
    "No contrastive": "#009988",
    "Frozen v0": "#999999",
    "TF-IDF Pareto": "#CC3311",
    "Embedding Pareto": "#0077BB",
}


def save_figure(figure, output_dir: Path, name: str) -> None:
    figure.tight_layout()
    figure.savefig(output_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def main_performance(data: dict, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    panels = (
        ("alfworld", "ALFWorld success rate (%)", (45, 94), "{:.2f}"),
        ("webshop", "WebShop mean reward", (0.62, 0.73), "{:.5f}"),
    )
    for axis, (benchmark, title, limits, number_format) in zip(axes, panels, strict=True):
        records = data["main_results"][benchmark]
        methods = [record["method"] for record in records]
        values = [record["score"] for record in records]
        bars = axis.bar(
            range(len(records)),
            values,
            color=[COLORS[method] for method in methods],
            width=0.68,
        )
        axis.set_title(title)
        axis.set_ylim(*limits)
        axis.set_xticks(range(len(records)), methods, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.22)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + (limits[1] - limits[0]) * 0.018,
                number_format.format(value),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    save_figure(figure, output_dir, "main-performance")


def cost_quality(data: dict, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    panels = (
        ("alfworld", "Success rate (%)", (48, 92)),
        ("webshop", "Mean reward", (0.64, 0.72)),
    )
    for axis, (benchmark, ylabel, limits) in zip(axes, panels, strict=True):
        for record in data["main_results"][benchmark]:
            method = record["method"]
            axis.scatter(
                record["input_tokens"] / 1000,
                record["score"],
                s=45 + record["steps"] * 5,
                color=COLORS[method],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            axis.annotate(
                method,
                (record["input_tokens"] / 1000, record["score"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_title(benchmark.upper())
        axis.set_xlabel("Mean input tokens (thousands)")
        axis.set_ylabel(ylabel)
        axis.set_ylim(*limits)
        axis.grid(alpha=0.22)
    save_figure(figure, output_dir, "quality-cost-tradeoff")


def alfworld_mechanisms(data: dict, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    build = data["alfworld_build_ablation"]
    labels = [record["method"] for record in build]
    values = [record["score"] or 0 for record in build]
    bars = axes[0].barh(
        range(len(build)),
        values,
        color=[COLORS[label] for label in labels],
        height=0.62,
    )
    bars[1].set_hatch("///")
    axes[0].set_yticks(range(len(build)), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 95)
    axes[0].set_xlabel("Success rate (%)")
    axes[0].set_title("Build-signal ablation")
    axes[0].grid(axis="x", alpha=0.22)
    for index, (bar, record) in enumerate(zip(bars, build, strict=True)):
        label = "No valid High" if record["score"] is None else f"{record['score']:.2f}"
        axes[0].text(max(bar.get_width(), 1) + 1.2, index, label, va="center", fontsize=8)

    runtime = data["alfworld_deployment_ablation"]
    runtime_bars = axes[1].bar(
        range(len(runtime)),
        [record["score"] for record in runtime],
        color=[COLORS[record["method"]] for record in runtime],
        width=0.58,
    )
    axes[1].set_xticks(range(len(runtime)), [record["method"] for record in runtime])
    axes[1].set_ylim(78, 91)
    axes[1].set_ylabel("Success rate (%)")
    axes[1].set_title("Runtime hierarchy ablation")
    axes[1].grid(axis="y", alpha=0.22)
    for bar, record in zip(runtime_bars, runtime, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            record["score"] + 0.35,
            f"{record['score']:.2f}\n{record['input_tokens'] / 1000:.1f}k tok",
            ha="center",
            fontsize=8,
        )
    save_figure(figure, output_dir, "alfworld-mechanisms")


def deployment_optimization(data: dict, output_dir: Path) -> None:
    deployment = data["deployment_optimization"]
    groups = deployment["groups"]
    x = np.arange(len(groups))
    width = 0.24
    figure, axis = plt.subplots(figsize=(7.2, 3.9))
    for index, series in enumerate(deployment["series"]):
        offsets = x + (index - 1) * width
        bars = axis.bar(
            offsets,
            series["scores"],
            width,
            label=series["method"],
            color=COLORS[series["method"]],
        )
        for bar, value in zip(bars, series["scores"], strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.25,
                f"{value:.2f}",
                ha="center",
                fontsize=7,
            )
    axis.set_xticks(x, groups)
    axis.set_ylim(77, 89)
    axis.set_ylabel("Success rate (%)")
    axis.set_title("Deployment graph optimization across frozen test sets")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    save_figure(figure, output_dir, "deployment-optimization")


def alfworld_family_heatmap(data: dict, output_dir: Path) -> None:
    family_data = data["alfworld_family_results"]
    task_counts = np.asarray(family_data["task_counts"], dtype=float)
    methods = [record["method"] for record in family_data["series"]]
    values = np.asarray(
        [
            100 * np.asarray(record["successes"], dtype=float) / task_counts
            for record in family_data["series"]
        ]
    )
    figure, axis = plt.subplots(figsize=(9.2, 4.3))
    image = axis.imshow(values, cmap="YlOrRd", vmin=20, vmax=100, aspect="auto")
    axis.set_xticks(range(len(family_data["families"])), family_data["families"])
    axis.set_yticks(range(len(methods)), methods)
    axis.tick_params(axis="x", rotation=24)
    axis.set_title("ALFWorld success rate by task family")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value >= 72 else "#222222",
            )
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.025, label="Success rate (%)")
    save_figure(figure, output_dir, "alfworld-family-heatmap")


def tower_compression(data: dict, output_dir: Path) -> None:
    structure = data["tower_structure_metrics"]
    utilization = data["tower_test_utilization"]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))

    stages = (
        ("Segment\ninstances", structure["segment_instances"], "#999999"),
        ("Quotient\nnodes", structure["quotient_nodes"], "#0077BB"),
        ("Mid\nclusters", structure["mid_count"], "#EE7733"),
        ("High\npaths", structure["high_count"], "#CC3311"),
    )
    x_positions = (0.08, 0.34, 0.62, 0.87)
    widths = (0.20, 0.17, 0.13, 0.15)
    heights = (0.62, 0.48, 0.30, 0.36)
    for index, ((label, count, color), x, width, height) in enumerate(
        zip(stages, x_positions, widths, heights, strict=True)
    ):
        rectangle = plt.Rectangle(
            (x - width / 2, 0.50 - height / 2),
            width,
            height,
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
        )
        axes[0].add_patch(rectangle)
        axes[0].text(x, 0.54, f"{count:,}", ha="center", va="center", color="white", weight="bold")
        axes[0].text(x, 0.42, label, ha="center", va="center", color="white", fontsize=8)
        if index < len(stages) - 1:
            axes[0].annotate(
                "",
                xy=(x_positions[index + 1] - widths[index + 1] / 2, 0.50),
                xytext=(x + width / 2, 0.50),
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.2},
            )
    axes[0].text(
        0.21,
        0.12,
        f"Duplicate collapse: {100 * structure['collapse_rate']:.1f}%",
        ha="center",
        fontsize=9,
    )
    axes[0].set_title("Evidence compression and hierarchy formation")
    axes[0].set_xlim(-0.04, 1.0)
    axes[0].set_ylim(0, 1)
    axes[0].axis("off")

    labels = ("High catalog", "Mid catalog")
    rates = (
        100 * utilization["high_catalog_coverage"],
        100 * utilization["mid_catalog_coverage"],
    )
    bars = axes[1].barh(labels, rates, color=("#CC3311", "#EE7733"), height=0.48)
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Catalog exposed on the test set (%)")
    axes[1].set_title("Runtime utilization")
    axes[1].grid(axis="x", alpha=0.22)
    for bar, rate in zip(bars, rates, strict=True):
        axes[1].text(rate + 1.5, bar.get_y() + bar.get_height() / 2, f"{rate:.1f}%", va="center")
    axes[1].text(
        0.5,
        -0.30,
        f"Mean selected per episode: {utilization['mean_high_per_episode']:.2f} High, "
        f"{utilization['mean_mid_per_episode']:.2f} Mid",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=8,
    )
    save_figure(figure, output_dir, "tower-compression-utilization")


def tower_structure(snapshot: dict, output_dir: Path) -> None:
    mid_cards = snapshot["mid_cards"]
    high_cards = snapshot["high_cards"]
    mid_ids = [card["skill_id"] for card in mid_cards]
    mid_index = {skill_id: index for index, skill_id in enumerate(mid_ids)}
    cluster_sizes = np.asarray(
        [len(card["member_segment_ids"]) for card in mid_cards], dtype=float
    )
    high_lengths = np.asarray(
        [len(card["ordered_mid_ids"]) for card in high_cards], dtype=int
    )
    high_support = np.zeros(len(mid_cards), dtype=int)
    adjacency = np.zeros((len(mid_cards), len(mid_cards)), dtype=int)
    for card in high_cards:
        ordered = [mid_index[item] for item in card["ordered_mid_ids"] if item in mid_index]
        for index in set(ordered):
            high_support[index] += 1
        for left, right in zip(ordered, ordered[1:]):
            adjacency[left, right] += 1
            adjacency[right, left] += 1

    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.7))
    axes[0, 0].hist(cluster_sizes, bins=10, color="#0077BB", edgecolor="white")
    axes[0, 0].set_title("Mid cluster coverage")
    axes[0, 0].set_xlabel("Collapsed segment members")
    axes[0, 0].set_ylabel("Number of Mid clusters")
    axes[0, 0].grid(axis="y", alpha=0.22)

    axes[0, 1].scatter(
        cluster_sizes,
        high_support,
        s=35,
        color="#CC3311",
        edgecolor="white",
        linewidth=0.7,
    )
    axes[0, 1].set_title("Mid coverage vs. High support")
    axes[0, 1].set_xlabel("Collapsed segment members")
    axes[0, 1].set_ylabel("High paths containing Mid")
    axes[0, 1].grid(alpha=0.22)

    counts = np.bincount(high_lengths, minlength=int(high_lengths.max()) + 1)
    axes[1, 0].bar(range(len(counts)), counts, color="#EE7733", width=0.7)
    axes[1, 0].set_title("High path length")
    axes[1, 0].set_xlabel("Ordered Mid steps")
    axes[1, 0].set_ylabel("Number of High paths")
    axes[1, 0].set_xticks(range(len(counts)))
    axes[1, 0].grid(axis="y", alpha=0.22)

    order = np.argsort(-high_support)
    heatmap = adjacency[np.ix_(order, order)]
    image = axes[1, 1].imshow(heatmap, cmap="magma", interpolation="nearest")
    axes[1, 1].set_title("High-path Mid co-occurrence")
    axes[1, 1].set_xlabel("Mid clusters (support order)")
    axes[1, 1].set_ylabel("Mid clusters (support order)")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Adjacent High paths")
    save_figure(figure, output_dir, "tower-structure")


def tower_embedding_map(snapshot: dict, output_dir: Path) -> None:
    mid_vectors = np.asarray(snapshot["mid_index"]["vectors"], dtype=float)
    high_vectors = np.asarray(snapshot["high_index"]["vectors"], dtype=float)
    vectors = np.vstack((mid_vectors, high_vectors))
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    components = np.linalg.svd(centered, full_matrices=False)[2][:2]
    coordinates = centered @ components.T
    mid_count = len(mid_vectors)
    figure, axis = plt.subplots(figsize=(6.7, 5.2))
    axis.scatter(
        coordinates[mid_count:, 0],
        coordinates[mid_count:, 1],
        s=18,
        alpha=0.28,
        color="#0077BB",
        label="High vectors",
    )
    axis.scatter(
        coordinates[:mid_count, 0],
        coordinates[:mid_count, 1],
        s=52,
        color="#CC3311",
        edgecolor="white",
        linewidth=0.7,
        label="Mid vectors",
    )
    axis.set_title("ALFWorld Tower embedding map")
    axis.set_xlabel("PCA component 1")
    axis.set_ylabel("PCA component 2")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    save_figure(figure, output_dir, "tower-embedding-map")


def expel_mini(expel_data: dict, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    alfworld = expel_data["alfworld"]["results"]
    alf_methods = ("no_skill", "skillx", "expel")
    display = {"no_skill": "No-Skill", "skillx": "SkillX", "expel": "ExpeL"}
    values = [100 * alfworld[method]["success_rate"] for method in alf_methods]
    bars = axes[0].bar(
        range(len(values)),
        values,
        color=[COLORS[display[method]] for method in alf_methods],
    )
    axes[0].set_xticks(range(len(values)), [display[method] for method in alf_methods])
    axes[0].set_ylim(45, 96)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_title("ALFWorld exploratory subset (n=20)")
    axes[0].grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center")

    webshop = expel_data["webshop"]["results"]
    web_methods = ("no_skill", "skillx", "expert_crafted", "expel")
    display["expert_crafted"] = "Expert-Crafted"
    values = [webshop[method]["mean_reward"] for method in web_methods]
    bars = axes[1].bar(
        range(len(values)),
        values,
        color=[COLORS[display[method]] for method in web_methods],
    )
    axes[1].set_xticks(
        range(len(values)),
        [display[method] for method in web_methods],
        rotation=18,
        ha="right",
    )
    axes[1].set_ylim(0.55, 0.69)
    axes[1].set_ylabel("Mean reward")
    axes[1].set_title("WebShop exploratory subset (n=20)")
    axes[1].grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.004,
            f"{value:.3f}",
            ha="center",
            fontsize=8,
        )
    save_figure(figure, output_dir, "expel-mini")


def main(options: argparse.Namespace) -> int:
    data = json.loads(options.data.read_text(encoding="utf-8"))
    expel_data = json.loads(options.expel_results.read_text(encoding="utf-8"))
    snapshot = json.loads(options.tower.read_text(encoding="utf-8"))
    options.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "figure.facecolor": "white",
        }
    )
    main_performance(data, options.output_dir)
    cost_quality(data, options.output_dir)
    alfworld_mechanisms(data, options.output_dir)
    deployment_optimization(data, options.output_dir)
    alfworld_family_heatmap(data, options.output_dir)
    tower_compression(data, options.output_dir)
    expel_mini(expel_data, options.output_dir)
    tower_structure(snapshot, options.output_dir)
    tower_embedding_map(snapshot, options.output_dir)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("clean_docs/figures/report-data.json"))
    parser.add_argument(
        "--expel-results",
        type=Path,
        default=Path("experiments/baselines/expel-mini/RESULTS.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("clean_docs/figures"))
    parser.add_argument(
        "--tower",
        type=Path,
        default=Path("artifacts/trace2tower/alfworld/original-concept-v17/p310/tower.json"),
    )
    raise SystemExit(main(parser.parse_args()))
