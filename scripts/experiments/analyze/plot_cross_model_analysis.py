from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
import numpy as np


RUN_IDS = {
    "DeepSeek V4 Flash": {
        "alfworld": {
            "tower": "alfworld-test-v1-flash-v18-budgeted-rewrite-gpt54-full-r0",
            "noskill": "alfworld-test-v1-flash-noskill-r0",
        },
        "webshop": {
            "tower": "webshop-alfworld-v17-replication-p100-validation-r0",
            "noskill": "webshop-original-concept-v1-validation-flash-noskill-r1",
        },
    },
    "GPT-5.4": {
        "alfworld": {
            "tower": "cross-dsflash-render-gpt54-agent-alfworld-tower-r0",
            "noskill": "cross-gpt54-agent-alfworld-noskill-r0",
        },
        "webshop": {
            "tower": "cross-dsflash-render-gpt54-agent-webshop-tower-r0",
            "noskill": "cross-gpt54-agent-webshop-noskill-r0",
        },
    },
    "DeepSeek V4 Pro": {
        "alfworld": {
            "tower": "generalize-gpt54-render-dspro-agent-alfworld-tower-r0",
            "noskill": "generalize-dspro-agent-alfworld-noskill-r0",
        },
        "webshop": {
            "tower": "generalize-gpt54-render-dspro-agent-webshop-tower-r0",
            "noskill": "generalize-dspro-agent-webshop-noskill-r0",
        },
    },
}

AUTHOR_EXECUTOR_RUN_IDS = {
    "GPT-5.4": {
        "DeepSeek V4 Flash": "alfworld-test-v1-flash-v18-budgeted-rewrite-gpt54-full-r0",
        "DeepSeek V4 Pro": "generalize-gpt54-render-dspro-agent-alfworld-tower-r0",
    },
    "DeepSeek V4 Flash": {
        "DeepSeek V4 Flash": "alfworld-author-matrix-dsflash-author-dsflash-user-r0",
        "DeepSeek V4 Pro": "alfworld-author-matrix-dsflash-author-dspro-user-r0",
    },
}

MODEL_COLORS = {
    "DeepSeek V4 Flash": "#009E73",
    "GPT-5.4": "#0072B2",
    "DeepSeek V4 Pro": "#D55E00",
}

FAMILY_LABELS = {
    "look_at_obj_in_light": "Look in light",
    "pick_and_place": "Pick & place",
    "pick_clean_then_place": "Clean & place",
    "pick_cool_then_place": "Cool & place",
    "pick_heat_then_place": "Heat & place",
    "pick_two_obj_and_place": "Pick two & place",
}


def load_run(root: Path, run_id: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted((root / "artifacts" / "runs" / run_id).rglob("results.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error"):
                raise ValueError(f"official row has an error: {run_id}:{row['sample_id']}")
            rows[row["sample_id"]] = row
    if not rows:
        raise ValueError(f"run has no official rows: {run_id}")
    return rows


def load_family_map(root: Path) -> dict[str, str]:
    path = root / "clean_artifacts" / "alfworld" / "manifests" / "alfworld_test.jsonl"
    return {
        row["sample_id"]: row["task_family"]
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    }


def score(row: Mapping[str, object]) -> float:
    return float(row["primary_score"])


def full_success(row: Mapping[str, object]) -> bool:
    return score(row) >= 1.0 - 1e-12


def mean_score(rows: Mapping[str, Mapping[str, object]]) -> float:
    return fmean(score(row) for row in rows.values())


def save_figure(figure, output_dir: Path, name: str) -> None:
    figure.savefig(output_dir / f"{name}.png", dpi=240, bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_model_method_interaction(data: dict, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    panels = (
        ("alfworld", "ALFWorld", 100, "Success rate (%)", (46, 92), "{:.1f}"),
        ("webshop", "WebShop", 1, "Mean reward", (0.58, 0.73), "{:.3f}"),
    )
    for axis, (benchmark, title, scale, ylabel, limits, value_format) in zip(
        axes, panels, strict=True
    ):
        gain_positions = {
            "DeepSeek V4 Flash": 0.42,
            "GPT-5.4": 0.58,
            "DeepSeek V4 Pro": 0.50,
        }
        endpoint_offsets = {
            "DeepSeek V4 Flash": 0.0,
            "GPT-5.4": -0.8 if scale == 100 else -0.002,
            "DeepSeek V4 Pro": 0.6 if scale == 100 else 0.002,
        }
        for model, color in MODEL_COLORS.items():
            values = [
                scale * data[model][benchmark][method]["mean_score"]
                for method in ("noskill", "tower")
            ]
            axis.plot(
                (0, 1),
                values,
                color=color,
                marker="o",
                markersize=7,
                linewidth=2.2,
                label=model,
            )
            offset = endpoint_offsets[model]
            axis.text(
                -0.04,
                values[0] + offset,
                value_format.format(values[0]),
                ha="right",
                va="center",
            )
            axis.text(
                1.04,
                values[1] + offset,
                value_format.format(values[1]),
                ha="left",
                va="center",
            )
            gain = values[1] - values[0]
            gain_x = gain_positions[model]
            axis.text(
                gain_x,
                values[0] + gain_x * gain,
                f"+{gain:.1f}" if scale == 100 else f"+{gain:.3f}",
                color=color,
                ha="center",
                va="bottom",
                fontsize=8,
                weight="semibold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8},
            )
        axis.set_xlim(-0.25, 1.25)
        axis.set_ylim(*limits)
        axis.set_xticks((0, 1), ("No-Skill", "Trace2Tower"))
        axis.set_ylabel(ylabel)
        axis.set_title(f"{title}: method-model interaction")
        axis.grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    figure.subplots_adjust(wspace=0.35)
    save_figure(figure, output_dir, "cross-model-tower-gains")


def family_rates(
    rows: Mapping[str, Mapping[str, object]],
    family_map: Mapping[str, str],
    family_ids: list[str],
) -> list[float]:
    return [
        100
        * fmean(
            full_success(rows[sample_id])
            for sample_id, family in family_map.items()
            if family == family_id
        )
        for family_id in family_ids
    ]


def annotate_heatmap(axis, values: np.ndarray, *, suffix: str = "") -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:+.1f}{suffix}" if suffix else f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 27 or (not suffix and value > 72) else "#222222",
            )


def plot_alfworld_family_spectrum(
    data: dict,
    family_map: Mapping[str, str],
    output_dir: Path,
) -> dict:
    family_ids = list(FAMILY_LABELS)
    models = list(MODEL_COLORS)
    tower = np.asarray(
        [family_rates(data[model]["alfworld"]["tower"]["rows"], family_map, family_ids) for model in models]
    )
    noskill = np.asarray(
        [family_rates(data[model]["alfworld"]["noskill"]["rows"], family_map, family_ids) for model in models]
    )
    gains = tower - noskill

    figure, axes = plt.subplots(2, 1, figsize=(10.6, 5.8), sharex=True)
    score_image = axes[0].imshow(tower, cmap="YlGn", vmin=45, vmax=100, aspect="auto")
    annotate_heatmap(axes[0], tower)
    axes[0].set_yticks(range(len(models)), models)
    axes[0].set_title("ALFWorld Tower success rate by task family")
    figure.colorbar(score_image, ax=axes[0], fraction=0.018, pad=0.02, label="Success rate (%)")

    limit = max(20, float(np.ceil(np.abs(gains).max() / 5) * 5))
    gain_image = axes[1].imshow(
        gains,
        cmap="RdBu",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    annotate_heatmap(axes[1], gains, suffix=" pp")
    axes[1].set_yticks(range(len(models)), models)
    axes[1].set_xticks(
        range(len(family_ids)),
        [FAMILY_LABELS[family_id] for family_id in family_ids],
        rotation=20,
        ha="right",
    )
    axes[1].set_title("Paired gain over each model's No-Skill baseline")
    figure.colorbar(gain_image, ax=axes[1], fraction=0.018, pad=0.02, label="Gain (pp)")
    figure.subplots_adjust(hspace=0.32)
    save_figure(figure, output_dir, "alfworld-cross-model-family-spectrum")
    return {
        model: {
            family_id: {
                "tower_success_rate": tower[model_index, family_index] / 100,
                "gain_pp": gains[model_index, family_index],
            }
            for family_index, family_id in enumerate(family_ids)
        }
        for model_index, model in enumerate(models)
    }


def plot_webshop_reward_deltas(data: dict, output_dir: Path) -> dict:
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharey=True)
    output = {}
    for axis, (model, model_color) in zip(axes, MODEL_COLORS.items(), strict=True):
        tower = data[model]["webshop"]["tower"]["rows"]
        noskill = data[model]["webshop"]["noskill"]["rows"]
        sample_ids = sorted(set(tower) & set(noskill))
        deltas = np.asarray([score(tower[item]) - score(noskill[item]) for item in sample_ids])
        sorted_deltas = np.sort(deltas)
        colors = np.where(sorted_deltas > 1e-12, model_color, np.where(sorted_deltas < -1e-12, "#CC6677", "#D0D0D0"))
        axis.bar(np.arange(len(sorted_deltas)), sorted_deltas, color=colors, width=1.0)
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.axhline(deltas.mean(), color=model_color, linewidth=1.4, linestyle="--")
        wins = int((deltas > 1e-12).sum())
        losses = int((deltas < -1e-12).sum())
        ties = len(deltas) - wins - losses
        axis.text(
            0.03,
            0.96,
            f"{wins} improved / {losses} worse / {ties} tied\nmean {deltas.mean():+.3f}",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
        )
        axis.set_title(model)
        axis.set_xlabel("Tasks ranked by paired reward change")
        axis.grid(axis="y", alpha=0.18)
        output[model] = {
            "improved": wins,
            "worse": losses,
            "tied": ties,
            "mean_delta": float(deltas.mean()),
        }
    axes[0].set_ylabel("Trace2Tower - No-Skill reward")
    figure.suptitle("WebShop paired reward changes", weight="semibold")
    figure.subplots_adjust(top=0.82, wspace=0.13)
    save_figure(figure, output_dir, "webshop-paired-reward-waterfalls")
    return output


def jaccard_success(
    left: Mapping[str, Mapping[str, object]],
    right: Mapping[str, Mapping[str, object]],
) -> float:
    left_success = {sample_id for sample_id, row in left.items() if full_success(row)}
    right_success = {sample_id for sample_id, row in right.items() if full_success(row)}
    return len(left_success & right_success) / len(left_success | right_success)


def plot_success_set_agreement(data: dict, output_dir: Path) -> dict:
    models = list(MODEL_COLORS)
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 7.0))
    output = {}
    for axis, (benchmark, method) in zip(
        axes.flat,
        (("alfworld", "noskill"), ("alfworld", "tower"), ("webshop", "noskill"), ("webshop", "tower")),
        strict=True,
    ):
        matrix = np.zeros((len(models), len(models)))
        for row_index, left_model in enumerate(models):
            for column_index, right_model in enumerate(models):
                matrix[row_index, column_index] = jaccard_success(
                    data[left_model][benchmark][method]["rows"],
                    data[right_model][benchmark][method]["rows"],
                )
        image = axis.imshow(matrix, cmap="Blues", vmin=0.35, vmax=1.0)
        for row_index in range(len(models)):
            for column_index in range(len(models)):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.72 else "#222222",
                )
        short_labels = ("Flash", "GPT-5.4", "Pro")
        axis.set_xticks(range(len(models)), short_labels)
        axis.set_yticks(range(len(models)), short_labels)
        axis.set_title(f"{benchmark.upper()} - {'Tower' if method == 'tower' else 'No-Skill'}")
        output[f"{benchmark}:{method}"] = matrix.tolist()
    figure.colorbar(image, ax=axes, fraction=0.025, pad=0.04, label="Jaccard similarity")
    figure.suptitle("Agreement between model full-success sets", weight="semibold")
    figure.subplots_adjust(top=0.90, wspace=0.28, hspace=0.30, right=0.88)
    save_figure(figure, output_dir, "cross-model-success-set-agreement")
    return output


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2 * tail / (2**discordant))


def plot_alfworld_author_executor_matrix(
    root: Path,
    data: dict,
    output_dir: Path,
) -> dict:
    authors = list(AUTHOR_EXECUTOR_RUN_IDS)
    executors = list(next(iter(AUTHOR_EXECUTOR_RUN_IDS.values())))
    rows_by_cell = {}
    success_rates = np.zeros((len(authors), len(executors)))
    gains = np.zeros_like(success_rates)
    cells = {}

    for author_index, author in enumerate(authors):
        cells[author] = {}
        for executor_index, executor in enumerate(executors):
            run_id = AUTHOR_EXECUTOR_RUN_IDS[author][executor]
            rows = load_run(root, run_id)
            if len(rows) != 134:
                raise ValueError(f"{run_id} has {len(rows)} rows, expected 134")
            rows_by_cell[author, executor] = rows
            success_rate = fmean(full_success(row) for row in rows.values())
            no_skill_rate = data[executor]["alfworld"]["noskill"]["full_success_rate"]
            success_rates[author_index, executor_index] = 100 * success_rate
            gains[author_index, executor_index] = 100 * (success_rate - no_skill_rate)
            cells[author][executor] = {
                "run_id": run_id,
                "n": len(rows),
                "success_count": sum(full_success(row) for row in rows.values()),
                "success_rate": success_rate,
                "gain_over_no_skill_pp": 100 * (success_rate - no_skill_rate),
            }

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    panels = (
        (axes[0], success_rates, "Success rate (%)", "YlGn", 50, 92),
        (axes[1], gains, "Gain over executor-matched No-Skill (pp)", "Blues", 0, 38),
    )
    short_authors = ("GPT-5.4", "DeepSeek Flash")
    short_executors = ("DeepSeek Flash", "DeepSeek Pro")
    for axis, values, title, color_map, minimum, maximum in panels:
        image = axis.imshow(values, cmap=color_map, vmin=minimum, vmax=maximum, aspect="auto")
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > (minimum + maximum) / 2 else "#222222",
                    weight="semibold",
                )
        axis.set_xticks(range(len(executors)), short_executors)
        axis.set_yticks(range(len(authors)), short_authors)
        axis.set_xlabel("Skill executor")
        axis.set_ylabel("Skill author")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axes[1].set_yticks([])
    axes[1].set_ylabel("")
    figure.suptitle("ALFWorld skill-author x executor matrix", weight="semibold")
    figure.subplots_adjust(top=0.82, wspace=0.48)
    save_figure(figure, output_dir, "alfworld-author-executor-matrix")

    comparisons = {}
    for executor in executors:
        gpt_rows = rows_by_cell["GPT-5.4", executor]
        flash_rows = rows_by_cell["DeepSeek V4 Flash", executor]
        sample_ids = sorted(set(gpt_rows) & set(flash_rows))
        gpt_only = sum(
            full_success(gpt_rows[sample_id]) and not full_success(flash_rows[sample_id])
            for sample_id in sample_ids
        )
        flash_only = sum(
            full_success(flash_rows[sample_id]) and not full_success(gpt_rows[sample_id])
            for sample_id in sample_ids
        )
        comparisons[executor] = {
            "gpt54_author_only": gpt_only,
            "deepseek_flash_author_only": flash_only,
            "tied": len(sample_ids) - gpt_only - flash_only,
            "gpt54_minus_flash_author_pp": 100
            * (
                fmean(full_success(gpt_rows[sample_id]) for sample_id in sample_ids)
                - fmean(full_success(flash_rows[sample_id]) for sample_id in sample_ids)
            ),
            "mcnemar_exact_p": exact_mcnemar_p(gpt_only, flash_only),
        }
    return {
        "authors": authors,
        "executors": executors,
        "cells": cells,
        "paired_author_comparisons": comparisons,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    output_dir = root / "clean_docs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    serializable = {}
    for model, benchmarks in RUN_IDS.items():
        data[model] = {}
        serializable[model] = {}
        for benchmark, methods in benchmarks.items():
            data[model][benchmark] = {}
            serializable[model][benchmark] = {}
            for method, run_id in methods.items():
                rows = load_run(root, run_id)
                expected = 134 if benchmark == "alfworld" else 100
                if len(rows) != expected:
                    raise ValueError(f"{run_id} has {len(rows)} rows, expected {expected}")
                aggregate = {
                    "run_id": run_id,
                    "n": len(rows),
                    "mean_score": mean_score(rows),
                    "full_success_rate": fmean(full_success(row) for row in rows.values()),
                }
                data[model][benchmark][method] = {**aggregate, "rows": rows}
                serializable[model][benchmark][method] = aggregate

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    plot_model_method_interaction(data, output_dir)
    serializable["alfworld_family_spectrum"] = plot_alfworld_family_spectrum(
        data, load_family_map(root), output_dir
    )
    serializable["webshop_reward_deltas"] = plot_webshop_reward_deltas(data, output_dir)
    serializable["success_set_agreement"] = plot_success_set_agreement(data, output_dir)
    serializable["alfworld_author_executor_matrix"] = plot_alfworld_author_executor_matrix(
        root,
        data,
        output_dir,
    )
    (output_dir / "cross-model-analysis-data.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
