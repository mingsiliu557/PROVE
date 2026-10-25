#!/usr/bin/env python3
"""Export one self-contained folder per selected Obs1 fragmentation case."""

from __future__ import annotations

import csv
import json
import shutil
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from build_observation_visualizations import (
    BLUE,
    GRAY,
    GREEN,
    INK,
    LIGHT_GRAY,
    ORANGE,
    RED,
    configure_plotting,
)


ROOT = Path(__file__).resolve().parent
SOURCE_JSONL = Path(
    "/path/to/datasets/experiments/obs1_density_dose_response_qwen_20260724/"
    "reconstruction/reward_dilution_all_750_questions.jsonl"
)
SELECTED_JSON = ROOT / "selected_cases.json"
OUTPUT_ROOT = ROOT / "obs1_cases"

CASE_ORDER = [
    ("2DQex53EkNGH2cfo3WPuPn", "case_01_phone"),
    ("MuwkW3j4kYfEZXwWuicLjv", "case_02_pizza_crust"),
    ("7Wj97TEQc4fnsxvvsbnUXH", "case_03_carrots"),
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact(text: object, width: int = 44) -> str:
    value = " ".join(str(text).split())
    return value if len(value) <= width else value[: width - 1] + "..."


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_rows(record: dict) -> tuple[list[dict], list[dict]]:
    valid_total = int(record["n_valid"])
    answer_to_cluster: dict[str, dict] = {}
    cluster_rows: list[dict] = []
    for cluster in record["semantic_clusters"]:
        variants = cluster["variants"]
        cluster_row = {
            "cluster_id": int(cluster["cluster_id"]),
            "mass": int(cluster["mass"]),
            "probability": float(cluster["mass"] / valid_total),
            "contains_evaluator_correct": bool(cluster["contains_evaluator_correct"]),
            "variants": [variant["answer"] for variant in variants],
        }
        cluster_rows.append(cluster_row)
        for variant in variants:
            answer_to_cluster[str(variant["answer"])] = {
                **cluster_row,
                "evaluator_correct": bool(variant["evaluator_correct"]),
            }

    max_string_count = max(record["exact_answer_counter"].values())
    max_cluster_mass = max(row["mass"] for row in cluster_rows)
    answer_rows = []
    for answer, count in record["exact_answer_counter"].items():
        cluster = answer_to_cluster[answer]
        answer_rows.append(
            {
                "answer": answer,
                "count": int(count),
                "ttrv_frequency_reward": float(count / valid_total),
                "semantic_cluster": f"C{cluster['cluster_id']}",
                "semantic_cluster_mass": int(cluster["mass"]),
                "density_reward": float(cluster["probability"]),
                "evaluator_exact_correct": bool(cluster["evaluator_correct"]),
                "semantic_correct_cluster": bool(cluster["contains_evaluator_correct"]),
                "ttrv_top_string": int(count) == max_string_count,
                "density_top_cluster": int(cluster["mass"]) == max_cluster_mass,
            }
        )
    answer_rows.sort(key=lambda row: (-row["count"], row["answer"]))
    cluster_rows.sort(key=lambda row: (-row["mass"], row["cluster_id"]))
    return answer_rows, cluster_rows


def write_csv(path: Path, answer_rows: list[dict], record: dict) -> None:
    fields = list(answer_rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(answer_rows)
        writer.writerow(
            {
                "answer": "<invalid outputs; excluded from both valid-only distributions>",
                "count": int(record["n_invalid"]),
            }
        )


def write_markdown_table(path: Path, answer_rows: list[dict], record: dict) -> None:
    header = (
        f"**Correct-semantic reward comparison:** TTRV mean reward "
        f"Y={record['frequency_candidate_reward_y']:.4f}; density reward "
        f"X={record['semantic_candidate_reward_x']:.4f}; reward retained "
        f"Y/X={record['retention_ratio']:.2%}; diluted {record['dilution_ratio']:.2%}.\n\n"
        "| Normalized response | Count | TTRV reward | Semantic cluster | "
        "Cluster mass | Density reward | Exact GT match | Correct semantic cluster | TTRV top | Density top |\n"
        "|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|\n"
    )
    lines = [header]
    for row in answer_rows:
        lines.append(
            "| {answer} | {count} | {ttrv:.4f} | {cluster} | {mass} | {density:.4f} | "
            "{exact} | {semantic} | {t_top} | {d_top} |\n".format(
                answer=markdown_escape(row["answer"]),
                count=row["count"],
                ttrv=row["ttrv_frequency_reward"],
                cluster=row["semantic_cluster"],
                mass=row["semantic_cluster_mass"],
                density=row["density_reward"],
                exact="yes" if row["evaluator_exact_correct"] else "no",
                semantic="yes" if row["semantic_correct_cluster"] else "no",
                t_top="yes" if row["ttrv_top_string"] else "no",
                d_top="yes" if row["density_top_cluster"] else "no",
            )
        )
    lines.append(
        f"| invalid (excluded) | {int(record['n_invalid'])} | - | - | - | - | - | - | - | - |\n"
    )
    path.write_text("".join(lines), encoding="utf-8")


def plot_case(
    case_dir: Path,
    image_path: Path,
    record: dict,
    selected: dict,
    answer_rows: list[dict],
    cluster_rows: list[dict],
) -> None:
    configure_plotting()
    fig = plt.figure(figsize=(16.5, 11.2))
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.0, 1.48], height_ratios=[1.0, 0.88], hspace=0.48, wspace=0.43
    )
    ax_image = fig.add_subplot(grid[0, 0])
    ax_ttrv = fig.add_subplot(grid[0, 1])
    ax_reward = fig.add_subplot(grid[1, 0])
    ax_density = fig.add_subplot(grid[1, 1])

    with Image.open(image_path) as image:
        ax_image.imshow(image.convert("RGB"))
    ax_image.set_xticks([])
    ax_image.set_yticks([])
    ax_image.set_title("Original image", loc="left", fontweight="bold", color=INK)
    ax_image.text(
        0.0,
        -0.08,
        "\n".join(textwrap.wrap(record["question"], width=46)),
        transform=ax_image.transAxes,
        va="top",
        fontsize=10,
        color=INK,
    )
    ax_image.text(
        0.0,
        -0.27,
        "Accepted: " + selected["accepted_reference_summary"],
        transform=ax_image.transAxes,
        va="top",
        fontsize=9,
        color="#555555",
    )

    answers = [row["answer"] for row in answer_rows]
    counts = [row["count"] for row in answer_rows]
    y = np.arange(len(answer_rows))
    colors = [GREEN if row["semantic_correct_cluster"] else RED for row in answer_rows]
    bars = ax_ttrv.barh(y, counts, color=colors, alpha=0.88)
    ax_ttrv.set_yticks(y, [compact(answer, 39) for answer in answers], fontsize=8.5)
    ax_ttrv.invert_yaxis()
    ax_ttrv.set_xlabel("Count among valid rollouts")
    ax_ttrv.set_title("TTRV: exact-string frequency", loc="left", fontweight="bold", color=INK)
    ax_ttrv.grid(axis="x", alpha=0.2)
    for bar, row in zip(bars, answer_rows):
        if row["ttrv_top_string"]:
            bar.set_edgecolor(BLUE)
            bar.set_linewidth(2.5)
    ax_ttrv.text(
        0.0,
        -0.12,
        f"valid={record['n_valid']}/32; invalid={record['n_invalid']} excluded. Blue outline = selected top string.",
        transform=ax_ttrv.transAxes,
        fontsize=8.5,
        color="#555555",
    )

    reward_values = [
        float(record["frequency_candidate_reward_y"]),
        float(record["semantic_candidate_reward_x"]),
    ]
    reward_bars = ax_reward.bar(
        ["TTRV\nmean $r_{freq}$", "Density\n$r_{density}$"],
        reward_values,
        color=[ORANGE, BLUE],
        width=0.58,
    )
    ax_reward.set_ylim(0, max(reward_values) * 1.30)
    ax_reward.set_ylabel("Reward per correct-semantic candidate")
    ax_reward.set_title("Reward recovered from fragmentation", loc="left", fontweight="bold", color=INK)
    ax_reward.grid(axis="y", alpha=0.2)
    for bar, value in zip(reward_bars, reward_values):
        ax_reward.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(reward_values) * 0.035,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax_reward.text(
        0.5,
        0.92,
        f"TTRV retains {record['retention_ratio']:.1%}; dilution = {record['dilution_ratio']:.1%}",
        transform=ax_reward.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )

    cluster_labels = [
        f"C{row['cluster_id']}: {compact(' / '.join(row['variants']), 48)}" for row in cluster_rows
    ]
    cluster_probs = [row["probability"] for row in cluster_rows]
    cy = np.arange(len(cluster_rows))
    cluster_colors = [GREEN if row["contains_evaluator_correct"] else RED for row in cluster_rows]
    cluster_bars = ax_density.barh(cy, cluster_probs, color=cluster_colors, alpha=0.88)
    ax_density.set_yticks(cy, cluster_labels, fontsize=8.5)
    ax_density.invert_yaxis()
    ax_density.set_xlim(0, 1.0)
    ax_density.set_xlabel("Semantic-cluster mass / valid rollouts")
    ax_density.set_title("Density: semantic-cluster distribution", loc="left", fontweight="bold", color=INK)
    ax_density.grid(axis="x", alpha=0.2)
    max_mass = max(row["mass"] for row in cluster_rows)
    for bar, row in zip(cluster_bars, cluster_rows):
        if row["mass"] == max_mass:
            bar.set_edgecolor(BLUE)
            bar.set_linewidth(2.5)
        ax_density.text(
            row["probability"] + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{row['mass']}/{record['n_valid']}",
            va="center",
            fontsize=9,
        )
    ax_density.text(
        0.0,
        -0.12,
        "Green = cluster containing an evaluator-correct response. Blue outline = selected top cluster.",
        transform=ax_density.transAxes,
        fontsize=8.5,
        color="#555555",
    )

    fig.suptitle(
        "Exact-string fragmentation vs. semantic density",
        fontsize=16,
        fontweight="bold",
        color=INK,
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "Same 32 responses. TTRV reward = exact-string frequency; density reward = semantic-cluster mass. Invalid outputs are excluded in both.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.055, right=0.98, top=0.91, bottom=0.10)
    fig.savefig(case_dir / "distribution_plot.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(case_dir / "distribution_plot.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def write_question(
    path: Path,
    image_name: str,
    record: dict,
    selected: dict,
    answer_rows: list[dict],
    cluster_rows: list[dict],
) -> None:
    ttrv_top = [row["answer"] for row in answer_rows if row["ttrv_top_string"]]
    density_top = [row for row in cluster_rows if row["mass"] == max(item["mass"] for item in cluster_rows)]
    density_variants = density_top[0]["variants"]
    text = f"""# {record['question_id']}

![Original image]({image_name})

## Original Question

{record['question']}

**Accepted reference summary:** {selected['accepted_reference_summary']}

## Same-Rollout Comparison

- Total samples: {record['n_total']}
- Valid direct answers: {record['n_valid']}
- Invalid outputs excluded from both valid-only distributions: {record['n_invalid']}
- TTRV exact-frequency top string: {', '.join(ttrv_top)}
- Density top semantic cluster: {', '.join(density_variants)}
- TTRV mean reward received by candidates in the correct semantic cluster: {record['frequency_candidate_reward_y']:.4f}
- Density reward received by those candidates: {record['semantic_candidate_reward_x']:.4f}
- TTRV reward retention: {record['retention_ratio']:.2%}; reward dilution: {record['dilution_ratio']:.2%}

TTRV reward = exact-string count / N_valid.  
Density reward = semantic-cluster mass / N_valid; every candidate in the same cluster receives that cluster-level reward.

See [distribution_table.md](distribution_table.md) for all responses and [distribution_plot.png](distribution_plot.png) for the visualization.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    selected_payload = json.loads(SELECTED_JSON.read_text(encoding="utf-8"))
    selected_by_id = {row["question_id"]: row for row in selected_payload["obs1"]}
    source_by_id = {row["question_id"]: row for row in read_jsonl(SOURCE_JSONL)}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = []
    for question_id, folder_name in CASE_ORDER:
        record = source_by_id[question_id]
        selected = selected_by_id[question_id]
        case_dir = OUTPUT_ROOT / folder_name
        case_dir.mkdir(parents=True, exist_ok=True)

        source_image = Path(selected["image"])
        image_name = "original_image" + source_image.suffix.lower()
        image_path = case_dir / image_name
        shutil.copy2(source_image, image_path)

        answer_rows, cluster_rows = build_rows(record)
        write_csv(case_dir / "distribution_table.csv", answer_rows, record)
        write_markdown_table(case_dir / "distribution_table.md", answer_rows, record)
        write_question(case_dir / "question.md", image_name, record, selected, answer_rows, cluster_rows)
        plot_case(case_dir, image_path, record, selected, answer_rows, cluster_rows)

        metadata = {
            "question_id": question_id,
            "benchmark": record["benchmark"],
            "question": record["question"],
            "accepted_reference_summary": selected["accepted_reference_summary"],
            "n_total": record["n_total"],
            "n_valid": record["n_valid"],
            "n_invalid": record["n_invalid"],
            "reward_comparison": {
                "ttrv_correct_semantic_mean_reward": record["frequency_candidate_reward_y"],
                "density_correct_semantic_reward": record["semantic_candidate_reward_x"],
                "retention_ratio": record["retention_ratio"],
                "dilution_ratio": record["dilution_ratio"],
            },
            "answer_distribution": answer_rows,
            "semantic_cluster_distribution": cluster_rows,
        }
        (case_dir / "case_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        manifest.append({"question_id": question_id, "folder": str(case_dir)})

    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(manifest), "output_root": str(OUTPUT_ROOT)}, indent=2))


if __name__ == "__main__":
    main()
