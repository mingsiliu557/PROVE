#!/usr/bin/env python3
"""Build publication-ready qualitative figures for Obs1, Obs2, and Obs3.

This script is intentionally isolated from the training code. It only reads
existing diagnostic artifacts and copies the selected source images.
"""

from __future__ import annotations

import json
import shutil
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path("/path/to/project/TTRV")
OUT = ROOT / "observation_visualizations_20260728"
IMAGE_OUT = OUT / "images"

OBS1_RECON = Path(
    "/path/to/datasets/experiments/obs1_density_dose_response_qwen_20260724/"
    "reconstruction/reward_dilution_all_750_questions.jsonl"
)
OBS2_RECORDS = ROOT / "outputs/diagnostic_obs2_fixed_bin_selection_20260723/question_selection.jsonl"
OBS3_CASES = ROOT / "case_studies/ours_correct_ttrv_wrong_20260726/curated_cases.jsonl"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
GRAY = "#8A8A8A"
LIGHT_GRAY = "#D9D9D9"
INK = "#202124"


OBS1_CASES = [
    {
        "question_id": "2DQex53EkNGH2cfo3WPuPn",
        "image": "/path/to/datasets/A-OKVQA/hf_validation_images/000004.jpg",
        "reference_summary": "phone / cell phone / mobile phone",
    },
    {
        "question_id": "MuwkW3j4kYfEZXwWuicLjv",
        "image": "/path/to/datasets/A-OKVQA/hf_validation_images/000480.jpg",
        "reference_summary": "crust (10/10 annotators)",
    },
    {
        "question_id": "7Wj97TEQc4fnsxvvsbnUXH",
        "image": "/path/to/datasets/A-OKVQA/hf_validation_images/000125.jpg",
        "reference_summary": "carrot/carrots (9/10 annotators)",
    },
]

OBS2_META = {
    ("ai2d20", "1463"): {
        "dataset": "AI2D",
        "question": "Which phase is shown at label E?",
        "options": "A waning gibbous | B full moon | C waxing crescent | D new moon",
        "image": "/path/to/datasets/ai2d/images_2/image_111.jpg",
    },
    ("mathvista20_hq", "150"): {
        "dataset": "MathVista",
        "question": "WTY is congruent to TWY. Find x.",
        "options": "A 2 | B 4 | C 5 | D 10",
        "image": "/path/to/datasets/mathvista/data/images/283.jpg",
    },
}

OBS3_IDS = ["mme_20-test-738", "mme_20-test-447", "ai2d_20-test-33"]


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_selected_jsonl(path: Path, ids: set[str], key: str = "question_id") -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not any(value in line for value in ids - rows.keys()):
                continue
            row = json.loads(line)
            value = str(row.get(key))
            if value in ids:
                rows[value] = row
                if len(rows) == len(ids):
                    break
    missing = ids - rows.keys()
    if missing:
        raise RuntimeError(f"Missing IDs in {path}: {sorted(missing)}")
    return rows


def read_all_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def copy_image(source: str, prefix: str) -> Path:
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    target = IMAGE_OUT / f"{prefix}{source_path.suffix.lower()}"
    shutil.copy2(source_path, target)
    return target


def show_image(ax: plt.Axes, path: Path) -> None:
    with Image.open(path) as image:
        ax.imshow(image.convert("RGB"))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def short_text(value: str, width: int = 58) -> str:
    return "\n".join(textwrap.wrap(value.replace(" Answer with a short direct answer.", ""), width=width))


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def build_obs1() -> list[dict]:
    selected = read_selected_jsonl(OBS1_RECON, {case["question_id"] for case in OBS1_CASES})
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 12.5), gridspec_kw={"width_ratios": [1.05, 1.45, 1.15]})
    exported: list[dict] = []

    for row_idx, case in enumerate(OBS1_CASES):
        row = selected[case["question_id"]]
        image_path = copy_image(case["image"], f"obs1_{row_idx + 1}_{case['question_id']}")
        show_image(axes[row_idx, 0], image_path)
        axes[row_idx, 0].set_title(
            f"Case {row_idx + 1}: {short_text(row['question'], 42)}\nAccepted references: {case['reference_summary']}",
            loc="left",
            color=INK,
            pad=8,
        )

        counter = Counter(row["exact_answer_counter"])
        top = counter.most_common(7)
        labels = [answer if len(answer) <= 30 else answer[:27] + "..." for answer, _ in reversed(top)]
        values = [count for _, count in reversed(top)]
        correct_variants = set(row["selected_correct_cluster_answers"])
        colors = [GREEN if answer in correct_variants else RED for answer, _ in reversed(top)]
        axes[row_idx, 1].barh(np.arange(len(values)), values, color=colors, edgecolor="none")
        axes[row_idx, 1].set_yticks(np.arange(len(labels)), labels)
        axes[row_idx, 1].set_xlabel("Exact-string count among valid rollouts")
        axes[row_idx, 1].set_title("String frequency fragments one correct meaning", loc="left", color=INK)
        axes[row_idx, 1].grid(axis="x", alpha=0.2)

        semantic = row["semantic_candidate_reward_x"]
        frequency = row["frequency_candidate_reward_y"]
        axes[row_idx, 2].bar(
            ["Exact-frequency\nreward", "Semantic-density\nreward"],
            [frequency, semantic],
            color=[ORANGE, BLUE],
            width=0.62,
        )
        axes[row_idx, 2].set_ylim(0, 1.02)
        axes[row_idx, 2].set_ylabel("Reward assigned to correct meaning")
        axes[row_idx, 2].grid(axis="y", alpha=0.2)
        axes[row_idx, 2].set_title(
            f"Correct cluster: {row['selected_correct_cluster_variant_count']} strings, "
            f"mass {row['selected_correct_cluster_mass']}/{row['n_valid']}",
            loc="left",
            color=INK,
        )
        for idx, value in enumerate([frequency, semantic]):
            axes[row_idx, 2].text(idx, value + 0.025, f"{value:.3f}", ha="center", fontsize=10)

        exported.append(
            {
                "question_id": case["question_id"],
                "question": row["question"],
                "accepted_reference_summary": case["reference_summary"],
                "image": str(image_path),
                "exact_answer_counter": row["exact_answer_counter"],
                "correct_semantic_cluster_answers": row["selected_correct_cluster_answers"],
                "correct_semantic_cluster_mass": row["selected_correct_cluster_mass"],
                "valid_rollouts": row["n_valid"],
                "frequency_reward": frequency,
                "semantic_density_reward": semantic,
            }
        )

    fig.suptitle(
        "Obs1: Semantic density restores reward lost to exact-string fragmentation",
        fontsize=16,
        fontweight="bold",
        color=INK,
        y=1.01,
    )
    fig.text(
        0.5,
        0.002,
        "A-OKVQA is used because each question has multiple accepted human answers. "
        "Green strings belong to the evaluator-correct semantic cluster; red strings do not.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(h_pad=2.2, w_pad=2.2)
    save_figure(fig, "obs1_semantic_fragmentation_cases")
    return exported


def normalize_choice(answer: str) -> str:
    answer = str(answer).strip().upper()
    return answer if answer in set("ABCDEF") else "invalid"


def aggregate_choice_counter(raw_counter: dict) -> Counter:
    counter: Counter = Counter()
    for answer, count in raw_counter.items():
        counter[normalize_choice(answer)] += int(count)
    return counter


def plot_counter(ax: plt.Axes, counter: Counter, gt: str, selected: str, title: str) -> None:
    labels = sorted((label for label in counter if label != "invalid"))
    if "invalid" in counter:
        labels.append("invalid")
    colors = [GREEN if label == gt else (ORANGE if label == selected else LIGHT_GRAY) for label in labels]
    values = [counter[label] for label in labels]
    ax.bar(labels, values, color=colors, width=0.65)
    ax.set_ylim(0, 32)
    ax.set_ylabel("Rollout count")
    ax.set_title(title, loc="left", color=INK)
    ax.grid(axis="y", alpha=0.2)


def build_obs2() -> list[dict]:
    rows = read_all_jsonl(OBS2_RECORDS)
    bins = ["low", "mid", "high"]
    labels = ["Low", "Mid", "High"]
    stats = {}
    for name in bins:
        group = [row for row in rows if row["h_bin"] == name]
        stats[name] = {
            "n": len(group),
            "majority_acc": np.mean([row["majority_correct"] for row in group]) * 100,
            "ours_acc": np.mean([row["ours_correct"] for row in group]) * 100,
            "invalid_rescue": sum(
                row["ours_correct"] and not row["majority_correct"] and row["majority_is_invalid"] for row in group
            ),
            "valid_rerank": sum(
                row["ours_correct"] and not row["majority_correct"] and not row["majority_is_invalid"] for row in group
            ),
            "loss": sum(row["majority_correct"] and not row["ours_correct"] for row in group),
        }

    chosen_keys = set(OBS2_META)
    chosen = {(row["benchmark"], str(row["question_id"])): row for row in rows if (row["benchmark"], str(row["question_id"])) in chosen_keys}
    if chosen.keys() != chosen_keys:
        raise RuntimeError(f"Missing Obs2 cases: {chosen_keys - chosen.keys()}")

    fig = plt.figure(figsize=(15.5, 10.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.1], hspace=0.36, wspace=0.28)
    ax_acc = fig.add_subplot(grid[0, 0])
    ax_dec = fig.add_subplot(grid[0, 1])
    ax_ai2d = fig.add_subplot(grid[1, 0])
    ax_math = fig.add_subplot(grid[1, 1])

    x = np.arange(3)
    majority = [stats[name]["majority_acc"] for name in bins]
    ours = [stats[name]["ours_acc"] for name in bins]
    ax_acc.plot(x, majority, marker="o", linewidth=2.4, color=ORANGE, label="Majority@32")
    ax_acc.plot(x, ours, marker="o", linewidth=2.4, color=BLUE, label="Density selection")
    ax_acc.set_xticks(x, [f"{label}\n(n={stats[name]['n']})" for label, name in zip(labels, bins)])
    ax_acc.set_ylabel("Selection accuracy (%)")
    ax_acc.set_title("A. Accuracy by fixed step-0 entropy bin", loc="left", fontweight="bold")
    ax_acc.grid(axis="y", alpha=0.2)
    ax_acc.legend(frameon=False)
    for idx, (maj, our) in enumerate(zip(majority, ours)):
        ax_acc.text(idx, our + 1.2, f"+{our - maj:.2f} pp", ha="center", color=BLUE, fontsize=9)

    invalid = np.array([stats[name]["invalid_rescue"] for name in bins])
    valid = np.array([stats[name]["valid_rerank"] for name in bins])
    loss = np.array([stats[name]["loss"] for name in bins])
    ax_dec.bar(x, invalid, color=GRAY, label="Invalid-majority rescue")
    ax_dec.bar(x, valid, bottom=invalid, color=GREEN, label="Valid-answer rerank")
    ax_dec.bar(x, -loss, color=RED, label="Majority correct, ours wrong")
    ax_dec.axhline(0, color=INK, linewidth=0.8)
    ax_dec.set_xticks(x, labels)
    ax_dec.set_ylabel("Number of questions")
    ax_dec.set_title("B. Source of selection changes", loc="left", fontweight="bold")
    ax_dec.legend(frameon=False, fontsize=9)
    ax_dec.grid(axis="y", alpha=0.2)

    exported = []
    for ax, key, panel in [(ax_ai2d, ("ai2d20", "1463"), "C"), (ax_math, ("mathvista20_hq", "150"), "D")]:
        row = chosen[key]
        meta = OBS2_META[key]
        image_path = copy_image(meta["image"], f"obs2_{meta['dataset'].lower()}_{key[1]}")
        counter = aggregate_choice_counter(row["parsed_counter_all_outputs"])
        plot_counter(
            ax,
            counter,
            str(row["ground_truth"]),
            str(row["ours_answer"]),
            f"{panel}. {meta['dataset']}: H={row['step0_all_output_entropy']:.3f}; "
            f"majority {row['majority_answer']} (wrong), density {row['ours_answer']} (correct)\n"
            f"{short_text(meta['question'], 70)} | GT={row['ground_truth']}",
        )
        ax.text(
            0.01,
            0.96,
            short_text(meta["options"], 75),
            transform=ax.transAxes,
            va="top",
            fontsize=8.5,
            color="#555555",
        )
        exported.append(
            {
                "benchmark": key[0],
                "question_id": key[1],
                "question": meta["question"],
                "options": meta["options"],
                "ground_truth": row["ground_truth"],
                "entropy": row["step0_all_output_entropy"],
                "temperature": row["density_temperature"],
                "counter": dict(counter),
                "majority_answer": row["majority_answer"],
                "density_answer": row["ours_answer"],
                "density_cluster_masses": row["density_cluster_masses"],
                "image": str(image_path),
            }
        )

    fig.suptitle(
        "Obs2: Gains increase with uncertainty, but are mostly invalid-majority rescue",
        fontsize=16,
        fontweight="bold",
        color=INK,
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "Bins are fixed from all 32 step-0 outputs (invalid is an ordinary entropy category). "
        "Only 2 of 247 corrected cases are valid-answer reranks, so this is reliability evidence, not isolated causal evidence for temperature.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    save_figure(fig, "obs2_entropy_reliability")
    return [{"bin_statistics": stats}, *exported]


def plot_vote_counter(ax: plt.Axes, raw_counter: dict, gt: str, title: str) -> None:
    counter = aggregate_choice_counter(raw_counter)
    labels = sorted((label for label in counter if label != "invalid"))
    if "invalid" in counter:
        labels.append("invalid")
    values = [counter[label] for label in labels]
    colors = [GREEN if label == gt else (GRAY if label == "invalid" else RED) for label in labels]
    ax.bar(labels, values, color=colors, width=0.62)
    ax.set_ylim(0, 8)
    ax.set_ylabel("Votes (N=8)")
    ax.set_title(title, loc="left", color=INK)
    ax.grid(axis="y", alpha=0.2)


def build_obs3() -> list[dict]:
    selected = read_selected_jsonl(OBS3_CASES, set(OBS3_IDS))
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 12.5), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    exported = []

    for row_idx, question_id in enumerate(OBS3_IDS):
        row = selected[question_id]
        image_path = copy_image(row["image_path"], f"obs3_{row_idx + 1}_{question_id}")
        show_image(axes[row_idx, 0], image_path)
        option_text = " | ".join(f"{key}: {value}" for key, value in row["options"].items())
        axes[row_idx, 0].set_title(
            f"Case {row_idx + 1}: {short_text(row['question'], 44)}\n{short_text(option_text, 55)}\nGT: {row['ground_truth']} ({row['ground_truth_option_text']})",
            loc="left",
            color=INK,
            pad=8,
        )
        plot_vote_counter(
            axes[row_idx, 1],
            row["ours_prediction_counter"],
            row["ground_truth"],
            f"Ours: {row['ours_prediction']} (correct)",
        )
        plot_vote_counter(
            axes[row_idx, 2],
            row["ttrv_prediction_counter"],
            row["ground_truth"],
            f"TTRV: {row['ttrv_prediction']} (wrong)",
        )
        exported.append(
            {
                "dataset": row["dataset"],
                "question_id": question_id,
                "question": row["question"],
                "options": row["options"],
                "ground_truth": row["ground_truth"],
                "ours_prediction": row["ours_prediction"],
                "ttrv_prediction": row["ttrv_prediction"],
                "ours_prediction_counter": row["ours_prediction_counter"],
                "ttrv_prediction_counter": row["ttrv_prediction_counter"],
                "image": str(image_path),
            }
        )

    fig.suptitle(
        "Obs3: Visual cases where ours is correct and TTRV is confidently wrong",
        fontsize=16,
        fontweight="bold",
        color=INK,
        y=1.01,
    )
    fig.text(
        0.5,
        0.002,
        "All panels use the same final majority@8 protocol. Green denotes the ground-truth option; red denotes a valid wrong option.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(h_pad=2.4, w_pad=2.0)
    save_figure(fig, "obs3_visual_grounding_cases")
    return exported


def write_readme(payload: dict) -> None:
    readme = """# Observation Visualizations

This directory contains three reproducible, publication-oriented figures built from existing diagnostic outputs. No model was retrained and no training code was modified.

## Outputs

- `obs1_semantic_fragmentation_cases.png/.pdf`: three A-OKVQA cases showing exact-string reward fragmentation and density-only semantic recovery.
- obs1_cases/: one self-contained folder per Obs1 case, with the original image, question, Markdown/CSV distribution table, reward metadata, and an individual PNG/PDF figure.
- `obs2_entropy_reliability.png/.pdf`: fixed step-0 entropy-bin accuracy, a decomposition of gains, and the two clean valid-answer reranking cases.
- `obs3_visual_grounding_cases.png/.pdf`: three visually explicit closed-choice cases where ours is correct and TTRV is wrong under the same final majority@8 protocol.
- `selected_cases.json`: compact machine-readable records for all selected cases and aggregate bin statistics.
- `images/`: copies of the eight selected source images.

## Evaluation Note for A-OKVQA

A-OKVQA is deliberately used for Obs1 because it supplies multiple human reference answers. In the diagnostic data, a prediction is marked correct when its normalized string matches any accepted reference. The training reward is a separate soft direct-answer score, `min(number_of_reference_matches / 3, 1)`. The figure uses semantic membership only to analyze fragmentation; it does not rewrite the official evaluator.

Obs1 is intentionally a component-level visualization. It compares exact-string frequency with semantic density on the same candidates and excludes entropy temperature, counterfactual/KL-as-reward, PAPO loss, and policy training. It therefore supports the claim that density repairs reward fragmentation, rather than claiming that the full PROVE selector alone caused each displayed recovery.

For each valid response a, the case tables report r_TTRV(a) = count(exact(a)) / N_valid and r_density(a) = mass(semantic_cluster(a)) / N_valid. The reward panel additionally compares the count-weighted mean TTRV reward received by candidates in the correct semantic cluster against their shared density reward. Invalid outputs are reported but excluded from both distributions, matching the completed Obs1 protocol.

## Obs2 Interpretation Constraint

The entropy-bin gain rises from Low to High, but the audit shows that almost all corrected questions are invalid-majority rescues. Only two corrected questions are clean rerankings between valid answer options, and their density margins are small. Therefore the figure supports reliability-aware selection under uncertainty, but it must not be presented as isolated causal proof that the temperature transform alone improves argmax selection.

## Reproduce

```bash
/root/miniconda3/bin/python observation_visualizations_20260728/build_observation_visualizations.py
/root/miniconda3/bin/python observation_visualizations_20260728/build_obs1_case_folders.py
```
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    (OUT / "selected_cases.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    IMAGE_OUT.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    payload = {"obs1": build_obs1(), "obs2": build_obs2(), "obs3": build_obs3()}
    write_readme(payload)
    print(json.dumps({key: len(value) for key, value in payload.items()}, indent=2))


if __name__ == "__main__":
    main()
