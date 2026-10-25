#!/usr/bin/env python3
"""Export valid-only low/high-entropy Obs2 visualization cases."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from build_observation_visualizations import (
    BLUE,
    GREEN,
    INK,
    LIGHT_GRAY,
    ORANGE,
    RED,
    configure_plotting,
)


ROOT = Path(__file__).resolve().parent
SELECTION_JSONL = ROOT.parent / "outputs/diagnostic_obs2_fixed_bin_selection_20260723/question_selection.jsonl"
DUMP_ROOT = Path(
    "/path/to/datasets/experiments/diagnostic_stage_a_20260718_043946/dumps"
)
OUTPUT_ROOT = ROOT / "obs2_cases"

CASE_ORDER = [
    ("ai2d20", "10", "case_01_ai2d_low_entropy", "low", "sharpening control"),
    ("ai2d20", "1463", "case_02_ai2d_high_entropy", "high", "valid-answer rerank"),
    ("mathvista20_hq", "116", "case_03_mathvista_low_entropy", "low", "sharpening control"),
    ("mathvista20_hq", "150", "case_04_mathvista_high_entropy", "high", "valid-answer rerank"),
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_label(value: object) -> str:
    text = str(value).strip().upper()
    return text if len(text) == 1 and text in set("ABCDEF") else "invalid"


def clean_prompt(prompt: object) -> str:
    text = re.sub(r"<IMG_CONTEXT>|<[^>]+>", " ", str(prompt or ""))
    text = " ".join(text.split())
    text = re.sub(r"^user\s+", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+assistant$", "", text, flags=re.IGNORECASE)


def extract_question_and_options(prompt: object) -> tuple[str, dict[str, str]]:
    text = clean_prompt(prompt)
    options: dict[str, str] = {}
    if "Options:" in text:
        prefix, option_text = text.split("Options:", 1)
        matches = re.findall(r"([A-F])\.\s*(.*?)(?=\s+[A-F]\.\s|$)", option_text)
        options = {label: value.strip() for label, value in matches}
        marker = "Choose the correct answer from the options below"
        question = prefix.split(marker, 1)[0].strip()
    elif "Choices:" in text:
        prefix, option_text = text.split("Choices:", 1)
        matches = re.findall(r"\(([A-F])\)\s*(.*?)(?=\s+\([A-F]\)\s|$)", option_text)
        options = {label: value.strip() for label, value in matches}
        question = prefix.rsplit("Question:", 1)[-1].strip()
    else:
        question = text
    return question, options


def valid_samples(samples: list[dict]) -> list[dict]:
    return [sample for sample in sorted(samples, key=lambda row: int(row.get("sample_idx", 0))) if normalize_label(sample.get("parsed_answer")) != "invalid"]


def normalized_entropy(values: list[float]) -> float:
    positive = [float(value) for value in values if float(value) > 0]
    if len(positive) <= 1:
        return 0.0
    total = sum(positive)
    probabilities = [value / total for value in positive]
    return float(-sum(value * math.log(value) for value in probabilities) / math.log(len(probabilities)))


def top2_margin(values: list[float]) -> float:
    ordered = sorted((float(value) for value in values), reverse=True)
    return ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]


def frequency_top_labels(rows: list[dict]) -> list[str]:
    peak = max(row["ttrv_valid_frequency_reward"] for row in rows)
    return [
        row["answer"]
        for row in rows
        if abs(row["ttrv_valid_frequency_reward"] - peak) <= 1e-12
    ]


def build_rows(record: dict, samples: list[dict]) -> tuple[list[dict], str]:
    valid = valid_samples(samples)
    counter = Counter(normalize_label(sample["parsed_answer"]) for sample in valid)
    if not counter:
        raise RuntimeError("No valid samples")
    valid_majority = counter.most_common(1)[0][0]
    labels = sorted(
        label
        for label in set(counter) | set(record["density_cluster_masses"])
        if normalize_label(label) != "invalid"
        and (counter.get(label, 0) > 0 or float(record["density_cluster_masses"].get(label, 0.0)) > 0)
    )
    gt = normalize_label(record["ground_truth"])
    ours = normalize_label(record["ours_answer"])
    rows = []
    for label in labels:
        count = int(counter.get(label, 0))
        rows.append(
            {
                "answer": label,
                "valid_count": count,
                "ttrv_valid_frequency_reward": float(count / len(valid)),
                "ours_density_temperature_reward": float(record["density_cluster_masses"].get(label, 0.0)),
                "ground_truth": label == gt,
                "ttrv_valid_majority": label == valid_majority,
                "ours_selected": label == ours,
            }
        )
    return rows, valid_majority


def write_tables(case_dir: Path, rows: list[dict]) -> None:
    with (case_dir / "distribution_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Valid answer | Count | TTRV valid-frequency reward | Ours density-temperature reward | GT | TTRV selected | Ours selected |\n",
        "|---|---:|---:|---:|:---:|:---:|:---:|\n",
    ]
    for row in rows:
        lines.append(
            "| {answer} | {count} | {freq:.4f} | {density:.4f} | {gt} | {ttrv} | {ours} |\n".format(
                answer=row["answer"],
                count=row["valid_count"],
                freq=row["ttrv_valid_frequency_reward"],
                density=row["ours_density_temperature_reward"],
                gt="yes" if row["ground_truth"] else "no",
                ttrv="yes" if row["ttrv_valid_majority"] else "no",
                ours="yes" if row["ours_selected"] else "no",
            )
        )
    (case_dir / "distribution_table.md").write_text("".join(lines), encoding="utf-8")


def write_valid_samples(case_dir: Path, samples: list[dict], candidate_rewards: list[float]) -> None:
    rows = []
    for sample in sorted(samples, key=lambda row: int(row.get("sample_idx", 0))):
        sample_idx = int(sample.get("sample_idx", 0))
        parsed = normalize_label(sample.get("parsed_answer"))
        if parsed == "invalid":
            continue
        rows.append(
            {
                "sample_idx": sample_idx,
                "raw_text": str(sample.get("raw_text", "")),
                "parsed_answer": parsed,
                "is_correct": bool(sample.get("is_correct")),
                "ours_reward": float(candidate_rewards[sample_idx]),
            }
        )
    with (case_dir / "valid_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def outcome_text(regime: str, gt: str, valid_majority: str, ours: str) -> str:
    ttrv_mark = "correct" if valid_majority == gt else "wrong"
    ours_mark = "correct" if ours == gt else "wrong"
    if regime == "low":
        mechanism = "Low answer entropy -> lower temperature -> concentrated reward"
    else:
        mechanism = "High answer entropy -> higher temperature -> flatter reward"
    return (
        f"{mechanism}\n\n"
        f"Ground truth: {gt}\n"
        f"TTRV valid-majority: {valid_majority} ({ttrv_mark})\n"
        f"Ours selection: {ours} ({ours_mark})"
    )


def plot_case(
    case_dir: Path,
    image_path: Path,
    record: dict,
    question: str,
    options: dict[str, str],
    rows: list[dict],
    valid_majority: str,
    regime: str,
) -> None:
    configure_plotting()
    fig = plt.figure(figsize=(16.2, 10.8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.45], hspace=0.38, wspace=0.34)
    ax_image = fig.add_subplot(grid[0, 0])
    ax_counts = fig.add_subplot(grid[0, 1])
    ax_summary = fig.add_subplot(grid[1, 0])
    ax_rewards = fig.add_subplot(grid[1, 1])

    with Image.open(image_path) as image:
        ax_image.imshow(image.convert("RGB"))
    ax_image.set_xticks([])
    ax_image.set_yticks([])
    ax_image.set_title("Original image", loc="left", fontweight="bold", color=INK)
    ax_image.text(
        0.0,
        -0.08,
        "\n".join(textwrap.wrap(question, width=50)),
        transform=ax_image.transAxes,
        va="top",
        fontsize=10,
        color=INK,
    )
    options_line = " | ".join(f"{key}. {value}" for key, value in options.items())
    ax_image.text(
        0.0,
        -0.27,
        "\n".join(textwrap.wrap(options_line, width=58)),
        transform=ax_image.transAxes,
        va="top",
        fontsize=9,
        color="#555555",
    )

    labels = [row["answer"] for row in rows]
    counts = [row["valid_count"] for row in rows]
    colors = [GREEN if row["ground_truth"] else LIGHT_GRAY for row in rows]
    bars = ax_counts.bar(labels, counts, color=colors, width=0.64)
    valid_total = sum(counts)
    ax_counts.set_ylim(0, max(4, valid_total * 1.12))
    ax_counts.set_ylabel("Count among valid rollouts")
    ax_counts.set_title(
        f"Valid-only answer distribution (N={valid_total})",
        loc="left",
        fontweight="bold",
        color=INK,
    )
    ax_counts.grid(axis="y", alpha=0.2)
    for bar, row in zip(bars, rows):
        if row["ttrv_valid_majority"]:
            bar.set_edgecolor(ORANGE)
            bar.set_linewidth(3.0)
        ax_counts.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + valid_total * 0.018,
            str(row["valid_count"]),
            ha="center",
        )

    gt = normalize_label(record["ground_truth"])
    ours = normalize_label(record["ours_answer"])
    reward_values = [row["ours_density_temperature_reward"] for row in rows]
    reward_h = normalized_entropy(reward_values)
    reward_margin = top2_margin(reward_values)
    top_labels = frequency_top_labels(rows)
    summary = (
        outcome_text(regime, gt, valid_majority, ours)
        + f"\nFrequency top labels = {', '.join(top_labels)}"
        + f"\n\nH_answer(valid) = {record['density_answer_entropy_valid_only']:.4f}"
        + f"\nTemperature = {record['density_temperature']:.4f}"
        + f"\nH_reward = {reward_h:.4f}"
        + f"\nPeak reward = {max(reward_values):.4f}"
        + f"\nTop-2 reward margin = {reward_margin:.6f}"
    )
    ax_summary.axis("off")
    ax_summary.text(
        0.04,
        0.96,
        summary,
        va="top",
        fontsize=12.5,
        linespacing=1.42,
        color=INK,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#F7F7F7", "edgecolor": LIGHT_GRAY},
    )

    x = np.arange(len(rows))
    width = 0.36
    freq = [row["ttrv_valid_frequency_reward"] for row in rows]
    density = reward_values
    ax_rewards.bar(x - width / 2, freq, width, color=ORANGE, label="TTRV valid frequency")
    ax_rewards.bar(x + width / 2, density, width, color=BLUE, label="Ours density + temperature")
    ax_rewards.set_xticks(x, labels)
    ax_rewards.set_ylim(0, 1.0)
    ax_rewards.set_ylabel("Reward")
    ax_rewards.set_title("Reward profile over valid labels", loc="left", fontweight="bold", color=INK)
    ax_rewards.grid(axis="y", alpha=0.2)
    ax_rewards.legend(frameon=False)

    title_regime = "Low entropy: sharpened reward" if regime == "low" else "High entropy: flattened reward"
    fig.suptitle(
        f"Obs2 {title_regime}",
        fontsize=16,
        fontweight="bold",
        color=INK,
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "All statistics, entropy, majority votes, and rewards in this figure use valid A-F rollouts only; invalid outputs are excluded.",
        ha="center",
        fontsize=9.3,
        color="#555555",
    )
    fig.subplots_adjust(left=0.055, right=0.98, top=0.91, bottom=0.10)
    fig.savefig(case_dir / "entropy_temperature_visualization.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(case_dir / "entropy_temperature_visualization.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def write_question(
    case_dir: Path,
    image_name: str,
    record: dict,
    question: str,
    options: dict[str, str],
    rows: list[dict],
    valid_majority: str,
    regime: str,
    role: str,
) -> None:
    gt = normalize_label(record["ground_truth"])
    ours = normalize_label(record["ours_answer"])
    reward_values = [row["ours_density_temperature_reward"] for row in rows]
    top_labels = frequency_top_labels(rows)
    option_lines = "\n".join(f"- {key}. {value}" for key, value in options.items())
    text = f"""# {record['benchmark']} / {record['question_id']}

![Original image]({image_name})

## Original Question

{question}

{option_lines}

**Ground truth:** {gt}

## Valid-Only Entropy and Reward

- Visualization role: {role}
- Regime: {regime} entropy
- Valid rollout count: {sum(row['valid_count'] for row in rows)}
- Valid-answer normalized entropy: {record['density_answer_entropy_valid_only']:.4f}
- Entropy-controlled temperature: {record['density_temperature']:.4f}
- Reward entropy: {normalized_entropy(reward_values):.4f}
- Maximum density-temperature reward: {max(reward_values):.4f}
- Top-2 density-temperature reward margin: {top2_margin(reward_values):.6f}
- TTRV top-frequency labels: {', '.join(top_labels)}{' (tie)' if len(top_labels) > 1 else ''}
- TTRV valid-majority: {valid_majority} ({'correct' if valid_majority == gt else 'wrong'})
- Ours selection: {ours} ({'correct' if ours == gt else 'wrong'})

TTRV reward is the valid-label frequency. Ours computes semantic density on valid candidate embeddings and uses valid-answer entropy to set the temperature. Invalid outputs are excluded from every quantity in this case folder.

See [distribution_table.md](distribution_table.md), [valid_samples.csv](valid_samples.csv), and [entropy_temperature_visualization.png](entropy_temperature_visualization.png).
"""
    (case_dir / "question.md").write_text(text, encoding="utf-8")


def plot_overview(cases: list[dict]) -> None:
    configure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2), sharey=True)
    for ax, case in zip(axes.flat, cases):
        rows = case["rows"]
        labels = [row["answer"] for row in rows]
        x = np.arange(len(rows))
        width = 0.36
        ax.bar(
            x - width / 2,
            [row["ttrv_valid_frequency_reward"] for row in rows],
            width,
            color=ORANGE,
            label="TTRV frequency",
        )
        ax.bar(
            x + width / 2,
            [row["ours_density_temperature_reward"] for row in rows],
            width,
            color=BLUE,
            label="Ours",
        )
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.2)
        ax.set_title(
            f"{case['dataset']} {case['regime'].upper()} | H={case['entropy']:.3f}, T={case['temperature']:.3f}\n"
            f"TTRV={case['valid_majority']} ({'correct' if case['valid_majority'] == case['gt'] else 'wrong'}), "
            f"Ours={case['ours']} ({'correct' if case['ours'] == case['gt'] else 'wrong'})",
            loc="left",
            fontsize=10.5,
        )
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_ylabel("Reward")
    axes[1, 0].set_ylabel("Reward")
    fig.suptitle(
        "Obs2: Entropy-controlled temperature adapts reward sharpness",
        fontsize=16,
        fontweight="bold",
        color=INK,
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "Matched within AI2D and MathVista. Low-entropy rows are sharpening controls; high-entropy rows are valid-only corrections.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0.03, 0.04, 0.99, 0.94], h_pad=2.0, w_pad=1.6)
    fig.savefig(OUTPUT_ROOT / "entropy_temperature_pairs.png", dpi=300, facecolor="white")
    fig.savefig(OUTPUT_ROOT / "entropy_temperature_pairs.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    selection = {
        (row["benchmark"], str(row["question_id"])): row for row in read_jsonl(SELECTION_JSONL)
    }
    dump_cache = {benchmark: read_jsonl(DUMP_ROOT / benchmark / "real.jsonl") for benchmark, *_ in CASE_ORDER}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []
    overview_cases = []

    for benchmark, question_id, folder_name, regime, role in CASE_ORDER:
        record = selection[(benchmark, question_id)]
        samples = [row for row in dump_cache[benchmark] if str(row["question_id"]) == question_id]
        if len(samples) != 32:
            raise RuntimeError(f"Expected 32 samples for {benchmark}/{question_id}, found {len(samples)}")
        source = samples[0]
        question, options = extract_question_and_options(source["prompt"])
        source_image = Path(source["target_image_path"])
        if not source_image.exists():
            raise FileNotFoundError(source_image)

        rows, valid_majority = build_rows(record, samples)
        gt = normalize_label(record["ground_truth"])
        ours = normalize_label(record["ours_answer"])
        if ours != gt:
            raise RuntimeError(f"Ours is not correct for {benchmark}/{question_id}")
        if regime == "high" and valid_majority == gt:
            raise RuntimeError(f"High-entropy case is not a correction: {benchmark}/{question_id}")
        if regime == "low" and valid_majority != gt:
            raise RuntimeError(f"Low-entropy control is not majority-correct: {benchmark}/{question_id}")

        case_dir = OUTPUT_ROOT / folder_name
        case_dir.mkdir(parents=True, exist_ok=True)
        image_name = "original_image" + source_image.suffix.lower()
        image_path = case_dir / image_name
        shutil.copy2(source_image, image_path)
        write_tables(case_dir, rows)
        write_valid_samples(case_dir, samples, record["candidate_rewards"])
        write_question(case_dir, image_name, record, question, options, rows, valid_majority, regime, role)
        plot_case(case_dir, image_path, record, question, options, rows, valid_majority, regime)

        metadata = {
            "benchmark": benchmark,
            "question_id": question_id,
            "question": question,
            "options": options,
            "ground_truth": gt,
            "role": role,
            "regime": regime,
            "valid_rollout_count": sum(row["valid_count"] for row in rows),
            "excluded_invalid_count": 32 - sum(row["valid_count"] for row in rows),
            "valid_answer_entropy": record["density_answer_entropy_valid_only"],
            "temperature": record["density_temperature"],
            "reward_entropy": normalized_entropy([row["ours_density_temperature_reward"] for row in rows]),
            "reward_top2_margin": top2_margin([row["ours_density_temperature_reward"] for row in rows]),
            "ttrv_top_frequency_labels": frequency_top_labels(rows),
            "ttrv_valid_majority": valid_majority,
            "ours_selection": ours,
            "distribution": rows,
            "reward_variant": record["reward_variant"],
            "counterfactual_available": record["counterfactual_available"],
        }
        (case_dir / "case_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        manifest.append({"benchmark": benchmark, "question_id": question_id, "folder": str(case_dir), "regime": regime, "role": role})
        overview_cases.append(
            {
                "dataset": "AI2D" if benchmark == "ai2d20" else "MathVista",
                "regime": regime,
                "entropy": record["density_answer_entropy_valid_only"],
                "temperature": record["density_temperature"],
                "gt": gt,
                "valid_majority": valid_majority,
                "ours": ours,
                "rows": rows,
            }
        )

    plot_overview(overview_cases)
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = """# Obs2 Valid-Only Entropy-Temperature Cases

All artifacts are reconstructed from existing step-0 real-image N=32 rollouts. No inference was rerun.

- AI2D and MathVista each contribute a matched low/high-entropy pair.
- Low-entropy cases are sharpening controls where both selectors are correct.
- High-entropy cases are the two available clean valid-only corrections: TTRV valid-majority is wrong and ours is correct.
- No low-entropy valid-only correction exists in the diagnostic records; all five such corrections have H_valid >= 0.875.
- All five valid-only corrections are exact-frequency ties; the selected high-entropy cases show semantic-density tie-breaking, not strict minority-over-majority reversal.
- Invalid outputs are excluded from counts, entropy, majority selection, reward tables, and plots.
- Counterfactual token log-probabilities are unavailable here, so the displayed selector is density + repository entropy temperature only.
"""
    (OUTPUT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"cases": len(manifest), "output_root": str(OUTPUT_ROOT)}, indent=2))


if __name__ == "__main__":
    main()
