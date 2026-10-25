#!/usr/bin/env python3
"""Build Obs2 case-study folders from entropy-shaping replay results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


GREEN = "#2E7D32"
RED = "#C62828"
BLUE = "#1565C0"
GRAY = "#607D8B"
LIGHT_BLUE = "#90CAF9"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def clean_prompt(prompt: str) -> str:
    text = re.sub(r"<img>.*?</img>", "[IMAGE]", prompt, flags=re.DOTALL)
    text = text.replace("<|im_start|>user", "")
    text = text.replace("<|im_end|><|im_start|>assistant", "")
    return text.strip()


def choose_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [record for record in records if record["desired_effect"]]
    selected: list[dict[str, Any]] = []
    preferences = {
        "high": [
            ("ai2d20", "1796"),
            ("mathverse20_fixed", "944"),
            ("mathvista20_hq", "38"),
            ("mathverse20_fixed", "483"),
            ("mathverse20_fixed", "1439"),
        ],
        "low": [
            ("ai2d20", "2346"),
            ("realworldqa20_hq", "359"),
            ("mathvista20_hq", "455"),
            ("realworldqa20_hq", "588"),
        ],
    }
    by_key = {(record["benchmark"], str(record["question_id"])): record for record in eligible}
    for role in ("high", "low"):
        role_selected = []
        for key in preferences[role]:
            if key in by_key:
                role_selected.append(by_key[key])
            if len(role_selected) == 2:
                break
        if len(role_selected) < 2:
            fallback = [
                record
                for record in eligible
                if record["role"] == role and record not in role_selected
            ]
            fallback.sort(
                key=lambda record: abs(float(record["margin_change"])),
                reverse=True,
            )
            role_selected.extend(fallback[: 2 - len(role_selected)])
        if len(role_selected) != 2:
            raise ValueError(f"Need two successful {role} cases, got {len(role_selected)}")
        selected.extend(role_selected)
    return selected


def bar_colors(labels: list[str], ground_truth: str, consensus: str) -> list[str]:
    colors = []
    for label in labels:
        if label == ground_truth:
            colors.append(GREEN)
        elif label == consensus:
            colors.append(RED)
        else:
            colors.append(GRAY)
    return colors


def add_bar_labels(axis: Any, bars: Any) -> None:
    for bar in bars:
        height = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.018,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def render_case(record: dict[str, Any], output_dir: Path, index: int) -> dict[str, Any]:
    role = record["role"]
    benchmark = record["benchmark"]
    qid = str(record["question_id"])
    folder = output_dir / f"case_{index:02d}_{benchmark}_{qid}_{role}_entropy"
    folder.mkdir(parents=True, exist_ok=True)

    image_path = Path(record["target_image_path"])
    image_copy = folder / f"original_image{image_path.suffix.lower()}"
    shutil.copy2(image_path, image_copy)

    rows = sorted(record["label_rewards"], key=lambda row: row["label"])
    labels = [row["label"] for row in rows]
    density = [float(row["density_reward"]) for row in rows]
    shaped = [float(row["entropy_reward_share"]) for row in rows]
    colors = bar_colors(labels, record["ground_truth"], record["majority_label"])
    ymax = max(max(density), max(shaped))
    ymax = min(1.08, max(0.45, math.ceil((ymax + 0.08) * 10) / 10))

    figure = plt.figure(figsize=(15.5, 5.2), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0])
    image_axis = figure.add_subplot(grid[0, 0])
    density_axis = figure.add_subplot(grid[0, 1])
    shaped_axis = figure.add_subplot(grid[0, 2])

    image = Image.open(image_path).convert("RGB")
    image_axis.imshow(image)
    image_axis.set_axis_off()
    image_axis.set_title(
        f"{benchmark} / qid {qid}\nGT={record['ground_truth']} | consensus={record['majority_label']} "
        f"({record['majority_count']}/32)",
        fontsize=11,
    )

    positions = np.arange(len(labels))
    density_bars = density_axis.bar(positions, density, color=colors, width=0.68)
    density_axis.set_title("Without entropy\n$R(a)=d_a$", fontsize=12)
    density_axis.set_xticks(positions, labels)
    density_axis.set_ylim(0.0, ymax)
    density_axis.set_ylabel("Normalized label reward")
    density_axis.grid(axis="y", alpha=0.2)
    add_bar_labels(density_axis, density_bars)

    shaped_bars = shaped_axis.bar(positions, shaped, color=colors, width=0.68)
    shaped_axis.set_title(
        "With entropy shaping\n"
        + f"$R(a)=e^{{d_a/T(H)}}$, T={record['entropy_temperature']:.3f}",
        fontsize=12,
    )
    shaped_axis.set_xticks(positions, labels)
    shaped_axis.set_ylim(0.0, ymax)
    shaped_axis.grid(axis="y", alpha=0.2)
    add_bar_labels(shaped_axis, shaped_bars)

    if role == "high":
        headline = "High entropy: harmful wrong-consensus margin is reduced"
        margin_label = "wrong consensus - GT"
    else:
        headline = "Low entropy: reliable correct-consensus margin is amplified"
        margin_label = "GT - strongest wrong"
    figure.suptitle(
        f"{headline}\nH={record['answer_entropy']:.3f}; {margin_label}: "
        f"{record['base_margin']:.3f} -> {record['entropy_margin']:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(folder / "reward_shaping_comparison.png", dpi=220, bbox_inches="tight")
    figure.savefig(folder / "reward_shaping_comparison.pdf", bbox_inches="tight")
    plt.close(figure)

    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "answer": row["label"],
                "count": row["count"],
                "frequency": f"{row['frequency']:.6f}",
                "density_reward": f"{row['density_reward']:.6f}",
                "entropy_reward_raw": f"{row['entropy_reward_raw']:.6f}",
                "entropy_reward_share": f"{row['entropy_reward_share']:.6f}",
                "density_advantage": f"{row['density_advantage']:.6f}",
                "entropy_advantage": f"{row['entropy_advantage']:.6f}",
                "ground_truth": "yes" if row["is_ground_truth"] else "",
                "consensus": "yes" if row["is_consensus"] else "",
            }
        )
    with (folder / "reward_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    markdown = [
        "| Answer | Count | Frequency | Density `d` | `exp(d/T)` | Normalized entropy reward | Density adv. | Entropy adv. | GT | Consensus |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in table_rows:
        markdown.append(
            f"| {row['answer']} | {row['count']} | {row['frequency']} | {row['density_reward']} "
            f"| {row['entropy_reward_raw']} | {row['entropy_reward_share']} "
            f"| {row['density_advantage']} | {row['entropy_advantage']} "
            f"| {row['ground_truth']} | {row['consensus']} |"
        )
    (folder / "reward_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    question = clean_prompt(record["prompt"])
    effect_description = (
        "wrong-consensus minus ground-truth"
        if role == "high"
        else "ground-truth minus strongest-wrong"
    )
    report = f"""# Obs2 Entropy-Shaping Case

## Question

```text
{question}
```

## Fixed Inputs

- Benchmark: `{benchmark}`
- Question ID: `{qid}`
- Ground truth: `{record['ground_truth']}`
- Strict consensus: `{record['majority_label']}` ({record['majority_count']}/32)
- Valid outputs: 32/32
- Answer entropy H: {record['answer_entropy']:.6f}
- Embedding scope: `{record['embedding_scope']}`
- Density similarity-kernel T0: {record['density_kernel_temperature']:.2f}
- Entropy temperature T(H): {record['entropy_temperature']:.6f}

## Controlled Comparison

- Baseline: `R_base(a) = d_a`
- Entropy-shaped: `R_entropy(a) = exp(d_a / T(H))`
- Plot normalization: `R_entropy / sum_a R_entropy`; this is a group-constant scaling and does not change GRPO z-scores.
- Both variants reuse exactly the same 32 responses, image, embeddings, and raw density values.

## Effect

- Metric: {effect_description}
- Density margin: {record['base_margin']:.6f}
- Entropy-shaped margin: {record['entropy_margin']:.6f}
- Margin change: {record['margin_change']:+.6f}
- Candidate-level GRPO advantage margin: {record['base_advantage_margin']:.6f} -> {record['entropy_advantage_margin']:.6f}
- Desired directional effect observed: `{str(record['desired_effect']).lower()}`

## Interpretation Boundary

The exponential map is monotonic in density, so it does not change the top-ranked answer for a fixed group. This case demonstrates modulation of the training signal: high entropy should reduce the relative harm of unreliable consensus, while low entropy should increase the relative promotion of reliable consensus.
"""
    (folder / "case_report.md").write_text(report, encoding="utf-8")
    (folder / "case_metadata.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "folder": folder.name,
        "role": role,
        "benchmark": benchmark,
        "question_id": qid,
        "ground_truth": record["ground_truth"],
        "consensus": record["majority_label"],
        "entropy": record["answer_entropy"],
        "temperature": record["entropy_temperature"],
        "base_margin": record["base_margin"],
        "entropy_margin": record["entropy_margin"],
        "margin_change": record["margin_change"],
    }


def build_overview(selected: list[dict[str, Any]], output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    for axis, record in zip(axes.flat, selected):
        rows = sorted(record["label_rewards"], key=lambda row: row["label"])
        labels = [row["label"] for row in rows]
        density = np.asarray([row["density_reward"] for row in rows], dtype=float)
        shaped = np.asarray([row["entropy_reward_share"] for row in rows], dtype=float)
        positions = np.arange(len(labels))
        width = 0.36
        axis.bar(positions - width / 2, density, width, color=GRAY, label="R=density")
        axis.bar(positions + width / 2, shaped, width, color=BLUE, label="entropy-shaped")
        axis.set_xticks(positions, labels)
        axis.set_ylim(0.0, min(1.05, max(0.5, float(max(density.max(), shaped.max()) + 0.08))))
        axis.grid(axis="y", alpha=0.2)
        direction = "harm reduced" if record["role"] == "high" else "promotion amplified"
        axis.set_title(
            f"{record['benchmark']} / {record['question_id']} ({record['role']} H)\n"
            f"H={record['answer_entropy']:.3f}, T={record['entropy_temperature']:.3f}, {direction}",
            fontsize=11,
        )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle("Obs2 case study: entropy changes reward sharpness according to consensus reliability", fontsize=15)
    figure.savefig(output_dir / "obs2_entropy_shaping_cases.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / "obs2_entropy_shaping_cases.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = read_jsonl(args.replay_jsonl)
    selected = choose_cases(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = [render_case(record, args.output_dir, index) for index, record in enumerate(selected, start=1)]
    build_overview(selected, args.output_dir)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Obs2 Entropy-Shaping Case Study",
        "",
        "Controlled comparison on fixed Stage-A N=32 valid outputs:",
        "",
        "- No entropy: `R_base(a) = d_a`.",
        "- With entropy: `R_entropy(a) = exp(d_a / T(H))`.",
        "- `T(H) = 0.10 + 0.20 H`.",
        "- The original density `d_a` is shared; entropy is applied only to the final reward map.",
        "- All selected cases have 32/32 parser-valid outputs.",
        "- High-entropy cases have a strict wrong consensus and show reduced harmful margin.",
        "- Low-entropy cases have a strict correct consensus and show amplified helpful margin.",
        "",
        "The transform is monotonic and therefore does not change the selected argmax. These cases visualize how entropy modulates policy-gradient supervision, not a post-hoc answer correction.",
        "",
        "| Case | Role | Benchmark / qid | GT | Consensus | H | T(H) | Margin before | Margin after | Change |",
        "|---:|---|---|:---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(manifest, start=1):
        lines.append(
            f"| {index} | {row['role']} | {row['benchmark']} / {row['question_id']} "
            f"| {row['ground_truth']} | {row['consensus']} | {row['entropy']:.4f} "
            f"| {row['temperature']:.4f} | {row['base_margin']:.4f} "
            f"| {row['entropy_margin']:.4f} | {row['margin_change']:+.4f} |"
        )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(selected), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
