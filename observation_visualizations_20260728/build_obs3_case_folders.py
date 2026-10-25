#!/usr/bin/env python3
"""Build self-contained Obs3 visual-grounding case folders."""

from __future__ import annotations

import os as _anon_os
_ANON_PROJECT_ROOT = _anon_os.environ.get('PROJECT_ROOT', '/path/to/project')


import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(f'{_ANON_PROJECT_ROOT}/TTRV')
SOURCE = ROOT / "case_studies/ours_correct_ttrv_wrong_20260726/curated_cases.jsonl"
OUTPUT = ROOT / "observation_visualizations_20260728/obs3_cases"
CASE_IDS = ("mme_20-test-738", "mme_20-test-447", "ai2d_20-test-33")

GREEN = "#009E73"
RED = "#D55E00"
GRAY = "#8A8A8A"


def read_cases() -> list[dict[str, Any]]:
    selected = {}
    with SOURCE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["question_id"])
            if qid in CASE_IDS:
                selected[qid] = row
    missing = set(CASE_IDS) - set(selected)
    if missing:
        raise ValueError(f"Missing Obs3 cases: {sorted(missing)}")
    return [selected[qid] for qid in CASE_IDS]


def normalized_counter(raw: dict[str, int], options: dict[str, str]) -> dict[str, int]:
    result = Counter({label: 0 for label in options})
    invalid = 0
    for label, count in raw.items():
        if label in options:
            result[label] += int(count)
        else:
            invalid += int(count)
    if invalid:
        result["invalid"] = invalid
    return dict(result)


def colors(labels: list[str], ground_truth: str) -> list[str]:
    return [GREEN if label == ground_truth else (GRAY if label == "invalid" else RED) for label in labels]


def plot_counter(axis: Any, counter: dict[str, int], ground_truth: str, title: str) -> None:
    labels = [label for label in counter if label != "invalid"]
    if "invalid" in counter:
        labels.append("invalid")
    values = [counter[label] for label in labels]
    bars = axis.bar(np.arange(len(labels)), values, color=colors(labels, ground_truth), width=0.65)
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_ylim(0, 8.8)
    axis.set_ylabel("Votes (N=8)")
    axis.set_title(title, fontsize=12)
    axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.15, str(value), ha="center", fontsize=10)


def write_sample_csv(path: Path, ours: list[dict[str, Any]], ttrv: list[dict[str, Any]]) -> None:
    rows = []
    for method, samples in (("ours", ours), ("ttrv", ttrv)):
        for sample in samples:
            rows.append(
                {
                    "method": method,
                    "sample_index": sample["sample_index"],
                    "prediction": sample["prediction"],
                    "correct": sample["correct"],
                    "reward": sample["reward"],
                    "response_token_len": sample["response_token_len"],
                    "response_raw": sample["response_raw"],
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_case(row: dict[str, Any], index: int) -> dict[str, Any]:
    qid = str(row["question_id"])
    folder = OUTPUT / f"case_{index:02d}_{qid}"
    folder.mkdir(parents=True, exist_ok=True)

    source_image = Path(row["image_path"])
    copied_image = folder / f"original_image{source_image.suffix.lower()}"
    shutil.copy2(source_image, copied_image)
    ours_counter = normalized_counter(row["ours_prediction_counter"], row["options"])
    ttrv_counter = normalized_counter(row["ttrv_prediction_counter"], row["options"])

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    image = Image.open(source_image).convert("RGB")
    axes[0].imshow(image)
    axes[0].set_axis_off()
    options = " | ".join(f"{label}: {text}" for label, text in row["options"].items())
    axes[0].set_title(
        f"{row['dataset']} / {qid}\n{row['question']}\n{options}\n"
        f"GT: {row['ground_truth']} ({row['ground_truth_option_text']})",
        fontsize=10.5,
        loc="left",
    )
    plot_counter(
        axes[1],
        ours_counter,
        row["ground_truth"],
        f"Ours: {row['ours_prediction']} (correct)",
    )
    plot_counter(
        axes[2],
        ttrv_counter,
        row["ground_truth"],
        f"TTRV: {row['ttrv_prediction']} (wrong)",
    )
    figure.suptitle("Visual-grounding case under the same final majority@8 protocol", fontsize=14, fontweight="bold")
    figure.tight_layout(w_pad=2.0)
    figure.savefig(folder / "vote_comparison.png", dpi=220, bbox_inches="tight")
    figure.savefig(folder / "vote_comparison.pdf", bbox_inches="tight")
    plt.close(figure)

    labels = sorted(row["options"])
    if "invalid" in ours_counter or "invalid" in ttrv_counter:
        labels.append("invalid")
    table_rows = [
        {
            "answer": label,
            "option_text": row["options"].get(label, "parser invalid"),
            "ours_votes": ours_counter.get(label, 0),
            "ttrv_votes": ttrv_counter.get(label, 0),
            "ground_truth": "yes" if label == row["ground_truth"] else "",
        }
        for label in labels
    ]
    with (folder / "vote_distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    markdown = [
        "| Answer | Option text | Ours votes | TTRV votes | GT |",
        "|---|---|---:|---:|:---:|",
    ]
    for item in table_rows:
        markdown.append(
            f"| {item['answer']} | {item['option_text']} | {item['ours_votes']} "
            f"| {item['ttrv_votes']} | {item['ground_truth']} |"
        )
    (folder / "vote_distribution.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    write_sample_csv(folder / "all_16_responses.csv", row["ours_samples"], row["ttrv_samples"])

    option_lines = "\n".join(f"- {label}: {text}" for label, text in row["options"].items())
    report = f"""# Obs3 Visual-Grounding Case

## Question

{row['question']}

{option_lines}

Ground truth: **{row['ground_truth']} ({row['ground_truth_option_text']})**

## Result

- Protocol: `{row['protocol']}`
- Ours: `{row['ours_prediction']}` (correct), votes `{json.dumps(ours_counter, ensure_ascii=False)}`
- TTRV: `{row['ttrv_prediction']}` (wrong), votes `{json.dumps(ttrv_counter, ensure_ascii=False)}`
- TTRV error type: `{row['ttrv_error_type']}`

## Provenance

- Ours validation groups: `{row['ours_source']}`
- TTRV validation groups: `{row['ttrv_source']}`
- Source image: `{row['image_path']}`

Both methods are compared under the same final majority@8 protocol. Invalid responses remain visible as an explicit category and are not silently reassigned.
"""
    (folder / "case_report.md").write_text(report, encoding="utf-8")
    (folder / "case_metadata.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "folder": folder.name,
        "dataset": row["dataset"],
        "question_id": qid,
        "ground_truth": row["ground_truth"],
        "ours_prediction": row["ours_prediction"],
        "ttrv_prediction": row["ttrv_prediction"],
        "ours_counter": ours_counter,
        "ttrv_counter": ttrv_counter,
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = [render_case(row, index) for index, row in enumerate(read_cases(), start=1)]
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = """# Obs3 Visual-Grounding Cases

Three closed-choice cases where Ours is correct and TTRV is wrong under the same final majority@8 protocol. Each case folder contains the source image, full question/options, all 16 raw responses, vote tables, provenance paths, and PNG/PDF figures.

The cases cover spatial relation, visual numerical reasoning, and diagram counting. They are qualitative examples rather than an aggregate causal estimate.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"cases": len(manifest), "output": str(OUTPUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
