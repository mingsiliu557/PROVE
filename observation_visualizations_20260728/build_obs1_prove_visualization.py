#!/usr/bin/env python3
"""Build Obs1 cases from full PROVE selection_groups.jsonl records."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
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
    OUT,
    RED,
    configure_plotting,
    short_text,
)


RUN_ROOT = OUT / "prove_selection_runs"


def normalize(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    return " ".join(text.split()) or "unknown"


def clean_prompt(prompt: object) -> str:
    text = re.sub(r"<IMG_CONTEXT>|<[^>]+>", " ", str(prompt or ""))
    return " ".join(text.split()).removeprefix("user").strip()


def latest_groups() -> Path:
    paths = sorted(RUN_ROOT.glob("aokvqa_*/selection_groups.jsonl"), key=lambda path: path.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"No selection_groups.jsonl below {RUN_ROOT}")
    return paths[-1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def final_score(sample: dict) -> float:
    for key in ("final_selection_reward", "selection_reward", "reward"):
        if sample.get(key) is not None:
            return float(sample[key])
    return 0.0


def analyze(group: dict) -> dict:
    samples = list(group.get("samples") or [])
    counts = Counter(normalize(sample.get("prediction")) for sample in samples)
    correct_strings = {
        normalize(sample.get("prediction")) for sample in samples if bool(sample.get("correct"))
    }
    by_answer: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        by_answer[normalize(sample.get("prediction"))].append(sample)
    answer_scores = []
    for answer, answer_samples in by_answer.items():
        best = max(answer_samples, key=final_score)
        answer_scores.append(
            {
                "answer": answer,
                "count": len(answer_samples),
                "correct": any(bool(sample.get("correct")) for sample in answer_samples),
                "base": float(best.get("base_reward") or 0.0),
                "bonus": float(best.get("visual_dep_bonus") or 0.0),
                "final": final_score(best),
            }
        )
    extra = group.get("extra_info") or {}
    majority = normalize(group.get("majority_prediction"))
    selected = normalize(group.get("reward_selection_prediction"))
    return {
        "question_id": str(extra.get("diagnostic_question_id") or extra.get("question_id") or group.get("index")),
        "question": clean_prompt(group.get("prompt")),
        "image_path": str(extra.get("image_path") or extra.get("diagnostic_target_image_path") or ""),
        "official_answers": list(extra.get("official_answers") or []),
        "majority": majority,
        "majority_count": int(group.get("majority_count") or counts.get(majority, 0)),
        "majority_correct": bool(group.get("majority_correct")),
        "prove": selected,
        "prove_correct": bool(group.get("reward_selection_correct")),
        "prove_margin": float(group.get("reward_selection_margin") or 0.0),
        "correct_strings": sorted(correct_strings),
        "correct_mass": sum(counts[value] for value in correct_strings),
        "counts": dict(counts),
        "answer_scores": answer_scores,
    }


def select_cases(groups: list[dict], limit: int) -> tuple[list[dict], dict]:
    rows = [analyze(group) for group in groups]
    cases = [
        row
        for row in rows
        if row["prove_correct"]
        and not row["majority_correct"]
        and row["majority"] != "unknown"
        and len(row["correct_strings"]) >= 2
        and row["correct_mass"] > row["majority_count"]
        and Path(row["image_path"]).exists()
    ]
    cases.sort(
        key=lambda row: (
            len(row["correct_strings"]),
            row["correct_mass"] - row["majority_count"],
            row["prove_margin"],
        ),
        reverse=True,
    )
    audit = {
        "groups": len(rows),
        "prove_correct_majority_wrong": sum(row["prove_correct"] and not row["majority_correct"] for row in rows),
        "prove_correct_valid_majority_wrong": sum(
            row["prove_correct"] and not row["majority_correct"] and row["majority"] != "unknown" for row in rows
        ),
        "strong_fragmentation_cases": len(cases),
    }
    if len(cases) < limit:
        raise RuntimeError(f"Need {limit} strong cases, found {len(cases)}; audit={audit}")
    return cases[:limit], audit


def plot_counts(ax: plt.Axes, case: dict) -> None:
    counter = Counter(case["counts"])
    answers = [answer for answer, _ in counter.most_common(8)]
    for answer in (case["majority"], case["prove"]):
        if answer not in answers:
            answers.append(answer)
    correct = set(case["correct_strings"])
    y = np.arange(len(answers))
    bars = ax.barh(y, [counter[a] for a in answers], color=[GREEN if a in correct else RED for a in answers])
    ax.set_yticks(y, [short_text(answer, 25) for answer in answers])
    ax.invert_yaxis()
    ax.set_xlabel("Exact-string count")
    ax.set_title(
        f"String majority: {case['majority']} x{case['majority_count']} (wrong)\n"
        f"Correct mass: {case['correct_mass']} across {len(correct)} accepted strings",
        loc="left",
    )
    ax.grid(axis="x", alpha=0.2)
    for bar, answer in zip(bars, answers):
        if answer == case["prove"]:
            bar.set_edgecolor(BLUE)
            bar.set_linewidth(2.5)


def plot_scores(ax: plt.Axes, case: dict) -> None:
    rows = sorted(case["answer_scores"], key=lambda row: row["final"], reverse=True)[:8]
    present = {row["answer"] for row in rows}
    rows += [row for row in case["answer_scores"] if row["answer"] in {case["majority"], case["prove"]} - present]
    rows = list({row["answer"]: row for row in rows}.values())
    rows.sort(key=lambda row: row["final"], reverse=True)
    y = np.arange(len(rows))
    for idx, row in enumerate(rows):
        color = GREEN if row["correct"] else RED
        ax.plot([row["base"], row["final"]], [idx, idx], color=LIGHT_GRAY, linewidth=2)
        ax.scatter(row["base"], idx, color=ORANGE, marker="o", s=45, zorder=3)
        selected = row["answer"] == case["prove"]
        ax.scatter(
            row["final"], idx, color=color, marker="*" if selected else "D",
            s=150 if selected else 48, edgecolor=INK, linewidth=0.5, zorder=4,
        )
    ax.set_yticks(y, [short_text(f"{row['answer']} (x{row['count']})", 27) for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Candidate reward")
    ax.set_title(
        f"PROVE selects: {case['prove']} (correct)\n"
        "circle: density base; diamond/star: base + visual bonus",
        loc="left",
    )
    ax.grid(axis="x", alpha=0.2)


def build(groups_path: Path, output_dir: Path, limit: int) -> None:
    cases, audit = select_cases(read_jsonl(groups_path), limit)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    fig, axes = plt.subplots(limit, 3, figsize=(16, 4.3 * limit), gridspec_kw={"width_ratios": [1.05, 1.35, 1.5]})
    axes = np.asarray([axes]) if limit == 1 else axes
    for idx, case in enumerate(cases):
        source = Path(case["image_path"])
        target = image_dir / f"obs1_prove_{idx + 1}_{case['question_id']}{source.suffix.lower()}"
        shutil.copy2(source, target)
        case["copied_image"] = str(target)
        with Image.open(target) as image:
            axes[idx, 0].imshow(image.convert("RGB"))
        axes[idx, 0].set_xticks([])
        axes[idx, 0].set_yticks([])
        refs = ", ".join(str(value) for value in case["official_answers"][:8])
        axes[idx, 0].set_title(
            f"Case {idx + 1}: {short_text(case['question'], 46)}\nAccepted: {short_text(refs, 54)}",
            loc="left",
        )
        plot_counts(axes[idx, 1], case)
        plot_scores(axes[idx, 2], case)
    fig.suptitle(
        "Obs1: Full PROVE selection recovers consensus split across accepted strings",
        fontsize=16, fontweight="bold", color=INK, y=1.01,
    )
    fig.text(
        0.5, 0.002,
        "Same 32 candidates for both selectors. A-OKVQA accepts any normalized human reference. "
        "PROVE = density/temperature base reward + valid-only visual-dependency bonus.",
        ha="center", fontsize=9, color="#555555",
    )
    fig.tight_layout(h_pad=2.4, w_pad=2.0)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"obs1_prove_selection_cases.{suffix}", facecolor="white", dpi=300)
    plt.close(fig)
    payload = {"source": str(groups_path), "audit": audit, "cases": cases}
    (output_dir / "obs1_prove_selection_cases.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = (
        "# Obs1 PROVE Selection Visualization\n\n"
        f"- Source: `{groups_path}`\n"
        f"- Questions: {audit['groups']}\n"
        f"- PROVE correct / majority wrong: {audit['prove_correct_majority_wrong']}\n"
        f"- Valid wrong majority / PROVE correct: {audit['prove_correct_valid_majority_wrong']}\n"
        f"- Strong fragmentation cases: {audit['strong_fragmentation_cases']}\n"
    )
    (output_dir / "obs1_prove_selection_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-groups", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    build(args.selection_groups or latest_groups(), args.output_dir, args.limit)


if __name__ == "__main__":
    main()
