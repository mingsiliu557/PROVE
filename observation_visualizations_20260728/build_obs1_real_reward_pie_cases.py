#!/usr/bin/env python3
"""Build CPU-only Obs1 figures from an already-completed feature-density run."""

from __future__ import annotations

import os as _anon_os
_ANON_DATASET_ROOT = _anon_os.environ.get('DATASET_ROOT', '/path/to/datasets')


import argparse
import csv
import json
import math
import shutil
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path(
    f'{_ANON_DATASET_ROOT}/experiments/'
    "internvl3_obs_overnight_20260730_041611/stages/"
    "10_obs1_prove_selection32/output/selection_groups.jsonl"
)
DEFAULT_OUTPUT = ROOT / "obs1_real_reward_pie_cases_20260730"

TOTAL_ROLLOUTS = 32
LONG_ANSWER_MAX_CHARS = 28
LONG_ANSWER_MAX_WORDS = 5

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
GRAY = "#8A8F98"
LIGHT_GRAY = "#F1F3F4"
INK = "#202124"
PIE_COLORS = [
    "#0072B2",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#E69F00",
    "#CC79A7",
    "#7A5195",
    "#3B8EA5",
    "#8CB369",
    "#D55E00",
    "#4C78A8",
    "#72B7B2",
]


# Membership is exact after parser normalization. Each list was manually
# audited against the image, question, official answers and all 32 predictions.
# Broader or ambiguous answers are deliberately excluded.
CURATED_CASES: list[dict[str, Any]] = [
    {
        "qid": "VZg6oeEVpweNk9fihBXS8m",
        "slug": "case_01_checkerboard",
        "recommended": True,
        "variants": (
            "checkerboard",
            "black and white checkered",
            "checkered",
            "floor has checkered pattern",
            "black and white checkered pattern",
            "floor has checkerboard pattern with alternating black and white tiles",
        ),
    },
    {
        "qid": "8Q69Bei8UReA69MSuQAcLT",
        "slug": "case_02_photograph",
        "recommended": True,
        "variants": (
            "photo",
            "photograph",
            "picture with his camera",
            "photographs",
            "picture",
            "man is taking picture",
            "photos",
            "picture of himself",
            "man is taking photograph with camera",
            "man is taking photograph",
        ),
    },
    {
        "qid": "eBgzKWWRWbkzRJsrfBqGBV",
        "slug": "case_03_transportation",
        "recommended": True,
        "variants": (
            "primary purpose for this train is to provide public transportation",
            "primary purpose for this train is to transport passengers",
            "to transport people",
            "public transportation",
            "commuter service",
            "commuter or regional transportation",
            "public transport",
            "urban transportation",
            "public transit",
            "primary purpose of train is to facilitate public transportation",
            "transit",
            "transportation",
            "transport",
            "primary purpose for this train is to provide public transportation to general population",
            "primary purpose for this train is to transport people",
        ),
    },
    {
        "qid": "8LVHUcGX7daYbXPh59c8pW",
        "slug": "case_04_plane",
        "recommended": True,
        "variants": (
            "fighter jet",
            "military fighter jet",
            "jet",
            "airplane",
            "aircraft",
            "us air force jet fighter",
            "aircraft usaf united states air force jet",
            "military jet",
            "jumbo jet",
            "jet plane",
        ),
    },
    {
        "qid": "7hKGdKpiEqfqB8dEeAw9hu",
        "slug": "case_05_rain",
        "recommended": True,
        "variants": (
            "rainy",
            "rainy weather",
            "answer rainy",
            "rain",
            "raining",
            "evening rainy weather",
            "it was rainy",
            "there was rainy weather",
            "night rain",
        ),
    },
    {
        "qid": "Z5xx32ZiWMMosQmfjqKaGC",
        "slug": "case_06_hotel_room",
        "recommended": True,
        "variants": (
            "this is hotel room",
            "this appears to be hotel room",
            "room in hotel",
        ),
    },
    {
        "qid": "AYNy264QNVzJ6rgeCkW4SU",
        "slug": "case_07_throw_ball",
        "recommended": False,
        "variants": (
            "throw",
            "pitch ball",
            "pitch",
            "throw ball",
            "throw baseball",
            "throw pitch",
        ),
    },
    {
        "qid": "E7LafAUEeJM9NiMjemXwRW",
        "slug": "case_08_zebras",
        "recommended": False,
        "variants": (
            "zebras",
            "zebra",
            "animal is zebra",
            "animals on grass are zebras",
            "type of animals on grass are zebras",
            "three zebra",
            "three zebras",
        ),
    },
    {
        "qid": "8KqQoveNzvvZSgxxPeepUY",
        "slug": "case_09_bus",
        "recommended": False,
        "variants": (
            "bus",
            "red bus is vehicle closest to rectangular blue sign",
            "red bus",
            "red public bus",
            "two red busses",
            "red double decker bus",
        ),
    },
    {
        "qid": "YibQmkwXJ4TPVqZjhqmV9r",
        "slug": "case_10_powdered_sugar",
        "recommended": False,
        "variants": (
            "white powder sprinkled over dessert is likely powdered sugar",
            "white powder sprinkled over dessert is powdered sugar",
            "white powder is powdered sugar",
            "dessert is sprinkled with powdered sugar",
            "it is white powdered sugar",
        ),
    },
    {
        "qid": "QU5GGV7TZU2xWvdPz946Ka",
        "slug": "case_11_bike",
        "recommended": False,
        "variants": ("bicycle", "bike", "bicycles"),
    },
    {
        "qid": "ec8EjipayMWjJLnwsWsrD3",
        "slug": "case_12_surfboards",
        "recommended": False,
        "variants": (
            "surfboards",
            "part of same image people in both photos are using surfboards",
            "they are using surfboards",
            "people are using surfboards",
            "surf boards",
            "surfboard",
        ),
    },
    {
        "qid": "S2UN3nzSfNpeyBMH2GUyCk",
        "slug": "case_13_dining_plane",
        "recommended": False,
        "variants": (
            "airplane",
            "this person appears to be dining on airplane",
            "this person appears to be dining on flight",
            "this person appears to be dining on commercial airplane",
            "person appears to be dining on airplane",
            "based on contents and presentation it seems person is dining on airplane",
            "in airplane",
            "in plane",
            "this person appears to be dining in airplane",
        ),
    },
    {
        "qid": "QJvRFzyGr6FEDrTqLohf7D",
        "slug": "case_14_hydrants",
        "recommended": False,
        "variants": (
            "hydrants",
            "fire hydrants",
            "fire hydrant",
            "four fire hydrants",
            "water hydrant",
        ),
    },
    {
        "qid": "Zpnu9g7eUGf6yYwwFN2Hk5",
        "slug": "case_15_cakes",
        "recommended": False,
        "variants": (
            "cake",
            "cakes",
            "answer cakes",
            "birthday cakes",
            "caption does not provide specific information on type of food but it mentions that it is decorated cakes for sale",
        ),
    },
    {
        "qid": "2DQex53EkNGH2cfo3WPuPn",
        "slug": "case_16_phone",
        "recommended": False,
        "variants": (
            "device sitting next to laptop is mobile phone",
            "cell phone",
            "cellphone",
            "device sitting next to laptop is vodafone camera phone",
            "flip phone",
            "device sitting next to laptop is vodafone cell phone",
            "device sitting next to laptop is flip phone",
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def normalize(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_question(prompt: str) -> str:
    value = prompt
    if "</img>" in value:
        value = value.split("</img>", 1)[1]
    value = value.replace("<|im_end|>", "").replace("<|im_start|>assistant", "")
    value = value.split("Answer with a short direct answer.", 1)[0]
    return " ".join(value.split())


def is_long_answer(answer: str) -> bool:
    return (
        len(answer) > LONG_ANSWER_MAX_CHARS
        or len(answer.split()) > LONG_ANSWER_MAX_WORDS
    )


def display_buckets(counts: Counter[str]) -> list[dict[str, Any]]:
    short_rows: list[dict[str, Any]] = []
    long_rows: list[tuple[str, int]] = []
    for answer, count in counts.items():
        if is_long_answer(answer):
            long_rows.append((answer, int(count)))
        else:
            short_rows.append(
                {
                    "label": answer,
                    "count": int(count),
                    "answers": [answer],
                    "is_others": False,
                }
            )
    short_rows.sort(key=lambda row: (-row["count"], row["label"]))
    if long_rows:
        long_rows.sort(key=lambda item: (-item[1], item[0]))
        short_rows.append(
            {
                "label": "Others",
                "count": sum(count for _, count in long_rows),
                "answers": [answer for answer, _ in long_rows],
                "is_others": True,
                "variant_count": len(long_rows),
            }
        )
    return short_rows


def configure_plotting() -> None:
    return None
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def analyze_case(
    spec: dict[str, Any],
    group: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    samples = group["samples"]
    if len(samples) != TOTAL_ROLLOUTS:
        raise ValueError(
            f"{spec['qid']}: expected {TOTAL_ROLLOUTS} samples, got {len(samples)}"
        )

    semantic_set = {normalize(value) for value in spec["variants"]}
    all_counter = Counter(normalize(sample["prediction"]) for sample in samples)
    missing = sorted(answer for answer in semantic_set if all_counter[answer] == 0)
    if missing:
        raise ValueError(f"{spec['qid']}: curated variants absent from run: {missing}")

    valid_samples = [
        sample for sample in samples if normalize(sample["prediction"]) != "unknown"
    ]
    invalid_count = len(samples) - len(valid_samples)
    logged_invalid = int(group["invalid_count"])
    if invalid_count != logged_invalid:
        raise ValueError(
            f"{spec['qid']}: invalid mismatch, "
            f"derived={invalid_count}, logged={logged_invalid}"
        )
    n_valid = len(valid_samples)
    if n_valid <= 0:
        raise ValueError(f"{spec['qid']}: no valid responses")

    valid_counter = Counter(normalize(sample["prediction"]) for sample in valid_samples)
    semantic_samples = [
        sample
        for sample in valid_samples
        if normalize(sample["prediction"]) in semantic_set
    ]
    if not semantic_samples:
        raise ValueError(f"{spec['qid']}: no GT-similar responses")

    semantic_counter = Counter(
        normalize(sample["prediction"]) for sample in semantic_samples
    )
    m = len(semantic_samples)
    f_count = len(semantic_counter)
    frequency_values = [
        valid_counter[normalize(sample["prediction"])] / n_valid
        for sample in semantic_samples
    ]
    base_values = [float(sample["base_reward"]) for sample in semantic_samples]
    final_values = [
        float(sample["final_selection_reward"]) for sample in semantic_samples
    ]
    visual_bonus_values = [
        float(sample.get("visual_dep_bonus") or 0.0) for sample in semantic_samples
    ]

    frequency_mean = sum(frequency_values) / m
    base_mean = sum(base_values) / m
    final_mean = sum(final_values) / m
    visual_bonus_mean = sum(visual_bonus_values) / m

    # Candidate-weighted exact-frequency mean:
    # sum_a n_a^2 / (N_valid * m).
    closed_form_frequency = (
        sum(count * count for count in semantic_counter.values()) / (n_valid * m)
    )
    if not math.isclose(
        frequency_mean, closed_form_frequency, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AssertionError(f"{spec['qid']}: frequency formula mismatch")

    by_prediction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in semantic_samples:
        by_prediction[normalize(sample["prediction"])].append(sample)

    variant_rows: list[dict[str, Any]] = []
    for answer, count in semantic_counter.most_common():
        answer_samples = by_prediction[answer]
        base_answer = [float(sample["base_reward"]) for sample in answer_samples]
        final_answer = [
            float(sample["final_selection_reward"]) for sample in answer_samples
        ]
        variant_rows.append(
            {
                "normalized_answer": answer,
                "count": int(count),
                "share_within_gt_similar": count / m,
                "frequency_reward": valid_counter[answer] / n_valid,
                "feature_density_base_reward_mean": sum(base_answer)
                / len(base_answer),
                "feature_density_base_reward_min": min(base_answer),
                "feature_density_base_reward_max": max(base_answer),
                "full_prove_reward_mean": sum(final_answer) / len(final_answer),
                "displayed_as_others": is_long_answer(answer),
            }
        )

    buckets = display_buckets(semantic_counter)
    question = clean_question(str(group["prompt"]))
    extra = group["extra_info"]
    image_path = Path(extra["image_path"])
    if not image_path.is_file():
        raise FileNotFoundError(f"{spec['qid']}: missing image {image_path}")

    rollout_rows: list[dict[str, Any]] = []
    for sample in samples:
        prediction = normalize(sample["prediction"])
        valid = prediction != "unknown"
        gt_similar = valid and prediction in semantic_set
        rollout_rows.append(
            {
                "sample_index": int(sample["sample_index"]),
                "response_raw": sample["response_raw"],
                "prediction": prediction,
                "valid": valid,
                "gt_similar": gt_similar,
                "frequency_reward": (
                    valid_counter[prediction] / n_valid if valid else 0.0
                ),
                "feature_density_base_reward": float(sample["base_reward"]),
                "visual_dep_bonus": float(sample.get("visual_dep_bonus") or 0.0),
                "full_prove_reward": float(sample["final_selection_reward"]),
            }
        )

    return {
        "case_slug": spec["slug"],
        "recommended_semantic_clarity": bool(spec["recommended"]),
        "question_id": spec["qid"],
        "question": question,
        "ground_truth": normalize(group["ground_truth"]),
        "official_answers": list(extra.get("official_answers", [])),
        "source_jsonl": str(source_path),
        "source_image": str(image_path),
        "n_total": len(samples),
        "n_valid": n_valid,
        "n_invalid": invalid_count,
        "n_wrong_valid": n_valid - m,
        "gt_similar_mass_m": m,
        "gt_similar_variant_count_f": f_count,
        "gt_similar_ratio_among_valid": m / n_valid,
        "frequency_correct_candidate_mean": frequency_mean,
        "feature_density_correct_candidate_mean": base_mean,
        "density_minus_frequency": base_mean - frequency_mean,
        "full_prove_correct_candidate_mean": final_mean,
        "visual_dep_bonus_correct_candidate_mean": visual_bonus_mean,
        "reward_selection_prediction": normalize(
            group["reward_selection_prediction"]
        ),
        "reward_selection_score": float(group["reward_selection_score"]),
        "majority_prediction": normalize(group["majority_prediction"]),
        "pie_buckets": buckets,
        "variant_rows": variant_rows,
        "rollout_rows": rollout_rows,
        "all_valid_prediction_counter": dict(valid_counter),
        "gt_similar_prediction_counter": dict(semantic_counter),
        "membership_rule": (
            "Conservative frozen exact membership after parser normalization; "
            "each listed string was manually checked to entail the GT."
        ),
        "others_rule": (
            f"Only GT-similar answers longer than {LONG_ANSWER_MAX_CHARS} "
            f"characters or {LONG_ANSWER_MAX_WORDS} whitespace tokens are "
            "merged for display."
        ),
        "frequency_formula": "sum_a(n_a^2)/(N_valid*m)",
        "ours_formula": "mean(base_reward_i over curated GT-similar candidates)",
        "f_definition": (
            "number of distinct normalized GT-similar surface forms"
        ),
    }


def copy_original_image(case_dir: Path, source: Path) -> Path:
    suffix = source.suffix.lower() or ".jpg"
    destination = case_dir / f"original_image{suffix}"
    shutil.copy2(source, destination)
    return destination


def _render_case_matplotlib_unavailable(
    record: dict[str, Any],
    copied_image: Path,
    case_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    configure_plotting()
    fig = plt.figure(figsize=(16.8, 5.8))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.08, 1.30, 0.92],
        left=0.035,
        right=0.985,
        top=0.82,
        bottom=0.17,
        wspace=0.30,
    )
    ax_image = fig.add_subplot(grid[0, 0])
    ax_pie = fig.add_subplot(grid[0, 1])
    ax_table = fig.add_subplot(grid[0, 2])

    with Image.open(copied_image) as image:
        ax_image.imshow(ImageOps.exif_transpose(image).convert("RGB"))
    ax_image.set_xticks([])
    ax_image.set_yticks([])
    ax_image.set_title("Original image", loc="left", fontweight="bold", color=INK)
    for spine in ax_image.spines.values():
        spine.set_color("#DADCE0")
        spine.set_linewidth(0.8)

    buckets = record["pie_buckets"]
    values = [int(bucket["count"]) for bucket in buckets]
    colors = [
        (GRAY if bucket["is_others"] else PIE_COLORS[index % len(PIE_COLORS)])
        for index, bucket in enumerate(buckets)
    ]
    wedges, _ = ax_pie.pie(
        values,
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops={"width": 0.39, "edgecolor": "white", "linewidth": 1.4},
    )
    ax_pie.text(
        0,
        0.05,
        f"{record['gt_similar_mass_m']}/{record['n_valid']}",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    ax_pie.text(
        0,
        -0.13,
        "GT-similar\nvalid answers",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#5F6368",
    )
    legend_labels = []
    for bucket in buckets:
        if bucket["is_others"]:
            label = (
                f"Others: {bucket['variant_count']} long variants "
                f"(n={bucket['count']})"
            )
        else:
            label = f"{bucket['label']} (n={bucket['count']})"
        legend_labels.append(label)
    ax_pie.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.98, 0.50),
        frameon=False,
        fontsize=8.1,
        handlelength=1.0,
        labelspacing=0.72,
    )
    ax_pie.set_title(
        "GT-similar answer variants",
        loc="left",
        fontweight="bold",
        color=INK,
    )

    ax_table.axis("off")
    ax_table.set_title(
        "Correct-answer reward",
        loc="left",
        fontweight="bold",
        color=INK,
        pad=12,
    )
    table_rows = [
        ["Frequency", f"{record['frequency_correct_candidate_mean']:.4f}"],
        [
            "Ours (feature density)",
            f"{record['feature_density_correct_candidate_mean']:.4f}",
        ],
        ["F (# variants)", str(record["gt_similar_variant_count_f"])],
    ]
    table = ax_table.table(
        cellText=table_rows,
        colLabels=["Metric", "Value"],
        colWidths=[0.68, 0.32],
        cellLoc="left",
        colLoc="left",
        loc="upper center",
        bbox=[0.0, 0.43, 1.0, 0.47],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(2.0)
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(LIGHT_GRAY if row % 2 else "#E8F1F8")
            if col == 1:
                cell.get_text().set_fontweight("bold")
                if row == 1:
                    cell.get_text().set_color(ORANGE)
                elif row == 2:
                    cell.get_text().set_color(BLUE)
    ax_table.text(
        0.0,
        0.34,
        f"Feature-density gain: {record['density_minus_frequency']:+.4f}",
        transform=ax_table.transAxes,
        fontsize=10,
        fontweight="bold",
        color=GREEN if record["density_minus_frequency"] >= 0 else RED,
    )
    ax_table.text(
        0.0,
        0.25,
        (
            "Full PROVE mean (with visual bonus): "
            f"{record['full_prove_correct_candidate_mean']:.4f}"
        ),
        transform=ax_table.transAxes,
        fontsize=8.5,
        color="#5F6368",
    )
    ax_table.text(
        0.0,
        0.16,
        (
            f"Omitted from pie: wrong-valid={record['n_wrong_valid']}, "
            f"invalid={record['n_invalid']}"
        ),
        transform=ax_table.transAxes,
        fontsize=8.5,
        color="#5F6368",
    )
    ax_table.text(
        0.0,
        0.02,
        "F counts distinct normalized forms in the\n"
        "curated GT-similar set before Others merging.",
        transform=ax_table.transAxes,
        fontsize=8.2,
        color="#5F6368",
        va="bottom",
    )

    fig.suptitle(
        f"Obs1 case: {record['question']}",
        x=0.035,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.035,
        0.865,
        (
            f"GT: {record['ground_truth']}   |   "
            f"same {record['n_total']} step-0 real-image rollouts   |   "
            f"question ID: {record['question_id']}"
        ),
        ha="left",
        fontsize=9.5,
        color="#5F6368",
    )
    fig.text(
        0.035,
        0.055,
        (
            "The donut is conditional on curated GT-similar valid responses; "
            "its slices do not represent all 32 rollouts. "
            "Others contains only long GT-similar answers."
        ),
        ha="left",
        fontsize=8.6,
        color="#5F6368",
    )
    fig.text(
        0.035,
        0.026,
        (
            "Frequency and Ours are candidate-weighted means on the same "
            "GT-similar subset. Ours uses logged base_reward; "
            "visual-dependency bonus is excluded."
        ),
        ha="left",
        fontsize=8.6,
        color="#5F6368",
    )

    png_path = case_dir / f"{record['case_slug']}_gt_similar_pie.png"
    pdf_path = case_dir / f"{record['case_slug']}_gt_similar_pie.pdf"
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        (
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/dejavu/DejaVuSans.ttf"
        ),
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _wrap_by_pixels(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width = draw.textbbox((0, 0), candidate, font=font)[2]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    spacing: int = 8,
) -> int:
    x, y = position
    lines = _wrap_by_pixels(draw, text, font, max_width)
    line_height = draw.textbbox((0, 0), "Ag", font=font)[3]
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height + spacing
    return y


def render_case(
    record: dict[str, Any],
    copied_image: Path,
    case_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    """Render the three-panel figure with Pillow only."""
    width, height = 5040, 1740
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    font_title = _load_font(48, bold=True)
    font_subtitle = _load_font(29)
    font_panel = _load_font(34, bold=True)
    font_body = _load_font(29)
    font_body_bold = _load_font(29, bold=True)
    font_small = _load_font(24)
    font_center_big = _load_font(58, bold=True)
    font_center_small = _load_font(23)
    font_table_header = _load_font(28, bold=True)
    font_table_value = _load_font(32, bold=True)

    _draw_wrapped(
        draw,
        (105, 42),
        f"Obs1 case: {record['question']}",
        font_title,
        INK,
        4800,
        spacing=4,
    )
    draw.text(
        (105, 137),
        (
            f"GT: {record['ground_truth']}   |   "
            f"same {record['n_total']} step-0 real-image rollouts   |   "
            f"question ID: {record['question_id']}"
        ),
        font=font_subtitle,
        fill="#5F6368",
    )

    image_box = (105, 275, 1455, 1395)
    pie_box = (1555, 275, 3395, 1395)
    table_box = (3515, 275, 4935, 1395)
    for box in (image_box, pie_box, table_box):
        draw.rounded_rectangle(
            box,
            radius=24,
            fill="#FFFFFF",
            outline="#DADCE0",
            width=3,
        )

    draw.text((150, 315), "Original image", font=font_panel, fill=INK)
    target_box = (
        image_box[0] + 45,
        image_box[1] + 100,
        image_box[2] - 45,
        image_box[3] - 45,
    )
    with Image.open(copied_image) as source:
        source_rgb = ImageOps.exif_transpose(source).convert("RGB")
        fitted = ImageOps.contain(
            source_rgb,
            (
                target_box[2] - target_box[0],
                target_box[3] - target_box[1],
            ),
            Image.Resampling.LANCZOS,
        )
    image_x = target_box[0] + (
        target_box[2] - target_box[0] - fitted.width
    ) // 2
    image_y = target_box[1] + (
        target_box[3] - target_box[1] - fitted.height
    ) // 2
    canvas.paste(fitted, (image_x, image_y))

    draw.text(
        (1600, 315),
        "GT-similar answer variants",
        font=font_panel,
        fill=INK,
    )
    buckets = record["pie_buckets"]
    total = sum(int(bucket["count"]) for bucket in buckets)
    center_x, center_y, radius = 2135, 810, 365
    bounds = (
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
    )
    start_angle = -90.0
    wedge_colors: list[str] = []
    for index, bucket in enumerate(buckets):
        color = (
            GRAY
            if bucket["is_others"]
            else PIE_COLORS[index % len(PIE_COLORS)]
        )
        wedge_colors.append(color)
        sweep = 360.0 * int(bucket["count"]) / total
        draw.pieslice(
            bounds,
            start=start_angle,
            end=start_angle + sweep,
            fill=color,
            outline="white",
            width=4,
        )
        start_angle += sweep
    inner_radius = 205
    draw.ellipse(
        (
            center_x - inner_radius,
            center_y - inner_radius,
            center_x + inner_radius,
            center_y + inner_radius,
        ),
        fill="white",
    )
    center_text = (
        f"{record['gt_similar_mass_m']}/{record['n_valid']}"
    )
    center_bounds = draw.textbbox(
        (0, 0), center_text, font=font_center_big
    )
    draw.text(
        (
            center_x - (center_bounds[2] - center_bounds[0]) // 2,
            center_y - 62,
        ),
        center_text,
        font=font_center_big,
        fill=INK,
    )
    for offset, line in enumerate(("GT-similar", "valid answers")):
        line_bounds = draw.textbbox(
            (0, 0), line, font=font_center_small
        )
        draw.text(
            (
                center_x - (line_bounds[2] - line_bounds[0]) // 2,
                center_y + 25 + offset * 31,
            ),
            line,
            font=font_center_small,
            fill="#5F6368",
        )

    legend_x, legend_y = 2550, 435
    legend_step = 75 if len(buckets) <= 9 else 62
    for index, bucket in enumerate(buckets):
        y = legend_y + index * legend_step
        color = wedge_colors[index]
        draw.rounded_rectangle(
            (legend_x, y + 5, legend_x + 32, y + 37),
            radius=5,
            fill=color,
        )
        if bucket["is_others"]:
            label = (
                f"Others: {bucket['variant_count']} long variants "
                f"(n={bucket['count']})"
            )
        else:
            label = f"{bucket['label']} (n={bucket['count']})"
        draw.text(
            (legend_x + 49, y),
            label,
            font=font_small,
            fill=INK,
        )

    draw.text(
        (3560, 315),
        "Correct-answer reward",
        font=font_panel,
        fill=INK,
    )
    table_x0, table_x1 = 3560, 4890
    split_x = 4450
    table_y = 425
    header_h = 82
    row_h = 126
    draw.rectangle(
        (table_x0, table_y, table_x1, table_y + header_h),
        fill=INK,
    )
    draw.text(
        (table_x0 + 24, table_y + 21),
        "Metric",
        font=font_table_header,
        fill="white",
    )
    draw.text(
        (split_x + 24, table_y + 21),
        "Value",
        font=font_table_header,
        fill="white",
    )
    table_rows = [
        (
            "Frequency",
            f"{record['frequency_correct_candidate_mean']:.4f}",
            ORANGE,
        ),
        (
            "Ours (feature density)",
            f"{record['feature_density_correct_candidate_mean']:.4f}",
            BLUE,
        ),
        (
            "F (# variants)",
            str(record["gt_similar_variant_count_f"]),
            INK,
        ),
    ]
    for index, (label, value, value_color) in enumerate(table_rows):
        y0 = table_y + header_h + index * row_h
        y1 = y0 + row_h
        fill = LIGHT_GRAY if index % 2 == 0 else "#E8F1F8"
        draw.rectangle((table_x0, y0, table_x1, y1), fill=fill)
        draw.line((split_x, y0, split_x, y1), fill="white", width=4)
        draw.text(
            (table_x0 + 24, y0 + 42),
            label,
            font=font_body,
            fill=INK,
        )
        draw.text(
            (split_x + 24, y0 + 36),
            value,
            font=font_table_value,
            fill=value_color,
        )

    gain_color = (
        GREEN if record["density_minus_frequency"] >= 0 else RED
    )
    draw.text(
        (3560, 930),
        (
            "Feature-density gain: "
            f"{record['density_minus_frequency']:+.4f}"
        ),
        font=font_body_bold,
        fill=gain_color,
    )
    draw.text(
        (3560, 1000),
        (
            "Full PROVE mean (with visual bonus): "
            f"{record['full_prove_correct_candidate_mean']:.4f}"
        ),
        font=font_small,
        fill="#5F6368",
    )
    draw.text(
        (3560, 1060),
        (
            f"Omitted from pie: wrong-valid={record['n_wrong_valid']}, "
            f"invalid={record['n_invalid']}"
        ),
        font=font_small,
        fill="#5F6368",
    )
    _draw_wrapped(
        draw,
        (3560, 1140),
        (
            "F counts distinct normalized forms in the curated "
            "GT-similar set before Others merging."
        ),
        font_small,
        "#5F6368",
        1260,
        spacing=6,
    )

    _draw_wrapped(
        draw,
        (105, 1460),
        (
            "The donut is conditional on curated GT-similar valid responses; "
            "its slices do not represent all 32 rollouts. "
            "Others contains only long GT-similar answers."
        ),
        font_small,
        "#5F6368",
        4800,
        spacing=4,
    )
    _draw_wrapped(
        draw,
        (105, 1545),
        (
            "Frequency and Ours are candidate-weighted means on the same "
            "GT-similar subset. Ours uses logged base_reward; "
            "visual-dependency bonus is excluded."
        ),
        font_small,
        "#5F6368",
        4800,
        spacing=4,
    )

    png_path = case_dir / f"{record['case_slug']}_gt_similar_pie.png"
    pdf_path = case_dir / f"{record['case_slug']}_gt_similar_pie.pdf"
    canvas.save(png_path, dpi=(dpi, dpi), optimize=True)
    canvas.save(pdf_path, "PDF", resolution=float(dpi))
    return png_path, pdf_path


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_case_artifacts(
    record: dict[str, Any],
    case_dir: Path,
    copied_image: Path,
    png_path: Path,
    pdf_path: Path,
) -> None:
    variant_fields = [
        "normalized_answer",
        "count",
        "share_within_gt_similar",
        "frequency_reward",
        "feature_density_base_reward_mean",
        "feature_density_base_reward_min",
        "feature_density_base_reward_max",
        "full_prove_reward_mean",
        "displayed_as_others",
    ]
    write_csv(
        case_dir / "semantic_variants.csv",
        record["variant_rows"],
        variant_fields,
    )
    rollout_fields = [
        "sample_index",
        "response_raw",
        "prediction",
        "valid",
        "gt_similar",
        "frequency_reward",
        "feature_density_base_reward",
        "visual_dep_bonus",
        "full_prove_reward",
    ]
    write_csv(
        case_dir / "rollout_rewards.csv",
        record["rollout_rows"],
        rollout_fields,
    )

    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"variant_rows", "rollout_rows"}
    }
    metadata["copied_image"] = str(copied_image)
    metadata["figure_png"] = str(png_path)
    metadata["figure_pdf"] = str(pdf_path)
    (case_dir / "case_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    readme = f"""# {record['case_slug']}

Question: {record['question']}

Ground truth: {record['ground_truth']}

This case is reconstructed without rerunning the model. It uses the same
{record['n_total']} step-0 real-image rollouts from the completed Obs1 PROVE
selection run.

## Main values

- GT-similar valid mass m: {record['gt_similar_mass_m']}/{record['n_valid']}
- Invalid outputs: {record['n_invalid']}
- Wrong valid outputs: {record['n_wrong_valid']}
- Frequency correct-candidate mean: {record['frequency_correct_candidate_mean']:.8f}
- Ours feature-density correct-candidate mean: {record['feature_density_correct_candidate_mean']:.8f}
- Ours minus Frequency: {record['density_minus_frequency']:+.8f}
- F: {record['gt_similar_variant_count_f']} distinct normalized GT-similar forms
- Full PROVE correct-candidate mean: {record['full_prove_correct_candidate_mean']:.8f}

## Interpretation

The pie is normalized only within the curated GT-similar set. Wrong valid
answers and invalid outputs are not pie slices. Others contains only long
answers that are already members of the GT-similar set.

Frequency is sum_a n_a^2/(N_valid*m). Ours is the mean logged base_reward over
the same m candidate instances. The main Ours value therefore isolates the
feature-density Obs1 mechanism. Full PROVE, which also contains the
visual-dependency bonus, is retained in the CSV and metadata but is not used in
the main comparison.

## Files

- {copied_image.name}: copied original image
- {png_path.name}: 300-dpi case figure
- {pdf_path.name}: vector PDF case figure
- semantic_variants.csv: one row per exact GT-similar normalized answer
- rollout_rewards.csv: all 32 raw/parsed responses and reward components
- case_metadata.json: provenance, definitions, counters and metrics
"""
    (case_dir / "README.md").write_text(readme, encoding="utf-8")


def build_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    columns: int,
    cell_width: int = 1450,
) -> None:
    if not image_paths:
        return
    gap = 24
    rendered: list[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            ratio = cell_width / image.width
            target = image.resize(
                (cell_width, max(1, round(image.height * ratio))),
                getattr(Image, "Resampling", Image).LANCZOS,
            )
            rendered.append(target)
    cell_height = max(image.height for image in rendered)
    rows = math.ceil(len(rendered) / columns)
    canvas = Image.new(
        "RGB",
        (
            columns * cell_width + (columns + 1) * gap,
            rows * cell_height + (rows + 1) * gap,
        ),
        (246, 247, 249),
    )
    for index, image in enumerate(rendered):
        row, column = divmod(index, columns)
        x = gap + column * (cell_width + gap)
        y = gap + row * (cell_height + gap)
        canvas.paste(image, (x, y))
    canvas.save(output_path, quality=95)


def summary_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_slug": record["case_slug"],
        "question_id": record["question_id"],
        "recommended_semantic_clarity": record[
            "recommended_semantic_clarity"
        ],
        "question": record["question"],
        "ground_truth": record["ground_truth"],
        "official_answers": " | ".join(
            str(value) for value in record["official_answers"]
        ),
        "n_total": record["n_total"],
        "n_valid": record["n_valid"],
        "n_invalid": record["n_invalid"],
        "n_wrong_valid": record["n_wrong_valid"],
        "gt_similar_mass_m": record["gt_similar_mass_m"],
        "gt_similar_variant_count_f": record[
            "gt_similar_variant_count_f"
        ],
        "frequency_correct_candidate_mean": record[
            "frequency_correct_candidate_mean"
        ],
        "feature_density_correct_candidate_mean": record[
            "feature_density_correct_candidate_mean"
        ],
        "density_minus_frequency": record["density_minus_frequency"],
        "full_prove_correct_candidate_mean": record[
            "full_prove_correct_candidate_mean"
        ],
        "visual_dep_bonus_correct_candidate_mean": record[
            "visual_dep_bonus_correct_candidate_mean"
        ],
        "majority_prediction": record["majority_prediction"],
        "reward_selection_prediction": record[
            "reward_selection_prediction"
        ],
        "reward_selection_score": record["reward_selection_score"],
        "source_image": record["source_image"],
    }


def write_root_readme(
    output_root: Path,
    source_path: Path,
    records: list[dict[str, Any]],
    top_gain: list[dict[str, Any]],
) -> None:
    recommended_lines = "\n".join(
        "- " + record["case_slug"]
        for record in records
        if record["recommended_semantic_clarity"]
    )
    gain_lines = "\n".join(
        "- " + record["case_slug"] for record in top_gain
    )
    if not gain_lines:
        gain_lines = "- 当前候选没有正增益"
    content = f"""# Obs1 real feature-density pie cases

这套图不需要重新跑实验，也没有重新采样。数据来自已完成的真实图片
selection run：

{source_path}

共输出 {len(records)} 个候选 case。语义最干净的 6 个候选是：

{recommended_lines}

按真实 feature-density 相对 frequency 的增益排序，前几个正增益候选是：

{gain_lines}

## 图应该怎么读

1. 左侧是该问题的原图和问题。
2. 中间扇形图只统计与 GT 语义一致的 normalized answer variants。
   扇形图在 C+ 内部归一化，不代表全部 32 个 rollout。
3. 错误但可解析的回答、unknown/invalid 都不进入扇形图；数量在图中另报。
4. 超过 {LONG_ANSWER_MAX_CHARS} 个字符或 {LONG_ANSWER_MAX_WORDS} 个词的
   GT-similar 回答仅在显示时合并为 Others。Others 不含错误答案。
5. Frequency 是同一正确语义集合中每个候选拿到的 exact-string frequency
   reward 的 candidate-weighted mean：

   Y = sum_a n_a^2 / (N_valid * m)

6. Ours 使用 selection_groups.jsonl 中已经保存的真实 base_reward，并在同一
   C+ 候选集合上取 candidate-weighted mean。它是 feature-density 基础奖励；
   不含 visual-dependency bonus。
7. F 是 C+ 中不同 normalized surface forms 的数量，且在合并 Others 前计算。

## 为什么主图不用 final_selection_reward

final_selection_reward = base_reward + visual_dep_bonus。Obs1 要隔离语义
碎片化下的 feature-density reward，所以主表使用 base_reward。完整 PROVE
均值仍保留在 case_summary.csv、semantic_variants.csv 和 case_metadata.json，
可用于补充材料，但不要把它当作纯 Obs1 效应。

## 文件结构

- case_*/：每例的原图、PNG、PDF、逐 variant CSV、32-rollout CSV、元数据和说明。
- recommended_semantic_clarity/：语义边界最干净的 6 例图。
- top_feature_density_gain/：按 Ours-Frequency 排序的正增益例图。
- contact_sheet_recommended_6.png：推荐 6 例总览。
- contact_sheet_all_16.png：全部候选总览。
- case_summary.csv：做论文表格最方便的主表。
- ranking_by_density_gain.csv：按真实 feature-density 增益排序。
- figure_manifest.json：所有文件与 provenance 的机器可读索引。

## 科学边界

- GT-similar membership 是逐题人工保守审计后冻结的 exact normalized string
  集合；没有使用 selection dump 中不可靠的 sample.correct 字段。
- 饼图展示 response wording fragmentation，不是模型对所有答案的概率分布。
- 这些是固定 rollout 的 case analysis，不等同于端到端训练准确率提升。
- 旧包中的理想化 semantic-cluster mass 不能冒充这里的真实 base_reward。

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate + run
- Origin Date: 2026-07-30
- Verification Status: generated from completed real feature-density logs; CPU-only
- Version Label: obs1_real_reward_pie_v1
"""
    (output_root / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_path = args.source_jsonl.resolve()
    output_root = args.output_root.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    groups = read_jsonl(source_path)
    by_qid = {
        str(group["extra_info"]["id"]): group
        for group in groups
    }
    missing_qids = [
        spec["qid"] for spec in CURATED_CASES if spec["qid"] not in by_qid
    ]
    if missing_qids:
        raise KeyError(f"Missing curated qids: {missing_qids}")

    records: list[dict[str, Any]] = []
    case_pngs: list[Path] = []
    recommended_pngs: list[Path] = []
    for spec in CURATED_CASES:
        record = analyze_case(spec, by_qid[spec["qid"]], source_path)
        case_dir = output_root / record["case_slug"]
        case_dir.mkdir(parents=True, exist_ok=False)
        copied_image = copy_original_image(
            case_dir, Path(record["source_image"])
        )
        png_path, pdf_path = render_case(
            record, copied_image, case_dir, args.dpi
        )
        write_case_artifacts(
            record,
            case_dir,
            copied_image,
            png_path,
            pdf_path,
        )
        record["figure_png"] = str(png_path)
        record["figure_pdf"] = str(pdf_path)
        records.append(record)
        case_pngs.append(png_path)
        if record["recommended_semantic_clarity"]:
            recommended_pngs.append(png_path)

    summary_rows = [summary_row(record) for record in records]
    summary_fields = list(summary_rows[0])
    write_csv(output_root / "case_summary.csv", summary_rows, summary_fields)

    ranked = sorted(
        records,
        key=lambda record: (
            -record["density_minus_frequency"],
            -record["gt_similar_variant_count_f"],
            record["case_slug"],
        ),
    )
    ranking_rows = []
    for rank, record in enumerate(ranked, start=1):
        row = {"density_gain_rank": rank, **summary_row(record)}
        ranking_rows.append(row)
    write_csv(
        output_root / "ranking_by_density_gain.csv",
        ranking_rows,
        list(ranking_rows[0]),
    )

    recommended_dir = output_root / "recommended_semantic_clarity"
    recommended_dir.mkdir()
    for record in records:
        if not record["recommended_semantic_clarity"]:
            continue
        shutil.copy2(record["figure_png"], recommended_dir)
        shutil.copy2(record["figure_pdf"], recommended_dir)

    top_gain = [
        record for record in ranked if record["density_minus_frequency"] > 0
    ][:6]
    gain_dir = output_root / "top_feature_density_gain"
    gain_dir.mkdir()
    for record in top_gain:
        shutil.copy2(record["figure_png"], gain_dir)
        shutil.copy2(record["figure_pdf"], gain_dir)

    build_contact_sheet(
        recommended_pngs,
        output_root / "contact_sheet_recommended_6.png",
        columns=1,
    )
    build_contact_sheet(
        case_pngs,
        output_root / "contact_sheet_all_16.png",
        columns=2,
    )

    manifest = {
        "version": "obs1_real_reward_pie_v1",
        "source_jsonl": str(source_path),
        "source_groups": len(groups),
        "generated_cases": len(records),
        "configuration": {
            "total_rollouts_per_case": TOTAL_ROLLOUTS,
            "long_answer_max_chars": LONG_ANSWER_MAX_CHARS,
            "long_answer_max_words": LONG_ANSWER_MAX_WORDS,
            "main_ours_field": "samples[].base_reward",
            "supplemental_full_field": (
                "samples[].final_selection_reward"
            ),
            "reward_style": "density_cluster_answer_entropy",
            "embedding_scope": "evidence_query_mean_pool",
            "density_temperature": {"t0": 0.2, "min": 0.1, "max": 0.3},
            "visual_dependency_bonus_in_main_table": False,
        },
        "cases": [
            {
                **summary_row(record),
                "figure_png": record["figure_png"],
                "figure_pdf": record["figure_pdf"],
            }
            for record in records
        ],
    }
    (output_root / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_root_readme(output_root, source_path, records, top_gain)

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "generated_cases": len(records),
                "positive_density_gain_cases": len(
                    [
                        record
                        for record in records
                        if record["density_minus_frequency"] > 0
                    ]
                ),
                "recommended_semantic_clarity": [
                    record["case_slug"]
                    for record in records
                    if record["recommended_semantic_clarity"]
                ],
                "top_feature_density_gain": [
                    {
                        "case": record["case_slug"],
                        "delta": record["density_minus_frequency"],
                    }
                    for record in top_gain
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
