#!/usr/bin/env python3
"""Build four large-gap Obs1 selector case tables from an existing PROVE run.

This script is CPU-only. It does not rerun inference or recompute embeddings.
It reconstructs valid-only TTRV frequency scores from stored normalized
predictions and reads the actual PROVE full selection reward from the stored
rollout records.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path(
    "/path/to/datasets/experiments/"
    "internvl3_obs_overnight_20260730_041611/stages/"
    "10_obs1_prove_selection32/output/selection_groups.jsonl"
)
DEFAULT_OUTPUT = ROOT / "obs1_large_reward_gap_cases_20260730"
DEFAULT_ARCHIVE = ROOT.parent / "obv1_large_reward_gap_cases_20260730.zip"

UNKNOWN = "unknown"
TIE_TOLERANCE = 1e-8

INK = "#202124"
MUTED = "#5F6368"
GRID = "#DADCE0"
NAVY = "#17324D"
BLUE = "#0072B2"
GREEN = "#138A62"
GREEN_BG = "#EAF7F1"
GREEN_BG_LIGHT = "#F4FBF7"
RED = "#C74440"
RED_BG = "#FDEEEE"
GRAY_BG = "#F3F4F6"
WHITE = "#FFFFFF"


# These exact normalized strings were manually audited against the image,
# question, benchmark reference answer, official answer list, and all stored
# rollouts. Only cases where the full PROVE selector is semantically correct
# and the valid-only TTRV frequency selector is semantically wrong are kept.
# Large-gap, manually audited stage-10 cases.
CASE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "qid": "P4cavrtzjFu5Ur3DCjn6YK",
        "slug": "case_01_ferris_wheel_rotates_vs_amusement",
        "evidence_grade": "A-",
        "expected_prove": "rotates",
        "expected_ttrv": "amusement",
        "correct_answers": {
            "go round",
            "rotates",
            "rotate",
            "spin round and round",
            "spin",
        },
        "audit_note": (
            "The question asks what the wheel does (motion); rotates answers "
            "that directly, while amusement describes its purpose."
        ),
    },
    {
        "qid": "5aRkxnYoakzDQFyDGb9qJ2",
        "slug": "case_02_green_vs_wrong_actress",
        "evidence_grade": "A-",
        "expected_prove": "green",
        "expected_ttrv": (
            "green street is named after scarlett o hara from movie gone with wind"
        ),
        "correct_answers": {"green"},
        "audit_note": (
            "The street sign gives the surname Green, matching actress Eva Green; "
            "the TTRV response invents a Scarlett O Hara attribution."
        ),
    },
    {
        "qid": "chXZTHvYp7pFsdQhLyAC9F",
        "slug": "case_03_tennis_vs_soccer",
        "evidence_grade": "A",
        "expected_prove": "tennis",
        "expected_ttrv": "soccer",
        "correct_answers": {
            "tennis",
            "inter school tennis tournament",
        },
        "audit_note": (
            "The image shows a tennis court and rackets; soccer is a "
            "different activity."
        ),
    },
    {
        "qid": "FeTsEoSpyGhqi8ui4i8tTy",
        "slug": "case_04_construction_vs_cranes",
        "evidence_grade": "A-",
        "expected_prove": "for construction",
        "expected_ttrv": "tallest structures are cranes",
        "correct_answers": {"for construction"},
        "audit_note": (
            "The question asks the purpose of the two tallest structures; "
            "for construction answers the purpose, while the TTRV response "
            "only identifies them as cranes."
        ),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_question(prompt: str) -> str:
    question = prompt.split("</img>")[-1]
    question = question.split("Answer with a short direct answer.")[0]
    question = question.replace("<|im_end|>", "").strip()
    return " ".join(question.split())


def reward_value(sample: dict[str, Any]) -> float:
    for field in ("final_selection_reward", "selection_reward", "reward"):
        value = sample.get(field)
        if value is not None:
            return float(value)
    raise KeyError("sample has no full selection reward field")


def first_max_by_score(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("cannot select from an empty sample list")
    maximum = max(reward_value(sample) for sample in sample_list)
    return next(
        sample
        for sample in sample_list
        if maximum - reward_value(sample) <= TIE_TOLERANCE
    )


def select_ttrv(valid_samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not valid_samples:
        raise ValueError("valid-only TTRV selection requires a valid response")
    counts = Counter(str(sample["prediction"]) for sample in valid_samples)
    maximum = max(counts.values())
    return next(
        sample
        for sample in valid_samples
        if counts[str(sample["prediction"])] == maximum
    )


def analyze_case(group: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    samples = list(group["samples"])
    if len(samples) != 32:
        raise AssertionError(f"{spec['qid']}: expected 32 rollouts, got {len(samples)}")

    for position, sample in enumerate(samples):
        if int(sample["sample_index"]) != position:
            raise AssertionError(f"{spec['qid']}: sample_index is not list order")

    valid_samples = [
        sample for sample in samples if str(sample["prediction"]) != UNKNOWN
    ]
    valid_counts = Counter(str(sample["prediction"]) for sample in valid_samples)
    unknown_count = len(samples) - len(valid_samples)

    ttrv_sample = select_ttrv(valid_samples)
    ttrv_answer = str(ttrv_sample["prediction"])

    logged_prove_index = int(group["reward_selection_sample_index"])
    prove_sample = samples[logged_prove_index]
    recomputed_prove_sample = first_max_by_score(samples)
    if int(recomputed_prove_sample["sample_index"]) != logged_prove_index:
        raise AssertionError(f"{spec['qid']}: logged and recomputed PROVE selectors differ")

    prove_answer = str(prove_sample["prediction"])
    prove_reward = reward_value(prove_sample)
    if prove_answer != str(group["reward_selection_prediction"]):
        raise AssertionError(f"{spec['qid']}: logged PROVE prediction mismatch")
    if abs(prove_reward - float(group["reward_selection_score"])) > TIE_TOLERANCE:
        raise AssertionError(f"{spec['qid']}: logged PROVE score mismatch")
    if prove_answer != spec["expected_prove"]:
        raise AssertionError(
            f"{spec['qid']}: expected PROVE={spec['expected_prove']!r}, "
            f"got {prove_answer!r}"
        )
    if ttrv_answer != spec["expected_ttrv"]:
        raise AssertionError(
            f"{spec['qid']}: expected TTRV={spec['expected_ttrv']!r}, "
            f"got {ttrv_answer!r}"
        )

    correct_answers = set(spec["correct_answers"])
    observed_answers = {str(sample["prediction"]) for sample in samples}
    missing = correct_answers - observed_answers
    if missing:
        raise AssertionError(f"{spec['qid']}: configured correct answers absent: {missing}")
    if prove_answer not in correct_answers:
        raise AssertionError(f"{spec['qid']}: PROVE selector is not semantically correct")
    if ttrv_answer in correct_answers:
        raise AssertionError(f"{spec['qid']}: TTRV selector is not semantically wrong")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["prediction"])].append(sample)

    answer_rows: list[dict[str, Any]] = []
    for answer, answer_samples in grouped.items():
        best = first_max_by_score(answer_samples)
        count = valid_counts.get(answer, 0)
        semantic_status = (
            "invalid"
            if answer == UNKNOWN
            else "correct"
            if answer in correct_answers
            else "incorrect"
        )
        selectors: list[str] = []
        if answer == prove_answer:
            selectors.append("PROVE")
        if answer == ttrv_answer:
            selectors.append("TTRV")
        answer_rows.append(
            {
                "answer": answer,
                "count": len(answer_samples) if answer == UNKNOWN else count,
                "semantic_status": semantic_status,
                "ttrv_reward": 0.0 if answer == UNKNOWN else count / len(valid_samples),
                "prove_max_reward": reward_value(best),
                "prove_best_sample_index": int(best["sample_index"]),
                "selected_by": " + ".join(selectors) if selectors else "",
                "prove_selected": answer == prove_answer,
                "ttrv_selected": answer == ttrv_answer,
            }
        )

    semantic_rank = {"correct": 0, "incorrect": 1, "invalid": 2}
    answer_rows.sort(
        key=lambda row: (
            0
            if row["prove_selected"]
            else 1
            if row["ttrv_selected"]
            else 2,
            semantic_rank[row["semantic_status"]],
            -int(row["count"]),
            str(row["answer"]),
        )
    )

    correct_vote_count = sum(valid_counts[answer] for answer in correct_answers)
    if correct_vote_count <= 0:
        raise AssertionError(f"{spec['qid']}: no semantically correct valid answers")

    n_valid = len(valid_samples)
    ttrv_count = valid_counts[ttrv_answer]
    prove_answer_ttrv_count = valid_counts[prove_answer]
    ttrv_answer_prove_reward = max(
        reward_value(sample) for sample in grouped[ttrv_answer]
    )

    return {
        "qid": spec["qid"],
        "slug": spec["slug"],
        "evidence_grade": spec["evidence_grade"],
        "audit_note": spec["audit_note"],
        "question": clean_question(str(group["prompt"])),
        "ground_truth": str(group["ground_truth"]),
        "official_answers": list(group.get("extra_info", {}).get("official_answers", [])),
        "image_path": Path(group["extra_info"]["image_path"]),
        "n_total": len(samples),
        "n_valid": n_valid,
        "n_unknown": unknown_count,
        "prove_answer": prove_answer,
        "prove_sample_index": int(prove_sample["sample_index"]),
        "prove_reward": prove_reward,
        "ttrv_answer": ttrv_answer,
        "ttrv_sample_index": int(ttrv_sample["sample_index"]),
        "ttrv_count": ttrv_count,
        "ttrv_reward": ttrv_count / n_valid,
        "prove_answer_ttrv_count": prove_answer_ttrv_count,
        "prove_answer_ttrv_reward": prove_answer_ttrv_count / n_valid,
        "ttrv_answer_prove_reward": ttrv_answer_prove_reward,
        "answer_rows": answer_rows,
    }


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        Path(
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/dejavu/DejaVuSans.ttf"
        ),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def wrap_by_pixels(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_lines(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str],
    font: ImageFont.ImageFont,
    fill: str,
    line_height: int,
) -> int:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def draw_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    answer: str,
    score: str,
    accent: str,
    body_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    bold_font: ImageFont.ImageFont,
) -> None:
    draw.rounded_rectangle(box, radius=20, fill=WHITE, outline=accent, width=4)
    x0, y0, x1, _ = box
    draw.text((x0 + 24, y0 + 20), title, font=small_font, fill=accent)
    all_answer_lines = wrap_by_pixels(draw, answer, bold_font, x1 - x0 - 48)
    answer_lines = all_answer_lines[:3]
    if len(all_answer_lines) > 3:
        answer_lines[-1] = answer_lines[-1].rstrip(" .") + "..."
    draw_lines(draw, (x0 + 24, y0 + 62), answer_lines, bold_font, INK, 43)
    score_y = y0 + 205 if len(answer_lines) > 2 else y0 + 162 if len(answer_lines) > 1 else y0 + 123
    draw.text((x0 + 24, score_y), score, font=body_font, fill=INK)


def render_reward_table(record: dict[str, Any], output_path: Path) -> None:
    width = 2500
    margin = 70
    columns = (
        ("Normalized answer", 970),
        ("Count", 150),
        ("Semantic audit", 250),
        ("TTRV frequency reward", 330),
        ("PROVE max full reward", 360),
        ("Selected by", 300),
    )
    if sum(column_width for _, column_width in columns) != width - 2 * margin:
        raise AssertionError("table column widths do not match canvas width")

    title_font = load_font(48, bold=True)
    question_font = load_font(36, bold=True)
    body_font = load_font(30)
    body_bold = load_font(30, bold=True)
    small_font = load_font(25)
    small_bold = load_font(25, bold=True)
    table_header_font = load_font(25, bold=True)
    table_font = load_font(27)
    table_bold = load_font(27, bold=True)
    footnote_font = load_font(23)

    measure = ImageDraw.Draw(Image.new("RGB", (10, 10), WHITE))
    question_lines = wrap_by_pixels(
        measure,
        f"Q: {record['question']}",
        question_font,
        width - 2 * margin,
    )
    header_height = 185 + 46 * len(question_lines) + 355
    table_title_height = 78
    table_header_height = 110

    row_layout: list[tuple[dict[str, Any], int, list[str]]] = []
    for row in record["answer_rows"]:
        answer_lines = wrap_by_pixels(
            measure,
            str(row["answer"]),
            table_bold if row["selected_by"] else table_font,
            columns[0][1] - 40,
        )
        row_height = max(72, 38 * len(answer_lines) + 28)
        row_layout.append((row, row_height, answer_lines))

    footnote_height = 175
    height = (
        header_height
        + table_title_height
        + table_header_height
        + sum(row_height for _, row_height, _ in row_layout)
        + footnote_height
        + 40
    )
    canvas = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(canvas)

    y = 45
    draw.text((margin, y), "Obs1 selector reward audit", font=title_font, fill=INK)
    badge = f"Evidence grade {record['evidence_grade']}"
    badge_box = draw.textbbox((0, 0), badge, font=small_bold)
    badge_width = badge_box[2] - badge_box[0] + 42
    draw.rounded_rectangle(
        (width - margin - badge_width, y + 2, width - margin, y + 48),
        radius=20,
        fill=GREEN_BG,
        outline=GREEN,
        width=2,
    )
    draw.text(
        (width - margin - badge_width + 21, y + 10),
        badge,
        font=small_bold,
        fill=GREEN,
    )
    y += 72
    y = draw_lines(draw, (margin, y), question_lines, question_font, INK, 46)
    draw.text(
        (margin, y + 8),
        (
            f"Reference answer: {record['ground_truth']}   |   "
            f"Question ID: {record['qid']}   |   "
            f"rollouts: {record['n_total']} total / {record['n_valid']} valid / "
            f"{record['n_unknown']} unknown"
        ),
        font=small_font,
        fill=MUTED,
    )
    y += 64

    gap = 18
    card_width = (width - 2 * margin - 3 * gap) // 4
    card_height = 260
    cards = [
        (
            "Our selection under PROVE (correct)",
            record["prove_answer"],
            "full reward = {:.6f}".format(record["prove_reward"]),
            GREEN,
        ),
        (
            "Majority selection under PROVE (wrong)",
            record["ttrv_answer"],
            "max full reward = {:.6f}".format(record["ttrv_answer_prove_reward"]),
            RED,
        ),
        (
            "Our selection under TTRV",
            record["prove_answer"],
            "{}/{} = {:.6f}".format(
                record["prove_answer_ttrv_count"],
                record["n_valid"],
                record["prove_answer_ttrv_reward"],
            ),
            BLUE,
        ),
        (
            "Majority selection under TTRV (wrong)",
            record["ttrv_answer"],
            "{}/{} = {:.6f}".format(
                record["ttrv_count"],
                record["n_valid"],
                record["ttrv_reward"],
            ),
            RED,
        ),
    ]

    for index, (title, answer, score, accent) in enumerate(cards):
        x0 = margin + index * (card_width + gap)
        draw_card(
            draw,
            (x0, y, x0 + card_width, y + card_height),
            title,
            str(answer),
            str(score),
            accent,
            body_font,
            small_font,
            body_bold,
        )
    y += card_height + 35

    draw.text((margin, y), "All observed normalized answers", font=question_font, fill=INK)
    draw.text(
        (width - margin - 930, y + 8),
        "PROVE row reward = max final reward among rollouts with that exact answer",
        font=small_font,
        fill=MUTED,
    )
    y += table_title_height

    x = margin
    for title, column_width in columns:
        draw.rectangle((x, y, x + column_width, y + table_header_height), fill=NAVY)
        header_lines = wrap_by_pixels(
            draw,
            title,
            table_header_font,
            column_width - 30,
        )
        draw_lines(
            draw,
            (x + 15, y + 18),
            header_lines,
            table_header_font,
            WHITE,
            34,
        )
        x += column_width
    y += table_header_height

    for row, row_height, answer_lines in row_layout:
        if row["prove_selected"]:
            fill = GREEN_BG
        elif row["ttrv_selected"]:
            fill = RED_BG
        elif row["semantic_status"] == "correct":
            fill = GREEN_BG_LIGHT
        elif row["semantic_status"] == "invalid":
            fill = GRAY_BG
        else:
            fill = WHITE

        values = (
            answer_lines,
            [str(row["count"])],
            [str(row["semantic_status"])],
            (
                ["0.000000 (invalid)"]
                if row["semantic_status"] == "invalid"
                else [f"{row['count']}/{record['n_valid']} = {row['ttrv_reward']:.6f}"]
            ),
            [f"{row['prove_max_reward']:.6f}"],
            [str(row["selected_by"]) if row["selected_by"] else "—"],
        )
        x = margin
        for column_index, ((_, column_width), cell_lines) in enumerate(
            zip(columns, values)
        ):
            draw.rectangle(
                (x, y, x + column_width, y + row_height),
                fill=fill,
                outline=GRID,
                width=2,
            )
            font = table_bold if row["selected_by"] else table_font
            if column_index == 2:
                status_color = (
                    GREEN
                    if row["semantic_status"] == "correct"
                    else MUTED
                    if row["semantic_status"] == "invalid"
                    else RED
                )
            elif column_index == 5 and row["selected_by"]:
                status_color = GREEN if row["prove_selected"] else RED
            else:
                status_color = INK
            text_height = 38 * len(cell_lines)
            cell_y = y + max(12, (row_height - text_height) // 2)
            draw_lines(
                draw,
                (x + 16, cell_y),
                list(cell_lines),
                font,
                status_color,
                38,
            )
            x += column_width
        y += row_height

    y += 28
    footnotes = (
        "TTRV frequency excludes unknown and is computed on exact normalized strings. "
        "All four selected cases are frequency ties resolved by the earliest valid rollout.",
        "PROVE reward is the logged final selector reward. For each exact answer, the table "
        "reports the maximum final reward among rollouts producing that answer.",
        "Audit note: {}".format(record["audit_note"]),
    )

    for footnote in footnotes:
        lines = wrap_by_pixels(draw, footnote, footnote_font, width - 2 * margin)
        y = draw_lines(draw, (margin, y), lines, footnote_font, MUTED, 31) + 5

    canvas.save(output_path, format="PNG", optimize=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def build_archive(output_dir: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(
        archive_path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, Path(output_dir.name) / path.relative_to(output_dir))


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    archive = args.archive.resolve()

    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {archive}")

    groups = read_jsonl(source)
    by_qid = {
        str(group.get("extra_info", {}).get("question_id")): group for group in groups
    }
    missing_qids = [spec["qid"] for spec in CASE_SPECS if spec["qid"] not in by_qid]
    if missing_qids:
        raise KeyError(f"candidate qids absent from source: {missing_qids}")

    # Complete validation happens before creating any output path.
    records = [analyze_case(by_qid[spec["qid"]], spec) for spec in CASE_SPECS]
    for record in records:
        if not record["image_path"].is_file():
            raise FileNotFoundError(record["image_path"])

    output.mkdir(parents=True)
    summary_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []

    for record in records:
        case_dir = output / record["slug"]
        case_dir.mkdir()
        image_output = case_dir / "original_image.jpg"
        table_output = case_dir / "reward_table.png"
        shutil.copy2(record["image_path"], image_output)
        render_reward_table(record, table_output)

        summary_rows.append(
            {
                "case": record["slug"],
                "evidence_grade": record["evidence_grade"],
                "question_id": record["qid"],
                "source_image_id": record["image_path"].name,
                "question": record["question"],
                "reference_answer": record["ground_truth"],
                "n_total": record["n_total"],
                "n_valid": record["n_valid"],
                "n_unknown": record["n_unknown"],
                "prove_answer": record["prove_answer"],
                "prove_sample_index": record["prove_sample_index"],
                "prove_full_reward": f"{record['prove_reward']:.9f}",
                "ttrv_answer": record["ttrv_answer"],
                "ttrv_sample_index": record["ttrv_sample_index"],
                "ttrv_count": record["ttrv_count"],
                "ttrv_frequency_reward": f"{record['ttrv_reward']:.9f}",
                "prove_answer_ttrv_count": record["prove_answer_ttrv_count"],
                "prove_answer_ttrv_reward": (
                    f"{record['prove_answer_ttrv_reward']:.9f}"
                ),
                "ttrv_answer_prove_max_full_reward": "{:.9f}".format(
                    record["ttrv_answer_prove_reward"]
                ),
                "semantic_audit_note": record["audit_note"],
            }
        )
        for row in record["answer_rows"]:
            answer_rows.append(
                {
                    "case": record["slug"],
                    "question_id": record["qid"],
                    "answer": row["answer"],
                    "count": row["count"],
                    "semantic_status": row["semantic_status"],
                    "ttrv_frequency_reward": f"{row['ttrv_reward']:.9f}",
                    "prove_max_full_reward": f"{row['prove_max_reward']:.9f}",
                    "prove_best_sample_index": row["prove_best_sample_index"],
                    "selected_by": row["selected_by"],
                }
            )

    summary_fields = [
        "case",
        "evidence_grade",
        "question_id",
        "source_image_id",
        "question",
        "reference_answer",
        "n_total",
        "n_valid",
        "n_unknown",
        "prove_answer",
        "prove_sample_index",
        "prove_full_reward",
        "ttrv_answer",
        "ttrv_sample_index",
        "ttrv_count",
        "ttrv_frequency_reward",
        "prove_answer_ttrv_count",
        "prove_answer_ttrv_reward",
        "ttrv_answer_prove_max_full_reward",
        "semantic_audit_note",
    ]
    answer_fields = [
        "case",
        "question_id",
        "answer",
        "count",
        "semantic_status",
        "ttrv_frequency_reward",
        "prove_max_full_reward",
        "prove_best_sample_index",
        "selected_by",
    ]
    write_csv(output / "case_summary.csv", summary_rows, summary_fields)
    write_csv(output / "all_answer_rewards.csv", answer_rows, answer_fields)

    for case_dir in (output / record["slug"] for record in records):
        filenames = sorted(path.name for path in case_dir.iterdir() if path.is_file())
        if filenames != ["original_image.jpg", "reward_table.png"]:
            raise AssertionError(f"unexpected case files in {case_dir}: {filenames}")

    build_archive(output, archive)
    print(
        json.dumps(
            {
                "source": str(source),
                "output": str(output),
                "archive": str(archive),
                "case_count": len(records),
                "cases": [
                    {
                        "qid": record["qid"],
                        "prove": record["prove_answer"],
                        "ttrv": record["ttrv_answer"],
                        "prove_on_ours": round(record["prove_reward"], 9),
                        "prove_on_majority": round(record["ttrv_answer_prove_reward"], 9),
                        "ttrv_on_ours": round(record["prove_answer_ttrv_reward"], 9),
                        "ttrv_on_majority": round(record["ttrv_reward"], 9),
                    }
                    for record in records
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
