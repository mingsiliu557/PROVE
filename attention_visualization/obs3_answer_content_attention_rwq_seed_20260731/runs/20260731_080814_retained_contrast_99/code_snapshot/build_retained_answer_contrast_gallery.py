#!/usr/bin/env python3
"""Render strict Ours-correct/TTRV-wrong retained-answer attention contrasts.

The two model panels intentionally use different teacher-forced targets:
Ours uses its retained correct option text, while TTRV uses its retained wrong
option text.  This is therefore a per-model retained-answer grounding contrast,
not a same-target checkpoint comparison.

All ranking metrics are computed from raw T=1 attention.  Temperature is used
only for display, with the audited support- and mass-preserving transform from
``build_grounding_pool_visualizations_v4``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from _lexical_contrast_renderer_impl import (
    FixedSetting,
    _contained_size,
    _font,
    _load_raw_window,
    _sanitize,
    _write_csv,
    _write_json,
)
from build_grounding_pool_visualizations_v4 import (
    mass_preserving_support_temperature,
)


QUERY = "content_answer_state_heads"
SETTING = FixedSetting(query=QUERY, head=2, center=15, layers=(14, 15, 16))
MODELS = ("ours", "ttrv")
VALID_LABEL = re.compile(r"^[A-F]$")
DISPLAY_TRANSFORM = (
    "m=sum(A); support={i:A_i>0}; on support, "
    "p_T=softmax(log(A_i/m)/T); displayed=m*p_T; off support=0"
)


@dataclass(frozen=True)
class PreparedCase:
    manifest: dict[str, Any]
    ours_record: dict[str, Any]
    ttrv_record: dict[str, Any]
    image_path: Path
    ours_npz: Path
    ttrv_npz: Path
    ours_raw: np.ndarray
    ttrv_raw: np.ndarray
    ours_metrics: dict[str, float]
    ttrv_metrics: dict[str, float]
    jsd_normalized: float
    ours_mass_share: float
    rank_score: float


def _compact(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalized_label(value: Any, field: str, case_id: str) -> str:
    label = _compact(value).upper()
    if not VALID_LABEL.fullmatch(label):
        raise ValueError(f"{case_id}: {field} must be A-F, got {value!r}")
    return label


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL contains no records: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_file(value: Any, parent: Path, field: str, case_id: str) -> Path:
    text = _compact(value)
    if not text:
        raise ValueError(f"{case_id}: {field} is empty")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{case_id}: {field} does not exist: {path}")
    return path


def _ensure_clean_output(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Output exists and is not a directory: {path}")
    if path.is_dir() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite prior results: {path}"
        )


def _question_only(value: Any) -> str:
    text = str(value or "").replace("<image>", " ").strip()
    kept: list[str] = []
    for line in text.splitlines():
        if re.match(r"^\s*[A-F]\s*[\.:\)]\s*", line):
            break
        if re.search(r"please\s+(?:answer|respond)", line, flags=re.IGNORECASE):
            break
        kept.append(line)
    return _compact(" ".join(kept)) or _compact(value)


def _record_map(
    records: Sequence[dict[str, Any]],
    manifest_ids: set[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    unexpected_cases: set[str] = set()
    for record in records:
        case_id = _compact(record.get("case_id"))
        model = _compact(record.get("model")).lower()
        if case_id not in manifest_ids:
            unexpected_cases.add(case_id or "<empty>")
            continue
        if model not in MODELS:
            raise ValueError(f"{case_id}: unexpected model={model!r}; expected {MODELS}")
        key = (case_id, model)
        if key in result:
            raise ValueError(f"Duplicate attention record for case/model={key}")
        result[key] = record
    if unexpected_cases:
        preview = sorted(unexpected_cases)[:10]
        raise ValueError(
            "Attention records contain cases absent from the strict manifest: "
            f"{preview} (count={len(unexpected_cases)})"
        )
    return result


def _retained_label(case: Mapping[str, Any], model: str, case_id: str) -> str:
    retained = case.get("retained_predictions")
    if not isinstance(retained, Mapping):
        raise ValueError(f"{case_id}: manifest retained_predictions is missing")
    model_row = retained.get(model)
    if not isinstance(model_row, Mapping):
        raise ValueError(f"{case_id}: retained_predictions.{model} is missing")
    return _normalized_label(
        model_row.get("parsed_answer"),
        f"retained_predictions.{model}.parsed_answer",
        case_id,
    )


def _spatial_metrics(grid: np.ndarray) -> dict[str, float]:
    array = np.asarray(grid, dtype=np.float64)
    mass = float(array.sum(dtype=np.float64))
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"Raw image attention mass must be positive, got {mass}")
    probability = array / mass
    flat = probability.reshape(-1)
    positive = flat > 0.0
    entropy = -float(np.sum(flat[positive] * np.log(flat[positive]), dtype=np.float64))
    entropy_normalized = entropy / math.log(flat.size) if flat.size > 1 else 0.0
    top_k = max(1, int(math.ceil(flat.size * 0.10)))
    top10_mass_fraction = float(np.partition(flat, -top_k)[-top_k:].sum(dtype=np.float64))

    height, width = array.shape
    border_y = max(1, int(math.ceil(height * 0.125)))
    border_x = max(1, int(math.ceil(width * 0.125)))
    edge = np.zeros((height, width), dtype=bool)
    edge[:border_y, :] = True
    edge[-border_y:, :] = True
    edge[:, :border_x] = True
    edge[:, -border_x:] = True
    edge_fraction = float(probability[edge].sum(dtype=np.float64))

    yy, xx = np.indices(array.shape, dtype=np.float64)
    center_x = float((probability * xx).sum(dtype=np.float64)) / max(1, width - 1)
    center_y = float((probability * yy).sum(dtype=np.float64)) / max(1, height - 1)
    return {
        "raw_image_attention_mass": mass,
        "raw_peak_attention": float(array.max()),
        "raw_conditional_entropy_normalized": entropy_normalized,
        "raw_top10pct_mass_fraction": top10_mass_fraction,
        "raw_edge_mass_fraction": edge_fraction,
        "raw_center_x_normalized": center_x,
        "raw_center_y_normalized": center_y,
    }


def _jsd_normalized(left: np.ndarray, right: np.ndarray) -> float:
    left_mass = float(left.sum(dtype=np.float64))
    right_mass = float(right.sum(dtype=np.float64))
    p = np.asarray(left, dtype=np.float64).reshape(-1) / left_mass
    q = np.asarray(right, dtype=np.float64).reshape(-1) / right_mass
    middle = 0.5 * (p + q)

    def kl(source: np.ndarray) -> float:
        support = source > 0.0
        return float(
            np.sum(
                source[support] * np.log(source[support] / middle[support]),
                dtype=np.float64,
            )
        )

    value = 0.5 * (kl(p) + kl(q)) / math.log(2.0)
    return float(np.clip(value, 0.0, 1.0))


def _validate_pair(
    case: dict[str, Any],
    ours: dict[str, Any],
    ttrv: dict[str, Any],
    manifest_parent: Path,
    records_parent: Path,
) -> PreparedCase:
    case_id = _compact(case.get("case_id"))
    if not case_id:
        raise ValueError("Manifest case_id is empty")
    dataset = _compact(case.get("dataset"))
    if not dataset:
        raise ValueError(f"{case_id}: dataset is empty")
    gt = _normalized_label(case.get("ground_truth"), "ground_truth", case_id)
    manifest_ours = _retained_label(case, "ours", case_id)
    manifest_ttrv = _retained_label(case, "ttrv", case_id)
    if manifest_ours != gt:
        raise ValueError(
            f"{case_id}: strict outcome violated: Ours={manifest_ours}, GT={gt}"
        )
    if manifest_ttrv == gt:
        raise ValueError(
            f"{case_id}: strict outcome violated: TTRV={manifest_ttrv} equals GT={gt}"
        )

    for model, record, expected_label in (
        ("ours", ours, manifest_ours),
        ("ttrv", ttrv, manifest_ttrv),
    ):
        if _compact(record.get("case_id")) != case_id:
            raise ValueError(f"{case_id}: {model} record case_id drift")
        if _compact(record.get("dataset")) != dataset:
            raise ValueError(f"{case_id}: {model} record dataset drift")
        if _compact(record.get("model")).lower() != model:
            raise ValueError(f"{case_id}: {model} record model drift")
        if _compact(record.get("target_source")) != "retained":
            raise ValueError(
                f"{case_id}: {model} target_source must be 'retained', got "
                f"{record.get('target_source')!r}"
            )
        record_label = _normalized_label(
            record.get("target_label"), f"{model}.target_label", case_id
        )
        retained_label = _normalized_label(
            record.get("retained_parsed_answer"),
            f"{model}.retained_parsed_answer",
            case_id,
        )
        if record_label != expected_label or retained_label != expected_label:
            raise ValueError(
                f"{case_id}: {model} retained label mismatch: manifest={expected_label}, "
                f"target={record_label}, record_retained={retained_label}"
            )
        record_gt = _normalized_label(
            record.get("ground_truth"), f"{model}.ground_truth", case_id
        )
        if record_gt != gt:
            raise ValueError(
                f"{case_id}: {model} ground_truth={record_gt} differs from manifest={gt}"
            )
        if not _compact(record.get("target_option_text")):
            raise ValueError(f"{case_id}: {model} target_option_text is empty")

    ours_text = _compact(ours["target_option_text"])
    ttrv_text = _compact(ttrv["target_option_text"])
    if ours_text.casefold() == ttrv_text.casefold():
        raise ValueError(
            f"{case_id}: different labels resolve to identical target text {ours_text!r}"
        )

    manifest_image = _resolve_file(
        case.get("image_path"), manifest_parent, "manifest.image_path", case_id
    )
    for model, record in (("ours", ours), ("ttrv", ttrv)):
        record_image = _resolve_file(
            record.get("image_path"), records_parent, f"{model}.image_path", case_id
        )
        if record_image != manifest_image:
            raise ValueError(
                f"{case_id}: {model} image path drift: {record_image} != {manifest_image}"
            )
    ours_npz = _resolve_file(ours.get("npz_path"), records_parent, "ours.npz_path", case_id)
    ttrv_npz = _resolve_file(ttrv.get("npz_path"), records_parent, "ttrv.npz_path", case_id)
    ours_raw = _load_raw_window(ours_npz, SETTING)
    ttrv_raw = _load_raw_window(ttrv_npz, SETTING)
    if ours_raw.shape != ttrv_raw.shape:
        raise ValueError(
            f"{case_id}: Ours/TTRV attention shapes differ: {ours_raw.shape} vs {ttrv_raw.shape}"
        )
    ours_metrics = _spatial_metrics(ours_raw)
    ttrv_metrics = _spatial_metrics(ttrv_raw)
    jsd = _jsd_normalized(ours_raw, ttrv_raw)
    ours_mass = ours_metrics["raw_image_attention_mass"]
    ttrv_mass = ttrv_metrics["raw_image_attention_mass"]
    ours_mass_share = ours_mass / (ours_mass + ttrv_mass)
    # A transparent attention-only browsing score.  It cannot establish causal
    # correctness; its only purpose is to put visually clear contrasts first.
    rank_score = (
        0.40 * ours_metrics["raw_top10pct_mass_fraction"]
        + 0.30 * jsd
        + 0.20 * ours_mass_share
        + 0.10 * (1.0 - ours_metrics["raw_edge_mass_fraction"])
    )
    return PreparedCase(
        manifest=case,
        ours_record=ours,
        ttrv_record=ttrv,
        image_path=manifest_image,
        ours_npz=ours_npz,
        ttrv_npz=ttrv_npz,
        ours_raw=ours_raw,
        ttrv_raw=ttrv_raw,
        ours_metrics=ours_metrics,
        ttrv_metrics=ttrv_metrics,
        jsd_normalized=jsd,
        ours_mass_share=ours_mass_share,
        rank_score=rank_score,
    )


def _viridis(fraction: np.ndarray) -> np.ndarray:
    import cv2

    values = np.round(np.clip(fraction, 0.0, 1.0) * 255.0).astype(np.uint8)
    colors = cv2.applyColorMap(values, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)


def _make_overlay(original: Any, grid: np.ndarray, vmax: float, alpha: float) -> Any:
    from PIL import Image

    normalized = np.clip(np.asarray(grid, dtype=np.float64) / vmax, 0.0, 1.0)
    heat = Image.fromarray(_viridis(normalized), mode="RGB")
    heat = heat.resize(original.size, resample=Image.Resampling.BILINEAR)
    return Image.blend(original, heat, alpha=alpha)


def _draw_colorbar(
    canvas: Any,
    x: int,
    y: int,
    height: int,
    vmax: float,
    temperature: float,
) -> None:
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(canvas)
    fractions = np.linspace(1.0, 0.0, max(2, height), dtype=np.float64).reshape(-1, 1)
    values = np.repeat(fractions, 28, axis=1)
    canvas.paste(Image.fromarray(_viridis(values), mode="RGB"), (x, y))
    draw.rectangle((x, y, x + 27, y + height - 1), outline="black", width=1)
    small = _font(13)
    draw.text((x + 34, y - 7), f"{vmax:.4g}", fill="black", font=small)
    draw.text((x + 34, y + height - 16), "0", fill="black", font=small)
    draw.text(
        (x - 4, y + height + 6),
        f"shared display scale, T={temperature:g}",
        fill=(60, 60, 60),
        font=small,
    )


def _render_side_by_side(
    prepared: PreparedCase,
    dataset_rank: int,
    original: Any,
    ours_overlay: Any,
    ttrv_overlay: Any,
    display_vmax: float,
    display_temperature: float,
    panel_max_width: int,
    panel_max_height: int,
) -> Any:
    from PIL import Image, ImageDraw

    case = prepared.manifest
    ours = prepared.ours_record
    ttrv = prepared.ttrv_record
    display_size = _contained_size(original.size, panel_max_width, panel_max_height)
    panels = (
        original.resize(display_size, resample=Image.Resampling.LANCZOS),
        ours_overlay.resize(display_size, resample=Image.Resampling.LANCZOS),
        ttrv_overlay.resize(display_size, resample=Image.Resampling.LANCZOS),
    )
    padding = 16
    panel_width = display_size[0] + 2 * padding
    colorbar_width = 180
    canvas_width = panel_width * 3 + colorbar_width

    title_font = _font(20, bold=True)
    body_font = _font(15)
    note_font = _font(13)
    question = _question_only(case.get("question") or ours.get("question"))
    ours_label = _compact(ours["target_label"])
    ttrv_label = _compact(ttrv["target_label"])
    ours_text = _compact(ours["target_option_text"])
    ttrv_text = _compact(ttrv["target_option_text"])
    lines: list[tuple[str, Any, tuple[int, int, int]]] = [
        (
            f"Rank {dataset_rank} | {case['dataset']}/{case['case_id']} | raw-T1 browsing score={prepared.rank_score:.5f}",
            title_font,
            (0, 0, 0),
        )
    ]
    lines.extend(
        (f"Question: {line}", body_font, (0, 0, 0))
        for line in (textwrap.wrap(question, width=155) or [""])
    )
    lines.extend(
        [
            (
                f"GT / Ours retained target (correct): {ours_label}. {ours_text}",
                body_font,
                (0, 95, 25),
            ),
            (
                f"TTRV retained target (wrong): {ttrv_label}. {ttrv_text}",
                body_font,
                (150, 25, 25),
            ),
            (
                "The two panels query different per-model retained answer texts; this is not a same-target checkpoint comparison.",
                note_font,
                (55, 55, 55),
            ),
            (
                "Answer-state probe = mean over alphanumeric option-text token states after each token is observed (teacher forced; non-causal).",
                note_font,
                (55, 55, 55),
            ),
            (
                f"Fixed query={QUERY}, head=2 (0-based), layers=[14, 15, 16]. Rank/metrics: raw T=1. Display only: mass-preserving T={display_temperature:g}.",
                note_font,
                (55, 55, 55),
            ),
        ]
    )
    line_height = 23
    header_height = 14 + line_height * len(lines)
    label_height = 86
    footer_height = 48
    canvas_height = header_height + label_height + display_size[1] + footer_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (line, font, color) in enumerate(lines):
        draw.text((14, 8 + index * line_height), line, fill=color, font=font)

    ours_mass = prepared.ours_metrics["raw_image_attention_mass"]
    ttrv_mass = prepared.ttrv_metrics["raw_image_attention_mass"]
    labels = (
        ["Original"],
        [f"Ours query: {ours_label}. {ours_text}", f"raw image mass={ours_mass:.6g}"],
        [f"TTRV query: {ttrv_label}. {ttrv_text}", f"raw image mass={ttrv_mass:.6g}"],
    )
    image_y = header_height + label_height
    for column, (label_lines, panel) in enumerate(zip(labels, panels)):
        x0 = column * panel_width
        for line_index, line in enumerate(label_lines):
            clipped = textwrap.shorten(line, width=76, placeholder="...")
            draw.text(
                (x0 + padding, header_height + 5 + line_index * 25),
                clipped,
                fill="black",
                font=title_font if line_index == 0 else body_font,
            )
        canvas.paste(panel, (x0 + padding, image_y))
    _draw_colorbar(
        canvas,
        panel_width * 3 + 12,
        image_y,
        display_size[1],
        display_vmax,
        display_temperature,
    )
    draw.text(
        (14, canvas_height - 38),
        "Per-case Ours/TTRV color scale is shared. Display sharpening preserves each raw map's total image mass and exact zero support.",
        fill=(60, 60, 60),
        font=note_font,
    )
    draw.text(
        (14, canvas_height - 19),
        "Attention is a qualitative grounding probe, not causal evidence that the highlighted region produced the answer.",
        fill=(60, 60, 60),
        font=note_font,
    )
    return canvas


def _render_contact_sheet(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    dataset: str,
    columns: int,
    display_temperature: float,
) -> None:
    from PIL import Image, ImageDraw

    if not rows:
        raise ValueError(f"Cannot make an empty contact sheet for {dataset}")
    cell_width, cell_height = 940, 590
    header_height = 82
    grid_rows = int(math.ceil(len(rows) / columns))
    canvas = Image.new(
        "RGB", (cell_width * columns, header_height + cell_height * grid_rows), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (16, 9),
        f"{dataset}: Ours-correct vs TTRV-wrong retained-answer grounding",
        fill="black",
        font=_font(22, bold=True),
    )
    draw.text(
        (16, 43),
        f"different retained targets | answer-state non-causal probe | raw-T1 rank | display T={display_temperature:g} | shared pairwise scale",
        fill=(55, 55, 55),
        font=_font(14),
    )
    for index, row in enumerate(rows):
        column = index % columns
        grid_row = index // columns
        x0 = column * cell_width
        y0 = header_height + grid_row * cell_height
        caption = (
            f"#{row['dataset_rank']} {row['case_id']} | "
            f"Ours={row['ours_target_label']} ({row['ours_target_option_text']}) | "
            f"TTRV={row['ttrv_target_label']} ({row['ttrv_target_option_text']})"
        )
        draw.text(
            (x0 + 10, y0 + 7),
            textwrap.shorten(caption, width=120, placeholder="..."),
            fill="black",
            font=_font(14, bold=True),
        )
        with Image.open(row["side_by_side_png"]) as handle:
            figure = handle.convert("RGB")
            target = _contained_size(figure.size, cell_width - 20, cell_height - 48)
            fitted = figure.resize(target, resample=Image.Resampling.LANCZOS)
        x = x0 + (cell_width - fitted.width) // 2
        y = y0 + 38 + (cell_height - 42 - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        draw.rectangle(
            (x0, y0, x0 + cell_width - 1, y0 + cell_height - 1),
            outline=(190, 190, 190),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def _temperature_validate(raw: np.ndarray, displayed: np.ndarray, temperature: float) -> None:
    raw_mass = float(raw.sum(dtype=np.float64))
    displayed_mass = float(displayed.sum(dtype=np.float64))
    tolerance = max(float(np.spacing(abs(raw_mass))) * 16.0, 1e-14)
    if not math.isclose(raw_mass, displayed_mass, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(
            f"T={temperature} display mass drift: raw={raw_mass}, display={displayed_mass}"
        )
    if np.any(displayed[raw <= 0.0] != 0.0):
        raise AssertionError("Display temperature changed exact zero support")


def _metric_columns(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def _dataset_summary(
    dataset: str,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    contact_path: Path,
    display_temperature: float,
    alpha: float,
) -> dict[str, Any]:
    return {
        "status": "PASS",
        "dataset": dataset,
        "rendered_cases": len(rows),
        "strict_outcome": "Ours retained label == GT; TTRV retained label is valid and != GT",
        "target_comparison": (
            "different per-model retained option texts: Ours correct target versus TTRV wrong target"
        ),
        "fixed_setting": {
            "query": QUERY,
            "head_index_zero_based": SETTING.head,
            "layer_center_zero_based": SETTING.center,
            "layer_window_zero_based": list(SETTING.layers),
            "layer_aggregation": "arithmetic mean of raw saved head grids",
        },
        "interpretation_boundary": (
            "teacher-forced content answer-state probe; each queried token is already observed; "
            "non-causal and not a same-target checkpoint comparison"
        ),
        "ranking_policy": {
            "source": "raw T=1 only",
            "purpose": "attention-only browsing order, not an accuracy or causal score",
            "formula": (
                "0.40*ours_raw_top10pct_mass_fraction + 0.30*raw_JSD_normalized + "
                "0.20*ours_raw_mass_share + 0.10*(1-ours_raw_edge_mass_fraction)"
            ),
        },
        "display_policy": {
            "temperature": display_temperature,
            "temperature_used_for_ranking_or_metrics": False,
            "transform": DISPLAY_TRANSFORM,
            "mass_preserved": True,
            "zero_support_preserved": True,
            "color_scale": "per case, shared between Ours and TTRV; vmin=0 and pairwise display vmax",
            "overlay_alpha": alpha,
            "aspect_ratio_preserved": True,
        },
        "outputs": {
            "dataset_dir": str(output_dir.resolve()),
            "original_dir": str((output_dir / "original").resolve()),
            "ours_overlay_dir": str((output_dir / "ours_overlay").resolve()),
            "ttrv_overlay_dir": str((output_dir / "ttrv_overlay").resolve()),
            "side_by_side_dir": str((output_dir / "side_by_side").resolve()),
            "index_csv": str((output_dir / "index.csv").resolve()),
            "contact_sheet_png": str(contact_path.resolve()),
            "summary_json": str((output_dir / "summary.json").resolve()),
        },
    }


def build_gallery(
    attention_records: str | Path,
    manifest: str | Path,
    output_dir: str | Path,
    display_temperature: float = 0.5,
    alpha: float = 0.55,
    panel_max_width: int = 620,
    panel_max_height: int = 620,
    contact_columns: int = 2,
    top_n_per_dataset: int = 0,
) -> dict[str, Any]:
    if not math.isfinite(display_temperature) or not 0.0 < display_temperature <= 1.0:
        raise ValueError("display_temperature must satisfy 0<T<=1")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0,1]")
    if panel_max_width <= 0 or panel_max_height <= 0:
        raise ValueError("panel dimensions must be positive")
    if contact_columns <= 0:
        raise ValueError("contact_columns must be positive")
    if top_n_per_dataset < 0:
        raise ValueError("top_n_per_dataset must be >=0; zero means render all")

    records_path = Path(attention_records).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    if not records_path.is_file():
        raise FileNotFoundError(f"Attention records do not exist: {records_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Strict manifest does not exist: {manifest_path}")
    destination = Path(output_dir).expanduser().resolve()
    _ensure_clean_output(destination)

    cases = _read_jsonl(manifest_path)
    manifest_by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = _compact(case.get("case_id"))
        if not case_id:
            raise ValueError("Manifest contains an empty case_id")
        if case_id in manifest_by_id:
            raise ValueError(f"Duplicate manifest case_id={case_id}")
        manifest_by_id[case_id] = case
    records = _read_jsonl(records_path)
    record_by_key = _record_map(records, set(manifest_by_id))
    expected_keys = {(case_id, model) for case_id in manifest_by_id for model in MODELS}
    missing = sorted(expected_keys - set(record_by_key))
    if missing:
        raise ValueError(f"Missing Ours/TTRV attention records: {missing[:10]} (count={len(missing)})")

    # Validate every input and load every raw map before creating output files.
    prepared: list[PreparedCase] = []
    for case_id, case in manifest_by_id.items():
        prepared.append(
            _validate_pair(
                case,
                record_by_key[(case_id, "ours")],
                record_by_key[(case_id, "ttrv")],
                manifest_path.parent,
                records_path.parent,
            )
        )

    by_dataset: dict[str, list[PreparedCase]] = defaultdict(list)
    for item in prepared:
        by_dataset[_compact(item.manifest["dataset"])].append(item)
    selected_by_dataset: dict[str, list[PreparedCase]] = {}
    for dataset, items in sorted(by_dataset.items()):
        ranked = sorted(
            items,
            key=lambda item: (-item.rank_score, _compact(item.manifest["case_id"])),
        )
        selected_by_dataset[dataset] = (
            ranked[:top_n_per_dataset] if top_n_per_dataset > 0 else ranked
        )

    destination.mkdir(parents=True, exist_ok=True)
    all_index_rows: list[dict[str, Any]] = []
    dataset_summaries: dict[str, Any] = {}
    for dataset, selected in selected_by_dataset.items():
        from PIL import Image

        dataset_dir = destination / _sanitize(dataset)
        directories = {
            name: dataset_dir / name
            for name in ("original", "ours_overlay", "ttrv_overlay", "side_by_side")
        }
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=False)
        dataset_rows: list[dict[str, Any]] = []
        for dataset_rank, item in enumerate(selected, start=1):
            case_id = _compact(item.manifest["case_id"])
            stem = f"{dataset_rank:03d}_{_sanitize(case_id)}"
            original_suffix = item.image_path.suffix.lower() or ".img"
            copied_original = directories["original"] / f"{stem}{original_suffix}"
            shutil.copy2(item.image_path, copied_original)
            with Image.open(item.image_path) as handle:
                original = handle.convert("RGB").copy()

            ours_display = mass_preserving_support_temperature(
                item.ours_raw, display_temperature
            )
            ttrv_display = mass_preserving_support_temperature(
                item.ttrv_raw, display_temperature
            )
            _temperature_validate(item.ours_raw, ours_display, display_temperature)
            _temperature_validate(item.ttrv_raw, ttrv_display, display_temperature)
            display_vmax = max(float(ours_display.max()), float(ttrv_display.max()))
            if not math.isfinite(display_vmax) or display_vmax <= 0.0:
                raise ValueError(f"{case_id}: shared display vmax is not positive")
            ours_overlay = _make_overlay(original, ours_display, display_vmax, alpha)
            ttrv_overlay = _make_overlay(original, ttrv_display, display_vmax, alpha)
            ours_overlay_path = directories["ours_overlay"] / f"{stem}_T{display_temperature:g}.png"
            ttrv_overlay_path = directories["ttrv_overlay"] / f"{stem}_T{display_temperature:g}.png"
            side_path = directories["side_by_side"] / f"{stem}_T{display_temperature:g}.png"
            ours_overlay.save(ours_overlay_path, format="PNG", optimize=True)
            ttrv_overlay.save(ttrv_overlay_path, format="PNG", optimize=True)
            side = _render_side_by_side(
                item,
                dataset_rank,
                original,
                ours_overlay,
                ttrv_overlay,
                display_vmax,
                display_temperature,
                panel_max_width,
                panel_max_height,
            )
            side.save(side_path, format="PNG", optimize=True)

            ours = item.ours_record
            ttrv = item.ttrv_record
            row: dict[str, Any] = {
                "dataset_rank": dataset_rank,
                "case_id": case_id,
                "dataset": dataset,
                "question_id": item.manifest.get("question_id"),
                "question": _question_only(item.manifest.get("question") or ours.get("question")),
                "ground_truth": _compact(item.manifest.get("ground_truth")),
                "ours_target_source": "retained",
                "ours_target_label": _compact(ours.get("target_label")),
                "ours_target_option_text": _compact(ours.get("target_option_text")),
                "ours_target_correct": True,
                "ttrv_target_source": "retained",
                "ttrv_target_label": _compact(ttrv.get("target_label")),
                "ttrv_target_option_text": _compact(ttrv.get("target_option_text")),
                "ttrv_target_correct": False,
                "targets_differ": True,
                "comparison_type": "different_per_model_retained_answer_targets",
                "query_mode": QUERY,
                "head_index_zero_based": SETTING.head,
                "layer_center_zero_based": SETTING.center,
                "layer_window_zero_based": list(SETTING.layers),
                "rank_metrics_temperature": 1.0,
                "raw_jsd_normalized": item.jsd_normalized,
                "ours_raw_mass_share": item.ours_mass_share,
                "raw_rank_score": item.rank_score,
                **_metric_columns("ours", item.ours_metrics),
                **_metric_columns("ttrv", item.ttrv_metrics),
                "display_temperature": display_temperature,
                "display_transform": DISPLAY_TRANSFORM,
                "display_temperature_used_for_rank_or_metrics": False,
                "shared_display_vmin": 0.0,
                "shared_display_vmax": display_vmax,
                "ours_display_mass": float(ours_display.sum(dtype=np.float64)),
                "ttrv_display_mass": float(ttrv_display.sum(dtype=np.float64)),
                "answer_state_interpretation": (
                    "teacher-forced token-conditioned grounding probe; token already observed; non-causal"
                ),
                "source_image_path": str(item.image_path),
                "source_npz_ours": str(item.ours_npz),
                "source_npz_ttrv": str(item.ttrv_npz),
                "copied_original": str(copied_original.resolve()),
                "ours_overlay_png": str(ours_overlay_path.resolve()),
                "ttrv_overlay_png": str(ttrv_overlay_path.resolve()),
                "side_by_side_png": str(side_path.resolve()),
                "source_image_width": original.width,
                "source_image_height": original.height,
                "aspect_ratio_preserved": (
                    original.size == ours_overlay.size == ttrv_overlay.size
                ),
            }
            dataset_rows.append(row)
            all_index_rows.append(row)

        dataset_index = dataset_dir / "index.csv"
        _write_csv(dataset_index, dataset_rows)
        contact_path = dataset_dir / f"contact_sheet_T{display_temperature:g}.png"
        _render_contact_sheet(
            dataset_rows,
            contact_path,
            dataset,
            contact_columns,
            display_temperature,
        )
        dataset_summary = _dataset_summary(
            dataset,
            dataset_rows,
            dataset_dir,
            contact_path,
            display_temperature,
            alpha,
        )
        dataset_summary["input_strict_cases"] = len(by_dataset[dataset])
        dataset_summary["top_n_per_dataset"] = top_n_per_dataset
        _write_json(dataset_dir / "summary.json", dataset_summary)
        dataset_summaries[dataset] = dataset_summary

    root_index_path = destination / "index.csv"
    _write_csv(root_index_path, all_index_rows)
    summary = {
        "status": "PASS",
        "attention_records_jsonl": str(records_path),
        "attention_records_sha256": _sha256(records_path),
        "strict_manifest_jsonl": str(manifest_path),
        "strict_manifest_sha256": _sha256(manifest_path),
        "manifest_cases": len(cases),
        "attention_records": len(records),
        "validated_complete_pairs": len(prepared),
        "rendered_cases": len(all_index_rows),
        "rendered_by_dataset": dict(
            sorted(Counter(row["dataset"] for row in all_index_rows).items())
        ),
        "strict_outcome_validation": {
            "ours_retained_equals_ground_truth": True,
            "ttrv_retained_is_valid_and_differs_from_ground_truth": True,
            "target_source_retained_for_both_models": True,
            "different_target_option_text_required": True,
        },
        "comparison_scope": (
            "Ours correct retained target versus TTRV wrong retained target; "
            "not a same-target checkpoint comparison"
        ),
        "fixed_setting": {
            "query": QUERY,
            "head_index_zero_based": SETTING.head,
            "layer_center_zero_based": SETTING.center,
            "layer_window_zero_based": list(SETTING.layers),
        },
        "rank_metrics_temperature": 1.0,
        "display_temperature": display_temperature,
        "display_transform": DISPLAY_TRANSFORM,
        "temperature_used_for_rank_or_metrics": False,
        "training": False,
        "checkpoint_mutation": False,
        "top_n_per_dataset": top_n_per_dataset,
        "datasets": dataset_summaries,
        "outputs": {
            "root_index_csv": str(root_index_path.resolve()),
            "root_summary_json": str((destination / "summary.json").resolve()),
        },
    }
    _write_json(destination / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-records", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--display-temperature", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--panel-max-width", type=int, default=620)
    parser.add_argument("--panel-max-height", type=int, default=620)
    parser.add_argument("--contact-columns", type=int, default=2)
    parser.add_argument(
        "--top-n-per-dataset",
        type=int,
        default=0,
        help="Render the highest raw-T1 attention-only scores per dataset; 0 renders all.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build_gallery(
        attention_records=args.attention_records,
        manifest=args.manifest,
        output_dir=args.output_dir,
        display_temperature=args.display_temperature,
        alpha=args.alpha,
        panel_max_width=args.panel_max_width,
        panel_max_height=args.panel_max_height,
        contact_columns=args.contact_columns,
        top_n_per_dataset=args.top_n_per_dataset,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
