#!/usr/bin/env python3
"""Render an outcome-blind lexical grounding review pool.

This v3 module is the executable public entry point for display-temperature
visualizations.  It preserves v2's validated rank selection and raw metrics,
while adding a display-only, mass-preserving spatial temperature transform.
Ranking and quality/contrast metrics always use the saved raw T=1 maps.

The input is the render-ready ``lexical_grounding_pool_ranked.csv`` companion
produced by ``screen_answer_content_v2.py``.  Every row carries two independent
attention-only ranks: ``quality_rank`` and ``contrast_rank``.  This renderer
selects top-N rows from each requested rank, materializes separate galleries,
and creates a deterministic round-robin, de-duplicated merged review pool.

Outcome fields are annotation only.  They are never read by the ranking
functions and never create a third rank.  Every rendered map uses the frozen
formal setting ``content_answer_state_heads``, head 2, layers [14, 15, 16]
(all zero-based).  Layers are arithmetic-mean aggregated as saved raw
attention at T=1.  No temperature or spatial probability normalization is
applied to ranking or metrics.  For display only, each model/case grid A is
converted with m=sum(A), p=A/m, and
p_T=softmax(log(p+eps)/T), then displayed as m*p_T.  Thus image attention mass
is preserved while 0<T<1 spatially sharpens the map.  Ours and TTRV share
vmin=0 and one exact per-case vmax after this display transform.

Interpretation boundary: these are GT teacher-forced,
answer-token-conditioned grounding probes.  The answer token is already
observed, so the maps are not same-token causal explanations and are not
attention for TTRV's retained wrong/unknown answer token.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import _lexical_contrast_renderer_impl as _helper


FORMAL_QUERY = "content_answer_state_heads"
FORMAL_HEAD = 2
FORMAL_CENTER = 15
FORMAL_LAYERS = (14, 15, 16)
DISPLAY_EPS = 1e-12
RANK_COLUMNS = {"quality": "quality_rank", "contrast": "contrast_rank"}
QUERY_SEMANTICS = (
    "GT teacher-forced answer-token-conditioned grounding probe; token already observed; "
    "non-causal for that same token; not TTRV wrong/unknown-token attention"
)
REQUIRED_COLUMNS = (
    "grounding_rank_type",
    "rank",
    "quality_rank",
    "contrast_rank",
    "case_id",
    "dataset",
    "question_id",
    "question",
    "image_path",
    "target_label",
    "target_option_text",
    "ground_truth",
    "ours_prediction",
    "ttrv_prediction",
    "outcome_tier",
    "quality_gate",
    "selection_query_mode",
    "selected_head_index_zero_based",
    "selected_layer_window_zero_based",
    "npz_ours",
    "npz_ttrv",
    "quality_score",
    "contrast_score",
    "grounding_attention_valid",
    "ranking_outcome_blind",
    "render_rank",
    "render_selection_sources",
)


def _compact(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _parse_bool(value: Any, field_name: str, row_number: int) -> bool:
    normalized = str(value or "").strip().casefold()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(
        f"CSV row {row_number}: {field_name} must be boolean, got {value!r}"
    )


def _parse_rank(value: Any, field_name: str, row_number: int) -> int:
    text = str(value or "").strip()
    if not re.fullmatch(r"\d+", text):
        raise ValueError(f"CSV row {row_number}: {field_name} must be a positive integer, got {value!r}")
    parsed = int(text)
    if parsed <= 0:
        raise ValueError(f"CSV row {row_number}: {field_name} must be > 0, got {parsed}")
    return parsed


def _parse_float(value: Any, field_name: str, row_number: int) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CSV row {row_number}: {field_name} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"CSV row {row_number}: {field_name} must be finite, got {value!r}")
    return parsed


def _parse_layers(value: Any, row_number: int) -> tuple[int, ...]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"CSV row {row_number}: selected_layer_window_zero_based must be JSON, got {value!r}"
        ) from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(
            f"CSV row {row_number}: selected_layer_window_zero_based must be a non-empty list"
        )
    if not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in parsed):
        raise ValueError(f"CSV row {row_number}: invalid layer list {parsed!r}")
    return tuple(parsed)


def _resolve_file(value: Any, csv_parent: Path, field_name: str, row_number: int) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"CSV row {row_number}: {field_name} is empty")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = csv_parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CSV row {row_number}: {field_name} does not exist: {path}")
    return path


def _case_tie_break(case_id: str) -> str:
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


def _temperature_text(value: float) -> str:
    return f"{value:.6g}"


def _temperature_slug(value: float) -> str:
    return _temperature_text(value).replace("-", "m").replace(".", "p")


def _mass_preserving_spatial_temperature(
    raw: np.ndarray,
    temperature: float,
    eps: float = DISPLAY_EPS,
) -> np.ndarray:
    """Apply display-only spatial sharpening while preserving sum(raw)."""

    array = np.asarray(raw, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Display-temperature input must be non-empty and finite")
    if np.any(array < 0.0):
        raise ValueError("Display-temperature input must be non-negative")
    mass = float(array.sum(dtype=np.float64))
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"Display-temperature input mass must be positive, got {mass}")
    probability = array / mass
    log_probability = np.log(probability + eps)
    logits = (log_probability - float(log_probability.max())) / temperature
    weights = np.exp(logits)
    weights /= float(weights.sum(dtype=np.float64))
    displayed = weights * mass

    # Correct the final floating-point reduction residual on the largest cell.
    # The correction changes neither selection nor any raw T=1 metric.
    flat = displayed.reshape(-1)
    correction_index = int(np.argmax(flat))
    for _ in range(3):
        residual = mass - float(displayed.sum(dtype=np.float64))
        if residual == 0.0:
            break
        flat[correction_index] += residual
    tolerance = max(float(np.spacing(abs(mass))) * 8.0, 1e-15)
    if not math.isclose(
        float(displayed.sum(dtype=np.float64)),
        mass,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise AssertionError("Display-temperature transform failed to preserve raw mass")
    return displayed


@dataclass(frozen=True)
class PoolRow:
    values: Mapping[str, str]
    csv_row_number: int
    quality_rank: int
    contrast_rank: int
    quality_score: float
    contrast_score: float

    @property
    def case_id(self) -> str:
        return str(self.values["case_id"])


@dataclass
class PreparedCase:
    row: PoolRow
    image_path: Path
    npz_ours: Path
    npz_ttrv: Path
    ours_raw: np.ndarray
    ttrv_raw: np.ndarray
    ours_display: np.ndarray
    ttrv_display: np.ndarray
    display_temperature: float
    vmax: float
    original_size: tuple[int, int]
    asset_paths: dict[str, Path] = field(default_factory=dict)


def _read_pool_csv(path: Path) -> tuple[list[PoolRow], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        if not header:
            raise ValueError(f"Grounding CSV has no header: {path}")
        duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
        if duplicates:
            raise ValueError(f"Grounding CSV has duplicate columns: {duplicates}")
        missing = sorted(set(REQUIRED_COLUMNS) - set(header))
        if missing:
            raise ValueError(
                f"Grounding CSV schema mismatch; missing required columns={missing}"
            )
        raw_rows = [dict(row) for row in reader if any(_compact(value) for value in row.values())]
    if not raw_rows:
        raise ValueError(f"Grounding CSV has no data rows: {path}")

    rows: list[PoolRow] = []
    seen_cases: dict[str, int] = {}
    for row_number, values in enumerate(raw_rows, start=2):
        case_id = _compact(values["case_id"])
        if not case_id:
            raise ValueError(f"CSV row {row_number}: case_id is empty")
        if case_id in seen_cases:
            raise ValueError(
                f"CSV row {row_number}: duplicate case_id={case_id!r}; first at row {seen_cases[case_id]}"
            )
        seen_cases[case_id] = row_number
        if not _parse_bool(values["ranking_outcome_blind"], "ranking_outcome_blind", row_number):
            raise ValueError(
                f"CSV row {row_number}: ranking_outcome_blind is false; refusing outcome-dependent ranking"
            )
        rows.append(
            PoolRow(
                values=values,
                csv_row_number=row_number,
                quality_rank=_parse_rank(values["quality_rank"], "quality_rank", row_number),
                contrast_rank=_parse_rank(values["contrast_rank"], "contrast_rank", row_number),
                quality_score=_parse_float(values["quality_score"], "quality_score", row_number),
                contrast_score=_parse_float(values["contrast_score"], "contrast_score", row_number),
            )
        )

    expected = list(range(1, len(rows) + 1))
    for field_name in ("quality_rank", "contrast_rank"):
        actual = sorted(getattr(row, field_name) for row in rows)
        if actual != expected:
            duplicates = sorted(rank for rank, count in Counter(actual).items() if count > 1)
            missing_ranks = sorted(set(expected) - set(actual))
            raise ValueError(
                f"{field_name} is not a complete 1..{len(rows)} permutation; "
                f"duplicates={duplicates}, missing={missing_ranks}"
            )
    return rows, header


def _parse_rank_types(value: str) -> tuple[str, ...]:
    rank_types = tuple(part.strip().casefold() for part in value.split(",") if part.strip())
    if not rank_types or len(set(rank_types)) != len(rank_types):
        raise ValueError(f"Invalid --rank-types={value!r}")
    unknown = sorted(set(rank_types) - set(RANK_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown rank types {unknown}; expected {sorted(RANK_COLUMNS)}")
    return rank_types


def _rank_value(row: PoolRow, rank_type: str) -> int:
    # Intentionally reads only an attention-only rank field.  Outcome and
    # correctness annotations are not touched by this selection path.
    return row.quality_rank if rank_type == "quality" else row.contrast_rank


def select_rank_pools(
    rows: Sequence[PoolRow], rank_types: Sequence[str], top_n: int
) -> dict[str, list[PoolRow]]:
    selected: dict[str, list[PoolRow]] = {}
    for rank_type in rank_types:
        selected[rank_type] = sorted(
            rows,
            key=lambda row: (_rank_value(row, rank_type), _case_tie_break(row.case_id)),
        )[:top_n]
    return selected


def merge_rank_pools(
    selected: Mapping[str, Sequence[PoolRow]], rank_types: Sequence[str], maximum: int
) -> tuple[list[PoolRow], dict[str, list[str]]]:
    merged: list[PoolRow] = []
    seen: set[str] = set()
    sources: dict[str, list[str]] = defaultdict(list)
    for rank_type in rank_types:
        for row in selected[rank_type]:
            sources[row.case_id].append(rank_type)
    max_length = max((len(selected[rank_type]) for rank_type in rank_types), default=0)
    for index in range(max_length):
        for rank_type in rank_types:
            if index >= len(selected[rank_type]):
                continue
            row = selected[rank_type][index]
            if row.case_id in seen:
                continue
            seen.add(row.case_id)
            merged.append(row)
            if len(merged) >= maximum:
                return merged, dict(sources)
    return merged, dict(sources)


def _validate_fixed_setting(row: PoolRow) -> None:
    values = row.values
    query = _compact(values["selection_query_mode"])
    if query != FORMAL_QUERY:
        raise ValueError(
            f"CSV row {row.csv_row_number}: expected selection_query_mode={FORMAL_QUERY!r}, got {query!r}"
        )
    head = _parse_rank(
        values["selected_head_index_zero_based"],
        "selected_head_index_zero_based",
        row.csv_row_number,
    )
    # _parse_rank is positive-only; the frozen head is 2, so that stricter
    # parser is valid here.
    if head != FORMAL_HEAD:
        raise ValueError(
            f"CSV row {row.csv_row_number}: expected frozen head={FORMAL_HEAD}, got {head}"
        )
    layers = _parse_layers(values["selected_layer_window_zero_based"], row.csv_row_number)
    if layers != FORMAL_LAYERS:
        raise ValueError(
            f"CSV row {row.csv_row_number}: expected frozen layers={list(FORMAL_LAYERS)}, got {list(layers)}"
        )
    if not _parse_bool(
        values["grounding_attention_valid"], "grounding_attention_valid", row.csv_row_number
    ):
        raise ValueError(
            f"CSV row {row.csv_row_number}: selected case has invalid grounding attention"
        )


def _open_oriented_rgb(path: Path):
    from PIL import Image, ImageOps

    with Image.open(path) as handle:
        return ImageOps.exif_transpose(handle).convert("RGB").copy()


def prepare_cases(
    rows: Sequence[PoolRow],
    csv_path: Path,
    display_temperature: float,
) -> dict[str, PreparedCase]:
    setting = _helper.FixedSetting(
        query=FORMAL_QUERY,
        head=FORMAL_HEAD,
        center=FORMAL_CENTER,
        layers=FORMAL_LAYERS,
    )
    prepared: dict[str, PreparedCase] = {}
    for row in rows:
        _validate_fixed_setting(row)
        image_path = _resolve_file(
            row.values["image_path"], csv_path.parent, "image_path", row.csv_row_number
        )
        npz_ours = _resolve_file(
            row.values["npz_ours"], csv_path.parent, "npz_ours", row.csv_row_number
        )
        npz_ttrv = _resolve_file(
            row.values["npz_ttrv"], csv_path.parent, "npz_ttrv", row.csv_row_number
        )
        ours_raw = _helper._load_raw_window(npz_ours, setting)
        ttrv_raw = _helper._load_raw_window(npz_ttrv, setting)
        if ours_raw.shape != ttrv_raw.shape:
            raise ValueError(
                f"CSV row {row.csv_row_number}: Ours/TTRV grid shape mismatch "
                f"{ours_raw.shape} vs {ttrv_raw.shape}"
            )
        ours_display = _mass_preserving_spatial_temperature(
            ours_raw, display_temperature
        )
        ttrv_display = _mass_preserving_spatial_temperature(
            ttrv_raw, display_temperature
        )
        vmax = max(float(ours_display.max()), float(ttrv_display.max()))
        if not math.isfinite(vmax) or vmax <= 0.0:
            raise ValueError(
                f"CSV row {row.csv_row_number}: shared transformed vmax must be positive, got {vmax}"
            )
        original = _open_oriented_rgb(image_path)
        prepared[row.case_id] = PreparedCase(
            row=row,
            image_path=image_path,
            npz_ours=npz_ours,
            npz_ttrv=npz_ttrv,
            ours_raw=ours_raw,
            ttrv_raw=ttrv_raw,
            ours_display=ours_display,
            ttrv_display=ttrv_display,
            display_temperature=display_temperature,
            vmax=vmax,
            original_size=original.size,
        )
    return prepared


def _viridis_fraction(values: np.ndarray) -> np.ndarray:
    import cv2

    normalized = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    intensities = np.round(normalized * 255.0).astype(np.uint8)
    colors = cv2.applyColorMap(intensities, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)


def _overlay(original: Any, raw: np.ndarray, vmax: float, alpha: float):
    from PIL import Image

    normalized = np.clip(np.asarray(raw, dtype=np.float64) / vmax, 0.0, 1.0)
    heat = Image.fromarray(_viridis_fraction(normalized))
    heat = heat.resize(original.size, resample=Image.Resampling.BILINEAR)
    return Image.blend(original, heat, alpha=alpha)


def _draw_colorbar(canvas: Any, x: int, y: int, height: int, vmax: float) -> None:
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(canvas)
    fractions = np.linspace(1.0, 0.0, max(2, height), dtype=np.float64).reshape(-1, 1)
    gradient = np.repeat(fractions, 28, axis=1)
    canvas.paste(Image.fromarray(_viridis_fraction(gradient)), (x, y))
    draw.rectangle((x, y, x + 27, y + height - 1), outline="black", width=1)
    font = _helper._font(13)
    draw.text((x + 34, y - 6), f"{vmax:.4g}", fill="black", font=font)
    draw.text((x + 34, y + height - 16), "0", fill="black", font=font)
    draw.text(
        (x - 4, y + height + 5),
        "shared transformed scale",
        fill=(70, 70, 70),
        font=font,
    )


def _render_side_by_side(
    case: PreparedCase,
    original: Any,
    ours_overlay: Any,
    ttrv_overlay: Any,
    panel_max_width: int,
    panel_max_height: int,
):
    from PIL import Image, ImageDraw

    values = case.row.values
    display_size = _helper._contained_size(original.size, panel_max_width, panel_max_height)
    panels = [
        original.resize(display_size, resample=Image.Resampling.LANCZOS),
        ours_overlay.resize(display_size, resample=Image.Resampling.LANCZOS),
        ttrv_overlay.resize(display_size, resample=Image.Resampling.LANCZOS),
    ]
    padding = 16
    panel_width = display_size[0] + 2 * padding
    bar_width = 126
    canvas_width = panel_width * 3 + bar_width
    wrap_width = max(48, (canvas_width - 34) // 8)
    logical_lines = [
        (
            f"{values['dataset']}/{case.row.case_id} | quality rank={case.row.quality_rank} "
            f"score={case.row.quality_score:.5g} | contrast rank={case.row.contrast_rank} "
            f"score={case.row.contrast_score:.5g}"
        ),
        f"Question: {_compact(values['question'])}",
        (
            f"GT teacher-forced target: {values['target_label']}. {_compact(values['target_option_text'])} | "
            f"Ours retained={_compact(values['ours_prediction'])} | "
            f"TTRV retained={_compact(values['ttrv_prediction'])} | "
            f"outcome={_compact(values['outcome_tier'])} (annotation only)"
        ),
        (
            f"query={FORMAL_QUERY} | head={FORMAL_HEAD} | layers={list(FORMAL_LAYERS)} (0-based) | "
            f"display-only T={_temperature_text(case.display_temperature)}; mass-preserving spatial sharpening | "
            f"shared transformed vmin=0, vmax={case.vmax:.5g}"
        ),
        "Ranking/quality/contrast metrics use saved raw T=1; temperature changes display only.",
        f"Probe semantics: {QUERY_SEMANTICS}",
    ]
    lines: list[str] = []
    for line in logical_lines:
        lines.extend(textwrap.wrap(line, width=wrap_width) or [""])
    line_height = 24
    header_height = 18 + len(lines) * line_height
    label_height = 38
    footer_height = 32
    canvas_height = header_height + label_height + display_size[1] + footer_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _helper._font(20, bold=True)
    body_font = _helper._font(15)
    for index, line in enumerate(lines):
        draw.text(
            (14, 9 + index * line_height),
            line,
            fill="black",
            font=title_font if index == 0 else body_font,
        )
    labels = (
        "Original",
        (
            f"Ours overlay | displayed mass={float(case.ours_display.sum()):.5g} "
            f"(raw mass preserved)"
        ),
        (
            f"TTRV overlay | displayed mass={float(case.ttrv_display.sum()):.5g} "
            f"(raw mass preserved)"
        ),
    )
    image_y = header_height + label_height
    for column, (label, panel) in enumerate(zip(labels, panels)):
        x0 = column * panel_width
        draw.text((x0 + padding, header_height + 8), label, fill="black", font=title_font)
        canvas.paste(panel, (x0 + padding, image_y))
    _draw_colorbar(canvas, panel_width * 3 + 12, image_y, display_size[1], case.vmax)
    draw.text(
        (14, canvas_height - 24),
        (
            f"Map=fused local+thumbnail; display-only T={_temperature_text(case.display_temperature)}; "
            "mass-preserving spatial sharpening; ranking/metrics use raw T=1."
        ),
        fill=(70, 70, 70),
        font=_helper._font(13),
    )
    return canvas


def render_union_assets(
    prepared: Mapping[str, PreparedCase],
    destination: Path,
    alpha: float,
    panel_max_width: int,
    panel_max_height: int,
) -> None:
    for case_id in sorted(prepared):
        case = prepared[case_id]
        case_dir = destination / "assets" / _helper._sanitize(case_id)
        case_dir.mkdir(parents=True, exist_ok=False)
        suffix = case.image_path.suffix.lower() or ".img"
        copied_original = case_dir / f"original{suffix}"
        shutil.copy2(case.image_path, copied_original)
        original = _open_oriented_rgb(case.image_path)
        ours = _overlay(original, case.ours_display, case.vmax, alpha)
        ttrv = _overlay(original, case.ttrv_display, case.vmax, alpha)
        temperature_slug = _temperature_slug(case.display_temperature)
        ours_path = case_dir / f"ours_overlay_display_T{temperature_slug}.png"
        ttrv_path = case_dir / f"ttrv_overlay_display_T{temperature_slug}.png"
        side_path = case_dir / f"side_by_side_display_T{temperature_slug}.png"
        ours.save(ours_path, format="PNG", optimize=True)
        ttrv.save(ttrv_path, format="PNG", optimize=True)
        side = _render_side_by_side(
            case, original, ours, ttrv, panel_max_width, panel_max_height
        )
        side.save(side_path, format="PNG", optimize=True)
        case.asset_paths = {
            "original": copied_original.resolve(),
            "ours_overlay": ours_path.resolve(),
            "ttrv_overlay": ttrv_path.resolve(),
            "side_by_side": side_path.resolve(),
        }


def _copy_case_assets(case: PreparedCase, destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=False)
    result: dict[str, Path] = {}
    for name, source in case.asset_paths.items():
        target = destination / Path(source).name
        shutil.copy2(source, target)
        result[name] = target.resolve()
    return result


def _pool_index_row(
    case: PreparedCase,
    pool_name: str,
    pool_order: int,
    copied: Mapping[str, Path],
    rank_sources: Sequence[str],
) -> dict[str, Any]:
    values = case.row.values
    pool_rank = (
        case.row.quality_rank
        if pool_name == "quality"
        else case.row.contrast_rank
        if pool_name == "contrast"
        else pool_order
    )
    return {
        "pool_name": pool_name,
        "pool_order": pool_order,
        "pool_rank": pool_rank,
        "rank_selection_sources": "|".join(rank_sources),
        "quality_rank": case.row.quality_rank,
        "quality_score": case.row.quality_score,
        "contrast_rank": case.row.contrast_rank,
        "contrast_score": case.row.contrast_score,
        "case_id": case.row.case_id,
        "dataset": values["dataset"],
        "question_id": values["question_id"],
        "question": _compact(values["question"]),
        "target_label": values["target_label"],
        "target_option_text": _compact(values["target_option_text"]),
        "ground_truth": values["ground_truth"],
        "ours_prediction": values["ours_prediction"],
        "ttrv_prediction": values["ttrv_prediction"],
        "outcome_tier_annotation_only": values["outcome_tier"],
        "quality_gate_annotation_only": values["quality_gate"],
        "ranking_outcome_blind": values["ranking_outcome_blind"],
        "selection_query_mode": FORMAL_QUERY,
        "selected_head_index_zero_based": FORMAL_HEAD,
        "selected_layer_window_zero_based": list(FORMAL_LAYERS),
        "query_semantics": QUERY_SEMANTICS,
        "map_component": "fused_local_plus_thumbnail",
        "display_temperature": case.display_temperature,
        "display_transform": "m=sum(A); p=A/m; p_T=softmax(log(p+eps)/T); displayed=m*p_T",
        "display_only_mass_preserving_spatial_sharpening": True,
        "temperature_transform_applied": True,
        "spatial_probability_normalization_applied_for_display": True,
        "ranking_and_quality_metrics_temperature": 1.0,
        "ranking_and_quality_metrics_use_raw_maps": True,
        "shared_vmin": 0.0,
        "shared_vmax": case.vmax,
        "ours_raw_attention_mass": float(case.ours_raw.sum(dtype=np.float64)),
        "ttrv_raw_attention_mass": float(case.ttrv_raw.sum(dtype=np.float64)),
        "ours_displayed_attention_mass": float(case.ours_display.sum(dtype=np.float64)),
        "ttrv_displayed_attention_mass": float(case.ttrv_display.sum(dtype=np.float64)),
        "ours_mass_preservation_error": float(
            case.ours_display.sum(dtype=np.float64)
            - case.ours_raw.sum(dtype=np.float64)
        ),
        "ttrv_mass_preservation_error": float(
            case.ttrv_display.sum(dtype=np.float64)
            - case.ttrv_raw.sum(dtype=np.float64)
        ),
        "source_image_path": str(case.image_path),
        "source_npz_ours": str(case.npz_ours),
        "source_npz_ttrv": str(case.npz_ttrv),
        "copied_original": str(copied["original"]),
        "ours_overlay_png": str(copied["ours_overlay"]),
        "ttrv_overlay_png": str(copied["ttrv_overlay"]),
        "side_by_side_png": str(copied["side_by_side"]),
        "source_image_width": case.original_size[0],
        "source_image_height": case.original_size[1],
        "aspect_ratio_preserved": True,
    }


def _render_contact_sheet(
    index_rows: Sequence[Mapping[str, Any]], output_path: Path, columns: int, title: str
) -> None:
    from PIL import Image, ImageDraw

    cell_width, cell_height = 900, 520
    header_height = 104
    grid_rows = int(math.ceil(len(index_rows) / columns))
    canvas = Image.new(
        "RGB", (cell_width * columns, header_height + cell_height * grid_rows), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 9), title, fill="black", font=_helper._font(22, bold=True))
    display_temperature = float(index_rows[0]["display_temperature"])
    draw.text(
        (16, 40),
        (
            f"answer-token-conditioned grounding | h{FORMAL_HEAD} L{list(FORMAL_LAYERS)} | "
            f"display-only T={_temperature_text(display_temperature)}; mass-preserving spatial sharpening"
        ),
        fill=(60, 60, 60),
        font=_helper._font(14),
    )
    draw.text(
        (16, 65),
        (
            "Ranking/metrics use raw T=1; non-causal for observed token; "
            "maps target GT content, not TTRV wrong/unknown text."
        ),
        fill=(130, 0, 0),
        font=_helper._font(14),
    )
    for index, row in enumerate(index_rows):
        column = index % columns
        grid_row = index // columns
        x0 = column * cell_width
        y0 = header_height + grid_row * cell_height
        caption = (
            f"#{row['pool_order']} {row['case_id']} | q-rank={row['quality_rank']} "
            f"c-rank={row['contrast_rank']}"
        )
        draw.text((x0 + 10, y0 + 8), caption, fill="black", font=_helper._font(15, bold=True))
        with Image.open(str(row["side_by_side_png"])) as handle:
            figure = handle.convert("RGB")
            target = _helper._contained_size(figure.size, cell_width - 20, cell_height - 48)
            fitted = figure.resize(target, resample=Image.Resampling.LANCZOS)
        x = x0 + (cell_width - fitted.width) // 2
        y = y0 + 40 + (cell_height - 44 - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        draw.rectangle((x0, y0, x0 + cell_width - 1, y0 + cell_height - 1), outline=(190, 190, 190))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def materialize_pool(
    pool_name: str,
    rows: Sequence[PoolRow],
    prepared: Mapping[str, PreparedCase],
    destination: Path,
    rank_sources: Mapping[str, Sequence[str]],
    contact_columns: int,
) -> dict[str, Any]:
    pool_dir = (
        destination / "merged_review_pool"
        if pool_name == "merged"
        else destination / "rankings" / pool_name
    )
    index_rows: list[dict[str, Any]] = []
    for order, row in enumerate(rows, start=1):
        case = prepared[row.case_id]
        rank_for_name = (
            row.quality_rank
            if pool_name == "quality"
            else row.contrast_rank
            if pool_name == "contrast"
            else order
        )
        case_dir = pool_dir / "cases" / f"{rank_for_name:03d}_{_helper._sanitize(row.case_id)}"
        copied = _copy_case_assets(case, case_dir)
        sources = rank_sources.get(row.case_id, [pool_name])
        index_rows.append(_pool_index_row(case, pool_name, order, copied, sources))
    index_path = pool_dir / "index.csv"
    _helper._write_csv(index_path, index_rows)
    display_temperature = prepared[rows[0].case_id].display_temperature
    contact_path = (
        pool_dir
        / f"contact_sheet_display_T{_temperature_slug(display_temperature)}.png"
    )
    _render_contact_sheet(
        index_rows,
        contact_path,
        contact_columns,
        title=f"Grounding review pool: {pool_name} ({len(index_rows)} cases)",
    )
    return {
        "pool_name": pool_name,
        "case_count": len(index_rows),
        "case_ids": [row.case_id for row in rows],
        "outcome_tier_annotation_counts": dict(
            sorted(Counter(_compact(row.values["outcome_tier"]) for row in rows).items())
        ),
        "quality_gate_annotation_counts": dict(
            sorted(Counter(_compact(row.values["quality_gate"]) for row in rows).items())
        ),
        "index_csv": str(index_path.resolve()),
        "contact_sheet_png": str(contact_path.resolve()),
        "case_root": str((pool_dir / "cases").resolve()),
    }


def _audit_outputs(pool_summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    from PIL import Image

    pngs: list[Path] = []
    originals: list[Path] = []
    index_row_count = 0
    shared_scale_failures: list[str] = []
    mass_preservation_failures: list[str] = []
    aspect_failures: list[str] = []
    for pool in pool_summaries.values():
        index_path = Path(str(pool["index_csv"]))
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != int(pool["case_count"]):
            raise AssertionError(f"Index count mismatch for {index_path}")
        index_row_count += len(rows)
        pngs.append(Path(str(pool["contact_sheet_png"])))
        for row in rows:
            originals.append(Path(row["copied_original"]))
            pngs.extend(
                [Path(row["ours_overlay_png"]), Path(row["ttrv_overlay_png"]), Path(row["side_by_side_png"])]
            )
            if float(row["shared_vmin"]) != 0.0 or float(row["shared_vmax"]) <= 0.0:
                shared_scale_failures.append(str(row["case_id"]))
            for model_name in ("ours", "ttrv"):
                raw_mass = float(row[f"{model_name}_raw_attention_mass"])
                displayed_mass = float(row[f"{model_name}_displayed_attention_mass"])
                if not math.isclose(
                    displayed_mass,
                    raw_mass,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
                    mass_preservation_failures.append(
                        f"{row['case_id']}:{model_name}"
                    )
            with Image.open(row["ours_overlay_png"]) as ours, Image.open(row["ttrv_overlay_png"]) as ttrv:
                expected = (int(row["source_image_width"]), int(row["source_image_height"]))
                if ours.size != expected or ttrv.size != expected:
                    aspect_failures.append(str(row["case_id"]))
    for path in pngs:
        if not path.is_file():
            raise FileNotFoundError(f"Missing rendered PNG: {path}")
        with Image.open(path) as handle:
            handle.verify()
    for path in originals:
        if not path.is_file():
            raise FileNotFoundError(f"Missing copied original: {path}")
        with Image.open(path) as handle:
            handle.verify()
    if shared_scale_failures:
        raise AssertionError(f"Shared scale audit failed: {shared_scale_failures}")
    if mass_preservation_failures:
        raise AssertionError(
            f"Mass-preservation audit failed: {mass_preservation_failures}"
        )
    if aspect_failures:
        raise AssertionError(f"Aspect/size audit failed: {aspect_failures}")
    return {
        "status": "PASS",
        "index_rows_checked": index_row_count,
        "png_files_decoded": len(pngs),
        "copied_originals_decoded": len(originals),
        "shared_scale_failures": 0,
        "mass_preservation_failures": 0,
        "aspect_or_overlay_size_failures": 0,
    }


def build_grounding_visualizations(
    grounding_csv: str | Path,
    output_dir: str | Path,
    rank_types: Sequence[str] = ("quality", "contrast"),
    top_n_per_rank: int = 16,
    merged_max_cases: int = 24,
    display_temperature: float = 1.0,
    alpha: float = 0.55,
    panel_max_width: int = 620,
    panel_max_height: int = 620,
    contact_columns: int = 2,
) -> dict[str, Any]:
    if not rank_types or len(set(rank_types)) != len(rank_types):
        raise ValueError(f"rank_types must be unique and non-empty, got {rank_types}")
    if any(rank_type not in RANK_COLUMNS for rank_type in rank_types):
        raise ValueError(f"Unknown rank_types={rank_types}")
    if top_n_per_rank <= 0 or merged_max_cases <= 0:
        raise ValueError("top_n_per_rank and merged_max_cases must be positive")
    if (
        not math.isfinite(display_temperature)
        or display_temperature <= 0.0
        or display_temperature > 1.0
    ):
        raise ValueError(
            f"display_temperature must satisfy 0 < T <= 1, got {display_temperature}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    if panel_max_width <= 0 or panel_max_height <= 0 or contact_columns <= 0:
        raise ValueError("panel/contact dimensions must be positive")

    csv_path = Path(grounding_csv).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Grounding CSV does not exist: {csv_path}")
    destination = Path(output_dir).expanduser().resolve()
    _helper._ensure_clean_output(destination)
    rows, header = _read_pool_csv(csv_path)
    selected = select_rank_pools(rows, rank_types, top_n_per_rank)
    merged, source_membership = merge_rank_pools(selected, rank_types, merged_max_cases)
    union_by_id = {row.case_id: row for rank_type in rank_types for row in selected[rank_type]}
    prepared = prepare_cases(
        list(union_by_id.values()),
        csv_path,
        display_temperature=display_temperature,
    )

    destination.mkdir(parents=True, exist_ok=True)
    render_union_assets(
        prepared,
        destination,
        alpha=alpha,
        panel_max_width=panel_max_width,
        panel_max_height=panel_max_height,
    )
    pool_summaries: dict[str, dict[str, Any]] = {}
    for rank_type in rank_types:
        membership = {row.case_id: [rank_type] for row in selected[rank_type]}
        pool_summaries[rank_type] = materialize_pool(
            rank_type,
            selected[rank_type],
            prepared,
            destination,
            membership,
            contact_columns,
        )
    pool_summaries["merged"] = materialize_pool(
        "merged",
        merged,
        prepared,
        destination,
        source_membership,
        contact_columns,
    )
    audit = _audit_outputs(pool_summaries)
    overlap = set(selected[rank_types[0]]) if False else set()
    if len(rank_types) >= 2:
        overlap = set(row.case_id for row in selected[rank_types[0]])
        for rank_type in rank_types[1:]:
            overlap &= set(row.case_id for row in selected[rank_type])
    summary = {
        "status": "PASS",
        "grounding_csv": str(csv_path),
        "grounding_csv_sha256": _helper._sha256(csv_path),
        "grounding_csv_header": header,
        "source_rows": len(rows),
        "rank_integrity": {
            "quality_rank": f"verified permutation 1..{len(rows)}",
            "contrast_rank": f"verified permutation 1..{len(rows)}",
            "ranking_outcome_blind_all_rows": True,
        },
        "selection": {
            "rank_types": list(rank_types),
            "top_n_per_rank": top_n_per_rank,
            "rank_columns": {rank_type: RANK_COLUMNS[rank_type] for rank_type in rank_types},
            "selection_fields_used": [
                "quality_rank or contrast_rank",
                "case_id only for deterministic tie-break",
            ],
            "outcome_or_correctness_used_for_selection": False,
            "per_rank_counts": {rank_type: len(selected[rank_type]) for rank_type in rank_types},
            "all_rank_intersection_count": len(overlap),
            "merged_policy": "deterministic round-robin in --rank-types order, deduplicated by case_id",
            "merged_max_cases": merged_max_cases,
            "merged_case_count": len(merged),
        },
        "fixed_setting": {
            "selection_query_mode": FORMAL_QUERY,
            "selected_head_index_zero_based": FORMAL_HEAD,
            "selected_layer_center_zero_based": FORMAL_CENTER,
            "selected_layer_window_zero_based": list(FORMAL_LAYERS),
            "shared_across_models_cases_and_ranks": True,
        },
        "interpretation_boundary": {
            "query_semantics": QUERY_SEMANTICS,
            "target_source": "GT teacher-forced option content",
            "retained_predictions": "annotation only; maps do not target TTRV wrong/unknown answer text",
            "same_token_causality": False,
        },
        "display_policy": {
            "map_component": "fused_local_plus_thumbnail",
            "layer_aggregation_before_display": "arithmetic mean of saved raw head grids",
            "display_temperature": display_temperature,
            "display_transform": "m=sum(A); p=A/m; p_T=softmax(log(p+eps)/T); displayed=m*p_T",
            "display_only_mass_preserving_spatial_sharpening": True,
            "ranking_and_quality_metrics_use_raw_T1": True,
            "temperature_does_not_change_rank_selection_or_metrics": True,
            "color_scale": (
                "per case Ours/TTRV share vmin=0 and exact pairwise vmax "
                "after the display-only transform"
            ),
            "overlay_alpha": alpha,
            "aspect_ratio": "preserved; overlay dimensions equal EXIF-oriented source dimensions",
        },
        "pools": pool_summaries,
        "output_audit": audit,
        "outputs": {
            "output_dir": str(destination),
            "union_asset_root": str((destination / "assets").resolve()),
            "summary_json": str((destination / "summary.json").resolve()),
        },
        "cpu_only": True,
        "model_inference_run": False,
    }
    _helper._write_json(destination / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounding-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank-types", default="quality,contrast")
    parser.add_argument("--top-n-per-rank", type=int, default=16)
    parser.add_argument("--merged-max-cases", type=int, default=24)
    parser.add_argument(
        "--display-temperature",
        type=float,
        default=1.0,
        help=(
            "Display-only spatial temperature, 0<T<=1. Preserves each raw map's "
            "mass; ranking and metrics remain raw T=1."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--panel-max-width", type=int, default=620)
    parser.add_argument("--panel-max-height", type=int, default=620)
    parser.add_argument("--contact-columns", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build_grounding_visualizations(
        grounding_csv=args.grounding_csv,
        output_dir=args.output_dir,
        rank_types=_parse_rank_types(args.rank_types),
        top_n_per_rank=args.top_n_per_rank,
        merged_max_cases=args.merged_max_cases,
        display_temperature=args.display_temperature,
        alpha=args.alpha,
        panel_max_width=args.panel_max_width,
        panel_max_height=args.panel_max_height,
        contact_columns=args.contact_columns,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
