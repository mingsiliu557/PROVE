#!/usr/bin/env python3
"""Render shortlisted lexical answer-content attention contrasts.

This CPU-only renderer consumes ``lexical_contrast_ranked.csv`` produced by
the answer-content screen.  It does not search for a query, head, or layer
window: those settings must already be present in the CSV and must be fixed
across every quality-gated row selected for rendering.

For each selected case the script writes an exact copy of the source image,
two raw-attention overlays, and an annotated side-by-side PNG.  Ours and TTRV
always share ``vmin=0`` and the same per-case ``vmax``.  Saved arrays are
arithmetic-mean aggregated across the selected layers and displayed at raw
``T=1``; no temperature or spatial probability normalization is applied.

Example (run only after the ranked CSV exists)::

    python attention_visualization/build_lexical_contrast_visualizations.py \
      --ranked-csv /path/to/lexical_contrast_ranked.csv \
      --output-dir /path/to/new_render_directory \
      --top-n 12

The output directory must be absent or empty, which prevents accidental
overwriting of an earlier visualization run.
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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_QUERY = "content_answer_state_heads"
ALLOWED_QUERIES = (
    "content_answer_state_heads",
    "content_prediction_query_heads",
)

# These are the columns the renderer actually consumes.  The screen may add
# quantitative diagnostic columns without requiring renderer changes.
REQUIRED_COLUMNS = (
    "rank",
    "tier_rank",
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
    "analysis_tier",
    "eligible",
    "selection_query_mode",
    "selected_head_index_zero_based",
    "selected_layer_center_zero_based",
    "selected_layer_window_zero_based",
    "rank_score",
    "quality_gate",
    "exclusion_reason",
    "npz_ours",
    "npz_ttrv",
)

PASS_VALUES = frozenset({"1", "true", "yes", "y", "pass", "passed", "keep", "eligible"})
FAIL_VALUES = frozenset({"0", "false", "no", "n", "fail", "failed", "drop", "excluded"})
ABSENT_VALUES = frozenset({"", "na", "n/a", "none", "null", "absent", "missing"})
UNKNOWN_VALUES = frozenset({"unknown", "undetermined", "unavailable"})


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty render index")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(value, ensure_ascii=False, default=_json_default)
                    if isinstance(value, (list, tuple, dict))
                    else value
                    for field, value in row.items()
                }
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize(value: Any) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return result[:120] or "case"


def _compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _parse_integer(value: Any, field: str, row_number: int, minimum: int = 0) -> int:
    text = str(value or "").strip()
    if not re.fullmatch(r"[+-]?\d+", text):
        raise ValueError(f"CSV row {row_number}: {field} must be an integer, got {value!r}")
    parsed = int(text)
    if parsed < minimum:
        raise ValueError(
            f"CSV row {row_number}: {field} must be >= {minimum}, got {parsed}"
        )
    return parsed


def _parse_float(value: Any, field: str, row_number: int) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CSV row {row_number}: {field} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"CSV row {row_number}: {field} must be finite, got {value!r}")
    return parsed


def _parse_bool(value: Any, field: str, row_number: int) -> bool:
    normalized = str(value or "").strip().casefold()
    if normalized in PASS_VALUES:
        return True
    if normalized in FAIL_VALUES:
        return False
    raise ValueError(
        f"CSV row {row_number}: {field} must be boolean/pass-fail; got {value!r}"
    )


def _quality_gate_state(value: Any, row_number: int) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized in PASS_VALUES:
        return "pass"
    if normalized in FAIL_VALUES:
        return "fail"
    if normalized in ABSENT_VALUES:
        return "absent"
    if normalized in UNKNOWN_VALUES:
        return "unknown"
    raise ValueError(
        f"CSV row {row_number}: unknown quality_gate={value!r}; expected pass/fail, "
        "an explicit absent value, or 'unknown'"
    )


def _prediction_status(value: Any) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized in ABSENT_VALUES:
        return "absent"
    if normalized in UNKNOWN_VALUES:
        return "unknown"
    return "valid"


def _parse_layers(value: Any, row_number: int) -> tuple[int, ...]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"CSV row {row_number}: selected_layer_window_zero_based must be a JSON list "
            f"such as [12, 13, 14], got {value!r}"
        ) from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(
            f"CSV row {row_number}: selected_layer_window_zero_based must be a non-empty JSON list"
        )
    layers: list[int] = []
    for index, layer in enumerate(parsed):
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ValueError(
                f"CSV row {row_number}: layer[{index}] must be a non-negative integer, got {layer!r}"
            )
        layers.append(layer)
    if layers != sorted(set(layers)):
        raise ValueError(
            f"CSV row {row_number}: layer window must be strictly increasing and unique, got {layers}"
        )
    return tuple(layers)


def _resolve_input(value: Any, csv_parent: Path, field: str, row_number: int) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"CSV row {row_number}: {field} is empty")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = csv_parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CSV row {row_number}: {field} file does not exist: {path}")
    return path


@dataclass(frozen=True)
class FixedSetting:
    query: str
    head: int
    center: int
    layers: tuple[int, ...]

    def npz_key(self, layer: int) -> str:
        return f"{self.query}__layer_{layer:02d}_head_{self.head:02d}"


@dataclass
class RankedCandidate:
    source_row: dict[str, str]
    csv_row_number: int
    rank: int
    tier_rank: int
    rank_score: float
    setting: FixedSetting
    image_path: Path
    npz_ours: Path
    npz_ttrv: Path


def _read_ranked_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        if not header:
            raise ValueError(f"Ranked CSV has no header: {path}")
        duplicates = sorted(field for field, count in Counter(header).items() if count > 1)
        if duplicates:
            raise ValueError(f"Ranked CSV has duplicate columns: {duplicates}")
        missing = sorted(set(REQUIRED_COLUMNS) - set(header))
        if missing:
            raise ValueError(
                "Ranked CSV schema mismatch: missing required columns "
                f"{missing}. Expected content_screen_design renderer contract."
            )
        rows = [dict(row) for row in reader if any(str(value or "").strip() for value in row.values())]
    if not rows:
        raise ValueError(f"Ranked CSV contains no data rows: {path}")
    return rows, header


def _setting_from_row(row: Mapping[str, str], row_number: int) -> FixedSetting:
    query = str(row["selection_query_mode"] or "").strip()
    if query not in ALLOWED_QUERIES:
        raise ValueError(
            f"CSV row {row_number}: selection_query_mode={query!r} is unsupported; "
            f"expected one of {list(ALLOWED_QUERIES)}"
        )
    head = _parse_integer(
        row["selected_head_index_zero_based"],
        "selected_head_index_zero_based",
        row_number,
    )
    center = _parse_integer(
        row["selected_layer_center_zero_based"],
        "selected_layer_center_zero_based",
        row_number,
    )
    layers = _parse_layers(row["selected_layer_window_zero_based"], row_number)
    if center not in layers:
        raise ValueError(
            f"CSV row {row_number}: selected layer center {center} is absent from window {list(layers)}"
        )
    if len(layers) % 2 == 1 and layers[len(layers) // 2] != center:
        raise ValueError(
            f"CSV row {row_number}: selected layer center {center} is not the middle of window {list(layers)}"
        )
    return FixedSetting(query=query, head=head, center=center, layers=layers)


def _select_candidates(
    rows: Sequence[dict[str, str]],
    csv_path: Path,
    top_n: int,
    expected_query: str,
) -> tuple[list[RankedCandidate], Counter[str]]:
    gate_states: Counter[str] = Counter()
    passing: list[tuple[int, int, dict[str, str]]] = []
    seen_cases: dict[str, int] = {}
    for offset, row in enumerate(rows, start=2):
        case_id = str(row["case_id"] or "").strip()
        if not case_id:
            raise ValueError(f"CSV row {offset}: case_id is empty")
        if case_id in seen_cases:
            raise ValueError(
                f"CSV row {offset}: duplicate case_id={case_id!r}; first seen at row {seen_cases[case_id]}"
            )
        seen_cases[case_id] = offset
        state = _quality_gate_state(row["quality_gate"], offset)
        gate_states[state] += 1
        if state != "pass":
            continue
        if not _parse_bool(row["eligible"], "eligible", offset):
            raise ValueError(
                f"CSV row {offset}: quality_gate passes but eligible={row['eligible']!r}"
            )
        rank = _parse_integer(row["rank"], "rank", offset, minimum=1)
        passing.append((rank, offset, row))

    passing.sort(key=lambda item: (item[0], item[1], str(item[2]["case_id"])))
    chosen_rows = passing[:top_n]
    if not chosen_rows:
        raise ValueError(
            f"No quality_gate-passing rows are available in {csv_path}; gate counts={dict(gate_states)}"
        )

    selected: list[RankedCandidate] = []
    for rank, row_number, row in chosen_rows:
        setting = _setting_from_row(row, row_number)
        if setting.query != expected_query:
            raise ValueError(
                f"CSV row {row_number}: selection_query_mode={setting.query!r} does not match "
                f"--expected-query={expected_query!r}; the renderer will not choose a substitute"
            )
        selected.append(
            RankedCandidate(
                source_row=row,
                csv_row_number=row_number,
                rank=rank,
                tier_rank=_parse_integer(row["tier_rank"], "tier_rank", row_number, minimum=1),
                rank_score=_parse_float(row["rank_score"], "rank_score", row_number),
                setting=setting,
                image_path=_resolve_input(row["image_path"], csv_path.parent, "image_path", row_number),
                npz_ours=_resolve_input(row["npz_ours"], csv_path.parent, "npz_ours", row_number),
                npz_ttrv=_resolve_input(row["npz_ttrv"], csv_path.parent, "npz_ttrv", row_number),
            )
        )

    settings = {candidate.setting for candidate in selected}
    if len(settings) != 1:
        details = [
            {
                "case_id": candidate.source_row["case_id"],
                "query": candidate.setting.query,
                "head": candidate.setting.head,
                "center": candidate.setting.center,
                "layers": list(candidate.setting.layers),
            }
            for candidate in selected
        ]
        raise ValueError(
            "Quality-gated top-N rows do not share one fixed query/head/layer window. "
            "The renderer will not select settings itself. Rows="
            + json.dumps(details, ensure_ascii=False)
        )
    return selected, gate_states


def _validate_grid(value: np.ndarray, path: Path, key: str) -> np.ndarray:
    grid = np.asarray(value, dtype=np.float64)
    if grid.ndim != 2:
        raise ValueError(f"{path}: NPZ key {key!r} must be 2-D, got shape={grid.shape}")
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{path}: NPZ key {key!r} contains NaN or infinity")
    minimum = float(grid.min())
    if minimum < -1e-8:
        raise ValueError(f"{path}: NPZ key {key!r} contains negative attention min={minimum:.6g}")
    return np.maximum(grid, 0.0)


def _load_raw_window(path: Path, setting: FixedSetting) -> np.ndarray:
    grids: list[np.ndarray] = []
    with np.load(path, allow_pickle=False) as arrays:
        missing = [setting.npz_key(layer) for layer in setting.layers if setting.npz_key(layer) not in arrays]
        if missing:
            examples = [key for key in arrays.files if key.startswith(setting.query)][:8]
            raise KeyError(
                f"{path}: missing fixed-setting NPZ keys {missing}; "
                f"available {setting.query!r} examples={examples}"
            )
        for layer in setting.layers:
            key = setting.npz_key(layer)
            grids.append(_validate_grid(arrays[key], path, key))
    shapes = {grid.shape for grid in grids}
    if len(shapes) != 1:
        raise ValueError(f"{path}: fixed layer-window grids have inconsistent shapes={sorted(shapes)}")
    # Existing Obs3 screens define a layer-window map as the arithmetic mean
    # of raw saved attention grids.  Do not normalize or apply temperature.
    return np.mean(grids, axis=0, dtype=np.float64)


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _viridis(rgb_fraction: np.ndarray) -> np.ndarray:
    import cv2

    values = np.round(np.clip(rgb_fraction, 0.0, 1.0) * 255).astype(np.uint8)
    colors = cv2.applyColorMap(values, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)


def _make_overlay(original: Any, grid: np.ndarray, vmax: float, alpha: float):
    from PIL import Image

    normalized = np.clip(np.asarray(grid, dtype=np.float64) / vmax, 0.0, 1.0)
    heat = Image.fromarray(_viridis(np.round(normalized * 255).astype(np.uint8)))
    heat = heat.resize(original.size, resample=Image.Resampling.BILINEAR)
    return Image.blend(original, heat, alpha=alpha)


def _contained_size(size: tuple[int, int], max_width: int, max_height: int) -> tuple[int, int]:
    width, height = size
    scale = min(max_width / width, max_height / height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _draw_colorbar(canvas: Any, x: int, y: int, height: int, vmax: float) -> None:
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(canvas)
    fractions = np.linspace(1.0, 0.0, max(2, height), dtype=np.float64).reshape(-1, 1)
    values = np.repeat(fractions, 28, axis=1)
    canvas.paste(Image.fromarray(_viridis(np.round(values * 255).astype(np.uint8))), (x, y))
    draw.rectangle((x, y, x + 27, y + height - 1), outline="black", width=1)
    small = _font(13)
    draw.text((x + 34, y - 7), f"{vmax:.4g}", fill="black", font=small)
    draw.text((x + 34, y + height - 16), "0", fill="black", font=small)
    draw.text((x - 5, y + height + 6), "shared raw scale", fill=(60, 60, 60), font=small)


def _render_side_by_side(
    candidate: RankedCandidate,
    original: Any,
    ours_overlay: Any,
    ttrv_overlay: Any,
    vmax: float,
    ours_mass: float,
    ttrv_mass: float,
    panel_max_width: int,
    panel_max_height: int,
) -> Any:
    from PIL import Image, ImageDraw

    row = candidate.source_row
    display_size = _contained_size(original.size, panel_max_width, panel_max_height)
    panels = [
        original.resize(display_size, resample=Image.Resampling.LANCZOS),
        ours_overlay.resize(display_size, resample=Image.Resampling.LANCZOS),
        ttrv_overlay.resize(display_size, resample=Image.Resampling.LANCZOS),
    ]
    panel_padding = 16
    panel_width = display_size[0] + 2 * panel_padding
    bar_width = 125
    canvas_width = panel_width * 3 + bar_width
    question = _compact_text(row["question"])
    target_text = _compact_text(row["target_option_text"])
    gt_text = _compact_text(row["ground_truth"])
    ours_text = _compact_text(row["ours_prediction"])
    ttrv_text = _compact_text(row["ttrv_prediction"])
    annotation_lines = [
        f"Rank {candidate.rank} | {row['dataset']}/{row['case_id']} | score={candidate.rank_score:.5g} | tier={row['analysis_tier']}",
        *[f"Question: {line}" for line in textwrap.wrap(question, width=150) or [""]],
        (
            f"GT: {gt_text} ({row['target_label']}. {target_text}) | "
            f"Ours: {ours_text} [{_prediction_status(ours_text)}] | "
            f"TTRV: {ttrv_text} [{_prediction_status(ttrv_text)}] | outcome={row['outcome_tier']}"
        ),
        (
            f"query={candidate.setting.query} | head={candidate.setting.head} (0-based) | "
            f"layers={list(candidate.setting.layers)} | raw T=1, no temperature | shared vmin=0, vmax={vmax:.5g}"
        ),
    ]
    title_font = _font(20, bold=True)
    body_font = _font(15)
    line_height = 24
    header_height = 18 + line_height * len(annotation_lines)
    label_height = 38
    footer_height = 32
    canvas_height = header_height + label_height + display_size[1] + footer_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(annotation_lines):
        draw.text(
            (14, 9 + index * line_height),
            line,
            fill="black",
            font=title_font if index == 0 else body_font,
        )
    labels = (
        "Original",
        f"Ours overlay | raw mass={ours_mass:.5g}",
        f"TTRV overlay | raw mass={ttrv_mass:.5g}",
    )
    image_y = header_height + label_height
    for column, (label, panel) in enumerate(zip(labels, panels)):
        x0 = column * panel_width
        draw.text((x0 + panel_padding, header_height + 8), label, fill="black", font=title_font)
        canvas.paste(panel, (x0 + panel_padding, image_y))
    _draw_colorbar(canvas, panel_width * 3 + 12, image_y, display_size[1], vmax)
    draw.text(
        (14, canvas_height - 24),
        "Layer aggregation: arithmetic mean of saved raw head grids; image geometry is never stretched.",
        fill=(70, 70, 70),
        font=_font(13),
    )
    return canvas


def _render_contact_sheet(
    rendered: Sequence[Mapping[str, Any]],
    output_path: Path,
    columns: int,
    setting: FixedSetting,
) -> None:
    from PIL import Image, ImageDraw

    cell_width, cell_height = 900, 520
    header_height = 76
    rows = int(math.ceil(len(rendered) / columns))
    canvas = Image.new("RGB", (cell_width * columns, header_height + cell_height * rows), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (16, 10),
        "Lexical answer-content contrast candidates | raw T=1 | no temperature",
        fill="black",
        font=_font(22, bold=True),
    )
    draw.text(
        (16, 42),
        f"fixed query={setting.query} | head={setting.head} | layers={list(setting.layers)} | per-case shared Ours/TTRV scale",
        fill=(60, 60, 60),
        font=_font(14),
    )
    for index, row in enumerate(rendered):
        column = index % columns
        grid_row = index // columns
        x0 = column * cell_width
        y0 = header_height + grid_row * cell_height
        caption = f"#{row['render_order']} rank={row['rank']} | {row['dataset']}/{row['case_id']}"
        draw.text((x0 + 10, y0 + 8), caption, fill="black", font=_font(15, bold=True))
        with Image.open(row["side_by_side_png"]) as handle:
            figure = handle.convert("RGB")
            target = _contained_size(figure.size, cell_width - 20, cell_height - 48)
            fitted = figure.resize(target, resample=Image.Resampling.LANCZOS)
        x = x0 + (cell_width - fitted.width) // 2
        y = y0 + 40 + (cell_height - 44 - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        draw.rectangle((x0, y0, x0 + cell_width - 1, y0 + cell_height - 1), outline=(190, 190, 190))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def _ensure_clean_output(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {path}")
    if path.is_dir() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; choose a new directory to avoid overwriting: {path}"
        )


def build_visualizations(
    ranked_csv: str | Path,
    output_dir: str | Path,
    top_n: int = 12,
    expected_query: str = DEFAULT_QUERY,
    alpha: float = 0.55,
    panel_max_width: int = 620,
    panel_max_height: int = 620,
    contact_columns: int = 2,
) -> dict[str, Any]:
    """Validate, select, and render quality-gated rows from the ranked CSV."""

    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")
    if expected_query not in ALLOWED_QUERIES:
        raise ValueError(f"expected_query must be one of {list(ALLOWED_QUERIES)}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if panel_max_width <= 0 or panel_max_height <= 0:
        raise ValueError("panel dimensions must be positive")
    if contact_columns <= 0:
        raise ValueError("contact_columns must be positive")

    csv_path = Path(ranked_csv).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Ranked CSV does not exist: {csv_path}")
    destination = Path(output_dir).expanduser().resolve()
    _ensure_clean_output(destination)
    rows, header = _read_ranked_csv(csv_path)
    selected, gate_states = _select_candidates(rows, csv_path, top_n, expected_query)
    setting = selected[0].setting

    # Validate every selected NPZ before creating output files.  This avoids a
    # half-rendered directory on schema/key errors.
    prepared: list[tuple[RankedCandidate, np.ndarray, np.ndarray]] = []
    for candidate in selected:
        ours_raw = _load_raw_window(candidate.npz_ours, candidate.setting)
        ttrv_raw = _load_raw_window(candidate.npz_ttrv, candidate.setting)
        if ours_raw.shape != ttrv_raw.shape:
            raise ValueError(
                f"CSV row {candidate.csv_row_number}: Ours/TTRV map shapes differ: "
                f"{ours_raw.shape} vs {ttrv_raw.shape}"
            )
        vmax = max(float(ours_raw.max()), float(ttrv_raw.max()))
        if not math.isfinite(vmax) or vmax <= 0.0:
            raise ValueError(
                f"CSV row {candidate.csv_row_number}: shared raw vmax must be positive, got {vmax}"
            )
        prepared.append((candidate, ours_raw, ttrv_raw))

    destination.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, Any]] = []
    for render_order, (candidate, ours_raw, ttrv_raw) in enumerate(prepared, start=1):
        from PIL import Image

        row = candidate.source_row
        case_dir = destination / "cases" / f"{render_order:03d}_{_sanitize(row['case_id'])}"
        case_dir.mkdir(parents=True, exist_ok=False)
        source_suffix = candidate.image_path.suffix or ".img"
        copied_original = case_dir / f"original{source_suffix.lower()}"
        shutil.copy2(candidate.image_path, copied_original)
        with Image.open(candidate.image_path) as handle:
            original = handle.convert("RGB").copy()
        vmax = max(float(ours_raw.max()), float(ttrv_raw.max()))
        ours_overlay = _make_overlay(original, ours_raw, vmax, alpha)
        ttrv_overlay = _make_overlay(original, ttrv_raw, vmax, alpha)
        ours_overlay_path = case_dir / "ours_overlay_raw_T1.png"
        ttrv_overlay_path = case_dir / "ttrv_overlay_raw_T1.png"
        ours_overlay.save(ours_overlay_path, format="PNG", optimize=True)
        ttrv_overlay.save(ttrv_overlay_path, format="PNG", optimize=True)
        side_by_side = _render_side_by_side(
            candidate,
            original,
            ours_overlay,
            ttrv_overlay,
            vmax,
            float(ours_raw.sum()),
            float(ttrv_raw.sum()),
            panel_max_width,
            panel_max_height,
        )
        side_by_side_path = case_dir / "side_by_side_raw_T1.png"
        side_by_side.save(side_by_side_path, format="PNG", optimize=True)
        rendered.append(
            {
                "render_order": render_order,
                "rank": candidate.rank,
                "tier_rank": candidate.tier_rank,
                "case_id": row["case_id"],
                "dataset": row["dataset"],
                "question_id": row["question_id"],
                "question": _compact_text(row["question"]),
                "ground_truth": row["ground_truth"],
                "target_label": row["target_label"],
                "target_option_text": _compact_text(row["target_option_text"]),
                "ours_prediction": row["ours_prediction"],
                "ours_prediction_status": _prediction_status(row["ours_prediction"]),
                "ttrv_prediction": row["ttrv_prediction"],
                "ttrv_prediction_status": _prediction_status(row["ttrv_prediction"]),
                "outcome_tier": row["outcome_tier"],
                "analysis_tier": row["analysis_tier"],
                "quality_gate": row["quality_gate"],
                "rank_score": candidate.rank_score,
                "selection_query_mode": setting.query,
                "selected_head_index_zero_based": setting.head,
                "selected_layer_center_zero_based": setting.center,
                "selected_layer_window_zero_based": list(setting.layers),
                "display_temperature": 1.0,
                "temperature_transform_applied": False,
                "layer_aggregation": "arithmetic_mean_raw_attention",
                "shared_vmin": 0.0,
                "shared_vmax": vmax,
                "ours_raw_attention_mass": float(ours_raw.sum()),
                "ttrv_raw_attention_mass": float(ttrv_raw.sum()),
                "source_image_path": str(candidate.image_path),
                "source_npz_ours": str(candidate.npz_ours),
                "source_npz_ttrv": str(candidate.npz_ttrv),
                "copied_original": str(copied_original.resolve()),
                "ours_overlay_png": str(ours_overlay_path.resolve()),
                "ttrv_overlay_png": str(ttrv_overlay_path.resolve()),
                "side_by_side_png": str(side_by_side_path.resolve()),
                "source_image_width": original.width,
                "source_image_height": original.height,
                "overlay_width": ours_overlay.width,
                "overlay_height": ours_overlay.height,
                "aspect_ratio_preserved": ours_overlay.size == original.size == ttrv_overlay.size,
            }
        )

    index_path = destination / "index.csv"
    _write_csv(index_path, rendered)
    contact_path = destination / "contact_sheet_raw_T1.png"
    _render_contact_sheet(rendered, contact_path, contact_columns, setting)
    summary = {
        "status": "PASS",
        "ranked_csv": str(csv_path),
        "ranked_csv_sha256": _sha256(csv_path),
        "ranked_csv_header": header,
        "source_rows": len(rows),
        "quality_gate_state_counts": dict(sorted(gate_states.items())),
        "top_n": top_n,
        "rendered_cases": len(rendered),
        "fixed_setting": {
            "selection_query_mode": setting.query,
            "selected_head_index_zero_based": setting.head,
            "selected_layer_center_zero_based": setting.center,
            "selected_layer_window_zero_based": list(setting.layers),
        },
        "display_policy": {
            "attention": "raw saved attention",
            "layer_aggregation": "arithmetic mean",
            "temperature": 1.0,
            "temperature_transform_applied": False,
            "spatial_probability_normalization_applied": False,
            "color_scale": "per case: Ours/TTRV share vmin=0 and exact pairwise raw vmax",
            "overlay_alpha": alpha,
            "aspect_ratio": "preserved; overlay PNG dimensions exactly equal source image dimensions",
        },
        "prediction_status_counts": {
            "ours": dict(sorted(Counter(row["ours_prediction_status"] for row in rendered).items())),
            "ttrv": dict(sorted(Counter(row["ttrv_prediction_status"] for row in rendered).items())),
        },
        "outputs": {
            "index_csv": str(index_path.resolve()),
            "contact_sheet_png": str(contact_path.resolve()),
            "case_root": str((destination / "cases").resolve()),
        },
        "formal_ranked_render_executed": True,
    }
    summary_path = destination / "summary.json"
    summary["outputs"]["summary_json"] = str(summary_path.resolve())
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument(
        "--expected-query",
        choices=ALLOWED_QUERIES,
        default=DEFAULT_QUERY,
        help="Validate the screen-selected fixed query; never substitutes another query.",
    )
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--panel-max-width", type=int, default=620)
    parser.add_argument("--panel-max-height", type=int, default=620)
    parser.add_argument("--contact-columns", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build_visualizations(
        ranked_csv=args.ranked_csv,
        output_dir=args.output_dir,
        top_n=args.top_n,
        expected_query=args.expected_query,
        alpha=args.alpha,
        panel_max_width=args.panel_max_width,
        panel_max_height=args.panel_max_height,
        contact_columns=args.contact_columns,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
