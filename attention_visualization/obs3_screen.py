#!/usr/bin/env python3
"""Strict Obs3 screening for saved InternVL image-attention layers.

This script consumes ``attention_records.jsonl`` plus the NPZ files produced by
``attention_viz.py extract --save-individual-layers``.  It is deliberately
separate from extraction: it never loads a checkpoint and never runs a GPU.

Selection is quantitative at raw temperature T=1 only.  A fixed T=0.5 view is
rendered beside the raw view as *display-only* sharpening.  It cannot affect
the selected layer window, tier, or rank.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ATTENTION_MODE = "first_decision_query"
REQUIRED_MODELS = ("base", "ttrv", "ours")
EXPECTED_LAYERS = tuple(range(28))
DISPLAY_TEMPERATURES = (1.0, 0.5)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, default=_json_default) + "\n")


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return cleaned[:120] or "case"


def _distribution(grid: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(grid, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D attention grid, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Attention grid contains NaN or infinity")
    # Tiny negative values can arise from serialization/conversion noise.  A
    # materially negative attention is invalid rather than silently repaired.
    if float(values.min()) < -1e-8:
        raise ValueError(f"Attention grid contains negative values (min={values.min():.3g})")
    values = np.maximum(values, 0.0)
    image_mass = float(values.sum())
    if image_mass <= 0.0:
        raise ValueError("Attention grid has zero image-attention mass")
    return values / image_mass, image_mass


def display_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Conditionally sharpen a spatial distribution without preserving raw mass."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    p, _ = _distribution(probabilities)
    if temperature == 1.0:
        return p
    powered = np.power(p, 1.0 / float(temperature), dtype=np.float64)
    total = float(powered.sum())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("Temperature transform produced an invalid distribution")
    return powered / total


def _entropy(p: np.ndarray) -> tuple[float, float]:
    flat = np.asarray(p, dtype=np.float64).reshape(-1)
    positive = flat[flat > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    normalized = entropy / math.log(flat.size) if flat.size > 1 else 0.0
    return entropy, float(normalized)


def _top_mask(p: np.ndarray, fraction: float = 0.10) -> tuple[np.ndarray, float]:
    flat = np.asarray(p, dtype=np.float64).reshape(-1)
    count = max(1, int(math.ceil(flat.size * fraction)))
    # Stable sorting makes ties reproducible; exactly ceil(10%) cells are used.
    indices = np.argsort(-flat, kind="stable")[:count]
    mask = np.zeros(flat.size, dtype=bool)
    mask[indices] = True
    return mask.reshape(p.shape), float(flat[indices].sum())


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    left = np.asarray(p, dtype=np.float64).reshape(-1)
    right = np.asarray(q, dtype=np.float64).reshape(-1)
    middle = 0.5 * (left + right)

    def kl(values: np.ndarray) -> float:
        positive = values > 0
        return float(np.sum(values[positive] * np.log(values[positive] / middle[positive])))

    return 0.5 * (kl(left) + kl(right))


def _peak(p: np.ndarray) -> tuple[int, int]:
    index = int(np.argmax(np.asarray(p).reshape(-1)))
    return tuple(int(value) for value in np.unravel_index(index, p.shape))


def _peak_distance(left: np.ndarray, right: np.ndarray) -> tuple[float, list[int], list[int]]:
    left_peak = _peak(left)
    right_peak = _peak(right)
    height, width = left.shape
    diagonal = math.hypot(max(0, height - 1), max(0, width - 1))
    distance = math.hypot(left_peak[0] - right_peak[0], left_peak[1] - right_peak[1])
    normalized = distance / diagonal if diagonal > 0 else 0.0
    return float(normalized), list(left_peak), list(right_peak)


def _roi_mask(
    roi: Mapping[str, Any],
    grid_shape: tuple[int, int],
    image_size: tuple[int, int],
) -> np.ndarray:
    bbox = roi.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("Each ROI needs bbox=[x0,y0,x1,y1]")
    coordinates = roi.get("coordinate_space")
    if coordinates not in {"normalized", "pixels"}:
        raise ValueError("Each ROI needs coordinate_space='normalized' or 'pixels'")
    x0, y0, x1, y1 = (float(value) for value in bbox)
    if coordinates == "pixels":
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("Pixel ROI requires positive image dimensions")
        x0, x1 = x0 / width, x1 / width
        y0, y1 = y0 / height, y1 / height
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"ROI must lie inside the image, got normalized bbox={[x0, y0, x1, y1]}")
    height, width = grid_shape
    xs = (np.arange(width, dtype=np.float64) + 0.5) / width
    ys = (np.arange(height, dtype=np.float64) + 0.5) / height
    return (ys[:, None] >= y0) & (ys[:, None] <= y1) & (xs[None, :] >= x0) & (xs[None, :] <= x1)


def _normalize_roi_spec(spec: Any) -> list[dict[str, Any]]:
    if spec is None:
        return []
    if isinstance(spec, dict) and "rois" in spec:
        spec = spec["rois"]
    elif isinstance(spec, dict) and "bbox" in spec:
        spec = [spec]
    if not isinstance(spec, list):
        raise ValueError("ROI case entry must be an ROI object or {'rois': [...]} object")
    result = []
    for index, roi in enumerate(spec):
        if not isinstance(roi, dict):
            raise ValueError("Every ROI must be an object")
        item = dict(roi)
        item.setdefault("label", f"ROI {index + 1}")
        result.append(item)
    return result


def load_rois(path: str | Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    payload = read_json(path)
    if isinstance(payload, dict) and "cases" in payload:
        payload = payload["cases"]
    if isinstance(payload, list):
        mapping: dict[str, Any] = {}
        for item in payload:
            if not isinstance(item, dict) or "case_id" not in item:
                raise ValueError("ROI list items need a case_id")
            mapping[str(item["case_id"])] = item
        payload = mapping
    if not isinstance(payload, dict):
        raise ValueError("ROI JSON must be a case_id mapping, {'cases': mapping}, or a list")
    return {str(case_id): _normalize_roi_spec(spec) for case_id, spec in payload.items()}


@dataclass
class CaseData:
    case_id: str
    records: dict[str, dict[str, Any]]
    raw_grids: dict[str, list[np.ndarray]]
    probabilities: dict[str, list[np.ndarray]]
    image_masses: dict[str, list[float]]
    layer_metrics: list[dict[str, Any]]
    window_metrics: list[dict[str, Any]]
    rois: list[dict[str, Any]]


def _resolve_npz_path(record: Mapping[str, Any], records_path: Path) -> Path:
    path = Path(str(record.get("npz_path", ""))).expanduser()
    if not path.is_absolute():
        path = records_path.parent / path
    return path.resolve()


def _strict_gate(records: Mapping[str, Mapping[str, Any]]) -> tuple[bool, str]:
    if set(records) != set(REQUIRED_MODELS):
        missing = sorted(set(REQUIRED_MODELS) - set(records))
        extra = sorted(set(records) - set(REQUIRED_MODELS))
        return False, f"incomplete_models:missing={missing},extra={extra}"
    ours = records["ours"]
    ttrv = records["ttrv"]
    ground_truth = str(ours.get("ground_truth"))
    if str(ttrv.get("ground_truth")) != ground_truth:
        return False, "ground_truth_mismatch"
    if not bool(ours.get("correct")) or str(ours.get("parsed_answer")) != ground_truth:
        return False, "ours_not_correct"
    ttrv_answer = str(ttrv.get("parsed_answer", "unknown"))
    if ttrv_answer.lower() in {"", "none", "null", "unknown"}:
        return False, "ttrv_invalid"
    if bool(ttrv.get("correct")) or ttrv_answer == ground_truth:
        return False, "ttrv_not_wrong"
    if not bool(ours.get("retained_match")):
        return False, "ours_retained_mismatch"
    if not bool(ttrv.get("retained_match")):
        return False, "ttrv_retained_mismatch"
    return True, "strict_gate_pass"


def _load_layers(record: Mapping[str, Any], records_path: Path) -> list[np.ndarray]:
    individual = tuple(int(value) for value in record.get("individual_layers", []))
    if individual != EXPECTED_LAYERS:
        raise ValueError(f"individual_layers must be exactly 0..27, got {individual}")
    npz_path = _resolve_npz_path(record, records_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing NPZ: {npz_path}")
    layers = []
    with np.load(npz_path, allow_pickle=False) as arrays:
        for layer in EXPECTED_LAYERS:
            key = f"{ATTENTION_MODE}__layer_{layer:02d}"
            if key not in arrays:
                raise KeyError(f"{npz_path} lacks {key}; re-extract after adding first_decision_query")
            layers.append(np.asarray(arrays[key], dtype=np.float64).copy())
    return layers


def _layer_metrics(
    layer: int,
    ours_p: np.ndarray,
    ttrv_p: np.ndarray,
    ours_mass: float,
    ttrv_mass: float,
    rois: Sequence[Mapping[str, Any]],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    if ours_p.shape != ttrv_p.shape:
        raise ValueError(f"Ours/TTRV grid mismatch at layer {layer}: {ours_p.shape} != {ttrv_p.shape}")
    ours_top, ours_top_mass = _top_mask(ours_p)
    ttrv_top, ttrv_top_mass = _top_mask(ttrv_p)
    union = int(np.logical_or(ours_top, ttrv_top).sum())
    intersection = int(np.logical_and(ours_top, ttrv_top).sum())
    top_iou = intersection / union if union else 1.0
    peak_distance, ours_peak, ttrv_peak = _peak_distance(ours_p, ttrv_p)
    ours_entropy, ours_normalized_entropy = _entropy(ours_p)
    ttrv_entropy, ttrv_normalized_entropy = _entropy(ttrv_p)
    jsd = _jensen_shannon(ours_p, ttrv_p)
    tv = 0.5 * float(np.abs(ours_p - ttrv_p).sum())
    components = {
        "jsd_normalized": jsd / math.log(2.0),
        "total_variation": tv,
        "top10_disagreement": 1.0 - top_iou,
        "peak_normalized_distance": peak_distance,
    }
    separation = float(np.mean(list(components.values())))
    roi_metrics = []
    for roi in rois:
        mask = _roi_mask(roi, ours_p.shape, image_size)
        area_fraction = float(mask.mean())
        if area_fraction <= 0:
            raise ValueError(f"ROI {roi.get('label', 'ROI')!r} contains no attention-grid cell centers")
        ours_roi = float(ours_p[mask].sum())
        ttrv_roi = float(ttrv_p[mask].sum())
        ours_peak_in_roi = bool(mask[tuple(ours_peak)])
        ttrv_peak_in_roi = bool(mask[tuple(ttrv_peak)])
        roi_metrics.append(
            {
                "label": str(roi.get("label", "ROI")),
                "bbox": roi["bbox"],
                "coordinate_space": roi["coordinate_space"],
                "grid_area_fraction": area_fraction,
                "ours_conditional_mass": ours_roi,
                "ttrv_conditional_mass": ttrv_roi,
                "ours_minus_ttrv_conditional_mass_delta": ours_roi - ttrv_roi,
                "ours_area_normalized_lift": ours_roi / area_fraction,
                "ttrv_area_normalized_lift": ttrv_roi / area_fraction,
                "ours_minus_ttrv_area_normalized_lift_delta": (ours_roi - ttrv_roi) / area_fraction,
                "ours_ttrv_ratio": ours_roi / ttrv_roi if ttrv_roi > 0 else None,
                "ours_peak_in_roi": ours_peak_in_roi,
                "ttrv_peak_in_roi": ttrv_peak_in_roi,
            }
        )
    return {
        "layer": layer,
        "temperature_used_for_metrics": 1.0,
        "jsd": jsd,
        "jsd_normalized": components["jsd_normalized"],
        "total_variation": tv,
        "top10_iou": float(top_iou),
        "top10_cell_count": int(ours_top.sum()),
        "peak_normalized_distance": peak_distance,
        "ours_peak_row_col": ours_peak,
        "ttrv_peak_row_col": ttrv_peak,
        "ours_entropy": ours_entropy,
        "ttrv_entropy": ttrv_entropy,
        "ours_normalized_entropy": ours_normalized_entropy,
        "ttrv_normalized_entropy": ttrv_normalized_entropy,
        "ours_top10_mass": ours_top_mass,
        "ttrv_top10_mass": ttrv_top_mass,
        "ours_raw_image_mass": ours_mass,
        "ttrv_raw_image_mass": ttrv_mass,
        "raw_image_mass_ours_ttrv_ratio": ours_mass / ttrv_mass,
        "raw_separation_components": components,
        "raw_separation_score": separation,
        "raw_peak_differs": ours_peak != ttrv_peak,
        "raw_top10_differs": top_iou < 1.0,
        "roi_metrics": roi_metrics,
    }


def _window_metrics(layer_metrics: Sequence[Mapping[str, Any]], stability_penalty: float) -> list[dict[str, Any]]:
    windows = []
    for center in range(1, len(layer_metrics) - 1):
        layers = [center - 1, center, center + 1]
        values = np.asarray([layer_metrics[index]["raw_separation_score"] for index in layers], dtype=np.float64)
        median = float(np.median(values))
        q25, q75 = (float(value) for value in np.quantile(values, [0.25, 0.75]))
        iqr = q75 - q25
        roi_window_metrics = []
        roi_count = len(layer_metrics[layers[0]].get("roi_metrics", []))
        for roi_index in range(roi_count):
            per_layer_roi = [layer_metrics[index]["roi_metrics"][roi_index] for index in layers]

            def aggregate(key: str) -> dict[str, float]:
                roi_values = np.asarray([float(item[key]) for item in per_layer_roi], dtype=np.float64)
                roi_q25, roi_q75 = (float(value) for value in np.quantile(roi_values, [0.25, 0.75]))
                return {
                    "median": float(np.median(roi_values)),
                    "iqr": roi_q75 - roi_q25,
                    "min": float(roi_values.min()),
                    "max": float(roi_values.max()),
                }

            roi_window_metrics.append(
                {
                    "label": per_layer_roi[0]["label"],
                    "bbox": per_layer_roi[0]["bbox"],
                    "coordinate_space": per_layer_roi[0]["coordinate_space"],
                    "grid_area_fraction": per_layer_roi[0]["grid_area_fraction"],
                    "ours_conditional_mass": aggregate("ours_conditional_mass"),
                    "ttrv_conditional_mass": aggregate("ttrv_conditional_mass"),
                    "ours_minus_ttrv_conditional_mass_delta": aggregate(
                        "ours_minus_ttrv_conditional_mass_delta"
                    ),
                    "ours_area_normalized_lift": aggregate("ours_area_normalized_lift"),
                    "ttrv_area_normalized_lift": aggregate("ttrv_area_normalized_lift"),
                    "ours_minus_ttrv_area_normalized_lift_delta": aggregate(
                        "ours_minus_ttrv_area_normalized_lift_delta"
                    ),
                    "ours_peak_in_roi_layer_count": sum(bool(item["ours_peak_in_roi"]) for item in per_layer_roi),
                    "ttrv_peak_in_roi_layer_count": sum(bool(item["ttrv_peak_in_roi"]) for item in per_layer_roi),
                    "window_layer_count": len(layers),
                }
            )
        windows.append(
            {
                "center_layer": center,
                "layers": layers,
                "raw_score_median": median,
                "raw_score_iqr": iqr,
                "raw_stability_score": median - stability_penalty * iqr,
                "raw_peak_difference_layers": sum(
                    bool(layer_metrics[index]["raw_peak_differs"]) for index in layers
                ),
                "raw_top10_difference_layers": sum(
                    bool(layer_metrics[index]["raw_top10_differs"]) for index in layers
                ),
                "raw_peak_distance_median": float(
                    np.median([layer_metrics[index]["peak_normalized_distance"] for index in layers])
                ),
                "raw_top10_iou_median": float(np.median([layer_metrics[index]["top10_iou"] for index in layers])),
                "raw_jsd_median": float(np.median([layer_metrics[index]["jsd"] for index in layers])),
                "raw_tv_median": float(
                    np.median([layer_metrics[index]["total_variation"] for index in layers])
                ),
                "roi_window_metrics": roi_window_metrics,
            }
        )
    return windows


def _best_window(windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(
        max(
            windows,
            key=lambda item: (
                float(item["raw_stability_score"]),
                float(item["raw_score_median"]),
                -float(item["raw_score_iqr"]),
                -int(item["center_layer"]),
            ),
        )
    )


def _shared_window(cases: Sequence[CaseData], stability_penalty: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    for center in range(1, 27):
        per_case = [case.window_metrics[center - 1]["raw_stability_score"] for case in cases]
        values = np.asarray(per_case, dtype=np.float64)
        median = float(np.median(values))
        q25, q75 = (float(value) for value in np.quantile(values, [0.25, 0.75]))
        iqr = q75 - q25
        candidates.append(
            {
                "center_layer": center,
                "layers": [center - 1, center, center + 1],
                "case_count": len(cases),
                "raw_case_stability_median": median,
                "raw_case_stability_iqr": iqr,
                "raw_shared_window_score": median - stability_penalty * iqr,
            }
        )
    candidates.sort(
        key=lambda item: (
            -float(item["raw_shared_window_score"]),
            -float(item["raw_case_stability_median"]),
            float(item["raw_case_stability_iqr"]),
            int(item["center_layer"]),
        )
    )
    return dict(candidates[0]), candidates


def _tier(
    window: Mapping[str, Any],
    tier_a_min_stability: float,
    tier_b_min_stability: float,
    tier_b_min_peak_distance: float,
    tier_b_max_top10_iou: float,
) -> tuple[str, str]:
    stability = float(window["raw_stability_score"])
    if stability >= tier_a_min_stability:
        return "A", "raw_T1_stable_difference"
    raw_peak_and_topk_differ = bool(
        int(window["raw_peak_difference_layers"]) >= 2
        and int(window["raw_top10_difference_layers"]) >= 2
        and float(window["raw_peak_distance_median"]) >= tier_b_min_peak_distance
        and float(window["raw_top10_iou_median"]) <= tier_b_max_top10_iou
    )
    if stability >= tier_b_min_stability and raw_peak_and_topk_differ:
        return "B", "raw_T1_difference_present_fixed_T0.5_display_only"
    return "unqualified", "raw_difference_below_prespecified_thresholds"


def _load_case(
    case_id: str,
    records: dict[str, dict[str, Any]],
    records_path: Path,
    rois: list[dict[str, Any]],
    stability_penalty: float,
) -> CaseData:
    raw_grids: dict[str, list[np.ndarray]] = {}
    probabilities: dict[str, list[np.ndarray]] = {}
    image_masses: dict[str, list[float]] = {}
    shapes = set()
    for model in REQUIRED_MODELS:
        raw_grids[model] = _load_layers(records[model], records_path)
        probabilities[model] = []
        image_masses[model] = []
        for grid in raw_grids[model]:
            p, mass = _distribution(grid)
            probabilities[model].append(p)
            image_masses[model].append(mass)
            shapes.add(p.shape)
    if len(shapes) != 1:
        raise ValueError(f"Layer/model grids have mismatched shapes: {sorted(shapes)}")
    reference = records["ours"]
    grid_meta = reference.get("grid", {})
    image_size = (
        int(grid_meta.get("original_width", 0)),
        int(grid_meta.get("original_height", 0)),
    )
    metrics = [
        _layer_metrics(
            layer,
            probabilities["ours"][layer],
            probabilities["ttrv"][layer],
            image_masses["ours"][layer],
            image_masses["ttrv"][layer],
            rois,
            image_size,
        )
        for layer in EXPECTED_LAYERS
    ]
    windows = _window_metrics(metrics, stability_penalty=stability_penalty)
    return CaseData(
        case_id=case_id,
        records=records,
        raw_grids=raw_grids,
        probabilities=probabilities,
        image_masses=image_masses,
        layer_metrics=metrics,
        window_metrics=windows,
        rois=rois,
    )


def _aggregate_window(case: CaseData, model: str, layers: Sequence[int]) -> tuple[np.ndarray, float]:
    p = np.mean([case.probabilities[model][layer] for layer in layers], axis=0)
    p /= p.sum()
    mass = float(np.mean([case.image_masses[model][layer] for layer in layers]))
    return p, mass


def _upsample(grid: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(np.asarray(grid, dtype=np.float32), mode="F")
    return np.asarray(image.resize(size, resample=Image.Resampling.BILINEAR), dtype=np.float32)


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _short_path(value: Any, components: int = 4) -> str:
    path = Path(str(value))
    return "/".join(path.parts[-components:])


def _draw_rois(image: Any, rois: Sequence[Mapping[str, Any]]) -> Any:
    from PIL import ImageDraw

    if not rois:
        return image
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for roi in rois:
        x0, y0, x1, y1 = (float(value) for value in roi["bbox"])
        if roi["coordinate_space"] == "normalized":
            x0, x1 = x0 * image.width, x1 * image.width
            y0, y1 = y0 * image.height, y1 * image.height
        draw.rectangle((x0, y0, x1, y1), outline=(255, 64, 64), width=max(2, image.width // 240))
    return result


def _build_temperature_figure(
    case: CaseData,
    layers: Sequence[int],
    tier: str,
    temperature: float,
    diagnostic_only: bool,
) -> tuple[Any, dict[str, Any]]:
    import cv2
    from PIL import Image, ImageDraw

    reference = case.records["ours"]
    with Image.open(reference["image_path"]) as handle:
        original = handle.convert("RGB").copy()
    original = _draw_rois(original, case.rois)
    maps: dict[str, np.ndarray] = {}
    raw_masses = {}
    for model in ("ttrv", "ours"):
        raw_p, raw_mass = _aggregate_window(case, model, layers)
        maps[model] = display_temperature(raw_p, temperature)
        raw_masses[model] = raw_mass
    vmax = float(max(maps["ttrv"].max(), maps["ours"].max()))
    if vmax <= 0:
        vmax = 1.0

    panel_width = 500
    image_height = 340
    text_height = 230
    title_height = 72
    bar_width = 82
    canvas = Image.new("RGB", (panel_width * 3 + bar_width, title_height + image_height + text_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(22, bold=True)
    header_font = _font(24, bold=True)
    body_font = _font(15)
    small_font = _font(13)
    display_note = "T=1 RAW METRIC/DISPLAY" if temperature == 1.0 else "T=0.5 DISPLAY-ONLY (NOT USED FOR RANKING)"
    fallback_note = " | DIAGNOSTIC PER-CASE WINDOW" if diagnostic_only else " | SHARED WINDOW"
    draw.text(
        (14, 10),
        f"Obs3 Tier {tier} | layers {list(layers)} | {display_note}{fallback_note}",
        fill="black",
        font=title_font,
    )
    draw.text(
        (14, 43),
        "Conditional spatial attention p=a/sum(a); Ours and TTRV share vmin=0 and vmax within this case.",
        fill=(70, 70, 70),
        font=small_font,
    )
    for column, label in enumerate(("Original image", "TTRV", "Ours")):
        bounds = draw.textbbox((0, 0), label, font=header_font)
        x = column * panel_width + (panel_width - (bounds[2] - bounds[0])) // 2
        draw.text((x, title_height + 2), label, fill="black", font=header_font)

    image_y = title_height + 36
    usable_image_height = image_height - 36

    def fit(image: Any) -> tuple[Any, int, int]:
        scale = min((panel_width - 22) / image.width, (usable_image_height - 8) / image.height)
        width = max(1, int(round(image.width * scale)))
        height = max(1, int(round(image.height * scale)))
        resized = image.resize((width, height), resample=Image.Resampling.LANCZOS)
        return resized, (panel_width - width) // 2, (usable_image_height - height) // 2

    resized, x_pad, y_pad = fit(original)
    canvas.paste(resized, (x_pad, image_y + y_pad))
    for column, model in enumerate(("ttrv", "ours"), start=1):
        values = np.clip(maps[model] / vmax, 0.0, 1.0)
        heat_bgr = cv2.applyColorMap(np.round(values * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        heat = Image.fromarray(heat_rgb).resize(original.size, resample=Image.Resampling.BILINEAR)
        overlay = Image.blend(original, heat, alpha=0.52)
        overlay = _draw_rois(overlay, case.rois)
        fitted, overlay_x, overlay_y = fit(overlay)
        canvas.paste(fitted, (column * panel_width + overlay_x, image_y + overlay_y))

    text_y = title_height + image_height + 4
    question = re.sub(r"\s+", " ", str(reference.get("question", ""))).strip()
    original_lines = [
        f"{reference.get('dataset')} | {reference.get('question_id')} | {case.case_id}",
        *textwrap.wrap(question, width=58)[:6],
        f"Ground truth: {reference.get('ground_truth')}",
    ]
    if case.rois:
        original_lines.append("Explicit ROI: " + ", ".join(str(roi.get("label", "ROI")) for roi in case.rois))
    for index, line in enumerate(original_lines[:9]):
        draw.text((12, text_y + index * 20), line, fill="black", font=body_font)

    for column, model in enumerate(("ttrv", "ours"), start=1):
        record = case.records[model]
        lines = [
            f"Raw response: {str(record.get('raw_response', '')).strip()[:62]}",
            f"Parsed: {record.get('parsed_answer')} | correct={bool(record.get('correct'))}",
            f"retained_match={bool(record.get('retained_match'))}",
            f"mean raw image mass (layers): {raw_masses[model]:.4g}",
            f"checkpoint: {_short_path(record.get('checkpoint'))}",
            f"source: {_short_path(record.get('source_checkpoint'))}",
        ]
        if record.get("validation_provenance") is not None:
            provenance = json.dumps(record["validation_provenance"], ensure_ascii=False, default=_json_default)
            lines.extend(textwrap.wrap("validation: " + provenance, width=60)[:3])
        for index, line in enumerate(lines[:9]):
            draw.text((column * panel_width + 12, text_y + index * 20), line, fill="black", font=body_font)

    bar_x = panel_width * 3 + 22
    bar_top = image_y + 12
    bar_bottom = image_y + usable_image_height - 12
    gradient = np.linspace(255, 0, max(2, bar_bottom - bar_top), dtype=np.uint8).reshape(-1, 1)
    gradient = np.repeat(gradient, 25, axis=1)
    colorbar = cv2.applyColorMap(gradient, cv2.COLORMAP_VIRIDIS)
    colorbar = cv2.cvtColor(colorbar, cv2.COLOR_BGR2RGB)
    canvas.paste(Image.fromarray(colorbar), (bar_x, bar_top))
    draw.rectangle((bar_x, bar_top, bar_x + 24, bar_bottom), outline="black", width=1)
    draw.text((bar_x - 5, bar_top - 20), f"{vmax:.3g}", fill="black", font=small_font)
    draw.text((bar_x + 3, bar_bottom + 4), "0", fill="black", font=small_font)
    return canvas, {"vmin": 0.0, "vmax": vmax, "shared_between": ["ttrv", "ours"]}


def _save_png_pdf(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)
    image.convert("RGB").save(path.with_suffix(".pdf"), format="PDF", resolution=220.0)


def _render_case(
    case: CaseData,
    rank: int,
    window: Mapping[str, Any],
    tier: str,
    output_dir: Path,
    diagnostic_only: bool,
) -> dict[str, Any]:
    layers = [int(value) for value in window["layers"]]
    stem = f"{rank:02d}_{sanitize_name(case.case_id)}_layers_{layers[0]:02d}_{layers[-1]:02d}"
    folder = output_dir / ("diagnostic_per_case_fallback" if diagnostic_only else "figures")
    figures = {}
    canvases = []
    for temperature, suffix in ((1.0, "T1_raw"), (0.5, "T0p5_display_only")):
        canvas, scale = _build_temperature_figure(
            case,
            layers=layers,
            tier=tier,
            temperature=temperature,
            diagnostic_only=diagnostic_only,
        )
        path = folder / f"{stem}_{suffix}.png"
        _save_png_pdf(canvas, path)
        figures[suffix] = {
            "png": str(path.resolve()),
            "pdf": str(path.with_suffix(".pdf").resolve()),
            "temperature": temperature,
            "display_only": temperature != 1.0,
            "color_scale": scale,
        }
        canvases.append(canvas)
    from PIL import Image

    comparison = Image.new(
        "RGB",
        (max(image.width for image in canvases), sum(image.height for image in canvases)),
        "white",
    )
    y = 0
    for canvas in canvases:
        comparison.paste(canvas, (0, y))
        y += canvas.height
    comparison_path = folder / f"{stem}_T1_vs_T0p5_comparison.png"
    _save_png_pdf(comparison, comparison_path)
    figures["comparison"] = {
        "png": str(comparison_path.resolve()),
        "pdf": str(comparison_path.with_suffix(".pdf").resolve()),
        "note": "T=0.5 is display-only; selection and ranking use raw T=1 metrics.",
    }
    return figures


def screen_records(
    records_path: str | Path,
    output_dir: str | Path,
    roi_json: str | Path | None = None,
    max_figures: int = 12,
    stability_penalty: float = 0.5,
    tier_a_min_stability: float = 0.30,
    tier_b_min_stability: float = 0.10,
    tier_b_min_peak_distance: float = 0.05,
    tier_b_max_top10_iou: float = 0.80,
) -> dict[str, Any]:
    if max_figures < 0:
        raise ValueError("max_figures must be >= 0")
    if not (tier_a_min_stability >= tier_b_min_stability >= 0):
        raise ValueError("Require tier_a_min_stability >= tier_b_min_stability >= 0")
    records_file = Path(records_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    roi_by_case = load_rois(roi_json)

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicate_cases = set()
    for record in read_jsonl(records_file):
        case_id = str(record.get("case_id", ""))
        model = str(record.get("model", ""))
        if model in grouped[case_id]:
            duplicate_cases.add(case_id)
        grouped[case_id][model] = record

    exclusions: Counter[str] = Counter()
    audit_rows = []
    cases: list[CaseData] = []
    for case_id in sorted(grouped):
        records = grouped[case_id]
        if case_id in duplicate_cases:
            exclusions["duplicate_model_record"] += 1
            audit_rows.append({"case_id": case_id, "included": False, "reason": "duplicate_model_record"})
            continue
        passes, reason = _strict_gate(records)
        if not passes:
            exclusions[reason] += 1
            audit_rows.append({"case_id": case_id, "included": False, "reason": reason})
            continue
        try:
            case = _load_case(
                case_id,
                records,
                records_file,
                roi_by_case.get(case_id, []),
                stability_penalty=stability_penalty,
            )
            image_path = Path(str(records["ours"].get("image_path", ""))).expanduser()
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing original image: {image_path}")
        except Exception as exc:
            key = f"artifact_error:{type(exc).__name__}"
            exclusions[key] += 1
            audit_rows.append(
                {"case_id": case_id, "included": False, "reason": key, "detail": str(exc)}
            )
            continue
        cases.append(case)
        audit_rows.append({"case_id": case_id, "included": True, "reason": "strict_complete_case"})

    ranked_rows: list[dict[str, Any]] = []
    shared: dict[str, Any] | None = None
    shared_candidates: list[dict[str, Any]] = []
    fallback = False
    chosen: list[tuple[CaseData, dict[str, Any], str, str]] = []
    if cases:
        shared, shared_candidates = _shared_window(cases, stability_penalty=stability_penalty)
        center = int(shared["center_layer"])
        for case in cases:
            window = dict(case.window_metrics[center - 1])
            tier, tier_reason = _tier(
                window,
                tier_a_min_stability,
                tier_b_min_stability,
                tier_b_min_peak_distance,
                tier_b_max_top10_iou,
            )
            if tier != "unqualified":
                chosen.append((case, window, tier, tier_reason))
        if not chosen:
            fallback = True
            for case in cases:
                window = _best_window(case.window_metrics)
                tier, tier_reason = _tier(
                    window,
                    tier_a_min_stability,
                    tier_b_min_stability,
                    tier_b_min_peak_distance,
                    tier_b_max_top10_iou,
                )
                if tier != "unqualified":
                    chosen.append((case, window, tier, tier_reason))

        window_by_case = {case.case_id: (window, tier, reason) for case, window, tier, reason in chosen}
        ordered_cases = sorted(
            cases,
            key=lambda case: (
                -float(
                    (window_by_case[case.case_id][0] if case.case_id in window_by_case else (
                        _best_window(case.window_metrics) if fallback else case.window_metrics[center - 1]
                    ))["raw_stability_score"]
                ),
                case.case_id,
            ),
        )
        qualified_rank = 0
        for overall_rank, case in enumerate(ordered_cases, start=1):
            if case.case_id in window_by_case:
                window, tier, tier_reason = window_by_case[case.case_id]
                qualified_rank += 1
            else:
                window = _best_window(case.window_metrics) if fallback else dict(case.window_metrics[center - 1])
                tier, tier_reason = "unqualified", "raw_difference_below_prespecified_thresholds"
            best_diagnostic = _best_window(case.window_metrics)
            ours = case.records["ours"]
            ttrv = case.records["ttrv"]
            selected_roi_metrics = list(window.get("roi_window_metrics", []))
            positive_roi_count = sum(
                float(item["ours_minus_ttrv_conditional_mass_delta"]["median"]) > 0
                for item in selected_roi_metrics
            )
            roi_manual_review_signal = {
                "available": bool(selected_roi_metrics),
                "affects_default_gate_or_rank": False,
                "positive_ours_minus_ttrv_delta_count": positive_roi_count,
                "roi_count": len(selected_roi_metrics),
                "all_explicit_rois_have_positive_median_delta": bool(selected_roi_metrics)
                and positive_roi_count == len(selected_roi_metrics),
                "mean_median_ours_minus_ttrv_delta": (
                    float(
                        np.mean(
                            [
                                item["ours_minus_ttrv_conditional_mass_delta"]["median"]
                                for item in selected_roi_metrics
                            ]
                        )
                    )
                    if selected_roi_metrics
                    else None
                ),
            }
            ranked_rows.append(
                {
                    "rank": overall_rank,
                    "qualified_rank": qualified_rank if tier != "unqualified" else None,
                    "case_id": case.case_id,
                    "dataset": ours.get("dataset"),
                    "question_id": ours.get("question_id"),
                    "question": ours.get("question"),
                    "ground_truth": ours.get("ground_truth"),
                    "tier": tier,
                    "tier_reason": tier_reason,
                    "qualified": tier != "unqualified",
                    "selection_policy": (
                        "per_case_three_layer_diagnostic_fallback" if fallback else "shared_three_layer_window"
                    ),
                    "diagnostic_only": fallback,
                    "selected_window": window,
                    "selected_window_roi_metrics": selected_roi_metrics,
                    "roi_manual_review_signal": roi_manual_review_signal,
                    "best_per_case_window_diagnostic": best_diagnostic,
                    "rank_metric": "raw_T1_selected_window_stability_score",
                    "rank_value": window["raw_stability_score"],
                    "display_temperature": 1.0 if tier == "A" else (0.5 if tier == "B" else None),
                    "display_temperature_affects_ranking": False,
                    "layer_metrics": case.layer_metrics,
                    "models": {
                        model: {
                            "checkpoint": case.records[model].get("checkpoint"),
                            "source_checkpoint": case.records[model].get("source_checkpoint"),
                            "raw_response": case.records[model].get("raw_response"),
                            "parsed_answer": case.records[model].get("parsed_answer"),
                            "correct": case.records[model].get("correct"),
                            "retained_match": case.records[model].get("retained_match"),
                            "validation_provenance": case.records[model].get("validation_provenance"),
                        }
                        for model in REQUIRED_MODELS
                    },
                    "image_path": ours.get("image_path"),
                    "explicit_rois": case.rois,
                }
            )

    ranked_rows.sort(key=lambda row: int(row["rank"]))
    chosen.sort(key=lambda item: (-float(item[1]["raw_stability_score"]), item[0].case_id))
    figures_by_case = {}
    for render_rank, (case, window, tier, _reason) in enumerate(chosen[:max_figures], start=1):
        figures_by_case[case.case_id] = _render_case(
            case,
            rank=render_rank,
            window=window,
            tier=tier,
            output_dir=destination,
            diagnostic_only=fallback,
        )
    for row in ranked_rows:
        row["figures"] = figures_by_case.get(row["case_id"])

    ranked_path = destination / "ranked_cases.jsonl"
    audit_path = destination / "screen_audit.jsonl"
    write_jsonl(ranked_path, ranked_rows)
    write_jsonl(audit_path, audit_rows)
    summary = {
        "records": str(records_file),
        "output_dir": str(destination),
        "attention_mode": ATTENTION_MODE,
        "required_models": list(REQUIRED_MODELS),
        "required_individual_layers_zero_based": list(EXPECTED_LAYERS),
        "input_case_count": len(grouped),
        "strict_complete_case_count": len(cases),
        "qualified_case_count": len(chosen),
        "tier_counts": dict(sorted(Counter(item[2] for item in chosen).items())),
        "exclusion_counts": dict(sorted(exclusions.items())),
        "selection_policy": (
            "per_case_three_layer_diagnostic_fallback" if fallback else "shared_three_layer_window"
        ) if cases else "no_strict_complete_cases",
        "diagnostic_only": fallback,
        "fallback_disclosure": (
            "The all-candidate shared three-layer window produced no qualified cases. "
            "Per-case adjacent three-layer windows are emitted only as diagnostics and must not be presented "
            "as a shared-layer final comparison."
            if fallback
            else None
        ),
        "shared_window": shared,
        "shared_window_candidates_ranked_raw_T1": shared_candidates,
        "thresholds": {
            "stability_penalty": stability_penalty,
            "tier_a_min_raw_stability": tier_a_min_stability,
            "tier_b_min_raw_stability": tier_b_min_stability,
            "tier_b_min_raw_peak_distance_median": tier_b_min_peak_distance,
            "tier_b_max_raw_top10_iou_median": tier_b_max_top10_iou,
            "tier_b_requires_peak_and_top10_difference_in_at_least_two_of_three_layers": True,
        },
        "metric_definition": {
            "quantitative_temperature": 1.0,
            "conditional_spatial_distribution": "p = max(attention,0) / sum(max(attention,0))",
            "raw_separation_score": (
                "mean(JSD/log(2), total_variation, 1-top10%_IoU, normalized_peak_distance)"
            ),
            "three_layer_stability": "median(raw separation) - stability_penalty * IQR(raw separation)",
            "shared_window_selection": (
                "maximize median(per-case stability) - stability_penalty * IQR(per-case stability)"
            ),
            "top10_definition": "exactly ceil(0.10 * grid_cells), stable descending ties",
            "raw_image_mass": "sum of unnormalized saved image-attention grid",
        },
        "display_policy": {
            "temperatures": list(DISPLAY_TEMPERATURES),
            "tier_a_primary": "T=1 raw conditional map",
            "tier_b_sharpening": "fixed T=0.5 display-only",
            "temperature_formula": "q(T) = p^(1/T) / sum(p^(1/T))",
            "temperature_used_for_selection_or_ranking": False,
            "color_scale": "Within each case and temperature, Ours/TTRV share vmin=0 and one vmax.",
        },
        "roi_policy": {
            "roi_json": str(Path(roi_json).resolve()) if roi_json else None,
            "automatic_roi_inference": False,
            "metrics": [
                "grid area fraction",
                "conditional mass",
                "area-normalized lift",
                "Ours-minus-TTRV mass/lift delta",
                "peak-in-ROI",
                "three-layer median/IQR/min/max aggregation",
            ],
            "manual_selection_guidance": (
                "A positive predeclared-ROI Ours-minus-TTRV median mass delta may be prioritized during "
                "question-relevance review; ROI never changes the default gate, shared window, tier, or rank."
            ),
            "accepted_schema": {
                "cases": {
                    "CASE_ID": {
                        "rois": [
                            {
                                "label": "predeclared target",
                                "bbox": [0.1, 0.2, 0.4, 0.6],
                                "coordinate_space": "normalized",
                            }
                        ]
                    }
                }
            },
        },
        "interpretation_warning": (
            "Spatial attention is correlational evidence, not proof of causal image use. "
            "T=0.5 cannot change cell ordering or peak locations and is never used quantitatively."
        ),
        "ranked_cases": str(ranked_path.resolve()),
        "screen_audit": str(audit_path.resolve()),
        "rendered_case_count": len(figures_by_case),
    }
    write_json(destination / "screen_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="attention_records.jsonl from extract")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--roi-json", default=None, help="Optional, explicitly predeclared ROI JSON")
    parser.add_argument("--max-figures", type=int, default=12)
    parser.add_argument("--stability-penalty", type=float, default=0.5)
    parser.add_argument("--tier-a-min-stability", type=float, default=0.30)
    parser.add_argument("--tier-b-min-stability", type=float, default=0.10)
    parser.add_argument("--tier-b-min-peak-distance", type=float, default=0.05)
    parser.add_argument("--tier-b-max-top10-iou", type=float, default=0.80)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = screen_records(
        records_path=args.records,
        output_dir=args.output_dir,
        roi_json=args.roi_json,
        max_figures=args.max_figures,
        stability_penalty=args.stability_penalty,
        tier_a_min_stability=args.tier_a_min_stability,
        tier_b_min_stability=args.tier_b_min_stability,
        tier_b_min_peak_distance=args.tier_b_min_peak_distance,
        tier_b_max_top10_iou=args.tier_b_max_top10_iou,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
