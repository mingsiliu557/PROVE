#!/usr/bin/env python3
"""Correctness-blind, multi-scheme screening of saved Obs3 attention maps.

The script is CPU-only.  It consumes ``attention_records.jsonl`` and the NPZ
files emitted by ``attention_viz.py extract --save-individual-layers``.  Case
ranking deliberately uses only case/dataset identifiers and saved attention
arrays; answer, prediction, correctness, and ground-truth fields are never
read by the ranking path.

Every quantitative metric is computed from raw T=1 attention.  Temperature is
applied only while rendering a separately labelled display-only panel.
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
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


MODELS = ("base", "ttrv", "ours")
RANK_MODELS = ("ttrv", "ours")
QUERIES = ("first_decision_query", "decision_query", "answer_state")
PRIMARY_QUERY = "first_decision_query"
LAYERS = tuple(range(28))
WINDOWS = tuple((center - 1, center, center + 1) for center in range(1, 27))
FORBIDDEN_RANK_FIELDS = (
    "answer",
    "correct",
    "ground_truth",
    "parsed_answer",
    "prediction",
    "raw_response",
    "retained_parsed_answer",
    "retained_prediction_counter",
    "retained_raw_response",
    "validation_provenance",
)


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


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
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


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, default=_json_default)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return cleaned[:120] or "case"


def tie_break(case_id: str) -> str:
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


def resolve_npz(record: Mapping[str, Any], records_path: Path) -> Path:
    path = Path(str(record.get("npz_path", ""))).expanduser()
    if not path.is_absolute():
        path = records_path.parent / path
    return path.resolve()


@dataclass
class GridMetric:
    probability: np.ndarray
    mass: float
    normalized_entropy: float
    concentration: float
    top10_mass: float
    corner_penalty: float
    peak_in_corner: bool


@dataclass
class CaseData:
    case_id: str
    dataset: str
    question: str
    image_path: str
    safe_records: dict[str, dict[str, Any]]
    raw: dict[str, dict[str, list[np.ndarray]]]
    metrics: dict[str, dict[str, list[GridMetric]]]
    head_raw: dict[str, dict[str, dict[int, list[np.ndarray]]]] = field(default_factory=dict)
    head_metrics: dict[str, dict[str, dict[int, list[GridMetric]]]] = field(default_factory=dict)


def _validate_grid(value: np.ndarray, key: str) -> np.ndarray:
    grid = np.asarray(value, dtype=np.float32)
    if grid.ndim != 2:
        raise ValueError(f"{key} must be 2-D, got {grid.shape}")
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{key} contains NaN/inf")
    if float(grid.min()) < -1e-8:
        raise ValueError(f"{key} contains negative attention: min={grid.min():.4g}")
    return np.maximum(grid, 0.0)


def _corner_mask(shape: tuple[int, int], fraction: float) -> np.ndarray:
    height, width = shape
    xs = (np.arange(width, dtype=np.float64) + 0.5) / width
    ys = (np.arange(height, dtype=np.float64) + 0.5) / height
    x_corner = (xs < fraction) | (xs > 1.0 - fraction)
    y_corner = (ys < fraction) | (ys > 1.0 - fraction)
    return y_corner[:, None] & x_corner[None, :]


def grid_metric(raw: np.ndarray, corner_fraction: float) -> GridMetric:
    values = np.asarray(raw, dtype=np.float64)
    mass = float(values.sum())
    if mass <= 0.0:
        probability = np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    else:
        probability = values / mass
    positive = probability[probability > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    normalized_entropy = entropy / math.log(probability.size) if probability.size > 1 else 0.0
    concentration = float(np.clip(1.0 - normalized_entropy, 0.0, 1.0))
    flat = probability.reshape(-1)
    top_count = max(1, int(math.ceil(flat.size * 0.10)))
    top_indices = np.argsort(-flat, kind="stable")[:top_count]
    top10_mass = float(flat[top_indices].sum())
    mask = _corner_mask(probability.shape, corner_fraction)
    area = float(mask.mean())
    corner_mass = float(probability[mask].sum())
    excess = max(0.0, (corner_mass - area) / max(1e-12, 1.0 - area))
    peak = np.unravel_index(int(np.argmax(probability)), probability.shape)
    peak_in_corner = bool(mask[peak])
    penalty = float(np.clip(0.75 * excess + 0.25 * float(peak_in_corner), 0.0, 1.0))
    return GridMetric(
        probability=probability.astype(np.float32),
        mass=mass,
        normalized_entropy=float(normalized_entropy),
        concentration=concentration,
        top10_mass=top10_mass,
        corner_penalty=penalty,
        peak_in_corner=peak_in_corner,
    )


HEAD_PATTERNS = (
    re.compile(r"^(?P<query>.+)__layer_(?P<layer>\d+)__head_(?P<head>\d+)$"),
    re.compile(r"^(?P<query>.+)__head_(?P<head>\d+)__layer_(?P<layer>\d+)$"),
    re.compile(r"^(?P<query>.+)__layer_(?P<layer>\d+)_head_(?P<head>\d+)$"),
)
HEAD_QUERY_ALIASES = {f"{query}_heads": query for query in QUERIES}


def _head_key(value: str) -> tuple[str, int, int] | None:
    for pattern in HEAD_PATTERNS:
        match = pattern.fullmatch(value)
        if match:
            return match.group("query"), int(match.group("head")), int(match.group("layer"))
    return None


def _load_npz(
    record: Mapping[str, Any],
    records_path: Path,
    corner_fraction: float,
) -> tuple[
    dict[str, list[np.ndarray]],
    dict[str, list[GridMetric]],
    dict[str, dict[int, list[np.ndarray]]],
    dict[str, dict[int, list[GridMetric]]],
]:
    path = resolve_npz(record, records_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing NPZ: {path}")
    raw: dict[str, list[np.ndarray]] = {}
    metrics: dict[str, list[GridMetric]] = {}
    head_by_key: dict[str, dict[int, dict[int, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    with np.load(path, allow_pickle=False) as arrays:
        for query in QUERIES:
            expected_keys = [f"{query}__layer_{layer:02d}" for layer in LAYERS]
            present = [key in arrays for key in expected_keys]
            if not any(present):
                continue
            if not all(present):
                missing = [key for key, exists in zip(expected_keys, present) if not exists]
                raise KeyError(f"{path} has partial {query} layers; missing={missing}")
            grids: list[np.ndarray] = []
            for key in expected_keys:
                grids.append(_validate_grid(arrays[key], key))
            raw[query] = grids
            metrics[query] = [grid_metric(grid, corner_fraction) for grid in grids]
        if PRIMARY_QUERY not in raw:
            raise KeyError(f"{path} lacks required primary query {PRIMARY_QUERY}")
        for key in arrays.files:
            parsed = _head_key(key)
            if parsed is None:
                continue
            query, head, layer = parsed
            query = HEAD_QUERY_ALIASES.get(query, query)
            if query in QUERIES and layer in LAYERS:
                head_by_key[query][head][layer] = _validate_grid(arrays[key], key)
    head_raw: dict[str, dict[int, list[np.ndarray]]] = defaultdict(dict)
    head_metrics: dict[str, dict[int, list[GridMetric]]] = defaultdict(dict)
    for query, by_head in head_by_key.items():
        for head, by_layer in by_head.items():
            if set(by_layer) != set(LAYERS):
                continue
            grids = [by_layer[layer] for layer in LAYERS]
            head_raw[query][head] = grids
            head_metrics[query][head] = [grid_metric(grid, corner_fraction) for grid in grids]
    return raw, metrics, dict(head_raw), dict(head_metrics)


def load_cases(
    records_path: str | Path,
    corner_fraction: float,
    limit: int = 0,
) -> tuple[list[CaseData], list[dict[str, Any]]]:
    source = Path(records_path).resolve()
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in read_jsonl(source):
        case_id = str(record.get("case_id", ""))
        model = str(record.get("model", ""))
        if not case_id or model not in MODELS:
            continue
        if model in grouped[case_id]:
            raise ValueError(f"Duplicate record for {case_id}/{model}")
        grouped[case_id][model] = record
    case_ids = sorted(grouped, key=tie_break)
    if limit > 0:
        case_ids = case_ids[:limit]
    cases: list[CaseData] = []
    failures: list[dict[str, Any]] = []
    for case_id in case_ids:
        records = grouped[case_id]
        if set(records) != set(MODELS):
            failures.append({"case_id": case_id, "reason": "incomplete_models", "models": sorted(records)})
            continue
        try:
            datasets = {str(records[model].get("dataset", "unknown")) for model in MODELS}
            images = {str(records[model].get("image_path", "")) for model in MODELS}
            if len(datasets) != 1 or len(images) != 1:
                raise ValueError(f"safe metadata mismatch: datasets={datasets}, images={images}")
            all_raw: dict[str, dict[str, list[np.ndarray]]] = {}
            all_metrics: dict[str, dict[str, list[GridMetric]]] = {}
            all_head_raw: dict[str, dict[str, dict[int, list[np.ndarray]]]] = {}
            all_head_metrics: dict[str, dict[str, dict[int, list[GridMetric]]]] = {}
            safe_records: dict[str, dict[str, Any]] = {}
            for model in MODELS:
                raw, metrics, head_raw, head_metrics = _load_npz(records[model], source, corner_fraction)
                all_raw[model] = raw
                all_metrics[model] = metrics
                all_head_raw[model] = head_raw
                all_head_metrics[model] = head_metrics
                safe_records[model] = {
                    "model": model,
                    "checkpoint": str(records[model].get("checkpoint", "")),
                    "source_checkpoint": str(records[model].get("source_checkpoint", "")),
                    "npz_path": str(resolve_npz(records[model], source)),
                }
            reference = records["base"]
            cases.append(
                CaseData(
                    case_id=case_id,
                    dataset=next(iter(datasets)),
                    question=str(reference.get("question", "")),
                    image_path=next(iter(images)),
                    safe_records=safe_records,
                    raw=all_raw,
                    metrics=all_metrics,
                    head_raw=all_head_raw,
                    head_metrics=all_head_metrics,
                )
            )
        except Exception as exc:  # preserve a complete, auditable failure list
            failures.append({"case_id": case_id, "reason": type(exc).__name__, "message": str(exc)})
    return cases, failures


def normalized_jsd(left: np.ndarray, right: np.ndarray) -> float:
    p = np.asarray(left, dtype=np.float64).reshape(-1)
    q = np.asarray(right, dtype=np.float64).reshape(-1)
    middle = 0.5 * (p + q)

    def kl(values: np.ndarray) -> float:
        positive = values > 0
        return float(np.sum(values[positive] * np.log(values[positive] / middle[positive])))

    return float(np.clip((0.5 * (kl(p) + kl(q))) / math.log(2.0), 0.0, 1.0))


def top_mask(probability: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    flat = np.asarray(probability).reshape(-1)
    count = max(1, int(math.ceil(flat.size * fraction)))
    indices = np.argsort(-flat, kind="stable")[:count]
    mask = np.zeros(flat.size, dtype=bool)
    mask[indices] = True
    return mask.reshape(probability.shape)


def spatial_separation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"Grid mismatch: {p.shape} != {q.shape}")
    jsd = normalized_jsd(p, q)
    tv = 0.5 * float(np.abs(p - q).sum())
    pmask, qmask = top_mask(p), top_mask(q)
    union = int(np.logical_or(pmask, qmask).sum())
    intersection = int(np.logical_and(pmask, qmask).sum())
    top_disagreement = 1.0 - (intersection / union if union else 1.0)
    ppeak = np.unravel_index(int(np.argmax(p)), p.shape)
    qpeak = np.unravel_index(int(np.argmax(q)), q.shape)
    diagonal = math.hypot(max(0, p.shape[0] - 1), max(0, p.shape[1] - 1))
    peak_distance = math.hypot(ppeak[0] - qpeak[0], ppeak[1] - qpeak[1]) / max(diagonal, 1e-12)
    score = float(np.mean([jsd, tv, top_disagreement, peak_distance]))
    return {
        "score": score,
        "jsd_normalized": jsd,
        "total_variation": tv,
        "top10_disagreement": float(top_disagreement),
        "peak_normalized_distance": float(peak_distance),
    }


def robust(values: Sequence[float], penalty: float = 0.5, absolute_center: bool = False) -> float:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    q25, q75 = np.quantile(array, [0.25, 0.75])
    center = abs(median) if absolute_center else median
    return max(0.0, center - penalty * float(q75 - q25))


def _metric_source(case: CaseData, model: str, query: str, head: int | None = None) -> list[GridMetric]:
    if head is None:
        return case.metrics[model][query]
    return case.head_metrics[model][query][head]


def _raw_source(case: CaseData, model: str, query: str, head: int | None = None) -> list[np.ndarray]:
    if head is None:
        return case.raw[model][query]
    return case.head_raw[model][query][head]


def map_stability(metrics: Sequence[GridMetric], layers: Sequence[int]) -> float:
    values = [
        normalized_jsd(metrics[left].probability, metrics[right].probability)
        for left, right in zip(layers[:-1], layers[1:])
    ]
    return float(np.clip(1.0 - np.median(values), 0.0, 1.0)) if values else 1.0


def aggregate_raw(case: CaseData, model: str, query: str, layers: Sequence[int], head: int | None = None) -> np.ndarray:
    grids = _raw_source(case, model, query, head)
    return np.mean([grids[layer] for layer in layers], axis=0, dtype=np.float64)


def aggregate_probability(
    case: CaseData,
    model: str,
    query: str,
    layers: Sequence[int],
    head: int | None = None,
) -> np.ndarray:
    raw = aggregate_raw(case, model, query, layers, head)
    total = float(raw.sum())
    return raw / total if total > 0 else np.full(raw.shape, 1.0 / raw.size)


def build_mass_percentiles(cases: Sequence[CaseData]) -> dict[tuple[str, str, str, int], float]:
    populations: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for case in cases:
        for model in RANK_MODELS:
            for query in case.metrics[model]:
                for layer in LAYERS:
                    populations[(case.dataset, query, layer, model)].append(case.metrics[model][query][layer].mass)
    sorted_values = {key: np.sort(values) for key, values in populations.items()}
    result: dict[tuple[str, str, str, int], float] = {}
    for case in cases:
        for model in RANK_MODELS:
            for query in case.metrics[model]:
                for layer in LAYERS:
                    values = sorted_values[(case.dataset, query, layer, model)]
                    mass = case.metrics[model][query][layer].mass
                    result[(case.case_id, model, query, layer)] = float(
                        np.searchsorted(values, mass, side="right") / len(values)
                    )
    return result


def engagement(case: CaseData, model: str, query: str, layers: Sequence[int], head: int | None = None) -> dict[str, float]:
    metrics = _metric_source(case, model, query, head)
    per_layer = [
        metrics[layer].mass * metrics[layer].concentration * (1.0 - metrics[layer].corner_penalty)
        for layer in layers
    ]
    stability = map_stability(metrics, layers)
    return {
        "score": robust(per_layer) * stability,
        "raw_image_mass_median": float(np.median([metrics[layer].mass for layer in layers])),
        "normalized_spatial_entropy_median": float(
            np.median([metrics[layer].normalized_entropy for layer in layers])
        ),
        "concentration_median": float(np.median([metrics[layer].concentration for layer in layers])),
        "top10_percent_conditional_mass_median": float(
            np.median([metrics[layer].top10_mass for layer in layers])
        ),
        "corner_penalty_median": float(np.median([metrics[layer].corner_penalty for layer in layers])),
        "cross_layer_stability": stability,
    }


def separation_row(
    case: CaseData,
    query: str,
    layers: Sequence[int],
    percentiles: Mapping[tuple[str, str, str, int], float],
) -> dict[str, Any]:
    ours = case.metrics["ours"][query]
    ttrv = case.metrics["ttrv"][query]
    components = [spatial_separation(ours[layer].probability, ttrv[layer].probability) for layer in layers]
    base = robust([item["score"] for item in components])
    ours_rho = float(np.median([percentiles[(case.case_id, "ours", query, layer)] for layer in layers]))
    ttrv_rho = float(np.median([percentiles[(case.case_id, "ttrv", query, layer)] for layer in layers]))
    support = math.sqrt(ours_rho * ttrv_rho)
    ours_stability = map_stability(ours, layers)
    ttrv_stability = map_stability(ttrv, layers)
    corner = max(
        float(np.median([ours[layer].corner_penalty for layer in layers])),
        float(np.median([ttrv[layer].corner_penalty for layer in layers])),
    )
    score = base * support * math.sqrt(ours_stability * ttrv_stability) * (1.0 - corner)
    return {
        "score": score,
        "raw_separation_robust": base,
        "mass_percentile_ours": ours_rho,
        "mass_percentile_ttrv": ttrv_rho,
        "mass_support": support,
        "stability_ours": ours_stability,
        "stability_ttrv": ttrv_stability,
        "corner_penalty_pair": corner,
        "components_median": {
            key: float(np.median([item[key] for item in components]))
            for key in ("jsd_normalized", "total_variation", "top10_disagreement", "peak_normalized_distance")
        },
    }


def safe_row(case: CaseData, scheme: str, score: float, layers: Sequence[int], query: str) -> dict[str, Any]:
    diagnostic_query = query if query in QUERIES else PRIMARY_QUERY
    diagnostics = {}
    for model in MODELS:
        metrics = case.metrics[model][diagnostic_query]
        diagnostics[model] = {
            "raw_image_mass_median": float(np.median([metrics[layer].mass for layer in layers])),
            "normalized_spatial_entropy_median": float(
                np.median([metrics[layer].normalized_entropy for layer in layers])
            ),
            "top10_percent_conditional_mass_median": float(
                np.median([metrics[layer].top10_mass for layer in layers])
            ),
            "corner_penalty_median": float(
                np.median([metrics[layer].corner_penalty for layer in layers])
            ),
        }
    jsd_values = [
        normalized_jsd(
            case.metrics["ours"][diagnostic_query][layer].probability,
            case.metrics["ttrv"][diagnostic_query][layer].probability,
        )
        for layer in layers
    ]
    return {
        "scheme": scheme,
        "case_id": case.case_id,
        "dataset": case.dataset,
        "question": case.question,
        "image_path": case.image_path,
        "score": float(score),
        "query": query,
        "layers_zero_based": [int(value) for value in layers],
        "deterministic_tie_break": tie_break(case.case_id),
        "head_mean_T1_window_diagnostics": diagnostics,
        "ours_ttrv_jsd_normalized_median": float(np.median(jsd_values)),
    }


def balanced_location(rows: Sequence[Mapping[str, Any]]) -> float:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(float(row["score"]))
    return float(np.mean([np.median(values) for _, values in sorted(by_dataset.items())]))


def choose_window(
    cases: Sequence[CaseData],
    builder: Callable[[CaseData, Sequence[int]], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    rows_by_layers: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for layers in WINDOWS:
        rows = [builder(case, layers) for case in cases]
        rows_by_layers[layers] = rows
        candidates.append(
            {
                "layers_zero_based": list(layers),
                "dataset_balanced_median_score": balanced_location(rows),
                "case_score_median": float(np.median([row["score"] for row in rows])),
            }
        )
    candidates.sort(
        key=lambda row: (
            -float(row["dataset_balanced_median_score"]),
            -float(row["case_score_median"]),
            tuple(row["layers_zero_based"]),
        )
    )
    chosen = tuple(int(value) for value in candidates[0]["layers_zero_based"])
    return rows_by_layers[chosen], candidates


def global_rank(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (-float(row["score"]), str(row["deterministic_tie_break"])))
    result = []
    for rank, row in enumerate(ranked, start=1):
        item = dict(row)
        item["global_rank"] = rank
        result.append(item)
    return result


def balanced_select(
    ranked: Sequence[dict[str, Any]],
    quota: int,
    group_key: str | None = None,
) -> list[dict[str, Any]]:
    datasets = sorted({str(row["dataset"]) for row in ranked})
    groups = sorted({str(row.get(group_key, "all")) for row in ranked}) if group_key else ["all"]
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for group in groups:
        for dataset in datasets:
            buckets[(group, dataset)] = [
                row
                for row in ranked
                if str(row["dataset"]) == dataset and (not group_key or str(row.get(group_key)) == group)
            ]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    positions = {key: 0 for key in buckets}
    while len(selected) < quota:
        progressed = False
        for key in sorted(buckets):
            values = buckets[key]
            while positions[key] < len(values) and values[positions[key]]["case_id"] in selected_ids:
                positions[key] += 1
            if positions[key] >= len(values):
                continue
            row = values[positions[key]]
            positions[key] += 1
            selected.append(dict(row))
            selected_ids.add(str(row["case_id"]))
            progressed = True
            if len(selected) >= quota:
                break
        if not progressed:
            break
    if len(selected) < quota:
        for row in ranked:
            if row["case_id"] not in selected_ids:
                selected.append(dict(row))
                selected_ids.add(str(row["case_id"]))
                if len(selected) >= quota:
                    break
    for order, row in enumerate(selected, start=1):
        row["selection_order"] = order
    return selected


def strong_direct_scheme(cases: Sequence[CaseData]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def build(case: CaseData, layers: Sequence[int]) -> dict[str, Any]:
        values = {model: engagement(case, model, PRIMARY_QUERY, layers) for model in RANK_MODELS}
        dominant = max(RANK_MODELS, key=lambda model: (values[model]["score"], model))
        row = safe_row(case, "strong_direct_mass", values[dominant]["score"], layers, PRIMARY_QUERY)
        row.update({"selection_group": f"{dominant}_dominant", "dominant_model": dominant, "model_metrics": values})
        return row

    rows, windows = choose_window(cases, build)
    return rows, {"window_candidates": windows, "balanced_group_key": "selection_group"}


def mass_contrast_scheme(
    cases: Sequence[CaseData],
    percentiles: Mapping[tuple[str, str, str, int], float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def build(case: CaseData, layers: Sequence[int]) -> dict[str, Any]:
        ours = case.metrics["ours"][PRIMARY_QUERY]
        ttrv = case.metrics["ttrv"][PRIMARY_QUERY]
        log_ratios = [math.log((ours[layer].mass + 1e-12) / (ttrv[layer].mass + 1e-12)) for layer in layers]
        median_ratio = float(np.median(log_ratios))
        ratio_strength = robust(log_ratios, absolute_center=True)
        rho_ours = float(np.median([percentiles[(case.case_id, "ours", PRIMARY_QUERY, layer)] for layer in layers]))
        rho_ttrv = float(np.median([percentiles[(case.case_id, "ttrv", PRIMARY_QUERY, layer)] for layer in layers]))
        stability_ours, stability_ttrv = map_stability(ours, layers), map_stability(ttrv, layers)
        corner = max(
            float(np.median([ours[layer].corner_penalty for layer in layers])),
            float(np.median([ttrv[layer].corner_penalty for layer in layers])),
        )
        score = ratio_strength * max(rho_ours, rho_ttrv) * math.sqrt(stability_ours * stability_ttrv) * (1.0 - corner)
        direction = "ours_higher" if median_ratio >= 0 else "ttrv_higher"
        row = safe_row(case, "bidirectional_raw_mass_contrast", score, layers, PRIMARY_QUERY)
        row.update(
            {
                "selection_group": direction,
                "direction": direction,
                "median_log_mass_ratio_ours_over_ttrv": median_ratio,
                "robust_absolute_log_mass_ratio": ratio_strength,
                "mass_percentile_ours": rho_ours,
                "mass_percentile_ttrv": rho_ttrv,
                "corner_penalty_pair": corner,
                "stability_ours": stability_ours,
                "stability_ttrv": stability_ttrv,
            }
        )
        return row

    rows, windows = choose_window(cases, build)
    return rows, {"window_candidates": windows, "balanced_group_key": "selection_group"}


def spatial_scheme(
    cases: Sequence[CaseData],
    percentiles: Mapping[tuple[str, str, str, int], float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def build(case: CaseData, layers: Sequence[int]) -> dict[str, Any]:
        metrics = separation_row(case, PRIMARY_QUERY, layers, percentiles)
        row = safe_row(case, "mass_supported_spatial_separation", metrics["score"], layers, PRIMARY_QUERY)
        row.update(metrics)
        return row

    rows, windows = choose_window(cases, build)
    return rows, {"window_candidates": windows, "balanced_group_key": None}


def ours_focus_quality_gain_scheme(
    cases: Sequence[CaseData],
    percentiles: Mapping[tuple[str, str, str, int], float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank positive Ours-vs-TTRV mass/focus gains at raw T=1."""

    def build(case: CaseData, layers: Sequence[int]) -> dict[str, Any]:
        ours = case.metrics["ours"][PRIMARY_QUERY]
        ttrv = case.metrics["ttrv"][PRIMARY_QUERY]
        mass_gains, entropy_reductions, top10_gains, quality_gains = [], [], [], []
        for layer in layers:
            mass_gain = (ours[layer].mass - ttrv[layer].mass) / (
                ours[layer].mass + ttrv[layer].mass + 1e-12
            )
            entropy_reduction = ttrv[layer].normalized_entropy - ours[layer].normalized_entropy
            top10_gain = ours[layer].top10_mass - ttrv[layer].top10_mass
            quality_gain = 0.4 * mass_gain + 0.3 * entropy_reduction + 0.3 * top10_gain
            mass_gains.append(mass_gain)
            entropy_reductions.append(entropy_reduction)
            top10_gains.append(top10_gain)
            quality_gains.append(quality_gain)
        gain = robust(quality_gains)
        mass_support = float(
            np.median([percentiles[(case.case_id, "ours", PRIMARY_QUERY, layer)] for layer in layers])
        )
        stability = map_stability(ours, layers)
        corner = float(np.median([ours[layer].corner_penalty for layer in layers]))
        score = gain * mass_support * stability * (1.0 - corner)
        row = safe_row(case, "ours_focus_quality_gain", score, layers, PRIMARY_QUERY)
        row.update(
            {
                "raw_T1_quality_gain_robust": gain,
                "ours_mass_support_percentile": mass_support,
                "ours_cross_layer_stability": stability,
                "ours_corner_penalty": corner,
                "mass_gain_median": float(np.median(mass_gains)),
                "normalized_entropy_reduction_median": float(np.median(entropy_reductions)),
                "top10_percent_mass_gain_median": float(np.median(top10_gains)),
            }
        )
        return row

    rows, windows = choose_window(cases, build)
    return rows, {
        "window_candidates": windows,
        "balanced_group_key": None,
        "direction_is_preregistered": "positive Ours minus TTRV quality gain",
        "quality_formula": "0.4 normalized raw-mass gain + 0.3 entropy reduction + 0.3 Top10% mass gain",
    }


def query_consensus_scheme(
    cases: Sequence[CaseData],
    percentiles: Mapping[tuple[str, str, str, int], float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def build(case: CaseData, layers: Sequence[int]) -> dict[str, Any]:
        per_query = {query: separation_row(case, query, layers, percentiles) for query in QUERIES}
        robust_queries = robust([per_query[query]["score"] for query in QUERIES])
        consistencies = {}
        for model in RANK_MODELS:
            maps = [aggregate_probability(case, model, query, layers) for query in QUERIES]
            pairwise = [normalized_jsd(maps[left], maps[right]) for left in range(3) for right in range(left + 1, 3)]
            consistencies[model] = float(np.clip(1.0 - np.median(pairwise), 0.0, 1.0))
        score = robust_queries * math.sqrt(consistencies["ours"] * consistencies["ttrv"])
        row = safe_row(case, "three_query_consensus", score, layers, "all_three_queries")
        row.update(
            {
                "per_query": per_query,
                "robust_query_separation": robust_queries,
                "query_map_consistency": consistencies,
            }
        )
        return row

    rows, windows = choose_window(cases, build)
    return rows, {"window_candidates": windows, "balanced_group_key": None, "queries": list(QUERIES)}


def query_trajectory_scheme(cases: Sequence[CaseData]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first, answer = QUERIES[0], QUERIES[2]

    def build(case: CaseData, layers: Sequence[int]) -> dict[str, Any]:
        gains: dict[str, list[float]] = {}
        gain_metrics: dict[str, dict[str, float]] = {}
        for model in RANK_MODELS:
            values = []
            mass_gains, concentration_gains, spatial_shifts = [], [], []
            for layer in layers:
                f = case.metrics[model][first][layer]
                a = case.metrics[model][answer][layer]
                mass_gain = (a.mass - f.mass) / (a.mass + f.mass + 1e-12)
                concentration_gain = a.concentration - f.concentration
                shift = normalized_jsd(a.probability, f.probability)
                mass_gains.append(mass_gain)
                concentration_gains.append(concentration_gain)
                spatial_shifts.append(shift)
                values.append(0.5 * mass_gain + 0.3 * concentration_gain + 0.2 * shift)
            gains[model] = values
            gain_metrics[model] = {
                "gain_median": float(np.median(values)),
                "mass_gain_median": float(np.median(mass_gains)),
                "concentration_gain_median": float(np.median(concentration_gains)),
                "spatial_shift_median": float(np.median(spatial_shifts)),
            }
        deltas = [left - right for left, right in zip(gains["ours"], gains["ttrv"])]
        median_delta = float(np.median(deltas))
        score = robust(deltas, absolute_center=True)
        direction = "ours_larger_transition" if median_delta >= 0 else "ttrv_larger_transition"
        row = safe_row(case, "query_stage_trajectory", score, layers, "first_to_answer_with_decision_middle")
        row.update(
            {
                "selection_group": direction,
                "direction": direction,
                "median_ours_minus_ttrv_transition": median_delta,
                "model_transition_metrics": gain_metrics,
            }
        )
        return row

    rows, windows = choose_window(cases, build)
    return rows, {"window_candidates": windows, "balanced_group_key": "selection_group", "queries": list(QUERIES)}


def depth_persistence_scheme(
    cases: Sequence[CaseData],
    percentiles: Mapping[tuple[str, str, str, int], float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    display_layers = (0, 13, 27)
    for case in cases:
        model_values = {}
        for model in RANK_MODELS:
            metrics = case.metrics[model][PRIMARY_QUERY]
            engagement_values = [
                item.mass * item.concentration * (1.0 - item.corner_penalty) for item in metrics
            ]
            coverage = float(
                np.mean(
                    [percentiles[(case.case_id, model, PRIMARY_QUERY, layer)] >= 0.5 for layer in LAYERS]
                )
            )
            stability = map_stability(metrics, LAYERS)
            model_values[model] = {
                "score": float(np.mean(engagement_values)) * coverage * stability,
                "mean_mass_weighted_concentration": float(np.mean(engagement_values)),
                "above_median_mass_layer_coverage": coverage,
                "adjacent_layer_stability": stability,
                "raw_image_mass_by_layer": [item.mass for item in metrics],
                "concentration_by_layer": [item.concentration for item in metrics],
                "corner_penalty_by_layer": [item.corner_penalty for item in metrics],
            }
        dominant = max(RANK_MODELS, key=lambda model: (model_values[model]["score"], model))
        row = safe_row(case, "depth_persistence_atlas", model_values[dominant]["score"], display_layers, PRIMARY_QUERY)
        row.update({"selection_group": f"{dominant}_dominant", "dominant_model": dominant, "model_metrics": model_values})
        rows.append(row)
    return rows, {
        "balanced_group_key": "selection_group",
        "all_layers_used_for_ranking": list(LAYERS),
        "display_anchor_layers": list(display_layers),
    }


def per_head_scheme(cases: Sequence[CaseData]) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    common: set[int] | None = None
    for case in cases:
        for model in MODELS:
            heads = set(case.head_metrics.get(model, {}).get(PRIMARY_QUERY, {}))
            common = heads if common is None else common & heads
    if not common:
        return None, {
            "status": "skipped",
            "reason": "No common complete per-head keys were found for Base/TTRV/Ours and all 28 layers.",
            "recognized_key_patterns": [pattern.pattern for pattern in HEAD_PATTERNS],
        }
    candidates = []
    rows_by_setting: dict[tuple[int, tuple[int, ...]], list[dict[str, Any]]] = {}
    for head in sorted(common):
        for layers in WINDOWS:
            rows = []
            for case in cases:
                values = {
                    model: engagement(case, model, PRIMARY_QUERY, layers, head=head) for model in RANK_MODELS
                }
                dominant = max(RANK_MODELS, key=lambda model: (values[model]["score"], model))
                row = safe_row(case, "per_head_specialist", values[dominant]["score"], layers, PRIMARY_QUERY)
                row.update(
                    {
                        "selection_group": f"{dominant}_dominant",
                        "dominant_model": dominant,
                        "head_index_zero_based": head,
                        "model_metrics": values,
                    }
                )
                rows.append(row)
            rows_by_setting[(head, layers)] = rows
            candidates.append(
                {
                    "head_index_zero_based": head,
                    "layers_zero_based": list(layers),
                    "dataset_balanced_median_score": balanced_location(rows),
                }
            )
    candidates.sort(
        key=lambda row: (
            -float(row["dataset_balanced_median_score"]),
            int(row["head_index_zero_based"]),
            tuple(row["layers_zero_based"]),
        )
    )
    best = candidates[0]
    setting = (int(best["head_index_zero_based"]), tuple(int(value) for value in best["layers_zero_based"]))
    return rows_by_setting[setting], {
        "status": "available",
        "balanced_group_key": "selection_group",
        "common_heads": sorted(common),
        "setting_candidates": candidates,
    }


def per_case_layer_head_discovery_scheme(
    cases: Sequence[CaseData],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Attention-only post-hoc discovery; never present this as a shared setting."""

    rows: list[dict[str, Any]] = []
    skipped_cases: list[str] = []
    head_counts: dict[str, int] = {}
    for case in cases:
        common: set[int] | None = None
        for model in MODELS:
            heads = set(case.head_metrics.get(model, {}).get(PRIMARY_QUERY, {}))
            common = heads if common is None else common & heads
        if not common:
            skipped_cases.append(case.case_id)
            continue
        head_counts[case.case_id] = len(common)
        settings: list[tuple[float, int, int, str, tuple[int, ...], dict[str, dict[str, float]]]] = []
        for head in sorted(common):
            for selected_layer in LAYERS:
                support_layers = tuple(
                    sorted({max(0, selected_layer - 1), selected_layer, min(27, selected_layer + 1)})
                )
                model_metrics = {
                    model: engagement(case, model, PRIMARY_QUERY, support_layers, head=head)
                    for model in RANK_MODELS
                }
                dominant = max(RANK_MODELS, key=lambda model: (model_metrics[model]["score"], model))
                settings.append(
                    (
                        float(model_metrics[dominant]["score"]),
                        head,
                        selected_layer,
                        dominant,
                        support_layers,
                        model_metrics,
                    )
                )
        settings.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        score, head, selected_layer, dominant, support_layers, model_metrics = settings[0]
        row = safe_row(
            case,
            "per_case_layer_head_discovery",
            score,
            [selected_layer],
            PRIMARY_QUERY,
        )
        row.update(
            {
                "selection_group": f"{dominant}_dominant",
                "dominant_model": dominant,
                "head_index_zero_based": head,
                "selected_layer_zero_based": selected_layer,
                "adjacent_support_layers_zero_based": list(support_layers),
                "per_head_model_metrics_with_adjacent_support": model_metrics,
                "diagnostic_post_hoc": True,
                "correctness_blind": True,
                "shared_setting": False,
                "settings_examined_for_case": len(settings),
            }
        )
        rows.append(row)
    if not rows:
        return None, {
            "status": "skipped",
            "reason": "No case had common complete per-head keys for Base/TTRV/Ours.",
            "diagnostic_post_hoc": True,
        }
    return rows, {
        "status": "available",
        "balanced_group_key": "selection_group",
        "diagnostic_post_hoc": True,
        "correctness_blind": True,
        "shared_setting": False,
        "warning": (
            "Each case independently maximizes raw-T1 attention quality over layer and head. "
            "This is an attention-only discovery diagnostic, not a preregistered shared-setting comparison."
        ),
        "search_space": "28 target layers x common heads x Ours/TTRV, with clipped adjacent-layer support",
        "head_count_by_case": head_counts,
        "skipped_case_ids": skipped_cases,
    }


def display_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("display temperature must be positive")
    p = np.asarray(probability, dtype=np.float64)
    if temperature == 1.0:
        return p
    transformed = np.power(np.maximum(p, 0.0), 1.0 / temperature)
    total = float(transformed.sum())
    return transformed / total if total > 0 else np.full(p.shape, 1.0 / p.size)


def _resize(grid: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(np.asarray(grid, dtype=np.float32), mode="F")
    return np.asarray(image.resize(size, resample=Image.Resampling.BILINEAR), dtype=np.float32)


def _open_image(case: CaseData):
    from PIL import Image

    with Image.open(case.image_path) as handle:
        return handle.convert("RGB").copy()


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _save_canvas(canvas: Any, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = Path(str(stem) + ".png")
    pdf = Path(str(stem) + ".pdf")
    canvas.save(png, format="PNG", optimize=True)
    canvas.convert("RGB").save(pdf, format="PDF", resolution=220.0)
    return {"png": str(png.resolve()), "pdf": str(pdf.resolve())}


def _fit(image: Any, width: int, height: int) -> tuple[Any, int, int]:
    from PIL import Image

    scale = min(width / image.width, height / image.height)
    resized_width = max(1, int(round(image.width * scale)))
    resized_height = max(1, int(round(image.height * scale)))
    resized = image.resize((resized_width, resized_height), resample=Image.Resampling.LANCZOS)
    return resized, (width - resized_width) // 2, (height - resized_height) // 2


def _overlay(original: Any, grid: np.ndarray, vmax: float, alpha: float = 0.55):
    import cv2
    from PIL import Image

    normalized = np.clip(np.asarray(grid, dtype=np.float64) / max(vmax, 1e-12), 0.0, 1.0)
    colors = cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
    heat = Image.fromarray(colors).resize(original.size, resample=Image.Resampling.BILINEAR)
    return Image.blend(original, heat, alpha=alpha)


def _draw_colorbar(canvas: Any, x: int, y: int, height: int, vmax: float, label: str) -> None:
    import cv2
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(canvas)
    gradient = np.linspace(255, 0, max(2, height), dtype=np.uint8).reshape(-1, 1)
    gradient = np.repeat(gradient, 24, axis=1)
    colors = cv2.applyColorMap(gradient, cv2.COLORMAP_VIRIDIS)
    colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
    canvas.paste(Image.fromarray(colors), (x, y))
    draw.rectangle((x, y, x + 23, y + height), outline="black", width=1)
    small = _font(12)
    draw.text((x - 6, y - 17), f"{vmax:.3g}", fill="black", font=small)
    draw.text((x + 5, y + height + 2), "0", fill="black", font=small)
    draw.text((x - 10, y + height + 20), label[:14], fill=(60, 60, 60), font=small)


def render_main(case: CaseData, row: Mapping[str, Any], output_dir: Path, display_t: float) -> dict[str, str]:
    from PIL import Image, ImageDraw

    original = _open_image(case)
    layers = [int(value) for value in row["layers_zero_based"]]
    query = str(row.get("query", PRIMARY_QUERY))
    if query not in QUERIES:
        query = PRIMARY_QUERY
    head = int(row["head_index_zero_based"]) if "head_index_zero_based" in row else None
    raw_maps = {model: aggregate_raw(case, model, query, layers, head) for model in MODELS}
    conditional = {
        model: raw / float(raw.sum()) if float(raw.sum()) > 0 else np.full(raw.shape, 1.0 / raw.size)
        for model, raw in raw_maps.items()
    }
    sharpened = {model: display_temperature(value, display_t) for model, value in conditional.items()}
    modes = (
        ("Absolute raw attention, T=1", raw_maps, False),
        ("Conditional spatial p, T=1", conditional, False),
        (f"Conditional p, T={display_t:g} DISPLAY-ONLY", sharpened, True),
    )
    panel_width, panel_height = 410, 290
    title_height, bar_width = 126, 78
    canvas = Image.new(
        "RGB",
        (panel_width * 4 + bar_width, title_height + panel_height * 3),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font, header_font, body_font, small_font = _font(20, True), _font(17, True), _font(14), _font(12)
    question = " ".join(case.question.split())
    title_lines = [
        f"{row['scheme']} | global rank {row.get('global_rank')} | {case.dataset}/{case.case_id}",
        f"raw-T1 score={float(row['score']):.5g} | query={query} | layers={layers}"
        + (f" | shared head={head}" if head is not None else " | head-mean"),
        *textwrap.wrap(question, width=150)[:2],
    ]
    for index, line in enumerate(title_lines):
        draw.text((14, 8 + index * 25), line, fill="black", font=title_font if index == 0 else body_font)
    for row_index, (label, maps, display_only) in enumerate(modes):
        vmax = max(float(np.max(value)) for value in maps.values())
        vmax = vmax if vmax > 0 else 1.0
        row_y = title_height + row_index * panel_height
        draw.text((12, row_y + 5), "Original image" if row_index == 0 else label, fill="black", font=header_font)
        fitted, x_pad, y_pad = _fit(original, panel_width - 22, panel_height - 52)
        canvas.paste(fitted, (11 + x_pad, row_y + 38 + y_pad))
        for column, model in enumerate(MODELS, start=1):
            mass = float(raw_maps[model].sum())
            draw.text(
                (column * panel_width + 12, row_y + 5),
                f"{model.upper()} | raw mass={mass:.4g}",
                fill="black",
                font=header_font,
            )
            overlay = _overlay(original, maps[model], vmax)
            fitted, x_pad, y_pad = _fit(overlay, panel_width - 22, panel_height - 52)
            canvas.paste(fitted, (column * panel_width + 11 + x_pad, row_y + 38 + y_pad))
        _draw_colorbar(
            canvas,
            panel_width * 4 + 18,
            row_y + 44,
            panel_height - 100,
            vmax,
            "display-only" if display_only else "raw T=1",
        )
        if display_only:
            draw.text(
                (12, row_y + panel_height - 18),
                "Temperature never affects selection, layers, or rank.",
                fill=(150, 0, 0),
                font=small_font,
            )
    stem = output_dir / f"{int(row['selection_order']):02d}_{sanitize(case.case_id)}_main"
    return _save_canvas(canvas, stem)


def render_query_atlas(
    case: CaseData,
    row: Mapping[str, Any],
    output_dir: Path,
    temperature: float,
) -> dict[str, str]:
    from PIL import Image, ImageDraw

    original = _open_image(case)
    layers = [int(value) for value in row["layers_zero_based"]]
    panel_width, panel_height = 410, 290
    title_height, bar_width = 92, 78
    canvas = Image.new("RGB", (panel_width * 4 + bar_width, title_height + panel_height * 3), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, header_font, small_font = _font(20, True), _font(15, True), _font(12)
    note = "raw T=1" if temperature == 1.0 else f"T={temperature:g} DISPLAY-ONLY"
    draw.text(
        (14, 8),
        f"Three-query atlas | {note} | {case.dataset}/{case.case_id} | layers={layers}",
        fill="black",
        font=title_font,
    )
    draw.text(
        (14, 40),
        "decision_query/answer_state are corroborative and content/length conditioned.",
        fill=(120, 0, 0),
        font=small_font,
    )
    for row_index, query in enumerate(QUERIES):
        maps = {
            model: display_temperature(aggregate_probability(case, model, query, layers), temperature)
            for model in MODELS
        }
        vmax = max(float(value.max()) for value in maps.values()) or 1.0
        row_y = title_height + row_index * panel_height
        draw.text((12, row_y + 5), query, fill="black", font=header_font)
        fitted, x_pad, y_pad = _fit(original, panel_width - 22, panel_height - 48)
        canvas.paste(fitted, (11 + x_pad, row_y + 34 + y_pad))
        for column, model in enumerate(MODELS, start=1):
            mass = float(aggregate_raw(case, model, query, layers).sum())
            draw.text(
                (column * panel_width + 12, row_y + 5),
                f"{model.upper()} | mass={mass:.4g}",
                fill="black",
                font=header_font,
            )
            overlay = _overlay(original, maps[model], vmax)
            fitted, x_pad, y_pad = _fit(overlay, panel_width - 22, panel_height - 48)
            canvas.paste(fitted, (column * panel_width + 11 + x_pad, row_y + 34 + y_pad))
        _draw_colorbar(canvas, panel_width * 4 + 18, row_y + 42, panel_height - 96, vmax, note)
    suffix = "T1" if temperature == 1.0 else f"T{temperature:g}_display_only"
    stem = output_dir / f"{int(row['selection_order']):02d}_{sanitize(case.case_id)}_query_atlas_{suffix}"
    return _save_canvas(canvas, stem)


def render_depth_atlas(case: CaseData, row: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    from PIL import Image, ImageDraw

    original = _open_image(case)
    anchors = [int(value) for value in row["layers_zero_based"]]
    panel_width, panel_height = 410, 285
    title_height = 92
    canvas = Image.new("RGB", (panel_width * 4, title_height + panel_height * 4), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, header_font, small_font = _font(20, True), _font(15, True), _font(12)
    draw.text(
        (14, 8),
        f"Depth persistence atlas | rank {row.get('global_rank')} | {case.dataset}/{case.case_id}",
        fill="black",
        font=title_font,
    )
    draw.text(
        (14, 40),
        f"raw-T1 score={float(row['score']):.5g} | all 28 layers used for ranking",
        fill=(60, 60, 60),
        font=small_font,
    )
    fitted, x_pad, y_pad = _fit(original, panel_width - 22, panel_height - 44)
    canvas.paste(fitted, (11 + x_pad, title_height + 32 + y_pad))
    draw.text((12, title_height + 5), "Original", fill="black", font=header_font)
    curve_left = panel_width + 55
    curve_top = title_height + 38
    curve_right = panel_width * 4 - 35
    curve_bottom = title_height + panel_height - 42
    draw.rectangle((curve_left, curve_top, curve_right, curve_bottom), outline=(80, 80, 80), width=1)
    colors = {"base": (70, 70, 70), "ttrv": (220, 90, 45), "ours": (40, 120, 210)}
    all_masses = [metric.mass for model in MODELS for metric in case.metrics[model][PRIMARY_QUERY]]
    ymax = max(all_masses) if max(all_masses) > 0 else 1.0
    for model in MODELS:
        masses = [metric.mass for metric in case.metrics[model][PRIMARY_QUERY]]
        points = []
        for layer, mass in enumerate(masses):
            x = curve_left + int(round(layer / 27 * (curve_right - curve_left)))
            y = curve_bottom - int(round(mass / ymax * (curve_bottom - curve_top)))
            points.append((x, y))
        draw.line(points, fill=colors[model], width=3)
    draw.text((curve_left, title_height + 5), "All 28 layers: raw image attention mass", fill="black", font=header_font)
    draw.text((curve_left, curve_bottom + 7), "layer 0", fill="black", font=small_font)
    draw.text((curve_right - 48, curve_bottom + 7), "layer 27", fill="black", font=small_font)
    draw.text((curve_right - 210, curve_top + 5), f"shared ymax={ymax:.4g}", fill=(60, 60, 60), font=small_font)
    legend_x = curve_left + 10
    for index, model in enumerate(MODELS):
        y = curve_top + 8 + index * 20
        draw.line((legend_x, y + 6, legend_x + 30, y + 6), fill=colors[model], width=3)
        draw.text((legend_x + 38, y), model.upper(), fill="black", font=small_font)
    for row_index, layer in enumerate(anchors, start=1):
        maps = {model: case.metrics[model][PRIMARY_QUERY][layer].probability for model in MODELS}
        vmax = max(float(value.max()) for value in maps.values()) or 1.0
        row_y = title_height + row_index * panel_height
        draw.text((105, row_y + 105), f"Layer {layer}\nconditional T=1", fill="black", font=header_font)
        for column, model in enumerate(MODELS, start=1):
            draw.text(
                (column * panel_width + 12, row_y + 5),
                f"{model.upper()} | mass={case.metrics[model][PRIMARY_QUERY][layer].mass:.4g}",
                fill="black",
                font=header_font,
            )
            overlay = _overlay(original, maps[model], vmax)
            fitted, x_pad, y_pad = _fit(overlay, panel_width - 22, panel_height - 44)
            canvas.paste(fitted, (column * panel_width + 11 + x_pad, row_y + 32 + y_pad))
    stem = output_dir / f"{int(row['selection_order']):02d}_{sanitize(case.case_id)}_depth_atlas"
    return _save_canvas(canvas, stem)


def render_selected(
    cases: Mapping[str, CaseData],
    selected: Sequence[dict[str, Any]],
    scheme_dir: Path,
    display_temperature_value: float,
) -> list[dict[str, Any]]:
    rendered = []
    for row in selected:
        case = cases[str(row["case_id"])]
        source_original = Path(case.image_path)
        suffix = source_original.suffix if source_original.suffix else ".png"
        copied_original = (
            scheme_dir
            / "originals"
            / f"{int(row['selection_order']):02d}_{sanitize(case.case_id)}_original{suffix.lower()}"
        )
        copied_original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_original, copied_original)
        figures = {
            "main_absolute_conditional": render_main(
                case, row, scheme_dir / "figures", display_temperature_value
            )
        }
        if row["scheme"] in {"three_query_consensus", "query_stage_trajectory"}:
            figures["query_atlas_T1"] = render_query_atlas(case, row, scheme_dir / "figures", 1.0)
            figures["query_atlas_display_only"] = render_query_atlas(
                case, row, scheme_dir / "figures", display_temperature_value
            )
        if row["scheme"] == "depth_persistence_atlas":
            figures["depth_atlas"] = render_depth_atlas(case, row, scheme_dir / "figures")
        item = dict(row)
        item["figures"] = figures
        item["copied_original"] = str(copied_original.resolve())
        rendered.append(item)
    return rendered


def process_scheme(
    name: str,
    rows: Sequence[dict[str, Any]],
    extra_config: Mapping[str, Any],
    cases_by_id: Mapping[str, CaseData],
    output_root: Path,
    quota: int,
    display_temperature_value: float,
    render: bool,
) -> dict[str, Any]:
    destination = output_root / name
    ranked = global_rank(rows)
    group_key = extra_config.get("balanced_group_key")
    selected = balanced_select(ranked, min(quota, len(ranked)), str(group_key) if group_key else None)
    write_jsonl(destination / "ranked_all.jsonl", ranked)
    write_jsonl(destination / "selected.jsonl", selected)
    write_csv(destination / "selected.csv", selected)
    config = {
        "scheme": name,
        "candidate_count": len(ranked),
        "selection_quota": quota,
        "selected_count": len(selected),
        "selection_temperature": 1.0,
        "display_temperature": display_temperature_value,
        "display_temperature_affects_ranking": False,
        "dataset_balancing": "deterministic round-robin fixed quota, then global backfill",
        "group_balancing_key": group_key,
        "deterministic_order": "score descending, then sha256(case_id)",
        "forbidden_ranking_fields": list(FORBIDDEN_RANK_FIELDS),
        "ranking_fields_used": [
            "saved raw attention arrays",
            "case_id",
            "dataset",
            "query mode",
            "layer index",
        ],
        **dict(extra_config),
    }
    write_json(destination / "selection_config.json", config)
    rendered = render_selected(cases_by_id, selected, destination, display_temperature_value) if render else selected
    if render:
        write_jsonl(destination / "selected_rendered.jsonl", rendered)
    return {
        "scheme": name,
        "status": "complete",
        "candidate_count": len(ranked),
        "selected_count": len(selected),
        "dataset_counts": dict(Counter(row["dataset"] for row in selected)),
        "group_counts": dict(Counter(str(row.get(group_key)) for row in selected)) if group_key else {},
        "ranked_all": str((destination / "ranked_all.jsonl").resolve()),
        "selected": str((destination / "selected.jsonl").resolve()),
        "selection_config": str((destination / "selection_config.json").resolve()),
        "rendered": bool(render),
        "selected_rows": rendered,
    }


def run(
    records_path: str | Path,
    output_dir: str | Path,
    quota_per_scheme: int = 8,
    display_temperature_value: float = 0.5,
    corner_fraction: float = 0.15,
    limit: int = 0,
    render: bool = True,
) -> dict[str, Any]:
    if quota_per_scheme <= 0:
        raise ValueError("quota_per_scheme must be positive")
    if not 0.0 < corner_fraction < 0.5:
        raise ValueError("corner_fraction must lie in (0, 0.5)")
    if display_temperature_value <= 0:
        raise ValueError("display temperature must be positive")
    records_file = Path(records_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cases, failures = load_cases(records_file, corner_fraction=corner_fraction, limit=limit)
    if not cases:
        raise RuntimeError("No complete cases with first_decision_query and all 28 layers")
    cases_by_id = {case.case_id: case for case in cases}
    available_queries = set(QUERIES)
    for case in cases:
        for model in MODELS:
            available_queries &= set(case.metrics[model])
    percentiles = build_mass_percentiles(cases)
    builders: list[tuple[str, tuple[list[dict[str, Any]], dict[str, Any]]]] = [
        ("01_strong_direct_mass", strong_direct_scheme(cases)),
        ("02_bidirectional_raw_mass_contrast", mass_contrast_scheme(cases, percentiles)),
        ("03_mass_supported_spatial_separation", spatial_scheme(cases, percentiles)),
        ("04_ours_focus_quality_gain", ours_focus_quality_gain_scheme(cases, percentiles)),
        ("07_depth_persistence_atlas", depth_persistence_scheme(cases, percentiles)),
    ]
    skipped_schemes: dict[str, dict[str, Any]] = {}
    if set(QUERIES).issubset(available_queries):
        builders.extend(
            [
                ("05_three_query_consensus", query_consensus_scheme(cases, percentiles)),
                ("06_query_stage_trajectory", query_trajectory_scheme(cases)),
            ]
        )
    else:
        missing = sorted(set(QUERIES) - available_queries)
        note = {
            "status": "skipped",
            "reason": f"Missing query modes: {missing}",
            "available_queries": sorted(available_queries),
            "extraction_note": (
                "Use max_new_tokens=16 for full three-query extraction. First-query-only artifacts remain "
                "valid for schemes 01-04 and 07-09."
            ),
        }
        skipped_schemes["05_three_query_consensus"] = dict(note)
        skipped_schemes["06_query_stage_trajectory"] = dict(note)
    head_rows, head_config = per_head_scheme(cases)
    if head_rows is not None:
        builders.append(("08_shared_per_head_specialist", (head_rows, head_config)))
    else:
        skipped_schemes["08_shared_per_head_specialist"] = head_config
    discovery_rows, discovery_config = per_case_layer_head_discovery_scheme(cases)
    if discovery_rows is not None:
        builders.append(("09_per_case_layer_head_discovery", (discovery_rows, discovery_config)))
    else:
        skipped_schemes["09_per_case_layer_head_discovery"] = discovery_config
    builders.sort(key=lambda item: item[0])
    scheme_summaries = []
    all_selected = []
    for name, (rows, config) in builders:
        summary = process_scheme(
            name=name,
            rows=rows,
            extra_config=config,
            cases_by_id=cases_by_id,
            output_root=destination,
            quota=quota_per_scheme,
            display_temperature_value=display_temperature_value,
            render=render,
        )
        all_selected.extend(summary.pop("selected_rows"))
        scheme_summaries.append(summary)
    for name, config in sorted(skipped_schemes.items()):
        scheme_summaries.append({"scheme": name, **config})
        write_json(destination / name / "selection_config.json", config)
        write_jsonl(destination / name / "ranked_all.jsonl", [])
        write_jsonl(destination / name / "selected.jsonl", [])
    scheme_summaries.sort(key=lambda item: str(item["scheme"]))
    write_jsonl(destination / "load_failures.jsonl", failures)
    write_jsonl(destination / "selected_all_schemes.jsonl", all_selected)
    flat_selected = []
    for row in all_selected:
        diagnostics = row.get("head_mean_T1_window_diagnostics", {})
        flat_selected.append(
            {
                "scheme": row["scheme"],
                "selection_order": row.get("selection_order"),
                "global_rank": row.get("global_rank"),
                "case_id": row["case_id"],
                "dataset": row["dataset"],
                "question": row["question"],
                "score": row["score"],
                "query": row["query"],
                "layers_zero_based": row["layers_zero_based"],
                "selection_group": row.get("selection_group"),
                "head_index_zero_based": row.get("head_index_zero_based"),
                "selected_layer_zero_based": row.get("selected_layer_zero_based"),
                "diagnostic_post_hoc": row.get("diagnostic_post_hoc", False),
                "raw_image_mass_base": diagnostics.get("base", {}).get("raw_image_mass_median"),
                "raw_image_mass_ttrv": diagnostics.get("ttrv", {}).get("raw_image_mass_median"),
                "raw_image_mass_ours": diagnostics.get("ours", {}).get("raw_image_mass_median"),
                "normalized_spatial_entropy_base": diagnostics.get("base", {}).get(
                    "normalized_spatial_entropy_median"
                ),
                "normalized_spatial_entropy_ttrv": diagnostics.get("ttrv", {}).get(
                    "normalized_spatial_entropy_median"
                ),
                "normalized_spatial_entropy_ours": diagnostics.get("ours", {}).get(
                    "normalized_spatial_entropy_median"
                ),
                "top10_percent_mass_base": diagnostics.get("base", {}).get(
                    "top10_percent_conditional_mass_median"
                ),
                "top10_percent_mass_ttrv": diagnostics.get("ttrv", {}).get(
                    "top10_percent_conditional_mass_median"
                ),
                "top10_percent_mass_ours": diagnostics.get("ours", {}).get(
                    "top10_percent_conditional_mass_median"
                ),
                "ours_ttrv_jsd_normalized": row.get("ours_ttrv_jsd_normalized_median"),
                "image_path": row["image_path"],
                "figures": row.get("figures", {}),
                "copied_original": row.get("copied_original"),
            }
        )
    write_csv(destination / "selected_all_schemes.csv", flat_selected)
    summary_rows = [
        {
            "scheme": item["scheme"],
            "status": item.get("status"),
            "candidate_count": item.get("candidate_count", 0),
            "selected_count": item.get("selected_count", 0),
            "dataset_counts": item.get("dataset_counts", {}),
            "group_counts": item.get("group_counts", {}),
            "rendered": item.get("rendered", False),
            "reason": item.get("reason", ""),
        }
        for item in scheme_summaries
    ]
    write_csv(destination / "summary.csv", summary_rows)
    summary = {
        "records_path": str(records_file),
        "output_dir": str(destination),
        "cpu_only": True,
        "loaded_complete_cases": len(cases),
        "load_failure_count": len(failures),
        "datasets": dict(Counter(case.dataset for case in cases)),
        "available_queries": sorted(available_queries),
        "defined_queries": list(QUERIES),
        "full_three_query_extraction_recommended_max_new_tokens": 16,
        "first_query_only_supported": True,
        "layers_zero_based": list(LAYERS),
        "selection_temperature": 1.0,
        "display_temperature": display_temperature_value,
        "display_temperature_affects_selection": False,
        "corner_fraction": corner_fraction,
        "forbidden_ranking_fields": list(FORBIDDEN_RANK_FIELDS),
        "correctness_blind_ranking": True,
        "candidate_pool_warning": (
            "This script is correctness-blind, but it cannot remove upstream outcome conditioning from the supplied manifest."
        ),
        "schemes": scheme_summaries,
        "selected_all_schemes_jsonl": str((destination / "selected_all_schemes.jsonl").resolve()),
        "selected_all_schemes_csv": str((destination / "selected_all_schemes.csv").resolve()),
        "summary_csv": str((destination / "summary.csv").resolve()),
    }
    write_json(destination / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only correctness-blind multi-scheme screening of saved Obs3 attention."
    )
    parser.add_argument("--records", required=True, help="attention_records.jsonl from attention_viz.py extract")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quota-per-scheme", type=int, default=8)
    parser.add_argument("--display-temperature", type=float, default=0.5)
    parser.add_argument("--corner-fraction", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=0, help="Deterministic sha256(case_id) limit; 0 means all")
    parser.add_argument("--no-render", action="store_true", help="Compute all rankings/configs without PNG/PDF")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(
        records_path=args.records,
        output_dir=args.output_dir,
        quota_per_scheme=args.quota_per_scheme,
        display_temperature_value=args.display_temperature,
        corner_fraction=args.corner_fraction,
        limit=args.limit,
        render=not args.no_render,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
