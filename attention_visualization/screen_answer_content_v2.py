#!/usr/bin/env python3
"""CPU-only, globally fixed screening for v2 answer-content attention.

The script consumes the ``ground_truth`` records/NPZ files emitted by
``answer_content_attention_v2.py``.  It first makes a deterministic SHA256
calibration/evaluation split using case IDs only.  A single layer window and
head is then selected from calibration attention arrays without reading model
predictions or correctness.  Only after that setting is frozen is the
selection manifest reopened to construct Ours-correct/TTRV-wrong tiers.

The default selection query is ``content_answer_state_heads`` because the
requested visualization conditions on the semantic answer tokens themselves.
That state has already observed the token and must not be described as a
same-token causal explanation.  The same frozen setting is always evaluated
with ``content_prediction_query_heads`` as a preceding-state causal audit.

No model, torch, CUDA, training, display temperature, or ground-truth reward is
used here.  Per-head maps are the fused local+thumbnail maps saved by v2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MODELS = ("ours", "ttrv")
ANSWER_QUERY = "content_answer_state_heads"
CAUSAL_QUERY = "content_prediction_query_heads"
QUERY_MODES = (ANSWER_QUERY, CAUSAL_QUERY)
QUERY_SHORT = {ANSWER_QUERY: "answer", CAUSAL_QUERY: "causal"}
QUERY_SEMANTICS = {
    ANSWER_QUERY: (
        "teacher-forced token-conditioned answer state; the content token is "
        "already observed and this is not a same-token causal explanation"
    ),
    CAUSAL_QUERY: (
        "teacher-forced immediately preceding causal state that predicts each "
        "content token; used as a causal-query audit"
    ),
}
DEFAULT_DATASET = "AI2D"
DEFAULT_CALIBRATION_FRACTION = 0.25
DEFAULT_TEMPLATE_SIZE = 32
DEFAULT_WINDOW_RADIUS = 1
EPSILON = 1e-12

YES_NO_VALUES = {"yes", "no", "true", "false"}
GENERIC_OPTION_VALUES = {
    "all of the above",
    "none of the above",
    "all the above",
    "none of these",
    "all of these",
}
OPTION_LINE = re.compile(r"(?:^|\n)\s*([A-F])[\.:]\s*", flags=re.IGNORECASE)
TRAILING_INSTRUCTION_LINE = re.compile(
    r"\n\s*(?:"
    r"please\s+(?:answer|respond)|"
    r"answer\s+(?:directly|with|using)|"
    r"respond\s+(?:directly|with|using)|"
    r"do\s+not\s+|"
    r"only\s+(?:answer|output|respond)"
    r")\b",
    flags=re.IGNORECASE,
)
LABEL_LIKE = re.compile(r"^[A-Za-z](?:\s*[-/,>]\s*[A-Za-z])*[.]?$", flags=re.ASCII)
PURE_NUMBER = re.compile(
    r"^[+-]?(?:\d+(?:[.,]\d+)?|\d+\s*/\s*\d+)(?:\s*[%°])?[.]?$"
)
ALPHA_WORD = re.compile(r"[A-Za-z]{2,}", flags=re.ASCII)
VALID_LABEL = re.compile(r"^[A-F]$")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Cannot serialize {type(value)!r}")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {source}:{line_number}")
            rows.append(value)
    return rows


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
            writer.writeheader()
            for source in rows:
                row: dict[str, Any] = {}
                for field in fields:
                    value = source.get(field, "")
                    if isinstance(value, float) and not math.isfinite(value):
                        value = ""
                    elif isinstance(value, (list, tuple, dict)):
                        value = json.dumps(value, ensure_ascii=False, default=json_default)
                    row[field] = value
                writer.writerow(row)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def case_hash_fraction(case_id: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}\0{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def deterministic_tie_break(case_id: str) -> str:
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().upper()
    return label if VALID_LABEL.fullmatch(label) else "unknown"


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"<\|[^>]+\|>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_lexical_gate(text: str, max_words: int) -> dict[str, Any]:
    normalized = normalize_text(text)
    simplified = normalized.lower().strip(" \t\r\n.,;:!?()[]{}\"'")
    words = ALPHA_WORD.findall(normalized)
    reason = "pass"
    if not normalized:
        reason = "empty_option_text"
    elif simplified in YES_NO_VALUES:
        reason = "yes_no_boolean"
    elif simplified in GENERIC_OPTION_VALUES:
        reason = "generic_option_phrase"
    elif PURE_NUMBER.fullmatch(normalized):
        reason = "pure_number"
    elif LABEL_LIKE.fullmatch(normalized):
        reason = "label_like"
    elif not words:
        reason = "no_lexical_word"
    elif len(words) > max_words:
        reason = "too_many_words"
    return {
        "pass": reason == "pass",
        "reason": reason,
        "normalized": normalized,
        "word_count": len(words),
        "words": words,
    }


def target_lexical_gate(record: "RecordMeta", max_words: int, max_tokens: int) -> dict[str, Any]:
    text_gate = text_lexical_gate(record.target_option_text, max_words=max_words)
    token_count = len(record.content_token_ids)
    reason = str(text_gate["reason"])
    if reason == "pass" and not 1 <= token_count <= max_tokens:
        reason = "content_token_count_out_of_range"
    if not (
        len(record.content_token_ids)
        == len(record.content_token_pieces)
        == len(record.content_target_indices)
    ):
        reason = "content_token_metadata_length_mismatch"
    result = dict(text_gate)
    result.update({"pass": reason == "pass", "reason": reason, "token_count": token_count})
    return result


def parse_prompt_options(raw_prompt: str) -> dict[str, str]:
    body = raw_prompt.split("Options:", 1)[1] if "Options:" in raw_prompt else raw_prompt
    body = body.split("<|im_end|>", 1)[0]
    matches = list(OPTION_LINE.finditer(body))
    options: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        value = body[match.end() : end]
        instruction = TRAILING_INSTRUCTION_LINE.search(value)
        if instruction is not None:
            value = value[: instruction.start()]
        value = normalize_text(value)
        if value:
            options[match.group(1).upper()] = value
    return options


def resolve_npz_path(value: Any, records_path: Path, case_id: str, model: str) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        path = records_path.parent / path
    if path.is_file():
        return path.resolve()
    fallback = records_path.parent / case_id / path.name
    if fallback.is_file():
        return fallback.resolve()
    conventional = records_path.parent / case_id / f"{model}_ground_truth.npz"
    return conventional.resolve()


@dataclass(frozen=True)
class RecordMeta:
    case_id: str
    dataset: str
    question_id: str
    model: str
    target_source: str
    target_label: str
    target_option_text: str
    ground_truth: str
    question: str
    image_path: str
    npz_path: Path
    content_target_indices: tuple[int, ...]
    content_token_ids: tuple[int, ...]
    content_token_pieces: tuple[str, ...]
    raw_prompt: str
    num_layers: int
    num_heads: int
    grid_height: int
    grid_width: int
    tile_rows: int
    tile_columns: int
    has_thumbnail: bool
    tokens_per_tile: int
    head_masses: Mapping[str, Mapping[str, Mapping[str, float]]]

    @property
    def layout_signature(self) -> str:
        return (
            f"r{self.tile_rows}_c{self.tile_columns}_"
            f"thumb{int(self.has_thumbnail)}_g{self.grid_height}x{self.grid_width}"
        )


def trim_record(value: Mapping[str, Any], records_path: Path) -> RecordMeta:
    target = value.get("target_tokens", {})
    attention = value.get("attention", {})
    grid = value.get("grid", {})
    model = str(value.get("model", ""))
    case_id = str(value.get("case_id", ""))
    masses = grid.get("attention_mass", {})
    head_masses = {
        query: masses.get(query, {}) if isinstance(masses, Mapping) else {}
        for query in QUERY_MODES
    }
    return RecordMeta(
        case_id=case_id,
        dataset=str(value.get("dataset", "")),
        question_id=str(value.get("question_id", "")),
        model=model,
        target_source=str(value.get("target_source", "")),
        target_label=normalize_label(value.get("target_label")),
        target_option_text=str(value.get("target_option_text", "")),
        ground_truth=normalize_label(value.get("ground_truth")),
        question=str(value.get("question", "")),
        image_path=str(value.get("image_path", "")),
        npz_path=resolve_npz_path(value.get("npz_path"), records_path, case_id, model),
        content_target_indices=tuple(int(item) for item in target.get("content_target_indices", [])),
        content_token_ids=tuple(int(item) for item in target.get("content_token_ids", [])),
        content_token_pieces=tuple(str(item) for item in target.get("content_token_pieces", [])),
        raw_prompt=str(target.get("raw_prompt", "")),
        num_layers=int(attention.get("num_layers", 0)),
        num_heads=int(attention.get("num_heads", 0)),
        grid_height=int(grid.get("grid_height", 0)),
        grid_width=int(grid.get("grid_width", 0)),
        tile_rows=int(grid.get("tile_rows", 0)),
        tile_columns=int(grid.get("tile_columns", 0)),
        has_thumbnail=bool(grid.get("has_thumbnail", False)),
        tokens_per_tile=int(grid.get("tokens_per_tile", 0)),
        head_masses=head_masses,
    )


def load_trimmed_records(records_path: Path) -> tuple[dict[tuple[str, str], RecordMeta], list[str]]:
    records: dict[tuple[str, str], RecordMeta] = {}
    errors: list[str] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                record = trim_record(value, records_path)
                key = (record.case_id, record.model)
                if key in records:
                    raise ValueError(f"duplicate record {key}")
                records[key] = record
            except Exception as exc:
                errors.append(f"{records_path}:{line_number}: {type(exc).__name__}: {exc}")
    return records, errors


def query_key(query: str, layer: int, head: int) -> str:
    return f"{query}__layer_{layer:02d}_head_{head:02d}"


def mass_key(layer: int, head: int) -> str:
    return f"layer_{layer:02d}_head_{head:02d}"


def validate_record_schema(record: RecordMeta, load_values: bool = False) -> list[str]:
    errors: list[str] = []
    prefix = f"{record.case_id}/{record.model}"
    if record.model not in MODELS:
        errors.append(f"{prefix}: unexpected model")
    if record.target_source != "ground_truth":
        errors.append(f"{prefix}: target_source={record.target_source!r}, expected ground_truth")
    if record.num_layers != 28 or record.num_heads != 12:
        errors.append(f"{prefix}: expected 28x12, got {record.num_layers}x{record.num_heads}")
    if not record.content_token_ids or not (
        len(record.content_token_ids)
        == len(record.content_token_pieces)
        == len(record.content_target_indices)
    ):
        errors.append(f"{prefix}: invalid content token metadata lengths")
    if record.grid_height <= 0 or record.grid_width <= 0:
        errors.append(f"{prefix}: invalid grid {record.grid_height}x{record.grid_width}")
    side = int(round(math.sqrt(max(0, record.tokens_per_tile))))
    if side * side != record.tokens_per_tile:
        errors.append(f"{prefix}: tokens_per_tile={record.tokens_per_tile} is not square")
    if not record.npz_path.is_file():
        errors.append(f"{prefix}: missing NPZ {record.npz_path}")
        return errors
    try:
        with np.load(record.npz_path, allow_pickle=False) as archive:
            available = set(archive.files)
            for query in QUERY_MODES:
                expected = {
                    query_key(query, layer, head)
                    for layer in range(record.num_layers)
                    for head in range(record.num_heads)
                }
                missing = expected - available
                if missing:
                    errors.append(f"{prefix}: {query} missing {len(missing)} keys")
                    continue
                masses = record.head_masses.get(query, {})
                missing_mass = [
                    mass_key(layer, head)
                    for layer in range(record.num_layers)
                    for head in range(record.num_heads)
                    if mass_key(layer, head) not in masses
                ]
                if missing_mass:
                    errors.append(f"{prefix}: {query} missing {len(missing_mass)} mass entries")
                if load_values:
                    for key in sorted(expected):
                        array = np.asarray(archive[key])
                        if array.shape != (record.grid_height, record.grid_width):
                            errors.append(f"{prefix}: {key} shape={array.shape}")
                            break
                        if not np.all(np.isfinite(array)) or float(array.min()) < -1e-8:
                            errors.append(f"{prefix}: {key} contains invalid values")
                            break
    except Exception as exc:
        errors.append(f"{prefix}: {type(exc).__name__}: {exc}")
    return errors


def assert_pair_consistency(ours: RecordMeta, ttrv: RecordMeta) -> None:
    fields = (
        "dataset",
        "question_id",
        "target_source",
        "target_label",
        "target_option_text",
        "ground_truth",
        "content_target_indices",
        "content_token_ids",
        "content_token_pieces",
        "grid_height",
        "grid_width",
        "tile_rows",
        "tile_columns",
        "has_thumbnail",
        "tokens_per_tile",
        "num_layers",
        "num_heads",
    )
    mismatches = [field for field in fields if getattr(ours, field) != getattr(ttrv, field)]
    if mismatches:
        raise ValueError(f"{ours.case_id}: Ours/TTRV mismatch in {mismatches}")


@dataclass
class FeatureBlock:
    valid_mask: np.ndarray
    mass: np.ndarray
    mass_rank: np.ndarray
    outer1_lift: np.ndarray
    outer10_lift: np.ndarray
    tile_boundary_lift: np.ndarray
    leading_edge_lift: np.ndarray
    boundary_lift: np.ndarray
    boundary_quality: np.ndarray
    layer_stability: np.ndarray
    local_fraction: np.ndarray
    thumbnail_fraction: np.ndarray
    entropy_norm: np.ndarray
    top10_mass: np.ndarray
    template_jsd: np.ndarray
    template_tv: np.ndarray
    template_corr: np.ndarray
    template_source: str = "pending"


def normalized_jsd_batch(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"JSD shape mismatch: {p.shape} != {q.shape}")
    middle = 0.5 * (p + q)
    left_term = np.where(
        p > 0.0,
        p * np.log(np.maximum(p, EPSILON) / np.maximum(middle, EPSILON)),
        0.0,
    )
    right_term = np.where(
        q > 0.0,
        q * np.log(np.maximum(q, EPSILON) / np.maximum(middle, EPSILON)),
        0.0,
    )
    axes = tuple(range(p.ndim - 2, p.ndim))
    result = 0.5 * (left_term.sum(axis=axes) + right_term.sum(axis=axes)) / math.log(2.0)
    return np.clip(result, 0.0, 1.0)


def total_variation_batch(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    axes = tuple(range(left.ndim - 2, left.ndim))
    return np.clip(
        0.5 * np.abs(np.asarray(left) - np.asarray(right)).sum(axis=axes),
        0.0,
        1.0,
    )


def pearson_batch(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    p = p.reshape(*p.shape[:-2], -1)
    q = q.reshape(*q.shape[:-2], -1)
    p_centered = p - p.mean(axis=-1, keepdims=True)
    q_centered = q - q.mean(axis=-1, keepdims=True)
    numerator = (p_centered * q_centered).sum(axis=-1)
    denominator = np.sqrt(
        (p_centered**2).sum(axis=-1) * (q_centered**2).sum(axis=-1)
    )
    return np.where(denominator > EPSILON, numerator / denominator, 0.0)


def resize_batch_bilinear(values: np.ndarray, output_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    height, width = array.shape[-2:]
    if height == output_size and width == output_size:
        return array.copy()
    ys = np.linspace(0.0, max(0, height - 1), output_size)
    xs = np.linspace(0.0, max(0, width - 1), output_size)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    wy = (ys - y0).reshape((1,) * (array.ndim - 2) + (output_size, 1))
    wx = (xs - x0).reshape((1,) * (array.ndim - 2) + (1, output_size))
    horizontal = array[..., :, x0] * (1.0 - wx) + array[..., :, x1] * wx
    resized = horizontal[..., y0, :] * (1.0 - wy) + horizontal[..., y1, :] * wy
    totals = resized.sum(axis=(-2, -1), keepdims=True)
    return np.divide(
        resized,
        totals,
        out=np.zeros_like(resized),
        where=totals > EPSILON,
    )


def normalize_maps(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("attention maps contain NaN/inf")
    if float(values.min()) < -1e-8:
        raise ValueError(f"attention maps contain negative value {values.min():.4g}")
    values = np.maximum(values, 0.0)
    mass = values.sum(axis=(-2, -1))
    probability = np.divide(
        values,
        mass[..., None, None],
        out=np.zeros_like(values),
        where=mass[..., None, None] > EPSILON,
    )
    return probability, mass


def density_lift(
    probability: np.ndarray,
    mask: np.ndarray,
    comparison: np.ndarray | None = None,
) -> np.ndarray:
    if mask.shape != probability.shape[-2:]:
        raise ValueError(f"mask shape {mask.shape} != map shape {probability.shape[-2:]}")
    if comparison is None:
        comparison = ~mask
    if not np.any(mask) or not np.any(comparison):
        return np.ones(probability.shape[:-2], dtype=np.float64)
    selected = probability[..., mask].mean(axis=-1)
    reference = probability[..., comparison].mean(axis=-1)
    return selected / np.maximum(reference, EPSILON)


def boundary_masks(record: RecordMeta) -> dict[str, np.ndarray]:
    height, width = record.grid_height, record.grid_width
    rows, columns = np.indices((height, width))
    outer1 = (
        (rows == 0)
        | (rows == height - 1)
        | (columns == 0)
        | (columns == width - 1)
    )
    band = max(1, int(math.ceil(0.10 * min(height, width))))
    outer10 = (
        (rows < band)
        | (rows >= height - band)
        | (columns < band)
        | (columns >= width - band)
    )
    side = int(round(math.sqrt(record.tokens_per_tile)))
    if side * side != record.tokens_per_tile:
        raise ValueError(f"tokens_per_tile={record.tokens_per_tile} is not square")
    tile_all = (
        (rows % side == 0)
        | (rows % side == side - 1)
        | (columns % side == 0)
        | (columns % side == side - 1)
    )
    tile_internal = tile_all & ~outer1
    leading_width = min(2, max(1, min(height, width) // 4))
    leading = (rows < leading_width) | (columns < leading_width)
    trailing = (rows >= height - leading_width) | (columns >= width - leading_width)
    trailing &= ~leading
    return {
        "outer1": outer1,
        "outer10": outer10,
        "tile_internal": tile_internal,
        "leading": leading,
        "trailing": trailing,
    }


def rank_percentiles(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = (np.arange(flat.size, dtype=np.float64) + 1.0) / flat.size
    return ranks.reshape(values.shape)


def load_query_cube(archive: Any, record: RecordMeta, query: str) -> np.ndarray:
    per_head: list[np.ndarray] = []
    for head in range(record.num_heads):
        per_layer = [
            np.asarray(archive[query_key(query, layer, head)], dtype=np.float64)
            for layer in range(record.num_layers)
        ]
        cube = np.stack(per_layer, axis=0)
        if cube.shape != (record.num_layers, record.grid_height, record.grid_width):
            raise ValueError(
                f"{record.case_id}/{record.model}/{query}: unexpected shape {cube.shape}"
            )
        per_head.append(cube)
    result = np.stack(per_head, axis=0)
    if not np.all(np.isfinite(result)) or float(result.min()) < -1e-8:
        raise ValueError(f"{record.case_id}/{record.model}/{query}: invalid values")
    return np.maximum(result, 0.0)


def component_mass_arrays(
    record: RecordMeta,
    query: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = np.empty((record.num_heads, record.num_layers), dtype=np.float64)
    local = np.empty_like(total)
    thumbnail = np.empty_like(total)
    source = record.head_masses.get(query, {})
    for head in range(record.num_heads):
        for layer in range(record.num_layers):
            item = source.get(mass_key(layer, head), {})
            total[head, layer] = float(item.get("total_image_mass", float("nan")))
            local[head, layer] = float(item.get("local_mass", float("nan")))
            thumbnail[head, layer] = float(item.get("thumbnail_mass", float("nan")))
    if not (
        np.all(np.isfinite(total))
        and np.all(np.isfinite(local))
        and np.all(np.isfinite(thumbnail))
    ):
        raise ValueError(
            f"{record.case_id}/{record.model}/{query}: invalid component mass metadata"
        )
    if not np.allclose(local + thumbnail, total, rtol=5e-4, atol=5e-8):
        raise ValueError(
            f"{record.case_id}/{record.model}/{query}: local+thumbnail mass drift"
        )
    return total, local, thumbnail


def compute_feature_block(
    cube: np.ndarray,
    record: RecordMeta,
    query: str,
    radius: int,
) -> tuple[FeatureBlock, np.ndarray]:
    centers = tuple(range(radius, record.num_layers - radius))
    aggregated = np.stack(
        [
            cube[:, center - radius : center + radius + 1].mean(axis=1)
            for center in centers
        ],
        axis=1,
    )
    probability, mass = normalize_maps(aggregated)
    layer_probability, layer_mass = normalize_maps(cube)
    window_valid = mass > EPSILON
    layer_window_valid = np.stack(
        [
            np.all(
                layer_mass[:, center - radius : center + radius + 1] > EPSILON,
                axis=1,
            )
            for center in centers
        ],
        axis=1,
    )
    valid_mask = window_valid & layer_window_valid
    expected_mass = np.stack(
        [
            layer_mass[:, center - radius : center + radius + 1].mean(axis=1)
            for center in centers
        ],
        axis=1,
    )
    if not np.allclose(mass, expected_mass, rtol=5e-5, atol=5e-8):
        raise AssertionError("window raw-map mass does not equal mean layer mass")

    masks = boundary_masks(record)
    outer1 = density_lift(probability, masks["outer1"])
    outer10 = density_lift(probability, masks["outer10"])
    tile = (
        density_lift(probability, masks["tile_internal"])
        if np.any(masks["tile_internal"])
        else np.ones_like(outer1)
    )
    leading = density_lift(probability, masks["leading"], masks["trailing"])
    composite = np.maximum.reduce([outer1, outer10, tile, leading])
    boundary_quality = np.exp(
        -np.maximum(0.0, np.log(np.maximum(composite, EPSILON)))
    )

    stability_parts = []
    for center in centers:
        adjacent = [
            normalized_jsd_batch(
                layer_probability[:, left],
                layer_probability[:, right],
            )
            for left, right in zip(
                range(center - radius, center + radius),
                range(center - radius + 1, center + radius + 1),
            )
        ]
        stability_parts.append(
            1.0 - np.median(np.stack(adjacent, axis=0), axis=0)
        )
    stability = np.where(
        valid_mask,
        np.clip(np.stack(stability_parts, axis=1), 0.0, 1.0),
        0.0,
    )

    total_meta, local_meta, thumbnail_meta = component_mass_arrays(record, query)
    total_window = np.stack(
        [
            total_meta[:, center - radius : center + radius + 1].mean(axis=1)
            for center in centers
        ],
        axis=1,
    )
    local_window = np.stack(
        [
            local_meta[:, center - radius : center + radius + 1].mean(axis=1)
            for center in centers
        ],
        axis=1,
    )
    thumbnail_window = np.stack(
        [
            thumbnail_meta[:, center - radius : center + radius + 1].mean(axis=1)
            for center in centers
        ],
        axis=1,
    )
    if not np.allclose(total_window, mass, rtol=5e-4, atol=5e-8):
        difference = float(np.max(np.abs(total_window - mass)))
        raise ValueError(
            f"{record.case_id}/{record.model}/{query}: "
            f"NPZ/JSON mass drift max={difference:.4g}"
        )
    local_fraction = local_window / np.maximum(total_window, EPSILON)
    thumbnail_fraction = thumbnail_window / np.maximum(total_window, EPSILON)

    positive = np.maximum(probability, EPSILON)
    entropy = -(probability * np.log(positive)).sum(axis=(-2, -1))
    entropy /= math.log(record.grid_height * record.grid_width)
    flat = probability.reshape(*probability.shape[:-2], -1)
    top_count = max(1, int(math.ceil(0.10 * flat.shape[-1])))
    top10 = np.partition(
        flat,
        flat.shape[-1] - top_count,
        axis=-1,
    )[..., -top_count:].sum(axis=-1)
    pending = np.full(mass.shape, np.nan, dtype=np.float64)
    block = FeatureBlock(
        valid_mask=valid_mask,
        mass=mass,
        mass_rank=rank_percentiles(mass),
        outer1_lift=outer1,
        outer10_lift=outer10,
        tile_boundary_lift=tile,
        leading_edge_lift=leading,
        boundary_lift=composite,
        boundary_quality=boundary_quality,
        layer_stability=stability,
        local_fraction=local_fraction,
        thumbnail_fraction=thumbnail_fraction,
        entropy_norm=np.clip(entropy, 0.0, 1.0),
        top10_mass=np.clip(top10, 0.0, 1.0),
        template_jsd=pending.copy(),
        template_tv=pending.copy(),
        template_corr=pending.copy(),
    )
    return block, probability


def geometric_quality(block: FeatureBlock) -> np.ndarray:
    components = np.stack(
        [
            np.clip(block.mass_rank, EPSILON, 1.0),
            np.clip(block.boundary_quality, EPSILON, 1.0),
            np.clip(block.template_jsd, EPSILON, 1.0),
            np.clip(block.layer_stability, EPSILON, 1.0),
        ],
        axis=0,
    )
    quality = np.exp(np.log(components).mean(axis=0))
    return np.where(block.valid_mask, quality, 0.0)


def normalize_template(values: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = array.sum(axis=(-2, -1), keepdims=True)
    return np.divide(
        array,
        total,
        out=np.zeros_like(array),
        where=total > EPSILON,
    )


def robust_setting_score(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    return np.maximum(0.0, median - 0.25 * (q75 - q25))


def ecdf(values: np.ndarray, target: float) -> float:
    population = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if population.size == 0 or not math.isfinite(target):
        return float("nan")
    return float(np.searchsorted(population, target, side="right") / population.size)


def selected_probability(
    record: RecordMeta,
    query: str,
    head: int,
    center: int,
    radius: int,
) -> np.ndarray:
    with np.load(record.npz_path, allow_pickle=False) as archive:
        raw = np.mean(
            [
                np.asarray(
                    archive[query_key(query, layer, head)],
                    dtype=np.float64,
                )
                for layer in range(center - radius, center + radius + 1)
            ],
            axis=0,
        )
    probability, _ = normalize_maps(raw)
    return probability


def load_manifest_static(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Load only correctness-independent manifest fields used before freezing."""

    rows: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                case_id = str(value["case_id"])
                if case_id in rows:
                    raise ValueError(f"duplicate case_id={case_id}")
                rows[case_id] = {
                    "case_id": case_id,
                    "dataset": str(value.get("dataset", "")),
                    "question_id": str(value.get("question_id", "")),
                    "category": str(value.get("category", "")),
                    "labels": str(value.get("labels", "")),
                    "question": str(value.get("question", "")),
                    "image_path": str(value.get("image_path", "")),
                }
            except Exception as exc:
                errors.append(f"{path}:{line_number}: {type(exc).__name__}: {exc}")
    return rows, errors


def load_manifest_outcomes(path: Path) -> dict[str, dict[str, Any]]:
    """Second-pass loader; called only after the global setting is frozen."""

    rows: dict[str, dict[str, Any]] = {}
    for value in read_jsonl(path):
        case_id = str(value["case_id"])
        retained = value.get("retained_predictions", {})
        rows[case_id] = {
            "ground_truth": value.get("ground_truth"),
            "ours_prediction_raw": retained.get("ours", {}).get("parsed_answer")
            if isinstance(retained, Mapping)
            else None,
            "ttrv_prediction_raw": retained.get("ttrv", {}).get("parsed_answer")
            if isinstance(retained, Mapping)
            else None,
        }
    return rows


def manifest_prediction(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return "absent"
    label = normalize_label(value)
    return label if label != "unknown" else "unknown"


def outcome_tier(
    static: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> tuple[str, str, str, str, bool, bool]:
    ground_truth = normalize_label(outcome.get("ground_truth"))
    ours = manifest_prediction(outcome.get("ours_prediction_raw"))
    ttrv = manifest_prediction(outcome.get("ttrv_prediction_raw"))
    allowed = {character for character in str(static.get("labels", "")).upper() if VALID_LABEL.fullmatch(character)}
    if not allowed and ground_truth != "unknown":
        allowed = {ground_truth}
    ours_correct = ours == ground_truth and ground_truth != "unknown"
    ttrv_correct = ttrv == ground_truth and ground_truth != "unknown"
    if not ours_correct:
        tier = "not_contrast"
    elif ttrv in allowed and ttrv != ground_truth:
        tier = "valid_wrong"
    elif ttrv in {"unknown", "absent"}:
        tier = "unknown_wrong"
    else:
        tier = "not_contrast"
    return (
        tier,
        ground_truth,
        ours,
        ttrv,
        ours_correct,
        ttrv_correct,
    )


def build_lexical_cases(
    records: Mapping[tuple[str, str], RecordMeta],
    manifest_static: Mapping[str, Mapping[str, Any]],
    dataset: str,
    max_words: int,
    max_tokens: int,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, list[str]],
]:
    lexical: dict[str, dict[str, Any]] = {}
    exclusions: dict[str, str] = {}
    pair_errors: dict[str, list[str]] = defaultdict(list)
    case_ids = sorted(
        case_id
        for case_id, static in manifest_static.items()
        if str(static.get("dataset")) == dataset
    )
    for case_id in case_ids:
        ours = records.get((case_id, "ours"))
        ttrv = records.get((case_id, "ttrv"))
        source = ours or ttrv
        if source is None:
            exclusions[case_id] = "missing_both_attention_records"
            continue
        gate = target_lexical_gate(source, max_words=max_words, max_tokens=max_tokens)
        if not gate["pass"]:
            exclusions[case_id] = str(gate["reason"])
            continue
        if ours is None or ttrv is None:
            exclusions[case_id] = "missing_model_attention_record"
            continue
        try:
            assert_pair_consistency(ours, ttrv)
        except Exception as exc:
            pair_errors[case_id].append(str(exc))
            exclusions[case_id] = "pair_schema_mismatch"
            continue
        lexical[case_id] = {
            "record": ours,
            "target_gate": gate,
            "options": parse_prompt_options(ours.raw_prompt),
        }
    return lexical, exclusions, dict(pair_errors)


def validate_lexical_npzs(
    lexical: Mapping[str, Mapping[str, Any]],
    records: Mapping[tuple[str, str], RecordMeta],
) -> tuple[list[str], dict[str, list[str]]]:
    valid: list[str] = []
    failures: dict[str, list[str]] = {}
    for case_id in sorted(lexical):
        errors: list[str] = []
        for model in MODELS:
            errors.extend(validate_record_schema(records[(case_id, model)], load_values=False))
        if errors:
            failures[case_id] = errors
        else:
            valid.append(case_id)
    return valid, failures


def add_template_sum(
    sums: dict[tuple[str, str, str], np.ndarray],
    counts: Counter[tuple[str, str, str]],
    query: str,
    model: str,
    signature: str,
    probability: np.ndarray,
) -> None:
    key = (query, model, signature)
    if key in sums:
        sums[key] += probability
    else:
        sums[key] = probability.copy()
    counts[key] += 1
    global_key = (query, model, "__global__")
    if global_key in sums:
        sums[global_key] += probability
    else:
        sums[global_key] = probability.copy()
    counts[global_key] += 1


def resolve_template(
    sums: Mapping[tuple[str, str, str], np.ndarray],
    counts: Mapping[tuple[str, str, str], int],
    query: str,
    model: str,
    signature: str,
    probability: np.ndarray,
    calibration_member: bool,
    min_group: int,
) -> tuple[np.ndarray, str]:
    group_key = (query, model, signature)
    group_remaining = int(counts.get(group_key, 0)) - int(calibration_member)
    if group_remaining >= min_group:
        total = np.asarray(sums[group_key], dtype=np.float64)
        if calibration_member:
            total = total - probability
        return normalize_template(total), f"layout:{signature}"
    global_key = (query, model, "__global__")
    global_remaining = int(counts.get(global_key, 0)) - int(calibration_member)
    if global_remaining < 1:
        raise ValueError(f"No non-self template for {query}/{model}/{signature}")
    total = np.asarray(sums[global_key], dtype=np.float64)
    if calibration_member:
        total = total - probability
    return normalize_template(total), "global_fallback"


def extract_all_features(
    case_ids: Sequence[str],
    calibration_ids: set[str],
    records: Mapping[tuple[str, str], RecordMeta],
    radius: int,
    template_size: int,
    min_template_group: int,
) -> tuple[
    dict[str, dict[tuple[str, str], FeatureBlock]],
    dict[tuple[str, str, str], np.ndarray],
    Counter[tuple[str, str, str]],
]:
    features: dict[str, dict[tuple[str, str], FeatureBlock]] = {
        query: {} for query in QUERY_MODES
    }
    template_sums: dict[tuple[str, str, str], np.ndarray] = {}
    template_counts: Counter[tuple[str, str, str]] = Counter()
    total_records = len(case_ids) * len(MODELS)
    processed = 0
    print(
        f"[screen] pass1 features/templates records={total_records} "
        f"calibration_cases={len(calibration_ids)}",
        flush=True,
    )
    for case_id in case_ids:
        for model in MODELS:
            record = records[(case_id, model)]
            with np.load(record.npz_path, allow_pickle=False) as archive:
                for query in QUERY_MODES:
                    cube = load_query_cube(archive, record, query)
                    block, probability = compute_feature_block(cube, record, query, radius)
                    features[query][(case_id, model)] = block
                    if case_id in calibration_ids:
                        resized = resize_batch_bilinear(probability, template_size)
                        add_template_sum(
                            template_sums,
                            template_counts,
                            query,
                            model,
                            record.layout_signature,
                            resized,
                        )
            processed += 1
            if processed % 20 == 0 or processed == total_records:
                print(f"[screen] pass1 progress={processed}/{total_records}", flush=True)

    print("[screen] pass2 leave-one-out/frozen template metrics", flush=True)
    processed = 0
    for case_id in case_ids:
        for model in MODELS:
            record = records[(case_id, model)]
            with np.load(record.npz_path, allow_pickle=False) as archive:
                for query in QUERY_MODES:
                    cube = load_query_cube(archive, record, query)
                    _, probability = compute_feature_block(cube, record, query, radius)
                    resized = resize_batch_bilinear(probability, template_size)
                    template, source = resolve_template(
                        template_sums,
                        template_counts,
                        query,
                        model,
                        record.layout_signature,
                        resized,
                        calibration_member=case_id in calibration_ids,
                        min_group=min_template_group,
                    )
                    block = features[query][(case_id, model)]
                    template_valid = template.sum(axis=(-2, -1)) > EPSILON
                    block.valid_mask &= template_valid
                    block.template_jsd = np.where(
                        block.valid_mask,
                        normalized_jsd_batch(resized, template),
                        0.0,
                    )
                    block.template_tv = np.where(
                        block.valid_mask,
                        total_variation_batch(resized, template),
                        0.0,
                    )
                    block.template_corr = np.where(
                        block.valid_mask,
                        pearson_batch(resized, template),
                        0.0,
                    )
                    block.template_source = source
            processed += 1
            if processed % 20 == 0 or processed == total_records:
                print(f"[screen] pass2 progress={processed}/{total_records}", flush=True)
    return features, template_sums, template_counts


def setting_statistics(
    query: str,
    calibration_ids: Sequence[str],
    features: Mapping[str, Mapping[tuple[str, str], FeatureBlock]],
    folds: int,
    top_k: int,
    split_salt: str,
) -> dict[str, Any]:
    quality_by_model: dict[str, np.ndarray] = {}
    for model in MODELS:
        quality_by_model[model] = np.stack(
            [
                geometric_quality(features[query][(case_id, model)])
                for case_id in calibration_ids
            ],
            axis=0,
        )
    robust_by_model = {
        model: robust_setting_score(values) for model, values in quality_by_model.items()
    }
    global_score = np.minimum(robust_by_model["ours"], robust_by_model["ttrv"])

    flat_count = global_score.size
    fold_ranks: list[np.ndarray] = []
    effective_folds = max(2, min(folds, len(calibration_ids)))
    assignments = {
        case_id: int(
            hashlib.sha256(f"{split_salt}\0cv\0{case_id}".encode("utf-8")).hexdigest(),
            16,
        )
        % effective_folds
        for case_id in calibration_ids
    }
    for fold in range(effective_folds):
        keep = [
            index
            for index, case_id in enumerate(calibration_ids)
            if assignments[case_id] != fold
        ]
        if not keep:
            continue
        fold_model = {
            model: robust_setting_score(quality_by_model[model][keep])
            for model in MODELS
        }
        fold_score = np.minimum(fold_model["ours"], fold_model["ttrv"]).reshape(-1)
        order = np.argsort(-fold_score, kind="stable")
        ranks = np.empty(flat_count, dtype=np.int64)
        ranks[order] = np.arange(1, flat_count + 1)
        fold_ranks.append(ranks.reshape(global_score.shape))
    stacked_ranks = np.stack(fold_ranks, axis=0)
    return {
        "quality_by_model": quality_by_model,
        "robust_by_model": robust_by_model,
        "global_score": global_score,
        "cv_median_rank": np.median(stacked_ranks, axis=0),
        "cv_rank_iqr": np.quantile(stacked_ranks, 0.75, axis=0)
        - np.quantile(stacked_ranks, 0.25, axis=0),
        "cv_topk_rate": np.mean(stacked_ranks <= top_k, axis=0),
        "fold_assignments": assignments,
        "effective_folds": len(fold_ranks),
    }


def median_feature(
    query: str,
    model: str,
    name: str,
    calibration_ids: Sequence[str],
    features: Mapping[str, Mapping[tuple[str, str], FeatureBlock]],
) -> np.ndarray:
    return np.median(
        np.stack(
            [getattr(features[query][(case_id, model)], name) for case_id in calibration_ids],
            axis=0,
        ),
        axis=0,
    )


def choose_global_setting(
    stats_by_query: Mapping[str, Mapping[str, Any]],
    selection_query: str,
    min_topk_rate: float,
) -> tuple[int, int, bool, str]:
    stats = stats_by_query[selection_query]
    score = np.asarray(stats["global_score"])
    topk = np.asarray(stats["cv_topk_rate"])
    eligible = topk >= min_topk_rate
    fallback = False
    reason = "cv_stable"
    if not np.any(eligible):
        eligible = np.ones_like(score, dtype=bool)
        fallback = True
        reason = "no_setting_met_cv_topk_rate; selected_full_calibration_best"
    candidates = []
    for head, center_index in zip(*np.nonzero(eligible)):
        candidates.append(
            (
                -float(score[head, center_index]),
                -float(topk[head, center_index]),
                int(head),
                int(center_index),
            )
        )
    candidates.sort()
    _, _, head, center_index = candidates[0]
    return head, center_index, fallback, reason


SETTING_DIAGNOSTICS = (
    "mass_rank",
    "outer1_lift",
    "outer10_lift",
    "tile_boundary_lift",
    "leading_edge_lift",
    "boundary_lift",
    "boundary_quality",
    "template_jsd",
    "template_tv",
    "template_corr",
    "layer_stability",
    "local_fraction",
    "thumbnail_fraction",
)


def make_setting_rows(
    stats_by_query: Mapping[str, Mapping[str, Any]],
    features: Mapping[str, Mapping[tuple[str, str], FeatureBlock]],
    calibration_ids: Sequence[str],
    selection_query: str,
    radius: int,
    min_topk_rate: float,
    selected_head: int,
    selected_center_index: int,
) -> list[dict[str, Any]]:
    ranks_by_query: dict[str, np.ndarray] = {}
    for query in QUERY_MODES:
        score = np.asarray(stats_by_query[query]["global_score"])
        order = np.argsort(-score.reshape(-1), kind="stable")
        ranks = np.empty(score.size, dtype=np.int64)
        ranks[order] = np.arange(1, score.size + 1)
        ranks_by_query[query] = ranks.reshape(score.shape)

    medians: dict[tuple[str, str, str], np.ndarray] = {}
    for query in QUERY_MODES:
        for model in MODELS:
            for metric in SETTING_DIAGNOSTICS:
                medians[(query, model, metric)] = median_feature(
                    query,
                    model,
                    metric,
                    calibration_ids,
                    features,
                )

    rows: list[dict[str, Any]] = []
    centers = tuple(range(radius, 28 - radius))
    for head in range(12):
        for center_index, center in enumerate(centers):
            row: dict[str, Any] = {
                "selection_query_mode": selection_query,
                "head_index_zero_based": head,
                "layer_center_zero_based": center,
                "layer_window_zero_based": list(
                    range(center - radius, center + radius + 1)
                ),
                "calibration_case_count": len(calibration_ids),
                "selected": head == selected_head
                and center_index == selected_center_index,
            }
            for query in QUERY_MODES:
                short = QUERY_SHORT[query]
                stats = stats_by_query[query]
                row.update(
                    {
                        f"{short}_global_score": float(
                            stats["global_score"][head, center_index]
                        ),
                        f"{short}_ours_robust_score": float(
                            stats["robust_by_model"]["ours"][head, center_index]
                        ),
                        f"{short}_ttrv_robust_score": float(
                            stats["robust_by_model"]["ttrv"][head, center_index]
                        ),
                        f"{short}_global_rank": int(
                            ranks_by_query[query][head, center_index]
                        ),
                        f"{short}_cv_median_rank": float(
                            stats["cv_median_rank"][head, center_index]
                        ),
                        f"{short}_cv_rank_iqr": float(
                            stats["cv_rank_iqr"][head, center_index]
                        ),
                        f"{short}_cv_topk_rate": float(
                            stats["cv_topk_rate"][head, center_index]
                        ),
                    }
                )
                for model in MODELS:
                    for metric in SETTING_DIAGNOSTICS:
                        row[f"{short}_{model}_median_{metric}"] = float(
                            medians[(query, model, metric)][head, center_index]
                        )
            selection_short = QUERY_SHORT[selection_query]
            row["selection_score"] = row[f"{selection_short}_global_score"]
            row["selection_cv_topk_rate"] = row[
                f"{selection_short}_cv_topk_rate"
            ]
            row["eligible"] = (
                float(row["selection_cv_topk_rate"]) >= min_topk_rate
            )
            rows.append(row)
    rows.sort(
        key=lambda row: (
            -float(row["selection_score"]),
            -float(row["selection_cv_topk_rate"]),
            int(row["head_index_zero_based"]),
            int(row["layer_center_zero_based"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["selection_rank"] = rank
    return rows


def setting_fields() -> list[str]:
    fields = [
        "selection_rank",
        "selection_query_mode",
        "head_index_zero_based",
        "layer_center_zero_based",
        "layer_window_zero_based",
        "calibration_case_count",
        "selection_score",
        "selection_cv_topk_rate",
        "eligible",
        "selected",
    ]
    for query in QUERY_MODES:
        short = QUERY_SHORT[query]
        fields.extend(
            [
                f"{short}_global_score",
                f"{short}_ours_robust_score",
                f"{short}_ttrv_robust_score",
                f"{short}_global_rank",
                f"{short}_cv_median_rank",
                f"{short}_cv_rank_iqr",
                f"{short}_cv_topk_rate",
            ]
        )
        for model in MODELS:
            fields.extend(
                f"{short}_{model}_median_{metric}"
                for metric in SETTING_DIAGNOSTICS
            )
    return fields


CASE_METRICS = (
    "mass",
    "mass_rank",
    "outer1_lift",
    "outer10_lift",
    "tile_boundary_lift",
    "leading_edge_lift",
    "boundary_lift",
    "boundary_quality",
    "local_fraction",
    "thumbnail_fraction",
    "template_jsd",
    "template_tv",
    "template_corr",
    "layer_stability",
    "entropy_norm",
    "top10_mass",
)


def feature_scalar(
    block: FeatureBlock,
    name: str,
    head: int,
    center_index: int,
) -> float:
    return float(getattr(block, name)[head, center_index])


def calibration_component_values(
    query: str,
    model: str,
    metric: str,
    calibration_ids: Sequence[str],
    features: Mapping[str, Mapping[tuple[str, str], FeatureBlock]],
    head: int,
    center_index: int,
) -> np.ndarray:
    return np.asarray(
        [
            feature_scalar(
                features[query][(case_id, model)],
                metric,
                head,
                center_index,
            )
            for case_id in calibration_ids
        ],
        dtype=np.float64,
    )


def case_quality(
    query: str,
    model: str,
    block: FeatureBlock,
    calibration_ids: Sequence[str],
    features: Mapping[str, Mapping[tuple[str, str], FeatureBlock]],
    head: int,
    center_index: int,
) -> tuple[float, dict[str, float], bool]:
    source_metrics = (
        "mass_rank",
        "boundary_quality",
        "template_jsd",
        "layer_stability",
    )
    ranks: dict[str, float] = {}
    for metric in source_metrics:
        population = calibration_component_values(
            query,
            model,
            metric,
            calibration_ids,
            features,
            head,
            center_index,
        )
        target = feature_scalar(block, metric, head, center_index)
        ranks[metric] = ecdf(population, target)
    quality = float(
        math.exp(
            np.log(
                np.clip(
                    np.asarray(list(ranks.values()), dtype=np.float64),
                    EPSILON,
                    1.0,
                )
            ).mean()
        )
    )
    gate = all(value >= 0.25 for value in ranks.values())
    return quality, ranks, gate


def blank_case_row(
    static: Mapping[str, Any],
    tier: str,
    ground_truth: str,
    ours_prediction: str,
    ttrv_prediction: str,
    ours_correct: bool,
    ttrv_correct: bool,
    selection_query: str,
    selected_head: int,
    selected_center: int,
    radius: int,
) -> dict[str, Any]:
    return {
        "rank": "",
        "tier_rank": "",
        "case_id": static.get("case_id", ""),
        "dataset": static.get("dataset", ""),
        "question_id": static.get("question_id", ""),
        "category": static.get("category", ""),
        "question": static.get("question", ""),
        "image_path": static.get("image_path", ""),
        "target_source": "ground_truth",
        "target_label": ground_truth,
        "target_option_text": "",
        "content_token_count": "",
        "content_token_ids": [],
        "content_token_pieces": [],
        "target_lexical_word_count": "",
        "target_lexical_gate": False,
        "target_lexical_reason": "missing_target_metadata",
        "ttrv_wrong_option_text": "",
        "ttrv_wrong_option_lexical_gate": False,
        "ttrv_wrong_option_lexical_reason": "unavailable",
        "allowed_labels": str(static.get("labels", "")),
        "ground_truth": ground_truth,
        "ours_prediction": ours_prediction,
        "ttrv_prediction": ttrv_prediction,
        "ours_correct": ours_correct,
        "ttrv_correct": ttrv_correct,
        "outcome_tier": tier,
        "analysis_tier": "pending",
        "split": "pending",
        "split_hash_fraction": "",
        "selection_query_mode": selection_query,
        "selection_query_semantics": QUERY_SEMANTICS[selection_query],
        "selected_head_index_zero_based": selected_head,
        "selected_layer_center_zero_based": selected_center,
        "selected_layer_window_zero_based": list(
            range(selected_center - radius, selected_center + radius + 1)
        ),
        "map_component": "fused_local_plus_thumbnail",
        "teacher_forced_target_note": (
            "Both maps use the same GT teacher-forced option text; this is not "
            "the TTRV generated wrong-token map."
        ),
        "grid_height": "",
        "grid_width": "",
        "tile_rows": "",
        "tile_columns": "",
        "has_thumbnail": "",
        "layout_signature": "",
        "answer_template_source_ours": "",
        "answer_template_source_ttrv": "",
        "causal_template_source_ours": "",
        "causal_template_source_ttrv": "",
        "selection_ours_quality": "",
        "selection_ttrv_quality": "",
        "selection_quality_gain": "",
        "selection_ours_component_percentiles": {},
        "selection_ttrv_component_percentiles": {},
        "quality_gate": False,
        "eligible": False,
        "rank_score": "",
        "exclusion_reason": "",
        "deterministic_tie_break": deterministic_tie_break(
            str(static.get("case_id", ""))
        ),
        "npz_ours": "",
        "npz_ttrv": "",
    }


def populate_case_metrics(
    row: dict[str, Any],
    case_id: str,
    records: Mapping[tuple[str, str], RecordMeta],
    features: Mapping[str, Mapping[tuple[str, str], FeatureBlock]],
    calibration_ids: Sequence[str],
    selected_head: int,
    selected_center_index: int,
    selected_center: int,
    radius: int,
    selection_query: str,
) -> None:
    direct_maps: dict[tuple[str, str], np.ndarray] = {}
    query_quality: dict[tuple[str, str], float] = {}
    query_gates: dict[tuple[str, str], bool] = {}
    query_ranks: dict[tuple[str, str], dict[str, float]] = {}
    for query in QUERY_MODES:
        short = QUERY_SHORT[query]
        for model in MODELS:
            block = features[query][(case_id, model)]
            for metric in CASE_METRICS:
                row[f"{short}_{model}_{metric}"] = feature_scalar(
                    block,
                    metric,
                    selected_head,
                    selected_center_index,
                )
            quality, ranks, gate = case_quality(
                query,
                model,
                block,
                calibration_ids,
                features,
                selected_head,
                selected_center_index,
            )
            row[f"{short}_{model}_quality"] = quality
            row[f"{short}_{model}_component_percentiles"] = ranks
            query_quality[(query, model)] = quality
            query_gates[(query, model)] = gate
            query_ranks[(query, model)] = ranks
            direct_maps[(query, model)] = selected_probability(
                records[(case_id, model)],
                query,
                selected_head,
                selected_center,
                radius,
            )
        row[f"{short}_ours_ttrv_jsd"] = float(
            normalized_jsd_batch(
                direct_maps[(query, "ours")],
                direct_maps[(query, "ttrv")],
            )
        )
        row[f"{short}_ours_ttrv_tv"] = float(
            total_variation_batch(
                direct_maps[(query, "ours")],
                direct_maps[(query, "ttrv")],
            )
        )
    for model in MODELS:
        row[f"{model}_answer_causal_jsd"] = float(
            normalized_jsd_batch(
                direct_maps[(ANSWER_QUERY, model)],
                direct_maps[(CAUSAL_QUERY, model)],
            )
        )
        row[f"{model}_answer_causal_tv"] = float(
            total_variation_batch(
                direct_maps[(ANSWER_QUERY, model)],
                direct_maps[(CAUSAL_QUERY, model)],
            )
        )
    ours_quality = query_quality[(selection_query, "ours")]
    ttrv_quality = query_quality[(selection_query, "ttrv")]
    separation = row[f"{QUERY_SHORT[selection_query]}_ours_ttrv_jsd"]
    row["selection_ours_quality"] = ours_quality
    row["selection_ttrv_quality"] = ttrv_quality
    row["selection_quality_gain"] = ours_quality - ttrv_quality
    row["selection_ours_component_percentiles"] = query_ranks[
        (selection_query, "ours")
    ]
    row["selection_ttrv_component_percentiles"] = query_ranks[
        (selection_query, "ttrv")
    ]
    row["quality_gate"] = query_gates[(selection_query, "ours")]
    row["rank_score"] = (
        0.50 * ours_quality
        + 0.30 * max(0.0, ours_quality - ttrv_quality)
        + 0.20 * float(separation)
    )


def case_fields() -> list[str]:
    fields = [
        "rank",
        "tier_rank",
        "case_id",
        "dataset",
        "question_id",
        "category",
        "question",
        "image_path",
        "target_source",
        "target_label",
        "target_option_text",
        "content_token_count",
        "content_token_ids",
        "content_token_pieces",
        "target_lexical_word_count",
        "target_lexical_gate",
        "target_lexical_reason",
        "ttrv_wrong_option_text",
        "ttrv_wrong_option_lexical_gate",
        "ttrv_wrong_option_lexical_reason",
        "allowed_labels",
        "ground_truth",
        "ours_prediction",
        "ttrv_prediction",
        "ours_correct",
        "ttrv_correct",
        "outcome_tier",
        "analysis_tier",
        "split",
        "split_hash_fraction",
        "eligible",
        "selection_query_mode",
        "selection_query_semantics",
        "selected_head_index_zero_based",
        "selected_layer_center_zero_based",
        "selected_layer_window_zero_based",
        "map_component",
        "teacher_forced_target_note",
        "grid_height",
        "grid_width",
        "tile_rows",
        "tile_columns",
        "has_thumbnail",
        "layout_signature",
        "answer_template_source_ours",
        "answer_template_source_ttrv",
        "causal_template_source_ours",
        "causal_template_source_ttrv",
    ]
    for query in QUERY_MODES:
        short = QUERY_SHORT[query]
        for model in MODELS:
            fields.extend(f"{short}_{model}_{metric}" for metric in CASE_METRICS)
            fields.extend(
                [
                    f"{short}_{model}_quality",
                    f"{short}_{model}_component_percentiles",
                ]
            )
        fields.extend(
            [
                f"{short}_ours_ttrv_jsd",
                f"{short}_ours_ttrv_tv",
            ]
        )
    fields.extend(
        [
            "ours_answer_causal_jsd",
            "ours_answer_causal_tv",
            "ttrv_answer_causal_jsd",
            "ttrv_answer_causal_tv",
            "selection_ours_quality",
            "selection_ttrv_quality",
            "selection_quality_gain",
            "selection_ours_component_percentiles",
            "selection_ttrv_component_percentiles",
            "rank_score",
            "quality_gate",
            "exclusion_reason",
            "deterministic_tie_break",
            "npz_ours",
            "npz_ttrv",
        ]
    )
    return fields


def build_case_rows(
    manifest_static: Mapping[str, Mapping[str, Any]],
    outcomes: Mapping[str, Mapping[str, Any]],
    records: Mapping[tuple[str, str], RecordMeta],
    lexical: Mapping[str, Mapping[str, Any]],
    valid_case_ids: set[str],
    calibration_ids: set[str],
    evaluation_ids: set[str],
    features: Mapping[str, Mapping[tuple[str, str], FeatureBlock]],
    calibration_order: Sequence[str],
    dataset: str,
    max_words: int,
    max_tokens: int,
    split_salt: str,
    selection_query: str,
    selected_head: int,
    selected_center_index: int,
    selected_center: int,
    radius: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id, static in sorted(manifest_static.items()):
        if str(static.get("dataset")) != dataset:
            continue
        tier, gt, ours_prediction, ttrv_prediction, ours_correct, ttrv_correct = outcome_tier(
            static,
            outcomes.get(case_id, {}),
        )
        if tier == "not_contrast":
            continue
        row = blank_case_row(
            static,
            tier,
            gt,
            ours_prediction,
            ttrv_prediction,
            ours_correct,
            ttrv_correct,
            selection_query,
            selected_head,
            selected_center,
            radius,
        )
        split_fraction = case_hash_fraction(case_id, split_salt)
        split = "calibration" if case_id in calibration_ids else "evaluation"
        row["split"] = split
        row["split_hash_fraction"] = split_fraction

        source = records.get((case_id, "ours")) or records.get((case_id, "ttrv"))
        if source is None:
            row["analysis_tier"] = "missing_attention"
            row["exclusion_reason"] = "missing_both_attention_records"
            rows.append(row)
            continue
        target_gate = target_lexical_gate(
            source,
            max_words=max_words,
            max_tokens=max_tokens,
        )
        row.update(
            {
                "target_label": source.target_label,
                "target_option_text": source.target_option_text,
                "content_token_count": len(source.content_token_ids),
                "content_token_ids": list(source.content_token_ids),
                "content_token_pieces": list(source.content_token_pieces),
                "target_lexical_word_count": target_gate["word_count"],
                "target_lexical_gate": bool(target_gate["pass"]),
                "target_lexical_reason": target_gate["reason"],
            }
        )
        options = parse_prompt_options(source.raw_prompt)
        wrong_text = options.get(ttrv_prediction, "") if tier == "valid_wrong" else ""
        wrong_gate = (
            text_lexical_gate(wrong_text, max_words=max_words)
            if wrong_text
            else {
                "pass": False,
                "reason": "unknown_or_unavailable_wrong_option",
                "word_count": 0,
            }
        )
        row.update(
            {
                "ttrv_wrong_option_text": wrong_text,
                "ttrv_wrong_option_lexical_gate": bool(wrong_gate["pass"]),
                "ttrv_wrong_option_lexical_reason": wrong_gate["reason"],
            }
        )

        if not target_gate["pass"]:
            row["analysis_tier"] = "nonlexical_target"
            row["exclusion_reason"] = str(target_gate["reason"])
            rows.append(row)
            continue
        if case_id not in valid_case_ids:
            row["analysis_tier"] = "missing_attention"
            row["exclusion_reason"] = "attention_pair_missing_or_invalid"
            rows.append(row)
            continue
        if case_id in calibration_ids:
            row["analysis_tier"] = "calibration_split_excluded"
            row["exclusion_reason"] = "case_used_in_correctness_blind_calibration"
            rows.append(row)
            continue
        if case_id not in evaluation_ids:
            row["analysis_tier"] = "schema_mismatch"
            row["exclusion_reason"] = "case_not_in_frozen_split"
            rows.append(row)
            continue
        if tier == "valid_wrong" and not wrong_gate["pass"]:
            row["analysis_tier"] = "nonlexical_wrong_option"
            row["exclusion_reason"] = str(wrong_gate["reason"])
            rows.append(row)
            continue
        selected_maps_valid = all(
            bool(
                features[query][(case_id, model)].valid_mask[
                    selected_head,
                    selected_center_index,
                ]
            )
            for query in QUERY_MODES
            for model in MODELS
        )
        if not selected_maps_valid:
            row["analysis_tier"] = "invalid_selected_setting_map"
            row["exclusion_reason"] = "zero_mass_or_missing_template_at_frozen_setting"
            rows.append(row)
            continue

        ours_record = records[(case_id, "ours")]
        ttrv_record = records[(case_id, "ttrv")]
        row.update(
            {
                "grid_height": ours_record.grid_height,
                "grid_width": ours_record.grid_width,
                "tile_rows": ours_record.tile_rows,
                "tile_columns": ours_record.tile_columns,
                "has_thumbnail": ours_record.has_thumbnail,
                "layout_signature": ours_record.layout_signature,
                "npz_ours": str(ours_record.npz_path),
                "npz_ttrv": str(ttrv_record.npz_path),
            }
        )
        for query in QUERY_MODES:
            short = QUERY_SHORT[query]
            row[f"{short}_template_source_ours"] = features[query][
                (case_id, "ours")
            ].template_source
            row[f"{short}_template_source_ttrv"] = features[query][
                (case_id, "ttrv")
            ].template_source
        populate_case_metrics(
            row,
            case_id,
            records,
            features,
            calibration_order,
            selected_head,
            selected_center_index,
            selected_center,
            radius,
            selection_query,
        )
        row["eligible"] = True
        row["analysis_tier"] = (
            "primary_ranked" if tier == "valid_wrong" else "secondary_unknown_ranked"
        )
        if not row["quality_gate"]:
            row["exclusion_reason"] = "below_predeclared_quality_quartile_gate"
        rows.append(row)

    ranked = [
        row
        for row in rows
        if bool(row.get("eligible")) and isinstance(row.get("rank_score"), (float, int))
    ]
    tier_order = {"valid_wrong": 0, "unknown_wrong": 1}
    ranked.sort(
        key=lambda row: (
            tier_order.get(str(row["outcome_tier"]), 9),
            -float(row["rank_score"]),
            str(row["deterministic_tie_break"]),
        )
    )
    tier_counts: Counter[str] = Counter()
    for global_rank, row in enumerate(ranked, start=1):
        tier_counts[str(row["outcome_tier"])] += 1
        row["rank"] = global_rank
        row["tier_rank"] = tier_counts[str(row["outcome_tier"])]
    ranked_ids = {str(row["case_id"]) for row in ranked}
    unranked = sorted(
        (row for row in rows if str(row["case_id"]) not in ranked_ids),
        key=lambda row: (
            str(row.get("analysis_tier")),
            str(row.get("deterministic_tie_break")),
        ),
    )
    return ranked + unranked


def template_hashes(
    template_sums: Mapping[tuple[str, str, str], np.ndarray],
    template_counts: Mapping[tuple[str, str, str], int],
    selected_head: int,
    selected_center_index: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key in sorted(template_sums):
        query, model, signature = key
        selected = normalize_template(
            template_sums[key][selected_head, selected_center_index]
        )
        digest = hashlib.sha256(
            np.ascontiguousarray(selected, dtype=np.float64).tobytes()
        ).hexdigest()
        result["|".join(key)] = {
            "count": int(template_counts[key]),
            "sha256_float64": digest,
        }
    return result


def prepare_output_paths(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "global_setting_candidates": output_dir / "global_setting_candidates.csv",
        "lexical_contrast_ranked": output_dir / "lexical_contrast_ranked.csv",
        "config": output_dir / "selection_config.json",
        "summary": output_dir / "summary.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")
    return paths


def run_validate(args: argparse.Namespace) -> int:
    records_path = Path(args.records).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    records, record_parse_errors = load_trimmed_records(records_path)
    manifest, manifest_errors = load_manifest_static(manifest_path)
    schema_errors: list[str] = []
    selected_records = list(records.values())
    if args.max_records > 0:
        selected_records = selected_records[: args.max_records]
    for index, record in enumerate(selected_records, start=1):
        schema_errors.extend(
            validate_record_schema(record, load_values=args.load_values)
        )
        if index % 50 == 0:
            print(f"[validate] records={index}/{len(selected_records)}", file=sys.stderr)
    pair_errors = []
    complete_pairs = 0
    for case_id in sorted(manifest):
        ours = records.get((case_id, "ours"))
        ttrv = records.get((case_id, "ttrv"))
        if ours is None or ttrv is None:
            continue
        try:
            assert_pair_consistency(ours, ttrv)
            complete_pairs += 1
        except Exception as exc:
            pair_errors.append(str(exc))
    result = {
        "records_path": str(records_path),
        "manifest_path": str(manifest_path),
        "record_count": len(records),
        "manifest_case_count": len(manifest),
        "complete_pair_count": complete_pairs,
        "record_parse_errors": record_parse_errors,
        "manifest_errors": manifest_errors,
        "schema_errors": schema_errors,
        "pair_errors": pair_errors,
        "query_modes": list(QUERY_MODES),
        "status": "PASS"
        if not (record_parse_errors or manifest_errors or schema_errors or pair_errors)
        else "FAIL",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))
    return 0 if result["status"] == "PASS" else 2


def run_screen(args: argparse.Namespace) -> int:
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction must lie in (0,1)")
    if args.window_radius != 1:
        raise ValueError("This preregistered screen requires --window-radius 1")
    fixed_values = (args.fixed_head_index, args.fixed_layer_center)
    if (fixed_values[0] is None) != (fixed_values[1] is None):
        raise ValueError(
            "--fixed-head-index and --fixed-layer-center must be supplied together"
        )
    fixed_setting = fixed_values[0] is not None
    if fixed_setting:
        if not 0 <= int(args.fixed_head_index) < 12:
            raise ValueError("--fixed-head-index must lie in [0,11]")
        if not args.window_radius <= int(args.fixed_layer_center) < 28 - args.window_radius:
            raise ValueError(
                "--fixed-layer-center does not admit the requested layer window"
            )
    records_path = Path(args.records).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    paths = prepare_output_paths(output_dir)

    print("[screen] loading correctness-blind metadata", flush=True)
    records, record_parse_errors = load_trimmed_records(records_path)
    manifest_static, manifest_errors = load_manifest_static(manifest_path)
    if record_parse_errors or manifest_errors:
        raise ValueError(
            f"Input parse errors: records={record_parse_errors[:3]} "
            f"manifest={manifest_errors[:3]}"
        )
    lexical, lexical_exclusions, pair_errors = build_lexical_cases(
        records,
        manifest_static,
        dataset=args.dataset,
        max_words=args.max_option_words,
        max_tokens=args.max_content_tokens,
    )
    valid_ids, npz_failures = validate_lexical_npzs(lexical, records)
    valid_set = set(valid_ids)
    if len(valid_ids) < args.min_lexical_cases:
        raise RuntimeError(
            f"Only {len(valid_ids)} valid lexical cases; "
            f"minimum is {args.min_lexical_cases}"
        )
    split_fraction = {
        case_id: case_hash_fraction(case_id, args.split_salt)
        for case_id in valid_ids
    }
    calibration_ids = {
        case_id
        for case_id in valid_ids
        if split_fraction[case_id] < args.calibration_fraction
    }
    evaluation_ids = valid_set - calibration_ids
    if len(calibration_ids) < args.min_calibration_cases:
        raise RuntimeError(
            f"Only {len(calibration_ids)} calibration cases; "
            f"minimum is {args.min_calibration_cases}"
        )
    calibration_order = sorted(calibration_ids)
    print(
        f"[screen] lexical={len(valid_ids)} calibration={len(calibration_ids)} "
        f"evaluation={len(evaluation_ids)}; predictions not loaded",
        flush=True,
    )
    features, template_sums, template_counts = extract_all_features(
        valid_ids,
        calibration_ids,
        records,
        radius=args.window_radius,
        template_size=args.template_size,
        min_template_group=args.min_template_group,
    )
    stats_by_query = {
        query: setting_statistics(
            query,
            calibration_order,
            features,
            folds=args.cv_folds,
            top_k=args.cv_top_k,
            split_salt=args.split_salt,
        )
        for query in QUERY_MODES
    }
    if fixed_setting:
        selected_head = int(args.fixed_head_index)
        selected_center = int(args.fixed_layer_center)
        selected_center_index = selected_center - args.window_radius
        fallback = False
        selection_reason = "precommitted_fixed_setting"
    else:
        (
            selected_head,
            selected_center_index,
            fallback,
            selection_reason,
        ) = choose_global_setting(
            stats_by_query,
            args.selection_query,
            args.min_cv_topk_rate,
        )
        selected_center = args.window_radius + selected_center_index
    setting_rows = make_setting_rows(
        stats_by_query,
        features,
        calibration_order,
        args.selection_query,
        args.window_radius,
        args.min_cv_topk_rate,
        selected_head,
        selected_center_index,
    )
    if sum(bool(row["selected"]) for row in setting_rows) != 1:
        raise AssertionError("Expected exactly one selected setting")
    selected_setting = next(row for row in setting_rows if row["selected"])
    print(
        f"[screen] setting frozen query={args.selection_query} "
        f"head={selected_head} center={selected_center} "
        f"score={selected_setting['selection_score']:.6g}",
        flush=True,
    )

    # Correctness/outcome metadata is deliberately loaded only after this point.
    outcomes = load_manifest_outcomes(manifest_path)
    case_rows = build_case_rows(
        manifest_static,
        outcomes,
        records,
        lexical,
        valid_set,
        calibration_ids,
        evaluation_ids,
        features,
        calibration_order,
        dataset=args.dataset,
        max_words=args.max_option_words,
        max_tokens=args.max_content_tokens,
        split_salt=args.split_salt,
        selection_query=args.selection_query,
        selected_head=selected_head,
        selected_center_index=selected_center_index,
        selected_center=selected_center,
        radius=args.window_radius,
    )

    config = {
        "created_at": utc_now(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {
            "records": str(records_path),
            "records_sha256": sha256_file(records_path),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "cpu_only": True,
        "torch_imported": False,
        "models": list(MODELS),
        "dataset": args.dataset,
        "selection_query_mode": args.selection_query,
        "selection_query_semantics": QUERY_SEMANTICS[args.selection_query],
        "causal_audit_query_mode": CAUSAL_QUERY,
        "causal_audit_query_semantics": QUERY_SEMANTICS[CAUSAL_QUERY],
        "map_component": "fused_local_plus_thumbnail",
        "selection_setting_shared_across_models_and_cases": True,
        "per_case_head_selection": False,
        "correctness_isolation": {
            "split_fields": ["case_id"],
            "split_hash": "sha256(salt + NUL + case_id), first 64 bits",
            "calibration_fraction": args.calibration_fraction,
            "split_salt": args.split_salt,
            "predictions_loaded_after_setting_frozen": True,
            "prediction_or_correctness_fields_used_in_setting_objective": False,
            "calibration_ids": calibration_order,
            "evaluation_ids": sorted(evaluation_ids),
        },
        "lexical_gate": {
            "max_content_tokens": args.max_content_tokens,
            "max_option_words": args.max_option_words,
            "excluded_exact_boolean_values": sorted(YES_NO_VALUES),
            "excluded_generic_values": sorted(GENERIC_OPTION_VALUES),
            "excluded_patterns": ["pure_number", "label_like", "no_lexical_word"],
            "content_tokens_from_saved_v2_metadata": True,
        },
        "setting_search": {
            "selection_policy": "precommitted_fixed" if fixed_setting else "calibration_selected",
            "heads_zero_based": list(range(12)),
            "layer_centers_zero_based": list(range(1, 27)),
            "window_radius": args.window_radius,
            "candidate_count": 312,
            "quality_formula": (
                "geometric_mean(within-case mass rank, boundary quality, "
                "template JSD case specificity, adjacent-layer stability)"
            ),
            "global_formula": (
                "min_model(median_case_quality - 0.25*IQR_case_quality)"
            ),
            "cv_folds": args.cv_folds,
            "cv_top_k": args.cv_top_k,
            "min_cv_topk_rate": args.min_cv_topk_rate,
            "selection_fallback": fallback,
            "selection_reason": selection_reason,
        },
        "boundary_metrics": {
            "density_lift": "mean probability in mask / mean probability in comparison",
            "outer1": "one-cell global outer ring vs complement",
            "outer10": "outer ceil(10%*min(H,W)) band vs complement",
            "tile_internal": (
                "one-cell token-tile edges derived from sqrt(tokens_per_tile), "
                "excluding global outer ring"
            ),
            "leading": "first two rows/columns vs disjoint last two rows/columns",
            "boundary_lift": "maximum of outer1, outer10, tile_internal, leading",
            "boundary_quality": "exp(-max(0, log(boundary_lift)))",
        },
        "template_metrics": {
            "size": args.template_size,
            "group": "(query,model,tile_rows,tile_columns,thumbnail,grid_shape)",
            "estimator": "pixelwise mean of resized conditional maps, renormalized",
            "calibration_policy": "leave-one-out",
            "evaluation_policy": "frozen full-calibration",
            "minimum_layout_group": args.min_template_group,
            "small_group_fallback": "model/query global calibration template",
            "jsd_normalized_by_log2": True,
            "template_hashes_selected_setting": template_hashes(
                template_sums,
                template_counts,
                selected_head,
                selected_center_index,
            ),
        },
        "candidate_rank": {
            "primary_tier": "valid_wrong",
            "secondary_tier": "unknown_wrong",
            "unknown_mixed_with_primary": False,
            "rank_formula": (
                "0.50*Ours_quality + 0.30*max(0,Ours_quality-TTRV_quality) "
                "+ 0.20*Ours_TTRV_normalized_JSD"
            ),
            "quality_gate": "all Ours component percentiles >= 0.25",
        },
        "selected_setting": {
            "precommitted": fixed_setting,
            "head_index_zero_based": selected_head,
            "layer_center_zero_based": selected_center,
            "layer_window_zero_based": list(
                range(
                    selected_center - args.window_radius,
                    selected_center + args.window_radius + 1,
                )
            ),
            "selection_score": selected_setting["selection_score"],
            "selection_cv_topk_rate": selected_setting["selection_cv_topk_rate"],
            "answer_global_rank": selected_setting["answer_global_rank"],
            "causal_global_rank": selected_setting["causal_global_rank"],
        },
    }
    summary = {
        "status": "complete",
        "created_at": utc_now(),
        "record_count": len(records),
        "manifest_case_count": len(manifest_static),
        "dataset_case_count": sum(
            str(row.get("dataset")) == args.dataset
            for row in manifest_static.values()
        ),
        "lexical_case_count_before_npz_validation": len(lexical),
        "valid_lexical_case_count": len(valid_ids),
        "calibration_case_count": len(calibration_ids),
        "evaluation_case_count": len(evaluation_ids),
        "lexical_exclusion_reason_counts": dict(
            sorted(Counter(lexical_exclusions.values()).items())
        ),
        "pair_error_count": len(pair_errors),
        "npz_failure_count": len(npz_failures),
        "npz_failures": npz_failures,
        "selected_setting": config["selected_setting"],
        "selection_fallback": fallback,
        "outcome_tier_counts": dict(
            sorted(Counter(str(row["outcome_tier"]) for row in case_rows).items())
        ),
        "analysis_tier_counts": dict(
            sorted(Counter(str(row["analysis_tier"]) for row in case_rows).items())
        ),
        "quality_gate_pass_counts": dict(
            sorted(
                Counter(
                    str(row["outcome_tier"])
                    for row in case_rows
                    if bool(row.get("eligible")) and bool(row.get("quality_gate"))
                ).items()
            )
        ),
        "top_ranked": [
            {
                "rank": row["rank"],
                "tier_rank": row["tier_rank"],
                "case_id": row["case_id"],
                "outcome_tier": row["outcome_tier"],
                "target_option_text": row["target_option_text"],
                "rank_score": row["rank_score"],
                "quality_gate": row["quality_gate"],
            }
            for row in case_rows
            if row.get("rank") != ""
        ][:20],
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    write_csv(paths["global_setting_candidates"], setting_rows, setting_fields())
    write_csv(paths["lexical_contrast_ranked"], case_rows, case_fields())
    write_json(paths["config"], config)
    write_json(paths["summary"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Validate v2 JSON/NPZ schema")
    validate.add_argument("--records", required=True)
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--max-records", type=int, default=0)
    validate.add_argument("--load-values", action="store_true")

    screen = subparsers.add_parser(
        "screen",
        help="Select one global setting and rank held-out lexical contrast cases",
    )
    screen.add_argument("--records", required=True)
    screen.add_argument("--manifest", required=True)
    screen.add_argument("--output-dir", required=True)
    screen.add_argument("--dataset", default=DEFAULT_DATASET)
    screen.add_argument(
        "--selection-query",
        choices=QUERY_MODES,
        default=ANSWER_QUERY,
    )
    screen.add_argument(
        "--calibration-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_FRACTION,
    )
    screen.add_argument(
        "--split-salt",
        default="answer_content_v2_global_setting_v1",
    )
    screen.add_argument("--fixed-head-index", type=int, default=None)
    screen.add_argument("--fixed-layer-center", type=int, default=None)
    screen.add_argument("--max-content-tokens", type=int, default=4)
    screen.add_argument("--max-option-words", type=int, default=4)
    screen.add_argument("--window-radius", type=int, default=DEFAULT_WINDOW_RADIUS)
    screen.add_argument("--template-size", type=int, default=DEFAULT_TEMPLATE_SIZE)
    screen.add_argument("--min-template-group", type=int, default=5)
    screen.add_argument("--min-lexical-cases", type=int, default=40)
    screen.add_argument("--min-calibration-cases", type=int, default=20)
    screen.add_argument("--cv-folds", type=int, default=5)
    screen.add_argument("--cv-top-k", type=int, default=10)
    screen.add_argument("--min-cv-topk-rate", type=float, default=0.8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "validate":
        return run_validate(args)
    if args.command == "screen":
        return run_screen(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
