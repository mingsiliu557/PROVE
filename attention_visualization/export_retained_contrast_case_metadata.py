#!/usr/bin/env python3
"""Export question, choice, answer, artifact, and provenance metadata.

The input manifest is the strict retained-answer contrast manifest (Ours is
correct and TTRV selects a valid wrong option).  The renderer root ``index.csv``
is the source of truth for the final rendered subset and its per-dataset rank.

Every renderer row is joined back to the manifest by ``case_id`` and checked
against the original parquet row.  In particular, this exporter verifies that:

* Ours equals the ground-truth option;
* TTRV is a parsed A--F option present in the question and differs from ground
  truth;
* the renderer's ground-truth and per-model target labels/text agree with the
  options parsed by :func:`answer_content_attention.parse_options`; and
* the renderer artifact paths exist.

The output directory may already exist, but none of the three output artifacts
may exist.  This script never overwrites an earlier export.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import unicodedata

from answer_content_attention import _load_rows, parse_options


CHOICE_LABELS = tuple("ABCDEF")
VALID_LABEL = re.compile(r"^[A-F]$")
OPTION_BOUNDARY = re.compile(
    r"^[ \t]*(?:options?\s*:|[A-F][\.:\)][ \t]+)",
    flags=re.IGNORECASE | re.MULTILINE,
)
INSTRUCTION_BOUNDARY = re.compile(
    r"^[ \t]*(?:"
    r"(?:choose|select)[ \t]+(?:the[ \t]+)?correct[ \t]+answer\b|"
    r"please[ \t]+(?:answer|respond)\b|"
    r"answer[ \t]+(?:directly|with|using|only)\b|"
    r"respond[ \t]+(?:directly|with|using|only)\b|"
    r"do[ \t]+not[ \t]+(?:include|provide|output)\b|"
    r"only[ \t]+(?:answer|output|respond)\b"
    r")",
    flags=re.IGNORECASE | re.MULTILINE,
)

RENDERER_REQUIRED_COLUMNS = (
    "dataset_rank",
    "case_id",
    "dataset",
    "question_id",
    "ground_truth",
    "ours_target_source",
    "ours_target_label",
    "ours_target_option_text",
    "ours_target_correct",
    "ttrv_target_source",
    "ttrv_target_label",
    "ttrv_target_option_text",
    "ttrv_target_correct",
    "targets_differ",
    "source_image_path",
    "copied_original",
    "ours_overlay_png",
    "ttrv_overlay_png",
    "side_by_side_png",
)

OUTPUT_COLUMNS = (
    "case_id",
    "dataset",
    "rank",
    "question_id",
    "question",
    "full_prompt_with_choices",
    "choice_A",
    "choice_B",
    "choice_C",
    "choice_D",
    "choice_E",
    "choice_F",
    "ground_truth_label",
    "ground_truth_text",
    "ours_label",
    "ours_text",
    "ttrv_label",
    "ttrv_text",
    "source_image_path",
    "copied_original",
    "ours_overlay",
    "ttrv_overlay",
    "side_by_side",
    # Outcome-dependent selection metadata from the strict manifest.
    "selection_policy",
    "selection_gate",
    "selection_outcome_dependent",
    "selection_correctness_blind",
    "selection_uses_model_predictions",
    "selection_dataset_max_option_words",
    "selection_rank",
    "selection_dataset_rank",
    # Dataset and retained-prediction provenance.
    "manifest_row_index",
    "test_parquet",
    "train_parquet",
    "task",
    "parser",
    "labels",
    "retained_target_source",
    "ours_raw_response",
    "ours_prediction_source_file",
    "ours_prediction_source_step",
    "ttrv_raw_response",
    "ttrv_prediction_source_file",
    "ttrv_prediction_source_step",
    "ours_checkpoint",
    "ours_source_checkpoint",
    "ttrv_checkpoint",
    "ttrv_source_checkpoint",
    # Renderer/attention provenance.  ``rank`` above is the renderer rank.
    "renderer_index_row",
    "comparison_type",
    "query_mode",
    "head_index_zero_based",
    "layer_center_zero_based",
    "layer_window_zero_based",
    "rank_metrics_temperature",
    "raw_rank_score",
    "display_temperature",
    "display_transform",
    "answer_state_interpretation",
    "source_npz_ours",
    "source_npz_ttrv",
    # Immutable input provenance is repeated so either output is standalone.
    "strict_manifest_path",
    "strict_manifest_sha256",
    "renderer_index_path",
    "renderer_index_sha256",
)


def _compact(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalized_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", _compact(value)).casefold()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_text(row: Mapping[str, Any], field: str, context: str) -> str:
    value = _compact(row.get(field))
    if not value:
        raise ValueError(f"{context}: {field} is empty")
    return value


def _label(value: Any, field: str, case_id: str) -> str:
    label = _compact(value).upper()
    if not VALID_LABEL.fullmatch(label):
        raise ValueError(f"{case_id}: {field} must be one of A-F, got {value!r}")
    return label


def _positive_integer(value: Any, field: str, context: str) -> int:
    text = _compact(value)
    if not re.fullmatch(r"[1-9][0-9]*", text):
        raise ValueError(f"{context}: {field} must be a positive integer, got {value!r}")
    return int(text)


def _finite_float_or_blank(value: Any, field: str, context: str) -> float | str:
    text = _compact(value)
    if not text:
        return ""
    try:
        result = float(text)
    except ValueError as exc:
        raise ValueError(f"{context}: {field} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{context}: {field} must be finite, got {value!r}")
    return result


def _bool(value: Any, field: str, context: str) -> bool:
    if isinstance(value, bool):
        return value
    text = _compact(value).casefold()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"{context}: {field} is not a boolean: {value!r}")


def _expect_bool(
    row: Mapping[str, Any], field: str, expected: bool, context: str
) -> bool:
    actual = _bool(row.get(field), field, context)
    if actual is not expected:
        raise ValueError(
            f"{context}: {field}={actual!r}, expected strict value {expected!r}"
        )
    return actual


def _mapping(value: Any, field: str, case_id: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{case_id}: {field} must be an object")
    return value


def _resolve_existing_file(
    value: Any, parent: Path, field: str, context: str
) -> Path:
    text = _required_text({field: value}, field, context)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{context}: {field} does not exist: {path}")
    return path


def _read_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
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
            case_id = _required_text(value, "case_id", f"{path}:{line_number}")
            if case_id in by_id:
                raise ValueError(f"{path}:{line_number}: duplicate case_id={case_id}")
            copied = dict(value)
            rows.append(copied)
            by_id[case_id] = copied
    if not rows:
        raise ValueError(f"Strict manifest contains no records: {path}")
    return rows, by_id


def _read_renderer_index(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if not fields:
            raise ValueError(f"Renderer index has no header: {path}")
        duplicate_fields = sorted(
            field for field, count in Counter(fields).items() if count > 1
        )
        if duplicate_fields:
            raise ValueError(f"Renderer index has duplicate columns: {duplicate_fields}")
        missing = sorted(set(RENDERER_REQUIRED_COLUMNS) - set(fields))
        if missing:
            raise ValueError(f"Renderer index is missing required columns: {missing}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Renderer index contains no rendered cases: {path}")
    return rows, fields


def _prompt_text(row: Mapping[str, Any]) -> str:
    """Return the original text content used by ``parse_options``.

    Parquet prompts are normally a list of chat messages with string content.
    The structured-content branch keeps text and image placeholders usable if a
    future parquet already stores OpenAI-style content segments.
    """

    messages = row.get("prompt")
    if not isinstance(messages, list):
        text = str(messages or "").strip()
        if not text:
            raise ValueError("Parquet row prompt is empty")
        return text
    pieces: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            segments: list[str] = []
            for segment in content:
                if not isinstance(segment, Mapping):
                    continue
                kind = _compact(segment.get("type")).casefold()
                if kind == "text":
                    segments.append(str(segment.get("text", "")))
                elif kind in {"image", "image_url"}:
                    segments.append("<image>")
                elif kind in {"video", "video_url"}:
                    segments.append("<video>")
            content_text = "".join(segments)
        else:
            content_text = str(content)
        if content_text.strip():
            pieces.append(content_text.strip())
    text = "\n".join(pieces).strip()
    if not text:
        raise ValueError("Parquet row prompt has no textual content")
    return text


def _question_only(full_prompt: str, case_id: str) -> str:
    text = full_prompt.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<(?:image|video)>", "\n", text, flags=re.IGNORECASE)
    boundaries = [
        match.start()
        for pattern in (OPTION_BOUNDARY, INSTRUCTION_BOUNDARY)
        if (match := pattern.search(text)) is not None
    ]
    prefix = text[: min(boundaries)] if boundaries else text
    question = _compact(prefix)
    if not question:
        raise ValueError(f"{case_id}: pure question stem is empty")
    if OPTION_BOUNDARY.search(question) or INSTRUCTION_BOUNDARY.search(question):
        raise ValueError(f"{case_id}: choices/instructions leaked into pure question stem")
    return question


def _same_text(
    actual: Any, expected: Any, field: str, case_id: str
) -> None:
    if _normalized_text(actual) != _normalized_text(expected):
        raise ValueError(
            f"{case_id}: {field} text mismatch: actual={actual!r}, expected={expected!r}"
        )


def _nested_model(
    case: Mapping[str, Any], container: str, model: str, case_id: str
) -> Mapping[str, Any]:
    parent = _mapping(case.get(container), container, case_id)
    return _mapping(parent.get(model), f"{container}.{model}", case_id)


def _json_list_or_text(value: Any, field: str, context: str) -> list[Any] | str:
    text = _compact(value)
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(parsed, list):
        raise ValueError(f"{context}: {field} must encode a JSON list, got {value!r}")
    return parsed


def _build_case_row(
    renderer: Mapping[str, Any],
    renderer_line: int,
    case: Mapping[str, Any],
    parquet_row: Mapping[str, Any],
    manifest_path: Path,
    renderer_path: Path,
    manifest_sha256: str,
    renderer_sha256: str,
) -> dict[str, Any]:
    context = f"{renderer_path}:{renderer_line}"
    case_id = _required_text(renderer, "case_id", context)
    manifest_case_id = _required_text(case, "case_id", f"manifest case {case_id}")
    if manifest_case_id != case_id:
        raise ValueError(f"{case_id}: renderer/manifest case_id drift")

    dataset = _required_text(renderer, "dataset", context)
    manifest_dataset = _required_text(case, "dataset", case_id)
    if dataset != manifest_dataset:
        raise ValueError(
            f"{case_id}: renderer dataset={dataset!r} != manifest={manifest_dataset!r}"
        )
    rank = _positive_integer(renderer.get("dataset_rank"), "dataset_rank", context)

    question_id = _required_text(case, "question_id", case_id)
    renderer_question_id = _required_text(renderer, "question_id", context)
    if renderer_question_id != question_id:
        raise ValueError(
            f"{case_id}: renderer question_id={renderer_question_id!r} "
            f"!= manifest={question_id!r}"
        )

    full_prompt = _prompt_text(parquet_row)
    manifest_prompt = _required_text(case, "question", case_id)
    manifest_prompt_without_media = re.sub(
        r"<(?:image|video)>", " ", manifest_prompt, flags=re.IGNORECASE
    )
    parquet_prompt_without_media = re.sub(
        r"<(?:image|video)>", " ", full_prompt, flags=re.IGNORECASE
    )
    _same_text(
        manifest_prompt_without_media,
        parquet_prompt_without_media,
        "manifest question versus parquet prompt",
        case_id,
    )
    question = _question_only(full_prompt, case_id)
    options = parse_options(parquet_row)
    unexpected_labels = sorted(set(options) - set(CHOICE_LABELS))
    if unexpected_labels:
        raise ValueError(f"{case_id}: parsed unsupported choices: {unexpected_labels}")

    ground_truth = _label(case.get("ground_truth"), "ground_truth", case_id)
    ours_prediction = _nested_model(
        case, "retained_predictions", "ours", case_id
    )
    ttrv_prediction = _nested_model(
        case, "retained_predictions", "ttrv", case_id
    )
    ours_label = _label(
        ours_prediction.get("parsed_answer"),
        "retained_predictions.ours.parsed_answer",
        case_id,
    )
    ttrv_label = _label(
        ttrv_prediction.get("parsed_answer"),
        "retained_predictions.ttrv.parsed_answer",
        case_id,
    )
    for field, label in (
        ("ground_truth", ground_truth),
        ("ours retained prediction", ours_label),
        ("ttrv retained prediction", ttrv_label),
    ):
        if label not in options:
            raise ValueError(
                f"{case_id}: {field} label {label} absent from parsed options "
                f"{sorted(options)}"
            )
    if ours_label != ground_truth:
        raise ValueError(
            f"{case_id}: strict outcome violated: Ours={ours_label}, GT={ground_truth}"
        )
    if ttrv_label == ground_truth:
        raise ValueError(
            f"{case_id}: strict outcome violated: TTRV={ttrv_label} equals GT={ground_truth}"
        )

    manifest_ours_label = _label(
        case.get("ours_target_label"), "ours_target_label", case_id
    )
    manifest_ttrv_label = _label(
        case.get("ttrv_target_label"), "ttrv_target_label", case_id
    )
    if manifest_ours_label != ours_label or manifest_ttrv_label != ttrv_label:
        raise ValueError(
            f"{case_id}: manifest target-label drift: "
            f"ours={manifest_ours_label}/{ours_label}, "
            f"ttrv={manifest_ttrv_label}/{ttrv_label}"
        )
    ours_text = str(options[ours_label])
    ttrv_text = str(options[ttrv_label])
    ground_truth_text = str(options[ground_truth])
    _same_text(case.get("ours_target_option_text"), ours_text, "manifest Ours target", case_id)
    _same_text(case.get("ttrv_target_option_text"), ttrv_text, "manifest TTRV target", case_id)
    _expect_bool(case, "ours_correct", True, case_id)
    _expect_bool(case, "ttrv_correct", False, case_id)
    selection_outcome_dependent = _expect_bool(
        case, "selection_outcome_dependent", True, case_id
    )
    selection_correctness_blind = _expect_bool(
        case, "selection_correctness_blind", False, case_id
    )
    selection_uses_predictions = _expect_bool(
        case, "selection_uses_model_predictions", True, case_id
    )

    renderer_gt = _label(renderer.get("ground_truth"), "renderer.ground_truth", case_id)
    renderer_ours = _label(
        renderer.get("ours_target_label"), "renderer.ours_target_label", case_id
    )
    renderer_ttrv = _label(
        renderer.get("ttrv_target_label"), "renderer.ttrv_target_label", case_id
    )
    if (renderer_gt, renderer_ours, renderer_ttrv) != (
        ground_truth,
        ours_label,
        ttrv_label,
    ):
        raise ValueError(
            f"{case_id}: renderer target labels disagree with manifest/parquet: "
            f"renderer={(renderer_gt, renderer_ours, renderer_ttrv)}, "
            f"expected={(ground_truth, ours_label, ttrv_label)}"
        )
    if _required_text(renderer, "ours_target_source", context) != "retained":
        raise ValueError(f"{case_id}: renderer ours_target_source must be 'retained'")
    if _required_text(renderer, "ttrv_target_source", context) != "retained":
        raise ValueError(f"{case_id}: renderer ttrv_target_source must be 'retained'")
    _same_text(renderer.get("ours_target_option_text"), ours_text, "renderer Ours target", case_id)
    _same_text(renderer.get("ttrv_target_option_text"), ttrv_text, "renderer TTRV target", case_id)
    _expect_bool(renderer, "ours_target_correct", True, context)
    _expect_bool(renderer, "ttrv_target_correct", False, context)
    _expect_bool(renderer, "targets_differ", True, context)

    manifest_image = _resolve_existing_file(
        case.get("image_path"), manifest_path.parent, "manifest.image_path", case_id
    )
    renderer_image = _resolve_existing_file(
        renderer.get("source_image_path"),
        renderer_path.parent,
        "source_image_path",
        case_id,
    )
    if manifest_image != renderer_image:
        raise ValueError(
            f"{case_id}: renderer source image {renderer_image} "
            f"!= manifest source image {manifest_image}"
        )
    copied_original = _resolve_existing_file(
        renderer.get("copied_original"),
        renderer_path.parent,
        "copied_original",
        case_id,
    )
    ours_overlay = _resolve_existing_file(
        renderer.get("ours_overlay_png"),
        renderer_path.parent,
        "ours_overlay_png",
        case_id,
    )
    ttrv_overlay = _resolve_existing_file(
        renderer.get("ttrv_overlay_png"),
        renderer_path.parent,
        "ttrv_overlay_png",
        case_id,
    )
    side_by_side = _resolve_existing_file(
        renderer.get("side_by_side_png"),
        renderer_path.parent,
        "side_by_side_png",
        case_id,
    )

    ours_model = _nested_model(case, "models", "ours", case_id)
    ttrv_model = _nested_model(case, "models", "ttrv", case_id)
    selection_rank = _positive_integer(
        case.get("selection_rank"), "selection_rank", case_id
    )
    selection_dataset_rank = _positive_integer(
        case.get("selection_dataset_rank"), "selection_dataset_rank", case_id
    )

    result: dict[str, Any] = {
        "case_id": case_id,
        "dataset": dataset,
        "rank": rank,
        "question_id": question_id,
        "question": question,
        "full_prompt_with_choices": full_prompt,
        **{f"choice_{label}": str(options.get(label, "")) for label in CHOICE_LABELS},
        "ground_truth_label": ground_truth,
        "ground_truth_text": ground_truth_text,
        "ours_label": ours_label,
        "ours_text": ours_text,
        "ttrv_label": ttrv_label,
        "ttrv_text": ttrv_text,
        "source_image_path": str(manifest_image),
        "copied_original": str(copied_original),
        "ours_overlay": str(ours_overlay),
        "ttrv_overlay": str(ttrv_overlay),
        "side_by_side": str(side_by_side),
        "selection_policy": _required_text(case, "selection_policy", case_id),
        "selection_gate": _required_text(case, "selection_gate", case_id),
        "selection_outcome_dependent": selection_outcome_dependent,
        "selection_correctness_blind": selection_correctness_blind,
        "selection_uses_model_predictions": selection_uses_predictions,
        "selection_dataset_max_option_words": _positive_integer(
            case.get("selection_dataset_max_option_words"),
            "selection_dataset_max_option_words",
            case_id,
        ),
        "selection_rank": selection_rank,
        "selection_dataset_rank": selection_dataset_rank,
        "manifest_row_index": int(case.get("row_index")),
        "test_parquet": str(Path(_required_text(case, "test_parquet", case_id)).expanduser().resolve()),
        "train_parquet": ""
        if case.get("train_parquet") in (None, "")
        else str(Path(str(case["train_parquet"])).expanduser().resolve()),
        "task": _compact(case.get("task")),
        "parser": _compact(case.get("parser")),
        "labels": _compact(case.get("labels")),
        "retained_target_source": _required_text(case, "retained_target_source", case_id),
        "ours_raw_response": str(ours_prediction.get("raw_response", "")),
        "ours_prediction_source_file": _compact(ours_prediction.get("source_file")),
        "ours_prediction_source_step": ours_prediction.get("source_step", ""),
        "ttrv_raw_response": str(ttrv_prediction.get("raw_response", "")),
        "ttrv_prediction_source_file": _compact(ttrv_prediction.get("source_file")),
        "ttrv_prediction_source_step": ttrv_prediction.get("source_step", ""),
        "ours_checkpoint": _required_text(ours_model, "checkpoint", case_id),
        "ours_source_checkpoint": _compact(ours_model.get("source_checkpoint")),
        "ttrv_checkpoint": _required_text(ttrv_model, "checkpoint", case_id),
        "ttrv_source_checkpoint": _compact(ttrv_model.get("source_checkpoint")),
        "renderer_index_row": renderer_line,
        "comparison_type": _compact(renderer.get("comparison_type")),
        "query_mode": _compact(renderer.get("query_mode")),
        "head_index_zero_based": _finite_float_or_blank(
            renderer.get("head_index_zero_based"), "head_index_zero_based", context
        ),
        "layer_center_zero_based": _finite_float_or_blank(
            renderer.get("layer_center_zero_based"),
            "layer_center_zero_based",
            context,
        ),
        "layer_window_zero_based": _json_list_or_text(
            renderer.get("layer_window_zero_based"),
            "layer_window_zero_based",
            context,
        ),
        "rank_metrics_temperature": _finite_float_or_blank(
            renderer.get("rank_metrics_temperature"),
            "rank_metrics_temperature",
            context,
        ),
        "raw_rank_score": _finite_float_or_blank(
            renderer.get("raw_rank_score"), "raw_rank_score", context
        ),
        "display_temperature": _finite_float_or_blank(
            renderer.get("display_temperature"), "display_temperature", context
        ),
        "display_transform": str(renderer.get("display_transform", "")),
        "answer_state_interpretation": str(
            renderer.get("answer_state_interpretation", "")
        ),
        "source_npz_ours": _compact(renderer.get("source_npz_ours")),
        "source_npz_ttrv": _compact(renderer.get("source_npz_ttrv")),
        "strict_manifest_path": str(manifest_path),
        "strict_manifest_sha256": manifest_sha256,
        "renderer_index_path": str(renderer_path),
        "renderer_index_sha256": renderer_sha256,
    }
    missing_output = sorted(set(OUTPUT_COLUMNS) - set(result))
    extra_output = sorted(set(result) - set(OUTPUT_COLUMNS))
    if missing_output or extra_output:
        raise AssertionError(
            f"Internal output schema drift: missing={missing_output}, extra={extra_output}"
        )
    return result


def _validate_renderer_keys(
    rows: Sequence[Mapping[str, Any]], manifest_by_id: Mapping[str, Mapping[str, Any]]
) -> tuple[list[str], dict[str, int]]:
    dataset_order: list[str] = []
    ranks_by_dataset: dict[str, set[int]] = defaultdict(set)
    seen_keys: set[tuple[str, int]] = set()
    seen_cases: set[str] = set()
    counts: Counter[str] = Counter()
    for offset, row in enumerate(rows, start=2):
        context = f"renderer index row {offset}"
        dataset = _required_text(row, "dataset", context)
        rank = _positive_integer(row.get("dataset_rank"), "dataset_rank", context)
        case_id = _required_text(row, "case_id", context)
        key = (dataset, rank)
        if key in seen_keys:
            raise ValueError(f"Duplicate renderer (dataset,dataset_rank) key: {key}")
        if case_id in seen_cases:
            raise ValueError(f"Duplicate rendered case_id={case_id}")
        if case_id not in manifest_by_id:
            raise ValueError(f"Renderer case_id absent from strict manifest: {case_id}")
        seen_keys.add(key)
        seen_cases.add(case_id)
        ranks_by_dataset[dataset].add(rank)
        counts[dataset] += 1
        if dataset not in dataset_order:
            dataset_order.append(dataset)
    for dataset, ranks in ranks_by_dataset.items():
        expected = set(range(1, len(ranks) + 1))
        if ranks != expected:
            raise ValueError(
                f"Renderer ranks for {dataset} must be contiguous 1..{len(ranks)}; "
                f"got {sorted(ranks)}"
            )
    return dataset_order, dict(counts)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row[field]) for field in OUTPUT_COLUMNS})


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=False))
            handle.write("\n")


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def export_case_metadata(
    strict_manifest: str | Path,
    renderer_index: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(strict_manifest).expanduser().resolve()
    renderer_path = Path(renderer_index).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Strict manifest does not exist: {manifest_path}")
    if not renderer_path.is_file():
        raise FileNotFoundError(f"Renderer root index does not exist: {renderer_path}")

    csv_path = destination / "case_metadata.csv"
    jsonl_path = destination / "case_metadata.jsonl"
    summary_path = destination / "case_metadata.summary.json"
    output_paths = (csv_path, jsonl_path, summary_path)
    for path in output_paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")

    manifest_rows, manifest_by_id = _read_manifest(manifest_path)
    renderer_rows, _ = _read_renderer_index(renderer_path)
    dataset_order, rendered_counts = _validate_renderer_keys(
        renderer_rows, manifest_by_id
    )
    manifest_sha256 = _sha256(manifest_path)
    renderer_sha256 = _sha256(renderer_path)

    # _load_rows reads only the parquet rows referenced by final rendered cases
    # and independently checks each question_id against the manifest.
    rendered_manifest_cases = [
        manifest_by_id[_required_text(row, "case_id", "renderer index")]
        for row in renderer_rows
    ]
    parquet_by_case = _load_rows(rendered_manifest_cases)
    expected_case_ids = {
        _required_text(case, "case_id", "rendered manifest case")
        for case in rendered_manifest_cases
    }
    if set(parquet_by_case) != expected_case_ids:
        raise ValueError(
            "Parquet loader returned a different case set: "
            f"expected={sorted(expected_case_ids)}, got={sorted(parquet_by_case)}"
        )

    order_index = {dataset: index for index, dataset in enumerate(dataset_order)}
    indexed_rows = list(enumerate(renderer_rows, start=2))
    indexed_rows.sort(
        key=lambda pair: (
            order_index[_required_text(pair[1], "dataset", f"row {pair[0]}")],
            _positive_integer(
                pair[1].get("dataset_rank"), "dataset_rank", f"row {pair[0]}"
            ),
        )
    )
    output_rows: list[dict[str, Any]] = []
    for renderer_line, renderer in indexed_rows:
        case_id = _required_text(renderer, "case_id", f"row {renderer_line}")
        output_rows.append(
            _build_case_row(
                renderer,
                renderer_line,
                manifest_by_id[case_id],
                parquet_by_case[case_id],
                manifest_path,
                renderer_path,
                manifest_sha256,
                renderer_sha256,
            )
        )

    if len(output_rows) != len(renderer_rows):
        raise AssertionError("Not every renderer row produced one metadata row")

    summary: dict[str, Any] = {
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strict_manifest": str(manifest_path),
        "strict_manifest_sha256": manifest_sha256,
        "strict_manifest_cases": len(manifest_rows),
        "renderer_root_index": str(renderer_path),
        "renderer_root_index_sha256": renderer_sha256,
        "rendered_cases": len(renderer_rows),
        "rendered_by_dataset": {
            dataset: rendered_counts[dataset] for dataset in dataset_order
        },
        "join_policy": (
            "renderer (dataset,dataset_rank) defines final order; case_id joins to "
            "strict manifest; parquet row_index/question_id supplies prompt and choices"
        ),
        "strict_validation": {
            "ours_equals_ground_truth": True,
            "ttrv_is_valid_option_and_differs_from_ground_truth": True,
            "renderer_target_labels_and_text_match_parquet": True,
            "renderer_artifacts_exist": True,
            "dataset_ranks_are_unique_and_contiguous": True,
        },
        "outputs": {
            "case_metadata_csv": str(csv_path),
            "case_metadata_jsonl": str(jsonl_path),
            "summary_json": str(summary_path),
        },
        "output_columns": list(OUTPUT_COLUMNS),
    }

    destination.mkdir(parents=True, exist_ok=True)
    # Repeat the no-overwrite check after mkdir to narrow the race window.  The
    # exclusive writers are the final guard.  On a later write failure, remove
    # only files created by this invocation.
    for path in output_paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    created: list[Path] = []
    try:
        _write_csv_exclusive(csv_path, output_rows)
        created.append(csv_path)
        _write_jsonl_exclusive(jsonl_path, output_rows)
        created.append(jsonl_path)
        summary["outputs"]["case_metadata_csv_sha256"] = _sha256(csv_path)
        summary["outputs"]["case_metadata_jsonl_sha256"] = _sha256(jsonl_path)
        _write_json_exclusive(summary_path, summary)
        created.append(summary_path)
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-manifest",
        "--manifest",
        dest="strict_manifest",
        required=True,
        help="Strict retained-contrast JSONL manifest",
    )
    parser.add_argument(
        "--renderer-index",
        required=True,
        help="Completed renderer root index.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for case_metadata.csv/jsonl/summary.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = export_case_metadata(
        args.strict_manifest, args.renderer_index, args.output_dir
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
