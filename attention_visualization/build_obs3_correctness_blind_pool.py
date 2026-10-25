#!/usr/bin/env python3
"""Build a held-out, correctness-blind Obs3 attention candidate pool.

The shortlist is selected *before* validation prediction files are opened.  It
uses only dataset identity, question text, a visual-keyword score, and a stable
SHA-256 key.  Ground truth and retained Ours/TTRV predictions are attached only
after selection as evaluation/provenance metadata.

The fixed checkpoint protocol is:

* MME: Ours step 20, TTRV step 20;
* AI2D: Ours step 20, TTRV step 10.

The default shortlist contains 200 cases per dataset: half selected by visual
keyword score and half by stable SHA-256 order from the remaining cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
SELECTION_POLICY = "correctness_blind_50pct_visual_50pct_sha256_v1"
SMOKE_SELECTION_POLICY = "per_dataset_visual_keyword_then_sha256_v1"
HASH_NAMESPACE = "obs3-correctness-blind-v1"
MODEL_ORDER = ("base", "ttrv", "ours")
SELECTION_ALLOWED_FIELDS = (
    "case_id",
    "dataset",
    "question_id",
    "question",
    "image_path",
    "category",
    "visual_keyword_score",
    "selection_hash",
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_REFERENCE_MANIFEST = (
    SCRIPT_DIR
    / "obs3_validation_attention_20260728"
    / "candidates_expanded_all.jsonl"
)

BASE_MODEL = Path(
    "/path/to/project/ttrv_runtime/models/"
    "OpenGVLab/InternVL3-2B"
)

VISUAL_KEYWORD_PATTERN = re.compile(
    r"\b(?:"
    r"color|colour|left|right|above|below|under|over|behind|front|"
    r"between|inside|outside|next\s+to|how\s+many|count|where|"
    r"which\s+object|shape|wearing|holding|diagram|picture|image|"
    r"depict|label|letter|arrow|position|location|top|bottom|"
    r"larger|smaller"
    r")\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    task: str
    labels: str
    test_parquet: Path
    train_parquet: Path
    ours_validation: Path
    ttrv_validation: Path
    ours_validation_selection: str
    ttrv_validation_selection: str
    ours_step: int
    ttrv_step: int
    ours_checkpoint: Path
    ttrv_checkpoint: Path
    ours_source_checkpoint: Path
    ttrv_source_checkpoint: Path


EXPERIMENT_ROOT = Path(
    "/path/to/datasets/experiments/"
    "closed_choice_pass8_ckpt_ttrv_20260709_052643"
)
OBS3_CHECKPOINT_ROOT = Path(
    "/path/to/datasets/experiments/"
    "internvl3_obs3_validation_attention_20260728/merged_checkpoints"
)

DATASETS = (
    DatasetSpec(
        key="MME",
        task="mme_20",
        labels="AB",
        test_parquet=Path("/path/to/datasets/verl_data/mme_20/test.parquet"),
        train_parquet=Path("/path/to/datasets/verl_data/mme_20/train.parquet"),
        ours_validation=EXPERIMENT_ROOT
        / "best_ckpt_pass8_mme20/validation_groups.jsonl",
        ttrv_validation=EXPERIMENT_ROOT
        / "ttrv_official_mme20_train_s10_s20_pass8/validation_groups.jsonl",
        ours_validation_selection="only",
        ttrv_validation_selection="last",
        ours_step=20,
        ttrv_step=20,
        ours_checkpoint=Path(
            "/path/to/datasets/experiments/"
            "internvl3_attention_viz_20260722_140000/merged_checkpoints/"
            "ours_mme20_step20"
        ),
        ttrv_checkpoint=OBS3_CHECKPOINT_ROOT / "ttrv_mme20_step20",
        ours_source_checkpoint=Path(
            "/path/to/datasets/experiments/"
            "visual_dep_reward_best_ckpt_20260706_141927/"
            "mme_coef002_ori000/checkpoints/global_step_20"
        ),
        ttrv_source_checkpoint=EXPERIMENT_ROOT
        / "ttrv_official_mme20_train_s10_s20_pass8/checkpoints/global_step_20",
    ),
    DatasetSpec(
        key="AI2D",
        task="ai2d_20",
        labels="A-D",
        test_parquet=Path("/path/to/datasets/verl_data/ai2d_20/test.parquet"),
        train_parquet=Path("/path/to/datasets/verl_data/ai2d_20/train.parquet"),
        ours_validation=EXPERIMENT_ROOT
        / "best_ckpt_pass8_ai2d20/validation_groups.jsonl",
        ttrv_validation=EXPERIMENT_ROOT
        / "ttrv_official_ai2d20_train_s10_s20_pass8/validation_groups.jsonl",
        ours_validation_selection="only",
        # The file contains a complete step-10 dump followed by a partial
        # step-20 dump (question IDs 0..279).  First occurrence is step 10.
        ttrv_validation_selection="first",
        ours_step=20,
        ttrv_step=10,
        ours_checkpoint=OBS3_CHECKPOINT_ROOT / "ours_ai2d20_step20",
        ttrv_checkpoint=OBS3_CHECKPOINT_ROOT / "ttrv_ai2d20_step10",
        ours_source_checkpoint=Path(
            "/path/to/datasets/experiments/"
            "visual_dep_reward_best_ckpt_20260706_141927/"
            "ai2d_coef003_ori000/checkpoints/global_step_20"
        ),
        ttrv_source_checkpoint=EXPERIMENT_ROOT
        / "ttrv_official_ai2d20_train_s10_s20_pass8/checkpoints/global_step_10",
    ),
)


QID_KEYS = (
    "question_id",
    "qid",
    "index",
    "id",
    "extra_info.question_id",
    "extra_info.index",
    "extra_info.id",
    "extra_info.source_index",
)


def nested_get(row: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = row
    for key in dotted_key.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def first_present(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = nested_get(row, key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def normalize_qid(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def row_qid(row: Mapping[str, Any], fallback: int) -> str:
    value = first_present(row, QID_KEYS)
    return normalize_qid(fallback if value is None else value)


def row_ground_truth(row: Mapping[str, Any]) -> str:
    for key in (
        "reward_model.ground_truth",
        "ground_truth",
        "answer",
        "extra_info.answer",
    ):
        value = nested_get(row, key)
        if value is not None:
            return str(value).strip()
    raise ValueError(f"No ground truth in row keys={sorted(row)}")


def row_image_path(row: Mapping[str, Any]) -> str:
    images = row.get("images")
    if isinstance(images, (list, tuple)) and images:
        first = images[0]
        value = first.get("image") or first.get("path") if isinstance(first, Mapping) else first
        if value:
            return str(Path(str(value)).expanduser().resolve())
    value = nested_get(row, "extra_info.image_path")
    if value:
        return str(Path(str(value)).expanduser().resolve())
    raise ValueError("No image path in dataset row")


def prompt_text(row: Mapping[str, Any]) -> str:
    messages = row.get("prompt")
    if not isinstance(messages, list):
        return str(messages or "")
    pieces: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        pieces.append(str(message.get("content", "")).replace("<image>", "").strip())
    text = "\n".join(piece for piece in pieces if piece)
    for marker in (
        "Please answer with yes or no.",
        "Choose the correct answer from the options below",
        "Answer with a short direct answer.",
        "Answer with the exact text from the image.",
    ):
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    return re.sub(r"\s+", " ", text).strip()


def simple_choice_parse(text: Any, labels: str) -> str:
    raw_labels = labels.upper()
    normalized_labels = raw_labels.replace("-", "")
    if raw_labels == "AB":
        allowed = tuple("AB")
    elif raw_labels in {"A-D", "ABCD"}:
        allowed = tuple("ABCD")
    elif raw_labels in {"A-F", "ABCDEF"}:
        allowed = tuple("ABCDEF")
    else:
        allowed = tuple(dict.fromkeys(re.findall(r"[A-Z]", normalized_labels)))
    upper = str(text or "").strip().upper()
    if upper in allowed:
        return upper
    triggered = re.findall(
        r"(?:ANSWER|CHOICE|OPTION)\s*(?:IS|:)?\s*([A-Z])\b", upper
    )
    if triggered and triggered[-1] in allowed:
        return triggered[-1]
    matches = [value for value in re.findall(r"\b([A-Z])\b", upper) if value in allowed]
    return matches[-1] if matches else "unknown"


def stable_case_id(dataset: str, question_id: str) -> str:
    digest = hashlib.sha1(f"{dataset}:{question_id}".encode("utf-8")).hexdigest()
    return f"{dataset}_{digest[:12]}"


def selection_hash(dataset: str, question_id: str) -> str:
    material = f"{HASH_NAMESPACE}:{dataset}:{question_id}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def json_line(row: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(row),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json_line(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def require_file(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"Empty {label}: {path}")
    return {"path": str(path.resolve()), "bytes": size}


def require_checkpoint(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Missing {label} checkpoint directory: {resolved}")
    weight = resolved / "model.safetensors"
    config = resolved / "config.json"
    weight_info = require_file(weight, f"{label} model weight")
    require_file(config, f"{label} model config")
    return {
        "path": str(resolved),
        "model_weight": weight_info,
        "config": str(config),
    }


def enforce_output_root(output_dir: Path) -> Path:
    resolved = output_dir.expanduser().resolve()
    try:
        resolved.relative_to(SCRIPT_DIR)
    except ValueError as exc:
        raise ValueError(
            f"Output directory must be inside {SCRIPT_DIR}; got {resolved}"
        ) from exc
    return resolved


def fixed_input_audit(reference_manifest: Path) -> dict[str, Any]:
    files: dict[str, Any] = {
        "reference_manifest": require_file(reference_manifest, "old160 reference manifest"),
        "datasets": {},
    }
    checkpoints: dict[str, Any] = {
        "base": require_checkpoint(BASE_MODEL, "Base"),
    }
    for spec in DATASETS:
        files["datasets"][spec.key] = {
            "test_parquet": require_file(spec.test_parquet, f"{spec.key} test parquet"),
            "train_parquet": require_file(spec.train_parquet, f"{spec.key} train parquet"),
            "ours_validation": require_file(
                spec.ours_validation, f"{spec.key} Ours validation_groups"
            ),
            "ttrv_validation": require_file(
                spec.ttrv_validation, f"{spec.key} TTRV validation_groups"
            ),
        }
        checkpoints[f"{spec.key}.ours"] = require_checkpoint(
            spec.ours_checkpoint, f"{spec.key} Ours"
        )
        checkpoints[f"{spec.key}.ttrv"] = require_checkpoint(
            spec.ttrv_checkpoint, f"{spec.key} TTRV"
        )
        for model_name, source in (
            ("ours", spec.ours_source_checkpoint),
            ("ttrv", spec.ttrv_source_checkpoint),
        ):
            if not source.expanduser().resolve().is_dir():
                raise FileNotFoundError(
                    f"Missing {spec.key} {model_name} source checkpoint: {source}"
                )
    return {"files": files, "runtime_checkpoints": checkpoints}


def build_parquet_cases(
    spec: DatasetSpec,
    max_test_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    test_table = pq.read_table(spec.test_parquet)
    train_rows = pq.read_table(spec.train_parquet).to_pylist()
    raw_test_rows = test_table.num_rows
    if max_test_rows > 0:
        test_table = test_table.slice(0, min(max_test_rows, raw_test_rows))
    test_rows = test_table.to_pylist()

    train_qids = {row_qid(row, index) for index, row in enumerate(train_rows)}
    train_images = {row_image_path(row) for row in train_rows}

    cases: list[dict[str, Any]] = []
    missing_images: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()
    for row_index, row in enumerate(test_rows):
        qid = row_qid(row, row_index)
        image_path = row_image_path(row)
        qid_overlap = qid in train_qids
        image_overlap = image_path in train_images
        if qid_overlap or image_overlap:
            if qid_overlap:
                exclusion_counts["train_qid_overlap"] += 1
            if image_overlap:
                exclusion_counts["train_image_overlap"] += 1
            exclusion_counts["excluded_union"] += 1
            continue
        if not Path(image_path).is_file() or Path(image_path).stat().st_size <= 0:
            missing_images.append(
                {"question_id": qid, "row_index": row_index, "image_path": image_path}
            )
            continue

        question = prompt_text(row)
        hits = [match.group(0).lower() for match in VISUAL_KEYWORD_PATTERN.finditer(question)]
        ground_truth = simple_choice_parse(row_ground_truth(row), spec.labels)
        if ground_truth == "unknown":
            raise ValueError(f"Invalid {spec.key} ground truth for {qid}: {row_ground_truth(row)!r}")
        case_id = stable_case_id(spec.key, qid)
        case = {
            "case_id": case_id,
            "dataset": spec.key,
            "question_id": qid,
            "row_index": row_index,
            "test_parquet": str(spec.test_parquet.resolve()),
            "train_parquet": str(spec.train_parquet.resolve()),
            "image_path": image_path,
            "question": question,
            "category": nested_get(row, "extra_info.mme_category"),
            "ground_truth": ground_truth,
            "parser": "legacy",
            "labels": spec.labels,
            "task": spec.task,
            "visual_keyword_score": len(hits),
            "visual_keyword_hits": hits,
            "selection_hash": selection_hash(spec.key, qid),
            "held_out_no_train_qid_or_image_overlap": True,
            "selection_policy": SELECTION_POLICY,
            "selection_correctness_blind": True,
            "models": {
                "base": {
                    "checkpoint": str(BASE_MODEL.resolve()),
                    "source_checkpoint": str(BASE_MODEL.resolve()),
                },
                "ttrv": {
                    "checkpoint": str(spec.ttrv_checkpoint.resolve()),
                    "source_checkpoint": str(spec.ttrv_source_checkpoint.resolve()),
                },
                "ours": {
                    "checkpoint": str(spec.ours_checkpoint.resolve()),
                    "source_checkpoint": str(spec.ours_source_checkpoint.resolve()),
                },
            },
        }
        cases.append(case)

    if missing_images:
        preview = missing_images[:10]
        raise FileNotFoundError(
            f"{spec.key} has {len(missing_images)} missing/empty images; first={preview}"
        )
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError(f"Duplicate stable case IDs in {spec.key}")

    stats = {
        "raw_test_rows": raw_test_rows,
        "scanned_test_rows": len(test_rows),
        "train_rows": len(train_rows),
        "train_qids": len(train_qids),
        "train_image_paths": len(train_images),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "held_out_rows": len(cases),
        "verified_image_paths": len(cases),
        "positive_visual_keyword_rows": sum(
            int(case["visual_keyword_score"] > 0) for case in cases
        ),
    }
    return cases, stats


def choose_shortlist(
    cases: list[dict[str, Any]],
    per_dataset_limit: int,
    visual_fraction: float,
) -> dict[str, Any]:
    if per_dataset_limit <= 0:
        raise ValueError("per_dataset_limit must be positive")
    if not 0.0 <= visual_fraction <= 1.0:
        raise ValueError("visual_fraction must be within [0, 1]")

    target = min(per_dataset_limit, len(cases))
    visual_quota = int(target * visual_fraction)
    hash_quota = target - visual_quota
    case_by_id = {str(case["case_id"]): case for case in cases}
    selection_rows = [
        {field: case.get(field) for field in SELECTION_ALLOWED_FIELDS}
        for case in cases
    ]
    visual_ranked = sorted(
        selection_rows,
        key=lambda case: (
            -int(case["visual_keyword_score"]),
            str(case["selection_hash"]),
            str(case["question_id"]),
        ),
    )
    visual_selection_rows = visual_ranked[:visual_quota]
    visual_ids = {str(case["case_id"]) for case in visual_selection_rows}
    hash_ranked = sorted(
        (case for case in selection_rows if str(case["case_id"]) not in visual_ids),
        key=lambda case: (str(case["selection_hash"]), str(case["question_id"])),
    )
    hash_selection_rows = hash_ranked[:hash_quota]

    if len(visual_selection_rows) + len(hash_selection_rows) != target:
        raise AssertionError("Shortlist quota accounting failed")

    for case in cases:
        case["shortlisted"] = False
        case["shortlist_source"] = None
        case["shortlist_source_rank"] = None
        case["shortlist_rank"] = None
    visual_selected = [case_by_id[str(row["case_id"])] for row in visual_selection_rows]
    hash_selected = [case_by_id[str(row["case_id"])] for row in hash_selection_rows]
    for source_rank, case in enumerate(visual_selected, start=1):
        case["shortlisted"] = True
        case["shortlist_source"] = "visual_keyword"
        case["shortlist_source_rank"] = source_rank
    for source_rank, case in enumerate(hash_selected, start=1):
        case["shortlisted"] = True
        case["shortlist_source"] = "sha256"
        case["shortlist_source_rank"] = source_rank
    selected = visual_selected + hash_selected
    for overall_rank, case in enumerate(selected, start=1):
        case["shortlist_rank"] = overall_rank

    return {
        "eligible_rows": len(cases),
        "target_rows": target,
        "visual_fraction": visual_fraction,
        "visual_quota": visual_quota,
        "hash_quota": hash_quota,
        "visual_selected": len(visual_selected),
        "hash_selected": len(hash_selected),
        "visual_selected_zero_score": sum(
            int(case["visual_keyword_score"] == 0) for case in visual_selected
        ),
    }


def vote_margin(counter: Mapping[str, Any]) -> int:
    counts = sorted((int(value) for value in counter.values()), reverse=True)
    if not counts:
        return 0
    return counts[0] - (counts[1] if len(counts) > 1 else 0)


def retained_record(
    row: Mapping[str, Any],
    labels: str,
    source_file: Path,
    source_step: int,
    source_line: int,
    source_occurrence: int,
    source_selection: str,
) -> dict[str, Any]:
    majority_raw = str(row.get("majority_prediction", ""))
    parsed = simple_choice_parse(majority_raw, labels)
    samples = row.get("samples", [])
    if not isinstance(samples, list):
        samples = []
    raw_response = ""
    for sample in samples:
        if isinstance(sample, Mapping) and str(sample.get("prediction", "")) == majority_raw:
            raw_response = str(sample.get("response_raw", ""))
            break
    if not raw_response:
        raw_response = str(
            row.get("single_response_raw", row.get("response_raw", "")) or ""
        )
    prediction_counter = row.get("prediction_counter", {})
    if not isinstance(prediction_counter, Mapping):
        prediction_counter = {}
    return {
        "raw_response": raw_response,
        "parsed_answer": parsed,
        "prediction_counter": dict(prediction_counter),
        "samples": samples,
        "source_file": str(source_file.resolve()),
        "source_step": source_step,
        "source_line": source_line,
        "source_occurrence": source_occurrence,
        "source_selection": source_selection,
        "vote_margin": vote_margin(prediction_counter),
        "selection_used_this_metadata": False,
    }


def load_validation_metadata(
    path: Path,
    labels: str,
    source_step: int,
    selection: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if selection not in {"first", "last", "only"}:
        raise ValueError(f"Unsupported validation selection={selection}")
    selected: dict[str, dict[str, Any]] = {}
    occurrences: Counter[str] = Counter()
    raw_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw_rows += 1
            row = json.loads(line)
            qid_value = row.get("index", row.get("question_id"))
            if qid_value is None:
                raise ValueError(f"Missing validation question ID at {path}:{line_number}")
            qid = normalize_qid(qid_value)
            occurrences[qid] += 1
            metadata = retained_record(
                row=row,
                labels=labels,
                source_file=path,
                source_step=source_step,
                source_line=line_number,
                source_occurrence=occurrences[qid],
                source_selection=selection,
            )
            if selection == "first":
                selected.setdefault(qid, metadata)
            else:
                selected[qid] = metadata

    if selection == "only" and any(count != 1 for count in occurrences.values()):
        duplicates = [qid for qid, count in occurrences.items() if count != 1]
        raise ValueError(
            f"Expected one validation row per ID in {path}; duplicates={duplicates[:10]}"
        )
    occurrence_histogram = Counter(occurrences.values())
    stats = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "raw_rows": raw_rows,
        "unique_question_ids": len(occurrences),
        "selected_question_ids": len(selected),
        "occurrence_histogram": {
            str(key): value for key, value in sorted(occurrence_histogram.items())
        },
        "selection": selection,
        "fixed_source_step": source_step,
    }
    return selected, stats


def base_retained_record() -> dict[str, Any]:
    return {
        "raw_response": "",
        "parsed_answer": "unknown",
        "prediction_counter": {},
        "samples": [],
        "source_file": None,
        "source_step": 0,
        "vote_margin": 0,
        "selection_used_this_metadata": False,
    }


def attach_retained_metadata(
    spec: DatasetSpec,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    ours, ours_stats = load_validation_metadata(
        path=spec.ours_validation,
        labels=spec.labels,
        source_step=spec.ours_step,
        selection=spec.ours_validation_selection,
    )
    ttrv, ttrv_stats = load_validation_metadata(
        path=spec.ttrv_validation,
        labels=spec.labels,
        source_step=spec.ttrv_step,
        selection=spec.ttrv_validation_selection,
    )

    missing_ours = [case["question_id"] for case in cases if case["question_id"] not in ours]
    missing_ttrv = [case["question_id"] for case in cases if case["question_id"] not in ttrv]
    if missing_ours or missing_ttrv:
        raise ValueError(
            f"Incomplete retained mapping for {spec.key}: "
            f"missing_ours={missing_ours[:10]} missing_ttrv={missing_ttrv[:10]}"
        )

    for case in cases:
        qid = str(case["question_id"])
        case["retained_predictions"] = {
            "base": base_retained_record(),
            "ttrv": ttrv[qid],
            "ours": ours[qid],
        }
        case["selection_gate"] = "correctness_blind"
        case["selection_reason"] = (
            "Held-out correctness-blind pool; shortlist selected before validation "
            "prediction files were opened. Retained predictions are metadata only."
        )
        case["validation_provenance"] = {
            "protocol": "final majority@8",
            "checkpoint_rule": (
                f"{spec.key} Ours step{spec.ours_step}; "
                f"TTRV step{spec.ttrv_step}"
            ),
            "ours_source": str(spec.ours_validation.resolve()),
            "ttrv_source": str(spec.ttrv_validation.resolve()),
            "ours_checkpoint_step": spec.ours_step,
            "ttrv_checkpoint_step": spec.ttrv_step,
            "held_out": True,
            "no_train_question_or_image_overlap": True,
            "correctness_used_for_selection": False,
            "retained_predictions_used_for_selection": False,
        }
    return {"ours": ours_stats, "ttrv": ttrv_stats}


REFERENCE_RETAINED_FIELDS = (
    "raw_response",
    "parsed_answer",
    "prediction_counter",
    "samples",
    "source_file",
    "source_step",
    "vote_margin",
)


def validate_reference_intersection(
    cases: list[dict[str, Any]],
    reference_manifest: Path,
    require_full_reference: bool,
) -> dict[str, Any]:
    reference_rows = read_jsonl(reference_manifest)
    reference = {str(row["case_id"]): row for row in reference_rows}
    current = {str(row["case_id"]): row for row in cases}
    intersection = sorted(set(reference) & set(current))
    missing_reference_ids = sorted(set(reference) - set(current))
    if require_full_reference and missing_reference_ids:
        raise ValueError(
            f"Full build omitted {len(missing_reference_ids)} old160 cases; "
            f"first={missing_reference_ids[:10]}"
        )

    mismatches: list[dict[str, Any]] = []
    scalar_fields = (
        "dataset",
        "question_id",
        "row_index",
        "image_path",
        "question",
        "ground_truth",
        "labels",
        "task",
    )
    for case_id in intersection:
        new = current[case_id]
        old = reference[case_id]
        for field in scalar_fields:
            if new.get(field) != old.get(field):
                mismatches.append(
                    {
                        "case_id": case_id,
                        "field": field,
                        "expected": old.get(field),
                        "actual": new.get(field),
                    }
                )
        if new.get("models") != old.get("models"):
            mismatches.append(
                {
                    "case_id": case_id,
                    "field": "models",
                    "expected": old.get("models"),
                    "actual": new.get("models"),
                }
            )
        old_base = old.get("retained_predictions", {}).get("base", {})
        new_base = new.get("retained_predictions", {}).get("base", {})
        for field in ("raw_response", "parsed_answer", "prediction_counter", "samples", "source_file", "source_step"):
            if new_base.get(field) != old_base.get(field):
                mismatches.append(
                    {
                        "case_id": case_id,
                        "field": f"retained_predictions.base.{field}",
                        "expected": old_base.get(field),
                        "actual": new_base.get(field),
                    }
                )
        for model_name in ("ours", "ttrv"):
            old_retained = old.get("retained_predictions", {}).get(model_name, {})
            new_retained = new.get("retained_predictions", {}).get(model_name, {})
            for field in REFERENCE_RETAINED_FIELDS:
                if new_retained.get(field) != old_retained.get(field):
                    mismatches.append(
                        {
                            "case_id": case_id,
                            "field": f"retained_predictions.{model_name}.{field}",
                            "expected": old_retained.get(field),
                            "actual": new_retained.get(field),
                        }
                    )
        if len(mismatches) >= 50:
            break
    if mismatches:
        raise ValueError(
            "Old160 retained/path mapping validation failed; "
            f"first mismatches={json.dumps(mismatches[:10], ensure_ascii=False)}"
        )
    return {
        "reference_manifest": str(reference_manifest.resolve()),
        "reference_sha256": sha256_file(reference_manifest),
        "reference_rows": len(reference_rows),
        "intersection_rows": len(intersection),
        "missing_reference_rows": len(missing_reference_ids),
        "retained_mapping_fields": list(REFERENCE_RETAINED_FIELDS),
        "mismatch_count": 0,
        "status": "PASS",
    }


def sort_full_manifest(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {spec.key: index for index, spec in enumerate(DATASETS)}
    return sorted(
        cases,
        key=lambda case: (
            order[str(case["dataset"])],
            str(case["question_id"]),
            str(case["case_id"]),
        ),
    )


def sort_attention_pool(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {spec.key: index for index, spec in enumerate(DATASETS)}
    return sorted(
        (case for case in cases if case["shortlisted"]),
        key=lambda case: (
            order[str(case["dataset"])],
            int(case["shortlist_rank"]),
            str(case["case_id"]),
        ),
    )


def select_cross_dataset_smoke(
    attention_pool: list[dict[str, Any]],
    per_dataset: int,
) -> list[tuple[str, int]]:
    if per_dataset <= 0:
        raise ValueError("smoke_per_dataset must be positive")
    smoke: list[tuple[str, int]] = []
    for spec in DATASETS:
        selection_rows = [
            {field: case.get(field) for field in SELECTION_ALLOWED_FIELDS}
            for case in attention_pool
            if case["dataset"] == spec.key
        ]
        ranked = sorted(
            selection_rows,
            key=lambda case: (
                -int(case["visual_keyword_score"]),
                str(case["selection_hash"]),
                str(case["question_id"]),
            ),
        )
        if len(ranked) < per_dataset:
            raise ValueError(
                f"{spec.key} attention pool has {len(ranked)} rows; "
                f"cannot select {per_dataset} smoke rows"
            )
        for dataset_rank, case in enumerate(ranked[:per_dataset], start=1):
            smoke.append((str(case["case_id"]), dataset_rank))
    return smoke


def materialize_cross_dataset_smoke(
    attention_pool: list[dict[str, Any]],
    selected: list[tuple[str, int]],
) -> list[dict[str, Any]]:
    case_by_id = {str(case["case_id"]): case for case in attention_pool}
    smoke: list[dict[str, Any]] = []
    for case_id, dataset_rank in selected:
        smoke_case = dict(case_by_id[case_id])
        smoke_case["smoke_selection_policy"] = SMOKE_SELECTION_POLICY
        smoke_case["smoke_dataset_rank"] = dataset_rank
        smoke_case["smoke_correctness_blind"] = True
        smoke.append(smoke_case)
    return smoke


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_line(row) + "\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_pool(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = enforce_output_root(Path(args.output_dir))
    reference_manifest = Path(args.reference_manifest).expanduser().resolve()
    input_audit = fixed_input_audit(reference_manifest)

    all_cases: list[dict[str, Any]] = []
    dataset_summaries: dict[str, Any] = {}
    cases_by_dataset: dict[str, list[dict[str, Any]]] = {}

    # Selection is deliberately completed before validation prediction files
    # are opened.  choose_shortlist receives no retained-prediction objects.
    for spec in DATASETS:
        cases, parquet_stats = build_parquet_cases(spec, args.max_test_rows)
        shortlist_stats = choose_shortlist(
            cases,
            per_dataset_limit=args.per_dataset_limit,
            visual_fraction=args.visual_fraction,
        )
        cases_by_dataset[spec.key] = cases
        all_cases.extend(cases)
        dataset_summaries[spec.key] = {
            "parquet": parquet_stats,
            "shortlist": shortlist_stats,
        }

    pre_metadata_attention_pool = sort_attention_pool(all_cases)
    smoke_selection = select_cross_dataset_smoke(
        pre_metadata_attention_pool, per_dataset=args.smoke_per_dataset
    )

    validation_summaries: dict[str, Any] = {}
    for spec in DATASETS:
        validation_summaries[spec.key] = attach_retained_metadata(
            spec, cases_by_dataset[spec.key]
        )

    full_manifest = sort_full_manifest(all_cases)
    attention_pool = sort_attention_pool(all_cases)
    cross_dataset_smoke = materialize_cross_dataset_smoke(
        attention_pool, selected=smoke_selection
    )
    if len({case["case_id"] for case in full_manifest}) != len(full_manifest):
        raise ValueError("Duplicate case IDs across datasets")
    if any(not case["held_out_no_train_qid_or_image_overlap"] for case in full_manifest):
        raise AssertionError("Non-held-out case reached full manifest")

    reference_summary = validate_reference_intersection(
        full_manifest,
        reference_manifest=reference_manifest,
        require_full_reference=args.max_test_rows == 0,
    )

    full_path = output_dir / "full_manifest.jsonl"
    pool_path = output_dir / "attention_pool.jsonl"
    smoke_path = output_dir / "cross_dataset_smoke.jsonl"
    summary_path = output_dir / "summary.json"
    output_hashes = {
        "full_manifest.jsonl": jsonl_digest(full_manifest),
        "attention_pool.jsonl": jsonl_digest(attention_pool),
        "cross_dataset_smoke.jsonl": jsonl_digest(cross_dataset_smoke),
    }
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "dry_run": bool(args.dry_run),
        "selection": {
            "policy": SELECTION_POLICY,
            "hash_namespace": HASH_NAMESPACE,
            "per_dataset_limit": args.per_dataset_limit,
            "visual_fraction": args.visual_fraction,
            "correctness_used": False,
            "ground_truth_used": False,
            "retained_predictions_used": False,
            "selection_completed_before_validation_files_opened": True,
            "allowed_fields": list(SELECTION_ALLOWED_FIELDS),
            "visual_sort": [
                "visual_keyword_score desc",
                "selection_hash asc",
                "question_id asc",
            ],
            "hash_sort": ["selection_hash asc", "question_id asc"],
            "smoke_policy": SMOKE_SELECTION_POLICY,
            "smoke_per_dataset": args.smoke_per_dataset,
            "smoke_correctness_used": False,
            "smoke_selection_completed_before_validation_files_opened": True,
        },
        "checkpoint_rule": {
            "MME": {"ours": 20, "ttrv": 20},
            "AI2D": {"ours": 20, "ttrv": 10},
        },
        "max_test_rows_per_dataset": args.max_test_rows,
        "full_manifest_rows": len(full_manifest),
        "attention_pool_rows": len(attention_pool),
        "cross_dataset_smoke_rows": len(cross_dataset_smoke),
        "datasets": dataset_summaries,
        "validation_metadata": validation_summaries,
        "integrity": {
            "unique_case_ids": len({case["case_id"] for case in full_manifest}),
            "verified_image_paths": len(full_manifest),
            "runtime_checkpoint_count": len(input_audit["runtime_checkpoints"]),
            "old160_reference": reference_summary,
        },
        "inputs": input_audit,
        "outputs": {
            "full_manifest": str(full_path),
            "attention_pool": str(pool_path),
            "cross_dataset_smoke": str(smoke_path),
            "summary": str(summary_path),
            "sha256": output_hashes,
            "persisted": not args.dry_run,
        },
    }

    if args.dry_run:
        return summary

    targets = (full_path, pool_path, smoke_path, summary_path)
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing outputs without --allow-overwrite: {existing}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(full_path, full_manifest)
    atomic_write_jsonl(pool_path, attention_pool)
    atomic_write_jsonl(smoke_path, cross_dataset_smoke)
    # Verify bytes on disk match the pre-write deterministic digests.
    actual_hashes = {
        "full_manifest.jsonl": sha256_file(full_path),
        "attention_pool.jsonl": sha256_file(pool_path),
        "cross_dataset_smoke.jsonl": sha256_file(smoke_path),
    }
    if actual_hashes != output_hashes:
        raise AssertionError(
            f"Output digest mismatch expected={output_hashes} actual={actual_hashes}"
        )
    atomic_write_json(summary_path, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help=f"Output directory; must be inside {SCRIPT_DIR}",
    )
    parser.add_argument("--per-dataset-limit", type=int, default=200)
    parser.add_argument("--visual-fraction", type=float, default=0.5)
    parser.add_argument("--smoke-per-dataset", type=int, default=1)
    parser.add_argument(
        "--reference-manifest",
        default=str(DEFAULT_REFERENCE_MANIFEST),
        help="Old frozen160 manifest used only for post-selection mapping validation",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=0,
        help="CPU smoke option: inspect only the first N test rows per dataset; 0 means all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate entirely in memory; write no persistent outputs",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing the three named output files in an existing output directory",
    )
    args = parser.parse_args(argv)
    if args.max_test_rows < 0:
        parser.error("--max-test-rows must be >= 0")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_pool(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
