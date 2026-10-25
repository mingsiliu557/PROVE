#!/usr/bin/env python3
"""InternVL3 answer-to-image attention extraction and visualization.

This module is intentionally independent from the training path.  It provides
five subcommands:

* ``merge``: merge verl FSDP actor shards into a local Hugging Face checkpoint;
* ``select``: join retained greedy outputs and build a candidate-case manifest;
* ``probe``: cheaply rerun exact greedy decoding before attention extraction;
* ``extract``: rerun exact greedy decoding and save answer/image attentions;
* ``render``: select one shared layer/temperature setting and render galleries.

The main visualization uses decoder self-attention from generated answer states
to the ``<IMG_CONTEXT>`` positions.  A causally shifted decision-query variant
is saved beside it because the state at position t predicts token t+1.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


IMAGE_CONTEXT_TOKEN = "<IMG_CONTEXT>"
DEFAULT_LAYER_WINDOWS = {
    "shallow": (6, 7, 8),
    "middle": (13, 14, 15),
    "deep": (20, 21, 22),
}
DEFAULT_TEMPERATURES = (1.0, 0.7, 0.5, 0.3)
MODEL_ORDER = ("base", "ttrv", "ours")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, default=_json_default) + "\n")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_npz(path: str | Path, payload: Mapping[str, np.ndarray]) -> None:
    """Atomically write a compressed NPZ without exposing a partial archive."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp.npz",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **payload)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_case_id(dataset: str, question_id: str) -> str:
    raw = f"{dataset}:{question_id}".encode("utf-8")
    return f"{dataset}_{hashlib.sha1(raw).hexdigest()[:12]}"


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return cleaned[:120] or "item"


def _find_actor_dir(checkpoint: str | Path) -> Path:
    root = Path(checkpoint).expanduser().resolve()
    candidates = (root / "actor", root)
    for candidate in candidates:
        if any(candidate.glob("model_world_size_*_rank_0.pt")):
            return candidate
    raise FileNotFoundError(f"No FSDP actor shards under {root} or {root / 'actor'}")


def _world_size(actor_dir: Path) -> int:
    matches = []
    for path in actor_dir.glob("model_world_size_*_rank_0.pt"):
        match = re.fullmatch(r"model_world_size_(\d+)_rank_0\.pt", path.name)
        if match:
            matches.append(int(match.group(1)))
    if len(set(matches)) != 1:
        raise ValueError(f"Could not determine one world size in {actor_dir}: {matches}")
    return matches[0]


def _merge_placement(tensors: Sequence[Any], placement: Any):
    import torch

    if placement.is_replicate():
        return tensors[0]
    if placement.is_partial():
        raise NotImplementedError("Partial DTensor placement is not supported")
    if placement.is_shard():
        return torch.cat(list(tensors), dim=placement.dim).contiguous()
    raise ValueError(f"Unsupported DTensor placement: {placement}")


def merge_fsdp_checkpoint(
    checkpoint: str | Path,
    base_model: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Merge a verl FSDP actor checkpoint and save an InternVL HF directory."""

    import torch
    from safetensors.torch import save_file

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        from torch.distributed._tensor import DTensor

    actor_dir = _find_actor_dir(checkpoint)
    world_size = _world_size(actor_dir)
    destination = Path(output_dir).expanduser().resolve()
    weight_path = destination / "model.safetensors"
    manifest_path = destination / "merge_manifest.json"
    if weight_path.exists() and manifest_path.exists() and not overwrite:
        manifest = read_json(manifest_path)
        manifest["reused"] = True
        return manifest

    destination.mkdir(parents=True, exist_ok=True)
    shard_paths = [actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt" for rank in range(world_size)]
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing FSDP shards: {missing}")

    def load_shard(path: Path):
        return torch.load(path, map_location="cpu", weights_only=False)

    with ThreadPoolExecutor(max_workers=min(world_size, 8)) as executor:
        shards = list(executor.map(load_shard, shard_paths))
    keys = set(shards[0])
    if any(set(shard) != keys for shard in shards[1:]):
        raise ValueError("FSDP shards do not contain identical parameter keys")

    merged: dict[str, torch.Tensor] = {}
    for key in sorted(keys):
        values = [shard.pop(key) for shard in shards]
        first = values[0]
        if isinstance(first, DTensor):
            placements = tuple(first.placements)
            if len(placements) != 1:
                raise NotImplementedError(f"Only 1-D FSDP is supported for {key}: {placements}")
            local_tensors = [value._local_tensor.detach().cpu().bfloat16() for value in values]
            tensor = _merge_placement(local_tensors, placements[0])
        else:
            local_tensors = [value.detach().cpu().bfloat16() for value in values]
            if world_size == 1:
                tensor = local_tensors[0]
            elif all(value.shape == first.shape for value in values) and first.ndim == 0:
                tensor = local_tensors[0]
            else:
                tensor = torch.cat(local_tensors, dim=0)
        merged[key] = tensor.contiguous()
    del shards

    base = Path(base_model).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Base model must be a local directory: {base}")
    for source in base.iterdir():
        if source.name.startswith("model") and source.suffix in {".bin", ".safetensors"}:
            continue
        if source.name.endswith(".index.json"):
            continue
        target = destination / source.name
        if source.is_file():
            shutil.copy2(source, target)
        elif source.is_dir() and source.name not in {".git"}:
            if target.exists() and overwrite:
                shutil.rmtree(target)
            if not target.exists():
                shutil.copytree(source, target)

    temporary = destination / "model.safetensors.tmp"
    save_file(merged, str(temporary), metadata={"format": "pt"})
    os.replace(temporary, weight_path)
    del merged

    manifest = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "actor_dir": str(actor_dir),
        "base_model": str(base),
        "output_dir": str(destination),
        "world_size": world_size,
        "shards": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in shard_paths
        ],
        "weight": {
            "path": str(weight_path),
            "bytes": weight_path.stat().st_size,
            "sha256": sha256_file(weight_path),
        },
        "reused": False,
    }
    write_json(manifest_path, manifest)
    return manifest


def nested_get(row: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = row
    for key in dotted_key.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


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
RESPONSE_KEYS = (
    "raw_response",
    "response_raw",
    "response",
    "model_response",
    "output",
    "solution",
    "prediction_raw",
    "prediction",
)
PARSED_KEYS = ("parsed_answer", "normalized_answer", "model_answer", "prediction")
STEP_KEYS = ("step", "global_step", "training_step", "trainer_step")


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


def _prediction_records(path: str | Path, desired_step: str | int = "final") -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    expanded: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row.get("records"), list):
            for child in row["records"]:
                if isinstance(child, dict):
                    expanded.append({**row, **child, "records": None})
        elif isinstance(row.get("samples"), list) and row.get("question_id") is not None:
            samples = row["samples"]
            if samples and isinstance(samples[0], dict):
                expanded.append({**row, **samples[0]})
            else:
                expanded.append(row)
        else:
            expanded.append(row)

    by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in expanded:
        qid = first_present(row, QID_KEYS)
        if qid is None:
            continue
        by_qid[normalize_qid(qid)].append(row)
    if not by_qid:
        raise ValueError(f"No question IDs found in {path}; checked {QID_KEYS}")

    selected: dict[str, dict[str, Any]] = {}
    for qid, candidates in by_qid.items():
        def step_value(item: Mapping[str, Any]) -> int:
            value = first_present(item, STEP_KEYS)
            if value is None:
                return -1
            match = re.search(r"\d+", str(value))
            return int(match.group()) if match else -1

        if desired_step == "first":
            row = candidates[0]
        elif desired_step in {"final", "last"}:
            # Break equal/missing-step ties by append order. Validation JSONL is
            # append-only, so the last occurrence is the latest checkpoint.
            row = max(enumerate(candidates), key=lambda pair: (step_value(pair[1]), pair[0]))[1]
        else:
            target = int(desired_step)
            matches = [item for item in candidates if step_value(item) == target]
            if not matches:
                continue
            row = matches[-1]
        response = first_present(row, RESPONSE_KEYS)
        parsed = first_present(row, PARSED_KEYS)
        selected[qid] = {
            "raw_response": "" if response is None else str(response),
            "parsed_answer": None if parsed is None else str(parsed),
            "source_step": step_value(row),
            "source_file": str(Path(path).resolve()),
            "source_row": row,
        }
    return selected


def _row_qid(row: Mapping[str, Any], fallback: int) -> str:
    value = first_present(row, QID_KEYS)
    return normalize_qid(fallback if value is None else value)


def _row_ground_truth(row: Mapping[str, Any]) -> str:
    for key in ("reward_model.ground_truth", "ground_truth", "answer", "extra_info.answer"):
        value = nested_get(row, key)
        if value is not None:
            return str(value).strip()
    raise ValueError(f"No ground truth in row keys={sorted(row)}")


def _row_image_path(row: Mapping[str, Any]) -> str:
    images = row.get("images")
    if isinstance(images, (list, tuple)) and images:
        first = images[0]
        if isinstance(first, Mapping):
            value = first.get("image") or first.get("path")
        else:
            value = first
        if value:
            return str(value)
    value = nested_get(row, "extra_info.image_path")
    if value:
        return str(value)
    raise ValueError("No image path in dataset row")


def _prompt_text(row: Mapping[str, Any]) -> str:
    messages = row.get("prompt")
    if not isinstance(messages, list):
        return str(messages or "")
    pieces = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content", "")
        pieces.append(str(content).replace("<image>", "").strip())
    return "\n".join(piece for piece in pieces if piece)


def _simple_choice_parse(text: str, labels: str) -> str:
    allowed = tuple(re.findall(r"[A-Z]", labels.upper().replace("-", "")))
    if labels.upper() in {"A-D", "ABCD"}:
        allowed = tuple("ABCD")
    elif labels.upper() in {"A-F", "ABCDEF"}:
        allowed = tuple("ABCDEF")
    elif labels.upper() == "AB":
        allowed = tuple("AB")
    upper = str(text or "").strip().upper()
    if upper in allowed:
        return upper
    trigger = re.findall(r"(?:ANSWER|CHOICE|OPTION)\s*(?:IS|:)?\s*([A-Z])\b", upper)
    if trigger and trigger[-1] in allowed:
        return trigger[-1]
    matches = re.findall(r"\b([A-Z])\b", upper)
    matches = [match for match in matches if match in allowed]
    return matches[-1] if matches else "unknown"


def _official_choice_parse(
    response: str,
    row: Mapping[str, Any],
    parser: str,
    labels: str,
    fallback_task: str,
) -> str:
    """Run the repository parser used by TTRL validation."""

    repo_root = Path(__file__).resolve().parents[1]
    verl_root = repo_root / "verl"
    if str(verl_root) not in sys.path:
        sys.path.insert(0, str(verl_root))
    from verl.utils.reward_score.ttrl.ttt_metrics import (
        _extract_answers,
        _normalize_choice_answer,
    )

    data_source = str(row.get("data_source", ""))
    task_by_source = {
        "GPQA-TTT": "gpqa",
        "MATH-TTT": "math",
        "AIME-TTT": "math",
        "AMC-TTT": "math",
        "data/AIME-TTT": "math",
        "bbox": "bbox",
        "VQA-DA-TTT": "vqa_da",
        "VisualSimpleQA-TTT": "vqa_da",
        "AOKVQA-DA-TTT": "vqa_da",
        "OCR-TTT": "ocr",
        "OCRBench-TTT": "ocr",
    }
    task = task_by_source.get(data_source, fallback_task)
    ground_truth = _row_ground_truth(row)
    parsed = _extract_answers(
        task,
        [response],
        ground_truth,
        extra_info=[row.get("extra_info", {})],
        answer_parse_mode=parser,
        choice_labels=labels,
    )[0]
    return _normalize_choice_answer(parsed, choice_labels=labels)


def _parsed_prediction(record: Mapping[str, Any], labels: str) -> str:
    parsed = record.get("parsed_answer")
    parsed_value = _simple_choice_parse("" if parsed is None else str(parsed), labels)
    if parsed_value != "unknown":
        return parsed_value
    return _simple_choice_parse(str(record.get("raw_response", "")), labels)


def select_candidates(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Build a candidate manifest from parquet data and retained greedy JSONL."""

    import pyarrow.parquet as pq

    config = read_json(config_path)
    base_model = str(Path(config["base_model"]).expanduser().resolve())
    per_dataset_limit = int(config.get("per_dataset_limit", 20))
    require_held_out = bool(config.get("exclude_train", True))
    all_candidates: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"base_model": base_model, "datasets": {}}

    for spec in config["datasets"]:
        key = str(spec["key"])
        labels = str(spec["labels"])
        test_path = Path(spec["test_parquet"]).expanduser().resolve()
        rows = pq.read_table(test_path).to_pylist()
        train_qids: set[str] = set()
        train_image_paths: set[str] = set()
        train_path = spec.get("train_parquet")
        if require_held_out and train_path:
            train_rows = pq.read_table(Path(train_path).expanduser().resolve()).to_pylist()
            train_qids = {_row_qid(row, index) for index, row in enumerate(train_rows)}
            train_image_paths = {_row_image_path(row) for row in train_rows}

        predictions = {
            model_name: _prediction_records(spec["prediction_files"][model_name], spec.get("prediction_step", {}).get(model_name, "final"))
            for model_name in MODEL_ORDER
        }
        qid_match_mode = str(spec.get("qid_match_mode", "exact"))
        if qid_match_mode == "numeric_suffix":
            remapped = {}
            for model_name, model_predictions in predictions.items():
                normalized = {}
                for prediction_qid, record in model_predictions.items():
                    match = re.search(r"(\d+)$", prediction_qid)
                    if match is None:
                        continue
                    lookup_key = match.group(1)
                    if lookup_key in normalized:
                        raise ValueError(f"Duplicate numeric question ID {lookup_key} in {spec['prediction_files'][model_name]}")
                    normalized[lookup_key] = record
                remapped[model_name] = normalized
            predictions = remapped
        elif qid_match_mode != "exact":
            raise ValueError(f"Unsupported qid_match_mode={qid_match_mode}")
        found = 0
        eligible = 0
        excluded_train = 0
        prefilter_mode = str(spec.get("prefilter_mode", "historical_pattern"))
        for row_index, row in enumerate(rows):
            qid = _row_qid(row, row_index)
            if qid in train_qids or _row_image_path(row) in train_image_paths:
                excluded_train += 1
                continue
            lookup_qid = qid
            if qid_match_mode == "numeric_suffix":
                match = re.search(r"(\d+)$", qid)
                if match is None:
                    continue
                lookup_qid = match.group(1)
            if any(lookup_qid not in predictions[name] for name in MODEL_ORDER):
                continue
            found += 1
            gt = _simple_choice_parse(_row_ground_truth(row), labels)
            parsed = {name: _parsed_prediction(predictions[name][lookup_qid], labels) for name in MODEL_ORDER}
            allow_proxy_invalid = bool(spec.get("allow_proxy_invalid", False))
            require_ttrv_wrong = bool(spec.get("prefilter_requires_ttrv_wrong", True))
            if gt == "unknown":
                continue
            if prefilter_mode == "all":
                eligible += 1
            elif prefilter_mode == "historical_pattern":
                if not allow_proxy_invalid and any(value == "unknown" for value in parsed.values()):
                    continue
                if parsed["base"] == gt or parsed["ours"] != gt:
                    continue
                if require_ttrv_wrong and parsed["ttrv"] == gt:
                    continue
                eligible += 1
            else:
                raise ValueError(f"Unsupported prefilter_mode={prefilter_mode}")
            question = _prompt_text(row)
            visual_keywords = len(
                re.findall(
                    r"\b(color|colour|left|right|above|below|behind|front|how many|count|where|which object|shape|wearing|holding)\b",
                    question,
                    flags=re.IGNORECASE,
                )
            )
            case = {
                "case_id": stable_case_id(key, qid),
                "dataset": key,
                "question_id": qid,
                "row_index": row_index,
                "test_parquet": str(test_path),
                "train_parquet": None if not train_path else str(Path(train_path).expanduser().resolve()),
                "image_path": _row_image_path(row),
                "question": question,
                "ground_truth": gt,
                "parser": str(spec.get("parser", "legacy")),
                "labels": labels,
                "task": str(spec.get("task", key)),
                "visual_keyword_score": visual_keywords,
                "retained_predictions": {
                    name: {
                        "raw_response": predictions[name][lookup_qid]["raw_response"],
                        "parsed_answer": parsed[name],
                        "source_file": predictions[name][lookup_qid]["source_file"],
                        "source_step": predictions[name][lookup_qid]["source_step"],
                    }
                    for name in MODEL_ORDER
                },
                "models": {
                    "base": {"checkpoint": base_model, "source_checkpoint": base_model},
                    "ttrv": {
                        "checkpoint": str(Path(spec["checkpoints"]["ttrv"]).expanduser().resolve()),
                        "source_checkpoint": str(Path(spec["source_checkpoints"]["ttrv"]).expanduser().resolve()),
                    },
                    "ours": {
                        "checkpoint": str(Path(spec["checkpoints"]["ours"]).expanduser().resolve()),
                        "source_checkpoint": str(Path(spec["source_checkpoints"]["ours"]).expanduser().resolve()),
                    },
                },
            }
            all_candidates.append(case)

        dataset_candidates = [case for case in all_candidates if case["dataset"] == key]
        dataset_candidates.sort(key=lambda item: (-item["visual_keyword_score"], item["question_id"]))
        keep_ids = {item["case_id"] for item in dataset_candidates[:per_dataset_limit]}
        for case in dataset_candidates:
            case["shortlisted"] = case["case_id"] in keep_ids
        summary["datasets"][key] = {
            "test_rows": len(rows),
            "excluded_train_rows": excluded_train,
            "joined_rows": found,
            "eligible_rows": eligible,
            "shortlisted_rows": min(eligible, per_dataset_limit),
        }

    all_candidates.sort(key=lambda item: (item["dataset"], not item["shortlisted"], -item["visual_keyword_score"], item["question_id"]))
    write_jsonl(output_path, all_candidates)
    summary["candidate_manifest"] = str(Path(output_path).resolve())
    summary["total_eligible"] = len(all_candidates)
    summary["total_shortlisted"] = sum(bool(item.get("shortlisted")) for item in all_candidates)
    write_json(Path(output_path).with_suffix(".summary.json"), summary)
    return summary


def _build_messages(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    messages = copy.deepcopy(row.get("prompt"))
    if not isinstance(messages, list):
        raise ValueError("Dataset row prompt must be a message list")
    has_images = bool(row.get("images"))
    if has_images:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, str):
                continue
            segments = [segment for segment in re.split(r"(<image>|<video>)", content) if segment]
            converted = []
            for segment in segments:
                if segment == "<image>":
                    converted.append({"type": "image"})
                elif segment == "<video>":
                    converted.append({"type": "video"})
                else:
                    converted.append({"type": "text", "text": segment})
            message["content"] = converted
    return messages


def _load_pil_images(row: Mapping[str, Any]):
    from PIL import Image

    result = []
    for value in row.get("images") or []:
        if isinstance(value, Mapping):
            value = value.get("image") or value.get("path")
        if not value:
            continue
        text = str(value)
        if text.startswith("file://"):
            text = text[7:]
        with Image.open(text) as image:
            result.append(image.convert("RGB").copy())
    return result


def _prepare_inputs(processor: Any, row: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    messages = _build_messages(row)
    raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    raw_prompt_for_processor = raw_prompt.replace("<image>", IMAGE_CONTEXT_TOKEN)
    images = _load_pil_images(row)
    batch = dict(processor(text=[raw_prompt_for_processor], images=images or None, return_tensors="pt"))
    image_processor = processor.image_processor
    batch["_attention_viz_original_sizes"] = [list(image.size) for image in images]
    batch["_attention_viz_tiling"] = {
        "min_patches": int(image_processor.min_patches),
        "max_patches": int(image_processor.max_patches),
        "use_thumbnail": True,
    }
    return batch, raw_prompt


def _move_inputs(batch: Mapping[str, Any], device: Any, dtype: Any) -> dict[str, Any]:
    import torch

    result = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            result[key] = value
        elif value.is_floating_point():
            result[key] = value.to(device=device, dtype=dtype)
        else:
            result[key] = value.to(device=device)
    return result


def _extract_dataset_row(case: Mapping[str, Any]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(case["test_parquet"])
    row_index = int(case["row_index"])
    if not 0 <= row_index < table.num_rows:
        raise IndexError(f"row_index={row_index} outside {table.num_rows}")
    row = table.slice(row_index, 1).to_pylist()[0]
    qid = _row_qid(row, row_index)
    if qid != str(case["question_id"]):
        raise ValueError(f"Question ID drift for {case['case_id']}: manifest={case['question_id']} parquet={qid}")
    return row


def _load_model_and_processor(checkpoint: str | Path, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    repo_root = Path(__file__).resolve().parents[1]
    verl_root = repo_root / "verl"
    if str(verl_root) not in sys.path:
        sys.path.insert(0, str(verl_root))
    from verl.utils.tokenizer import hf_processor

    path = str(Path(checkpoint).expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    processor = hf_processor(path, trust_remote_code=True)
    if processor is None:
        raise RuntimeError(f"verl hf_processor returned None for {path}")
    model = AutoModel.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
    )
    model.config.llm_config._attn_implementation = "eager"
    model.language_model.config._attn_implementation = "eager"
    model.language_model.config.output_attentions = True
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMAGE_CONTEXT_TOKEN)
    model.eval().to(device)
    return model, tokenizer, processor


def _stop_token_ids(tokenizer: Any) -> set[int]:
    values = {tokenizer.eos_token_id}
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id >= 0:
            values.add(int(token_id))
    return {int(value) for value in values if value is not None and int(value) >= 0}


def _manual_greedy(
    model: Any,
    tokenizer: Any,
    batch: Mapping[str, Any],
    max_new_tokens: int,
) -> tuple[list[int], str]:
    """Deterministic greedy decode with the same InternVL visual embeddings."""

    import torch

    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask", torch.ones_like(input_ids))
    pixel_values = batch.get("pixel_values")
    if pixel_values is None:
        raise ValueError("InternVL input has no pixel_values")
    with torch.inference_mode():
        visual = model.extract_feature(pixel_values)
        embeds = model.language_model.get_input_embeddings()(input_ids).clone()
        selected = input_ids.eq(model.img_context_token_id)
        expected = int(selected.sum().item())
        actual = int(visual.shape[0] * visual.shape[1])
        if expected != actual:
            raise ValueError(f"Visual token mismatch: prompt={expected}, encoder={actual}, pixel_values={tuple(pixel_values.shape)}")
        embeds[selected] = visual.reshape(-1, visual.shape[-1]).to(embeds.dtype)

        generated: list[int] = []
        past = None
        next_embeds = embeds
        current_mask = attention_mask
        stop_ids = _stop_token_ids(tokenizer)
        for _ in range(max_new_tokens):
            outputs = model.language_model(
                inputs_embeds=next_embeds,
                attention_mask=current_mask,
                past_key_values=past,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            token_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
            if token_id in stop_ids:
                break
            generated.append(token_id)
            past = outputs.past_key_values
            token_tensor = torch.tensor([[token_id]], device=input_ids.device)
            next_embeds = model.language_model.get_input_embeddings()(token_tensor)
            current_mask = torch.cat(
                [current_mask, torch.ones((1, 1), dtype=current_mask.dtype, device=current_mask.device)], dim=1
            )
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return generated, response


def _attention_windows(
    model: Any,
    batch: Mapping[str, Any],
    generated_ids: Sequence[int],
    layer_windows: Mapping[str, Sequence[int]],
    save_individual_layers: bool = False,
    save_first_decision_heads: bool = False,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    import torch

    input_ids = batch["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise NotImplementedError(f"Expected one input sequence, got input_ids shape={tuple(input_ids.shape)}")
    prompt_length = input_ids.shape[1]
    if prompt_length <= 0:
        raise ValueError("Prompt has no tokens")
    empty_generation = not generated_ids
    answer_tensor = torch.tensor([list(generated_ids)], dtype=input_ids.dtype, device=input_ids.device)
    full_ids = torch.cat([input_ids, answer_tensor], dim=1)
    prompt_mask = batch.get("attention_mask", torch.ones_like(input_ids))
    if prompt_mask.shape != input_ids.shape:
        raise ValueError(
            f"attention_mask shape={tuple(prompt_mask.shape)} does not match input_ids shape={tuple(input_ids.shape)}"
        )
    valid_prompt_positions = prompt_mask[0].ne(0).nonzero(as_tuple=False).flatten()
    if valid_prompt_positions.numel() == 0:
        raise ValueError("Prompt attention_mask has no valid tokens")
    first_decision_index = int(valid_prompt_positions[-1].item())
    full_mask = torch.cat(
        [prompt_mask, torch.ones_like(answer_tensor, dtype=prompt_mask.dtype, device=prompt_mask.device)], dim=1
    )
    forward_batch = {
        "pixel_values": batch["pixel_values"],
        "input_ids": full_ids,
        "attention_mask": full_mask,
        "image_flags": batch.get(
            "image_flags",
            torch.ones((batch["pixel_values"].shape[0], 1), dtype=torch.long, device=input_ids.device),
        ),
        "use_cache": False,
        "output_attentions": True,
        "return_dict": True,
    }
    with torch.inference_mode():
        outputs = model(**forward_batch)
    if not outputs.attentions or any(value is None for value in outputs.attentions):
        raise RuntimeError("Model returned no attention weights; eager attention is required")

    image_positions = full_ids[0].eq(model.img_context_token_id).nonzero(as_tuple=False).flatten()
    if image_positions.numel() == 0:
        raise ValueError("No <IMG_CONTEXT> positions found")
    answer_positions = torch.arange(prompt_length, prompt_length + len(generated_ids), device=input_ids.device)
    decision_positions = answer_positions - 1
    # decision_query preserves the original implementation for backward
    # compatibility: it averages the states immediately preceding every
    # generated answer token. first_decision_query is the comparable
    # causal state used by the Obs3 validation case study. It is always the
    # final valid prompt position, before any checkpoint-specific answer token
    # has been observed, so response length/content cannot change the query set.
    first_decision_position = torch.tensor([first_decision_index], device=input_ids.device)
    if empty_generation:
        # A model may immediately emit a stop token. The prompt-end state is
        # still a real causal query, but answer_state and decision_query have
        # no valid generated positions and must not be fabricated.
        query_modes = (("first_decision_query", first_decision_position),)
    else:
        query_modes = (
            ("answer_state", answer_positions),
            ("decision_query", decision_positions),
            ("first_decision_query", first_decision_position),
        )
    result: dict[str, dict[str, np.ndarray]] = {mode: {} for mode, _ in query_modes}
    head_mode = "first_decision_query_heads"
    if save_first_decision_heads:
        result[head_mode] = {}
    for window_name, layer_indices in layer_windows.items():
        layers = []
        for layer_index in layer_indices:
            if not 0 <= int(layer_index) < len(outputs.attentions):
                raise IndexError(f"Layer {layer_index} outside 0..{len(outputs.attentions) - 1}")
            layers.append(outputs.attentions[int(layer_index)][0].float())
        stacked = torch.stack(layers, dim=0)  # layer, head, query, key
        for mode, positions in query_modes:
            cross = stacked[:, :, positions, :][..., image_positions]
            vector = cross.mean(dim=(0, 1, 2))
            result[mode][window_name] = vector.detach().cpu().numpy().astype(np.float32)
    if save_individual_layers or save_first_decision_heads:
        for layer_index, layer_attention in enumerate(outputs.attentions):
            layer = layer_attention[0].float()
            if save_individual_layers:
                for mode, positions in query_modes:
                    cross = layer[:, positions, :][..., image_positions]
                    vector = cross.mean(dim=(0, 1))
                    result[mode][f"layer_{layer_index:02d}"] = (
                        vector.detach().cpu().numpy().astype(np.float32)
                    )
            if save_first_decision_heads:
                head_vectors = layer[:, first_decision_position, :][..., image_positions].squeeze(1)
                for head_index, vector in enumerate(head_vectors):
                    result[head_mode][f"layer_{layer_index:02d}_head_{head_index:02d}"] = (
                        vector.detach().cpu().numpy().astype(np.float32)
                    )
    num_layers = len(outputs.attentions)
    num_heads = int(outputs.attentions[0].shape[1])
    metadata = {
        "prompt_length": int(prompt_length),
        "answer_token_count": len(generated_ids),
        "empty_generation": bool(empty_generation),
        "available_query_modes": [mode for mode, _ in query_modes],
        "prompt_valid_token_count": int(valid_prompt_positions.numel()),
        "prompt_padding_token_count": int(prompt_length - valid_prompt_positions.numel()),
        "first_decision_position_source": "attention_mask_last_valid_prompt_token",
        "full_length": int(full_ids.shape[1]),
        "image_token_count": int(image_positions.numel()),
        "image_position_start": int(image_positions.min().item()),
        "image_position_end": int(image_positions.max().item()) + 1,
        "image_positions_contiguous": bool(
            torch.equal(
                image_positions,
                torch.arange(image_positions.min(), image_positions.max() + 1, device=image_positions.device),
            )
        ),
        "answer_positions": answer_positions.detach().cpu().tolist(),
        "decision_positions": decision_positions.detach().cpu().tolist(),
        "first_decision_position": int(first_decision_position.item()),
        "num_layers": num_layers,
        "num_heads": num_heads,
        "saved_individual_layers": list(range(num_layers)) if save_individual_layers else [],
        "saved_first_decision_heads": bool(save_first_decision_heads),
        "saved_first_decision_head_layers": list(range(num_layers)) if save_first_decision_heads else [],
        "saved_first_decision_head_indices": list(range(num_heads)) if save_first_decision_heads else [],
        "first_decision_head_key_format": (
            f"{head_mode}__layer_{{layer:02d}}_head_{{head:02d}}" if save_first_decision_heads else None
        ),
    }
    del outputs
    return result, metadata


def _token_maps_to_grid(vectors: Mapping[str, Mapping[str, np.ndarray]], batch: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    import cv2
    from transformers.models.got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas

    pixel_values = batch["pixel_values"]
    tile_count = int(pixel_values.shape[0])
    first_vector = next(iter(next(iter(vectors.values())).values()))
    token_count = int(first_vector.shape[0])
    if token_count % tile_count != 0:
        raise ValueError(f"image tokens={token_count} not divisible by tiles={tile_count}")
    tokens_per_tile = token_count // tile_count
    side = int(round(math.sqrt(tokens_per_tile)))
    if side * side != tokens_per_tile:
        raise ValueError(f"tokens_per_tile={tokens_per_tile} is not a square grid")

    original_sizes = batch.get("_attention_viz_original_sizes", [])
    if len(original_sizes) != 1:
        raise NotImplementedError(f"Expected exactly one source image, got {len(original_sizes)}")
    original_width, original_height = (int(value) for value in original_sizes[0])
    tiling = batch.get("_attention_viz_tiling", {})
    tile_height, tile_width = (int(value) for value in pixel_values.shape[-2:])
    num_columns, num_rows = get_optimal_tiled_canvas(
        (original_height, original_width),
        (tile_height, tile_width),
        int(tiling.get("min_patches", 1)),
        int(tiling.get("max_patches", 12)),
    )
    num_local_tiles = int(num_columns * num_rows)
    has_thumbnail = bool(tiling.get("use_thumbnail", True) and num_local_tiles != 1)
    expected_tiles = num_local_tiles + int(has_thumbnail)
    if tile_count != expected_tiles:
        raise ValueError(
            f"Processor tile reconstruction mismatch: actual={tile_count}, expected={expected_tiles}, "
            f"layout={num_columns}x{num_rows}, thumbnail={has_thumbnail}"
        )

    grids: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for mode, windows in vectors.items():
        for window, vector in windows.items():
            tile_grids = vector.reshape(tile_count, side, side)
            rows = []
            for row_index in range(num_rows):
                start = row_index * num_columns
                rows.append(np.concatenate(tile_grids[start : start + num_columns], axis=1))
            canvas = np.concatenate(rows, axis=0)
            if has_thumbnail:
                thumbnail = tile_grids[-1]
                projected = cv2.resize(
                    thumbnail,
                    (num_columns * side, num_rows * side),
                    interpolation=cv2.INTER_LINEAR,
                )
                projected_sum = float(projected.sum())
                if projected_sum > 0:
                    projected *= float(thumbnail.sum()) / projected_sum
                canvas = canvas + projected
            grids[mode][window] = canvas.astype(np.float32, copy=False)
    metadata = {
        "pixel_values_shape": list(pixel_values.shape),
        "tile_count": tile_count,
        "local_tile_count": num_local_tiles,
        "tile_columns": num_columns,
        "tile_rows": num_rows,
        "has_thumbnail": has_thumbnail,
        "original_width": original_width,
        "original_height": original_height,
        "tokens_per_tile": tokens_per_tile,
        "grid_height": num_rows * side,
        "grid_width": num_columns * side,
    }
    return dict(grids), metadata


def _npz_payload(grids: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {f"{mode}__{window}": array for mode, windows in grids.items() for window, array in windows.items()}


def probe_greedy(
    candidates_path: str | Path,
    output_dir: str | Path,
    device: str,
    max_new_tokens: int,
    limit: int,
    include_all: bool,
) -> dict[str, Any]:
    """Rerun exact greedy decoding without materializing full attention tensors."""

    import torch

    cases = read_jsonl(candidates_path)
    if not include_all:
        cases = [case for case in cases if case.get("shortlisted", False)]
    if limit > 0:
        cases = cases[:limit]
    if not cases:
        raise ValueError("No candidate cases selected")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    jobs_by_checkpoint: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for case in cases:
        for model_name in MODEL_ORDER:
            jobs_by_checkpoint[case["models"][model_name]["checkpoint"]].append((case, model_name))

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for checkpoint, jobs in jobs_by_checkpoint.items():
        print(f"[probe] loading checkpoint={checkpoint} jobs={len(jobs)}", flush=True)
        model, tokenizer, processor = _load_model_and_processor(checkpoint, device)
        for job_index, (case, model_name) in enumerate(jobs, start=1):
            try:
                row = _extract_dataset_row(case)
                batch_cpu, raw_prompt = _prepare_inputs(processor, row)
                batch = _move_inputs(batch_cpu, torch.device(device), torch.bfloat16)
                generated_ids, response = _manual_greedy(
                    model,
                    tokenizer,
                    batch,
                    max_new_tokens=max_new_tokens,
                )
                parsed = _official_choice_parse(
                    response,
                    row,
                    parser=str(case["parser"]),
                    labels=str(case["labels"]),
                    fallback_task=str(case["task"]),
                )
                records.append(
                    {
                        "case_id": case["case_id"],
                        "dataset": case["dataset"],
                        "question_id": case["question_id"],
                        "model": model_name,
                        "checkpoint": checkpoint,
                        "source_checkpoint": case["models"][model_name]["source_checkpoint"],
                        "raw_prompt": raw_prompt,
                        "raw_response": response,
                        "parsed_answer": parsed,
                        "ground_truth": case["ground_truth"],
                        "correct": parsed == str(case["ground_truth"]),
                        "generated_token_ids": generated_ids,
                        "generated_tokens": tokenizer.convert_ids_to_tokens(generated_ids),
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "case_id": case["case_id"],
                        "dataset": case["dataset"],
                        "question_id": case["question_id"],
                        "model": model_name,
                        "checkpoint": checkpoint,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if job_index % 25 == 0 or job_index == len(jobs):
                print(f"[probe] checkpoint progress={job_index}/{len(jobs)}", flush=True)
        del model, tokenizer, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    records.sort(key=lambda item: (item["dataset"], item["case_id"], MODEL_ORDER.index(item["model"])))
    write_jsonl(destination / "probe_records.jsonl", records)
    write_jsonl(destination / "failures.jsonl", failures)

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[record["case_id"]][record["model"]] = record
    all_candidates = []
    exact_candidates = []
    exact_by_dataset: dict[str, int] = defaultdict(int)
    for case in cases:
        updated = copy.deepcopy(case)
        model_records = grouped.get(case["case_id"], {})
        complete = set(model_records) == set(MODEL_ORDER)
        retained_match = {
            model_name: bool(
                complete
                and model_records[model_name]["parsed_answer"]
                == case["retained_predictions"][model_name]["parsed_answer"]
            )
            for model_name in MODEL_ORDER
        }
        selection_gate = str(case.get("selection_gate", "base_wrong_ttrv_wrong_ours_correct"))
        if selection_gate == "ours_correct_ttrv_valid_wrong":
            # Obs3 uses the exact checkpoints and majority@8 validation records.
            # The greedy attention rerun must reproduce both retained decisions;
            # Base is descriptive context and is deliberately not part of the gate.
            retained_ours = str(case["retained_predictions"]["ours"]["parsed_answer"])
            retained_ttrv = str(case["retained_predictions"]["ttrv"]["parsed_answer"])
            strict_eligible = bool(
                complete
                and retained_ours == str(case["ground_truth"])
                and retained_ttrv not in {str(case["ground_truth"]), "unknown"}
                and retained_match["ours"]
                and retained_match["ttrv"]
                and model_records["ours"]["correct"]
                and not model_records["ttrv"]["correct"]
                and model_records["ttrv"]["parsed_answer"] != "unknown"
            )
        elif selection_gate == "base_wrong_ttrv_wrong_ours_correct":
            strict_eligible = bool(
                complete
                and not model_records["base"]["correct"]
                and not model_records["ttrv"]["correct"]
                and model_records["ours"]["correct"]
            )
        else:
            raise ValueError(f"Unsupported selection_gate={selection_gate}")
        updated["shortlisted"] = strict_eligible
        updated["probe_complete"] = complete
        updated["strict_eligible"] = strict_eligible
        updated["probe_retained_match"] = retained_match
        updated["probe_predictions"] = {
            model_name: {
                "raw_response": record["raw_response"],
                "parsed_answer": record["parsed_answer"],
                "correct": record["correct"],
            }
            for model_name, record in model_records.items()
        }
        all_candidates.append(updated)
        if strict_eligible:
            exact_candidates.append(updated)
            exact_by_dataset[str(updated["dataset"])] += 1

    write_jsonl(destination / "probe_candidates_all.jsonl", all_candidates)
    write_jsonl(destination / "exact_candidates.jsonl", exact_candidates)
    summary = {
        "candidate_manifest": str(Path(candidates_path).resolve()),
        "output_dir": str(destination),
        "requested_cases": len(cases),
        "successful_model_cases": len(records),
        "failed_model_cases": len(failures),
        "complete_cases": sum(bool(case["probe_complete"]) for case in all_candidates),
        "strict_eligible_cases": len(exact_candidates),
        "strict_eligible_by_dataset": dict(sorted(exact_by_dataset.items())),
        "exact_candidates": str(destination / "exact_candidates.jsonl"),
    }
    write_json(destination / "probe_summary.json", summary)
    return summary

def extract_attentions(
    candidates_path: str | Path,
    output_dir: str | Path,
    device: str,
    max_new_tokens: int,
    limit: int,
    include_all: bool,
    save_individual_layers: bool = False,
    save_first_decision_heads: bool = False,
) -> dict[str, Any]:
    import torch

    cases = read_jsonl(candidates_path)
    if not include_all:
        cases = [case for case in cases if case.get("shortlisted", False)]
    if limit > 0:
        cases = cases[:limit]
    if not cases:
        raise ValueError("No candidate cases selected")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    records_by_checkpoint: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for case in cases:
        for model_name in MODEL_ORDER:
            records_by_checkpoint[case["models"][model_name]["checkpoint"]].append((case, model_name))

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_model_cases = sum(len(jobs) for jobs in records_by_checkpoint.values())
    processed_model_cases = 0
    for checkpoint, jobs in records_by_checkpoint.items():
        print(f"[extract] loading checkpoint={checkpoint} jobs={len(jobs)}", flush=True)
        model, tokenizer, processor = _load_model_and_processor(checkpoint, device)
        for case, model_name in jobs:
            artifact_dir = destination / sanitize_name(case["case_id"])
            artifact_dir.mkdir(parents=True, exist_ok=True)
            npz_path = artifact_dir / f"{model_name}.npz"
            json_path = artifact_dir / f"{model_name}.json"
            try:
                row = _extract_dataset_row(case)
                batch_cpu, raw_prompt = _prepare_inputs(processor, row)
                batch = _move_inputs(batch_cpu, torch.device(device), torch.bfloat16)
                generated_ids, response = _manual_greedy(model, tokenizer, batch, max_new_tokens=max_new_tokens)
                vectors, attention_meta = _attention_windows(
                    model,
                    batch,
                    generated_ids,
                    DEFAULT_LAYER_WINDOWS,
                    save_individual_layers=save_individual_layers,
                    save_first_decision_heads=save_first_decision_heads,
                )
                grids, grid_meta = _token_maps_to_grid(vectors, batch)
                parsed = _official_choice_parse(
                    response,
                    row,
                    parser=str(case["parser"]),
                    labels=str(case["labels"]),
                    fallback_task=str(case["task"]),
                )
                correct = parsed == str(case["ground_truth"])
                retained = case["retained_predictions"][model_name]
                record = {
                    "case_id": case["case_id"],
                    "dataset": case["dataset"],
                    "question_id": case["question_id"],
                    "model": model_name,
                    "checkpoint": checkpoint,
                    "source_checkpoint": case["models"][model_name]["source_checkpoint"],
                    "raw_prompt": raw_prompt,
                    "raw_response": response,
                    "parsed_answer": parsed,
                    "ground_truth": case["ground_truth"],
                    "correct": correct,
                    "retained_raw_response": retained["raw_response"],
                    "retained_parsed_answer": retained["parsed_answer"],
                    "retained_match": parsed == retained["parsed_answer"],
                    "retained_prediction_counter": retained.get("prediction_counter"),
                    "selection_gate": case.get("selection_gate"),
                    "validation_provenance": case.get("validation_provenance"),
                    "generated_token_ids": generated_ids,
                    "generated_tokens": tokenizer.convert_ids_to_tokens(generated_ids),
                    "attention": attention_meta,
                    "grid": grid_meta,
                    "layer_windows": DEFAULT_LAYER_WINDOWS,
                    "individual_layers": attention_meta["saved_individual_layers"],
                    "npz_path": str(npz_path),
                    "image_path": case["image_path"],
                    "question": case["question"],
                    "visual_keyword_score": case["visual_keyword_score"],
                }
                write_npz(npz_path, _npz_payload(grids))
                write_json(json_path, record)
                records.append(record)
            except Exception as exc:
                failure = {
                    "case_id": case["case_id"],
                    "dataset": case["dataset"],
                    "question_id": case["question_id"],
                    "model": model_name,
                    "checkpoint": checkpoint,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                write_json(artifact_dir / f"{model_name}.failed.json", failure)
            finally:
                processed_model_cases += 1
                report_progress = (
                    processed_model_cases % 10 == 0 or processed_model_cases == total_model_cases
                )
                if report_progress:
                    gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if report_progress:
                    print(
                        "[extract] progress="
                        f"{processed_model_cases}/{total_model_cases} "
                        f"successes={len(records)} failures={len(failures)} "
                        f"case_id={case['case_id']} model={model_name}",
                        flush=True,
                    )
        del model, tokenizer, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    records.sort(key=lambda item: (item["dataset"], item["case_id"], MODEL_ORDER.index(item["model"])))
    write_jsonl(destination / "attention_records.jsonl", records)
    write_jsonl(destination / "failures.jsonl", failures)
    summary = {
        "candidate_manifest": str(Path(candidates_path).resolve()),
        "output_dir": str(destination),
        "requested_cases": len(cases),
        "successful_model_cases": len(records),
        "failed_model_cases": len(failures),
        "save_individual_layers": bool(save_individual_layers),
        "save_first_decision_heads": bool(save_first_decision_heads),
        "artifact_write_policy": {
            "per_file_atomic_replace": True,
            "npz_json_pair_transactional": False,
            "commit_order": ["npz", "json", "aggregate_jsonl", "summary_json"],
        },
        "complete_cases": sum(
            1
            for case in cases
            if sum(record["case_id"] == case["case_id"] for record in records) == len(MODEL_ORDER)
        ),
    }
    write_json(destination / "extraction_summary.json", summary)
    return summary


def attention_temperature(grid: np.ndarray, temperature: float, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(grid, dtype=np.float64)
    values = np.maximum(values, 0.0)
    original_mass = float(values.sum())
    if original_mass <= 0 or not np.isfinite(original_mass):
        return np.zeros_like(values, dtype=np.float32)
    logits = np.log(values + eps) / float(temperature)
    logits -= logits.max()
    probabilities = np.exp(logits)
    total = probabilities.sum()
    if total <= 0 or not np.isfinite(total):
        return np.zeros_like(values, dtype=np.float32)
    # Preserve total image attention and adjust only spatial sharpness.
    return (original_mass * probabilities / total).astype(np.float32)


def jensen_shannon(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(left, dtype=np.float64).reshape(-1)
    q = np.asarray(right, dtype=np.float64).reshape(-1)
    p = np.maximum(p, 0.0) + eps
    q = np.maximum(q, 0.0) + eps
    p /= p.sum()
    q /= q.sum()
    middle = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / middle)) + 0.5 * np.sum(q * np.log(q / middle)))


@dataclass
class CompleteCase:
    case_id: str
    records: dict[str, dict[str, Any]]
    grids: dict[str, dict[str, np.ndarray]]


def _load_complete_cases(
    records_path: str | Path,
    attention_mode: str,
) -> tuple[list[CompleteCase], list[dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in read_jsonl(records_path):
        grouped[record["case_id"]][record["model"]] = record
    complete = []
    skipped_missing_mode = []
    required_keys = {f"{attention_mode}__{window}" for window in DEFAULT_LAYER_WINDOWS}
    for case_id, records in grouped.items():
        if set(records) != set(MODEL_ORDER):
            continue
        grids: dict[str, dict[str, np.ndarray]] = {}
        missing_models = []
        for model_name in MODEL_ORDER:
            record = records[model_name]
            npz_path = Path(str(record["npz_path"])).expanduser().resolve()
            with np.load(npz_path, allow_pickle=False) as arrays:
                missing_keys = sorted(required_keys - set(arrays.files))
                if missing_keys:
                    available_modes = record.get("attention", {}).get("available_query_modes")
                    if available_modes is None:
                        available_modes = sorted({key.split("__", 1)[0] for key in arrays.files if "__" in key})
                    missing_models.append(
                        {
                            "model": model_name,
                            "npz_path": str(npz_path),
                            "missing_keys": missing_keys,
                            "available_query_modes": available_modes,
                        }
                    )
                    continue
                grids[model_name] = {
                    window: arrays[f"{attention_mode}__{window}"] for window in DEFAULT_LAYER_WINDOWS
                }
        if missing_models:
            skipped_missing_mode.append(
                {
                    "case_id": case_id,
                    "reason": "missing_attention_mode",
                    "requested_attention_mode": attention_mode,
                    "models": missing_models,
                }
            )
            continue
        complete.append(CompleteCase(case_id=case_id, records=records, grids=grids))
    return complete, skipped_missing_mode


def choose_shared_setting(
    cases: Sequence[CompleteCase],
    temperatures: Sequence[float],
) -> tuple[str, float, list[dict[str, Any]]]:
    scores = []
    for window in DEFAULT_LAYER_WINDOWS:
        for temperature in temperatures:
            per_case = []
            for case in cases:
                transformed = {
                    model: attention_temperature(case.grids[model][window], temperature) for model in MODEL_ORDER
                }
                ours_base = jensen_shannon(transformed["ours"], transformed["base"])
                ours_ttrv = jensen_shannon(transformed["ours"], transformed["ttrv"])
                base_ttrv = jensen_shannon(transformed["base"], transformed["ttrv"])
                score = 0.5 * (ours_base + ours_ttrv) - 0.25 * base_ttrv
                per_case.append(score)
            scores.append(
                {
                    "window": window,
                    "temperature": float(temperature),
                    "mean_discriminability": float(np.mean(per_case)),
                    "median_discriminability": float(np.median(per_case)),
                }
            )
    scores.sort(key=lambda item: (-item["mean_discriminability"], item["window"], -item["temperature"]))
    best = scores[0]
    return str(best["window"]), float(best["temperature"]), scores


def _upsample(grid: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    width, height = size
    image = Image.fromarray(np.asarray(grid, dtype=np.float32), mode="F")
    return np.asarray(image.resize((width, height), resample=Image.Resampling.BILINEAR), dtype=np.float32)


def _case_score(case: CompleteCase, window: str, temperature: float) -> float:
    transformed = {
        model: attention_temperature(case.grids[model][window], temperature) for model in MODEL_ORDER
    }
    return 0.5 * (
        jensen_shannon(transformed["ours"], transformed["base"])
        + jensen_shannon(transformed["ours"], transformed["ttrv"])
    ) - 0.25 * jensen_shannon(transformed["base"], transformed["ttrv"])


def _render_figure(
    cases: Sequence[CompleteCase],
    window: str,
    temperature: float,
    output_path: Path,
    title: str | None,
) -> dict[str, Any]:
    import textwrap

    import cv2
    from PIL import Image, ImageDraw, ImageFont

    prepared = []
    all_maps = []
    for case in cases:
        reference = case.records["base"]
        with Image.open(reference["image_path"]) as image:
            original = image.convert("RGB").copy()
        maps = {}
        for model_name in MODEL_ORDER:
            grid = attention_temperature(case.grids[model_name][window], temperature)
            maps[model_name] = _upsample(grid, original.size)
            all_maps.append(maps[model_name])
        prepared.append((case, original, maps))
    global_min = float(min(array.min() for array in all_maps))
    global_max = float(max(array.max() for array in all_maps))
    if global_max <= global_min:
        global_max = global_min + 1e-8

    def font(size: int, bold: bool = False):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
        for candidate in candidates:
            if Path(candidate).is_file():
                return ImageFont.truetype(candidate, size=size)
        return ImageFont.load_default()

    header_font = font(28, bold=True)
    label_font = font(17, bold=True)
    body_font = font(16)
    small_font = font(14)
    panel_width = 480
    image_height = 330
    text_height = 142
    header_height = 58
    row_height = image_height + text_height
    colorbar_width = 92
    canvas_width = panel_width * 4 + colorbar_width
    canvas_height = header_height + row_height * len(cases)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    headers = ("Image / Question / GT", "Base", "TTRV", "Ours")
    for column, header in enumerate(headers):
        box = draw.textbbox((0, 0), header, font=header_font)
        x = column * panel_width + (panel_width - (box[2] - box[0])) // 2
        draw.text((x, 12), header, fill="black", font=header_font)

    def fit_image(image: Image.Image) -> tuple[Image.Image, int, int]:
        scale = min((panel_width - 20) / image.width, (image_height - 14) / image.height)
        width = max(1, int(round(image.width * scale)))
        height = max(1, int(round(image.height * scale)))
        resized = image.resize((width, height), resample=Image.Resampling.LANCZOS)
        return resized, (panel_width - width) // 2, (image_height - height) // 2

    for row_index, (case, original, maps) in enumerate(prepared):
        y0 = header_height + row_index * row_height
        resized, x_pad, y_pad = fit_image(original)
        canvas.paste(resized, (x_pad, y0 + y_pad))
        reference = case.records["base"]
        question = re.sub(r"\s+", " ", str(reference["question"])).strip()
        lines = [
            f"{reference['dataset']} | {reference['question_id']}",
            *textwrap.wrap(question, width=54)[:4],
            f"GT: {reference['ground_truth']}",
        ]
        text_y = y0 + image_height + 4
        for line_index, line in enumerate(lines):
            draw.text((12, text_y + line_index * 20), line, fill="black", font=small_font)

        for column, model_name in enumerate(MODEL_ORDER, start=1):
            values = (maps[model_name] - global_min) / (global_max - global_min)
            values = np.clip(values, 0.0, 1.0)
            heat_bgr = cv2.applyColorMap(np.round(values * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
            heat = Image.fromarray(heat_rgb).resize(original.size, resample=Image.Resampling.BILINEAR)
            overlay = Image.blend(original, heat, alpha=0.52)
            overlay_resized, overlay_x, overlay_y = fit_image(overlay)
            x0 = column * panel_width
            canvas.paste(overlay_resized, (x0 + overlay_x, y0 + overlay_y))
            record = case.records[model_name]
            mark = "✓" if record["correct"] else "✗"
            answer = re.sub(r"\s+", " ", str(record["raw_response"])).strip()
            if len(answer) > 58:
                answer = answer[:55] + "..."
            answer_lines = textwrap.wrap(
                f"Answer: {answer}",
                width=43,
                break_long_words=True,
                break_on_hyphens=False,
            )[:2]
            answer_y = y0 + image_height + 10
            for line_index, line in enumerate(answer_lines):
                draw.text((x0 + 14, answer_y + line_index * 21), line, fill="black", font=label_font)
            parsed_y = answer_y + max(1, len(answer_lines)) * 21 + 3
            draw.text(
                (x0 + 14, parsed_y),
                f"Parsed: {record['parsed_answer']} {mark}",
                fill="black",
                font=label_font,
            )
            retained = "retained match" if record.get("retained_match") else "retained mismatch"
            draw.text((x0 + 14, parsed_y + 25), retained, fill=(80, 80, 80), font=small_font)

    bar_x = panel_width * 4 + 22
    bar_top = header_height + 36
    bar_bottom = canvas_height - 48
    gradient = np.linspace(255, 0, max(2, bar_bottom - bar_top), dtype=np.uint8).reshape(-1, 1)
    gradient = np.repeat(gradient, 28, axis=1)
    bar_bgr = cv2.applyColorMap(gradient, cv2.COLORMAP_VIRIDIS)
    bar = Image.fromarray(cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2RGB))
    canvas.paste(bar, (bar_x, bar_top))
    draw.rectangle((bar_x, bar_top, bar_x + 27, bar_bottom), outline="black", width=1)
    draw.text((bar_x - 6, bar_top - 24), f"{global_max:.3g}", fill="black", font=small_font)
    draw.text((bar_x - 6, bar_bottom + 6), f"{global_min:.3g}", fill="black", font=small_font)
    draw.text((bar_x - 8, bar_bottom + 27), "attention", fill="black", font=small_font)
    if title:
        draw.text((12, 12), title, fill="black", font=body_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    canvas.save(output_path.with_suffix(".pdf"), format="PDF", resolution=220.0)
    return {"global_min": global_min, "global_max": global_max}


def _attention_mode_caption(attention_mode: str, layers: Sequence[int]) -> str:
    descriptions = {
        "answer_state": (
            "Answer-to-image decoder self-attention queried from generated answer-token states and "
            "averaged over generated answer positions, attention heads, and layers"
        ),
        "decision_query": (
            "Answer-to-image decoder self-attention queried from the causal states immediately preceding "
            "each generated answer token and averaged over those decision positions, attention heads, and layers"
        ),
        "first_decision_query": (
            "Answer-to-image decoder self-attention queried from the prompt-end (final valid prompt token) "
            "first-answer decision state before any checkpoint-specific generated token, averaged over "
            "attention heads and layers"
        ),
    }
    try:
        description = descriptions[attention_mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported attention_mode={attention_mode!r}") from exc
    return (
        f"{description} {list(layers)} (zero-based). All panels share the same color scale. "
        "Model answers are shown below each panel (correct/incorrect versus ground truth)."
    )


def render_outputs(
    records_path: str | Path,
    output_dir: str | Path,
    attention_mode: str,
    temperatures: Sequence[float],
    selected_case_ids: Sequence[str] | None = None,
    layer_window: str | None = None,
    temperature_override: float | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cases, skipped_missing_mode = _load_complete_cases(records_path, attention_mode=attention_mode)
    if skipped_missing_mode:
        print(
            f"[render] skipped {len(skipped_missing_mode)} complete case(s) that do not provide "
            f"attention_mode={attention_mode!r}",
            flush=True,
        )
    if not cases:
        examples = [
            {
                "case_id": item["case_id"],
                "models": [detail["model"] for detail in item["models"]],
            }
            for item in skipped_missing_mode[:3]
        ]
        raise ValueError(
            f"No complete Base/TTRV/Ours case provides attention_mode={attention_mode!r}; "
            f"skipped_missing_mode={len(skipped_missing_mode)} examples={examples}"
        )
    eligible = []
    for case in cases:
        selection_gate = case.records["ours"].get("selection_gate")
        if selection_gate == "ours_correct_ttrv_valid_wrong":
            passes = bool(
                case.records["ours"]["correct"]
                and not case.records["ttrv"]["correct"]
                and case.records["ttrv"]["parsed_answer"] != "unknown"
                and case.records["ours"].get("retained_match", False)
                and case.records["ttrv"].get("retained_match", False)
            )
        else:
            passes = bool(
                not case.records["base"]["correct"]
                and not case.records["ttrv"]["correct"]
                and case.records["ours"]["correct"]
            )
        if passes:
            eligible.append(case)
    if not eligible:
        raise ValueError("No complete rerun case satisfies its configured validation gate")
    auto_window, auto_temperature, setting_scores = choose_shared_setting(eligible, temperatures)
    window = layer_window or auto_window
    temperature = float(temperature_override) if temperature_override is not None else auto_temperature
    ranked = sorted(eligible, key=lambda case: -_case_score(case, window, temperature))

    if selected_case_ids:
        eligible_by_id = {case.case_id: case for case in eligible}
        skipped_by_id = {item["case_id"]: item for item in skipped_missing_mode}
        unavailable = [case_id for case_id in selected_case_ids if case_id in skipped_by_id]
        if unavailable:
            raise ValueError(
                f"Requested cases do not provide attention_mode={attention_mode!r}: {unavailable}"
            )
        missing = [case_id for case_id in selected_case_ids if case_id not in eligible_by_id]
        if missing:
            raise ValueError(f"Requested cases are not complete strict candidates: {missing}")
        if len(selected_case_ids) != 3 or len(set(selected_case_ids)) != 3:
            raise ValueError("Exactly three distinct --selected-case-ids are required")
        selected = [eligible_by_id[case_id] for case_id in selected_case_ids]
        selection_policy = "explicit_strict_case_ids"
    else:
        selected = []
        datasets = sorted({case.records["base"]["dataset"] for case in ranked})
        for dataset in datasets:
            matches = [case for case in ranked if case.records["base"]["dataset"] == dataset]
            if matches:
                selected.append(matches[0])
        if len(selected) < 3:
            raise ValueError(f"Only {len(selected)} datasets have eligible complete cases; need 3")
        selected = selected[:3]
        selection_policy = "top_discriminability_per_dataset"

    scale = _render_figure(
        selected,
        window,
        temperature,
        destination / f"final_3x4_{attention_mode}.png",
        title=None,
    )
    gallery_rows = []
    for case in ranked:
        score = _case_score(case, window, temperature)
        gallery_rows.append(
            {
                "case_id": case.case_id,
                "dataset": case.records["base"]["dataset"],
                "question_id": case.records["base"]["question_id"],
                "discriminability": score,
                "selected_final": case.case_id in {item.case_id for item in selected},
            }
        )
    write_jsonl(destination / "ranked_cases.jsonl", gallery_rows)

    # Diagnostic sheets preserve every layer/temperature choice while enforcing
    # one scale per sheet across all three checkpoints.
    diagnostic_cases = ranked[: min(len(ranked), 12)]
    for test_window in DEFAULT_LAYER_WINDOWS:
        for test_temperature in temperatures:
            for case in diagnostic_cases:
                name = f"gallery_{case.case_id}_{attention_mode}_{test_window}_T{test_temperature:g}.png"
                _render_figure([case], test_window, test_temperature, destination / "gallery" / name, title=None)

    summary = {
        "records": str(Path(records_path).resolve()),
        "attention_mode": attention_mode,
        "attention_mode_caption": _attention_mode_caption(
            attention_mode,
            DEFAULT_LAYER_WINDOWS[window],
        ),
        "complete_cases_with_attention_mode": len(cases),
        "skipped_missing_attention_mode_count": len(skipped_missing_mode),
        "skipped_missing_attention_mode_cases": skipped_missing_mode,
        "eligible_complete_cases": len(eligible),
        "selection_policy": selection_policy,
        "shared_layer_window": window,
        "shared_layer_indices_zero_based": list(DEFAULT_LAYER_WINDOWS[window]),
        "shared_temperature": temperature,
        "shared_color_scale": scale,
        "selected_cases": [
            {
                "case_id": case.case_id,
                "dataset": case.records["base"]["dataset"],
                "question_id": case.records["base"]["question_id"],
                "discriminability": _case_score(case, window, temperature),
            }
            for case in selected
        ],
        "setting_scores": setting_scores,
        "figure_png": str(destination / f"final_3x4_{attention_mode}.png"),
        "figure_pdf": str(destination / f"final_3x4_{attention_mode}.pdf"),
        "caption": _attention_mode_caption(attention_mode, DEFAULT_LAYER_WINDOWS[window]),
    }
    write_json(destination / "render_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser("merge", help="Merge a verl FSDP actor checkpoint")
    merge_parser.add_argument("--checkpoint", required=True)
    merge_parser.add_argument("--base-model", required=True)
    merge_parser.add_argument("--output-dir", required=True)
    merge_parser.add_argument("--overwrite", action="store_true")

    select_parser = subparsers.add_parser("select", help="Build candidate manifest from retained greedy outputs")
    select_parser.add_argument("--config", required=True)
    select_parser.add_argument("--output", required=True)

    probe_parser = subparsers.add_parser("probe", help="Rerun exact greedy decoding without full attentions")
    probe_parser.add_argument("--candidates", required=True)
    probe_parser.add_argument("--output-dir", required=True)
    probe_parser.add_argument("--device", default="cuda:0")
    probe_parser.add_argument("--max-new-tokens", type=int, default=64)
    probe_parser.add_argument("--limit", type=int, default=0)
    probe_parser.add_argument("--include-all", action="store_true")

    extract_parser = subparsers.add_parser("extract", help="Extract answer-to-image attentions")
    extract_parser.add_argument("--candidates", required=True)
    extract_parser.add_argument("--output-dir", required=True)
    extract_parser.add_argument("--device", default="cuda:0")
    extract_parser.add_argument("--max-new-tokens", type=int, default=64)
    extract_parser.add_argument("--limit", type=int, default=0)
    extract_parser.add_argument("--include-all", action="store_true")
    extract_parser.add_argument("--save-individual-layers", action="store_true")
    extract_parser.add_argument(
        "--save-first-decision-heads",
        action="store_true",
        help="Save per-head first_decision_query maps for every decoder layer",
    )

    render_parser = subparsers.add_parser("render", help="Rank settings and render final/gallery figures")
    render_parser.add_argument("--records", required=True)
    render_parser.add_argument("--output-dir", required=True)
    render_parser.add_argument(
        "--attention-mode",
        choices=("answer_state", "decision_query", "first_decision_query"),
        default="first_decision_query",
    )
    render_parser.add_argument("--temperatures", default="1.0,0.7,0.5,0.3")
    render_parser.add_argument(
        "--selected-case-ids",
        default="",
        help="Comma-separated strict candidate IDs; when set, exactly three distinct IDs are required",
    )
    render_parser.add_argument("--layer-window", choices=tuple(DEFAULT_LAYER_WINDOWS), default=None)
    render_parser.add_argument("--temperature", type=float, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "merge":
        result = merge_fsdp_checkpoint(args.checkpoint, args.base_model, args.output_dir, overwrite=args.overwrite)
    elif args.command == "select":
        result = select_candidates(args.config, args.output)
    elif args.command == "probe":
        result = probe_greedy(
            args.candidates,
            args.output_dir,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            include_all=args.include_all,
        )
    elif args.command == "extract":
        result = extract_attentions(
            args.candidates,
            args.output_dir,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            include_all=args.include_all,
            save_individual_layers=args.save_individual_layers,
            save_first_decision_heads=args.save_first_decision_heads,
        )
    elif args.command == "render":
        temperatures = tuple(float(value) for value in args.temperatures.split(",") if value.strip())
        selected_case_ids = tuple(value.strip() for value in args.selected_case_ids.split(",") if value.strip())
        result = render_outputs(
            args.records,
            args.output_dir,
            args.attention_mode,
            temperatures,
            selected_case_ids=selected_case_ids,
            layer_window=args.layer_window,
            temperature_override=args.temperature,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
