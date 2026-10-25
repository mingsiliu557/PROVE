#!/usr/bin/env python3
"""Extract teacher-forced option-content-to-image attention for InternVL3.

The original benchmark response is a choice letter.  This diagnostic expands a
choice to a canonical assistant response such as ``C. Seal`` and keeps only the
semantic option-text tokens (``Seal``) as queries.  It deliberately saves two
different query semantics:

* ``content_answer_state``: the state at the teacher-forced content token.  This
  is token-conditioned and is closest to the qualitative maps used by TITA.
* ``content_prediction_query``: the immediately preceding causal state which
  predicts that content token.

The former is a grounding probe, not evidence that the same state caused the
already-observed token.  The latter preserves the autoregressive causal shift.
No model parameters are updated by this script.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from attention_viz import (
    DEFAULT_LAYER_WINDOWS,
    IMAGE_CONTEXT_TOKEN,
    MODEL_ORDER,
    _build_messages,
    _load_model_and_processor,
    _load_pil_images,
    _move_inputs,
    _row_qid,
    read_jsonl,
    sanitize_name,
    write_json,
    write_jsonl,
    write_npz,
)


CONTENT_LAYER_WINDOWS = {
    "early_visual": (1, 2, 3, 4, 5),
    **DEFAULT_LAYER_WINDOWS,
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


def _prompt_body(row: Mapping[str, Any]) -> str:
    messages = row.get("prompt")
    if not isinstance(messages, list):
        return str(messages or "")
    pieces: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            for segment in content:
                if isinstance(segment, Mapping) and segment.get("type") == "text":
                    pieces.append(str(segment.get("text", "")))
        else:
            pieces.append(str(content))
    return "\n".join(pieces)


def parse_options(row: Mapping[str, Any]) -> dict[str, str]:
    """Parse local A--F option lines without changing the model prompt.

    AI2D/MME use an ``Options:`` block with ``A.`` markers.  RealWorldQA uses
    the same line-oriented MCQ structure but may omit the heading and may use
    ``A:`` markers.  The final option is also followed by a response-format
    instruction, which is metadata rather than part of that option.
    """

    body = _prompt_body(row)
    option_text = body.split("Options:", 1)[1] if "Options:" in body else body
    matches = list(OPTION_LINE.finditer(option_text))
    options: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(option_text)
        label = match.group(1).upper()
        value = option_text[match.end() : end]
        instruction = TRAILING_INSTRUCTION_LINE.search(value)
        if instruction is not None:
            value = value[: instruction.start()]
        value = re.sub(r"\s+", " ", value).strip()
        if not value:
            raise ValueError(f"Empty option text for {label}")
        if label in options and options[label] != value:
            raise ValueError(f"Duplicate option label {label}")
        options[label] = value
    if not options:
        raise ValueError("No option lines parsed")
    return options


def resolve_target_label(case: Mapping[str, Any], model_name: str, source: str) -> str:
    if source == "ground_truth":
        value = case.get("ground_truth")
    elif source == "retained":
        value = case.get("retained_predictions", {}).get(model_name, {}).get("parsed_answer")
    else:
        raise ValueError(f"Unsupported target source: {source}")
    label = str(value or "").strip().upper()
    if not re.fullmatch(r"[A-F]", label):
        raise ValueError(f"No valid {source} choice for {model_name}: {value!r}")
    return label


def canonical_choice(label: str, options: Mapping[str, str]) -> tuple[str, str]:
    if label not in options:
        raise ValueError(f"Target label {label} absent from options={sorted(options)}")
    option_text = re.sub(r"\s+", " ", str(options[label])).strip()
    return f"{label}. {option_text}", option_text


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _first_stop_offset(tokenizer: Any, token_ids: Sequence[int]) -> int:
    stop_ids = {tokenizer.eos_token_id}
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and int(token_id) >= 0:
            stop_ids.add(int(token_id))
    stop_ids = {int(value) for value in stop_ids if value is not None and int(value) >= 0}
    for index, token_id in enumerate(token_ids):
        if int(token_id) in stop_ids:
            return index
    return len(token_ids)


def _content_token_indices(
    tokenizer: Any,
    target_ids: Sequence[int],
    label: str,
    canonical: str,
) -> tuple[list[int], dict[str, Any]]:
    decoded = tokenizer.decode(list(target_ids), skip_special_tokens=False)
    if _normalized_text(decoded) != _normalized_text(canonical):
        raise ValueError(f"Teacher-forced suffix drift: expected={canonical!r} decoded={decoded!r}")

    prefix = f"{label}."
    prefix_end: int | None = None
    for end in range(1, len(target_ids) + 1):
        partial = tokenizer.decode(list(target_ids[:end]), skip_special_tokens=False).lstrip()
        if partial.startswith(prefix) and len(partial) >= len(prefix):
            prefix_end = end
            break
    if prefix_end is None:
        raise ValueError(f"Could not locate label prefix {prefix!r} in target tokens")

    pieces = [tokenizer.decode([int(token_id)], skip_special_tokens=False) for token_id in target_ids]
    indices = [
        index
        for index in range(prefix_end, len(target_ids))
        if any(character.isalnum() for character in pieces[index])
    ]
    if not indices:
        raise ValueError(f"No alphanumeric content tokens in target={canonical!r} pieces={pieces!r}")
    metadata = {
        "target_decoded": decoded,
        "target_token_ids": [int(value) for value in target_ids],
        "target_tokens": tokenizer.convert_ids_to_tokens(list(target_ids)),
        "target_token_pieces": pieces,
        "label_prefix_token_end": int(prefix_end),
        "content_target_indices": indices,
        "content_token_ids": [int(target_ids[index]) for index in indices],
        "content_tokens": tokenizer.convert_ids_to_tokens([int(target_ids[index]) for index in indices]),
        "content_token_pieces": [pieces[index] for index in indices],
    }
    return indices, metadata


def prepare_teacher_forced_inputs(
    processor: Any,
    tokenizer: Any,
    row: Mapping[str, Any],
    label: str,
    canonical: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build prompt-only and full-chat IDs, then align the assistant suffix."""

    messages = _build_messages(row)
    raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    full_messages = copy.deepcopy(messages)
    full_messages.append({"role": "assistant", "content": canonical})
    raw_full = processor.apply_chat_template(full_messages, add_generation_prompt=False, tokenize=False)
    images = _load_pil_images(row)
    prompt_batch = dict(
        processor(
            text=[raw_prompt.replace("<image>", IMAGE_CONTEXT_TOKEN)],
            images=images or None,
            return_tensors="pt",
        )
    )
    full_batch = dict(
        processor(
            text=[raw_full.replace("<image>", IMAGE_CONTEXT_TOKEN)],
            images=images or None,
            return_tensors="pt",
        )
    )
    prompt_ids = prompt_batch["input_ids"][0]
    full_ids = full_batch["input_ids"][0]
    prompt_length = int(prompt_ids.numel())
    if full_ids.numel() <= prompt_length:
        raise ValueError(f"Full chat length {full_ids.numel()} <= prompt length {prompt_length}")
    if not bool((full_ids[:prompt_length] == prompt_ids).all().item()):
        mismatch = (full_ids[:prompt_length] != prompt_ids).nonzero(as_tuple=False).flatten()
        first = int(mismatch[0].item()) if mismatch.numel() else -1
        raise ValueError(f"Prompt/full chat prefix mismatch at token {first}")

    suffix = [int(value) for value in full_ids[prompt_length:].tolist()]
    suffix = suffix[: _first_stop_offset(tokenizer, suffix)]
    if not suffix:
        raise ValueError("Teacher-forced assistant suffix is empty")
    target_indices, token_meta = _content_token_indices(tokenizer, suffix, label, canonical)

    image_processor = processor.image_processor
    full_batch["_attention_viz_original_sizes"] = [list(image.size) for image in images]
    full_batch["_attention_viz_tiling"] = {
        "min_patches": int(image_processor.min_patches),
        "max_patches": int(image_processor.max_patches),
        "use_thumbnail": True,
    }
    metadata = {
        "raw_prompt": raw_prompt,
        "raw_full_chat": raw_full,
        "prompt_length": prompt_length,
        "target_sequence_start": prompt_length,
        **token_meta,
    }
    return full_batch, metadata


def _attention_vectors(
    model: Any,
    batch: Mapping[str, Any],
    target_meta: Mapping[str, Any],
    layer_windows: Mapping[str, Sequence[int]],
    save_individual_layers: bool,
    save_content_answer_heads: bool,
    save_token_maps: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    import torch

    full_ids = batch["input_ids"]
    if full_ids.ndim != 2 or full_ids.shape[0] != 1:
        raise NotImplementedError(f"Expected one sequence, got {tuple(full_ids.shape)}")
    prompt_length = int(target_meta["prompt_length"])
    target_indices = [int(value) for value in target_meta["content_target_indices"]]
    answer_positions = torch.tensor(
        [prompt_length + index for index in target_indices],
        dtype=torch.long,
        device=full_ids.device,
    )
    prediction_positions = answer_positions - 1
    first_decision_position = torch.tensor([prompt_length - 1], dtype=torch.long, device=full_ids.device)
    image_positions = full_ids[0].eq(model.img_context_token_id).nonzero(as_tuple=False).flatten()
    if image_positions.numel() == 0:
        raise ValueError("No <IMG_CONTEXT> positions found")

    forward_batch = {
        "pixel_values": batch["pixel_values"],
        "input_ids": full_ids,
        "attention_mask": batch.get("attention_mask", torch.ones_like(full_ids)),
        "image_flags": batch.get(
            "image_flags",
            torch.ones((batch["pixel_values"].shape[0], 1), dtype=torch.long, device=full_ids.device),
        ),
        "use_cache": False,
        "output_attentions": True,
        "return_dict": True,
    }
    with torch.inference_mode():
        outputs = model(**forward_batch)
    if not outputs.attentions or any(value is None for value in outputs.attentions):
        raise RuntimeError("Model returned no attentions; eager attention is required")

    query_modes: list[tuple[str, Any]] = [
        ("content_answer_state", answer_positions),
        ("content_prediction_query", prediction_positions),
        ("first_decision_query", first_decision_position),
    ]
    if save_token_maps:
        for token_index, (answer_position, prediction_position) in enumerate(
            zip(answer_positions, prediction_positions)
        ):
            query_modes.extend(
                [
                    (f"content_token_{token_index:02d}_answer_state", answer_position.reshape(1)),
                    (f"content_token_{token_index:02d}_prediction_query", prediction_position.reshape(1)),
                ]
            )

    result: dict[str, dict[str, np.ndarray]] = {mode: {} for mode, _ in query_modes}
    head_mode = "content_answer_state_heads"
    if save_content_answer_heads:
        result[head_mode] = {}
    for window_name, layer_indices in layer_windows.items():
        layers = [outputs.attentions[int(layer_index)][0].float() for layer_index in layer_indices]
        stacked = torch.stack(layers, dim=0)  # layer, head, query, key
        for mode, positions in query_modes:
            cross = stacked[:, :, positions, :][..., image_positions]
            result[mode][window_name] = cross.mean(dim=(0, 1, 2)).detach().cpu().numpy().astype(np.float32)

    if save_individual_layers or save_content_answer_heads:
        for layer_index, layer_attention in enumerate(outputs.attentions):
            layer = layer_attention[0].float()  # head, query, key
            if save_individual_layers:
                for mode, positions in query_modes:
                    cross = layer[:, positions, :][..., image_positions]
                    result[mode][f"layer_{layer_index:02d}"] = (
                        cross.mean(dim=(0, 1)).detach().cpu().numpy().astype(np.float32)
                    )
            if save_content_answer_heads:
                head_vectors = layer[:, answer_positions, :][..., image_positions].mean(dim=1)
                for head_index, vector in enumerate(head_vectors):
                    result[head_mode][f"layer_{layer_index:02d}_head_{head_index:02d}"] = (
                        vector.detach().cpu().numpy().astype(np.float32)
                    )

    num_layers = len(outputs.attentions)
    num_heads = int(outputs.attentions[0].shape[1])
    metadata = {
        "query_semantics": {
            "content_answer_state": "teacher-forced content-token state; token already observed",
            "content_prediction_query": "immediately preceding causal state that predicts the content token",
            "first_decision_query": "last prompt state before any answer token",
        },
        "available_query_modes": [mode for mode, _ in query_modes],
        "content_answer_positions": answer_positions.detach().cpu().tolist(),
        "content_prediction_positions": prediction_positions.detach().cpu().tolist(),
        "first_decision_position": int(first_decision_position.item()),
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
        "num_layers": num_layers,
        "num_heads": num_heads,
        "saved_individual_layers": list(range(num_layers)) if save_individual_layers else [],
        "saved_content_answer_heads": bool(save_content_answer_heads),
        "content_answer_head_key_format": (
            f"{head_mode}__layer_{{layer:02d}}_head_{{head:02d}}" if save_content_answer_heads else None
        ),
    }
    del outputs
    return result, metadata


def _spatial_payload(
    vectors: Mapping[str, Mapping[str, np.ndarray]],
    batch: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reconstruct maps while retaining local-tile and thumbnail components."""

    import cv2
    from transformers.models.got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas

    pixel_values = batch["pixel_values"]
    tile_count = int(pixel_values.shape[0])
    first_vector = next(iter(next(iter(vectors.values())).values()))
    token_count = int(first_vector.shape[0])
    if token_count % tile_count:
        raise ValueError(f"image tokens={token_count} not divisible by tiles={tile_count}")
    tokens_per_tile = token_count // tile_count
    side = int(round(math.sqrt(tokens_per_tile)))
    if side * side != tokens_per_tile:
        raise ValueError(f"tokens_per_tile={tokens_per_tile} is not square")

    original_sizes = batch.get("_attention_viz_original_sizes", [])
    if len(original_sizes) != 1:
        raise NotImplementedError(f"Expected one image, got {len(original_sizes)}")
    original_width, original_height = (int(value) for value in original_sizes[0])
    tiling = batch.get("_attention_viz_tiling", {})
    tile_height, tile_width = (int(value) for value in pixel_values.shape[-2:])
    num_columns, num_rows = get_optimal_tiled_canvas(
        (original_height, original_width),
        (tile_height, tile_width),
        int(tiling.get("min_patches", 1)),
        int(tiling.get("max_patches", 12)),
    )
    local_tile_count = int(num_columns * num_rows)
    has_thumbnail = bool(tiling.get("use_thumbnail", True) and local_tile_count != 1)
    if tile_count != local_tile_count + int(has_thumbnail):
        raise ValueError(
            f"Tile layout mismatch actual={tile_count} local={local_tile_count} thumbnail={has_thumbnail}"
        )

    payload: dict[str, np.ndarray] = {}
    masses: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for mode, windows in vectors.items():
        is_head_mode = mode.endswith("_heads")
        for window, vector in windows.items():
            tile_grids = vector.reshape(tile_count, side, side)
            rows = []
            for row_index in range(num_rows):
                start = row_index * num_columns
                rows.append(np.concatenate(tile_grids[start : start + num_columns], axis=1))
            local = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
            thumbnail = np.zeros_like(local)
            thumbnail_mass = 0.0
            if has_thumbnail:
                thumbnail_grid = tile_grids[-1]
                thumbnail = cv2.resize(
                    thumbnail_grid,
                    (num_columns * side, num_rows * side),
                    interpolation=cv2.INTER_LINEAR,
                ).astype(np.float32, copy=False)
                projected_sum = float(thumbnail.sum())
                thumbnail_mass = float(thumbnail_grid.sum())
                if projected_sum > 0:
                    thumbnail *= thumbnail_mass / projected_sum
            fused = (local + thumbnail).astype(np.float32, copy=False)
            vector_mass = float(vector.sum())
            if not np.isclose(float(fused.sum()), vector_mass, rtol=2e-4, atol=2e-7):
                raise AssertionError(
                    f"Spatial mass drift mode={mode} window={window} vector={vector_mass} fused={fused.sum()}"
                )
            key = f"{mode}__{window}"
            payload[key] = fused
            if not is_head_mode:
                payload[f"{key}__local"] = local
                payload[f"{key}__thumbnail"] = thumbnail
            masses[mode][window] = {
                "total_image_mass": vector_mass,
                "local_mass": float(local.sum()),
                "thumbnail_mass": thumbnail_mass,
            }

    metadata = {
        "pixel_values_shape": list(pixel_values.shape),
        "tile_count": tile_count,
        "local_tile_count": local_tile_count,
        "tile_columns": int(num_columns),
        "tile_rows": int(num_rows),
        "has_thumbnail": has_thumbnail,
        "tokens_per_tile": tokens_per_tile,
        "grid_height": int(num_rows * side),
        "grid_width": int(num_columns * side),
        "original_width": original_width,
        "original_height": original_height,
        "attention_mass": dict(masses),
        "component_policy": "fused=local+mass_preserving_projected_thumbnail; non-head modes retain all components",
    }
    return payload, metadata


def _load_rows(cases: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq

    by_path: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        by_path[str(case["test_parquet"])].append(case)
    rows: dict[str, dict[str, Any]] = {}
    for parquet_path, path_cases in by_path.items():
        table = pq.read_table(parquet_path)
        for case in path_cases:
            row_index = int(case["row_index"])
            if not 0 <= row_index < table.num_rows:
                raise IndexError(f"row_index={row_index} outside {table.num_rows}")
            row = table.slice(row_index, 1).to_pylist()[0]
            qid = _row_qid(row, row_index)
            if qid != str(case["question_id"]):
                raise ValueError(
                    f"Question ID drift case={case['case_id']} manifest={case['question_id']} parquet={qid}"
                )
            rows[str(case["case_id"])] = row
    return rows


def _parse_models(value: str) -> tuple[str, ...]:
    models = tuple(part.strip() for part in value.split(",") if part.strip())
    if not models or len(set(models)) != len(models):
        raise ValueError(f"Invalid --models value: {value!r}")
    unknown = set(models) - set(MODEL_ORDER)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    return models


def validate_manifest(
    candidates_path: str | Path,
    models: Sequence[str],
    target_source: str,
    limit: int,
) -> dict[str, Any]:
    cases = read_jsonl(candidates_path)
    if limit > 0:
        cases = cases[:limit]
    rows = _load_rows(cases)
    targets = 0
    errors: list[dict[str, Any]] = []
    by_dataset: dict[str, int] = defaultdict(int)
    for case in cases:
        row = rows[str(case["case_id"])]
        try:
            options = parse_options(row)
            for model_name in models:
                label = resolve_target_label(case, model_name, target_source)
                canonical_choice(label, options)
                targets += 1
            by_dataset[str(case["dataset"])] += 1
        except Exception as exc:
            errors.append(
                {
                    "case_id": case.get("case_id"),
                    "dataset": case.get("dataset"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return {
        "candidate_manifest": str(Path(candidates_path).resolve()),
        "cases": len(cases),
        "models": list(models),
        "target_source": target_source,
        "valid_targets": targets,
        "valid_cases_by_dataset": dict(sorted(by_dataset.items())),
        "errors": errors,
    }


def extract(
    candidates_path: str | Path,
    output_dir: str | Path,
    device: str,
    models: Sequence[str],
    target_source: str,
    limit: int,
    save_individual_layers: bool,
    save_content_answer_heads: bool,
    save_token_maps: bool,
) -> dict[str, Any]:
    import torch

    cases = read_jsonl(candidates_path)
    if limit > 0:
        cases = cases[:limit]
    if not cases:
        raise ValueError("No cases selected")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(cases)

    jobs_by_checkpoint: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for case in cases:
        for model_name in models:
            jobs_by_checkpoint[str(case["models"][model_name]["checkpoint"])].append((case, model_name))

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_jobs = sum(len(jobs) for jobs in jobs_by_checkpoint.values())
    processed = 0
    for checkpoint, jobs in jobs_by_checkpoint.items():
        print(f"[content-extract] loading checkpoint={checkpoint} jobs={len(jobs)}", flush=True)
        model, tokenizer, processor = _load_model_and_processor(checkpoint, device)
        for case, model_name in jobs:
            artifact_dir = destination / sanitize_name(str(case["case_id"]))
            artifact_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{model_name}_{target_source}"
            npz_path = artifact_dir / f"{stem}.npz"
            json_path = artifact_dir / f"{stem}.json"
            try:
                row = rows[str(case["case_id"])]
                options = parse_options(row)
                label = resolve_target_label(case, model_name, target_source)
                canonical, option_text = canonical_choice(label, options)
                batch_cpu, target_meta = prepare_teacher_forced_inputs(
                    processor, tokenizer, row, label, canonical
                )
                batch = _move_inputs(batch_cpu, torch.device(device), torch.bfloat16)
                vectors, attention_meta = _attention_vectors(
                    model,
                    batch,
                    target_meta,
                    CONTENT_LAYER_WINDOWS,
                    save_individual_layers=save_individual_layers,
                    save_content_answer_heads=save_content_answer_heads,
                    save_token_maps=save_token_maps,
                )
                payload, grid_meta = _spatial_payload(vectors, batch)
                retained = case.get("retained_predictions", {}).get(model_name, {})
                record = {
                    "case_id": case["case_id"],
                    "dataset": case["dataset"],
                    "question_id": case["question_id"],
                    "model": model_name,
                    "checkpoint": checkpoint,
                    "source_checkpoint": case["models"][model_name].get("source_checkpoint"),
                    "target_source": target_source,
                    "target_label": label,
                    "target_option_text": option_text,
                    "canonical_target": canonical,
                    "ground_truth": case.get("ground_truth"),
                    "retained_parsed_answer": retained.get("parsed_answer"),
                    "retained_raw_response": retained.get("raw_response"),
                    "question": case.get("question"),
                    "image_path": case.get("image_path"),
                    "target_tokens": target_meta,
                    "attention": attention_meta,
                    "grid": grid_meta,
                    "layer_windows": CONTENT_LAYER_WINDOWS,
                    "npz_path": str(npz_path),
                    "interpretation_boundary": {
                        "content_answer_state": "teacher-forced token-conditioned grounding probe; not causal evidence for the same token",
                        "content_prediction_query": "causal query immediately before each teacher-forced content token",
                    },
                    "training": False,
                    "checkpoint_mutation": False,
                }
                write_npz(npz_path, payload)
                write_json(json_path, record)
                records.append(record)
            except Exception as exc:
                failure = {
                    "case_id": case.get("case_id"),
                    "dataset": case.get("dataset"),
                    "question_id": case.get("question_id"),
                    "model": model_name,
                    "checkpoint": checkpoint,
                    "target_source": target_source,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                write_json(artifact_dir / f"{stem}.failed.json", failure)
            finally:
                processed += 1
                if processed % 10 == 0 or processed == total_jobs:
                    gc.collect()
                    print(
                        f"[content-extract] progress={processed}/{total_jobs} "
                        f"successes={len(records)} failures={len(failures)}",
                        flush=True,
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        del model, tokenizer, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_rank = {name: index for index, name in enumerate(MODEL_ORDER)}
    records.sort(key=lambda row: (row["dataset"], row["case_id"], model_rank[row["model"]]))
    failures.sort(key=lambda row: (str(row["dataset"]), str(row["case_id"]), model_rank[row["model"]]))
    write_jsonl(destination / "attention_records.jsonl", records)
    write_jsonl(destination / "failures.jsonl", failures)
    summary = {
        "candidate_manifest": str(Path(candidates_path).resolve()),
        "output_dir": str(destination),
        "requested_cases": len(cases),
        "requested_models": list(models),
        "target_source": target_source,
        "requested_model_cases": len(cases) * len(models),
        "successful_model_cases": len(records),
        "failed_model_cases": len(failures),
        "complete_cases": sum(
            all(any(row["case_id"] == case["case_id"] and row["model"] == model for row in records) for model in models)
            for case in cases
        ),
        "save_individual_layers": bool(save_individual_layers),
        "save_content_answer_heads": bool(save_content_answer_heads),
        "save_token_maps": bool(save_token_maps),
        "query_modes": ["content_answer_state", "content_prediction_query", "first_decision_query"],
        "training": False,
        "checkpoint_mutation": False,
    }
    write_json(destination / "extraction_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate option parsing and target labels without GPU")
    validate_parser.add_argument("--candidates", required=True)
    validate_parser.add_argument("--models", default="ours,ttrv")
    validate_parser.add_argument("--target-source", choices=("ground_truth", "retained"), required=True)
    validate_parser.add_argument("--limit", type=int, default=0)

    extract_parser = subparsers.add_parser("extract", help="Run teacher-forced content-token attention inference")
    extract_parser.add_argument("--candidates", required=True)
    extract_parser.add_argument("--output-dir", required=True)
    extract_parser.add_argument("--device", default="cuda:0")
    extract_parser.add_argument("--models", default="ours,ttrv")
    extract_parser.add_argument("--target-source", choices=("ground_truth", "retained"), required=True)
    extract_parser.add_argument("--limit", type=int, default=0)
    extract_parser.add_argument("--save-individual-layers", action="store_true")
    extract_parser.add_argument("--save-content-answer-heads", action="store_true")
    extract_parser.add_argument("--save-token-maps", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    models = _parse_models(args.models)
    if args.command == "validate":
        result = validate_manifest(args.candidates, models, args.target_source, args.limit)
        if result["errors"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 2
    elif args.command == "extract":
        result = extract(
            args.candidates,
            args.output_dir,
            args.device,
            models,
            args.target_source,
            args.limit,
            args.save_individual_layers,
            args.save_content_answer_heads,
            args.save_token_maps,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
