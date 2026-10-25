#!/usr/bin/env python3
"""Hardened entry point for answer-content attention extraction.

This module reuses the validated v1 data/token/grid path and replaces two
functions: per-head extraction now saves both the token-conditioned answer
state and the preceding causal prediction query; manifest validation counts
invalid retained targets per model rather than stopping at the first invalid
model in a case.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import answer_content_attention as base


def attention_vectors_v2(
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
    answer_head_mode = "content_answer_state_heads"
    prediction_head_mode = "content_prediction_query_heads"
    if save_content_answer_heads:
        result[answer_head_mode] = {}
        result[prediction_head_mode] = {}

    for window_name, layer_indices in layer_windows.items():
        layers = [outputs.attentions[int(layer_index)][0].float() for layer_index in layer_indices]
        stacked = torch.stack(layers, dim=0)
        for mode, positions in query_modes:
            cross = stacked[:, :, positions, :][..., image_positions]
            result[mode][window_name] = cross.mean(dim=(0, 1, 2)).detach().cpu().numpy().astype(np.float32)

    if save_individual_layers or save_content_answer_heads:
        for layer_index, layer_attention in enumerate(outputs.attentions):
            layer = layer_attention[0].float()
            if save_individual_layers:
                for mode, positions in query_modes:
                    cross = layer[:, positions, :][..., image_positions]
                    result[mode][f"layer_{layer_index:02d}"] = (
                        cross.mean(dim=(0, 1)).detach().cpu().numpy().astype(np.float32)
                    )
            if save_content_answer_heads:
                answer_heads = layer[:, answer_positions, :][..., image_positions].mean(dim=1)
                prediction_heads = layer[:, prediction_positions, :][..., image_positions].mean(dim=1)
                for head_index, vector in enumerate(answer_heads):
                    result[answer_head_mode][f"layer_{layer_index:02d}_head_{head_index:02d}"] = (
                        vector.detach().cpu().numpy().astype(np.float32)
                    )
                for head_index, vector in enumerate(prediction_heads):
                    result[prediction_head_mode][f"layer_{layer_index:02d}_head_{head_index:02d}"] = (
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
        "saved_content_prediction_heads": bool(save_content_answer_heads),
        "content_answer_head_key_format": (
            f"{answer_head_mode}__layer_{{layer:02d}}_head_{{head:02d}}"
            if save_content_answer_heads
            else None
        ),
        "content_prediction_head_key_format": (
            f"{prediction_head_mode}__layer_{{layer:02d}}_head_{{head:02d}}"
            if save_content_answer_heads
            else None
        ),
    }
    del outputs
    return result, metadata


def validate_manifest_v2(
    candidates_path: str | Path,
    models: Sequence[str],
    target_source: str,
    limit: int,
) -> dict[str, Any]:
    cases = base.read_jsonl(candidates_path)
    if limit > 0:
        cases = cases[:limit]
    rows = base._load_rows(cases)
    targets = 0
    errors: list[dict[str, Any]] = []
    complete_by_dataset: dict[str, int] = defaultdict(int)
    for case in cases:
        row = rows[str(case["case_id"])]
        case_valid = True
        try:
            options = base.parse_options(row)
        except Exception as exc:
            errors.append(
                {
                    "case_id": case.get("case_id"),
                    "dataset": case.get("dataset"),
                    "model": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        for model_name in models:
            try:
                label = base.resolve_target_label(case, model_name, target_source)
                base.canonical_choice(label, options)
                targets += 1
            except Exception as exc:
                case_valid = False
                errors.append(
                    {
                        "case_id": case.get("case_id"),
                        "dataset": case.get("dataset"),
                        "model": model_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        if case_valid:
            complete_by_dataset[str(case["dataset"])] += 1
    return {
        "candidate_manifest": str(Path(candidates_path).resolve()),
        "cases": len(cases),
        "models": list(models),
        "target_source": target_source,
        "valid_targets": targets,
        "invalid_targets": len(errors),
        "complete_cases_by_dataset": dict(sorted(complete_by_dataset.items())),
        "errors": errors,
    }


base._attention_vectors = attention_vectors_v2
base.validate_manifest = validate_manifest_v2


if __name__ == "__main__":
    raise SystemExit(base.main())

