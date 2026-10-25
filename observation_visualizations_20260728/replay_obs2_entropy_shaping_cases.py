#!/usr/bin/env python3
"""Replay selected Obs2 cases for raw-density vs entropy-shaped rewards.

This script reuses fixed Stage-A N=32 responses. It does not generate responses
or train a model. Candidate embeddings are computed once, then shared by both
reward variants:

    R_base(a) = d_a
    R_entropy(a) = exp(d_a / T(H))
    T(H) = T_min + (T_max - T_min) * H

The normalized entropy-shaped share is reported for an apples-to-apples plot.
The unnormalized reward is used for candidate-level GRPO z-score diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = REPO_ROOT / "verl"
for value in (REPO_ROOT, VERL_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

HELPER_PATH = REPO_ROOT / "verl" / "scripts" / "replay_obs2_density_selection.py"
SPEC = importlib.util.spec_from_file_location("obs2_replay_helpers", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import replay helpers from {HELPER_PATH}")
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)

from verl.utils.reward_score.ttrl.ttt_metrics import (  # noqa: E402
    _density_masses_from_embeddings,
    _normalize_choice_answer,
    _normalized_entropy,
    _parse_choice_labels,
)


DEFAULT_CASES = (
    {"role": "high", "benchmark": "ai2d20", "question_id": "1796"},
    {"role": "high", "benchmark": "mathverse20_fixed", "question_id": "944"},
    {"role": "high", "benchmark": "mathverse20_fixed", "question_id": "483"},
    {"role": "high", "benchmark": "mathvista20_hq", "question_id": "38"},
    {"role": "high", "benchmark": "mathverse20_fixed", "question_id": "1439"},
    {"role": "low", "benchmark": "ai2d20", "question_id": "2346"},
    {"role": "low", "benchmark": "realworldqa20_hq", "question_id": "359"},
    {"role": "low", "benchmark": "mathvista20_hq", "question_id": "455"},
    {"role": "low", "benchmark": "realworldqa20_hq", "question_id": "588"},
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def zscore(values: list[float], eps: float = 1e-8) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return (array - float(array.mean())) / (float(array.std()) + eps)


def normalized_entropy_from_probabilities(values: list[float]) -> float:
    positive = [float(value) for value in values if float(value) > 0.0]
    if len(positive) <= 1:
        return 0.0
    total = sum(positive)
    probabilities = [value / total for value in positive]
    entropy = -sum(value * math.log(value) for value in probabilities)
    return entropy / math.log(len(probabilities))


def process_case(
    case: dict[str, str],
    records: list[dict[str, Any]],
    row: dict[str, Any],
    model: Any,
    tokenizer: Any,
    processor: Any,
    device: str,
    micro_batch_size: int,
    density_kernel_t0: float,
    t_min: float,
    t_max: float,
) -> dict[str, Any]:
    benchmark = case["benchmark"]
    qid = case["question_id"]
    role = case["role"]
    config = HELPERS.DATASET_CONFIG[benchmark]
    labels = _parse_choice_labels(config["labels"])
    parsed = [str(record["parsed_answer"]) for record in records]
    canonical = [
        _normalize_choice_answer(answer, choice_labels=config["labels"])
        for answer in parsed
    ]
    valid_indices = [index for index, answer in enumerate(canonical) if answer in labels]
    if len(valid_indices) != 32:
        raise ValueError(f"{benchmark}/{qid}: case study requires 32 valid outputs, got {len(valid_indices)}")

    counter = Counter(canonical[index] for index in valid_indices)
    answer_entropy = _normalized_entropy(counter, len(valid_indices))
    final_temperature = t_min + (t_max - t_min) * answer_entropy
    prepared, raw_prompt = HELPERS._prepare_inputs(processor, row)
    response_texts = [
        HELPERS.normalized_response_text(tokenizer, records[index]["raw_text"])
        for index in valid_indices
    ]
    queries = [
        HELPERS.evidence_query(config["scope"], raw_prompt, response)
        for response in response_texts
    ]
    unique_queries = sorted(set(queries))
    query_embeddings = HELPERS.embed_queries(
        model,
        tokenizer,
        prepared,
        unique_queries,
        device,
        micro_batch_size,
    )
    embeddings = np.stack([query_embeddings[query] for query in queries], axis=0)

    candidate_density, _, sim_mean, sim_std = _density_masses_from_embeddings(
        embeddings,
        density_kernel_t0,
    )
    label_density = {label: 0.0 for label in labels if counter.get(label, 0) > 0}
    for answer, density in zip((canonical[index] for index in valid_indices), candidate_density):
        label_density[answer] += float(density)
    density_total = sum(label_density.values())
    if not math.isclose(density_total, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError(f"{benchmark}/{qid}: label density sums to {density_total}")

    entropy_reward_raw = {
        label: math.exp(value / max(final_temperature, 1e-8))
        for label, value in label_density.items()
    }
    entropy_reward_total = sum(entropy_reward_raw.values())
    entropy_reward_share = {
        label: value / entropy_reward_total
        for label, value in entropy_reward_raw.items()
    }

    candidate_base_rewards = [label_density[canonical[index]] for index in valid_indices]
    candidate_entropy_rewards = [entropy_reward_raw[canonical[index]] for index in valid_indices]
    candidate_base_advantages = zscore(candidate_base_rewards)
    candidate_entropy_advantages = zscore(candidate_entropy_rewards)
    base_advantage = {
        label: float(np.mean([
            candidate_base_advantages[position]
            for position, index in enumerate(valid_indices)
            if canonical[index] == label
        ]))
        for label in label_density
    }
    entropy_advantage = {
        label: float(np.mean([
            candidate_entropy_advantages[position]
            for position, index in enumerate(valid_indices)
            if canonical[index] == label
        ]))
        for label in label_density
    }

    sorted_counter = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    if len(sorted_counter) > 1 and sorted_counter[0][1] == sorted_counter[1][1]:
        raise ValueError(f"{benchmark}/{qid}: majority must be strict for this case study")
    majority_label = sorted_counter[0][0]
    ground_truth = _normalize_choice_answer(records[0]["ground_truth"], choice_labels=config["labels"])
    if ground_truth not in label_density:
        raise ValueError(f"{benchmark}/{qid}: ground truth {ground_truth} is absent from valid outputs")

    wrong_labels = [label for label in label_density if label != ground_truth]
    strongest_wrong = max(wrong_labels, key=lambda label: label_density[label])
    if role == "high":
        if majority_label == ground_truth:
            raise ValueError(f"{benchmark}/{qid}: high-entropy case needs wrong consensus")
        comparison_label = majority_label
        base_margin = label_density[comparison_label] - label_density[ground_truth]
        entropy_margin = entropy_reward_share[comparison_label] - entropy_reward_share[ground_truth]
        base_advantage_margin = base_advantage[comparison_label] - base_advantage[ground_truth]
        entropy_advantage_margin = entropy_advantage[comparison_label] - entropy_advantage[ground_truth]
        effect_name = "wrong_consensus_minus_ground_truth"
        desired_effect = entropy_margin < base_margin
    elif role == "low":
        if majority_label != ground_truth:
            raise ValueError(f"{benchmark}/{qid}: low-entropy case needs correct consensus")
        comparison_label = strongest_wrong
        base_margin = label_density[ground_truth] - label_density[comparison_label]
        entropy_margin = entropy_reward_share[ground_truth] - entropy_reward_share[comparison_label]
        base_advantage_margin = base_advantage[ground_truth] - base_advantage[comparison_label]
        entropy_advantage_margin = entropy_advantage[ground_truth] - entropy_advantage[comparison_label]
        effect_name = "ground_truth_minus_strongest_wrong"
        desired_effect = entropy_margin > base_margin
    else:
        raise ValueError(f"Unsupported role: {role}")

    labels_sorted = sorted(label_density)
    label_rows = []
    for label in labels_sorted:
        label_rows.append(
            {
                "label": label,
                "count": int(counter[label]),
                "frequency": float(counter[label] / len(valid_indices)),
                "density_reward": float(label_density[label]),
                "entropy_reward_raw": float(entropy_reward_raw[label]),
                "entropy_reward_share": float(entropy_reward_share[label]),
                "density_advantage": float(base_advantage[label]),
                "entropy_advantage": float(entropy_advantage[label]),
                "is_ground_truth": label == ground_truth,
                "is_consensus": label == majority_label,
            }
        )

    return {
        "role": role,
        "benchmark": benchmark,
        "question_id": qid,
        "ground_truth": ground_truth,
        "majority_label": majority_label,
        "majority_count": int(counter[majority_label]),
        "valid_count": len(valid_indices),
        "valid_counter": dict(counter),
        "answer_entropy": float(answer_entropy),
        "density_kernel_temperature": float(density_kernel_t0),
        "entropy_temperature": float(final_temperature),
        "temperature_min": float(t_min),
        "temperature_max": float(t_max),
        "embedding_scope": config["scope"],
        "unique_embedding_queries": len(unique_queries),
        "similarity_mean": float(sim_mean),
        "similarity_std": float(sim_std),
        "base_formula": "R_base(a) = d_a",
        "entropy_formula": "R_entropy(a) = exp(d_a / T(H))",
        "normalization_note": "entropy_reward_share is normalized over observed labels only; this group-constant scaling does not change GRPO z-scores",
        "effect_name": effect_name,
        "comparison_label": comparison_label,
        "base_margin": float(base_margin),
        "entropy_margin": float(entropy_margin),
        "margin_change": float(entropy_margin - base_margin),
        "base_advantage_margin": float(base_advantage_margin),
        "entropy_advantage_margin": float(entropy_advantage_margin),
        "advantage_margin_change": float(entropy_advantage_margin - base_advantage_margin),
        "desired_effect": bool(desired_effect),
        "density_reward_entropy": normalized_entropy_from_probabilities(list(label_density.values())),
        "entropy_shaped_reward_entropy": normalized_entropy_from_probabilities(list(entropy_reward_share.values())),
        "label_rewards": label_rows,
        "target_image_path": records[0]["target_image_path"],
        "prompt": records[0]["prompt"],
        "raw_responses": [records[index]["raw_text"] for index in valid_indices],
        "parsed_answers": [canonical[index] for index in valid_indices],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--density-kernel-t0", type=float, default=0.20)
    parser.add_argument("--temperature-min", type=float, default=0.10)
    parser.add_argument("--temperature-max", type=float, default=0.30)
    args = parser.parse_args()

    parquet_paths = HELPERS.load_manifest(args.manifest)
    grouped_by_benchmark = {
        benchmark: HELPERS.load_dump_groups(args.dump_root / benchmark / "real.jsonl")
        for benchmark in sorted({case["benchmark"] for case in DEFAULT_CASES})
    }
    parquet_by_benchmark = {
        benchmark: HELPERS.load_parquet_rows(parquet_paths[benchmark])
        for benchmark in grouped_by_benchmark
    }
    model, tokenizer, processor = HELPERS.load_model(args.base_model, args.device)

    results = []
    for position, case in enumerate(DEFAULT_CASES, start=1):
        benchmark = case["benchmark"]
        qid = case["question_id"]
        result = process_case(
            case,
            grouped_by_benchmark[benchmark][qid],
            parquet_by_benchmark[benchmark][qid],
            model,
            tokenizer,
            processor,
            args.device,
            args.micro_batch_size,
            args.density_kernel_t0,
            args.temperature_min,
            args.temperature_max,
        )
        results.append(result)
        print(
            f"[case {position}/{len(DEFAULT_CASES)}] {benchmark}/{qid} "
            f"role={case['role']} H={result['answer_entropy']:.4f} "
            f"margin={result['base_margin']:.6f}->{result['entropy_margin']:.6f} "
            f"desired={result['desired_effect']}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(args.output_dir / "candidate_case_replay.jsonl", results)
    summary_rows = [
        {
            "role": result["role"],
            "benchmark": result["benchmark"],
            "question_id": result["question_id"],
            "ground_truth": result["ground_truth"],
            "majority_label": result["majority_label"],
            "answer_entropy": result["answer_entropy"],
            "entropy_temperature": result["entropy_temperature"],
            "base_margin": result["base_margin"],
            "entropy_margin": result["entropy_margin"],
            "margin_change": result["margin_change"],
            "base_advantage_margin": result["base_advantage_margin"],
            "entropy_advantage_margin": result["entropy_advantage_margin"],
            "advantage_margin_change": result["advantage_margin_change"],
            "desired_effect": result["desired_effect"],
        }
        for result in results
    ]
    atomic_csv(args.output_dir / "candidate_case_summary.csv", summary_rows)
    atomic_json(
        args.output_dir / "run_summary.json",
        {
            "status": "complete",
            "cases": len(results),
            "desired_effect_cases": sum(bool(result["desired_effect"]) for result in results),
            "response_generation": False,
            "training": False,
            "density_kernel_temperature": args.density_kernel_t0,
            "temperature_min": args.temperature_min,
            "temperature_max": args.temperature_max,
            "formula": {
                "base": "R_base(a) = d_a",
                "entropy": "R_entropy(a) = exp(d_a / T(H))",
                "temperature": "T(H) = T_min + (T_max - T_min) * H",
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
