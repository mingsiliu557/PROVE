#!/usr/bin/env python3
"""Probe current Ours/TTRV checkpoints and build retained-answer contrasts.

Only ``ours`` and ``ttrv`` are decoded. Selection is deliberately
outcome-dependent: current Ours must be correct, current TTRV must be a valid
wrong choice, and both retained option texts must pass a lexical gate. The
result is directly consumable by ``answer_content_attention_v2.py
--target-source retained``.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence

from answer_content_attention import _load_rows, parse_options
from attention_viz import (
    _load_model_and_processor,
    _manual_greedy,
    _move_inputs,
    _official_choice_parse,
    _prepare_inputs,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from screen_answer_content_v2 import text_lexical_gate


MODELS = ("ours", "ttrv")
DATASET_MAX_OPTION_WORDS = {"RealWorldQA": 16, "SEED": 4}
DEFAULT_SELECTION_SALT = "retained_contrast_current_checkpoint_20260731_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stable_selection_hash(case: Mapping[str, Any], salt: str) -> str:
    material = (
        salt
        + "\0"
        + str(case.get("dataset", ""))
        + "\0"
        + str(case.get("case_id", ""))
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def validate_manifest(cases: Sequence[Mapping[str, Any]]) -> None:
    if not cases:
        raise ValueError("Candidate manifest is empty")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        case_id = str(case.get("case_id", ""))
        if not case_id:
            raise ValueError(f"Input row {index} has no case_id")
        if case_id in seen:
            raise ValueError(f"Duplicate case_id in input: {case_id}")
        seen.add(case_id)
        dataset = str(case.get("dataset", ""))
        if dataset not in DATASET_MAX_OPTION_WORDS:
            raise ValueError(
                f"Unsupported dataset={dataset!r} for {case_id}; expected "
                f"{sorted(DATASET_MAX_OPTION_WORDS)}"
            )
        if not str(case.get("ground_truth", "")).strip():
            raise ValueError(f"No ground_truth for {case_id}")
        if "historical_retained_predictions" in case:
            raise ValueError(
                f"{case_id} already has historical_retained_predictions; "
                "use the original all-held-out manifest to avoid provenance loss"
            )
        if not isinstance(case.get("retained_predictions"), Mapping):
            raise ValueError(f"No retained_predictions mapping for {case_id}")
        models = case.get("models")
        if not isinstance(models, Mapping):
            raise ValueError(f"No models mapping for {case_id}")
        for model_name in MODELS:
            spec = models.get(model_name)
            if not isinstance(spec, Mapping) or not spec.get("checkpoint"):
                raise ValueError(f"No {model_name} checkpoint for {case_id}")


def _record_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record.get("case_id", "")), str(record.get("model", ""))


def updated_retained_prediction(
    record: Mapping[str, Any],
    *,
    probe_records_path: Path,
    input_manifest: Path,
    input_manifest_sha256: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    return {
        "raw_response": str(record["raw_response"]),
        "parsed_answer": str(record["parsed_answer"]),
        "checkpoint": str(record["checkpoint"]),
        "source_checkpoint": str(record.get("source_checkpoint", "")),
        "provenance": {
            "kind": "current_checkpoint_greedy_probe",
            "current_checkpoint_aligned": True,
            "decode": "deterministic_greedy",
            "max_new_tokens": int(max_new_tokens),
            "generated_token_ids": list(record.get("generated_token_ids", [])),
            "generated_tokens": list(record.get("generated_tokens", [])),
            "probe_records": str(probe_records_path.resolve()),
            "input_manifest": str(input_manifest.resolve()),
            "input_manifest_sha256": input_manifest_sha256,
        },
    }


def select_exact_candidates(
    cases: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    options_by_case: Mapping[str, Mapping[str, str]],
    *,
    max_per_dataset: int,
    selection_salt: str,
    probe_records_path: Path,
    input_manifest: Path,
    input_manifest_sha256: str,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the strict outcome/lexical gate and construct the new manifest."""

    if max_per_dataset < 0:
        raise ValueError("--max-per-dataset must be >= 0")
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    duplicate_records: list[tuple[str, str]] = []
    for record in records:
        key = _record_key(record)
        if key in indexed:
            duplicate_records.append(key)
        indexed[key] = record
    if duplicate_records:
        raise ValueError(f"Duplicate probe records: {duplicate_records[:10]}")

    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dataset_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for source_case in cases:
        case_id = str(source_case["case_id"])
        dataset = str(source_case["dataset"])
        counts = dataset_counts[dataset]
        counts["input_cases"] += 1
        ours = indexed.get((case_id, "ours"))
        ttrv = indexed.get((case_id, "ttrv"))
        if ours is None or ttrv is None:
            counts["excluded_incomplete_probe"] += 1
            continue
        gt = str(source_case["ground_truth"]).strip().upper()
        ours_label = str(ours.get("parsed_answer", "unknown")).strip().upper()
        ttrv_label = str(ttrv.get("parsed_answer", "unknown")).strip().upper()
        options = options_by_case.get(case_id)
        if not options:
            counts["excluded_missing_options"] += 1
            continue
        valid_labels = set(options)
        if ours_label != gt:
            counts["excluded_ours_not_correct"] += 1
            continue
        if ttrv_label not in valid_labels:
            counts["excluded_ttrv_invalid"] += 1
            continue
        if ttrv_label == gt:
            counts["excluded_ttrv_not_wrong"] += 1
            continue
        if ours_label not in valid_labels:
            counts["excluded_ours_label_absent"] += 1
            continue

        max_words = DATASET_MAX_OPTION_WORDS[dataset]
        ours_text = str(options[ours_label])
        ttrv_text = str(options[ttrv_label])
        ours_gate = text_lexical_gate(ours_text, max_words=max_words)
        ttrv_gate = text_lexical_gate(ttrv_text, max_words=max_words)
        if not bool(ours_gate["pass"]):
            counts[f"excluded_ours_lexical:{ours_gate['reason']}"] += 1
            continue
        if not bool(ttrv_gate["pass"]):
            counts[f"excluded_ttrv_lexical:{ttrv_gate['reason']}"] += 1
            continue
        ours_normalized = str(ours_gate["normalized"]).casefold()
        ttrv_normalized = str(ttrv_gate["normalized"]).casefold()
        if ours_normalized == ttrv_normalized:
            counts["excluded_identical_normalized_target_text"] += 1
            continue

        case = copy.deepcopy(dict(source_case))
        historical = copy.deepcopy(dict(case["retained_predictions"]))
        case["historical_retained_predictions"] = historical
        retained = copy.deepcopy(historical)
        retained["ours"] = updated_retained_prediction(
            ours, probe_records_path=probe_records_path,
            input_manifest=input_manifest,
            input_manifest_sha256=input_manifest_sha256,
            max_new_tokens=max_new_tokens,
        )
        retained["ttrv"] = updated_retained_prediction(
            ttrv, probe_records_path=probe_records_path,
            input_manifest=input_manifest,
            input_manifest_sha256=input_manifest_sha256,
            max_new_tokens=max_new_tokens,
        )
        case["retained_predictions"] = retained
        case["shortlisted"] = True
        case["strict_eligible"] = True
        case["selection_outcome_dependent"] = True
        case["selection_current_checkpoint_aligned"] = True
        case["selection_correctness_blind"] = False
        case["selection_uses_model_predictions"] = True
        case["selection_policy"] = (
            "current_greedy_ours_correct_ttrv_valid_wrong_dual_lexical_v1"
        )
        case["selection_hash"] = stable_selection_hash(case, selection_salt)
        case["ours_target_label"] = ours_label
        case["ours_target_option_text"] = str(ours_gate["normalized"])
        case["ours_target_lexical_word_count"] = int(ours_gate["word_count"])
        case["ttrv_target_label"] = ttrv_label
        case["ttrv_target_option_text"] = str(ttrv_gate["normalized"])
        case["ttrv_target_lexical_word_count"] = int(ttrv_gate["word_count"])
        case["target_texts_are_different"] = True
        eligible[dataset].append(case)
        counts["eligible_before_cap"] += 1

    selected: list[dict[str, Any]] = []
    for dataset in sorted(DATASET_MAX_OPTION_WORDS):
        ranked = sorted(
            eligible.get(dataset, []),
            key=lambda case: (str(case["selection_hash"]), str(case["case_id"])),
        )
        if max_per_dataset > 0:
            ranked = ranked[:max_per_dataset]
        for rank, case in enumerate(ranked, start=1):
            case["selection_rank_within_dataset"] = rank
            selected.append(case)
        dataset_counts[dataset]["selected_after_cap"] = len(ranked)

    summary = {
        "selection_outcome_dependent": True,
        "selection_current_checkpoint_aligned": True,
        "selection_correctness_blind": False,
        "selection_uses_model_predictions": True,
        "selection_policy": (
            "current_greedy_ours_correct_ttrv_valid_wrong_dual_lexical_v1"
        ),
        "selection_salt": selection_salt,
        "max_per_dataset": max_per_dataset,
        "max_per_dataset_zero_means": "retain all eligible cases",
        "lexical_max_words": dict(DATASET_MAX_OPTION_WORDS),
        "selected_total": len(selected),
        "datasets": {
            dataset: dict(sorted(dataset_counts[dataset].items()))
            for dataset in sorted(DATASET_MAX_OPTION_WORDS)
        },
    }
    return selected, summary


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    # The shared loader does not pass local_files_only, so force offline model
    # resolution before importing torch/transformers through that helper.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch

    input_manifest = Path(args.candidates).expanduser().resolve()
    if not input_manifest.is_file():
        raise FileNotFoundError(input_manifest)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if args.max_per_dataset < 0:
        raise ValueError("--max-per-dataset must be >= 0")

    cases = read_jsonl(input_manifest)
    validate_manifest(cases)
    # Load every requested parquet row once, grouped by parquet path.  The
    # previous per-case fallback repeatedly reopened/scanned the same parquet
    # files while looping over checkpoints.
    rows_by_case = _load_rows(cases)
    options_by_case = {
        str(case["case_id"]): parse_options(rows_by_case[str(case["case_id"])])
        for case in cases
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / "probe_records.jsonl"
    failures_path = output_dir / "failures.jsonl"
    exact_path = output_dir / "exact_candidates.jsonl"
    summary_path = output_dir / "probe_summary.json"
    manifest_hash = sha256_file(input_manifest)

    jobs_by_checkpoint: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for case in cases:
        for model_name in MODELS:
            checkpoint = str(
                Path(case["models"][model_name]["checkpoint"])
                .expanduser()
                .resolve()
            )
            jobs_by_checkpoint[checkpoint].append((dict(case), model_name))

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started_at = utc_now()

    for checkpoint in sorted(jobs_by_checkpoint):
        jobs = jobs_by_checkpoint[checkpoint]
        print(
            f"[retained-probe] loading checkpoint={checkpoint} jobs={len(jobs)}",
            flush=True,
        )
        try:
            model, tokenizer, processor = _load_model_and_processor(
                checkpoint, args.device
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(
                f"[retained-probe] CHECKPOINT LOAD FAILURE {checkpoint}: {message}",
                file=sys.stderr,
                flush=True,
            )
            stack = traceback.format_exc()
            for case, model_name in jobs:
                failures.append(
                    {
                        "case_id": case["case_id"],
                        "dataset": case["dataset"],
                        "question_id": case["question_id"],
                        "model": model_name,
                        "checkpoint": checkpoint,
                        "stage": "load_checkpoint",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": stack,
                    }
                )
            continue

        for job_index, (case, model_name) in enumerate(jobs, start=1):
            case_id = str(case["case_id"])
            try:
                row = rows_by_case[case_id]
                batch_cpu, raw_prompt = _prepare_inputs(processor, row)
                batch = _move_inputs(
                    batch_cpu, torch.device(args.device), torch.bfloat16
                )
                generated_ids, response = _manual_greedy(
                    model,
                    tokenizer,
                    batch,
                    max_new_tokens=args.max_new_tokens,
                )
                parsed = _official_choice_parse(
                    response,
                    row,
                    parser=str(case["parser"]),
                    labels=str(case["labels"]),
                    fallback_task=str(case["task"]),
                )
                source_checkpoint = str(
                    case["models"][model_name].get("source_checkpoint", "")
                )
                records.append(
                    {
                        "case_id": case_id,
                        "dataset": case["dataset"],
                        "question_id": case["question_id"],
                        "model": model_name,
                        "checkpoint": checkpoint,
                        "source_checkpoint": source_checkpoint,
                        "raw_prompt": raw_prompt,
                        "raw_response": response,
                        "parsed_answer": parsed,
                        "ground_truth": str(case["ground_truth"]),
                        "correct": parsed == str(case["ground_truth"]),
                        "generated_token_ids": generated_ids,
                        "generated_tokens": tokenizer.convert_ids_to_tokens(
                            generated_ids
                        ),
                        "decode": "deterministic_greedy",
                        "max_new_tokens": int(args.max_new_tokens),
                        "provenance": "current_manifest_checkpoint",
                    }
                )
            except Exception as exc:
                failure = {
                    "case_id": case_id,
                    "dataset": case["dataset"],
                    "question_id": case["question_id"],
                    "model": model_name,
                    "checkpoint": checkpoint,
                    "stage": "greedy_decode",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                print(
                    f"[retained-probe] CASE FAILURE {case_id}/{model_name}: "
                    f"{failure['error_type']}: {failure['error']}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if job_index % 25 == 0 or job_index == len(jobs):
                print(
                    f"[retained-probe] checkpoint progress={job_index}/{len(jobs)}",
                    flush=True,
                )
        del model, tokenizer, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_order = {name: index for index, name in enumerate(MODELS)}
    records.sort(
        key=lambda item: (
            str(item["dataset"]),
            str(item["case_id"]),
            model_order[str(item["model"])],
        )
    )
    failures.sort(
        key=lambda item: (
            str(item["dataset"]),
            str(item["case_id"]),
            model_order.get(str(item["model"]), 99),
        )
    )
    write_jsonl(records_path, records)
    write_jsonl(failures_path, failures)

    selected, selection_summary = select_exact_candidates(
        cases,
        records,
        options_by_case,
        max_per_dataset=args.max_per_dataset,
        selection_salt=args.selection_salt,
        probe_records_path=records_path,
        input_manifest=input_manifest,
        input_manifest_sha256=manifest_hash,
        max_new_tokens=args.max_new_tokens,
    )
    write_jsonl(exact_path, selected)

    expected_model_cases = len(cases) * len(MODELS)
    record_keys = {_record_key(record) for record in records}
    complete_case_ids = {
        str(case["case_id"])
        for case in cases
        if all((str(case["case_id"]), model_name) in record_keys for model_name in MODELS)
    }
    status = (
        "complete"
        if not failures and len(records) == expected_model_cases
        else "incomplete"
    )
    summary = {
        "status": status,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": manifest_hash,
        "output_dir": str(output_dir),
        "models_loaded": list(MODELS),
        "base_model_loaded": False,
        "checkpoint_job_counts": {
            checkpoint: len(jobs)
            for checkpoint, jobs in sorted(jobs_by_checkpoint.items())
        },
        "device": args.device,
        "decode": "deterministic_greedy",
        "max_new_tokens": int(args.max_new_tokens),
        "requested_cases": len(cases),
        "expected_model_cases": expected_model_cases,
        "successful_model_cases": len(records),
        "failed_model_cases": len(failures),
        "complete_cases": len(complete_case_ids),
        "probe_records": str(records_path),
        "probe_records_sha256": sha256_file(records_path),
        "failures": str(failures_path),
        "failures_sha256": sha256_file(failures_path),
        "exact_candidates": str(exact_path),
        "exact_candidates_sha256": sha256_file(exact_path),
        "historical_retained_predictions_preserved": True,
        "retained_predictions_updated_from_current_probe": True,
        **selection_summary,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if status != "complete":
        raise RuntimeError(
            f"Probe incomplete: {len(failures)} failures; artifacts were saved "
            f"under {output_dir} and will not be silently reused"
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=200,
        help="Cap exact cases per dataset; 0 keeps every eligible case",
    )
    parser.add_argument("--selection-salt", default=DEFAULT_SELECTION_SALT)
    return parser


def main() -> int:
    run_probe(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
