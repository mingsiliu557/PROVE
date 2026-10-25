#!/usr/bin/env python3
"""Build an outcome-dependent retained-answer attention contrast manifest.

The selected cases satisfy all of the following:

* the retained Ours prediction equals the ground-truth option;
* the retained TTRV prediction is a valid option and differs from ground truth;
* both retained option texts contain lexical content and are neither boolean nor
  purely numeric; and
* each option text stays within the dataset-specific word limit.

This selector intentionally uses model outcomes.  Every output row and the
summary record that fact explicitly so the result cannot be mistaken for a
correctness-blind evaluation sample.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from answer_content_attention import _load_rows, canonical_choice, parse_options
from attention_viz import read_jsonl, write_json, write_jsonl
from screen_answer_content_v2 import text_lexical_gate


DATASET_MAX_WORDS = {
    "RealWorldQA": 16,
    "SEED": 4,
}
SELECTION_POLICY = (
    "strict_ours_correct_ttrv_valid_wrong_dual_retained_text_lexical_gate_v1"
)
VALID_LABEL = re.compile(r"^[A-F]$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().upper()
    return label if VALID_LABEL.fullmatch(label) else "unknown"


def load_unique_cases(path: Path) -> list[dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for source in read_jsonl(path):
        case_id = str(source.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("Candidate row is missing case_id")
        if case_id in cases:
            raise ValueError(f"Duplicate case_id in {path}: {case_id}")
        cases[case_id] = dict(source)
    return list(cases.values())


def retained_label(case: Mapping[str, Any], model: str) -> str:
    predictions = case.get("retained_predictions", {})
    if not isinstance(predictions, Mapping):
        return "unknown"
    record = predictions.get(model, {})
    if not isinstance(record, Mapping):
        return "unknown"
    return normalize_label(record.get("parsed_answer"))


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    summary_path = (
        Path(args.summary).expanduser().resolve()
        if args.summary
        else output_path.with_suffix(".summary.json")
    )
    if output_path == summary_path:
        raise ValueError("--output and --summary must be different paths")
    for path in (output_path, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")

    datasets = tuple(part.strip() for part in args.datasets.split(",") if part.strip())
    if not datasets or len(datasets) != len(set(datasets)):
        raise ValueError("--datasets must contain unique, nonempty dataset names")
    unsupported = sorted(set(datasets) - set(DATASET_MAX_WORDS))
    if unsupported:
        raise ValueError(f"No lexical word limit configured for datasets: {unsupported}")

    all_cases = load_unique_cases(input_path)
    considered = [case for case in all_cases if str(case.get("dataset")) in datasets]
    rows = _load_rows(considered)
    exclusions: dict[str, Counter[str]] = {
        dataset: Counter() for dataset in datasets
    }
    selected: list[dict[str, Any]] = []

    for case in considered:
        dataset = str(case["dataset"])
        ground_truth = normalize_label(case.get("ground_truth"))
        ours_label = retained_label(case, "ours")
        ttrv_label = retained_label(case, "ttrv")

        if ground_truth == "unknown":
            exclusions[dataset]["invalid_ground_truth"] += 1
            continue
        if ours_label == "unknown":
            exclusions[dataset]["invalid_ours_prediction"] += 1
            continue
        if ours_label != ground_truth:
            exclusions[dataset]["ours_not_correct"] += 1
            continue
        if ttrv_label == "unknown":
            exclusions[dataset]["invalid_ttrv_prediction"] += 1
            continue
        if ttrv_label == ground_truth:
            exclusions[dataset]["ttrv_not_wrong"] += 1
            continue

        try:
            options = parse_options(rows[str(case["case_id"])])
            ours_canonical, ours_text = canonical_choice(ours_label, options)
            ttrv_canonical, ttrv_text = canonical_choice(ttrv_label, options)
        except Exception as exc:
            exclusions[dataset][f"option_parse_error:{type(exc).__name__}"] += 1
            continue

        max_words = DATASET_MAX_WORDS[dataset]
        ours_gate = text_lexical_gate(ours_text, max_words=max_words)
        if not bool(ours_gate["pass"]):
            exclusions[dataset][f"ours_text:{ours_gate['reason']}"] += 1
            continue
        ttrv_gate = text_lexical_gate(ttrv_text, max_words=max_words)
        if not bool(ttrv_gate["pass"]):
            exclusions[dataset][f"ttrv_text:{ttrv_gate['reason']}"] += 1
            continue

        item = dict(case)
        item.update(
            {
                "dataset": dataset,
                "ours_target_label": ours_label,
                "ours_target_option_text": str(ours_gate["normalized"]),
                "ours_target_canonical": ours_canonical,
                "ours_target_lexical_word_count": int(ours_gate["word_count"]),
                "ttrv_target_label": ttrv_label,
                "ttrv_target_option_text": str(ttrv_gate["normalized"]),
                "ttrv_target_canonical": ttrv_canonical,
                "ttrv_target_lexical_word_count": int(ttrv_gate["word_count"]),
                "retained_target_source": "per_model_retained_prediction",
                "ours_correct": True,
                "ttrv_correct": False,
                "selection_policy": SELECTION_POLICY,
                "selection_gate": "ours_correct_ttrv_valid_wrong",
                "selection_outcome_dependent": True,
                "selection_correctness_blind": False,
                "selection_uses_model_predictions": True,
                "selection_dataset_max_option_words": max_words,
            }
        )
        selected.append(item)

    dataset_order = {dataset: index for index, dataset in enumerate(datasets)}
    selected.sort(
        key=lambda case: (
            dataset_order[str(case["dataset"])],
            str(case["case_id"]),
        )
    )
    dataset_ranks: Counter[str] = Counter()
    for global_rank, case in enumerate(selected, start=1):
        dataset = str(case["dataset"])
        dataset_ranks[dataset] += 1
        case["selection_rank"] = global_rank
        case["selection_dataset_rank"] = dataset_ranks[dataset]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, selected)

    dataset_summary: dict[str, Any] = {}
    for dataset in datasets:
        selected_count = sum(str(case["dataset"]) == dataset for case in selected)
        input_count = sum(str(case.get("dataset")) == dataset for case in all_cases)
        dataset_summary[dataset] = {
            "input_cases": input_count,
            "considered_cases": input_count,
            "selected_cases": selected_count,
            "max_option_words_per_side": DATASET_MAX_WORDS[dataset],
            "exclusion_reasons": dict(sorted(exclusions[dataset].items())),
        }

    summary = {
        "status": "complete",
        "input_manifest": str(input_path),
        "input_manifest_sha256": sha256_file(input_path),
        "output_manifest": str(output_path),
        "output_manifest_sha256": sha256_file(output_path),
        "summary_path": str(summary_path),
        "selection_policy": SELECTION_POLICY,
        "selection_outcome_dependent": True,
        "selection_correctness_blind": False,
        "selection_uses_model_predictions": True,
        "selection_rule": (
            "Ours retained prediction equals GT; TTRV retained prediction is a valid "
            "different option; both retained option texts pass the dataset-specific "
            "lexical gate. All satisfying cases are retained."
        ),
        "dataset_max_option_words_per_side": {
            dataset: DATASET_MAX_WORDS[dataset] for dataset in datasets
        },
        "input_total": len(all_cases),
        "considered_total": len(considered),
        "selected_total": len(selected),
        "datasets": dataset_summary,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Candidate JSONL manifest")
    parser.add_argument("--output", required=True, help="Strict selected JSONL")
    parser.add_argument(
        "--summary",
        default=None,
        help="Summary JSON (default: <output stem>.summary.json)",
    )
    parser.add_argument("--datasets", default="RealWorldQA,SEED")
    return parser


def main() -> int:
    build_manifest(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
