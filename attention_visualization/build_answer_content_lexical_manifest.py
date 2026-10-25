#!/usr/bin/env python3
"""Build a deterministic, model-outcome-blind lexical attention manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from answer_content_attention import (
    _load_rows,
    canonical_choice,
    parse_options,
    resolve_target_label,
)
from attention_viz import read_jsonl, write_json, write_jsonl
from screen_answer_content_v2 import text_lexical_gate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selection_hash(case: Mapping[str, Any], salt: str) -> str:
    material = (
        salt
        + "\0"
        + str(case.get("dataset", ""))
        + "\0"
        + str(case.get("case_id", ""))
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def load_unique_cases(paths: Sequence[Path]) -> list[dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for path in paths:
        for case in read_jsonl(path):
            case_id = str(case["case_id"])
            if case_id in cases:
                raise ValueError(f"Duplicate case_id across inputs: {case_id}")
            cases[case_id] = dict(case)
    return list(cases.values())


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    inputs = [Path(value).expanduser().resolve() for value in args.candidate_manifests]
    output = Path(args.output).expanduser().resolve()
    summary_path = output.with_suffix(".summary.json")
    if output.exists() or summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {summary_path}")
    if not 0.0 <= args.visual_fraction <= 1.0:
        raise ValueError("--visual-fraction must lie in [0,1]")
    datasets = tuple(part.strip() for part in args.datasets.split(",") if part.strip())
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("--datasets must contain unique nonempty names")

    cases = load_unique_cases(inputs)
    cases = [case for case in cases if str(case.get("dataset")) in datasets]
    rows = _load_rows(cases)
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusions: dict[str, Counter[str]] = defaultdict(Counter)

    for case in cases:
        dataset = str(case["dataset"])
        try:
            options = parse_options(rows[str(case["case_id"])])
            label = resolve_target_label(case, "ours", "ground_truth")
            _, target_text = canonical_choice(label, options)
            gate = text_lexical_gate(target_text, max_words=args.max_option_words)
            if not bool(gate["pass"]):
                exclusions[dataset][str(gate["reason"])] += 1
                continue
            item = dict(case)
            item["target_option_text_for_selection"] = target_text
            item["target_lexical_word_count"] = int(gate["word_count"])
            item["selection_hash"] = selection_hash(item, args.selection_salt)
            eligible[dataset].append(item)
        except Exception as exc:
            exclusions[dataset][f"parse_error:{type(exc).__name__}"] += 1

    selected: list[dict[str, Any]] = []
    dataset_summary: dict[str, Any] = {}
    for dataset in datasets:
        pool = eligible.get(dataset, [])
        if len(pool) < args.per_dataset_limit:
            raise RuntimeError(
                f"{dataset} has only {len(pool)} lexical cases; "
                f"need {args.per_dataset_limit}"
            )
        visual_count = int(math.ceil(args.per_dataset_limit * args.visual_fraction))
        by_visual = sorted(
            pool,
            key=lambda case: (
                -int(case.get("visual_keyword_score", 0)),
                str(case["selection_hash"]),
                str(case["question_id"]),
            ),
        )
        visual = by_visual[:visual_count]
        visual_ids = {str(case["case_id"]) for case in visual}
        by_hash = sorted(
            (case for case in pool if str(case["case_id"]) not in visual_ids),
            key=lambda case: (
                str(case["selection_hash"]),
                str(case["question_id"]),
            ),
        )
        hashed = by_hash[: args.per_dataset_limit - len(visual)]
        for source, group in (("visual_keyword", visual), ("stable_hash", hashed)):
            for rank, source_case in enumerate(group, start=1):
                case = dict(source_case)
                case["shortlisted"] = True
                case["selection_source"] = source
                case["selection_source_rank"] = rank
                case["selection_correctness_blind"] = True
                case["selection_uses_model_predictions"] = False
                case["selection_policy"] = (
                    "lexical_gate_then_50pct_visual_keyword_50pct_sha256_v1"
                )
                selected.append(case)
        dataset_summary[dataset] = {
            "input_cases": sum(
                str(case.get("dataset")) == dataset for case in cases
            ),
            "eligible_lexical_cases": len(pool),
            "selected_cases": len(visual) + len(hashed),
            "selected_visual_keyword": len(visual),
            "selected_stable_hash": len(hashed),
            "exclusion_reasons": dict(sorted(exclusions[dataset].items())),
        }

    selected.sort(
        key=lambda case: (
            datasets.index(str(case["dataset"])),
            0 if case["selection_source"] == "visual_keyword" else 1,
            int(case["selection_source_rank"]),
            str(case["case_id"]),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, selected)
    summary = {
        "status": "complete",
        "candidate_manifests": [str(path) for path in inputs],
        "candidate_manifest_sha256": {
            str(path): sha256_file(path) for path in inputs
        },
        "output_manifest": str(output),
        "output_manifest_sha256": sha256_file(output),
        "selection_salt": args.selection_salt,
        "selection_correctness_blind": True,
        "selection_uses_model_predictions": False,
        "ground_truth_use": (
            "GT option text is used only for lexical eligibility and later "
            "teacher-forced query construction; model correctness is not used."
        ),
        "max_option_words": args.max_option_words,
        "per_dataset_limit": args.per_dataset_limit,
        "visual_fraction": args.visual_fraction,
        "selected_total": len(selected),
        "datasets": dataset_summary,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifests", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", default="RealWorldQA,SEED")
    parser.add_argument("--per-dataset-limit", type=int, default=200)
    parser.add_argument("--max-option-words", type=int, default=4)
    parser.add_argument("--visual-fraction", type=float, default=0.5)
    parser.add_argument(
        "--selection-salt",
        default="obs3_answer_content_rwq_seed_20260731_v1",
    )
    return parser


def main() -> int:
    build_manifest(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
