#!/usr/bin/env python3
"""Build an outcome-blind held-out lexical grounding pool from a frozen screen.

This companion deliberately does not re-run global setting selection.  It reads
and verifies ``selection_config.json``, reconstructs only the attention features
needed at that frozen setting, freezes both rankings, and only then loads model
predictions/outcomes for annotation columns.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import screen_answer_content_v2 as core


POOL_NAME = "lexical_grounding_pool_ranked.csv"
RENDER_SUBSET_NAME = "lexical_grounding_render_subset.csv"
SUMMARY_NAME = "grounding_pool_summary.json"


def prepare_paths(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "grounding_pool": output_dir / POOL_NAME,
        "render_subset": output_dir / RENDER_SUBSET_NAME,
        "summary": output_dir / SUMMARY_NAME,
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing grounding outputs: {existing}"
        )
    return paths


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def verify_frozen_config(
    config_path: Path,
) -> tuple[dict[str, Any], Path, Path, str, int, int, int]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs = require_mapping(config.get("inputs"), "config.inputs")
    records_path = Path(str(inputs["records"])).expanduser().resolve()
    manifest_path = Path(str(inputs["manifest"])).expanduser().resolve()
    if core.sha256_file(records_path) != str(inputs["records_sha256"]):
        raise ValueError("records SHA256 differs from frozen selection config")
    if core.sha256_file(manifest_path) != str(inputs["manifest_sha256"]):
        raise ValueError("manifest SHA256 differs from frozen selection config")
    frozen_script = Path(str(config["script"])).expanduser().resolve()
    if frozen_script != Path(core.__file__).resolve():
        raise ValueError(
            f"Unexpected frozen screen script: {frozen_script} != {core.__file__}"
        )
    if core.sha256_file(frozen_script) != str(config["script_sha256"]):
        raise ValueError("frozen screen script SHA256 no longer matches config")
    if not bool(config.get("selection_setting_shared_across_models_and_cases")):
        raise ValueError("config does not declare one shared global setting")
    if bool(config.get("per_case_head_selection")):
        raise ValueError("per-case head selection is forbidden")
    selection_query = str(config["selection_query_mode"])
    if selection_query not in core.QUERY_MODES:
        raise ValueError(f"Unknown frozen selection query: {selection_query}")
    selected = require_mapping(config.get("selected_setting"), "selected_setting")
    head = int(selected["head_index_zero_based"])
    center = int(selected["layer_center_zero_based"])
    setting_search = require_mapping(config.get("setting_search"), "setting_search")
    radius = int(setting_search["window_radius"])
    expected_window = list(range(center - radius, center + radius + 1))
    if list(selected["layer_window_zero_based"]) != expected_window:
        raise ValueError("frozen layer window is inconsistent with center/radius")
    if not (0 <= head < 12 and radius <= center < 28 - radius):
        raise ValueError("frozen setting lies outside the saved v2 tensor range")
    return (
        config,
        records_path,
        manifest_path,
        selection_query,
        head,
        center,
        radius,
    )


def populated_outcome_blind_row(
    case_id: str,
    static: Mapping[str, Any],
    records: Mapping[tuple[str, str], core.RecordMeta],
    lexical: Mapping[str, Mapping[str, Any]],
    features: Mapping[
        str, Mapping[tuple[str, str], core.FeatureBlock]
    ],
    calibration_order: Sequence[str],
    split_salt: str,
    selection_query: str,
    selected_head: int,
    selected_center: int,
    radius: int,
) -> dict[str, Any]:
    source = records[(case_id, "ours")]
    target_gate = lexical[case_id]["target_gate"]
    center_index = selected_center - radius
    row = core.blank_case_row(
        static,
        "withheld_until_attention_ranks_frozen",
        source.target_label,
        "withheld",
        "withheld",
        False,
        False,
        selection_query,
        selected_head,
        selected_center,
        radius,
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
            "split": "evaluation",
            "split_hash_fraction": core.case_hash_fraction(case_id, split_salt),
            "grid_height": source.grid_height,
            "grid_width": source.grid_width,
            "tile_rows": source.tile_rows,
            "tile_columns": source.tile_columns,
            "has_thumbnail": source.has_thumbnail,
            "layout_signature": source.layout_signature,
            "npz_ours": str(records[(case_id, "ours")].npz_path),
            "npz_ttrv": str(records[(case_id, "ttrv")].npz_path),
            "ranking_outcome_blind": True,
        }
    )
    for query in core.QUERY_MODES:
        short = core.QUERY_SHORT[query]
        row[f"{short}_template_source_ours"] = features[query][
            (case_id, "ours")
        ].template_source
        row[f"{short}_template_source_ttrv"] = features[query][
            (case_id, "ttrv")
        ].template_source
    selected_maps_valid = all(
        bool(
            features[query][(case_id, model)].valid_mask[
                selected_head,
                center_index,
            ]
        )
        for query in core.QUERY_MODES
        for model in core.MODELS
    )
    row["grounding_attention_valid"] = selected_maps_valid
    if selected_maps_valid:
        core.populate_case_metrics(
            row,
            case_id,
            records,
            features,
            calibration_order,
            selected_head,
            center_index,
            selected_center,
            radius,
            selection_query,
        )
        row["eligible"] = True
        row["analysis_tier"] = "grounding_pool_ranked"
        row["quality_score"] = float(row["selection_ours_quality"])
        row["contrast_score"] = float(row["rank_score"])
        if not bool(row["quality_gate"]):
            row["exclusion_reason"] = (
                "below_predeclared_quality_quartile_gate"
            )
    else:
        # Invalid frozen maps remain visible and deterministically rank last.
        # Zero is not treated as a normalized/uniform attention map.
        row["eligible"] = False
        row["analysis_tier"] = "grounding_pool_invalid_selected_setting"
        row["quality_gate"] = False
        row["quality_score"] = 0.0
        row["contrast_score"] = 0.0
        row["rank_score"] = 0.0
        row["exclusion_reason"] = (
            "zero_mass_or_missing_template_at_frozen_setting"
        )
    row["quality_rank"] = 0
    row["contrast_rank"] = 0
    return row


def freeze_attention_only_ranks(
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    def ordering(score_name: str) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: (
                not bool(row["grounding_attention_valid"]),
                -float(row[score_name]),
                str(row["deterministic_tie_break"]),
            ),
        )

    quality_order = ordering("quality_score")
    contrast_order = ordering("contrast_score")
    for rank, row in enumerate(quality_order, start=1):
        row["quality_rank"] = rank
    for rank, row in enumerate(contrast_order, start=1):
        row["contrast_rank"] = rank
        row["rank"] = rank
        row["tier_rank"] = ""
        row["grounding_rank_type"] = "contrast"

    def digest(rank_name: str, score_name: str) -> str:
        ordered = sorted(rows, key=lambda row: int(row[rank_name]))
        payload = "\n".join(
            f"{int(row[rank_name])}\0{row['case_id']}\0"
            f"{float(row[score_name]):.17g}"
            for row in ordered
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return {
        "quality_rank_sha256_before_outcomes": digest(
            "quality_rank", "quality_score"
        ),
        "contrast_rank_sha256_before_outcomes": digest(
            "contrast_rank", "contrast_score"
        ),
    }


def attach_display_only_outcomes(
    rows: Sequence[dict[str, Any]],
    manifest_path: Path,
    manifest_static: Mapping[str, Mapping[str, Any]],
    records: Mapping[tuple[str, str], core.RecordMeta],
    max_words: int,
) -> None:
    outcomes = core.load_manifest_outcomes(manifest_path)
    for row in rows:
        case_id = str(row["case_id"])
        tier, gt, ours, ttrv, ours_correct, ttrv_correct = core.outcome_tier(
            manifest_static[case_id],
            outcomes.get(case_id, {}),
        )
        source = records[(case_id, "ours")]
        if gt != source.target_label:
            raise ValueError(
                f"{case_id}: manifest GT {gt} != saved target {source.target_label}"
            )
        options = core.parse_prompt_options(source.raw_prompt)
        wrong_text = options.get(ttrv, "") if ttrv in options and ttrv != gt else ""
        wrong_gate = (
            core.text_lexical_gate(wrong_text, max_words=max_words)
            if wrong_text
            else {
                "pass": False,
                "reason": "unknown_or_unavailable_wrong_option",
            }
        )
        row.update(
            {
                "ground_truth": gt,
                "ours_prediction": ours,
                "ttrv_prediction": ttrv,
                "ours_correct": ours_correct,
                "ttrv_correct": ttrv_correct,
                "outcome_tier": tier,
                "ttrv_wrong_option_text": wrong_text,
                "ttrv_wrong_option_lexical_gate": bool(wrong_gate["pass"]),
                "ttrv_wrong_option_lexical_reason": wrong_gate["reason"],
            }
        )


EXTRA_FIELDS = [
    "quality_rank",
    "contrast_rank",
    "quality_score",
    "contrast_score",
    "grounding_attention_valid",
    "ranking_outcome_blind",
    "grounding_rank_type",
    "render_rank",
    "render_selection_sources",
]


def grounding_fields() -> list[str]:
    leading = [
        "grounding_rank_type",
        "rank",
        "quality_rank",
        "contrast_rank",
        "case_id",
        "dataset",
        "question_id",
        "question",
        "image_path",
        "target_label",
        "target_option_text",
        "ground_truth",
        "ours_prediction",
        "ttrv_prediction",
        "outcome_tier",
        "quality_gate",
        "selection_query_mode",
        "selected_head_index_zero_based",
        "selected_layer_window_zero_based",
        "npz_ours",
        "npz_ttrv",
    ]
    result = list(leading)
    for field in core.case_fields() + EXTRA_FIELDS:
        if field not in result:
            result.append(field)
    return result


def make_render_subset(
    rows: Sequence[dict[str, Any]],
    top_per_rank: int,
    min_cases: int,
    max_cases: int,
) -> list[dict[str, Any]]:
    if not (1 <= top_per_rank <= max_cases):
        raise ValueError("--top-per-rank must lie in [1, max-render-cases]")
    if not (1 <= min_cases <= max_cases):
        raise ValueError("render case bounds are inconsistent")
    by_quality = sorted(rows, key=lambda row: int(row["quality_rank"]))
    by_contrast = sorted(rows, key=lambda row: int(row["contrast_rank"]))
    quality_seed = {str(row["case_id"]) for row in by_quality[:top_per_rank]}
    contrast_seed = {str(row["case_id"]) for row in by_contrast[:top_per_rank]}
    selected_ids = quality_seed | contrast_seed
    if len(selected_ids) < min_cases:
        for index in range(top_per_rank, len(rows)):
            for ordered in (by_quality, by_contrast):
                selected_ids.add(str(ordered[index]["case_id"]))
                if len(selected_ids) >= min_cases:
                    break
            if len(selected_ids) >= min_cases:
                break
    candidates = [row for row in rows if str(row["case_id"]) in selected_ids]
    candidates.sort(
        key=lambda row: (
            min(int(row["quality_rank"]), int(row["contrast_rank"])),
            int(row["quality_rank"]) + int(row["contrast_rank"]),
            str(row["deterministic_tie_break"]),
        )
    )
    candidates = candidates[:max_cases]
    subset: list[dict[str, Any]] = []
    for render_rank, source in enumerate(candidates, start=1):
        row = dict(source)
        case_id = str(row["case_id"])
        sources = []
        if case_id in quality_seed:
            sources.append("quality_top")
        if case_id in contrast_seed:
            sources.append("contrast_top")
        if not sources:
            sources.append("deterministic_fill")
        row["grounding_rank_type"] = (
            "both"
            if len(sources) == 2
            else "quality"
            if sources == ["quality_top"]
            else "contrast"
            if sources == ["contrast_top"]
            else "fill"
        )
        row["render_selection_sources"] = sources
        row["render_rank"] = render_rank
        row["rank"] = render_rank
        subset.append(row)
    if not min_cases <= len(subset) <= max_cases:
        raise AssertionError(
            f"render subset size {len(subset)} outside [{min_cases},{max_cases}]"
        )
    return subset


def top_summary(
    rows: Sequence[dict[str, Any]],
    rank_name: str,
    score_name: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row[rank_name]))[:limit]
    return [
        {
            "rank": int(row[rank_name]),
            "case_id": row["case_id"],
            "target_option_text": row["target_option_text"],
            "score": row[score_name],
            "quality_gate": row["quality_gate"],
            "outcome_tier_annotation": row["outcome_tier"],
            "ours_prediction_annotation": row["ours_prediction"],
            "ttrv_prediction_annotation": row["ttrv_prediction"],
        }
        for row in ordered
    ]


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.selection_config).expanduser().resolve()
    (
        config,
        records_path,
        manifest_path,
        selection_query,
        selected_head,
        selected_center,
        radius,
    ) = verify_frozen_config(config_path)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else config_path.parent
    )
    paths = prepare_paths(output_dir)
    correctness = require_mapping(
        config.get("correctness_isolation"), "correctness_isolation"
    )
    lexical_config = require_mapping(config.get("lexical_gate"), "lexical_gate")
    template_config = require_mapping(
        config.get("template_metrics"), "template_metrics"
    )
    calibration_ids = set(map(str, correctness["calibration_ids"]))
    evaluation_ids = set(map(str, correctness["evaluation_ids"]))
    calibration_order = sorted(calibration_ids)
    split_salt = str(correctness["split_salt"])
    dataset = str(config["dataset"])
    max_words = int(lexical_config["max_option_words"])
    max_tokens = int(lexical_config["max_content_tokens"])

    print("[grounding] verifying correctness-blind frozen inputs", flush=True)
    records, record_errors = core.load_trimmed_records(records_path)
    manifest_static, manifest_errors = core.load_manifest_static(manifest_path)
    if record_errors or manifest_errors:
        raise ValueError(
            f"input parse errors: records={record_errors[:3]} "
            f"manifest={manifest_errors[:3]}"
        )
    lexical, _, pair_errors = core.build_lexical_cases(
        records,
        manifest_static,
        dataset=dataset,
        max_words=max_words,
        max_tokens=max_tokens,
    )
    valid_ids, npz_failures = core.validate_lexical_npzs(lexical, records)
    if pair_errors or npz_failures:
        raise ValueError(
            f"pair/NPZ failures: pair={pair_errors} npz={npz_failures}"
        )
    if set(valid_ids) != calibration_ids | evaluation_ids:
        raise ValueError("valid lexical IDs differ from frozen split membership")
    if calibration_ids & evaluation_ids:
        raise ValueError("frozen calibration/evaluation splits overlap")
    recomputed_calibration = {
        case_id
        for case_id in valid_ids
        if core.case_hash_fraction(case_id, split_salt)
        < float(correctness["calibration_fraction"])
    }
    if recomputed_calibration != calibration_ids:
        raise ValueError("SHA256 split does not reproduce frozen calibration IDs")

    print(
        f"[grounding] frozen setting query={selection_query} head={selected_head} "
        f"layers={list(range(selected_center-radius, selected_center+radius+1))}; "
        f"calibration={len(calibration_ids)} evaluation={len(evaluation_ids)}",
        flush=True,
    )
    features, _, _ = core.extract_all_features(
        valid_ids,
        calibration_ids,
        records,
        radius=radius,
        template_size=int(template_config["size"]),
        min_template_group=int(template_config["minimum_layout_group"]),
    )
    rows = [
        populated_outcome_blind_row(
            case_id,
            manifest_static[case_id],
            records,
            lexical,
            features,
            calibration_order,
            split_salt,
            selection_query,
            selected_head,
            selected_center,
            radius,
        )
        for case_id in sorted(evaluation_ids)
    ]
    if len(rows) != len(evaluation_ids):
        raise AssertionError("grounding pool does not cover the full evaluation split")
    rank_hashes = freeze_attention_only_ranks(rows)
    print(
        "[grounding] attention-only quality/contrast ranks frozen; "
        "loading outcome annotations now",
        flush=True,
    )
    attach_display_only_outcomes(
        rows,
        manifest_path,
        manifest_static,
        records,
        max_words,
    )
    rows.sort(key=lambda row: int(row["contrast_rank"]))
    render_subset = make_render_subset(
        rows,
        top_per_rank=args.top_per_rank,
        min_cases=args.min_render_cases,
        max_cases=args.max_render_cases,
    )

    summary = {
        "status": "complete",
        "created_at": core.utc_now(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": core.sha256_file(Path(__file__).resolve()),
        "frozen_selection_config": str(config_path),
        "frozen_selection_config_sha256": core.sha256_file(config_path),
        "selection_recomputed": False,
        "selected_setting": config["selected_setting"],
        "selection_query_mode": selection_query,
        "selection_query_semantics": core.QUERY_SEMANTICS[selection_query],
        "causal_audit_query_mode": core.CAUSAL_QUERY,
        "pool_case_count": len(rows),
        "expected_evaluation_case_count": len(evaluation_ids),
        "quality_gate_pass_count": sum(bool(row["quality_gate"]) for row in rows),
        "invalid_selected_setting_map_count": sum(
            not bool(row["grounding_attention_valid"]) for row in rows
        ),
        "ranking_isolation": {
            "ranks_frozen_before_manifest_outcome_loader_called": True,
            "prediction_or_correctness_fields_used_in_ranking": False,
            "display_only_outcome_fields": [
                "ground_truth",
                "ours_prediction",
                "ttrv_prediction",
                "ours_correct",
                "ttrv_correct",
                "outcome_tier",
            ],
            **rank_hashes,
        },
        "rank_definitions": {
            "quality_rank": (
                "descending frozen-query Ours attention quality; SHA256 "
                "case tie-break"
            ),
            "quality_score": (
                "geometric mean of calibration ECDF percentiles for mass rank, "
                "boundary quality, template JSD, and layer stability"
            ),
            "contrast_rank": (
                "descending attention-only contrast score; SHA256 case tie-break"
            ),
            "contrast_score": (
                "0.50*Ours_quality + 0.30*max(0,Ours_quality-TTRV_quality) "
                "+ 0.20*Ours_TTRV_normalized_JSD"
            ),
            "quality_gate": "all Ours component percentiles >= 0.25",
        },
        "outcome_tier_annotation_counts": dict(
            sorted(Counter(str(row["outcome_tier"]) for row in rows).items())
        ),
        "top_quality_attention_only": top_summary(
            rows, "quality_rank", "quality_score"
        ),
        "top_contrast_attention_only": top_summary(
            rows, "contrast_rank", "contrast_score"
        ),
        "render_subset": {
            "case_count": len(render_subset),
            "top_per_rank_seed": args.top_per_rank,
            "minimum_cases": args.min_render_cases,
            "maximum_cases": args.max_render_cases,
            "deduplicated": True,
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    fields = grounding_fields()
    core.write_csv(paths["grounding_pool"], rows, fields)
    core.write_csv(paths["render_subset"], render_subset, fields)
    core.write_json(paths["summary"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=core.json_default))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-config", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--top-per-rank", type=int, default=16)
    parser.add_argument("--min-render-cases", type=int, default=20)
    parser.add_argument("--max-render-cases", type=int, default=24)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
