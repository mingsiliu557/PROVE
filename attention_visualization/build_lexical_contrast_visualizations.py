#!/usr/bin/env python3
"""Build publication-ready visualizations from ``lexical_contrast_ranked.csv``.

This is the stable public entrypoint for the lexical contrast renderer.  The
implementation lives in ``_lexical_contrast_renderer_impl.py`` so this module
can enforce the exact ``content_screen_design`` contract:

* render only ``quality_gate=True`` and ``eligible=True`` primary rows;
* keep literal ``unknown`` and ``absent`` predictions in separate tiers;
* require one screen-selected query/head/layer window across the top-N;
* use raw saved attention at T=1 with no temperature transformation; and
* use one Ours/TTRV color scale per case while preserving image geometry.

No setting is selected or repaired by this renderer.  Missing columns,
invalid primary-tier invariants, missing NPZ keys, and mixed settings fail with
explicit row-level errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import _lexical_contrast_renderer_impl as _impl


OUTCOME_TIERS = frozenset({"valid_wrong", "unknown_wrong", "not_contrast"})
ANALYSIS_TIERS = frozenset(
    {
        "primary_ranked",
        "secondary_unknown_ranked",
        "calibration_split_excluded",
        "nonlexical_wrong_option",
        "missing_attention",
        "schema_mismatch",
        "quality_flagged",
    }
)

# Columns guaranteed by content_screen_design and required in addition to the
# implementation's rendering fields.  Diagnostic score columns remain
# forward-compatible extras.
CONTRACT_COLUMNS = (
    "category",
    "content_token_count",
    "content_token_ids",
    "content_token_pieces",
)
_impl.REQUIRED_COLUMNS = tuple(dict.fromkeys((*_impl.REQUIRED_COLUMNS, *CONTRACT_COLUMNS)))


def _viridis(values: np.ndarray) -> np.ndarray:
    """Map 0..255 intensities to viridis without probability transforms."""

    import cv2

    array = np.asarray(values)
    if not np.all(np.isfinite(array)):
        raise ValueError("Color-map input contains NaN or infinity")
    numeric = np.asarray(array, dtype=np.float64)
    # The private implementation supplies uint8 0..255 values.  Accept 0..1
    # floats as a convenience for direct unit tests without double scaling.
    if numeric.size and float(numeric.max()) <= 1.0 and float(numeric.min()) >= 0.0:
        numeric = numeric * 255.0
    intensities = np.round(np.clip(numeric, 0.0, 255.0)).astype(np.uint8)
    colors = cv2.applyColorMap(intensities, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)


_impl._viridis = _viridis
_original_select_candidates = _impl._select_candidates


def _normalized_label(value: Any) -> str:
    return str(value or "").strip().upper()


def _validate_tier_columns(rows: Sequence[Mapping[str, str]]) -> None:
    for row_number, row in enumerate(rows, start=2):
        outcome = str(row.get("outcome_tier") or "").strip()
        analysis = str(row.get("analysis_tier") or "").strip()
        if outcome not in OUTCOME_TIERS:
            raise ValueError(
                f"CSV row {row_number}: unknown outcome_tier={outcome!r}; "
                f"expected one of {sorted(OUTCOME_TIERS)}"
            )
        if analysis not in ANALYSIS_TIERS:
            raise ValueError(
                f"CSV row {row_number}: unknown analysis_tier={analysis!r}; "
                f"expected one of {sorted(ANALYSIS_TIERS)}"
            )


def _validate_primary_candidate(candidate: Any) -> None:
    row = candidate.source_row
    row_number = candidate.csv_row_number
    outcome = str(row["outcome_tier"] or "").strip()
    analysis = str(row["analysis_tier"] or "").strip()
    if outcome != "valid_wrong" or analysis != "primary_ranked":
        raise ValueError(
            f"CSV row {row_number}: quality_gate passes but tier is "
            f"outcome_tier={outcome!r}, analysis_tier={analysis!r}; only "
            "valid_wrong/primary_ranked may enter the primary renderer"
        )
    ours_status = _impl._prediction_status(row["ours_prediction"])
    ttrv_status = _impl._prediction_status(row["ttrv_prediction"])
    if ours_status != "valid" or ttrv_status != "valid":
        raise ValueError(
            f"CSV row {row_number}: passing primary rows require two valid predictions; "
            f"Ours={row['ours_prediction']!r} ({ours_status}), "
            f"TTRV={row['ttrv_prediction']!r} ({ttrv_status})"
        )
    ground_truth = _normalized_label(row["ground_truth"])
    target = _normalized_label(row["target_label"])
    ours = _normalized_label(row["ours_prediction"])
    ttrv = _normalized_label(row["ttrv_prediction"])
    if not all(re.fullmatch(r"[A-F]", label) for label in (ground_truth, target, ours, ttrv)):
        raise ValueError(
            f"CSV row {row_number}: primary labels must be A..F; GT={ground_truth!r}, "
            f"target={target!r}, Ours={ours!r}, TTRV={ttrv!r}"
        )
    if target != ground_truth or ours != ground_truth or ttrv == ground_truth:
        raise ValueError(
            f"CSV row {row_number}: primary invariant requires "
            "target_label == ours_prediction == ground_truth != ttrv_prediction; "
            f"got target={target}, Ours={ours}, GT={ground_truth}, TTRV={ttrv}"
        )
    if not _impl._compact_text(row["target_option_text"]):
        raise ValueError(f"CSV row {row_number}: passing target_option_text is empty")
    _impl._parse_integer(row["content_token_count"], "content_token_count", row_number, minimum=1)


def _select_candidates(
    rows: Sequence[dict[str, str]],
    csv_path: Path,
    top_n: int,
    expected_query: str,
):
    _validate_tier_columns(rows)
    selected, gate_states = _original_select_candidates(rows, csv_path, top_n, expected_query)
    for candidate in selected:
        _validate_primary_candidate(candidate)
    return selected, gate_states


_impl._select_candidates = _select_candidates


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _augment_index_and_summary(
    ranked_csv: Path,
    output_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    source_rows = _read_csv_rows(ranked_csv)
    source_by_case = {str(row["case_id"]): row for row in source_rows}
    index_path = output_dir / "index.csv"
    index_rows = _read_csv_rows(index_path)
    augmented: list[dict[str, Any]] = []
    for row in index_rows:
        source = source_by_case[str(row["case_id"])]
        item: dict[str, Any] = dict(row)
        item.update(
            {
                "category": source["category"],
                "content_token_count": _impl._parse_integer(
                    source["content_token_count"],
                    "content_token_count",
                    0,
                    minimum=1,
                ),
                "content_token_ids": source["content_token_ids"],
                "content_token_pieces": source["content_token_pieces"],
            }
        )
        augmented.append(item)
    _impl._write_csv(index_path, augmented)
    summary["source_outcome_tier_counts"] = dict(
        sorted(Counter(row["outcome_tier"] for row in source_rows).items())
    )
    summary["source_analysis_tier_counts"] = dict(
        sorted(Counter(row["analysis_tier"] for row in source_rows).items())
    )
    summary["tier_contract"] = {
        "primary": "quality_gate=true, eligible=true, outcome_tier=valid_wrong, analysis_tier=primary_ranked",
        "unknown_prediction": "literal unknown; never rendered in the primary pass",
        "absent_prediction": "literal absent; never rendered in the primary pass",
    }
    summary["outputs"]["index_csv"] = str(index_path.resolve())
    _impl._write_json(output_dir / "summary.json", summary)
    return summary


def build_visualizations(
    ranked_csv: str | Path,
    output_dir: str | Path,
    top_n: int = 12,
    expected_query: str = _impl.DEFAULT_QUERY,
    alpha: float = 0.55,
    panel_max_width: int = 620,
    panel_max_height: int = 620,
    contact_columns: int = 2,
) -> dict[str, Any]:
    csv_path = Path(ranked_csv).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    summary = _impl.build_visualizations(
        ranked_csv=csv_path,
        output_dir=destination,
        top_n=top_n,
        expected_query=expected_query,
        alpha=alpha,
        panel_max_width=panel_max_width,
        panel_max_height=panel_max_height,
        contact_columns=contact_columns,
    )
    return _augment_index_and_summary(csv_path, destination, summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument(
        "--expected-query",
        choices=_impl.ALLOWED_QUERIES,
        default=_impl.DEFAULT_QUERY,
        help="Validate the screen-selected query; never substitutes another setting.",
    )
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--panel-max-width", type=int, default=620)
    parser.add_argument("--panel-max-height", type=int, default=620)
    parser.add_argument("--contact-columns", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build_visualizations(
        ranked_csv=args.ranked_csv,
        output_dir=args.output_dir,
        top_n=args.top_n,
        expected_query=args.expected_query,
        alpha=args.alpha,
        panel_max_width=args.panel_max_width,
        panel_max_height=args.panel_max_height,
        contact_columns=args.contact_columns,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
