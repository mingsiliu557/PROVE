#!/usr/bin/env python3
"""Render grounding galleries with exact support-preserving display temperature.

This is the public temperature-rendering entry point.  It reuses the validated
v3 gallery pipeline, but replaces its display transform so that zero-valued
cells remain exactly zero and ``T=1`` is an elementwise identity.  Ranking,
metrics, case selection, and the frozen answer-state head/layer setting remain
the raw-T=1 values from the grounding CSV.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import build_grounding_pool_visualizations_v3 as base


def mass_preserving_support_temperature(
    raw: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Sharpen positive spatial mass while preserving total mass and support."""

    array = np.asarray(raw, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Display-temperature input must be non-empty and finite")
    if np.any(array < 0.0):
        raise ValueError("Display-temperature input must be non-negative")
    if not math.isfinite(temperature) or not 0.0 < temperature <= 1.0:
        raise ValueError(f"display temperature must satisfy 0<T<=1, got {temperature}")
    mass = float(array.sum(dtype=np.float64))
    if not math.isfinite(mass):
        raise ValueError(f"Display-temperature mass is non-finite: {mass}")
    if mass <= 0.0:
        return np.zeros_like(array, dtype=np.float64)
    if temperature == 1.0:
        return array.copy()

    positive = array > 0.0
    positive_values = array[positive]
    logits = np.log(positive_values / mass) / temperature
    logits -= float(logits.max())
    weights = np.exp(logits)
    weights /= float(weights.sum(dtype=np.float64))
    displayed = np.zeros_like(array, dtype=np.float64)
    displayed[positive] = weights * mass

    positive_flat = np.flatnonzero(positive.reshape(-1))
    flat = displayed.reshape(-1)
    correction_index = int(positive_flat[np.argmax(flat[positive_flat])])
    for _ in range(3):
        residual = mass - float(displayed.sum(dtype=np.float64))
        if residual == 0.0:
            break
        flat[correction_index] += residual
    tolerance = max(float(np.spacing(abs(mass))) * 8.0, 1e-15)
    if not math.isclose(
        float(displayed.sum(dtype=np.float64)),
        mass,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise AssertionError("Display-temperature transform failed to preserve raw mass")
    if np.any(displayed[~positive] != 0.0):
        raise AssertionError("Display-temperature transform changed zero support")
    return displayed


base._mass_preserving_spatial_temperature = mass_preserving_support_temperature


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounding-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank-types", default="quality,contrast")
    parser.add_argument("--top-n-per-rank", type=int, default=16)
    parser.add_argument("--merged-max-cases", type=int, default=24)
    parser.add_argument("--display-temperature", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--panel-max-width", type=int, default=620)
    parser.add_argument("--panel-max-height", type=int, default=620)
    parser.add_argument("--contact-columns", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = base.build_grounding_visualizations(
        grounding_csv=args.grounding_csv,
        output_dir=args.output_dir,
        rank_types=base._parse_rank_types(args.rank_types),
        top_n_per_rank=args.top_n_per_rank,
        merged_max_cases=args.merged_max_cases,
        display_temperature=args.display_temperature,
        alpha=args.alpha,
        panel_max_width=args.panel_max_width,
        panel_max_height=args.panel_max_height,
        contact_columns=args.contact_columns,
    )
    policy = summary["display_policy"]
    policy["display_transform"] = (
        "m=sum(A); support={i:A_i>0}; on support, "
        "p_T=softmax(log(A_i/m)/T); displayed=m*p_T; off support=0"
    )
    policy["zero_support_preserved"] = True
    policy["T1_elementwise_identity"] = True
    summary["renderer_entrypoint"] = str(Path(__file__).resolve())
    base._helper._write_json(Path(args.output_dir).resolve() / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
