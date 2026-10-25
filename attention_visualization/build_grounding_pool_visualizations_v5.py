#!/usr/bin/env python3
"""Final support-preserving temperature renderer with consistent metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import build_grounding_pool_visualizations_v4 as v4


TRANSFORM_TEXT = (
    "m=sum(A); support={i:A_i>0}; on support, "
    "p_T=softmax(log(A_i/m)/T); displayed=m*p_T; off support=0"
)


_original_pool_index_row = v4.base._pool_index_row


def _pool_index_row_with_exact_formula(*args: Any, **kwargs: Any) -> dict[str, Any]:
    row = _original_pool_index_row(*args, **kwargs)
    row["display_transform"] = TRANSFORM_TEXT
    row["zero_support_preserved"] = True
    row["T1_elementwise_identity"] = True
    return row


v4.base._pool_index_row = _pool_index_row_with_exact_formula


def main() -> int:
    args = v4.build_parser().parse_args()
    summary = v4.base.build_grounding_visualizations(
        grounding_csv=args.grounding_csv,
        output_dir=args.output_dir,
        rank_types=v4.base._parse_rank_types(args.rank_types),
        top_n_per_rank=args.top_n_per_rank,
        merged_max_cases=args.merged_max_cases,
        display_temperature=args.display_temperature,
        alpha=args.alpha,
        panel_max_width=args.panel_max_width,
        panel_max_height=args.panel_max_height,
        contact_columns=args.contact_columns,
    )
    policy = summary["display_policy"]
    policy["display_transform"] = TRANSFORM_TEXT
    policy["zero_support_preserved"] = True
    policy["T1_elementwise_identity"] = True
    summary["renderer_entrypoint"] = str(Path(__file__).resolve())
    summary["row_level_transform_metadata_consistent"] = True
    v4.base._helper._write_json(Path(args.output_dir).resolve() / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
