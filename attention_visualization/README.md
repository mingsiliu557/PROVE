# InternVL3-2B Attention Visualization

This directory is isolated from the TTRV training path. It converts retained
verl FSDP checkpoints, selects matched cases, extracts InternVL3 decoder
self-attention from answer queries to image-token keys, and renders a shared-scale
Base/TTRV/Ours comparison.

## Scientific convention

- `answer_state`: rows at the generated answer-token positions. This is the main
  view requested for the figure.
- `decision_query`: causally shifted rows that produced the answer tokens. This
  is retained as a consistency diagnostic.
- Layers are tested in three fixed zero-based windows: `6-8`, `13-15`, `20-22`.
- Display temperatures are global: `1.0`, `0.7`, `0.5`, `0.3`. Temperature is
  post-processing only and never changes model generation. It preserves each
  map's original total image-attention mass and only changes spatial sharpness.
- Every final panel shares one layer window, one temperature, and one global
  min-max color scale.
- Attention is supporting descriptive evidence, not a standalone causal
  attribution method.

## Commands

```bash
bash attention_visualization/run.sh static

CHECKPOINT=/path/to/global_step_20 \
MERGED_NAME=ttrv_mme20_step20 \
EXP_ROOT=/path/to/datasets/experiments/internvl3_attention_viz_RUN \
bash attention_visualization/run.sh merge

SELECTION_CONFIG=/path/to/selection_config.json \
EXP_ROOT=/path/to/datasets/experiments/internvl3_attention_viz_RUN \
bash attention_visualization/run.sh select

CANDIDATES=/path/to/candidates.jsonl \
EXP_ROOT=/path/to/datasets/experiments/internvl3_attention_viz_RUN \
bash attention_visualization/run.sh smoke
```

The selection config records dataset parquet files, retained greedy output files,
source FSDP checkpoints, and their merged Hugging Face destinations. Runtime
artifacts stay under `/path/to/datasets`; this directory contains code only.

The extractor reconstructs InternVL's dynamic row-major tile grid with the exact
`get_optimal_tiled_canvas` helper used by the configured processor. When a global
thumbnail is appended, its attention mass is projected back to the full grid and
added to the local-tile map while preserving the total thumbnail attention mass.
