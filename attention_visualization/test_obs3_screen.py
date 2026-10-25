#!/usr/bin/env python3
"""Small CPU-only synthetic test for obs3_screen.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from attention_visualization import obs3_screen
except ModuleNotFoundError:  # Direct script execution sets sys.path to this directory.
    import obs3_screen


class Obs3ScreenSyntheticTest(unittest.TestCase):
    def _write_case(
        self,
        root: Path,
        case_id: str,
        image_path: Path,
        *,
        ours_correct: bool = True,
        ttrv_answer: str = "B",
        ours_retained_match: bool = True,
    ) -> list[dict]:
        records = []
        for model in obs3_screen.REQUIRED_MODELS:
            arrays = {}
            for layer in obs3_screen.EXPECTED_LAYERS:
                grid = np.full((4, 4), 1.0, dtype=np.float32)
                if layer in {9, 10, 11}:
                    grid.fill(0.01)
                    if model == "ours":
                        grid[0, 0] = 10.0
                    elif model == "ttrv":
                        grid[3, 3] = 10.0
                scale = {"base": 0.15, "ttrv": 0.10, "ours": 0.20}[model]
                arrays[f"first_decision_query__layer_{layer:02d}"] = grid * scale
            npz_path = root / f"{case_id}_{model}.npz"
            np.savez_compressed(npz_path, **arrays)
            if model == "ours":
                parsed = "A" if ours_correct else "B"
                correct = ours_correct
                retained_match = ours_retained_match
            elif model == "ttrv":
                parsed = ttrv_answer
                correct = ttrv_answer == "A"
                retained_match = True
            else:
                parsed = "A"
                correct = True
                retained_match = True
            records.append(
                {
                    "case_id": case_id,
                    "dataset": "synthetic",
                    "question_id": case_id,
                    "model": model,
                    "checkpoint": f"/checkpoints/{model}",
                    "source_checkpoint": f"/source/{model}",
                    "raw_response": parsed,
                    "parsed_answer": parsed,
                    "ground_truth": "A",
                    "correct": correct,
                    "retained_match": retained_match,
                    "individual_layers": list(obs3_screen.EXPECTED_LAYERS),
                    "npz_path": str(npz_path),
                    "image_path": str(image_path),
                    "question": "Which marked region is correct?",
                    "grid": {"original_width": 64, "original_height": 48},
                    "validation_provenance": {"split": "synthetic-validation"},
                }
            )
        return records

    def test_strict_gate_shared_window_and_display_only_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="obs3_screen_test_") as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            Image.new("RGB", (64, 48), (180, 190, 200)).save(image_path)
            records = []
            records.extend(self._write_case(root, "eligible", image_path))
            records.extend(self._write_case(root, "invalid_ttrv", image_path, ttrv_answer="unknown"))
            records.extend(
                self._write_case(root, "retained_mismatch", image_path, ours_retained_match=False)
            )
            records_path = root / "attention_records.jsonl"
            with records_path.open("w", encoding="utf-8") as handle:
                for row in records:
                    handle.write(json.dumps(row) + "\n")

            roi_path = root / "rois.json"
            roi_path.write_text(
                json.dumps(
                    {
                        "cases": {
                            "eligible": {
                                "rois": [
                                    {
                                        "label": "synthetic target",
                                        "bbox": [0.0, 0.0, 0.5, 0.5],
                                        "coordinate_space": "normalized",
                                    }
                                ]
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            output_dir = root / "screen"
            summary = obs3_screen.screen_records(
                records_path, output_dir, roi_json=roi_path, max_figures=1
            )

            self.assertEqual(summary["strict_complete_case_count"], 1)
            self.assertEqual(summary["qualified_case_count"], 1)
            self.assertEqual(summary["selection_policy"], "shared_three_layer_window")
            self.assertEqual(summary["shared_window"]["layers"], [9, 10, 11])
            self.assertFalse(summary["display_policy"]["temperature_used_for_selection_or_ranking"])
            ranked = obs3_screen.read_jsonl(output_dir / "ranked_cases.jsonl")
            self.assertEqual(len(ranked), 1)
            self.assertEqual(ranked[0]["case_id"], "eligible")
            self.assertEqual(ranked[0]["tier"], "A")
            self.assertEqual(len(ranked[0]["layer_metrics"]), 28)
            self.assertEqual(ranked[0]["rank_metric"], "raw_T1_selected_window_stability_score")
            roi = ranked[0]["selected_window_roi_metrics"][0]
            self.assertGreater(roi["ours_minus_ttrv_conditional_mass_delta"]["median"], 0)
            self.assertEqual(roi["ours_peak_in_roi_layer_count"], 3)
            self.assertTrue(
                ranked[0]["roi_manual_review_signal"][
                    "all_explicit_rois_have_positive_median_delta"
                ]
            )
            self.assertFalse(
                ranked[0]["roi_manual_review_signal"]["affects_default_gate_or_rank"]
            )
            figures = ranked[0]["figures"]
            self.assertTrue(Path(figures["T1_raw"]["png"]).is_file())
            self.assertTrue(Path(figures["T0p5_display_only"]["pdf"]).is_file())
            self.assertTrue(figures["T0p5_display_only"]["display_only"])
            self.assertTrue(Path(figures["comparison"]["png"]).is_file())


if __name__ == "__main__":
    unittest.main()
