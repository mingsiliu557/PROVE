import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_lexical_contrast_visualizations as renderer


class LexicalContrastRendererTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        pixels = np.zeros((40, 80, 3), dtype=np.uint8)
        pixels[..., 0] = np.arange(80, dtype=np.uint8)[None, :]
        pixels[..., 1] = np.arange(40, dtype=np.uint8)[:, None] * 4
        pixels[..., 2] = 90
        self.image = self.root / "source.png"
        Image.fromarray(pixels).save(self.image)
        self.layers = (1, 2, 3)
        self.head = 2
        self.query = "content_answer_state_heads"
        ours = {}
        ttrv = {}
        template = np.arange(24, dtype=np.float32).reshape(4, 6) / 1000.0
        for layer in self.layers:
            key = f"{self.query}__layer_{layer:02d}_head_{self.head:02d}"
            ours[key] = template + layer * 0.002
            ttrv[key] = np.flip(template, axis=1) + layer * 0.001
        self.expected_ours = np.mean(list(ours.values()), axis=0)
        self.expected_ttrv = np.mean(list(ttrv.values()), axis=0)
        self.ours_npz = self.root / "ours.npz"
        self.ttrv_npz = self.root / "ttrv.npz"
        np.savez_compressed(self.ours_npz, **ours)
        np.savez_compressed(self.ttrv_npz, **ttrv)

    def tearDown(self):
        self.temporary.cleanup()

    def _row(self, case_id="CASE_PASS", rank=2, quality="True"):
        row = {field: "" for field in renderer._impl.REQUIRED_COLUMNS}
        row.update(
            {
                "rank": str(rank),
                "tier_rank": "1",
                "case_id": case_id,
                "dataset": "AI2D",
                "question_id": f"qid-{case_id}",
                "category": "synthetic",
                "question": "Which animal is at the top of the food chain?",
                "image_path": str(self.image),
                "target_label": "C",
                "target_option_text": "Seal",
                "content_token_count": "1",
                "content_token_ids": "[52525]",
                "content_token_pieces": '[" Seal"]',
                "ground_truth": "C",
                "ours_prediction": "C",
                "ttrv_prediction": "B",
                "outcome_tier": "valid_wrong",
                "analysis_tier": "primary_ranked",
                "eligible": "True",
                "selection_query_mode": self.query,
                "selected_head_index_zero_based": str(self.head),
                "selected_layer_center_zero_based": "2",
                "selected_layer_window_zero_based": json.dumps(self.layers),
                "rank_score": "0.75",
                "quality_gate": quality,
                "exclusion_reason": "",
                "npz_ours": str(self.ours_npz),
                "npz_ttrv": str(self.ttrv_npz),
            }
        )
        return row

    def _write_csv(self, path, rows, fields=None):
        fields = list(fields or renderer._impl.REQUIRED_COLUMNS)
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])

    def test_full_synthetic_render_uses_raw_shared_scale(self):
        failed = self._row("CASE_UNKNOWN", rank=1, quality="False")
        failed.update(
            {
                "ours_prediction": "A",
                "ttrv_prediction": "unknown",
                "outcome_tier": "unknown_wrong",
                "analysis_tier": "secondary_unknown_ranked",
                "eligible": "False",
                "npz_ours": "",
                "npz_ttrv": "",
            }
        )
        passed = self._row()
        ranked = self.root / "lexical_contrast_ranked.csv"
        self._write_csv(ranked, [failed, passed])
        destination = self.root / "rendered"
        summary = renderer.build_visualizations(
            ranked,
            destination,
            top_n=1,
            panel_max_width=180,
            panel_max_height=140,
            contact_columns=1,
        )
        self.assertEqual(summary["rendered_cases"], 1)
        self.assertEqual(summary["quality_gate_state_counts"], {"fail": 1, "pass": 1})
        self.assertEqual(summary["source_outcome_tier_counts"], {"unknown_wrong": 1, "valid_wrong": 1})
        self.assertFalse(summary["display_policy"]["temperature_transform_applied"])
        self.assertEqual(summary["display_policy"]["temperature"], 1.0)
        self.assertEqual(summary["fixed_setting"]["selected_head_index_zero_based"], self.head)
        self.assertEqual(
            summary["fixed_setting"]["selected_layer_window_zero_based"],
            list(self.layers),
        )

        with (destination / "index.csv").open("r", encoding="utf-8", newline="") as handle:
            index = list(csv.DictReader(handle))
        self.assertEqual(len(index), 1)
        row = index[0]
        expected_vmax = max(float(self.expected_ours.max()), float(self.expected_ttrv.max()))
        self.assertAlmostEqual(float(row["shared_vmax"]), expected_vmax, places=8)
        self.assertAlmostEqual(float(row["ours_raw_attention_mass"]), float(self.expected_ours.sum()), places=8)
        self.assertAlmostEqual(float(row["ttrv_raw_attention_mass"]), float(self.expected_ttrv.sum()), places=8)
        self.assertEqual(row["content_token_count"], "1")
        self.assertEqual(row["category"], "synthetic")
        self.assertEqual(row["aspect_ratio_preserved"], "True")

        copied = Path(row["copied_original"])
        ours_overlay = Path(row["ours_overlay_png"])
        ttrv_overlay = Path(row["ttrv_overlay_png"])
        side_by_side = Path(row["side_by_side_png"])
        self.assertEqual(copied.read_bytes(), self.image.read_bytes())
        self.assertTrue(ours_overlay.is_file())
        self.assertTrue(ttrv_overlay.is_file())
        self.assertTrue(side_by_side.is_file())
        self.assertTrue((destination / "contact_sheet_raw_T1.png").is_file())
        with Image.open(ours_overlay) as handle:
            self.assertEqual(handle.size, (80, 40))
        with Image.open(ttrv_overlay) as handle:
            self.assertEqual(handle.size, (80, 40))

        colors = renderer._viridis(np.array([[0, 64, 128, 255]], dtype=np.uint8))
        self.assertEqual(len({tuple(value) for value in colors.reshape(-1, 3)}), 4)

    def test_missing_required_column_has_explicit_error(self):
        ranked = self.root / "missing.csv"
        fields = [field for field in renderer._impl.REQUIRED_COLUMNS if field != "npz_ttrv"]
        self._write_csv(ranked, [self._row()], fields=fields)
        with self.assertRaisesRegex(ValueError, "missing required columns.*npz_ttrv"):
            renderer.build_visualizations(ranked, self.root / "missing_output", top_n=1)

    def test_mixed_fixed_settings_are_rejected_before_render(self):
        first = self._row("CASE_ONE", rank=1)
        second = self._row("CASE_TWO", rank=2)
        second["selected_head_index_zero_based"] = "3"
        ranked = self.root / "mixed.csv"
        self._write_csv(ranked, [first, second])
        with self.assertRaisesRegex(ValueError, "do not share one fixed query/head/layer window"):
            renderer.build_visualizations(ranked, self.root / "mixed_output", top_n=2)
        self.assertFalse((self.root / "mixed_output").exists())

    def test_unknown_prediction_cannot_enter_primary_pass(self):
        row = self._row()
        row["ttrv_prediction"] = "unknown"
        ranked = self.root / "unknown_pass.csv"
        self._write_csv(ranked, [row])
        with self.assertRaisesRegex(ValueError, "require two valid predictions"):
            renderer.build_visualizations(ranked, self.root / "unknown_output", top_n=1)


if __name__ == "__main__":
    unittest.main()
