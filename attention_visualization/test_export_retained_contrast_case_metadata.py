import csv
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from export_retained_contrast_case_metadata import export_case_metadata


class ExportRetainedContrastCaseMetadataTest(unittest.TestCase):
    def test_end_to_end_parquet_choices_rank_order_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parquet_path = root / "test.parquet"
            prompts = (
                (
                    "<image>\nWhich vehicle is nearest?\n"
                    "Choose the correct answer from the options below and respond with only the letter.\n"
                    "Options:\nA. Red car\nB. Blue bus"
                ),
                (
                    "<image>\nWhere is the bird flying?\n"
                    "A. Left\nB. Right\nC. Up\n"
                    "Please answer directly with only the letter of the correct option."
                ),
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "question_id": "q1",
                            "prompt": [{"role": "user", "content": prompts[0]}],
                        },
                        {
                            "question_id": "q2",
                            "prompt": [{"role": "user", "content": prompts[1]}],
                        },
                    ]
                ),
                parquet_path,
            )

            source_image = root / "source.png"
            source_image.write_bytes(b"synthetic source")
            artifact_names = (
                "one_original.png",
                "one_ours.png",
                "one_ttrv.png",
                "one_side.png",
                "two_original.png",
                "two_ours.png",
                "two_ttrv.png",
                "two_side.png",
            )
            for name in artifact_names:
                (root / name).write_bytes(name.encode("utf-8"))

            def manifest_case(
                case_id: str,
                question_id: str,
                row_index: int,
                prompt: str,
                ground_truth: str,
                ttrv: str,
                selection_rank: int,
            ) -> dict:
                options = {
                    "q1": {"A": "Red car", "B": "Blue bus"},
                    "q2": {"A": "Left", "B": "Right", "C": "Up"},
                }[question_id]
                return {
                    "case_id": case_id,
                    "dataset": "SYNTH",
                    "question_id": question_id,
                    "row_index": row_index,
                    "test_parquet": str(parquet_path),
                    "train_parquet": None,
                    "image_path": str(source_image),
                    "question": prompt.replace("<image>", "").strip(),
                    "ground_truth": ground_truth,
                    "parser": "synthetic",
                    "labels": "A-F",
                    "task": "synthetic",
                    "retained_predictions": {
                        "ours": {
                            "raw_response": ground_truth,
                            "parsed_answer": ground_truth,
                            "source_file": "/predictions/ours.jsonl",
                            "source_step": 10,
                        },
                        "ttrv": {
                            "raw_response": ttrv,
                            "parsed_answer": ttrv,
                            "source_file": "/predictions/ttrv.jsonl",
                            "source_step": 10,
                        },
                    },
                    "models": {
                        "ours": {
                            "checkpoint": "/checkpoints/ours",
                            "source_checkpoint": "/training/ours",
                        },
                        "ttrv": {
                            "checkpoint": "/checkpoints/ttrv",
                            "source_checkpoint": "/training/ttrv",
                        },
                    },
                    "ours_target_label": ground_truth,
                    "ours_target_option_text": options[ground_truth],
                    "ttrv_target_label": ttrv,
                    "ttrv_target_option_text": options[ttrv],
                    "retained_target_source": "per_model_retained_prediction",
                    "ours_correct": True,
                    "ttrv_correct": False,
                    "selection_policy": "synthetic_strict",
                    "selection_gate": "ours_correct_ttrv_valid_wrong",
                    "selection_outcome_dependent": True,
                    "selection_correctness_blind": False,
                    "selection_uses_model_predictions": True,
                    "selection_dataset_max_option_words": 16,
                    "selection_rank": selection_rank,
                    "selection_dataset_rank": selection_rank,
                }

            manifest_rows = [
                manifest_case("case_one", "q1", 0, prompts[0], "A", "B", 1),
                manifest_case("case_two", "q2", 1, prompts[1], "C", "A", 2),
            ]
            manifest_path = root / "strict.jsonl"
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in manifest_rows),
                encoding="utf-8",
            )

            def renderer_row(
                case: dict, dataset_rank: int, prefix: str
            ) -> dict[str, object]:
                ours = case["ours_target_label"]
                ttrv = case["ttrv_target_label"]
                return {
                    "dataset_rank": dataset_rank,
                    "case_id": case["case_id"],
                    "dataset": case["dataset"],
                    "question_id": case["question_id"],
                    "ground_truth": case["ground_truth"],
                    "ours_target_source": "retained",
                    "ours_target_label": ours,
                    "ours_target_option_text": case["ours_target_option_text"],
                    "ours_target_correct": True,
                    "ttrv_target_source": "retained",
                    "ttrv_target_label": ttrv,
                    "ttrv_target_option_text": case["ttrv_target_option_text"],
                    "ttrv_target_correct": False,
                    "targets_differ": True,
                    "comparison_type": "different_per_model_retained_answer_targets",
                    "query_mode": "content_answer_state_heads",
                    "head_index_zero_based": 2,
                    "layer_center_zero_based": 15,
                    "layer_window_zero_based": json.dumps([14, 15, 16]),
                    "rank_metrics_temperature": 1.0,
                    "raw_rank_score": 0.5,
                    "display_temperature": 0.5,
                    "display_transform": "synthetic",
                    "answer_state_interpretation": "non-causal",
                    "source_npz_ours": f"{prefix}_ours.npz",
                    "source_npz_ttrv": f"{prefix}_ttrv.npz",
                    "source_image_path": str(source_image),
                    "copied_original": str(root / f"{prefix}_original.png"),
                    "ours_overlay_png": str(root / f"{prefix}_ours.png"),
                    "ttrv_overlay_png": str(root / f"{prefix}_ttrv.png"),
                    "side_by_side_png": str(root / f"{prefix}_side.png"),
                }

            # Deliberately put rank 2 before rank 1 in the renderer CSV.
            renderer_rows = [
                renderer_row(manifest_rows[0], 2, "one"),
                renderer_row(manifest_rows[1], 1, "two"),
            ]
            renderer_path = root / "index.csv"
            with renderer_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(renderer_rows[0]))
                writer.writeheader()
                writer.writerows(renderer_rows)

            output_dir = root / "metadata"
            summary = export_case_metadata(manifest_path, renderer_path, output_dir)
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["rendered_by_dataset"], {"SYNTH": 2})
            exported = [
                json.loads(line)
                for line in (output_dir / "case_metadata.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual([row["case_id"] for row in exported], ["case_two", "case_one"])
            self.assertEqual([row["rank"] for row in exported], [1, 2])
            self.assertEqual(exported[0]["question"], "Where is the bird flying?")
            self.assertEqual(exported[0]["choice_C"], "Up")
            self.assertEqual(exported[0]["ground_truth_text"], "Up")
            self.assertEqual(exported[1]["question"], "Which vehicle is nearest?")
            self.assertEqual(exported[1]["choice_B"], "Blue bus")
            self.assertIn("Options:", exported[1]["full_prompt_with_choices"])

            with self.assertRaises(FileExistsError):
                export_case_metadata(manifest_path, renderer_path, output_dir)


if __name__ == "__main__":
    unittest.main()
