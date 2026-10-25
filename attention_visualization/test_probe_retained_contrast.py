import copy
from pathlib import Path
import unittest

from probe_retained_contrast import select_exact_candidates, validate_manifest


def make_case(case_id, dataset, ground_truth="B"):
    return {
        "case_id": case_id,
        "dataset": dataset,
        "question_id": f"question-{case_id}",
        "ground_truth": ground_truth,
        "retained_predictions": {
            "base": {"parsed_answer": "A", "raw_response": "old-base"},
            "ours": {"parsed_answer": "A", "raw_response": "old-ours"},
            "ttrv": {"parsed_answer": "B", "raw_response": "old-ttrv"},
        },
        "models": {
            "ours": {"checkpoint": "/models/ours"},
            "ttrv": {"checkpoint": "/models/ttrv"},
        },
    }


def make_records(case_id, ours="B", ttrv="A"):
    return [
        {
            "case_id": case_id,
            "model": "ours",
            "raw_response": ours,
            "parsed_answer": ours,
            "checkpoint": "/models/ours",
            "source_checkpoint": "/training/ours",
            "generated_token_ids": [1],
            "generated_tokens": [ours],
        },
        {
            "case_id": case_id,
            "model": "ttrv",
            "raw_response": ttrv,
            "parsed_answer": ttrv,
            "checkpoint": "/models/ttrv",
            "source_checkpoint": "/training/ttrv",
            "generated_token_ids": [2],
            "generated_tokens": [ttrv],
        },
    ]


def select(cases, records, options, max_per_dataset=0):
    return select_exact_candidates(
        cases,
        records,
        options,
        max_per_dataset=max_per_dataset,
        selection_salt="unit-test-salt",
        probe_records_path=Path("/tmp/probe_records.jsonl"),
        input_manifest=Path("/tmp/input.jsonl"),
        input_manifest_sha256="abc123",
        max_new_tokens=8,
    )


class ProbeRetainedContrastTest(unittest.TestCase):
    def test_strict_selection_updates_current_and_preserves_history(self):
        case = make_case("rwq-1", "RealWorldQA")
        old_retained = copy.deepcopy(case["retained_predictions"])
        options = {
            "rwq-1": {
                "A": "The item is located on the left side",
                "B": "The item is located on the right side of the road",
            }
        }
        selected, summary = select([case], make_records("rwq-1"), options)
        self.assertEqual(len(selected), 1)
        result = selected[0]
        self.assertEqual(result["historical_retained_predictions"], old_retained)
        self.assertEqual(result["retained_predictions"]["base"], old_retained["base"])
        self.assertEqual(result["retained_predictions"]["ours"]["parsed_answer"], "B")
        self.assertEqual(result["retained_predictions"]["ttrv"]["parsed_answer"], "A")
        self.assertEqual(result["ours_target_option_text"], options["rwq-1"]["B"])
        self.assertEqual(result["ttrv_target_option_text"], options["rwq-1"]["A"])
        self.assertTrue(
            result["retained_predictions"]["ours"]["provenance"][
                "current_checkpoint_aligned"
            ]
        )
        self.assertTrue(summary["selection_outcome_dependent"])
        self.assertTrue(summary["selection_current_checkpoint_aligned"])

    def test_seed_word_limit_and_boolean_gate(self):
        cases = [
            make_case("seed-pass", "SEED"),
            make_case("seed-long", "SEED"),
            make_case("seed-bool", "SEED"),
        ]
        records = sum((make_records(case["case_id"]) for case in cases), [])
        options = {
            "seed-pass": {"A": "Right of the team", "B": "Above the team"},
            "seed-long": {
                "A": "Right of the basketball team",
                "B": "Above the basketball team",
            },
            "seed-bool": {"A": "No", "B": "Yes"},
        }
        selected, summary = select(cases, records, options)
        self.assertEqual([case["case_id"] for case in selected], ["seed-pass"])
        counts = summary["datasets"]["SEED"]
        self.assertEqual(counts["eligible_before_cap"], 1)
        self.assertEqual(counts["excluded_ttrv_lexical:too_many_words"], 1)
        self.assertEqual(counts["excluded_ours_lexical:yes_no_boolean"], 1)

    def test_normalized_identical_target_text_is_excluded(self):
        case = make_case("rwq-same", "RealWorldQA")
        options = {
            "rwq-same": {
                "A": "  Right   lane  ",
                "B": "Right lane",
            }
        }
        selected, summary = select([case], make_records("rwq-same"), options)
        self.assertEqual(selected, [])
        self.assertEqual(
            summary["datasets"]["RealWorldQA"][
                "excluded_identical_normalized_target_text"
            ],
            1,
        )

    def test_stable_sha256_cap_is_input_order_independent(self):
        cases = [
            make_case("rwq-a", "RealWorldQA"),
            make_case("rwq-b", "RealWorldQA"),
        ]
        records = make_records("rwq-a") + make_records("rwq-b")
        options = {
            "rwq-a": {"A": "left lane", "B": "right lane"},
            "rwq-b": {"A": "green light", "B": "red light"},
        }
        first, _ = select(cases, records, options, max_per_dataset=1)
        second, _ = select(
            list(reversed(cases)), list(reversed(records)), options,
            max_per_dataset=1,
        )
        self.assertEqual(first[0]["case_id"], second[0]["case_id"])
        self.assertEqual(first[0]["selection_rank_within_dataset"], 1)

    def test_invalid_outcomes_and_reprobe_manifest_are_rejected(self):
        case = make_case("rwq-wrong", "RealWorldQA")
        options = {"rwq-wrong": {"A": "left lane", "B": "right lane"}}
        selected, summary = select(
            [case], make_records("rwq-wrong", ours="A", ttrv="B"), options
        )
        self.assertEqual(selected, [])
        self.assertEqual(
            summary["datasets"]["RealWorldQA"]["excluded_ours_not_correct"], 1
        )
        validate_manifest([case])
        case["historical_retained_predictions"] = {}
        with self.assertRaisesRegex(ValueError, "already has"):
            validate_manifest([case])


if __name__ == "__main__":
    unittest.main()
