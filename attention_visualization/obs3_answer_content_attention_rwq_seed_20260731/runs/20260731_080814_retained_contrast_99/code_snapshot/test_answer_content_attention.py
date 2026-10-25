import unittest

from answer_content_attention import (
    _parse_models,
    canonical_choice,
    parse_options,
    resolve_target_label,
)
from screen_answer_content_v2 import parse_prompt_options


class AnswerContentAttentionTest(unittest.TestCase):
    def test_parse_ai2d_options(self):
        row = {
            "prompt": [
                {
                    "role": "user",
                    "content": "<image>\nQuestion?\nOptions:\nA. Mussel\nB. Starfish\nC. Seal\nD. Herring",
                }
            ]
        }
        self.assertEqual(
            parse_options(row),
            {"A": "Mussel", "B": "Starfish", "C": "Seal", "D": "Herring"},
        )

    def test_parse_mme_options(self):
        row = {"prompt": [{"role": "user", "content": "Question?\nOptions:\nA. No \nB. Yes."}]}
        self.assertEqual(parse_options(row), {"A": "No", "B": "Yes."})

    def test_parse_realworldqa_dot_options_without_heading(self):
        content = (
            "<image>\nWhere is the remote pointed?\n"
            "A. Towards the right\nB. Centered\nC. Towards the left\n"
            "Please answer directly with only the letter of the correct option and nothing else."
        )
        expected = {
            "A": "Towards the right",
            "B": "Centered",
            "C": "Towards the left",
        }
        self.assertEqual(parse_options({"prompt": [{"role": "user", "content": content}]}), expected)
        self.assertEqual(parse_prompt_options(content), expected)

    def test_parse_realworldqa_colon_options_without_heading(self):
        content = (
            "<image>\nHow many cones are on the left?\n\n"
            "A: 8 orange cones\nB: 7 orange cones\nC: 6 orange cones\n"
            "Please respond with only the corresponding option letter."
        )
        expected = {
            "A": "8 orange cones",
            "B": "7 orange cones",
            "C": "6 orange cones",
        }
        self.assertEqual(parse_options({"prompt": [{"role": "user", "content": content}]}), expected)
        self.assertEqual(parse_prompt_options(content), expected)

    def test_target_sources_and_canonical(self):
        case = {
            "ground_truth": "C",
            "retained_predictions": {"ours": {"parsed_answer": "C"}, "ttrv": {"parsed_answer": "B"}},
        }
        options = {"A": "Mussel", "B": "Starfish", "C": "Seal", "D": "Herring"}
        self.assertEqual(resolve_target_label(case, "ours", "ground_truth"), "C")
        self.assertEqual(resolve_target_label(case, "ttrv", "retained"), "B")
        self.assertEqual(canonical_choice("C", options), ("C. Seal", "Seal"))

    def test_model_parser(self):
        self.assertEqual(_parse_models("ours,ttrv"), ("ours", "ttrv"))
        with self.assertRaises(ValueError):
            _parse_models("ours,ours")


if __name__ == "__main__":
    unittest.main()
