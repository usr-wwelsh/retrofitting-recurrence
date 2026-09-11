import unittest

from verifiers.math_reward import math_reward


class MathRewardTests(unittest.TestCase):
    def test_correct_boxed_answer_scores_one(self):
        rewards = math_reward(
            ["Reasoning... the answer is \\boxed{42}."],
            ["\\boxed{42}"],
        )
        self.assertEqual(rewards, [1.0])

    def test_wrong_answer_scores_zero(self):
        rewards = math_reward(
            ["Reasoning... the answer is \\boxed{7}."],
            ["\\boxed{42}"],
        )
        self.assertEqual(rewards, [0.0])

    def test_equivalent_but_differently_formatted_answer_scores_one(self):
        rewards = math_reward(
            ["The answer is \\boxed{1/2}."],
            ["\\boxed{0.5}"],
        )
        self.assertEqual(rewards, [1.0])

    def test_unparseable_completion_scores_zero_without_raising(self):
        rewards = math_reward(
            ["I don't know, sorry."],
            ["\\boxed{42}"],
        )
        self.assertEqual(rewards, [0.0])

    def test_scores_a_batch_independently(self):
        rewards = math_reward(
            ["\\boxed{4}", "\\boxed{5}"],
            ["\\boxed{4}", "\\boxed{4}"],
        )
        self.assertEqual(rewards, [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
