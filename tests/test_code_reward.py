import unittest

from verifiers.code_reward import code_reward

_ADD_COMPLETION = """Here's the function:
```python
def add(a, b):
    return a + b
```
"""

_BUGGY_ADD_COMPLETION = """Here's the function:
```python
def add(a, b):
    return a - b
```
"""


class CodeRewardTests(unittest.TestCase):
    def test_all_tests_pass_scores_one(self):
        rewards = code_reward(
            [_ADD_COMPLETION],
            [["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]],
        )
        self.assertEqual(rewards, [1.0])

    def test_some_tests_pass_scores_partial_credit(self):
        rewards = code_reward(
            [_BUGGY_ADD_COMPLETION],
            [["assert add(1, 2) == -1", "assert add(-1, 1) == 0"]],
        )
        self.assertEqual(rewards, [0.5])

    def test_no_code_block_scores_zero(self):
        rewards = code_reward(
            ["I refuse to write code."],
            [["assert add(1, 2) == 3"]],
        )
        self.assertEqual(rewards, [0.0])

    def test_syntax_error_scores_zero_without_raising(self):
        rewards = code_reward(
            ["```python\ndef add(a, b:\n    return a + b\n```"],
            [["assert add(1, 2) == 3"]],
        )
        self.assertEqual(rewards, [0.0])

    def test_infinite_loop_scores_zero_without_hanging(self):
        rewards = code_reward(
            ["```python\ndef add(a, b):\n    while True: pass\n```"],
            [["assert add(1, 2) == 3"]],
            timeout=1.0,
        )
        self.assertEqual(rewards, [0.0])

    def test_no_test_cases_scores_zero(self):
        rewards = code_reward([_ADD_COMPLETION], [[]])
        self.assertEqual(rewards, [0.0])

    def test_setup_code_runs_before_assertions(self):
        rewards = code_reward(
            [_ADD_COMPLETION],
            [["assert add(THREE, 2) == 5"]],
            setup_code=["THREE = 3"],
        )
        self.assertEqual(rewards, [1.0])


if __name__ == "__main__":
    unittest.main()
