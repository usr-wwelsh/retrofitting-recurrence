"""Integration test for mix_grpo_data.py -- downloads real gsm8k/mbpp train splits over the
network. Not part of the fast unit suite (tests/); run explicitly:
    python -m unittest tests.integration.test_mix_grpo_data
"""

import shutil
import tempfile
import unittest

from datasets import load_from_disk

from mix_grpo_data import process


class MixGrpoDataIntegrationTest(unittest.TestCase):
    def test_builds_combined_math_and_code_dataset(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            process(save_path=tmp_dir)
            ds = load_from_disk(tmp_dir)

            self.assertEqual(len(ds), 7473 + 374)
            self.assertEqual(
                set(ds.column_names),
                {"prompt", "task_type", "gold_answer", "test_list", "test_setup_code"},
            )

            task_types = set(ds["task_type"])
            self.assertEqual(task_types, {"math", "code"})

            math_row = next(r for r in ds if r["task_type"] == "math")
            self.assertTrue(math_row["gold_answer"].startswith("The answer is "))
            self.assertEqual(math_row["test_list"], [])

            code_row = next(r for r in ds if r["task_type"] == "code")
            self.assertEqual(code_row["gold_answer"], "")
            self.assertTrue(len(code_row["test_list"]) > 0)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
