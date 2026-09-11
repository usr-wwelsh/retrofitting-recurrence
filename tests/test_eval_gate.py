import json
import tempfile
import unittest
from pathlib import Path

from eval_gate import compute_verdict, run_gate

_TASKS = ("arc_easy", "hellaswag")


def _results(arc_easy_acc: float, hellaswag_acc_norm: float) -> dict:
    return {
        "results": {
            "arc_easy": {"acc,none": arc_easy_acc, "alias": "arc_easy"},
            "hellaswag": {"acc_norm,none": hellaswag_acc_norm, "alias": "hellaswag"},
        }
    }


class ComputeVerdictTests(unittest.TestCase):
    def test_matching_baseline_passes(self):
        baseline = _results(0.5, 0.4)
        verdict = compute_verdict(baseline, baseline, tasks=_TASKS)
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.relative_drop, 0.0)

    def test_improvement_over_baseline_passes(self):
        checkpoint = _results(0.6, 0.5)
        baseline = _results(0.5, 0.4)
        verdict = compute_verdict(checkpoint, baseline, tasks=_TASKS)
        self.assertTrue(verdict.passed)
        self.assertLess(verdict.relative_drop, 0.0)

    def test_small_drop_within_threshold_passes(self):
        checkpoint = _results(0.49, 0.39)
        baseline = _results(0.5, 0.4)
        verdict = compute_verdict(checkpoint, baseline, tasks=_TASKS, regression_threshold=0.05)
        self.assertTrue(verdict.passed)

    def test_large_drop_beyond_threshold_fails(self):
        checkpoint = _results(0.3, 0.2)
        baseline = _results(0.5, 0.4)
        verdict = compute_verdict(checkpoint, baseline, tasks=_TASKS, regression_threshold=0.05)
        self.assertFalse(verdict.passed)
        self.assertAlmostEqual(verdict.relative_drop, (0.45 - 0.25) / 0.45)

    def test_uses_acc_norm_when_present_else_acc(self):
        checkpoint = _results(0.5, 0.4)
        baseline = _results(0.5, 0.4)
        verdict = compute_verdict(checkpoint, baseline, tasks=_TASKS)
        self.assertEqual(verdict.per_task["arc_easy"]["checkpoint"], 0.5)
        self.assertEqual(verdict.per_task["hellaswag"]["checkpoint"], 0.4)


class RunGateTests(unittest.TestCase):
    def test_pass_writes_verdict_and_exits_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "ckpt"
            ckpt_path.mkdir()
            baseline_path = Path(tmp) / "baseline.json"
            baseline_path.write_text(json.dumps(_results(0.5, 0.4)))

            verdict = run_gate(
                str(ckpt_path),
                str(baseline_path),
                tasks=_TASKS,
                evaluate_fn=lambda *_args: _results(0.5, 0.4),
            )

            self.assertTrue(verdict.passed)
            written = json.loads((ckpt_path / "eval_gate_verdict.json").read_text())
            self.assertTrue(written["passed"])

    def test_regression_fails_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "ckpt"
            ckpt_path.mkdir()
            baseline_path = Path(tmp) / "baseline.json"
            baseline_path.write_text(json.dumps(_results(0.5, 0.4)))

            verdict = run_gate(
                str(ckpt_path),
                str(baseline_path),
                tasks=_TASKS,
                evaluate_fn=lambda *_args: _results(0.2, 0.1),
            )

            self.assertFalse(verdict.passed)
            written = json.loads((ckpt_path / "eval_gate_verdict.json").read_text())
            self.assertFalse(written["passed"])


if __name__ == "__main__":
    unittest.main()
