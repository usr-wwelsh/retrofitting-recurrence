import unittest

from grpo_train import _build_generation_kwargs


class BuildGenerationKwargsTests(unittest.TestCase):
    def test_default_criterion_only_sets_num_steps(self):
        # no criterion opt-in -> rollouts must keep using the existing fixed-depth
        # generate() path, not silently switch dispatch to generate_with_adaptive_compute
        kwargs = _build_generation_kwargs(mean_recurrence=8, criterion=None, exit_threshold="auto")
        self.assertEqual(kwargs, {"num_steps": 8})

    def test_criterion_opt_in_adds_exit_threshold(self):
        kwargs = _build_generation_kwargs(mean_recurrence=8, criterion="latent-diff", exit_threshold="auto")
        self.assertEqual(kwargs, {"num_steps": 8, "criterion": "latent-diff", "exit_threshold": "auto"})

    def test_explicit_exit_threshold_is_forwarded(self):
        kwargs = _build_generation_kwargs(mean_recurrence=16, criterion="entropy-diff", exit_threshold="0.002")
        self.assertEqual(
            kwargs, {"num_steps": 16, "criterion": "entropy-diff", "exit_threshold": "0.002"}
        )


if __name__ == "__main__":
    unittest.main()
