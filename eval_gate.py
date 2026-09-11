"""Regression gate between training stages.

Run after each stage (phase1, phase2, SFT) from shells/full_pipeline_vast.sh. Compares a
checkpoint against a cached base-SmolLM2-360M lm-eval-harness result on general-knowledge
tasks (deliberately excluding gsm8k -- that's the "did it improve" signal, not the
"did it regress" one, see GPU_PRICING_NOTES.md). Exits non-zero on regression beyond
`regression_threshold` so the calling shell script's `set -e` halts the pipeline before the
next stage's GPU-hours are spent, and fires a wandb alert if a run is active -- same
one-shot-alert-on-bad-news shape as train.py's non-finite-loss/crash alerts.

lm-eval-harness itself is not imported at module scope: it's heavy and GPU-bound, so
`evaluate_fn` is injectable for testing (the pure regression/threshold logic below is what's
actually worth covering with fast tests; lm-eval-harness's own correctness is out of scope).
"""

import dataclasses
import json
import sys
from pathlib import Path
from typing import Callable

DEFAULT_TASKS = ("arc_easy", "arc_challenge", "hellaswag", "mmlu", "piqa", "winogrande")


@dataclasses.dataclass
class GateVerdict:
    passed: bool
    checkpoint_mean: float
    baseline_mean: float
    relative_drop: float
    per_task: dict


def _primary_metric(task_result: dict) -> float:
    for key in ("acc_norm,none", "acc,none"):
        if key in task_result:
            return task_result[key]
    raise KeyError(f"No acc/acc_norm metric found in task result: {task_result.keys()}")


def compute_verdict(
    checkpoint_results: dict,
    baseline_results: dict,
    tasks=DEFAULT_TASKS,
    regression_threshold: float = 0.05,
) -> GateVerdict:
    per_task = {}
    for task in tasks:
        per_task[task] = {
            "checkpoint": _primary_metric(checkpoint_results["results"][task]),
            "baseline": _primary_metric(baseline_results["results"][task]),
        }
    checkpoint_mean = sum(v["checkpoint"] for v in per_task.values()) / len(per_task)
    baseline_mean = sum(v["baseline"] for v in per_task.values()) / len(per_task)
    relative_drop = (baseline_mean - checkpoint_mean) / baseline_mean if baseline_mean else 0.0
    return GateVerdict(
        passed=relative_drop <= regression_threshold,
        checkpoint_mean=checkpoint_mean,
        baseline_mean=baseline_mean,
        relative_drop=relative_drop,
        per_task=per_task,
    )


def _default_evaluate(ckpt_path: str, mean_recurrence: int, tasks) -> dict:
    import lm_eval

    return lm_eval.simple_evaluate(
        model="hf",
        model_args=(
            f"pretrained={ckpt_path},mean_recurrence={mean_recurrence},"
            "add_bos_token=True,dtype=float32,trust_remote_code=True"
        ),
        tasks=list(tasks),
    )


def run_gate(
    ckpt_path: str,
    baseline_results_path: str,
    mean_recurrence: int = 8,
    tasks: tuple = DEFAULT_TASKS,
    regression_threshold: float = 0.05,
    evaluate_fn: Callable[[str, int, tuple], dict] = _default_evaluate,
) -> GateVerdict:
    baseline_results = json.loads(Path(baseline_results_path).read_text())
    checkpoint_results = evaluate_fn(ckpt_path, mean_recurrence, tasks)
    verdict = compute_verdict(checkpoint_results, baseline_results, tasks, regression_threshold)

    verdict_path = Path(ckpt_path) / "eval_gate_verdict.json"
    verdict_path.write_text(json.dumps(dataclasses.asdict(verdict), indent=2))

    if not verdict.passed:
        try:
            import wandb

            if wandb.run is not None:
                wandb.alert(
                    title="Eval gate: regression vs base",
                    text=(
                        f"{ckpt_path}: mean knowledge-task score {verdict.checkpoint_mean:.4f} "
                        f"vs baseline {verdict.baseline_mean:.4f} "
                        f"({verdict.relative_drop:.1%} relative drop, "
                        f"threshold {regression_threshold:.1%})"
                    ),
                    level=wandb.AlertLevel.ERROR,
                )
        except ImportError:
            pass
        print(
            f"REGRESSION: {verdict.relative_drop:.1%} drop vs baseline "
            f"(threshold {regression_threshold:.1%}) -- halting.",
            file=sys.stderr,
        )
    else:
        print(
            f"PASS: checkpoint mean {verdict.checkpoint_mean:.4f} vs "
            f"baseline {verdict.baseline_mean:.4f} ({verdict.relative_drop:.1%} relative drop)."
        )
    return verdict


def main(
    ckpt_path: str,
    baseline_results_path: str,
    mean_recurrence: int = 8,
    tasks: tuple = DEFAULT_TASKS,
    regression_threshold: float = 0.05,
):
    verdict = run_gate(ckpt_path, baseline_results_path, mean_recurrence, tasks, regression_threshold)
    sys.exit(0 if verdict.passed else 1)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
