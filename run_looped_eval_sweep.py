"""Go/no-go sanity check for training-free looped inference (arXiv:2605.23872)
on SmolLM2-360M -- NOT the retrofitted-recurrence checkpoints this repo trains
elsewhere. Base model is the plain HuggingFaceTB/SmolLM2-360M pretrained
weights, completely frozen: no architecture conversion, no fine-tuning.

Sweeps a couple of damped mid-stack looping configs (see looped_inference.py)
against the same general-knowledge task set eval_gate.py uses for regression
checks (arc_easy, arc_challenge, hellaswag, mmlu, piqa, winogrande -- must NOT
regress) plus gsm8k-cot (this repo's eval_yamls/gsm8k-cot-sean.yaml if
present, else stock gsm8k_cot_zeroshot) as the "did it get smarter" signal.

Eval sets are heavily subsampled (see TASK_LIMITS) -- this machine is CPU-only
and generation is the expensive path. This is a go/no-go signal, not a
paper-quality result: treat exact numbers as noisy at these sample sizes.

Usage: .venv/bin/python run_looped_eval_sweep.py
"""

import json
import time
from pathlib import Path

import lm_eval
import lm_eval.tasks
import torch
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from looped_inference import LoopConfig, attach_loop_hook

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
BLOCK_START = 10
BLOCK_END = 17  # inclusive; 8-layer contiguous mid-stack block (of 32), away
# from layers 0/31 which analyze_layer_importance.py found to be BI outliers
# (~1.0 and ~0.68 vs ~0.02-0.06 for the rest) -- i.e. clearly load-bearing,
# not good candidates to loop in a naive contiguous block.

GENERAL_TASKS = ("arc_easy", "arc_challenge", "hellaswag", "mmlu", "piqa", "winogrande")
_GSM8K_YAML = Path("eval_yamls/gsm8k-cot-sean.yaml")
GSM8K_TASK = "gsm8k_cot_sean" if _GSM8K_YAML.exists() else "gsm8k_cot_zeroshot"

TASK_LIMITS = {
    "arc_easy": 20,
    "arc_challenge": 20,
    "hellaswag": 20,
    "mmlu": 2,  # per-subtask; mmlu expands to 57 subjects -> 114 docs total
    "piqa": 20,
    "winogrande": 20,
    GSM8K_TASK: 8,
}

CONFIGS = {
    "base": None,
    "K2_eta0.5": LoopConfig(start_idx=BLOCK_START, end_idx=BLOCK_END, k=2, eta=0.5),
    "K4_eta0.3": LoopConfig(start_idx=BLOCK_START, end_idx=BLOCK_END, k=4, eta=0.3),
}

_PRIMARY_METRIC_KEYS = ("exact_match,strict-match", "acc_norm,none", "acc,none")


def _primary_metric(task_result: dict) -> float:
    for key in _PRIMARY_METRIC_KEYS:
        if key in task_result:
            return task_result[key]
    raise KeyError(f"No known metric in {task_result.keys()}")


def _run_task(lm, task, limit, task_manager):
    res = lm_eval.simple_evaluate(
        model=lm,
        tasks=[task],
        limit=limit,
        verbosity="ERROR",
        task_manager=task_manager,
        bootstrap_iters=0,
    )
    return res["results"][task]


def print_table(results: dict, tasks, out=print):
    config_names = list(results.keys())
    base = results["base"]
    col_w = 22
    out("task".ljust(16) + "".join(c.ljust(col_w) for c in config_names))
    for task in tasks:
        row = task.ljust(16)
        for c in config_names:
            m = results[c][task]["metric"]
            cell = f"{m:.4f}" if c == "base" else f"{m:.4f} ({m - base[task]['metric']:+.4f})"
            row += cell.ljust(col_w)
        out(row)


def main(out_path: str = "looped_eval_results.json", batch_size: int = 4):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    lm = HFLM(pretrained=model, tokenizer=tok, device="cpu", batch_size=batch_size)

    task_manager = lm_eval.tasks.TaskManager(include_path="eval_yamls")
    all_tasks = list(GENERAL_TASKS) + [GSM8K_TASK]

    results = {}
    for cfg_name, cfg in CONFIGS.items():
        handle = attach_loop_hook(model, cfg) if cfg is not None else None
        results[cfg_name] = {}
        for task in all_tasks:
            t0 = time.time()
            task_result = _run_task(lm, task, TASK_LIMITS[task], task_manager)
            dt = time.time() - t0
            metric = _primary_metric(task_result)
            results[cfg_name][task] = {"metric": float(metric), "seconds": dt, "raw": task_result}
            print(f"[{cfg_name}] {task}: {metric:.4f} ({dt:.1f}s)", flush=True)
        if handle is not None:
            handle.remove()
        Path(out_path).write_text(json.dumps(results, indent=2, default=str))

    print()
    print_table(results, all_tasks)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
