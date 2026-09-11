"""
Builds the verifiable-reward dataset for grpo_train.py.

Deliberately NOT IFM/Math-Reasoning or IFM/Code-Reasoning (mix_smollm2_corpus.py's phase-2
sources) -- those are flat "text" columns with no gold answer or test cases, fine for
continued-pretraining but useless for a verifiable RL reward. GRPO needs a dataset where
correctness can actually be checked, so this uses two small, standard, train/test-split
datasets instead:
  - gsm8k (train split, 7,473 rows) for math -- gold answer extracted from the "#### N"
    suffix and reformatted as "The answer is N." to match both math_verify's parser and the
    exact phrasing eval_yamls/gsm8k-cot-sean.yaml's regex expects, so the SFT target format,
    GRPO reward parsing, and eval scoring all agree on one convention.
  - mbpp (train split, 374 rows) for code -- test_list is already assert-statement test
    cases, exactly what verifiers/code_reward.py's sandboxed execution expects.
Both datasets' *test* splits are reserved for eval, never touched here.

Small and non-streaming, unlike mix_smollm2_corpus.py's multi-billion-row streamed sources --
both fit in memory whole.
"""

import re

from datasets import Dataset, concatenate_datasets, load_dataset
from jsonargparse import CLI

_GSM8K_PROMPT = "Q: {question}\n\nA: Let's think step by step."
_MBPP_PROMPT = (
    "Write a Python function for the following problem. "
    "Return only the function in a single ```python code block.\n\n{text}"
)


def _gsm8k_gold_answer(answer: str) -> str:
    final = answer.split("####")[-1].strip()
    final = re.sub(r"[,$]", "", final)
    return f"The answer is {final}."


def _build_math_rows(split: str) -> Dataset:
    gsm8k = load_dataset("gsm8k", "main", split=split)
    return gsm8k.map(
        lambda row: {
            "prompt": _GSM8K_PROMPT.format(question=row["question"]),
            "task_type": "math",
            "gold_answer": _gsm8k_gold_answer(row["answer"]),
            "test_list": [],
            "test_setup_code": "",
        },
        remove_columns=gsm8k.column_names,
    )


def _build_code_rows(split: str) -> Dataset:
    mbpp = load_dataset("mbpp", split=split)
    return mbpp.map(
        lambda row: {
            "prompt": _MBPP_PROMPT.format(text=row["text"]),
            "task_type": "code",
            "gold_answer": "",
            "test_list": row["test_list"],
            "test_setup_code": row["test_setup_code"],
        },
        remove_columns=mbpp.column_names,
    )


def process(
    save_path: str = "data/grpo_mix",
    gsm8k_split: str = "train",
    mbpp_split: str = "train",
    seed: int = 42,
):
    """
    Args:
        save_path: output dir for the combined dataset (Dataset.save_to_disk format).
        gsm8k_split: which gsm8k split to pull math rows from -- must stay "train" for an
            actual GRPO run; only overridable for building a matching held-out eval set.
        mbpp_split: which mbpp split to pull code rows from -- same caveat.
        seed: shuffle seed, so math/code rows are interleaved rather than all-math-then-all-code.
    """
    math_rows = _build_math_rows(gsm8k_split)
    code_rows = _build_code_rows(mbpp_split)
    combined = concatenate_datasets([math_rows, code_rows]).shuffle(seed=seed)
    combined.save_to_disk(save_path)
    print(
        f"done: {len(math_rows)} math rows + {len(code_rows)} code rows "
        f"= {len(combined)} total, written to {save_path}"
    )


if __name__ == "__main__":
    CLI(process)
