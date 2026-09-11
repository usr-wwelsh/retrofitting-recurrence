"""Math correctness reward for GRPO, via math-verify (github.com/huggingface/Math-Verify).

Mirrors Open-R1's accuracy_reward (src/open_r1/rewards.py) exactly: parse the gold answer
and the completion, verify they're mathematically equivalent. Gold answers must already be in
a math-verify-parseable form (e.g. wrapped in \\boxed{...}) -- normalizing a raw dataset's gold
field into that form is mix_grpo_data.py's job, not this function's.
"""

from math_verify import parse, verify


def math_reward(completions: list[str], gold_answers: list[str]) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, gold_answers):
        try:
            gold_parsed = parse(gold)
            if not gold_parsed:
                rewards.append(0.0)
                continue
            answer_parsed = parse(completion)
            rewards.append(1.0 if verify(gold_parsed, answer_parsed) else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards
