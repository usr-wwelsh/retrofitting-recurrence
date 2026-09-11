"""Code correctness reward for GRPO: extract code from a completion, run it against
MBPP-style assert test cases inside sandbox_exec's bwrap sandbox, reward = fraction passed.

Modeled on Open-R1's code_reward (src/open_r1/rewards.py) -- same shape (extract code, run
tests, score by pass rate) -- but with a local bwrap sandbox as the execution backend instead
of a cloud provider (E2B/Piston/Morph), since GRPO here runs on a single rented GPU box rather
than needing a remotely-scalable execution service.
"""

import re

from sandbox_exec import run_sandboxed

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

_REWARD_MARKER_RE = re.compile(r"REWARD:(\d+)/(\d+)")


def _extract_code(completion: str) -> str:
    match = _CODE_BLOCK_RE.search(completion)
    return match.group(1) if match else completion


def _build_script(code: str, test_cases: list[str], setup_code: str = "") -> str:
    checks = "\n".join(
        f"try:\n    {test}\n    _passed += 1\nexcept Exception:\n    pass"
        for test in test_cases
    )
    setup = f"{setup_code}\n" if setup_code else ""
    return f"{code}\n{setup}\n_passed = 0\n{checks}\nprint(f'REWARD:{{_passed}}/{len(test_cases)}')\n"


def code_reward(
    completions: list[str],
    test_cases: list[list[str]],
    setup_code: list[str] | None = None,
    timeout: float = 10.0,
) -> list[float]:
    setup_code = setup_code or [""] * len(completions)
    rewards = []
    for completion, tests, setup in zip(completions, test_cases, setup_code):
        if not tests:
            rewards.append(0.0)
            continue
        code = _extract_code(completion)
        script = _build_script(code, tests, setup)
        result = run_sandboxed(script, timeout=timeout)
        match = _REWARD_MARKER_RE.search(result.stdout)
        rewards.append(int(match.group(1)) / int(match.group(2)) if match else 0.0)
    return rewards
