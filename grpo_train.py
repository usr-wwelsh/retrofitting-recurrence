"""
GRPO/RLVR stage: post-trains the SFT checkpoint on verifiable math (GSM8K) and code (MBPP)
reward, via TRL's GRPOTrainer -- the actual battle-tested implementation (DeepSeekMath's GRPO),
not a hand-rolled RL loop. Uses verifiers.math_reward (math-verify) and verifiers.code_reward
(bwrap-sandboxed execution) as TRL reward functions.

A separate script from train.py on purpose: GRPO's rollout/generation/group-advantage loop is
a fundamentally different shape than train.py's next-token training loop, and train.py has no
hook system to bolt an RL trainer onto (single procedural loop, no callbacks).

trl==0.19.1 pinned in requirements.txt, not the latest release: trl>=0.20.0 requires
transformers>=4.53.2, which conflicts with this repo's transformers==4.51.0 pin --
raven_modeling_minimal_llama.py subclasses transformers.cache_utils.Cache/DynamicCache
internals that break on newer transformers (see README.md's "built to work with
transformers==4.51.0" note). 0.19.1 is the newest trl release whose transformers floor
(>=4.51.0) is still satisfied by the pin.

--mean_recurrence is forwarded to the model's own generate() as the `num_steps` kwarg (see
raven_modeling_minimal.py's generate()/iterate_forward -- same kwarg multi_recurence_eval.py
and the lm-eval-harness model_args use) via GRPOConfig.generation_kwargs, so rollouts actually
exercise the trained recurrence depth instead of silently falling back to config default.

Default hyperparameters target a real RLVR lift, not a smoke test: TRL's GRPOConfig requires
`per_device_train_batch_size * num_processes * gradient_accumulation_steps` (the "generation
batch") to be evenly divisible by num_generations -- the previous defaults
(micro_batch_size=4, num_generations=8, no gradient_accumulation_steps) violated this and
GRPOTrainer() raised ValueError before training could start. Now: micro_batch_size=8,
gradient_accumulation_steps=8 -> generation_batch_size=64, giving 4 distinct prompts per
optimizer step (64/num_generations=16) instead of the degenerate single-prompt-per-step case
you'd get at generation_batch_size==num_generations. max_steps=2000 * 4 prompts/step ~= one
full epoch over the combined 7,847-row gsm8k+mbpp set (mix_grpo_data.py's default output) --
the crash-avoiding minimum (micro_batch_size=8, num_generations unchanged, max_steps=500)
would have covered well under a quarter of that. max_completion_length raised 512->768 so
GSM8K CoT + MBPP solutions aren't truncated mid-answer. save_total_limit=1 prunes local GRPO
checkpoints the same way train.py's save_checkpoint/save_model_only already do -- otherwise
HF Trainer keeps every save_interval checkpoint on disk forever.
"""

import os

from datasets import load_from_disk
from jsonargparse import CLI
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from verifiers.code_reward import code_reward
from verifiers.math_reward import math_reward


def _math_reward_fn(completions, gold_answer, **_kwargs):
    return math_reward(completions, gold_answer)


def _code_reward_fn(completions, test_list, test_setup_code, **_kwargs):
    return code_reward(completions, test_list, setup_code=test_setup_code)


def main(
    model_name: str,
    dataset_path: str,
    out_path: str,
    run_name: str,
    hub_checkpoint_repo: str | None = None,
    mean_recurrence: int = 8,
    max_steps: int = 2000,
    learning_rate: float = 2e-6,
    num_generations: int = 16,
    max_prompt_length: int = 512,
    max_completion_length: int = 768,
    micro_batch_size: int = 8,
    gradient_accumulation_steps: int = 8,
    save_interval: int = 100,
    save_total_limit: int = 1,
    wandb_disabled: bool = False,
    wandb_project: str = "smollm2-recurrent",
):
    """
    Args:
        model_name: SFT checkpoint to load (local model_only_chkpt_* dir or a Hub repo id).
        dataset_path: mix_grpo_data.py's output dir (Dataset.save_to_disk format), with
            prompt/task_type/gold_answer/test_list/test_setup_code columns.
        out_path: where GRPOTrainer writes checkpoints (HF Trainer's output_dir).
        run_name: wandb run name / checkpoint subdir name.
        hub_checkpoint_repo: if set, push the final model here (HF model repo).
        mean_recurrence: recurrence depth for rollout generation -- forwarded as generate()'s
            `num_steps` kwarg, should match the checkpoint's trained ceiling.
        max_steps, learning_rate, num_generations, max_prompt_length, max_completion_length,
            micro_batch_size, gradient_accumulation_steps, save_interval, save_total_limit:
            GRPOConfig hyperparameters -- see module docstring for why the defaults are sized
            the way they are. micro_batch_size * gradient_accumulation_steps must stay a
            multiple of num_generations or GRPOTrainer() raises at construction time.
        wandb_disabled, wandb_project: same convention as train.py.
    """
    os.environ["WANDB_PROJECT"] = wandb_project

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    dataset = load_from_disk(dataset_path)

    config = GRPOConfig(
        output_dir=f"{out_path}/{run_name}",
        run_name=run_name,
        max_steps=max_steps,
        learning_rate=learning_rate,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_interval,
        save_total_limit=save_total_limit,
        generation_kwargs={"num_steps": mean_recurrence},
        report_to=["wandb"] if not wandb_disabled else [],
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[_math_reward_fn, _code_reward_fn],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    final_path = f"{out_path}/{run_name}/model_only_final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    if hub_checkpoint_repo is not None:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(hub_checkpoint_repo, private=False, exist_ok=True)
        api.upload_folder(repo_id=hub_checkpoint_repo, folder_path=final_path)


if __name__ == "__main__":
    CLI(main)
