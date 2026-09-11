"""
Builds the SFT dataset for train.py's chat-template path (cfg.preprocessed_data_path=None,
dataset_args={"q_col": "question", "a_col": "answer"}).

Not IFM/SFT-Reasoning as GPU_PRICING_NOTES.md originally assumed -- inspected via
HfFileSystem + pyarrow (footer/row-group reads, no full 2GB-shard downloads) and found to be
flat "text"/"token_count" rows with instruction and response concatenated with no reliable
delimiter (filenames like "pretrain_output_cleaned_oss_removal*.parquet" -- this is raw
pretraining-shaped data, not curated SFT records), so it can't feed a prompt-masked SFT loss.

Uses open-thoughts/OpenThoughts-114k instead: a real, widely-used verified CoT-reasoning SFT
dataset (math/code/science; Apache-2.0) with a clean schema -- `system` (one fixed prompt
across the dataset, confirmed by sampling) and `conversations` (always exactly one
user turn then one assistant turn, confirmed by sampling). train.py's chat-template path
only supports a flat single-turn (question, answer) pair, not an arbitrary conversation list,
so this flattens conversations[0]/conversations[1] into that shape and writes the dataset's
own system prompt out separately for train.py's --sys_prompt flag (train.py hardcodes one
system prompt for the whole run, not a per-row column).
"""

from datasets import load_dataset
from jsonargparse import CLI


def process(save_path: str = "data/sft_mix", split: str = "train"):
    """
    Args:
        save_path: output dir; the flattened dataset is written to `{save_path}/dataset`
            (Dataset.save_to_disk format -- pass this as train.py's --dataset_location) and
            the dataset's system prompt to `{save_path}/system_prompt.txt` (pass its contents
            as train.py's --sys_prompt).
        split: which split to use -- OpenThoughts-114k only has "train"; kept as an arg for
            symmetry with mix_grpo_data.py rather than hardcoded.
    """
    dataset = load_dataset("open-thoughts/OpenThoughts-114k", split=split)

    system_prompts = set(dataset["system"])
    if len(system_prompts) != 1:
        raise ValueError(
            f"Expected one fixed system prompt across the dataset, found {len(system_prompts)} -- "
            "the assumption this script was built on (checked by sampling 500 rows) no longer "
            "holds, re-inspect before proceeding."
        )
    system_prompt = system_prompts.pop()

    def flatten(row):
        turns = row["conversations"]
        if len(turns) != 2 or turns[0]["from"] != "user" or turns[1]["from"] != "assistant":
            raise ValueError(f"Expected exactly [user, assistant] turns, got {turns}")
        return {"question": turns[0]["value"], "answer": turns[1]["value"]}

    flattened = dataset.map(flatten, remove_columns=dataset.column_names)
    flattened.save_to_disk(f"{save_path}/dataset")

    system_prompt_path = f"{save_path}/system_prompt.txt"
    with open(system_prompt_path, "w") as f:
        f.write(system_prompt)

    print(f"done: {len(flattened)} rows written to {save_path}/dataset")
    print(f"system prompt written to {system_prompt_path} -- pass its contents as train.py's --sys_prompt")


if __name__ == "__main__":
    CLI(process)
