"""
Streams and packs the SmolLM2-recurrent continued-pretraining mix at its
native 60% FineWeb-Edu / 40% DCLM / ~4% Cosmopedia-v2 ratio, without any of
the cluster-specific assumptions in mix_datasets.py / preprocess_data_packing.py
(no hardcoded /p/vast paths, no num_proc=96) -- meant to run on a single
Colab/Kaggle machine.

Note: HuggingFaceTB/smollm-corpus does NOT itself contain a DCLM split (it
only has cosmopedia-v2, fineweb-edu-dedup, python-edu) -- DCLM is pulled
from the separate, edu-filtered HuggingFaceTB/dclm-edu dataset instead.

Unlike preprocess_data_packing.py's use of trl.pack_dataset (which needs a
materialized, sharded Dataset), this packs a live IterableDataset stream
manually: concatenate token ids across documents (BOS-separated, "wrapped"
packing -- no padding, matching preprocess_data_packing.py's
wrapped_packing=True default) and cut into fixed max_length+1 blocks as they
become available, writing out a parquet shard every `rows_per_shard` blocks
so memory stays bounded regardless of total token budget.

Output layout matches what stateful_parquet_dataset.py (train.py's
--is_parquet_dataset loader) expects: a flat directory of
shard-*.parquet files, each row holding fixed-length `input_ids` /
`attention_mask` columns.
"""

import os

from datasets import Dataset, interleave_datasets, load_dataset
from jsonargparse import CLI
from transformers import AutoTokenizer

SOURCES = {
    "fineweb-edu": dict(path="HuggingFaceTB/smollm-corpus", name="fineweb-edu-dedup", weight=0.60),
    "dclm": dict(path="HuggingFaceTB/dclm-edu", name=None, weight=0.40),
    "cosmopedia-v2": dict(path="HuggingFaceTB/smollm-corpus", name="cosmopedia-v2", weight=0.04),
}


def load_source(spec, seed, shuffle_buffer_size):
    ds = load_dataset(spec["path"], spec["name"], split="train", streaming=True)
    if shuffle_buffer_size > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    if "text" not in ds.features:
        raise ValueError(
            f"{spec['path']}/{spec['name']} has no 'text' column -- "
            f"got {list(ds.features)}. Update the column mapping below."
        )
    return ds.select_columns(["text"])


def process(
    tokenizer_name: str = "convert_pretrained_model/models/SmolLM2-360M",
    save_path: str = "data/smollm2_recurrent_mix",
    max_length: int = 1024,
    token_budget: int = 500_000_000,
    rows_per_shard: int = 20_000,
    seed: int = 42,
    shuffle_buffer_size: int = 10_000,
    log_every: int = 1_000,
):
    """
    Args:
        tokenizer_name: SmolLM2 tokenizer (must match the converted model).
        save_path: output dir for shard-*.parquet files.
        max_length: sequence length; rows are stored at max_length+1 tokens
            (train.py slices off the last position for next-token labels).
        token_budget: stop once roughly this many packed tokens are written.
        rows_per_shard: packed rows buffered in memory before flushing a
            shard -- keeps peak memory bounded independent of token_budget.
        seed: shuffle/interleave seed.
        shuffle_buffer_size: per-source streaming shuffle buffer; a source
            yields nothing until this many of its examples have been fetched,
            so lower it (e.g. 0 to disable) for quick local smoke tests.
        log_every: print progress every this many raw documents consumed.
    """
    os.makedirs(save_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    weights = [s["weight"] for s in SOURCES.values()]
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    streams = [load_source(spec, seed, shuffle_buffer_size) for spec in SOURCES.values()]
    mixed = interleave_datasets(streams, probabilities=probabilities, seed=seed)

    block_len = max_length + 1
    buffer: list[int] = []
    rows: list[dict] = []
    shard_idx = 0
    total_tokens_written = 0

    def flush_shard():
        nonlocal rows, shard_idx
        if not rows:
            return
        Dataset.from_list(rows).to_parquet(f"{save_path}/shard-{shard_idx:05d}.parquet")
        print(f"wrote shard-{shard_idx:05d}.parquet ({len(rows)} rows, {total_tokens_written:,} total tokens)")
        rows = []
        shard_idx += 1

    for doc_idx, example in enumerate(mixed, start=1):
        if doc_idx % log_every == 0:
            print(f"consumed {doc_idx:,} docs, {total_tokens_written:,} tokens packed so far")

        ids = tokenizer(example["text"], add_special_tokens=False)["input_ids"]
        buffer.append(tokenizer.bos_token_id)
        buffer.extend(ids)

        while len(buffer) >= block_len:
            block = buffer[:block_len]
            buffer = buffer[block_len:]
            rows.append({"input_ids": block, "attention_mask": [1] * block_len})
            total_tokens_written += block_len

            if len(rows) >= rows_per_shard:
                flush_shard()

        if total_tokens_written >= token_budget:
            break

    flush_shard()
    print(f"done: {shard_idx} shards, {total_tokens_written:,} tokens written to {save_path}")


if __name__ == "__main__":
    CLI(process)
