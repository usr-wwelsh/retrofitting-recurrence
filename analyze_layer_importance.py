"""
ShortGPT-style layer importance analysis for SmolLM2-360M.

Computes Block Influence (BI) per layer: BI_i = 1 - mean cosine similarity
between a layer's input and output hidden states over a calibration set.
Low BI means the layer's transformation is close to identity (redundant,
safe to drop from the looped core). High BI means the layer does real work
and should be kept.

Fixes prelude/coda as the first/last N layers (structural convention used
by the retrofitting-recurrence repo's TinyLlama config), then ranks the
remaining middle layers by BI and drops the most redundant ones at the
same ~43% middle-layer drop rate the paper used for TinyLlama (6 of 14
middle layers dropped, keeping 8 as the core).

Usage: .venv/bin/python analyze_layer_importance.py
"""
import json

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
PRELUDE_SIZE = 4
CODA_SIZE = 4
NUM_CALIBRATION_DOCS = 32
SEQ_LEN = 512
MIDDLE_DROP_FRACTION = 6 / 14  # matches TinyLlama-1.1B config in this repo


def load_calibration_batches(tokenizer):
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    texts = []
    for row in ds:
        text = row["text"].strip()
        if len(text) < 200:
            continue
        texts.append(text)
        if len(texts) >= NUM_CALIBRATION_DOCS:
            break
    batches = []
    for text in texts:
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=SEQ_LEN
        )
        if enc["input_ids"].shape[1] < 32:
            continue
        batches.append(enc)
    return batches


def compute_block_influence(model, batches):
    num_layers = model.config.num_hidden_layers
    bi_sum = torch.zeros(num_layers)
    bi_count = 0

    with torch.no_grad():
        for enc in batches:
            out = model(**enc, output_hidden_states=True)
            hidden_states = out.hidden_states  # tuple: embeds, layer_1, ..., layer_N
            for i in range(num_layers):
                h_in = hidden_states[i][0]  # (seq_len, hidden)
                h_out = hidden_states[i + 1][0]
                cos_sim = torch.nn.functional.cosine_similarity(h_in, h_out, dim=-1)
                bi_sum[i] += (1.0 - cos_sim).sum().item()
            bi_count += hidden_states[0].shape[1]

    return (bi_sum / bi_count).tolist()


def select_layers(bi_scores, num_layers):
    prelude_idx = list(range(0, PRELUDE_SIZE))
    coda_idx = list(range(num_layers - CODA_SIZE, num_layers))
    middle_idx = list(range(PRELUDE_SIZE, num_layers - CODA_SIZE))

    num_drop = round(len(middle_idx) * MIDDLE_DROP_FRACTION)
    middle_by_bi = sorted(middle_idx, key=lambda i: bi_scores[i])  # ascending: most redundant first
    dropped_idx = sorted(middle_by_bi[:num_drop])
    core_idx = sorted(middle_by_bi[num_drop:])

    return prelude_idx, core_idx, coda_idx, dropped_idx


def main():
    print(f"Loading {MODEL_NAME} and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    num_layers = model.config.num_hidden_layers

    print(f"Streaming {NUM_CALIBRATION_DOCS} calibration docs from FineWeb-Edu...")
    batches = load_calibration_batches(tokenizer)
    print(f"Got {len(batches)} usable sequences.")

    print("Computing per-layer Block Influence (this may take a few minutes on CPU)...")
    bi_scores = compute_block_influence(model, batches)

    print("\nLayer  BI score  (low = redundant / safe to drop)")
    for i, score in enumerate(bi_scores):
        print(f"{i:5d}  {score:.5f}")

    prelude_idx, core_idx, coda_idx, dropped_idx = select_layers(bi_scores, num_layers)

    print(f"\nprelude_idx ({len(prelude_idx)}): {prelude_idx}")
    print(f"core_idx    ({len(core_idx)}): {core_idx}")
    print(f"coda_idx    ({len(coda_idx)}): {coda_idx}")
    print(f"dropped_idx ({len(dropped_idx)}): {dropped_idx}")

    mapping_cfg = {"prelude_idx": prelude_idx, "core_idx": core_idx, "coda_idx": coda_idx}
    result = {
        "model_name": MODEL_NAME,
        "num_layers": num_layers,
        "bi_scores": bi_scores,
        "mapping_cfg": mapping_cfg,
        "dropped_idx": dropped_idx,
        "num_calibration_sequences": len(batches),
    }
    with open("layer_importance_smollm2_360m.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to layer_importance_smollm2_360m.json")


if __name__ == "__main__":
    main()
