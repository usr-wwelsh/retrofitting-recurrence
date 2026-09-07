import json
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def get_edited_model(model_name, extra_args={}):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    is_llama_arch = "llama" in model_name.lower() or "LlamaForCausalLM" in getattr(
        config, "architectures", []
    )
    if is_llama_arch:
        config_args = {
            "model_type": "looped_llama2",
            "auto_map": {"AutoModelForCausalLM": "looped_llama.LoopedLlamaForCausalLM"},
            "architectures": ["LoopedLlamaForCausalLM"],
        }
    else:
        print("model not found")
        exit()

    config.__dict__.update(config_args)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        attn_implementation="eager",
        torch_dtype="bfloat16",
        trust_remote_code=True,
    )
    model.rec_post_init(extra_args, {})
    return model


def force_attn_impl(name):
    if name == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    elif name == "flash":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    else:
        print("attn impl not found")
        exit()
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)


def get_smollm2_huginn_config(smollm2_model_name, mapping_cfg):
    config = AutoConfig.from_pretrained("models/huginn-0125", trust_remote_code=True)
    llama_config = AutoConfig.from_pretrained(smollm2_model_name, trust_remote_code=True)
    if llama_config.tie_word_embeddings:
        print(
            "SmolLM2 has tied embeddings but the recurrent model won't "
            '("tie_embeddings": False) -- lm_head starts as a copy of the '
            "embedding matrix and is free to diverge under training."
        )
    n_prelude = len(mapping_cfg["prelude_idx"])
    n_core = len(mapping_cfg["core_idx"])
    n_coda = len(mapping_cfg["coda_idx"])
    update_dict = {
        "head_dim": int(llama_config.hidden_size / llama_config.num_attention_heads),
        "intermediate_size": llama_config.intermediate_size,
        "n_embd": llama_config.hidden_size,
        "n_heads": llama_config.num_attention_heads,
        "num_key_value_heads": llama_config.num_key_value_heads,
        "n_layers": n_prelude + n_core + n_coda,
        "n_layers_in_coda": n_coda,
        "n_layers_in_prelude": n_prelude,
        "n_layers_in_recurrent_block": n_core,
        "norm_eps": 1e-05,
        "vocab_size": llama_config.vocab_size,
        "padded_vocab_size": llama_config.vocab_size,
        "rope_base": 500000.0,
        "tie_embeddings": False,
        "torch_dtype": llama_config.torch_dtype,
        "qk_bias": False,
        "max_position_embeddings": llama_config.max_position_embeddings,
    }

    for key, value in update_dict.items():
        setattr(config, key, value)

    config.init_values["embed_scale"] = 1.0
    if llama_config.rope_theta:
        config.rope_theta = llama_config.rope_theta
    if llama_config.rope_scaling:
        config.rope_scaling = {
            "factor": llama_config.rope_scaling["factor"],
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }

    return config


def get_looped_smollm2(model_name, looped_args):
    model = get_edited_model(model_name, looped_args)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def weight_mapping(llama_state_dict, huginn_state_dict, mapping_cfg):
    huginn_state_dict["transformer.wte.weight"] = llama_state_dict["model.embed_tokens.weight"]
    huginn_state_dict["lm_head.weight"] = llama_state_dict["lm_head.weight"]
    huginn_state_dict["transformer.ln_f.weight"] = llama_state_dict["model.norm.weight"]

    def copy_layer(src_i, tgt_prefix):
        q_w = llama_state_dict[f"model.layers.{src_i}.self_attn.q_proj.weight"]
        k_w = llama_state_dict[f"model.layers.{src_i}.self_attn.k_proj.weight"]
        v_w = llama_state_dict[f"model.layers.{src_i}.self_attn.v_proj.weight"]

        huginn_state_dict[f"{tgt_prefix}.attn.Wqkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        huginn_state_dict[f"{tgt_prefix}.attn.proj.weight"] = llama_state_dict[
            f"model.layers.{src_i}.self_attn.o_proj.weight"
        ]

        gate_proj = llama_state_dict[f"model.layers.{src_i}.mlp.gate_proj.weight"]
        up_proj = llama_state_dict[f"model.layers.{src_i}.mlp.up_proj.weight"]
        huginn_state_dict[f"{tgt_prefix}.mlp.fc.weight"] = torch.cat([gate_proj, up_proj], dim=0)
        huginn_state_dict[f"{tgt_prefix}.mlp.proj.weight"] = llama_state_dict[
            f"model.layers.{src_i}.mlp.down_proj.weight"
        ]
        huginn_state_dict[f"{tgt_prefix}.norm_1.weight"] = llama_state_dict[
            f"model.layers.{src_i}.input_layernorm.weight"
        ]
        huginn_state_dict[f"{tgt_prefix}.norm_2.weight"] = llama_state_dict[
            f"model.layers.{src_i}.post_attention_layernorm.weight"
        ]

    for j, src_i in enumerate(mapping_cfg["prelude_idx"]):
        copy_layer(src_i, f"transformer.prelude.{j}")
    for j, src_i in enumerate(mapping_cfg["core_idx"]):
        copy_layer(src_i, f"transformer.core_block.{j}")
    for j, src_i in enumerate(mapping_cfg["coda_idx"]):
        copy_layer(src_i, f"transformer.coda.{j}")

    return huginn_state_dict


def get_smollm2_huginn(looped_smollm2_model, config_model_name, save_name, mapping_cfg):
    if save_name is not None and os.path.exists(save_name):
        return AutoModelForCausalLM.from_pretrained(
            save_name, trust_remote_code=True, torch_dtype=torch.bfloat16
        )

    config = get_smollm2_huginn_config(config_model_name, mapping_cfg)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    huginn_state_dict = weight_mapping(
        llama_state_dict=looped_smollm2_model.state_dict(),
        huginn_state_dict=model.state_dict(),
        mapping_cfg=mapping_cfg,
    )
    model.load_state_dict(huginn_state_dict)
    if save_name is not None:
        model.save_pretrained(save_name)
    return model


def check_same(looped_smollm2, smollm2_huginn, tokenizer):
    input_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(input_text, return_tensors="pt").to(smollm2_huginn.device)
    looped_inputs = {k: v.clone() for k, v in inputs.items()}
    huginn_inputs = {k: v.clone() for k, v in inputs.items()}

    with torch.no_grad():
        llama_out = looped_smollm2(**looped_inputs, output_hidden_states=True)
        logits_looped = llama_out.logits

        huginn_out = smollm2_huginn(
            **huginn_inputs,
            output_details={
                "return_logits": True,
                "return_latents": True,
                "return_head": True,
                "return_stats": False,
            },
            num_steps=1,
        )
        logits_huginn = huginn_out.logits

    same_shape = logits_looped.shape == logits_huginn.shape
    close_values = torch.allclose(logits_looped, logits_huginn, atol=1e-4, rtol=1e-4)
    mse = torch.nn.functional.mse_loss(logits_looped, logits_huginn).item()

    print(f"Same shape: {same_shape}")
    print(f"Values close: {close_values}")
    print(f"Mean Squared Error: {mse:.6f}")

    # The production modeling variant returns only the final hidden state
    # (not a per-layer list like the compare variant), so only the final
    # state is comparable to Llama's last hidden_states entry.
    hug_final = huginn_out.hidden_states
    llama_final = llama_out.hidden_states[-1]
    if hug_final.dim() != llama_final.dim():
        hug_final = hug_final.unsqueeze(0)
    close = torch.allclose(hug_final, llama_final, atol=1e-4, rtol=1e-4)
    final_mse = torch.nn.functional.mse_loss(hug_final, llama_final).item()
    print(f"final hidden state: {close}, {final_mse:.3f}")


def main():
    """
    Requires convert_pretrained_model/models/huginn-0125 to be present locally
    (see README step 1) before running -- this only builds the config
    template, no huginn weights are used in the final model.
    """
    force_attn_impl("math")

    with open("../layer_importance_smollm2_360m.json") as f:
        analysis = json.load(f)
    mapping_cfg = analysis["mapping_cfg"]

    smollm2_model_name = "models/SmolLM2-360M"
    n_prelude = len(mapping_cfg["prelude_idx"])
    n_core = len(mapping_cfg["core_idx"])
    n_coda = len(mapping_cfg["coda_idx"])
    save_name = f"models/Recurrent-SmolLM2-360M--{n_prelude}-{n_core}-{n_coda}"

    looped_args = {
        "prelude_idx": mapping_cfg["prelude_idx"],
        "core_idx": mapping_cfg["core_idx"],
        "coda_idx": mapping_cfg["coda_idx"],
        "num_rec": 1,
    }

    looped_smollm2_model, tokenizer = get_looped_smollm2(smollm2_model_name, looped_args)
    smollm2_huginn = get_smollm2_huginn(
        looped_smollm2_model, smollm2_model_name, save_name, mapping_cfg
    )
    total_params = sum(p.numel() for p in smollm2_huginn.parameters())
    print(f"Total params in converted model: {total_params:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    looped_smollm2_model.eval().to(device=device, dtype=torch.float32)
    smollm2_huginn.eval().to(device=device, dtype=torch.float32)

    if device.type == "cpu":
        # force_attn_impl's torch.backends.cuda.enable_*_sdp toggles only
        # govern CUDA dispatch; on CPU with this torch version SDPA needs the
        # modern context manager or no backend is considered viable.
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            check_same(looped_smollm2_model, smollm2_huginn, tokenizer)
    else:
        check_same(looped_smollm2_model, smollm2_huginn, tokenizer)


if __name__ == "__main__":
    main()
