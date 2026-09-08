"""Based on https://github.com/seal-rg/recurrent-pretraining/blob/main/finetuning_simple_example.py"""

####################################################################################################
# Imports.
####################################################################################################

import time

global_start_time = time.monotonic()
import os
import socket
from typing import Any, Optional
from functools import partial
import sys
import datetime
import shutil
import subprocess
import torch
import wandb
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, AutoConfig
from datasets import load_dataset, Dataset, load_from_disk
from contextlib import nullcontext
from stateful_parquet_dataset import get_parquet_dataloader
from dataclasses import dataclass, field
from jsonargparse import CLI
from ellisadam import ELLISAdam

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(True)

# Check device health immediately after loading torch and standard libraries without loading cuda/hip/dist:
nvml_count = torch.cuda.device_count()
if nvml_count < 1:
    raise ValueError(f"Node failure! Device manager init failed on {socket.gethostname()}")

end_time = time.monotonic()
if int(os.getenv("SLURM_PROCID", "0")) == 0:
    print(f"{time.ctime()[:-5]}: Time to load libraries: {end_time - global_start_time:.02f} seconds.")


@dataclass
class CLISettings:
    run_name: str = "default-run"
    out_path: str = "huginn_llama"
    resume_path: Optional[str] = None
    save_n_mins_before_timeout: Optional[int] = None
    # data
    preprocessed_data_path: Optional[str] = None
    dataset_location: str = "openai/gsm8k"
    dataset_args: dict[str, Any] = field(
        default_factory=lambda: dict(q_col="question", a_col="answer")
    )
    dataset_config: str = "main"
    max_length: Optional[int] = None
    max_samples: Optional[int] = None
    # impl
    micro_batch_size: int = 2
    compile: bool = False
    # training
    max_steps: int = 0
    epochs: int = 1
    batch_size: int = 32
    optim_config: dict[str, Any] = field(
        default_factory=lambda: dict(lr=5e-7, weight_decay=1e-4, betas=(0.9, 0.95), eps=1e-8)
    )
    scheduler_args: dict[float, Any] = field(
        default_factory=lambda: dict(warmup=0.1, cooldown=0.1, min_lr_ratio=0.001)
    ) # min_lr = min_lr_ratio * lr
    save_interval: int = -1
    hub_checkpoint_repo: Optional[str] = None # if set, push each full resumable checkpoint here (HF dataset repo) and keep only the latest, instead of accumulating checkpoints on local disk
    model_name: str = "smcleish/Recurrent-TinyLlama-3T-untrained"
    wandb_disabled: bool = False
    wandb_project: str = "huginn"
    seed: int = 74
    fix_num_steps: bool = False
    init_from_scratch: bool = False
    take_loss_over_all_tokens: bool = False # for chat templated datasets default is to only supervise assistant tokens
    max_grad_norm: float = 1.0
    bf16_true: bool = False
    compile_warmup_routine: bool = False
    no_amp: bool = True
    is_parquet_dataset: bool = False
    ignore_past_parquet_dataset: bool = False
    parquet_dataset_max_tokens: Optional[int] = None
    ignore_past_scheduler: bool = False
    mean_recurrence_schedule: dict[float, Any] = field(
        default_factory=lambda: dict(turn_on=False, warmup=0.1, max_mean_rec=32, warmup_type="linear")
    )
    mean_backprop_depth_schedule: dict[float, Any] = field(
        default_factory=lambda: dict(turn_on=False, warmup=0.1, max_backprop=8, start=1)
    )
    no_monkeypatch_on_jonas_init: bool = False
    throttle: bool = False
    non_recurrent_model: bool = False
    muon: dict[float, Any] = field(
        default_factory=lambda: dict(use_muon=False, lr=0.005, weight_decay=1e-4)
    )
    use_ellis_adam: dict[float, Any] = field(
        default_factory=lambda: dict(use_ellis_adam=False, decouple_wd=True, tensor_wise_gradient_normalization=False, tensor_wise_finite_check=False, running_init=True, atan_adam=True, update_clipping=True,)
    )
    parquet_epoching_flag_use_with_real_caution: int = 1

    def __post_init__(self):
        assert self.micro_batch_size <= self.batch_size, "batch size must be less than micro batch size"

        self.amp_args = {"device_type": "cuda", "dtype": torch.bfloat16}
        if self.no_amp:
            # https://github.com/Lightning-AI/pytorch-lightning/pull/20921
            # https://github.com/pytorch/pytorch/issues/65766
            self.amp_args["enabled"] = False
            self.amp_args["cache_enabled"] = False
        else:
            # i.e. we haven't turned amp off
            self.amp_args["enabled"] = True
            self.amp_args["cache_enabled"] = self.compile and (not self.bf16_true) # can only use cache if compiled and in float32

        assert self.batch_size % self.micro_batch_size == 0, "grad accum steps must be an int"
        if self.is_parquet_dataset:
            assert (self.parquet_dataset_max_tokens is not None) or (self.max_steps != 0), "if using parquet need to specify max tokens or max steps"
            assert self.max_length is not None, "if using parquet need to specify max_length of context"

        if self.non_recurrent_model:
            assert not self.throttle, "Can't use throttle with non_recurrent_model"
            assert not self.mean_backprop_depth_schedule["turn_on"], "Can't use mean_backprop_depth_schedule with non_recurrent_model"
            assert not self.mean_recurrence_schedule["turn_on"], "Can't use mean_recurrence_schedule with non_recurrent_model"
            assert not self.compile_warmup_routine, "Can't use compile_warmup_routine with non_recurrent_model"

            self.no_monkeypatch_on_jonas_init = True # turn off for normal models

@dataclass
class Message:
    role: str
    content: str

def get_flux_timeleft():
    result = subprocess.run(
        ["flux", "job", "timeleft"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True
    )
    return int(result.stdout.strip())

has_completed_timeout_save = False
def check_if_save(save_n_mins_before_timeout):
    global has_completed_timeout_save
    if (save_n_mins_before_timeout * 60 > get_flux_timeleft()) and (not has_completed_timeout_save):
        has_completed_timeout_save = True
        return True
    return False

def save_model_only(cfg, state, chkpt_name):
    unwrapped_model = get_unwrapped_model(state)
    unwrapped_model.save_pretrained(f"{cfg.out_path}/{cfg.run_name}/{chkpt_name}")
    state["tokenizer"].save_pretrained(f"{cfg.out_path}/{cfg.run_name}/{chkpt_name}")

    # keep only the most recent eval checkpoint on disk -- these are full fp32
    # weights and accumulate fast otherwise
    run_dir = f"{cfg.out_path}/{cfg.run_name}"
    for name in os.listdir(run_dir):
        if name.startswith("model_only_chkpt_") and name != chkpt_name:
            shutil.rmtree(f"{run_dir}/{name}", ignore_errors=True)

def save_checkpoint(state, agg_vars_dict, cfg):
    # agg_vars_dict = {"data_start_step": data_start_step, "optimizer_step": optimizer_step, "total_tokens": total_tokens, "total_tokens_with_loss": total_tokens_with_loss}
    step = agg_vars_dict["optimizer_step"]
    if cfg.is_parquet_dataset:
        # have to call this on all nodes as there is an internal gather
        dataloader_state = state["dataloader"].state_dict()
    else:
        dataloader_state = None
    
    if cfg.muon["use_muon"]:
        # muon does an all gather on saving
        optim_state_dict = state["optimizer"].state_dict()
    elif is_main_process():
        optim_state_dict = state["optimizer"].state_dict()

    if not is_main_process():
        return

    extras = {}
    if cfg.mean_recurrence_schedule["turn_on"]:
        extras["mean_recurrence_scheduler"] = state["mean_recurrence_scheduler"].state_dict()
    if cfg.mean_backprop_depth_schedule["turn_on"]:
        extras["mean_backprop_depth_scheduler"] = state["mean_backprop_depth_scheduler"].state_dict()

    unwrap = get_unwrapped_model(state)
    ckpt = dict(
        model=unwrap.state_dict(),
        optimizer=optim_state_dict,
        scheduler=state["scheduler"].state_dict(),
        dataloader=dataloader_state,
        rng_state=torch.get_rng_state(),
        cuda_rng_state=torch.cuda.get_rng_state_all(),
        agg_vars_dict=agg_vars_dict,
        cfg=cfg.__dict__, # for provenance
        **extras,
    )

    chkpt_dir = f"{cfg.out_path}/{cfg.run_name}/checkpoint_{step}"
    os.makedirs(chkpt_dir, exist_ok=True)
    torch.save(ckpt, f"{chkpt_dir}/chkpt.pt")
    print(f"[rank 0] Saved checkpoint @ step {step:,}")

    if cfg.hub_checkpoint_repo is not None:
        push_checkpoint_to_hub(cfg.hub_checkpoint_repo, cfg.run_name, chkpt_dir, step)

    # keep only the most recent full checkpoint on disk -- these include optimizer
    # state (~2x model size) and a full resume only ever needs the latest one
    run_dir = f"{cfg.out_path}/{cfg.run_name}"
    for name in os.listdir(run_dir):
        if name.startswith("checkpoint_") and name != f"checkpoint_{step}":
            shutil.rmtree(f"{run_dir}/{name}", ignore_errors=True)

def push_checkpoint_to_hub(repo_id, run_name, chkpt_dir, step):
    # namespaced by run_name so unrelated runs (e.g. a smoke test and the
    # real run) sharing one repo can never resume from each other's state
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=f"{chkpt_dir}/chkpt.pt",
        path_in_repo=f"{run_name}/chkpt.pt",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"{run_name} checkpoint @ step {step}",
    )
    print(f"[rank 0] Pushed checkpoint @ step {step:,} to {repo_id}/{run_name} (overwriting previous)")

def load_checkpoint(state, cfg, device):
    ckpt = torch.load(f"{cfg.resume_path}/chkpt.pt", map_location=device)
    unwrap = get_unwrapped_model(state)
    unwrap.load_state_dict(ckpt["model"], strict=True)
    state["optimizer"].load_state_dict(ckpt["optimizer"])

    if cfg.mean_recurrence_schedule["turn_on"] and ("mean_recurrence_scheduler" in ckpt):
        state["mean_recurrence_scheduler"].load_state_dict(ckpt["mean_recurrence_scheduler"])
    if cfg.mean_backprop_depth_schedule["turn_on"] and ("mean_backprop_depth_scheduler" in ckpt):
        state["mean_backprop_depth_scheduler"].load_state_dict(ckpt["mean_backprop_depth_scheduler"])

    if not cfg.ignore_past_scheduler:
        state["scheduler"].load_state_dict(ckpt["scheduler"])
    if cfg.is_parquet_dataset and not cfg.ignore_past_parquet_dataset:
        state["dataloader"].load_state_dict(ckpt["dataloader"])

    torch.set_rng_state(ckpt["rng_state"].to("cpu"))
    torch.cuda.set_rng_state_all([rng.to("cpu") for rng in ckpt["cuda_rng_state"]])
    print(f"Resumed from {cfg.resume_path}")
    return ckpt["agg_vars_dict"]

def is_main_process():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    else:
        return True

def seed_everything(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.manual_seed(seed)

def get_unwrapped_model(state):
    if isinstance(state, dict):
        return state["model"].module if state["distributed"] else state["model"]
    else:
        return state.module if torch.distributed.is_initialized() else state


####################################################################################################
# Main driver functions.
####################################################################################################
# DEFAULT_SYS_PROMPT = "You are a helpful assistant that can help users with mathematical reasoning."
DEFAULT_SYS_PROMPT = "You are a helpful assistant that can assist users with mathematical reasoning."

def initialize_state_monkeypatch(self, input_embeds, scale: float = 1.0, patched_std: float = 0.008703882797784892, patched_embed_scale: float = 1.0):
    """
    Patch to fixes the std to the Huginn value and remove the embed scaling
    """
    x = torch.randn_like(input_embeds)
    std = patched_std * scale
    if std > 0:
        torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if patched_embed_scale != 1:
            x = x * self.emb_scale
    else:
        x.zero_()
    return x


def startup(cfg: CLISettings):
    """The main setup function for the training script."""
    seed_everything(cfg.seed)
    ##########    Comms              ##############
    rank = int(os.getenv("SLURM_PROCID", os.getenv("RANK", "0")))
    local_device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        distributed = True
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=int(os.getenv("SLURM_NTASKS", os.getenv("WORLD_SIZE", -1))),
            device_id=local_device,  # this immediately forms the NCCL communicator, crucial based on Sean's testing
            timeout=datetime.timedelta(hours=0.5 if cfg.is_parquet_dataset else 2), # 2hrs should be good to process for ~20M samples-ish
        )
        world_size = torch.distributed.get_world_size()
        print(f"Comms formed on rank {rank} with device {local_device} out of world size {world_size}.")
    else:
        world_size = 1
        distributed = False

    weight_dtype = torch.float32
    if cfg.bf16_true:
        torch.set_default_dtype(torch.bfloat16)
        weight_dtype = torch.bfloat16
    torch.cuda.set_device(local_device)

    ########## Model and tokenizer ##############
    config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
    if cfg.init_from_scratch:
        # https://huggingface.co/smcleish/Recurrent-Llama-3.2-2-4-2-untrained/blob/main/raven_modeling_minimal_with_init.py
        if cfg.non_recurrent_model:
            pass
        else:
            config.auto_map["AutoModelForCausalLM"] = "raven_modeling_minimal_with_init.RavenForCausalLM"
            # Redirect to a different modelling file as for Llama we need to hardcode emb_scale=1.0, which we do in the regular modelling file
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        if not cfg.no_monkeypatch_on_jonas_init:
            from types import MethodType
            model.initialize_state = MethodType(initialize_state_monkeypatch, model)

        model.to(device=local_device, dtype=weight_dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=local_device,
            torch_dtype=weight_dtype,
            attn_implementation="sdpa",
            config=config,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ##########  Distribute model   ##############
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_device], find_unused_parameters=not cfg.compile, gradient_as_bucket_view=True)
    if cfg.compile:
        model = torch.compile(model, fullgraph=False, dynamic=False, mode="max-autotune-no-cudagraphs")
    ##########     Optimizer       ##############
    if cfg.use_ellis_adam["use_ellis_adam"]:
        optimizer = ELLISAdam(
            params=model.parameters(),
            **{k: v for k, v in cfg.optim_config.items() if k != "eps"},
            **{k: v for k, v in cfg.use_ellis_adam.items() if k != "use_ellis_adam"},
        )

    elif cfg.muon["use_muon"]:
        from muon import MuonWithAuxAdam

        body_params = []
        non_body_params = []
        norms = []

        if cfg.non_recurrent_model:
            if ("TinyLlama-1.1B-intermediate-step-1431k-3T" in cfg.model_name) or ("Llama-3.2-1B" in cfg.model_name) or ("OLMo-2" in cfg.model_name):
                for n, p in model.named_parameters():
                    if ("norm" in n) or ("bias" in n):
                        norms.append(p)
                    elif ("embed_tokens" in n) or ("lm_head" in n):
                        non_body_params.append(p)
                    else:
                        body_params.append((n,p))
            else:
                for n, p in model.named_parameters():
                    if ("norm" in n) or ("bias" in n):
                        norms.append(n)
                    elif ("embed_tokens" in n) or ("lm_head" in n):
                        non_body_params.append(n)
                    else:
                        body_params.append(n)
                if is_main_process():
                    print(model)
                    print("="*70)
                    print(norms)
                    print("="*70)
                    print(non_body_params)
                    print("="*70)
                    print(body_params)
                assert False, "Model not allowed for muon"
        else:
            # if a huginn
            for n, p in model.named_parameters():
                if ("norm" in n) or ("ln_f" in n) or ("Wqkv.bias" in n):
                    norms.append(p)
                elif ("wte" in n) or ("lm_head" in n):
                    non_body_params.append(p)
                else:
                    body_params.append((n,p))

        # body_params = sorted(body_params, key=lambda x: x.size(), reverse=True)
        # Took sorting out of the init so that it is deterministic
        body_params.sort(key=lambda np: (-np[1].numel(), tuple(np[1].shape), np[0]))
        body_params = [p for _, p in body_params]
        param_groups = [
            dict(params=body_params, use_muon=True, lr=cfg.muon["lr"], weight_decay=cfg.muon["weight_decay"], no_sorting_in_init=False),
            dict(params=non_body_params + norms, use_muon=False, lr=cfg.optim_config["lr"], betas=cfg.optim_config["betas"], weight_decay=cfg.optim_config["weight_decay"]),
        ]
        optimizer = MuonWithAuxAdam(param_groups)

        ## Need to save all states on all ranks, see: https://github.com/KellerJordan/Muon/issues/46
        def gather(self):
            if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
                return
            world = torch.distributed.get_world_size()

            for group in self.param_groups:
                if not group["use_muon"]:
                    continue

                params = group["params"]
                # Make sure every rank has a buffer tensor to receive the broadcast.
                for p in params:
                    st = self.state[p]
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = torch.zeros_like(p)

                # For param i, the canonical copy lives on rank (i % world).
                for i, p in enumerate(params):
                    src = i % world
                    torch.distributed.broadcast(self.state[p]["momentum_buffer"], src=src)

        optimizer.register_state_dict_pre_hook(gather)
    else:
        # print(model.named_parameters())
        if cfg.throttle:
            recur_params = []
            non_recur_params = []
            for n, p in model.named_parameters():
                if ("adapter" in n) or ("core_block" in n):
                    recur_params.append(p)
                else:
                    non_recur_params.append(p)
            params = [
                {"params": recur_params,  "lr": cfg.optim_config["lr"]},
                {"params": non_recur_params, "lr": cfg.optim_config["lr"]},
            ]
            optim_config = cfg.optim_config.copy()
            optim_config.pop("lr")
        else:
            params = model.parameters()
            optim_config = cfg.optim_config.copy()
        optimizer = torch.optim.AdamW(params, **optim_config)

    ##########     Data            ##############
    def format_and_tokenize_examples(examples):
        conversations = []
        for idx in range(len(examples[cfg.dataset_args["q_col"]])):
            if cfg.dataset_args["q_col"] != "text":
                messages = [
                    Message(role="system", content=DEFAULT_SYS_PROMPT),
                    Message(role="user", content=examples[cfg.dataset_args["q_col"]][idx].strip()),
                    Message(role="Huginn", content=examples[cfg.dataset_args["a_col"]][idx].strip()),
                ]
            else:
                messages = tokenizer.bos_token + examples[cfg.dataset_args["q_col"]][idx].strip()
            conversations.append(messages)
        
        if cfg.dataset_args["q_col"] != "text":
            chat_encoding = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
                padding="max_length",
                max_length=cfg.max_length + 1,
                return_tensors="pt",
                return_dict=True,
                truncation=True,
            )
            if cfg.take_loss_over_all_tokens:
                chat_encoding["assistant_masks"] = chat_encoding["attention_mask"]
        else:
            chat_encoding = tokenizer(
                conversations,
                padding="max_length",
                max_length=cfg.max_length + 1,
                return_tensors="pt",
                truncation=True,
                
            )
            chat_encoding["assistant_masks"] = chat_encoding["attention_mask"].clone()

        return {
            "token_ids": chat_encoding["input_ids"],
            "mask": chat_encoding["assistant_masks"],
            "attention_mask": chat_encoding["attention_mask"],
        }

    if cfg.preprocessed_data_path is None:
        cfg.token_id_col_name = "token_ids"
        dataset_save_dir = f"{cfg.out_path}/{cfg.run_name}/dataset"
        if is_main_process(): # only load to rank 0 to begin
            try:
                dataset: Dataset = load_dataset(cfg.dataset_location, cfg.dataset_config)["train"]  # type: ignore
            except:
                dataset: Dataset = load_from_disk(cfg.dataset_location, cfg.dataset_config)  # type: ignore

            if cfg.max_samples is not None:
                dataset = dataset.select(range(cfg.max_samples))

            if os.path.exists(dataset_save_dir): # delete any old dataset
                shutil.rmtree(dataset_save_dir)

            tokenized_dataset = dataset.map(
                format_and_tokenize_examples,
                num_proc=16,
                remove_columns=dataset.column_names,
                batched=True,
                batch_size=1024,
            )

        if distributed: # load the dataset to other ranks
            if is_main_process():
                tokenized_dataset.save_to_disk(dataset_save_dir)
            torch.distributed.barrier()
            tokenized_dataset = load_from_disk(dataset_save_dir)
            torch.distributed.barrier()
    else:
        cfg.token_id_col_name = "input_ids"
        if cfg.is_parquet_dataset:
            assert cfg.max_samples is None, "cannot have max samples for parquet dataset"
            tokenized_dataset = get_parquet_dataloader(world_size, rank, cfg.micro_batch_size, cfg.preprocessed_data_path, num_epochs=cfg.parquet_epoching_flag_use_with_real_caution)
        else:
            tokenized_dataset = load_from_disk(cfg.preprocessed_data_path)
            if cfg.max_samples is not None:
                dataset = dataset.select(range(cfg.max_samples))

    if not cfg.is_parquet_dataset:
        tokenized_dataset.set_format("pt")

    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(cfg.seed)
    if cfg.is_parquet_dataset:
        dataloader = tokenized_dataset
    elif distributed:
        sampler = torch.utils.data.DistributedSampler(
            tokenized_dataset,
            shuffle=not cfg.is_parquet_dataset,
            num_replicas=world_size,
            rank=rank,
            seed=cfg.seed,
        )
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,
            batch_size=cfg.micro_batch_size,
            sampler=sampler,
            pin_memory=True,
            generator=dataloader_generator,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,  # type: ignore
            batch_size=cfg.micro_batch_size,
            shuffle=not cfg.is_parquet_dataset,
            pin_memory=True,
            generator=dataloader_generator,
        )

    ##########     Scheduler       ##############
    if cfg.is_parquet_dataset:
        if cfg.max_steps:
            max_training_steps = cfg.max_steps
        else:
            max_training_steps = max(1, math.ceil(cfg.parquet_dataset_max_tokens / world_size / cfg.max_length))
        num_warmup_steps = math.ceil(cfg.scheduler_args["warmup"] * max_training_steps)
        num_decay_steps = math.ceil(cfg.scheduler_args["cooldown"] * max_training_steps)
    else:
        if cfg.max_steps:
            max_training_steps = cfg.max_steps
        else:
            accumulation_steps = max(1, cfg.batch_size // cfg.micro_batch_size)
            num_update_steps_per_epoch = math.ceil(len(dataloader) / accumulation_steps)
            max_training_steps = cfg.epochs * num_update_steps_per_epoch
        num_warmup_steps = math.ceil(cfg.scheduler_args["warmup"] * max_training_steps)
        num_decay_steps = math.ceil(cfg.scheduler_args["cooldown"] * max_training_steps)

    scheduler = get_scheduler(
        name="warmup_stable_decay",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_training_steps,
        scheduler_specific_kwargs={"num_decay_steps":num_decay_steps, "min_lr_ratio": cfg.scheduler_args["min_lr_ratio"]},
    )

    state = {
        "model": model,
        "optimizer": optimizer,
        "tokenizer": tokenizer,
        "dataloader": dataloader,
        "distributed": distributed,
        "scheduler": scheduler,
    }

    if cfg.mean_recurrence_schedule["turn_on"]:
        # make a dummy optimizer of one param 
        num_warmup_steps = math.ceil(cfg.mean_recurrence_schedule["warmup"] * max_training_steps)
        mean_recurrence_scheduler = get_scheduler(
            name="warmup_stable_decay",
            optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=float(cfg.mean_recurrence_schedule["max_mean_rec"])),
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_training_steps,
            scheduler_specific_kwargs={"num_decay_steps":0, "min_lr_ratio":0, "warmup_type": cfg.mean_recurrence_schedule["warmup_type"]},
        )
        state["mean_recurrence_scheduler"] = mean_recurrence_scheduler
    
    if cfg.mean_backprop_depth_schedule["turn_on"]:
        # make a dummy optimizer of one param 
        num_warmup_steps = math.ceil(cfg.mean_backprop_depth_schedule["warmup"] * max_training_steps)

        max_depth = cfg.mean_backprop_depth_schedule["max_backprop"]
        start = max(1.0, cfg.mean_backprop_depth_schedule["start"] - 1) # start at one below so we get the right value out of the scheduler after the first step
        min_lr_ratio = max(0.0, min(1.0, start / max_depth))

        mean_backprop_depth_scheduler = get_scheduler(
            name="warmup_stable_decay",
            optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=float(max_depth)),
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_training_steps,
            scheduler_specific_kwargs={"num_decay_steps":0, "min_lr_ratio":min_lr_ratio},
        )
        state["mean_backprop_depth_scheduler"] = mean_backprop_depth_scheduler
        state["mean_backprop_depth_scheduler"].step() # take the first step so we get 2 out of the scheduler and not 1

    cfg.world_size = world_size
    if is_main_process():
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.run_name,
            config=cfg,
            dir=cfg.out_path,
            mode="disabled" if cfg.wandb_disabled else "online",
        )

    return state, local_device


def distributed_and_agg_metrics(metrics_to_agg_data_step, metrics_to_agg_optim_step):
    keys_to_mean = ["loss", "log_ppl"]

    distributed = torch.distributed.is_initialized()
    rank = int(os.getenv("SLURM_PROCID", os.getenv("RANK", "0")))
    local_device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    def _sync(value: float, op=torch.distributed.ReduceOp.SUM) -> float:
        """Synchronise a scalar across ranks and return the reduced result."""
        if distributed:
            tensor = torch.tensor(value, dtype=torch.float64, device=local_device)
            torch.distributed.all_reduce(tensor, op=op)
            return tensor.item()
        return value
    

    aggregated = {}
    # metrics_to_agg_data_step
    for key, local_list in metrics_to_agg_data_step.items():
        if not local_list:
            continue

        local_sum = float(sum(local_list))
        local_count = float(len(local_list))

        global_sum = _sync(local_sum)
        global_count = _sync(local_count)

        aggregated[key] = global_sum / (max(global_count, 1.0) if key in keys_to_mean else 1.0)

        local_list.clear()

    # metrics_to_agg_optim_step
    for key, val in metrics_to_agg_optim_step.items():
        if key in keys_to_mean:
            # we don't pass this anymore as it is global anyway but is example of how to use avg
            aggregated[key] = _sync(val, op=torch.distributed.ReduceOp.AVG)
        else:
            aggregated[key] = _sync(val)

    return aggregated

def get_steps_compiling(data_step, device):
    if data_step > 600:
        exit()
    n = data_step % 300
    k =  min(8, n)
    print(f"Warming up sampling step={data_step}, n={n}, k={k}")
    return  torch.tensor([n,k], device=device)

def num_steps_sampler(data_step, mean_recurrence, mean_backprop_depth, cfg):
    """
    Sampling num steps in a checkpointable way
    https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/model_dynamic.py#L1250
    """
    t = max(mean_recurrence - mean_backprop_depth, 0)
    s = mean_backprop_depth
    
    seed_n = 514229 + data_step 
    seed_k = 317811 + data_step   

    n_generator = torch.Generator(device="cpu")
    n_generator.manual_seed(seed_n % (2**31 - 1))
    k_generator = torch.Generator(device="cpu")
    k_generator.manual_seed(seed_k % (2**31 - 1))

    sigma = 0.5
    mu = math.log(t + s) - (sigma**2 / 2)
    rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma, generator=n_generator)
    p = torch.poisson(torch.tensor([rate], dtype=torch.float), generator=n_generator) + 1
    n = torch.clamp(p - s, min=0)
    k = torch.as_tensor(torch.minimum(torch.as_tensor(s), p))

    return n.to(dtype=torch.long), k.to(dtype=torch.long)

def sheduler_n_k_handler(state, cfg, model_config):
    if cfg.mean_recurrence_schedule["turn_on"]:
        new_mean_rec = math.ceil(state["mean_recurrence_scheduler"].get_last_lr()[0])
    else:
        new_mean_rec = model_config.mean_recurrence

    if cfg.mean_backprop_depth_schedule["turn_on"]:
        mean_backprop_depth = math.ceil(state["mean_backprop_depth_scheduler"].get_last_lr()[0])
    else:
        mean_backprop_depth = model_config.mean_backprop_depth

    if new_mean_rec <= 0:
        # schedule starts at 0
        new_mean_rec = 1

    if (new_mean_rec - mean_backprop_depth) < 0:
        # t = max(mean_recurrence - mean_backprop_depth, 0) messes up the schedule so we catch that here
        return partial(num_steps_sampler, mean_recurrence=new_mean_rec, mean_backprop_depth=new_mean_rec, cfg=cfg), new_mean_rec, new_mean_rec
    else:
        return partial(num_steps_sampler, mean_recurrence=new_mean_rec, mean_backprop_depth=mean_backprop_depth, cfg=cfg), new_mean_rec, mean_backprop_depth

def train(state, device, cfg, data_start_step=1, optimizer_step=0, total_tokens_from_restart=0, total_tokens_with_loss_from_restart=0, elapsed_time_from_restart=0.0):
    model, optimizer = state["model"], state["optimizer"]
    model.train()

    accumulation_steps = cfg.batch_size // cfg.micro_batch_size
    optimizer_step = optimizer_step
    step_time = time.monotonic()
    total_tokens = 0
    total_tokens_with_loss = 0
    tokens_in_step = 0
    k_mean_tracker = [0,0]
    elapsed_time = 0.0

    output_details = {
        "return_logits": False,
        "return_latents": False,
        "return_head": False,
        "return_stats": True,
    }

    metrics_to_agg_data_step = {
        "loss": [],
        "log_ppl": [],
    }
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    model_config = get_unwrapped_model(model).config
    if cfg.mean_recurrence_schedule["turn_on"] or cfg.mean_backprop_depth_schedule["turn_on"]:
        num_steps_sampler_partial, new_mean_rec, new_backprop_depth = sheduler_n_k_handler(state, cfg, model_config)
    elif cfg.non_recurrent_model:
        new_mean_rec, new_backprop_depth = model_config.num_hidden_layers, model_config.num_hidden_layers
    else:
        new_mean_rec = model_config.mean_recurrence
        new_backprop_depth = model_config.mean_backprop_depth
        num_steps_sampler_partial = partial(num_steps_sampler, mean_recurrence=new_mean_rec, mean_backprop_depth=new_backprop_depth, cfg=cfg)

    for epoch in range(cfg.epochs):
        for data_step, inputs in enumerate(state["dataloader"], start=(data_start_step + 1) if cfg.is_parquet_dataset else 1):
            if (data_start_step != 1) and (not cfg.is_parquet_dataset) and (data_step <= data_start_step):
                # not first_run and not parquet_run and is less than the restart
                continue

            # Realize the input and labels tensors.
            input_ids = inputs[cfg.token_id_col_name][:, :-1].to(dtype=torch.long, device=device, non_blocking=True)
            # Need to take into account the assistant and attention if sequences are being padded
            if cfg.preprocessed_data_path is None:
                mask = ~(inputs["mask"].bool() & inputs["attention_mask"].bool())
            else:
                mask = ~inputs["attention_mask"].bool()

            labels = torch.where(mask[:, 1:], -100, inputs[cfg.token_id_col_name][:, 1:]).to(
                dtype=torch.long, device=device, non_blocking=True
            )
            total_tokens_with_loss += (labels != -100).sum().item()

            tokens_in_step += input_ids.numel()
            is_accumulating = (data_step % accumulation_steps != 0)
 
            if cfg.fix_num_steps:
                num_steps = torch.tensor([0,1], device=model.device)
            elif cfg.compile_warmup_routine:
                num_steps = get_steps_compiling(data_step, model.device)
            elif not cfg.non_recurrent_model:
                num_steps = num_steps_sampler_partial(data_step)

            if cfg.throttle:
                k_mean_tracker[0] += num_steps[1]
                k_mean_tracker[1] += 1

            # The actual compute step of  Forward, loss, and backward computation:
            def tightly_scoped_fwd_bwd(model, input_ids, labels):
                with model.no_sync() if is_accumulating and state["distributed"] else nullcontext():
                    with torch.autocast(**cfg.amp_args):
                        outputs = model(input_ids, labels=labels, num_steps=num_steps, output_details=output_details)

                    (outputs["loss"] / accumulation_steps).backward()
                    return outputs["loss"].detach(), outputs["log_ppl"].detach(), outputs["stats"]["num_steps_no_grad"], outputs["stats"]["num_steps_with_grad"]
            
            def non_rec_fwd_bwd(model, input_ids, labels):
                with model.no_sync() if is_accumulating and state["distributed"] else nullcontext():
                    with torch.autocast(**cfg.amp_args):
                        logits = model(input_ids).logits

                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100
                    ) # copied from Huginn code to be sure

                    (loss / accumulation_steps).backward()
                    log_ppl = loss.clone().detach().exp()
                    return loss.detach(), log_ppl, model_config.num_hidden_layers, model_config.num_hidden_layers

            fwd_bwd_func = non_rec_fwd_bwd if cfg.non_recurrent_model else tightly_scoped_fwd_bwd
            loss, log_ppl, num_steps_no_grad, num_steps_with_grad = fwd_bwd_func(model, input_ids, labels)

            # logging
            metrics_to_agg_data_step["loss"].append(loss.item())
            metrics_to_agg_data_step["log_ppl"].append(log_ppl.item())

            if not is_accumulating:
                if cfg.throttle:
                    # NOTE: this is only okay to do as k is the same at each step on all ranks
                    # this will break if k is not the same on all ranks at all steps

                    g = optimizer.param_groups[0] # recur params first, then non recur when initing optim
                    denom = max(1, int(k_mean_tracker[0] / k_mean_tracker[1])) # mean k for this batch
                    g["lr"] = g["lr"] / denom
                    k_mean_tracker  = [0, 0]

                    lrs = [pg["lr"] for pg in optimizer.param_groups]
                    wandb_lr_log  = {"train/lr_recur": lrs[0], "train/lr_nonrecur": lrs[1]}
                else:
                    lrs = [pg["lr"] for pg in optimizer.param_groups]
                    wandb_lr_log  = {"train/lr_recur": lrs[0], "train/lr_nonrecur": lrs[0]}


                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=cfg.max_grad_norm,
                    norm_type=2.0,
                ).item()
                grad_clip_coef = min(1.0, float(cfg.max_grad_norm) / (total_norm + 1e-12))
                optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                state["scheduler"].step()
                optimizer_step += 1

                if cfg.mean_recurrence_schedule["turn_on"] or cfg.mean_backprop_depth_schedule["turn_on"]:
                    if cfg.mean_recurrence_schedule["turn_on"]:
                        state["mean_recurrence_scheduler"].step()
                    if cfg.mean_backprop_depth_schedule["turn_on"]:
                        state["mean_backprop_depth_scheduler"].step()
                    num_steps_sampler_partial, new_mean_rec, new_backprop_depth = sheduler_n_k_handler(state, cfg, model_config)

            if not is_accumulating:
                time_taken = (time.monotonic() - step_time)
                time_interval = time_taken / accumulation_steps
                tok_sec = tokens_in_step / time_taken
                elapsed_time += time_taken
                print(
                    f"GPU: {model.device} | Step: {data_step:4d} | Updates: {optimizer_step:4d} | Time/step: {time_interval:2.4f}"
                    f" | Tok/sec={tok_sec:9.2f} | Loss: {loss:2.4f} / log-ppl: {log_ppl:2.4f} | Grad-Norm {total_norm:2.4f} | ClipCoef {grad_clip_coef:1.4f}"
                )
                total_tokens += tokens_in_step
                step_time = time.monotonic()
                tokens_in_step = 0

                agg_metrics = distributed_and_agg_metrics(metrics_to_agg_data_step, {"total_tokens_with_loss": total_tokens_with_loss, "total_tokens": total_tokens, "tokens_per_second": tok_sec})
                total_tokens_to_log = total_tokens_from_restart + agg_metrics.pop("total_tokens")
                total_tokens_with_loss_to_log = total_tokens_with_loss_from_restart + agg_metrics.pop("total_tokens_with_loss")
                elapsed_time_to_log = elapsed_time_from_restart + elapsed_time

                if is_main_process():
                    wandb.log({
                        "train/step": optimizer_step,
                        "train/epoch": epoch,
                        "train/lr": state["scheduler"].get_last_lr()[1 if cfg.throttle else 0],
                        "train/total_tokens": total_tokens_to_log,
                        "train/total_tokens_with_loss": total_tokens_with_loss_to_log,
                        "train/total_tokens_no_loss": total_tokens_to_log - total_tokens_with_loss_to_log,
                        "train/total_samples": data_step * cfg.micro_batch_size * world_size,
                        "train/num_steps_no_grad": num_steps_no_grad,
                        "train/num_steps_with_grad": num_steps_with_grad,
                        "train/total_norm": total_norm,
                        "train/grad_clip_coef": grad_clip_coef,
                        "train/grad_clip_max_norm": cfg.max_grad_norm,
                        "train/mean_recurrence": new_mean_rec,
                        "train/mean_backprop_depth": new_backprop_depth,
                        "train/elapsed_time": elapsed_time_to_log,
                        **{f"train/{k}": v for k,v in agg_metrics.items()},
                        **wandb_lr_log,
                    }, step=optimizer_step)

                    if (cfg.save_interval != -1) and (optimizer_step % cfg.save_interval == 0):
                        save_model_only(cfg, state, f"model_only_chkpt_{optimizer_step}")

                if (cfg.save_interval != -1) and (optimizer_step % (2 * cfg.save_interval) == 0):
                    # have to call save_checkpoint on all ranks for the dataloader
                    save_checkpoint(state, {"data_start_step": data_step, "optimizer_step": optimizer_step, "total_tokens": total_tokens_to_log, "total_tokens_with_loss": total_tokens_with_loss_to_log, "elapsed_time": elapsed_time_to_log}, cfg)

                if cfg.save_n_mins_before_timeout is not None:
                    if check_if_save(cfg.save_n_mins_before_timeout):
                        save_checkpoint(state, {"data_start_step": data_step, "optimizer_step": optimizer_step, "total_tokens": total_tokens_to_log, "total_tokens_with_loss": total_tokens_with_loss_to_log, "elapsed_time": elapsed_time_to_log}, cfg)
                        if torch.distributed.is_initialized():
                            torch.distributed.barrier()

            if cfg.max_steps and optimizer_step >= cfg.max_steps:
                break

    model.eval()
    return state


def main():
    """Encapsulates main scope away from import calls."""

    # Configuration loader
    cfg: CLISettings = CLI(CLISettings)

    # Print system setup
    if is_main_process():
        print("--------------------------------------------------------------------")
        print(f"------------------ Launching run {cfg.run_name}------------------")
        print("--------------------------------------------------------------------")
        print("--------------------------------------------------------------------")
        print(f"Platform: {sys.platform}, Python: {sys.version.split(' (')[0]}, PyTorch: {torch.__version__}")
        print(f"CPU threads: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.")
        driver = f"HIP/ROCM {torch.version.hip}" if torch.version.hip else f"CUDA: {torch.version.cuda}"
        print(f"GPU : {torch.cuda.get_device_name()}. {driver}.")

    # set flags
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Should be true anyway
    torch._dynamo.config.optimize_ddp = "python_reducer"
    # have to use the below two together as we do error if we compile the gradient states the no_grad/grad step
    torch._dynamo.config.compiled_autograd = False # didn't work for Jonas ever...
    # torch._dynamo.config.error_on_recompile = True # Here's to hoping

    train_time = time.monotonic()
    # Set up dist and load model and tokenizer into state
    state, device = startup(cfg)
    data_start_step, optimizer_step, total_tokens, total_tokens_with_loss, elapsed_time = 1, 0, 0, 0, 0.0
    if cfg.resume_path is not None:
        agg_dict = load_checkpoint(state, cfg, device)
        data_start_step, optimizer_step, total_tokens, total_tokens_with_loss, elapsed_time = agg_dict["data_start_step"], agg_dict["optimizer_step"], agg_dict["total_tokens"], agg_dict["total_tokens_with_loss"], agg_dict["elapsed_time"]
        # cfg.max_steps = optimizer_step + cfg.max_steps # make max_steps max NEW steps

    # train
    state = train(state, device, cfg, data_start_step, optimizer_step, total_tokens, total_tokens_with_loss, elapsed_time)
    save_model_only(cfg, state, "final_checkpoint")

    # Now exit
    if is_main_process():
        print("--------------------------------------------------------------------")
        print(f"Training time: {str(datetime.timedelta(seconds=time.monotonic() - train_time))} ")
        max_alloc = f"{torch.cuda.max_memory_allocated(device) / float(1024**3):,.3f} GB"
        max_reserved = f"{torch.cuda.max_memory_reserved(device) / float(1024**3):,.3f} GB"
        print(f"Max. Mem allocated: {max_alloc}. Max. Mem reserved: {max_reserved}.")
        print("--------------------------------------------------------------------")
        wandb.finish()
        dataset_save_dir = f"{cfg.out_path}/{cfg.run_name}/dataset"
        if os.path.exists(dataset_save_dir):
            shutil.rmtree(dataset_save_dir)


def shutdown():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    print(f"---------Total time: {str(datetime.timedelta(seconds=time.monotonic() - global_start_time))} ---------")
    print("-----------------Shutdown complete.--------------------------")


def guarded_main():
    try:
        run_name = main()
        print("--------------------------------------------------------------------")
        print(f"Run {run_name} finished without error.")
    except BaseException:
        print("--------------------------------------------------------------------")
        print("Run finished with errors.")
        raise
    finally:
        shutdown()  # guarantee NCCL deconstruction


if __name__ == "__main__":
    guarded_main()