# Continued-pretraining launch for Recurrent-SmolLM2-360M on a single Colab/Kaggle GPU.
#
# This is a single-phase reference invocation only -- colab_train_smollm2.ipynb
# is the authoritative full_run driver and additionally runs a two-phase
# FineWeb-Edu-heal / diversify-plus-math-code curriculum (see its MODE cell
# and mix_smollm2_corpus.py's docstring) that a single train.py call like
# this one can't express. Use this script for a quick single-mix run only.
#
# Differences from shells/tinyllama.sh (which targets an 8-32 GPU cluster):
#   - single process, no distributed launch
#   - no --save_n_mins_before_timeout: that flag shells out to `flux job
#     timeleft` (see README.md "Note the save_n_mins_before_timeout flag is
#     designed to work on flux scheduling systems only") -- on Colab/Kaggle
#     there is no `flux` binary, so passing it crashes at the first check.
#     Session-limit safety instead comes purely from --save_interval below
#     plus manual --resume_path on restart.
#   - mean_recurrence_schedule.max_mean_rec=8, mean_backprop_depth_schedule.max_backprop=4
#     (not the dataclass default of 32/8): the converted checkpoint's static
#     config already defaults mean_recurrence=32/mean_backprop_depth=8 (see
#     convert_pretrained_model/models/*/raven_config_minimal.py), which the
#     schedule overrides during training. A mean of 32 loops through the
#     14-layer core is ~450 effective layers per step on average -- far too
#     expensive for a free-tier single GPU. 8 is the budget-realistic
#     compromise between that and the paper's own single-GPU-scale reference
#     runs (shells/tinyllama.sh, Figure 5/7/8), which use max_mean_rec=4.
#   - --muon.use_muon=true: the paper's recipe trains with Muon, not plain
#     AdamW; train.py already supports it via the --muon.* CLI group.
#
# GPU dtype gotcha: no_amp=false hardcodes amp_args["dtype"]=torch.bfloat16
# (see train.py __post_init__). Free-tier Colab T4 and Kaggle T4/P100 are
# pre-Ampere and have no native bf16 tensor cores -- autocast will still run
# but slower/emulated. Check your assigned GPU first:
#   python -c "import torch; print(torch.cuda.get_device_capability())"
# capability >= (8, 0) (A100, L4) -> bf16 is native, use --no_amp=false.
# capability < (8, 0) (T4, P100)  -> use --no_amp=true (fp32, no autocast)
# below; slower per-step but correct. This script defaults to the safe
# (fp32) setting -- flip NO_AMP to false yourself once you've checked.

MODEL_PATH="usr-wwelsh/Recurrent-SmolLM2-360M-4-14-4"   # public HF Hub repo
DATA_PATH="data/smollm2_recurrent_mix"   # output of mix_smollm2_corpus.py
RUN_NAME="smollm2-recurrent-v1"
NO_AMP=true
HUB_CHECKPOINT_REPO="usr-wwelsh/smollm2-recurrent-checkpoints"   # private HF dataset repo; run `huggingface-cli login` first. Holds only the latest resumable checkpoint per RUN_NAME (at "${RUN_NAME}/chkpt.pt") -- train.py overwrites it in place and prunes the local copy on every save, so this never accumulates.

python train.py \
    --run_name="${RUN_NAME}" \
    --out_path=huginn_smollm2 \
    --model_name="${MODEL_PATH}" \
    --hub_checkpoint_repo="${HUB_CHECKPOINT_REPO}" \
    --preprocessed_data_path="${DATA_PATH}" \
    --is_parquet_dataset=true \
    --max_length=1024 \
    --micro_batch_size=8 \
    --batch_size=64 \
    --optim_config.lr=5e-5 \
    --scheduler_args.warmup=0.02 \
    --scheduler_args.cooldown=0.9 \
    --max_grad_norm=1.0 \
    --no_amp=${NO_AMP} \
    --max_steps=7500 \
    --compile=false \
    --save_interval=250 \
    --mean_recurrence_schedule.turn_on=true \
    --mean_recurrence_schedule.warmup=0.25 \
    --mean_recurrence_schedule.max_mean_rec=8 \
    --mean_backprop_depth_schedule.turn_on=true \
    --mean_backprop_depth_schedule.warmup=0.25 \
    --mean_backprop_depth_schedule.start=1 \
    --mean_backprop_depth_schedule.max_backprop=4 \
    --muon.use_muon=true

# max_steps=7500 * batch_size=64 * max_length=1024 ~= 492M tokens -- the low
# end of the "hundreds of millions to low billions" budget in the project
# notes, sized to fit in a handful of Colab/Kaggle sessions. Raise max_steps
# (and re-derive parquet_dataset_max_tokens accordingly, or just rely on
# max_steps directly since it takes precedence -- see train.py's scheduler
# section) once the pipeline is validated end-to-end on this slice.

# To resume after a session is killed:
#   python train.py ...(same args)... \
#       --resume_path=huginn_smollm2/${RUN_NAME}/checkpoint_<last_saved_step>
