# Full phase1-heal -> phase2-diverse -> SFT -> GRPO run for Recurrent-SmolLM2-360M, on a
# rented Vast.ai GPU (L4/A5000 per GPU_PRICING_NOTES.md), run over SSH/tmux -- not Colab.
# colab_train_smollm2.ipynb stays the free-tier smoke-test sanity check for phase1/2 plumbing
# only; this is where the real run lives, including the two stages the notebook never covers.
#
# `set -e`: every stage (mix, train, eval_gate) exits non-zero on failure, and eval_gate.py
# specifically exits non-zero on a knowledge-benchmark regression vs base -- so a regression
# halts this whole script immediately, before the next stage's GPU-hours are spent. This is
# the shell-script equivalent of the notebook's `assert _exit_code == 0` convention.
set -e

# --- one-time setup ---------------------------------------------------------
apt-get install -y bubblewrap   # code_reward's sandbox; not a pip dependency
huggingface-cli login   # needs HF_TOKEN with write access
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY or training runs without logging}"

MODEL_PATH="usr-wwelsh/Recurrent-SmolLM2-360M-4-14-4"
RUN_NAME="smollm2-recurrent-full"
OUT_PATH="huginn_smollm2"
DATA_PATH="data/${RUN_NAME}"
HUB_CHECKPOINT_REPO="usr-wwelsh/smollm2-recurrent-checkpoints"
FINAL_MODEL_REPO="usr-wwelsh/Recurrent-SmolLM2-360M-4-14-4-trained"

# TOKEN_BUDGET/MAX_STEPS below are the shells/smollm2_colab.sh placeholders, sized for the
# GPU_PRICING_NOTES.md T4-equivalent guess -- re-derive both once a real L4 tok/s number
# replaces that guess (see GPU_PRICING_NOTES.md's "Next step").
PHASE1_MAX_STEPS=3750
PHASE2_MAX_STEPS=3750
MEAN_RECURRENCE=8

# One-time baseline eval (cached, reused by every gate below -- see eval_gate.py's
# baseline_results_path docstring for why this isn't re-run per gate).
BASELINE_DIR="eval_outputs/SmolLM2-360M-base"
if [ ! -d "${BASELINE_DIR}" ]; then
    lm_eval --model hf \
        --model_args pretrained=HuggingFaceTB/SmolLM2-360M,add_bos_token=True,dtype="float32" \
        --tasks arc_easy,arc_challenge,hellaswag,mmlu,piqa,winogrande \
        --device cuda \
        --output_path "${BASELINE_DIR}" \
        --batch_size auto
fi
BASELINE_RESULTS=$(find "${BASELINE_DIR}" -name "results*.json" | head -1)

run_gate () {
    local ckpt_path="$1"
    lm_eval --model hf \
        --model_args pretrained=${ckpt_path},mean_recurrence=${MEAN_RECURRENCE},add_bos_token=True,dtype="float32",trust_remote_code=True \
        --tasks arc_easy,arc_challenge,hellaswag,mmlu,piqa,winogrande \
        --device cuda \
        --output_path "${ckpt_path}/eval_gate_lm_eval" \
        --batch_size auto
    local ckpt_results
    ckpt_results=$(find "${ckpt_path}/eval_gate_lm_eval" -name "results*.json" | head -1)
    python eval_gate.py --ckpt_path="${ckpt_path}" --baseline_results_path="${ckpt_results}" \
        --mean_recurrence=${MEAN_RECURRENCE}
    # eval_gate.py itself only computes the verdict from already-run lm_eval output; running
    # lm_eval here (rather than inside eval_gate.py) keeps the heavy GPU-bound eval invocation
    # in the same place every other eval call in this repo lives (shells/*.sh), not duplicated
    # inside a Python script.
}

# --- phase 1: heal on plain FineWeb-Edu --------------------------------------
python mix_smollm2_corpus.py --save_path="${DATA_PATH}/phase1" \
    --token_budget=500000000 --fineweb_edu_weight=1.0 --dclm_weight=0 --cosmopedia_v2_weight=0

python train.py \
    --run_name="${RUN_NAME}" --out_path="${OUT_PATH}" --model_name="${MODEL_PATH}" \
    --hub_checkpoint_repo="${HUB_CHECKPOINT_REPO}" \
    --preprocessed_data_path="${DATA_PATH}/phase1" --is_parquet_dataset=true \
    --max_length=1024 --micro_batch_size=8 --batch_size=64 \
    --optim_config.lr=5e-5 --scheduler_args.warmup=0.02 --scheduler_args.cooldown=0.9 \
    --max_grad_norm=1.0 --no_amp=false --max_steps=${PHASE1_MAX_STEPS} --compile=false \
    --save_interval=250 --wandb_project=smollm2-recurrent \
    --mean_recurrence_schedule.turn_on=true --mean_recurrence_schedule.warmup=0.25 \
    --mean_recurrence_schedule.max_mean_rec=${MEAN_RECURRENCE} \
    --mean_backprop_depth_schedule.turn_on=true --mean_backprop_depth_schedule.warmup=0.25 \
    --mean_backprop_depth_schedule.start=1 --mean_backprop_depth_schedule.max_backprop=4 \
    --muon.use_muon=true

PHASE1_CKPT="${OUT_PATH}/${RUN_NAME}/model_only_chkpt_${PHASE1_MAX_STEPS}"
run_gate "${PHASE1_CKPT}"

# --- phase 2: diversify into FineWeb-Edu/DCLM/Cosmopedia + math/code reasoning ---
python mix_smollm2_corpus.py --save_path="${DATA_PATH}/phase2" \
    --token_budget=500000000 \
    --fineweb_edu_weight=0.5 --dclm_weight=0.3 --cosmopedia_v2_weight=0.04 \
    --math_reasoning_weight=0.08 --code_reasoning_weight=0.08

python train.py \
    --run_name="${RUN_NAME}" --out_path="${OUT_PATH}" --model_name="${PHASE1_CKPT}" \
    --hub_checkpoint_repo="${HUB_CHECKPOINT_REPO}" \
    --preprocessed_data_path="${DATA_PATH}/phase2" --is_parquet_dataset=true \
    --max_length=1024 --micro_batch_size=8 --batch_size=64 \
    --optim_config.lr=5e-5 --scheduler_args.warmup=0.02 --scheduler_args.cooldown=0.9 \
    --max_grad_norm=1.0 --no_amp=false --max_steps=${PHASE2_MAX_STEPS} --compile=false \
    --save_interval=250 --wandb_project=smollm2-recurrent \
    --mean_recurrence_schedule.turn_on=true --mean_recurrence_schedule.warmup=0.25 \
    --mean_recurrence_schedule.max_mean_rec=${MEAN_RECURRENCE} \
    --mean_backprop_depth_schedule.turn_on=true --mean_backprop_depth_schedule.warmup=0.25 \
    --mean_backprop_depth_schedule.start=1 --mean_backprop_depth_schedule.max_backprop=4 \
    --muon.use_muon=true

PHASE2_CKPT="${OUT_PATH}/${RUN_NAME}/model_only_chkpt_${PHASE2_MAX_STEPS}"
run_gate "${PHASE2_CKPT}"

# --- SFT: CoT-reasoning instruction tuning, prerequisite for GRPO -------------
python mix_sft_data.py --save_path="${DATA_PATH}/sft"
SFT_SYS_PROMPT=$(cat "${DATA_PATH}/sft/system_prompt.txt")

python train.py \
    --run_name="${RUN_NAME}-sft" --out_path="${OUT_PATH}" --model_name="${PHASE2_CKPT}" \
    --hub_checkpoint_repo="${HUB_CHECKPOINT_REPO}" \
    --dataset_location="${DATA_PATH}/sft/dataset" \
    --sys_prompt="${SFT_SYS_PROMPT}" \
    --max_length=2048 --micro_batch_size=8 --batch_size=64 \
    --optim_config.lr=1e-5 --scheduler_args.warmup=0.02 --scheduler_args.cooldown=0.9 \
    --max_grad_norm=1.0 --no_amp=false --epochs=2 --compile=false \
    --save_interval=250 --wandb_project=smollm2-recurrent \
    --mean_recurrence_schedule.turn_on=true --mean_recurrence_schedule.warmup=0.05 \
    --mean_recurrence_schedule.max_mean_rec=${MEAN_RECURRENCE} \
    --mean_backprop_depth_schedule.turn_on=true --mean_backprop_depth_schedule.warmup=0.05 \
    --mean_backprop_depth_schedule.start=1 --mean_backprop_depth_schedule.max_backprop=4 \
    --muon.use_muon=true

SFT_CKPT=$(find "${OUT_PATH}/${RUN_NAME}-sft" -maxdepth 1 -name "model_only_chkpt_*" | sort -t_ -k4 -n | tail -1)
run_gate "${SFT_CKPT}"

# --- GRPO: verifiable math (GSM8K) + code (MBPP) reward -----------------------
python mix_grpo_data.py --save_path="${DATA_PATH}/grpo"

python grpo_train.py \
    --model_name="${SFT_CKPT}" --dataset_path="${DATA_PATH}/grpo" \
    --out_path="${OUT_PATH}" --run_name="${RUN_NAME}-grpo" \
    --hub_checkpoint_repo="${FINAL_MODEL_REPO}" \
    --mean_recurrence=${MEAN_RECURRENCE} --wandb_project=smollm2-recurrent
# No eval_gate after GRPO on purpose -- gsm8k/code accuracy here is the "did it work" signal
# (the whole point of this stage), not a regression check. Run the notebook's eval sweep
# (or shells/eval_smollm2.sh) against the final model to actually see those numbers.
