# Eval sweep for Recurrent-SmolLM2-360M checkpoints, adapted from shells/eval.sh
# (which targets a multi-GPU AMD/ROCm cluster via HIP_VISIBLE_DEVICES).
#
# Goal per the project notes: at an early checkpoint in the low-recurrence
# warmup phase, confirm the model is *recovering toward* base SmolLM2-360M
# (not stuck below it, not diverging further) -- so this sweeps
# mean_recurrence across several depths per checkpoint rather than assuming
# any single depth is representative.

OUT_ROOT="eval_outputs"
MODEL_PATH="huginn_smollm2/smollm2-recurrent-v1"
CHKPT=250   # first checkpoint saved; add more invocations as training progresses

for MEAN_RECURRENCE in 1 2 4 8; do
    CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
        --model_args pretrained=${MODEL_PATH}/model_only_chkpt_${CHKPT},mean_recurrence=${MEAN_RECURRENCE},add_bos_token=True,dtype="float32",trust_remote_code=True \
        --tasks arc_easy,arc_challenge,hellaswag,mmlu,piqa,winogrande \
        --device cuda \
        --output_path "${OUT_ROOT}/${MODEL_PATH}/model_only_chkpt_${CHKPT}/mean_recurrence_${MEAN_RECURRENCE}" \
        --batch_size auto
done

# Compare against base SmolLM2-360M as the non-inferiority reference point
# (no mean_recurrence arg -- it isn't a recurrent model):
CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args pretrained=HuggingFaceTB/SmolLM2-360M,add_bos_token=True,dtype="float32" \
    --tasks arc_easy,arc_challenge,hellaswag,mmlu,piqa,winogrande \
    --device cuda \
    --output_path "${OUT_ROOT}/SmolLM2-360M-base" \
    --batch_size auto
