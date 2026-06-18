#!/bin/bash
# ============================================================================
# stage2_env.sh  —  shared environment for all Stage 2 jobs
# ============================================================================
# Edit the three paths in the EDIT block below ONCE. Every job file sources this,
# so you do not need to touch the individual jobs. Submit jobs from this directory:
#     cd <repo>/stage2/job_scripts
#     mkdir -p log_ss2          # one time, for Slurm logs
#     sbatch exp7_geometry.job
# ============================================================================

# ── EDIT THESE (once) ────────────────────────────────────────────────────────
# REPO_ROOT must contain the stage2/ folder, model/, and gs_dataset_scenesplat.py
# (i.e. wherever you copied the Stage 2 code). If your code lives under
# Semantic-Scene-Completion rather than Can3Tok, point REPO_ROOT there.
export REPO_ROOT="/home/yli11/scratch-project/Hafeez_thesis/Semantic-Scene-Completion"
export CONDA_ENV="/home/yli11/.conda/envs/can3tok"
# DATA_PATH is the BASE dir (NOT the chunk dir). The chunk root, full root and val/
# are derived from it, exactly as in Stage 1.
export DATA_PATH="/home/yli7/scratch/datasets/gaussian_world/preprocessed/interior_gs"

# Replicate the Stage 1 training mix (chunks + the three extra full-scene roots).
# These are the SAME values used to train every Stage 1 checkpoint.
export TRAIN_DATA="chunks"
export TRAIN_SCENES="3800"
export EXTRA_TRAIN_PATHS="/home/yli7/scratch/datasets/gaussian_world/preprocessed/interior_gs/train:/home/yli7/scratch/datasets/gaussian_world/preprocessed/arkitscenes_mcmc_3dgs/train:/home/yli7/scratch/datasets/gaussian_world/preprocessed/scannetpp_v2_mcmc_3dgs_lang_large_new/train"
export EXTRA_TRAIN_SCENES="800:1290:906"
# ─────────────────────────────────────────────────────────────────────────────

export STAGE1_CONFIG="./model/configs/aligned_shape_latents/shapevae-256.yaml"
# Anchor the accelerate config to THIS directory (job_scripts), so it resolves no
# matter where REPO_ROOT is or where you submit from.
JOBDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ACCELERATE_CONFIG="$JOBDIR/accelerate_config.yaml"

module purge && module load 2023 && module load CUDA/12.1.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="${CONDA_ENV}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"

cd "$REPO_ROOT" || { echo "ERROR: REPO_ROOT not found: $REPO_ROOT"; exit 1; }

# Derive process count from Slurm (do NOT hardcode CUDA_VISIBLE_DEVICES on Snellius)
export NUM_GPUS="${SLURM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
echo "REPO_ROOT=$REPO_ROOT  NUM_GPUS=$NUM_GPUS  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ── Training launcher ────────────────────────────────────────────────────────
# Uses: STAGE, STAGE1_CKPT, RUN_TAG, and optional NUM_EPOCHS / MODEL_SIZE / ROPE_TYPE
run_stage2_train() {
    if [[ "$STAGE1_CKPT" == *"<YOUR"* || -z "$STAGE1_CKPT" ]]; then
        echo "ERROR: STAGE1_CKPT is not set."; exit 1
    fi
    local CMD="accelerate launch --config_file $ACCELERATE_CONFIG --num_processes $NUM_GPUS --num_machines 1 stage2/train_stage2.py"
    CMD="$CMD --stage $STAGE --stage1_checkpoint $STAGE1_CKPT --stage1_config $STAGE1_CONFIG"
    CMD="$CMD --run_tag ${RUN_TAG:-}"
    CMD="$CMD --model_size ${MODEL_SIZE:-B} --rope_type ${ROPE_TYPE:-learned_ape}"
    CMD="$CMD --batch_size ${BATCH_SIZE:-120} --num_epochs ${NUM_EPOCHS:-2000}"
    CMD="$CMD --lr ${LR:-1e-4} --weight_decay ${WEIGHT_DECAY:-1e-2}"
    CMD="$CMD --warmup_steps ${WARMUP_STEPS:-100} --lr_min_ratio ${LR_MIN_RATIO:-0.05}"
    CMD="$CMD --eval_every ${EVAL_EVERY:-50} --val_scenes ${VAL_SCENES:-88}"
    CMD="$CMD --train_data ${TRAIN_DATA:-chunks}"
    [ -n "${TRAIN_SCENES:-}" ] && CMD="$CMD --train_scenes $TRAIN_SCENES"
    [ -n "${EXTRA_TRAIN_PATHS:-}" ] && CMD="$CMD --extra_train_paths $EXTRA_TRAIN_PATHS --extra_train_scenes ${EXTRA_TRAIN_SCENES:-}"
    CMD="$CMD --vis_num_scenes ${VIS_NUM_SCENES:-4} --vis_num_steps ${VIS_NUM_STEPS:-50}"
    CMD="$CMD --flow_diag_freq ${FLOW_DIAG_FREQ:-50} --data_path $DATA_PATH"
    [ -n "${RESUME_CHECKPOINT:-}" ] && CMD="$CMD --resume_checkpoint $RESUME_CHECKPOINT"
    echo "Command: $CMD"; echo ""
    SECONDS=0; eval $CMD; local EXIT=$?
    echo ""; echo "Exit: $EXIT | Duration: $((SECONDS/60))m $((SECONDS%60))s | End: $(date)"
    return $EXIT
}

# ── Sampling launcher ────────────────────────────────────────────────────────
# Uses: OBJECTIVE, STAGE1_CKPT, and the relevant *_CKPT variables.
run_stage2_sample() {
    local CMD="python stage2/sample_stage2.py --objective $OBJECTIVE"
    CMD="$CMD --stage1_checkpoint $STAGE1_CKPT --stage1_config $STAGE1_CONFIG"
    CMD="$CMD --num_samples ${NUM_SAMPLES:-8} --num_steps ${NUM_STEPS:-50}"
    CMD="$CMD --output_dir ${OUTPUT_DIR:-./stage2_samples_${SLURM_JOB_ID}} --data_path $DATA_PATH"
    if [ "$OBJECTIVE" = "generation" ]; then
        [ -n "${SCENE_CKPT:-}" ]    && CMD="$CMD --scene_checkpoint $SCENE_CKPT"
        [ -n "${DC_CKPT:-}" ]       && CMD="$CMD --dc_checkpoint $DC_CKPT --dc_mode ${DC_MODE:-sample}"
        [ -n "${LAYOUT_CKPT:-}" ]   && CMD="$CMD --layout_checkpoint $LAYOUT_CKPT"
        [ -n "${GEOMETRY_CKPT:-}" ] && CMD="$CMD --geometry_checkpoint $GEOMETRY_CKPT"
    else
        CMD="$CMD --completion_checkpoint ${COMPLETION_CKPT} --coverage ${COVERAGE:-0.4}"
    fi
    echo "Command: $CMD"; echo ""
    SECONDS=0; eval $CMD; local EXIT=$?
    echo ""; echo "Exit: $EXIT | Duration: $((SECONDS/60))m $((SECONDS%60))s | End: $(date)"
    return $EXIT
}