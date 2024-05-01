#!/bin/bash
#SBATCH -A llmservice_nemo_long-context
#SBATCH -p batch_block1,batch_block2,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -J "llmservice_nemo_long-context:megatron_inteleave_conversion"
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

set -x

# Define paths
LUSTRE_PATH="/lustre/fsw/portfolios/llmservice/users"
MEGATRON_DIR="${LUSTRE_PATH}/shantanua/Megatron-Partition/megatron-lm" # Current supported branch: lmcafee/core-converter
RESULTS_DIR="${LUSTRE_PATH}/yihuih/moe-init/15b"

# Input model path
MODEL_DIR="/lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-8t/checkpoints/gpt3-15b-8t-shuffle-mup/base"

# Output model
TARGET_TP=8
TARGET_PP=8
SAVE_DIR="${RESULTS_DIR}/gpt3-15b-8t-CT2-tp${TARGET_TP}-pp${TARGET_PP}"

# Set the container
CONTAINER="gitlab-master.nvidia.com/adlr/megatron-lm/pytorch:23.09-py3-pretrain-draco_cw_ub_tot-te-apex"

# Command that will be executed
read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& export PYTHONPATH="${MEGATRON_DIR}:${PYTHONPATH}" \
&& export CUDA_DEVICE_MAX_CONNECTIONS=1 \
&& python ${MEGATRON_DIR}/tools/checkpoint/convert.py \
    --model-type GPT \
    --load-dir ${MODEL_DIR} \
    --loader mcore \
    --saver mcore \
    --save-dir ${SAVE_DIR} \
    --transformer-impl transformer_engine \
    --target-tensor-parallel-size ${TARGET_TP} \
    --target-pipeline-parallel-size ${TARGET_PP}
EOF

srun --jobid=457828 --ntasks-per-node=1 --gpus-per-node=8 \
    --container-image="$CONTAINER" \
    --no-container-mount-home \
    --container-mounts="/lustre:/lustre" bash -c "${cmd}"