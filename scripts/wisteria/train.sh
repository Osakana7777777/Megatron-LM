#!/bin/bash


# Use: ./train.sh <data-path> <tokenizer-path>

DATE=$(date "+%Y-%m-%d" --date="TZ=\"Asia/Tokyo\"")
echo "今日の日付は: $DATE"


MODEL_SCALE="2.7B" # or "8B"

case "${MODEL_SCALE}" in
    "800M")
        TENSOR_MODEL_PARALLEL_SIZE=4
        NUM_LAYERS=48
        HIDDEN_SIZE=1024
        NUM_ATTENTION_HEADS=16
        GLOBAL_BATCH_SIZE=32
        ;;
    "8B")
        TENSOR_MODEL_PARALLEL_SIZE=1
        NUM_LAYERS=28
        HIDDEN_SIZE=4096
        NUM_ATTENTION_HEADS=32
        GLOBAL_BATCH_SIZE=8
	PIPELINE_PARALLEL_SIZE=4
        ;;
    "2.7B")
	TENSOR_MODEL_PARALLEL_SIZE=2
	NUM_LAYERS=64
	HIDDEN_SIZE=2560
	NUM_ATTENTION_HEADS=32
	MICRO_BATCH_SIZE=1
        GLOBAL_BATCH_SIZE=1024
	PIPELINE_PARALLEL_SIZE=1
	;;
    *)
        echo "Invalid version specified"
        exit 1
        ;;
esac

DATA_PATH_LIST=(
"/workspace/megatron/scripts/wisteria/data/datasets/binarized/ja_wiki_test_text_document"
#"/workspace/megatron/scripts/wisteria/data/datasets/binarized/en_wiki_0_text_document"
)
DATA_PATH=$(IFS=' '; echo "${DAtA_PATH_LIST[*]}")
echo ${DATA_PATH}
TOKENIZER_PATH=/workspace/megatron/scripts/wisteria/data/tokenizer/code10K_en20K_ja30K.ver2.2.model

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4

CHECKPOINT_DIR="./checkpoints"
DATACACHE_DIR="./data-cache"
TENSORBOARD_DIR="./tensorboard"

mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

export TRITON_CACHE_DIR="./triton-cache/"
export TRITON_CACHE_MANAGER="megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"
export 0TORCH_NCCL_AVOID_RECORD_STREAMS=1
export 1TORCH_NCCL_AVOID_RECORD_STREAMS=1

SEQ_LEN=2048
TRAIN_SAMPLES=73242188  # 300B tokens / 4096
LR_WARMUP_SAMPLES=50000
LR_DECAY_SAMPLES=73192188 # TRAIN_SAMPLES - LR_WARMUP_SAMPLES

JOB_NAME="test-2.7b-ja_en_wiki-tp-${TENSOR_MODEL_PARALLEL_SIZE}-pp-${PIPELINE_PARALLEL_SIZE}-seqlen-${SEQ_LEN}-mbs-${MICRO_BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-1node-${DATE}"

options=" \
       --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE} \
       --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
       --sequence-parallel \
       --use-distributed-optimizer \
       --overlap-param-gather \
       --overlap-grad-reduce \
       --untie-embeddings-and-output-weights \
       --init-method-std 0.02 \
       --position-embedding-type none \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_ATTENTION_HEADS} \
       --group-query-attention \
       --num-query-groups 8 \
       --hybrid-attention-ratio 0 \
       --hybrid-mlp-ratio 0 \
       --seq-length ${SEQ_LEN} \
       --max-position-embeddings ${SEQ_LEN} \
       --train-samples ${TRAIN_SAMPLES} \
       --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
       --lr-decay-samples ${LR_DECAY_SAMPLES} \
       --save ${CHECKPOINT_DIR} \
       --load ${CHECKPOINT_DIR} \
       --data-path "/workspace/megatron/scripts/wisteria/data/datasets/binarized/ja_wiki_test_text_document" "/workspace/megatron/scripts/wisteria/data/datasets/binarized/en_wiki_0_text_document" \
       --data-cache-path ${DATACACHE_DIR} \
       --split 99,1,0 \
       --tokenizer-type SentencePieceTokenizer \
       --tokenizer-model ${TOKENIZER_PATH} \
       --distributed-backend nccl \
       --micro-batch-size ${MICRO_BATCH_SIZE} \
       --global-batch-size ${GLOBAL_BATCH_SIZE} \
       --lr 1.6e-4 \
       --min-lr 1.6e-5 \
       --lr-decay-style cosine \
       --weight-decay 0.1 \
       --clip-grad 1.0 \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --disable-bias-linear \
       --normalization RMSNorm \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 20000 \
       --eval-iters 10 \
       --bf16 \
       --use-mcore-models \
       --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
       --no-create-attention-mask-in-dataloader \
       --log-throughput \
       --tensorboard-dir ${TENSORBOARD_DIR} \
       --wandb-project wisteria-2024 \
       --wandb-exp-name ${JOB_NAME} "

MASTER_PORT=12345
NODE_RANK=0


if [[ ${NODE_RANK} -gt 0 ]]; then
    run_cmd="torchrun --nnodes 1 --nproc_per_node 8 --node_rank ${NODE_RANK} --master_addr localhost --master_port ${MASTER_PORT}  /workspace/megatron/pretrain_mamba.py ${options}" \
	    2>&1 | tee ./log.log
else
	run_cmd="torchrun --nnodes 1 --nproc_per_node 8 --node_rank ${NODE_RANK} --master_addr localhost --master_port ${MASTER_PORT}  /workspace/megatron/pretrain_mamba.py ${options}"
fi

echo ${run_cmd}
eval ${run_cmd}

set +x
