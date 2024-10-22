#!/bin/bash

SCRIPT_FILE=/workspace/megatron/scripts/wisteria

INPUT_FILE=${SCRIPT_FILE}/data/datasets/train_0.jsonl
OUTPUT_FILE=${SCRIPT_FILE}/data/datasets/binarized

mkdir -p ${OUTPUT_FILE}

python ../../tools/preprocess_data.py \
	--input ${INPUT_FILE} \
	--output-prefix ${OUTPUT_FILE}/ja_wiki_test \
	--tokenizer-type SentencePieceTokenizer \
	--tokenizer-model ${SCRIPT_FILE}/data/tokenizer/code10K_en20K_ja30K.ver2.2.model \
	--append-eod \
	--workers 72
