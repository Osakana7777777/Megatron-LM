#!/bin/bash

pwd

DATASET_PATH=./datasets/en/train_0.jsonl
OUTPUT_DIR=./datasets/binarized
TOKENIZER_PATH=./tokenizer/code10K_en20K_ja30K.ver2.2.model

python ../../../tools/preprocess_data.py \
	    --input ${DATASET_PATH} \
	    --output-prefix ${OUTPUT_DIR}/en_wiki_0 \
	    --tokenizer-type SentencePieceTokenizer \
	    --tokenizer-model ${TOKENIZER_PATH} \
            --workers 72 \
            --append-eod
