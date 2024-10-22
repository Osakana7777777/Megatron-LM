#!/bin/bash

cd ../../../

DATASET_PATH=

python megatron/tools/preprocess_data.py \
    --input $DATASET_PATH \
    --output-prefix $OUTPUT_DIR/ja_wiki \
    --tokenizer-type SentencePieceTokenizer \
    --workers \
    --append-eod 
