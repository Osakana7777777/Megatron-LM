#!/bin/bash

cd datasets
for i in {1..5}
do
    wget --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v2/-/raw/main/ja/ja_wiki/train_${i}.jsonl.gz
done