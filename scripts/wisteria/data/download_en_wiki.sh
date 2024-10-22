#!/bin/bash

mkdir -p datasets/en
cd datasets/en
for i in {0..3}
do
	    wget --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v2/-/raw/main/en/en_wiki/train_${i}.jsonl.gz
	    gunzip train_${i}.jsonl.gz
    done
