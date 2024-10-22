#!/bin/bash

python ../../tools/checkpoint/hybrid_conversion.py \
	--load-dir checkpoints \
	--save-dir ckpt_convert/checkpoints_iter100 \
	--d-model 2560 \
	--mamba-d-state 128 \
	--mamba2-n-groups 8 \
	--mamba2-head-dim 64
