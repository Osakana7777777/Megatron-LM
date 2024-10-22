cd ../../

python tools/checkpoint/convert.py \
	--model-type Mamba \
	--loader mcore \
	--saver mamba2_hf \
	--load-dir /workspace/megatron/scripts/wisteria/ckpt_convert/checkpoints_iter100 \
	--save-dir /workspace/megatron/scripts/wisteria/hf_ckpts \
	--save-dtype bfloat16 \
	--megatron-path /workspace/megatron
