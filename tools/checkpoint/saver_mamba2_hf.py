#Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import sys
import os
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, Mamba2Config, AutoTokenizer
from contextlib import contextmanager


def add_arguments(parser):
    group = parser.add_argument_group(title='Mamba2_hf saver.')
    group.add_argument('--hf-tokenizer-path', type=str, default=None,
                       help='Huggingface tokenizer path. eg. /models/state-spaces/mamba2-130m.')
    group.add_argument('--save-dtype', type=str, default='bfloat16')


@contextmanager
def suspend_nn_inits():
    """
    create context manager for loading without init
    see https://github.com/huggingface/transformers/issues/26258
    """
    skip = lambda *args, **kwargs: None  # noqa: E731
    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_   # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip   # replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring


def save_checkpoint(queue: mp.Queue, args):
    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    def check_message(msg):
        if not args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print("Exiting. If you want to ignore this, use the argument --no-checking.")
            exit(1)

    md = queue_get()

    # Verify compatibility of args
    assert hasattr(md, 'checkpoint_args')
    assert md.model_type == 'Mamba'
    mag_conf = md.checkpoint_args

    if args.save_dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif args.save_dtype == 'float32':
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    mamba2_conf = Mamba2Config(
        vocab_size=mag_conf.vocab_size,
        hidden_size=mag_conf.hidden_size,
        state_size=mag_conf.state_size,
        num_hidden_layers=mag_conf.encoder_num_layers,
        expand=mag_conf.expand,
        conv_kernel=mag_conf.conv_kernel,
        rms_norm_eps=mag_conf.norm_epsilon,
        tie_word_embeddings=not mag_conf.untie_embeddings_and_output_weights,
        use_conv_bias=mag_conf.use_conf_bias,
        torch_dtype=torch_dtype,
        chunk_size=mag_conf.chunk_size,
    )

    state_dict = {}

    def set_hf_param(name, tensor: torch.Tensor):
        weight_name = name
        state_dict[weight_name] = tensor.to(torch.bfloat16)

    set_hf_param('backbone.embeddings', queue_get("embeddings")["word embeddings"])
    for i_layer in range(mamba2_conf.num_hidden_layers):
        message = queue_get(f"mamba2 layer {i_layer}")
        suffix = f'backbone.layers.{i_layer}.'
        set_hf_param(suffix + 'mixer.A_log', message["mixer.A_log"])
        set_hf_param(suffix + 'mixer.conv1d.bias', message["mixer.conv1d.bias"])
        set_hf_param(suffix + 'mixer.conv1d.weight', message["mixer.conv1d.weight"])
        set_hf_param(suffix + 'mixer.D', message["mixer.D"])
        set_hf_param(suffix + 'mixer.dt_bias', message["mixer.dt_bias"])
        set_hf_param(suffix + 'mixer.in_proj.weight', message["mixer.in_proj.weight"])
        set_hf_param(suffix + 'mixer.norm.weight', message["mixer.norm.weight"])
        set_hf_param(suffix + 'mixer.out_proj.weight', message["mixer.out_proj.weight"])
        set_hf_param(suffix + 'norm.weight', message["norm.weight"])
    set_hf_param('backbone.norm_f', queue_get('final norm')['weight'])
    set_hf_param('lm_head', queue_get('output layer')['weight'])

    with suspend_nn_inits():
        print("Saving model to disk ...")
        model = AutoModelForCausalLM.from_pretrained(
            None,  # type: ignore
            config=mamba2_conf,
            state_dict=state_dict,
            torch_dtype=torch_dtype
        )
        model.save_pretrained(
            args.save_dir,
            safe_serialization=True,
        )

    #tokenizer = AutoTokenizer.from_pretrained(
    #    args.hf_tokenizer_path
    #)
    #tokenizer.save_pretrained(args.save_dir)
