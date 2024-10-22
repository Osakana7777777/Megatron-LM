import torch

ckpt = torch.load("ckpt_convert/checkpoints/iter_0000020/mp_rank_00/model_optim_rng.pt")
param = ckpt["model"]

for key in param.keys():
    if param[key] is not None:
        print(param[key].shape, key)
