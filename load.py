import torch
import os
print(os.getcwd())
a = torch.load("scripts/wisteria/ckpt_convert/checkpoints/iter_0002000/mp_rank_00/model_optim_rng.pt", map_location="cpu")
print(a.keys())
s = 0
for key in a["model"].keys():
    if a["model"][key] is not None:
        s += a["model"][key].numel()
        print(a["model"][key].shape, key)

print("==================================")
print(s)

