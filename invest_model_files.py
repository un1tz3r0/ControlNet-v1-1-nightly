import argparse
import torch
from safetensors.torch import load_file, save_file


st_state_dict = load_file("models/control_v11p_sd21_zoedepth.safetensors")
with open("st_state_dict.txt", "w") as f:
    key_lines = [f"{k} {v.shape}\n" for k, v in st_state_dict.items()]
    f.writelines(key_lines)



# ckpt_state_dict = torch.load("models/control_v11p_sd21_zoedepth.ckpt", map_location=torch.device("cpu"))["state_dict"]
# with open("ckpt_state_dict.txt", "w") as f:
#     key_lines = [f"{k} {v.shape}\n" for k, v in ckpt_state_dict.items()]
#     f.writelines(key_lines)

