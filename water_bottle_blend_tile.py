import argparse
import datetime
import os
import random

import cv2
import einops
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch

from pytorch_lightning import seed_everything

import config

from annotator.util import HWC3, resize_image
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from share import *


# from tqdm import tqdm, trange


@torch.no_grad()
def one_image(
    ann,
    model,
    ddim_sampler,
    input_image,
    detect_resolution,
    image_resolution,
    low_threshold,
    high_threshold,
    num_samples,
    denoise_strength,
    ddim_steps,
    guess_mode,
    strength,
    scale,
    seed,
    eta,
    prompt,
    a_prompt,
    n_prompt,
    config,
):
    input_image = HWC3(input_image)
    detected_map = input_image.copy()

    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, "b h w c -> b c h w").clone()

    img = torch.from_numpy(img.copy()).float().cuda() / 127.0 - 1.0
    img = torch.stack([img for _ in range(num_samples)], dim=0)
    img = einops.rearrange(img, "b h w c -> b c h w").clone()

    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    cond = {
        "c_concat": [control],
        "c_crossattn": [model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)],
    }
    un_cond = {
        "c_concat": None if guess_mode else [control],
        "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
    }

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=True)
    t_enc = min(int(denoise_strength * ddim_steps), ddim_steps - 1)
    z = model.get_first_stage_encoding(model.encode_first_stage(img))
    z_enc = ddim_sampler.stochastic_encode(z, torch.tensor([t_enc] * num_samples).to(model.device))

    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    model.control_scales = (
        [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
    )
    # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

    samples = ddim_sampler.decode(
        z_enc, cond, t_enc, unconditional_guidance_scale=scale, unconditional_conditioning=un_cond
    )

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (
        (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    )

    results = [x_samples[i] for i in range(num_samples)]
    return results


def save_samples(input_img, results, output_dir, img_file):
    img_basename = os.path.basename(img_file).split(".")[0]
    for i, result in enumerate(results):
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(output_dir, f"{img_basename}_{i}_tiled.png")
        cv2.imwrite(out_path, result)


def main(args):
    model_name = "control_v11f1e_sd15_tile"
    model = create_model(f"./models/{model_name}.yaml").cpu()
    model.load_state_dict(load_state_dict("./models/v1-5-pruned.ckpt", location="cuda"), strict=False)
    model.load_state_dict(load_state_dict(f"./models/{model_name}.pth", location="cuda"), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    image_files = sorted(os.listdir(args.data_dir))
    if args.debug:
        image_files = image_files[:1]

    cur_ann_output_dir = os.path.join(args.output_dir, args.ann)

    for img_i, img_file in enumerate(image_files):
        img_basename = os.path.splitext(img_file)[0]
        print("-" * 20, "Processing", img_basename, img_i, "/", len(image_files), "-" * 20)
        for i in range(args.num_samples):
            blend_img_file = os.path.join(cur_ann_output_dir, f"{img_basename}_{i}_blended.png")

            blend_img = cv2.imread(blend_img_file)
            blend_img = cv2.cvtColor(blend_img, cv2.COLOR_BGR2RGB)

            results = one_image(
                ann=args.ann,
                model=model,
                ddim_sampler=ddim_sampler,
                input_image=blend_img,
                detect_resolution=512,
                image_resolution=512,
                low_threshold=100,
                high_threshold=200,
                num_samples=1,  # we already have num_samples images
                denoise_strength=1.0,
                ddim_steps=32,
                guess_mode=False,
                strength=1.0,
                scale=9.0,
                seed=12345,
                eta=1.0,
                prompt="a bottle with water in it and label on it",
                a_prompt="best quality",
                n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
                config=config,
            )
            save_samples(blend_img, results, cur_ann_output_dir, blend_img_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A collection of util tools for AzureML")

    parser.add_argument("--ann", type=str, default="depth")
    parser.add_argument("--data_dir", type=str, default="../data/water_bottle_all_renamed/single", required=True)
    parser.add_argument("--output_dir", type=str, default="./water_bottle_output/", required=True)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    if args.ann == "all":
        for ann in ["depth", "normal", "canny", "seg", "lineart"]:
            args.ann = ann
            print("*" * 20, "Running for", ann, "*" * 20)
            main(args)
    else:
        main(args)
