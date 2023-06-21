import argparse
import datetime
import os
import random

import cv2
import einops
import gradio as gr
import numpy as np
import torch

from pytorch_lightning import seed_everything
from tqdm import tqdm

import config

from annotator.canny import CannyDetector
from annotator.lineart import LineartDetector
from annotator.midas import MidasDetector
from annotator.normalbae import NormalBaeDetector
from annotator.oneformer import OneformerADE20kDetector, OneformerCOCODetector
from annotator.uniformer import UniformerDetector
from annotator.util import HWC3, resize_image
from annotator.zoe import ZoeDetector
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from share import *


@torch.no_grad()
def one_image_batch(
    ann,
    model,
    ddim_sampler,
    input_image,
    preprocessor,
    detect_resolution,
    image_resolution,
    low_threshold,
    high_threshold,
    num_samples,
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

    if ann == "canny":
        detected_map = preprocessor(resize_image(input_image, detect_resolution), low_threshold, high_threshold)
    elif ann == "lineart":
        detected_map = preprocessor(resize_image(input_image, detect_resolution), coarse=False)
    else:
        detected_map = preprocessor(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)

    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    print(f"detection {detected_map.shape} {detected_map.dtype} {detected_map.min()} {detected_map.max()}")

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, "b h w c -> b c h w").clone()

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
    shape = (4, H // 8, W // 8)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    model.control_scales = (
        [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
    )
    # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

    samples, intermediates = ddim_sampler.sample(
        ddim_steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond,
    )

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (
        (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    )

    results = [x_samples[i] for i in range(num_samples)]

    re_detections = []
    re_detections_error = []
    for result in results:
        if ann == "canny":
            re_detection_map = preprocessor(resize_image(result, detect_resolution), low_threshold, high_threshold)
        elif ann == "lineart":
            re_detection_map = preprocessor(resize_image(result, detect_resolution), coarse=False)
        else:
            re_detection_map = preprocessor(resize_image(result, detect_resolution))
        re_detection_map = HWC3(re_detection_map)
        re_detection_map = cv2.resize(re_detection_map, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        detected_map_error = np.abs(re_detection_map - detected_map).astype(np.uint8)
        re_detections.append(re_detection_map)
        re_detections_error.append(detected_map_error)

    return detected_map, results, re_detections, re_detections_error


def save_samples(input_img, output, output_dir, img_basename):
    detected_map, results, re_detections, re_detections_error = output

    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"{img_basename}.png"), input_img)
    cv2.imwrite(os.path.join(output_dir, f"{img_basename}_detected.png"), detected_map)
    for i, result in enumerate(results):
        img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"{img_basename}_{i}.png"), img)
        detected_map = cv2.cvtColor(re_detections[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"{img_basename}_{i}_detected.png"), detected_map)
        cv2.imwrite(os.path.join(output_dir, f"{img_basename}_{i}_detected_error.png"), re_detections_error[i])


def main(args):
    if args.ann == "depth":
        model_name = "control_v11f1p_sd15_depth"
        preprocessor = ZoeDetector()
    elif args.ann == "normal":
        model_name = "control_v11p_sd15_normalbae"
        preprocessor = NormalBaeDetector()
    elif args.ann == "canny":
        model_name = "control_v11p_sd15_canny"
        preprocessor = CannyDetector()
    elif args.ann == "seg":
        model_name = "control_v11p_sd15_seg"
        preprocessor = OneformerADE20kDetector()
    elif args.ann == "lineart":
        model_name = "control_v11p_sd15_lineart"
        preprocessor = LineartDetector()
    else:
        raise NotImplementedError
    model = create_model(f"./models/{model_name}.yaml").cpu()
    model.load_state_dict(load_state_dict("./models/v1-5-pruned.ckpt", location="cuda"), strict=False)
    model.load_state_dict(load_state_dict(f"./models/{model_name}.pth", location="cuda"), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    cur_ann_output_dir = os.path.join(args.output_dir, args.ann)
    os.makedirs(cur_ann_output_dir, exist_ok=True)

    img_path = os.path.join(args.img_path)
    img_basename = os.path.basename(img_path).split(".")[0]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = one_image_batch(
        ann=args.ann,
        model=model,
        ddim_sampler=ddim_sampler,
        input_image=img,
        preprocessor=preprocessor,
        detect_resolution=512,
        image_resolution=512,
        low_threshold=100,
        high_threshold=200,
        num_samples=args.num_samples,
        ddim_steps=20,
        guess_mode=False,
        strength=1.0,
        scale=9.0,
        seed=12345,
        eta=1.0,
        prompt="a bottle on table with water in it and label on it",
        a_prompt="best quality",
        n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
        config=config,
    )
    save_samples(img, output, cur_ann_output_dir, img_basename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A parser for NOCS")

    parser.add_argument("--ann", type=str, default="depth")
    parser.add_argument("--img_path", type=str, default="", required=True)
    parser.add_argument("--output_dir", type=str, default="./nocs_output/", required=True)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    if args.ann == "all":
        for ann in tqdm(["depth", "normal", "canny", "seg", "lineart"]):
            args.ann = ann
            main(args)
    else:
        main(args)
