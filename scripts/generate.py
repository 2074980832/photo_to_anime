#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SDXL 图生图脚本（适配 Animagine XL 4.0 + 日志自动记录）
-------------------------------------------------------
功能：
- 图生图 SDXL (StableDiffusionXLImg2ImgPipeline)
- 支持 SDXL LoRA
- 支持 SDXL ControlNet
- 自动缩放输入图片避免 OOM（支持 8GB 显卡）
- 自动检测 GPU 显存决定缩放策略
- 自动启用 xformers / VAE slicing / attention slicing
- 不生成对比图
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
import logging
import os

# 导入相关类
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLControlNetImg2ImgPipeline
from controlnet_aux import CannyDetector, MidasDetector, OpenposeDetector, LineartDetector, PidiNetDetector, NormalBaeDetector

# ==============================
# 日志系统初始化
# ==============================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # 控制台输出
        logging.FileHandler(f"{LOG_DIR}/generate.log", mode="a", encoding="utf-8")  # 写入文件
    ]
)

log = logging.getLogger(__name__)

log.info("=== SDXL generate_xl.py 启动 ===")



# ==============================
# 自动 GPU 显存检测
# ==============================

def detect_max_input_size():
    if not torch.cuda.is_available():
        return 512

    mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    if mem <= 8:
        return 512     # 8GB 显卡
    elif mem <= 12:
        return 768     # 12GB 显卡
    else:
        return 1024    # 大显存


# ==============================
# 自动预处理 ControlNet
# ==============================

def preprocess_image(img, cnet_type):
    cnet_type = cnet_type.lower()

    if cnet_type == "canny":
        return CannyDetector()(img)

    if cnet_type == "depth":
        return MidasDetector()(img)

    if cnet_type == "openpose":
        return OpenposeDetector()(img)

    if cnet_type == "lineart":
        return LineartDetector()(img)

    if cnet_type == "softedge":
        return PidiNetDetector()(img)

    if cnet_type == "normal":
        return NormalBaeDetector()(img)

    log.warning(f"未知 ControlNet 类型：{cnet_type}，使用原图")
    return img


# ==============================
# 输入图自动 Resize（避免 OOM）
# ==============================

def resize_input_image(img, max_size):
    w, h = img.size
    max_dim = max(w, h)

    if max_dim <= max_size:
        return img

    scale = max_size / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)

    log.info(f"输入图片过大，已缩放至：{new_w}x{new_h}")
    return img.resize((new_w, new_h), Image.LANCZOS)


# ==============================
# 主流程
# ==============================

def main():

    parser = argparse.ArgumentParser(description="SDXL 图生图（适配 Animagine XL 4.0）")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=Path, required=False)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--prompt", type=str, default="anime style, masterpiece, high score, absurdres")
    parser.add_argument("--negative", type=str, default="lowres, bad anatomy, extra digits, blurry")

    parser.add_argument("--lora", type=str, nargs="*")
    parser.add_argument("--controlnet", type=str)
    parser.add_argument("--cnet-type", type=str)

    parser.add_argument("--strength", type=float, default=0.45)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=5.0)

    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=1216)

    parser.add_argument("--max-input-size", type=int, default=None)

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # ----------------------------------
    # 自动决定最大输入图片尺寸
    # ----------------------------------

    if args.max_input_size:
        max_input_size = args.max_input_size
    else:
        max_input_size = detect_max_input_size()

    log.info(f"输入图片最大尺寸：{max_input_size}")

    # ----------------------------------
    # 控制器加载（SDXL ControlNet）
    # ----------------------------------

    controlnet = None
    if args.controlnet:
        log.info(f"加载 SDXL ControlNet: {args.controlnet}")
        controlnet = ControlNetModel.from_pretrained(
            args.controlnet,
            torch_dtype=torch.float16,
        )

    # ----------------------------------
    # 加载 SDXL 图生图 Pipeline
    # ----------------------------------

    if controlnet:
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            args.model,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            add_watermarker=False
        )
    else:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            add_watermarker=False
        )

    # ----------------------------------
    # 优化显存
    # ----------------------------------

    try:
        pipe.enable_xformers_memory_efficient_attention()
        log.info("xformers 已启用")
    except:
        log.warning("xformers 未能启用")

    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()

    pipe.to(args.device)

    # ----------------------------------
    # LoRA 加载
    # ----------------------------------

    if args.lora:
        for l in args.lora:
            log.info(f"加载 SDXL LoRA: {l}")
            pipe.load_lora_weights(l)

    # ----------------------------------
    # 处理输入图片
    # ----------------------------------

    if args.input:
        if args.input.is_dir():
            images = [args.input / f for f in args.input.iterdir()]
        else:
            images = [args.input]
    else:
        images = [None]

    log.info(f"需要处理 {len(images)} 张图片")

    idx = 1
    for img_path in images:

        if img_path:
            img = Image.open(img_path).convert("RGB")
            img = resize_input_image(img, max_input_size)

            # ControlNet 预处理
            if args.controlnet and args.cnet_type:
                img_cnet = preprocess_image(img, args.cnet_type)
            else:
                img_cnet = None

            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative,
                image=img,
                control_image=img_cnet,
                strength=args.strength,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                width=args.width,
                height=args.height
            )
        else:
            # 文生图模式
            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                width=args.width,
                height=args.height
            )

        out_img = result.images[0]
        out_path = args.output / f"result_{idx}.png"
        out_img.save(out_path)

        log.info(f"已保存输出图像：{out_path}")
        idx += 1


if __name__ == "__main__":
    main()
