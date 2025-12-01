#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU-only LoRA training script for:
- Stable Diffusion 1.x base + optional ControlNet condition
- Train LoRA for UNet / text_encoder / (optional) ControlNet
- Save LoRA weights as safetensors

数据约定:
- 训练图片: data/processed/photos   (例如: realperson_000_image.png)
- 控制图:   data/processed/edges    (例如: realperson_000_edge.png)
  注意: edges 中混有动漫线条文件无妨；只有与 photos 成对匹配的 realperson_xxx_edge 会被用到
"""

import os
import argparse
import json
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, DDPMScheduler
from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load

DEVICE = torch.device("cpu")  # 强制 CPU


# =====================
# LoRA wrapper
# =====================
class LoRALinear(nn.Module):
    """
    Wrap an nn.Linear with LoRA adapters.
    out = orig(x) + scaling * (x @ A.T @ B.T)
    """

    def __init__(self, orig: nn.Linear, r: int = 4, alpha: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.orig = orig
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.r = r
        self.alpha = alpha if alpha is not None else r
        self.scaling = float(self.alpha) / max(1, self.r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
            nn.init.normal_(self.lora_A, std=0.01)
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        for p in self.orig.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.orig(x)
        if self.r > 0:
            xi = self.dropout(x) if self.dropout is not None else x
            lora_out = (xi @ self.lora_A.T) @ self.lora_B.T
            out = out + lora_out * self.scaling
        return out


def replace_linear_with_lora(
    module: nn.Module,
    r: int = 4,
    alpha: Optional[int] = None,
    dropout: float = 0.0,
    target_names: Optional[List[str]] = None,
    prefix: str = "",
) -> int:
    """
    递归地将 module 内部符合条件的 nn.Linear 替换为 LoRALinear.
    target_names: 只替换名字中包含这些子串的层；None 则替换所有线性层。
    """
    replaced = 0
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            should = target_names is None or any(t in full_name for t in target_names)
            if should:
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
        else:
            replaced += replace_linear_with_lora(
                child, r=r, alpha=alpha, dropout=dropout, target_names=target_names, prefix=full_name
            )
    return replaced


def collect_lora_state_dict(root_module: nn.Module, module_key_prefix: str) -> Dict[str, torch.Tensor]:
    """
    从 root_module 中收集所有 LoRALinear 的参数 (A/B/scaling/r)，
    返回用于保存的 state_dict。
    key 以 module_key_prefix 开头，用于区分组件 (text_encoder./unet./controlnet.)
    """
    out: Dict[str, torch.Tensor] = {}
    for name, m in root_module.named_modules():
        if isinstance(m, LoRALinear) and getattr(m, "r", 0) > 0:
            keybase = f"{module_key_prefix}{name.replace('.', '_')}"
            out[f"{keybase}.lora_A"] = m.lora_A.detach().cpu()
            out[f"{keybase}.lora_B"] = m.lora_B.detach().cpu()
            out[f"{keybase}.scaling"] = torch.tensor(m.scaling)
            out[f"{keybase}.r"] = torch.tensor(m.r, dtype=torch.int32)
    return out


def load_lora_into_module(root_module: nn.Module, state: Dict[str, torch.Tensor], module_key_prefix: str):
    """
    将 safetensors 中保存的 LoRA 权重加载回 root_module 内部的 LoRALinear.
    module_key_prefix 必须与保存时一致。
    """
    for name, m in root_module.named_modules():
        if isinstance(m, LoRALinear) and getattr(m, "r", 0) > 0:
            keybase = f"{module_key_prefix}{name.replace('.', '_')}"
            a_key = f"{keybase}.lora_A"
            b_key = f"{keybase}.lora_B"
            s_key = f"{keybase}.scaling"
            if a_key in state and b_key in state:
                a = state[a_key].to(m.lora_A.device)
                b = state[b_key].to(m.lora_B.device)
                if a.shape == m.lora_A.shape and b.shape == m.lora_B.shape:
                    m.lora_A.data.copy_(a)
                    m.lora_B.data.copy_(b)
                if s_key in state:
                    m.scaling = float(state[s_key].item())


# =====================
# Dataset
# =====================
class PhotoControlDataset(Dataset):
    """
    - images_dir: data/processed/photos   (预处理人像，如 realperson_000_image.png)
    - control_dir: data/processed/edges   (真实图线条 realperson_000_edge.png；目录里混有动漫线稿不会被用到)
    - captions_file: 可选 CSV/JSONL filename->caption；否则用模板或默认 prompt

    返回:
    {
        "pixel_values": (3,H,W) 张量, [-1,1]
        "control_image": (3,H,W) 张量 (无控制图时为全0占位，不返回 None)
        "input_ids": tokenizer 的 ids
        "filename": 文件名
    }
    """

    def __init__(
        self,
        images_dir: str,
        control_dir: Optional[str] = None,
        captions_file: Optional[str] = None,
        tokenizer: Optional[CLIPTokenizer] = None,
        resolution: int = 512,
        prompt_template: Optional[str] = None,
        flip_prob: float = 0.0,
    ):
        super().__init__()
        self.images_dir = images_dir
        self.control_dir = control_dir
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.flip_prob = flip_prob
        self.prompt_template = prompt_template

        captions_map: Dict[str, str] = {}
        if captions_file and os.path.exists(captions_file):
            if captions_file.endswith(".jsonl") or captions_file.endswith(".json"):
                with open(captions_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        fn = obj.get("filename") or obj.get("file") or obj.get("image")
                        caption = obj.get("caption") or obj.get("text") or ""
                        if fn:
                            captions_map[fn] = caption
            else:
                import csv

                with open(captions_file, newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 2:
                            captions_map[row[0]] = row[1]

        # 只匹配 realperson_xxx_image.png -> realperson_xxx_edge.png；其余边缘图自动忽略
        self.samples = []
        image_files = sorted(
            [fn for fn in os.listdir(images_dir) if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        )

        for fn in image_files:
            img_path = os.path.join(images_dir, fn)
            control_path = None

            if control_dir:
                stem, ext = os.path.splitext(fn)
                if stem.endswith("_image"):
                    edge_stem = stem.replace("_image", "_edge")
                    cand = os.path.join(control_dir, edge_stem + ext)
                    if os.path.exists(cand):
                        control_path = cand
                else:
                    # 兜底：同名匹配
                    cand_same = os.path.join(control_dir, fn)
                    if os.path.exists(cand_same):
                        control_path = cand_same

            caption = captions_map.get(fn)
            if caption is None:
                stem = os.path.splitext(fn)[0]
                if self.prompt_template:
                    caption = self.prompt_template.replace("{filename}", stem)
                else:
                    caption = "a high quality anime style portrait, detailed, beautiful lighting"

            self.samples.append((img_path, control_path, caption))

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.resolution, self.resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.control_transform = transforms.Compose(
            [
                transforms.Resize((self.resolution, self.resolution)),
                transforms.ToTensor(),  # [0,1]
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, control_path, caption = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if torch.rand(1).item() < self.flip_prob:
            image = transforms.functional.hflip(image)
        pixel_values = self.image_transform(image)

        # 永远返回 Tensor，不返回 None（无控制图则返回全0占位）
        if control_path:
            ci = Image.open(control_path).convert("RGB")
            if torch.rand(1).item() < self.flip_prob:
                ci = transforms.functional.hflip(ci)
            control_image = self.control_transform(ci)
        else:
            control_image = torch.zeros_like(pixel_values)

        if self.tokenizer is not None:
            toks = self.tokenizer(
                caption,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
        else:
            toks = caption

        return {
            "pixel_values": pixel_values,
            "control_image": control_image,
            "input_ids": toks,
            "filename": os.path.basename(img_path),
        }


# =====================
# Training
# =====================
def train(args):
    print("Using device:", DEVICE)

    # 1) 加载 Stable Diffusion 1.x 组件：tokenizer + text_encoder + VAE + UNet
    print("Loading base models (this may take a while on CPU)...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=True
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    ).to(DEVICE)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    ).to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    ).to(DEVICE)

    controlnet = None
    if args.controlnet_model:
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model).to(DEVICE)
        for p in controlnet.parameters():
            p.requires_grad = False

    # 冻结 base 权重
    for p in vae.parameters():
        p.requires_grad = False
    for p in unet.parameters():
        p.requires_grad = False
    for p in text_encoder.parameters():
        p.requires_grad = False

    # 2) 注入 LoRA（UNet + text_encoder；可选 ControlNet）
    print("Injecting LoRA adapters...")
    te_count = replace_linear_with_lora(
        text_encoder,
        r=args.rank,
        alpha=args.alpha,
        dropout=args.lora_dropout,
        target_names=args.te_target_names,
    )
    unet_count = replace_linear_with_lora(
        unet,
        r=args.rank,
        alpha=args.alpha,
        dropout=args.lora_dropout,
        target_names=args.unet_target_names,
    )
    cn_count = 0
    if controlnet and args.train_controlnet_lora:
        cn_count = replace_linear_with_lora(
            controlnet,
            r=args.rank,
            alpha=args.alpha,
            dropout=args.lora_dropout,
            target_names=args.cn_target_names,
        )
    print(f"LoRA injected: text_encoder={te_count}, unet={unet_count}, controlnet={cn_count}")

    # 3) resume
    if args.lora_checkpoint and os.path.exists(args.lora_checkpoint):
        print("Loading existing LoRA checkpoint:", args.lora_checkpoint)
        state = safetensors_load(args.lora_checkpoint)
        load_lora_into_module(text_encoder, state, "text_encoder.")
        load_lora_into_module(unet, state, "unet.")
        if controlnet:
            load_lora_into_module(controlnet, state, "controlnet.")

    # 4) trainable params = 所有 LoRA A/B
    def gather_lora_params(module: nn.Module):
        params = []
        for m in module.modules():
            if isinstance(m, LoRALinear) and getattr(m, "r", 0) > 0:
                params.append(m.lora_A)
                params.append(m.lora_B)
        return params

    trainable_params = []
    trainable_params += gather_lora_params(text_encoder)
    trainable_params += gather_lora_params(unet)
    if controlnet and args.train_controlnet_lora:
        trainable_params += gather_lora_params(controlnet)

    if len(trainable_params) == 0:
        raise RuntimeError("No LoRA parameters found to train. Check injection settings.")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # 5) dataset / dataloader
    dataset = PhotoControlDataset(
        images_dir=args.train_data_dir,
        control_dir=args.control_data_dir,
        captions_file=args.captions_file,
        tokenizer=tokenizer,
        resolution=args.resolution,
        prompt_template=args.prompt_template,
        flip_prob=args.flip_prob,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # 6) 噪声调度器（SD1 风格）
    if os.path.exists(os.path.join(args.pretrained_model_name_or_path, "scheduler")):
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
    else:
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )

    vae.eval()
    text_encoder.train()
    unet.train()
    if controlnet and args.train_controlnet_lora:
        controlnet.train()
    elif controlnet:
        controlnet.eval()

    global_step = 0
    print("Start training (CPU only, will be slow; 建议先用很小的 max_train_steps 做烟雾测试)...")

    for epoch in range(args.num_train_epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}", ncols=100)
        for batch in loop:
            if global_step >= args.max_train_steps:
                break

            pixel_values = batch["pixel_values"].to(DEVICE)
            control_image = batch["control_image"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)

            # encode image -> latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=DEVICE,
            ).long()
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 文本编码 (单路，768 维)
            enc_out = text_encoder(input_ids)
            encoder_hidden_states = (
                enc_out.last_hidden_state
                if hasattr(enc_out, "last_hidden_state")
                else enc_out[0]
            )

            # ControlNet 前向（可选，兼容新旧 diffusers 参数名）
            cn_out = None
            down_block_res_samples = None
            mid_block_res_sample = None

            if controlnet is not None:
                try:
                    cn_out = controlnet(
                        sample=latents,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=control_image,  # 老参数名
                        return_dict=True,
                    )
                except TypeError:
                    cn_out = controlnet(
                        sample=latents,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        conditioning_image=control_image,  # 新参数名
                        return_dict=True,
                    )

                # 兼容不同版本的输出字段
                if hasattr(cn_out, "down_block_res_samples"):
                    down_block_res_samples = cn_out.down_block_res_samples
                    mid_block_res_sample = cn_out.mid_block_res_sample
                if hasattr(cn_out, "down_block_additional_residuals"):
                    down_block_res_samples = cn_out.down_block_additional_residuals
                if hasattr(cn_out, "mid_block_additional_residual"):
                    mid_block_res_sample = cn_out.mid_block_additional_residual

            # UNet 预测
            unet_kwargs = {"encoder_hidden_states": encoder_hidden_states}
            if down_block_res_samples is not None:
                unet_kwargs["down_block_additional_residuals"] = down_block_res_samples
            if mid_block_res_sample is not None:
                unet_kwargs["mid_block_additional_residual"] = mid_block_res_sample

            model_pred = unet(noisy_latents, timesteps, **unet_kwargs).sample

            loss = F.mse_loss(model_pred, noise) / args.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % args.logging_steps == 0:
                loop.set_postfix({"step": global_step, "loss": float(loss.detach().cpu())})

            # 保存 checkpoint
            if global_step % args.save_steps == 0 and global_step > 0:
                os.makedirs(args.output_dir, exist_ok=True)
                sd = {}
                sd.update(collect_lora_state_dict(text_encoder, "text_encoder."))
                sd.update(collect_lora_state_dict(unet, "unet."))
                if controlnet:
                    sd.update(collect_lora_state_dict(controlnet, "controlnet."))
                ckpt_path = os.path.join(args.output_dir, f"lora_step_{global_step}.safetensors")
                safetensors_save(sd, ckpt_path)
                print(f"[INFO] Saved LoRA checkpoint: {ckpt_path}")

            global_step += 1

        if global_step >= args.max_train_steps:
            break

    # 最终保存
    os.makedirs(args.output_dir, exist_ok=True)
    final_state = {}
    final_state.update(collect_lora_state_dict(text_encoder, "text_encoder."))
    final_state.update(collect_lora_state_dict(unet, "unet."))
    if controlnet:
        final_state.update(collect_lora_state_dict(controlnet, "controlnet."))
    final_path = os.path.join(args.output_dir, "lora_final.safetensors")
    safetensors_save(final_state, final_path)
    print(f"[DONE] Training finished. Saved final LoRA to: {final_path}")


# =====================
# CLI
# =====================
def get_args():
    parser = argparse.ArgumentParser(
        description="CPU-only LoRA training for Stable Diffusion 1.x + optional ControlNet"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="SD1.x 基础模型目录或 HF repo id，如 runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument(
        "--controlnet_model",
        type=str,
        default=None,
        help="ControlNet 模型目录或 HF repo id (可选，如 lllyasviel/sd-controlnet-canny)",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="人像图片目录，如 data/processed/photos",
    )
    parser.add_argument(
        "--control_data_dir",
        type=str,
        default=None,
        help="控制图像目录(边缘/线稿)，如 data/processed/edges",
    )
    parser.add_argument(
        "--captions_file",
        type=str,
        default=None,
        help="可选 captions CSV/JSONL (filename, caption)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_out",
        help="输出目录，用于保存 safetensors 权重",
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help="可选已有 LoRA safetensors，resume 使用",
    )

    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=100,
        help="CPU 很慢，建议初版先设为几十~几百步",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=None, help="LoRA alpha 缩放")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker 数；Windows+CPU 建议为 0",
    )
    parser.add_argument("--flip_prob", type=float, default=0.0)
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=None,
        help="没有 caption 时的提示模板，例如 'an anime portrait of {filename}'",
    )
    parser.add_argument(
        "--train_controlnet_lora",
        action="store_true",
        help="是否给 ControlNet 也注入并训练 LoRA",
    )
    parser.add_argument(
        "--te_target_names",
        nargs="*",
        default=["q_proj", "k_proj", "v_proj", "out_proj", "proj"],
        help="text encoder 中要替换的 Linear 名称子串",
    )
    parser.add_argument(
        "--unet_target_names",
        nargs="*",
        default=["to_q", "to_k", "to_v", "to_out", "proj_out", "proj_in"],
        help="UNet 中要替换的 Linear 名称子串",
    )
    parser.add_argument(
        "--cn_target_names",
        nargs="*",
        default=["to_q", "to_k", "to_v", "to_out", "proj"],
        help="ControlNet 中要替换的 Linear 名称子串",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)