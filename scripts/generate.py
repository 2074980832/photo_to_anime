#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SD1 img2img 生成脚本（与 train_lora.py 的 LoRA 格式兼容）
增强点：
 - 支持标准 diffusers repo（model_index.json）和 hub snapshots 布局
 - 手动构造 pipeline 时补全 scheduler 与 feature_extractor（避免缺参数错误）
 - 中文注释、single/multiple 模式、可选 --lora、不强制 --output
保存为 scripts/generate.py，从项目根运行示例：
  python scripts/generate.py --model ./hub/models--runwayml--stable-diffusion-v1-5
"""
from pathlib import Path
import argparse
import logging
import os
from typing import Dict, Optional, List

import torch
import torch.nn as nn
from PIL import Image

# diffusers / transformers 组件
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor
from safetensors.torch import load_file as safetensors_load

# ---------------- LoRA helper（与 train_lora.py 保持一致） ----------------
class LoRALinear(nn.Module):
    """把原始 nn.Linear 包装成 LoRA 占位层（与训练脚本一致）"""
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
            # 确保 LoRA 参数创建在与 orig 相同的 device（避免后续运算的 device mismatch）
            try:
                orig_device = next(orig.parameters()).device
            except StopIteration:
                orig_device = torch.device("cpu")
            self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features, device=orig_device))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, device=orig_device))
            nn.init.normal_(self.lora_A, std=0.01)
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        # 冻结原始层参数
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
    递归替换 module 中匹配的 nn.Linear 为 LoRALinear（返回替换数量）。
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


def load_lora_into_module(root_module: nn.Module, state: Dict[str, torch.Tensor], module_key_prefix: str):
    """
    将 safetensors 中保存的 LoRA 权重加载回 LoRALinear。
    module_key_prefix: "text_encoder." / "unet." / "controlnet."
    """
    for name, m in root_module.named_modules():
        if isinstance(m, LoRALinear) and getattr(m, "r", 0) > 0:
            keybase = f"{module_key_prefix}{name.replace('.', '_')}"
            a_key = f"{keybase}.lora_A"
            b_key = f"{keybase}.lora_B"
            s_key = f"{keybase}.scaling"
            if a_key in state and b_key in state:
                a = state[a_key]
                b = state[b_key]
                try:
                    a = a.to(m.lora_A.device)
                    b = b.to(m.lora_B.device)
                    if a.shape == m.lora_A.shape and b.shape == m.lora_B.shape:
                        m.lora_A.data.copy_(a)
                        m.lora_B.data.copy_(b)
                    else:
                        logging.warning(f"LoRA shape mismatch for {keybase}: saved {a.shape}/{b.shape}, model {m.lora_A.shape}/{m.lora_B.shape}")
                except Exception as e:
                    logging.warning(f"Failed to load LoRA {keybase}: {e}")
            if s_key in state:
                try:
                    m.scaling = float(state[s_key].item())
                except Exception:
                    pass


# ---------------- 辅助工具 ----------------
def detect_default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def detect_max_input_size(device: str):
    if device == "cpu" or not torch.cuda.is_available():
        return 512
    mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if mem <= 8:
        return 512
    elif mem <= 12:
        return 768
    else:
        return 1024


def resize_input_image(img: Image.Image, max_size: int):
    """按最大边长等比缩放输入图，避免 OOM"""
    w, h = img.size
    max_dim = max(w, h)
    if max_dim <= max_size:
        return img
    scale = max_size / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def list_image_files_in_dir(dirpath: Path) -> List[Path]:
    if not dirpath.exists() or not dirpath.is_dir():
        return []
    return sorted([p for p in dirpath.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]])


# ---------------- pipeline loader（增强：自动查找 snapshots 并补全 scheduler/feature_extractor） ----------------
def find_latest_snapshot_dir(base: Path) -> Optional[Path]:
    """
    在 base 下寻找 snapshots/<hash>/ 的最新子目录，兼容 hub/.../snapshots/<hash> 布局。
    """
    snaps_root = base / "snapshots"
    if snaps_root.exists() and snaps_root.is_dir():
        subs = [d for d in snaps_root.iterdir() if d.is_dir()]
        if subs:
            subs_sorted = sorted(subs, key=lambda p: p.name)
            return subs_sorted[-1]
    # 递归查找更深层次的 snapshots（容错）
    for p in base.rglob("snapshots"):
        parent = Path(p)
        subs = [d for d in parent.iterdir() if d.is_dir()]
        if subs:
            subs_sorted = sorted(subs, key=lambda q: q.name)
            return subs_sorted[-1]
    return None


def load_sd1_pipeline_from_path(model_path: Path, controlnet: Optional[ControlNetModel], dtype: torch.dtype):
    """
    支持：
     - 标准 diffusers repo（有 model_index.json / pipeline_config.json）：直接 from_pretrained
     - snapshot 布局（会自动查找 latest snapshot）：逐组件加载，并且**补全 scheduler 与 feature_extractor**
    """
    # 优先用 from_pretrained 处理标准仓库布局
    if (model_path / "model_index.json").exists() or (model_path / "pipeline_config.json").exists():
        if controlnet:
            return StableDiffusionControlNetImg2ImgPipeline.from_pretrained(str(model_path), controlnet=controlnet, torch_dtype=dtype, safety_checker=None)
        else:
            return StableDiffusionImg2ImgPipeline.from_pretrained(str(model_path), torch_dtype=dtype, safety_checker=None)

    # 尝试查找 snapshot 目录（hub/.../snapshots/<hash>）
    snapshot = find_latest_snapshot_dir(model_path)
    # 如果用户直接指定了 snapshot（model_path 本身就是 snapshot），也支持
    if snapshot is None and (model_path / "text_encoder").exists():
        snapshot = model_path

    if snapshot is None:
        raise RuntimeError(f"Model directory {model_path} 没有 model_index.json，也找不到 snapshots 下的快照。请传入标准 diffusers 目录或 snapshot。")

    # 期望 snapshot 下存在 text_encoder, tokenizer, unet, vae 等子目录
    te_dir = snapshot / "text_encoder"
    tokenizer_dir = snapshot / "tokenizer"
    unet_dir = snapshot / "unet"
    vae_dir = snapshot / "vae"

    # 一些 snapshot 可能命名为 tokenizer_2 等，做容错
    if not tokenizer_dir.exists():
        if (snapshot / "tokenizer_2").exists():
            tokenizer_dir = snapshot / "tokenizer_2"
        elif (snapshot / "tokenizer").exists():
            tokenizer_dir = snapshot / "tokenizer"

    if not (te_dir.exists() and tokenizer_dir.exists() and unet_dir.exists() and vae_dir.exists()):
        raise RuntimeError(f"Snapshot 目录 {snapshot} 缺少必要子目录（text_encoder / tokenizer / unet / vae），无法构造 SD1 pipeline。")

    # 逐组件加载：tokenizer, text_encoder, vae, unet
    tokenizer = CLIPTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
    text_encoder = CLIPTextModel.from_pretrained(str(te_dir))
    vae = AutoencoderKL.from_pretrained(str(vae_dir))
    unet = UNet2DConditionModel.from_pretrained(str(unet_dir))

    # 重要：构造一个合适的 scheduler（用于推理），若 snapshot 中没有 scheduler，则使用 SD1 常用 DDPMScheduler 默认参数
    scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    # 重要：构造 feature_extractor（pipeline 需要）。优先尝试从 transformers hub 下载常见 CLIP extractor；
    # 若下载失败（离线环境），退回到直接实例化 CLIPFeatureExtractor()（可用但可能不含预处理参数）
    try:
        feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        logging.warning(f"无法从网络加载 CLIPFeatureExtractor（{e}），将使用默认实例回退。")
        feature_extractor = CLIPFeatureExtractor()

    # 构造 pipeline（将我们手动加载的组件传入）
    if controlnet:
        pipe = StableDiffusionControlNetImg2ImgPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=feature_extractor,
        )
    else:
        pipe = StableDiffusionImg2ImgPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=feature_extractor,
        )
    return pipe


# ---------------- 主流程（不变，省略重复注释） ----------------
def main():
    parser = argparse.ArgumentParser(description="SD1 img2img generate script (single/multiple) - fixed scheduler/fe")
    parser.add_argument("--model", type=str, required=True, help="SD1 模型目录或 HF repo id（支持 snapshot 风格）")
    parser.add_argument("--controlnet", type=str, default=None, help="可选 ControlNet 模型目录/ID（SD1 兼容）")
    parser.add_argument("--mode", type=str, choices=["single", "multiple"], default="single", help="single (单张) 或 multiple (批量)")
    parser.add_argument("--input", type=str, default=None, help="输入文件或目录（覆盖默认输入目录）")
    parser.add_argument("--control-input", type=str, default=None, help="单张 control 图（single 模式或作为所有图的统一 control）")
    parser.add_argument("--control-dir", type=str, default=None, help="control 图目录（multiple 模式下按同名或索引匹配）")
    parser.add_argument("--output", type=str, default=None, help="输出目录（可选，缺省使用 generate_output_photos/<mode>）")
    parser.add_argument("--prompt", type=str, default="anime style, masterpiece, high quality, detailed")
    parser.add_argument("--negative", type=str, default="lowres, bad anatomy, blurry")
    parser.add_argument("--lora", type=str, nargs="+", default=None, help="可选 LoRA safetensors 路径（可传多个）")
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-input-size", type=int, default=None)
    parser.add_argument("--rank", type=int, default=4, help="注入 LoRA 时使用的 rank（与训练时一致）")
    parser.add_argument("--unet-target-names", nargs="*", default=["to_q", "to_k", "to_v", "to_out", "proj_out", "proj_in"])
    parser.add_argument("--te-target-names", nargs="*", default=["q_proj", "k_proj", "v_proj", "out_proj", "proj"])
    parser.add_argument("--num-images", type=int, default=None, help="如果输入是目录，可限制最多处理多少张")
    parser.add_argument("--cnet-type", type=str, default=None, help="若要自动生成 control 图（需要 controlnet_aux）: canny/depth/openpose/lineart/softedge/normal")
    args = parser.parse_args()

    # 日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    # 使用当前工作目录作为 repo root（便于在 repo 根运行 scripts/generate.py）
    repo_root = Path.cwd()
    log.info(f"运行目录（repo root）: {repo_root}")

    device = args.device if args.device else detect_default_device()
    log.info(f"使用设备: {device}")

    max_input_size = args.max_input_size if args.max_input_size else detect_max_input_size(device)
    log.info(f"最大输入边长（用于 resize）: {max_input_size}")

    # 默认输入/输出目录（相对于 repo_root）
    if args.mode == "single":
        default_input_dir = repo_root / "generate_input_photos" / "single"
        default_output_dir = repo_root / "generate_output_photos" / "single"
    else:
        default_input_dir = repo_root / "generate_input_photos" / "multiple"
        default_output_dir = repo_root / "generate_output_photos" / "multiple"

    # 准备输入图片列表（支持未提供 input 的默认目录）
    images: List[Path] = []
    if args.input:
        in_path = Path(args.input)
        if in_path.is_file():
            images = [in_path]
        elif in_path.is_dir():
            files = list_image_files_in_dir(in_path)
            if args.mode == "single":
                if not files:
                    raise RuntimeError(f"输入目录 {in_path} 没有图片。")
                images = [files[0]]
            else:
                images = files
        else:
            raise RuntimeError(f"输入路径 {in_path} 不存在。")
    else:
        files = list_image_files_in_dir(default_input_dir)
        if not files:
            raise RuntimeError(f"缺省输入目录 {default_input_dir} 没有图片，请传入 --input。")
        if args.mode == "single":
            images = [files[0]]
        else:
            images = files

    if args.num_images:
        images = images[: args.num_images]

    # control 图处理（single/multiple 情况）
    control_single_path: Optional[Path] = None
    control_dir: Optional[Path] = None
    if args.control_input:
        control_single_path = Path(args.control_input)
        if not control_single_path.exists():
            log.warning(f"control-input {control_single_path} 不存在，忽略。")
            control_single_path = None
    elif args.control_dir:
        control_dir = Path(args.control_dir)
        if not control_dir.exists() or not control_dir.is_dir():
            log.warning(f"control-dir {control_dir} 不存在或不是目录，忽略。")
            control_dir = None

    # 加载 ControlNet 模型（若指定）
    controlnet_model = None
    if args.controlnet:
        log.info(f"加载 ControlNet: {args.controlnet}")
        controlnet_model = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=torch.float16 if device == "cuda" else torch.float32)

    # ---------- 加载 pipeline（增强兼容性） ----------
    log.info("正在加载 SD1 pipeline（可能需要一点时间）...")
    model_path = Path(args.model)
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        pipe = load_sd1_pipeline_from_path(model_path, controlnet_model, dtype)
    except Exception as e:
        log.error(f"从 {model_path} 构造 pipeline 失败：{e}")
        raise

    # 优化（xformers / slicing）并移动到设备
    try:
        pipe.enable_xformers_memory_efficient_attention()
        log.info("已启用 xformers（若可用）")
    except Exception:
        log.debug("xformers 未安装或不可用")
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    pipe.to(device)

    # 注入 LoRA 占位（以便后面加载权重时能找到 LoRALinear）
    log.info("注入 LoRA 占位层（UNet / text encoder / 可选 ControlNet）...")
    te_count = replace_linear_with_lora(pipe.text_encoder, r=args.rank, target_names=args.te_target_names)
    unet_count = replace_linear_with_lora(pipe.unet, r=args.rank, target_names=args.unet_target_names)
    cn_count = 0
    if controlnet_model and hasattr(pipe, "controlnet"):
        cn_count = replace_linear_with_lora(pipe.controlnet, r=args.rank, target_names=args.unet_target_names)
    log.info(f"已注入 LoRA 占位: text_encoder={te_count}, unet={unet_count}, controlnet={cn_count}")

    # 加载 LoRA 权重（可选）
    if args.lora:
        merged_state: Dict[str, torch.Tensor] = {}
        for lpath in args.lora:
            if not os.path.exists(lpath):
                log.warning(f"LoRA 文件 {lpath} 不存在，跳过。")
                continue
            log.info(f"读取 LoRA safetensors: {lpath}")
            st = safetensors_load(lpath)
            merged_state.update(st)
        if merged_state:
            load_lora_into_module(pipe.text_encoder, merged_state, "text_encoder.")
            load_lora_into_module(pipe.unet, merged_state, "unet.")
            if controlnet_model and hasattr(pipe, "controlnet"):
                load_lora_into_module(pipe.controlnet, merged_state, "controlnet.")
            log.info("已加载 LoRA 权重（匹配的键已应用）")
        else:
            log.info("未找到有效 LoRA 权重，继续不使用 LoRA。")

    # 可选：controlnet_aux 自动生成 control 图
    cnet_detector = None
    if args.cnet_type:
        try:
            from controlnet_aux import CannyDetector, MidasDetector, OpenposeDetector, LineartDetector, PidiNetDetector, NormalBaeDetector
            ct = args.cnet_type.lower()
            if ct == "canny":
                cnet_detector = CannyDetector()
            elif ct == "depth":
                cnet_detector = MidasDetector()
            elif ct == "openpose":
                cnet_detector = OpenposeDetector()
            elif ct == "lineart":
                cnet_detector = LineartDetector()
            elif ct == "softedge":
                cnet_detector = PidiNetDetector()
            elif ct == "normal":
                cnet_detector = NormalBaeDetector()
            else:
                log.warning(f"未知 cnet-type: {args.cnet_type}，跳过自动预处理。")
                cnet_detector = None
            if cnet_detector:
                log.info(f"使用 controlnet_aux 的检测器: {args.cnet_type}")
        except Exception:
            log.warning("controlnet_aux 未安装或导入失败，自动 control 预处理不可用。")
            cnet_detector = None

    # 输出目录
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = repo_root / "generate_output_photos" / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    # 辅助：按同名或按索引查找 control 图（multiple 模式）
    def find_control_for_image(img_path: Path, idx: int, control_dir: Optional[Path], control_common: Optional[Path]) -> Optional[Image.Image]:
        if control_common:
            try:
                ci = Image.open(control_common).convert("RGB")
                return resize_input_image(ci, max_input_size)
            except Exception as e:
                log.warning(f"打开 control common {control_common} 失败: {e}")
                return None
        if control_dir:
            cand = control_dir / img_path.name
            if cand.exists():
                try:
                    ci = Image.open(cand).convert("RGB")
                    return resize_input_image(ci, max_input_size)
                except Exception as e:
                    log.warning(f"打开 control 图 {cand} 失败: {e}")
            files = list_image_files_in_dir(control_dir)
            if idx < len(files):
                try:
                    ci = Image.open(files[idx]).convert("RGB")
                    return resize_input_image(ci, max_input_size)
                except Exception as e:
                    log.warning(f"打开 control 图 {files[idx]} 失败: {e}")
        return None

    # 逐张生成
    counter = 1
    for i, img_path in enumerate(images):
        log.info(f"处理: {img_path}")
        img = Image.open(img_path).convert("RGB")
        img = resize_input_image(img, max_input_size)

        control_image = None
        if control_single_path:
            try:
                ci = Image.open(control_single_path).convert("RGB")
                control_image = resize_input_image(ci, max_input_size)
            except Exception as e:
                log.warning(f"打开 control-input {control_single_path} 失败: {e}")
                control_image = None
        elif control_dir:
            control_image = find_control_for_image(img_path, i, control_dir, None)
        elif cnet_detector is not None:
            try:
                control_image = cnet_detector(img)
            except Exception as e:
                log.warning(f"controlnet_aux detector 运行失败: {e}; 本张不使用 control 图。")
                control_image = None

        log.info("运行 pipeline ...")
        if hasattr(pipe, "controlnet") and control_image is not None:
            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative,
                image=img,
                control_image=control_image,
                strength=args.strength,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                width=args.width,
                height=args.height,
            )
        else:
            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative,
                image=img,
                strength=args.strength,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                width=args.width,
                height=args.height,
            )

        out_img = result.images[0]
        out_path = out_dir / f"{img_path.stem}_result_{counter}.png"
        out_img.save(out_path)
        log.info(f"已保存: {out_path}")
        counter += 1

    log.info("全部完成。")


if __name__ == "__main__":
    main()