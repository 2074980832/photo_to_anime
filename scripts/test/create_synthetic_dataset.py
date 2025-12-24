# create_synthetic_dataset.py
"""
在本地生成合成测试数据（placeholder 图像 + metadata.csv）。
默认生成 n_originals * variants_per_original >= 150 条记录。
生成的目录结构：
exp_dir/
  originals/
    orig_0001.jpg ...
  generated/
    gen_0001_v1_photo_to_anime.png ...
  metadata.csv  # 相对路径（exp_dir 下）
使用示例:
python create_synthetic_dataset.py --exp_dir exp_lora_anime --n_originals 50 --variants 3
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import csv
import os
import math
from datetime import datetime

PROMPTS = [
    "国风少女，汉服，精致线描，水彩质感, 素雅",
    "传统汉服人像，油画感，柔和光影，清雅",
    "古风仕女，流动长裙，飘逸，细腻笔触",
    "anime style, Chinese traditional clothing, elegant, delicate",
    "portrait of a young woman in hanfu, intricate hairpin, soft lighting",
    "国风插画，人物半身像，淡雅配色，漫画风格",
    "wuxia heroine, flowing robes, ink wash texture, dramatic pose",
    "anime girl in hanfu, ornate patterns, cinematic lighting",
]

def make_placeholder_image(path: Path, text: str, size=(512,512), bgcolor=(240,235,230)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', size, color=bgcolor)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    # 自动换行
    lines = []
    max_chars = 24
    for i in range(0, len(text), max_chars):
        lines.append(text[i:i+max_chars])

    # 使用 Pillow >=10 的 textbbox API
    bbox = draw.textbbox((0, 0), lines[0], font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    total_h = h * len(lines)
    x = (size[0] - w) // 2
    y = (size[1] - total_h) // 2

    for i, line in enumerate(lines):
        draw.text((x, y + i*h), line, fill=(30,30,30), font=font)

    img.save(path, format='PNG')


def generate_metadata(exp_dir: Path, n_originals: int, variants: int, model_tag: str):
    originals_dir = exp_dir / "originals"
    generated_dir = exp_dir / "generated"
    originals_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    records = []
    rng = random.Random(12345)  # deterministic

    # Create originals
    for i in range(1, n_originals+1):
        orig_fn = f"orig_{i:04d}.png"
        orig_path = originals_dir / orig_fn
        if not orig_path.exists():
            make_placeholder_image(orig_path, f"ORIG {i:04d}")
        # create variants
        for v in range(1, variants+1):
            gen_fn = f"gen_{i:04d}_v{v}_{model_tag}.png"
            gen_path = generated_dir / gen_fn
            # Placeholder content indicates which model & seed
            seed = rng.randint(1000, 999999)
            prompt = rng.choice(PROMPTS)
            # Compose a short label to write on image
            label = f"GEN {i:04d} v{v}\\n{model_tag}\\nseed:{seed}"
            if not gen_path.exists():
                make_placeholder_image(gen_path, label)
            # inference time simulate between 0.8s - 3.2s
            inference_time = round(rng.uniform(0.8, 3.2), 3)
            records.append({
                "original_image_path": str(Path("originals") / orig_fn).replace("\\", "/"),
                "generated_image_path": str(Path("generated") / gen_fn).replace("\\", "/"),
                "prompt": prompt,
                "seed": seed,
                "inference_time": inference_time
            })

    # ensure we have at least 150 rows
    if len(records) < 150:
        raise ValueError(f"Generated only {len(records)} rows; increase n_originals or variants to reach >=150")

    # take exactly 150 entries (first 150)
    records = records[:150]

    # write metadata.csv
    csv_path = exp_dir / "metadata.csv"
    with csv_path.open("w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["original_image_path","generated_image_path","prompt","seed","inference_time"])
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"[{datetime.now().isoformat()}] Created {len(records)} metadata rows at {csv_path}")
    return csv_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, help="experiment directory to create (contains originals/ generated/ metadata.csv)")
    parser.add_argument("--n_originals", type=int, default=50, help="number of distinct original images (default 50)")
    parser.add_argument("--variants", type=int, default=3, help="generated variants per original (default 3) => 50*3=150")
    parser.add_argument("--model_tag", type=str, default="photo_to_anime", help="tag to include in generated filenames")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    csv_path = generate_metadata(exp_dir, args.n_originals, args.variants, args.model_tag)
    print("Done. To run evaluation:")
    print(f"  python evaluate.py --exp_dir {exp_dir} --output_dir eval_results --fid_ref_dir path/to/anime_reference --baseline_dir path/to/baseline")

if __name__ == "__main__":
    main()
