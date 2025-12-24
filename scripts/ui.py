import streamlit as st
import subprocess
import os
from pathlib import Path
import shutil
from PIL import Image

# ================= 路径配置 =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

GENERATE_SCRIPT = PROJECT_ROOT / "scripts" / "generate.py"
TEMP_INPUT_DIR = PROJECT_ROOT / "temp_web_inputs"
TEMP_OUTPUT_DIR = PROJECT_ROOT / "temp_web_outputs"

TEMP_INPUT_DIR.mkdir(exist_ok=True)
TEMP_OUTPUT_DIR.mkdir(exist_ok=True)

# ================= 页面配置 =================
st.set_page_config(
    page_title="Stable Diffusion LoRA UI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= Sidebar =================
with st.sidebar:
    st.header("⚙️ 模型设置")

    base_model = st.text_input(
        "Base Model",
        value=str(PROJECT_ROOT / "hub" / "models--runwayml--stable-diffusion-v1-5")
    )

    device = st.selectbox("Device", ["cuda", "cpu", "mps"], index=0)

    sampler = st.selectbox(
        "Sampler",
        ["default", "ddim", "euler", "euler_a", "dpmsolver"],
        index=4
    )

    use_autoprompt = st.checkbox(
        "启用 AutoPrompt",
        value=False,
        help="启用后由模型自动生成提示词"
    )

    st.markdown("---")
    st.subheader("🧩 LoRA 设置")

    lora_paths_input = st.text_area(
        "LoRA 文件路径（每行一个）",
        value=str(PROJECT_ROOT / "lora" / "lora_final.safetensors"),
        height=80
    )

    # 解析 LoRA 列表
    lora_paths = [x.strip() for x in lora_paths_input.splitlines() if x.strip()]

    st.markdown("**LoRA 权重（线性融合）**")

    lora_weights = []

    if len(lora_paths) == 0:
        st.info("未指定 LoRA，将使用基础模型推理")

    elif len(lora_paths) == 1:
        # 单 LoRA：简单滑条
        w = st.slider(
            f"Weight: {Path(lora_paths[0]).name}",
            min_value=0.0,
            max_value=1.5,
            value=1.0,
            step=0.05
        )
        lora_weights = [w]

    else:
        # 多 LoRA：逐个权重滑条（真实线性融合）
        for i, lp in enumerate(lora_paths):
            w = st.slider(
                f"[{i}] {Path(lp).name}",
                min_value=0.0,
                max_value=1.5,
                value=1.0,
                step=0.05
            )
            lora_weights.append(w)

    apply_lora_cnet = st.checkbox(
        "Apply LoRA to ControlNet",
        value=True
    )

    st.markdown("---")
    st.subheader("🕸 ControlNet")

    controlnet_paths_input = st.text_area(
        "ControlNet（每行一个）",
        value="lllyasviel/sd-controlnet-canny",
        height=60
    )

    controlnet_paths = [
        x.strip() for x in controlnet_paths_input.splitlines() if x.strip()
    ]

# ================= 主界面 =================
st.title("🎨 Stable Diffusion · LoRA Generator")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🖼 输入与提示词")

    uploaded_file = st.file_uploader(
        "上传原图（img2img）",
        type=["png", "jpg", "jpeg", "webp"]
    )

    control_image_file = st.file_uploader(
        "ControlNet 参考图（可选）",
        type=["png", "jpg", "jpeg", "webp"]
    )

    prompt = st.text_area(
        "Positive Prompt",
        value="chinese traditional style anime, hanfu, ink-wash painting elements, ultra detailed face, finely detailed eyes and eyelashes, crisp lineart, intricate hair strands, masterpiece, extremely detailed",
        height=90,
        disabled=use_autoprompt
    )

    negative_prompt = st.text_area(
        "Negative Prompt",
        value="低分辨率, 模糊, 透视错误, 畸形手, 畸形眼睛, 错位, 粗糙, 像素化, 现代服饰, 现实风格, 不协调背景",
        height=70
    )

    with st.expander("🔧 高级参数", expanded=True):
        c1, c2 = st.columns(2)
        width = c1.number_input("Width", 512, step=64)
        height = c2.number_input("Height", 768, step=64)

        steps = c1.slider("Steps", 10, 80, 28)
        guidance = c2.slider("CFG", 1.0, 20.0, 7.5)
        strength = st.slider("Denoise Strength", 0.0, 1.0, 0.6)

    generate_btn = st.button("🚀 Generate", use_container_width=True)

# ================= 生成逻辑 =================
with col2:
    st.subheader("🧪 生成结果")

    result_box = st.empty()
    log_box = st.empty()

    if generate_btn:
        if not uploaded_file:
            st.error("请先上传图片")
        else:
            # 保存输入图像
            input_path = TEMP_INPUT_DIR / uploaded_file.name
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # ControlNet 输入
            control_arg = []
            if control_image_file:
                cpath = TEMP_INPUT_DIR / f"control_{control_image_file.name}"
                with open(cpath, "wb") as f:
                    f.write(control_image_file.getbuffer())
                control_arg = ["--control-input", str(cpath)]

            # 清理输出目录
            shutil.rmtree(TEMP_OUTPUT_DIR, ignore_errors=True)
            TEMP_OUTPUT_DIR.mkdir(exist_ok=True)

            # 构建命令
            cmd = [
                "python", str(GENERATE_SCRIPT),
                "--model", base_model,
                "--input", str(input_path),
                "--output", str(TEMP_OUTPUT_DIR),
                "--negative", negative_prompt,
                "--device", device,
                "--width", str(width),
                "--height", str(height),
                "--steps", str(steps),
                "--guidance", str(guidance),
                "--strength", str(strength),
                "--mode", "single"
            ]

            if not use_autoprompt:
                cmd += ["--prompt", prompt]
            else:
                cmd.append("--auto-prompt")

            if sampler != "default":
                cmd += ["--sampler", sampler]

            if lora_paths:
                cmd += ["--lora", *lora_paths]
                cmd += ["--lora-weights", *[str(w) for w in lora_weights]]

            if controlnet_paths:
                cmd += ["--controlnets", *controlnet_paths]

            if apply_lora_cnet:
                cmd.append("--apply-lora-to-controlnet")

            cmd += control_arg

            result_box.info("⏳ 正在生成，请稍候...")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            logs = ""
            for line in process.stdout:
                logs += line
                log_box.code("\n".join(logs.splitlines()[-6:]))

            if process.wait() == 0:
                images = list(TEMP_OUTPUT_DIR.glob("*.png"))
                if images:
                    img = Image.open(max(images, key=os.path.getmtime))
                    result_box.image(img, use_container_width=True)
                else:
                    result_box.error("未找到输出图片")
            else:
                result_box.error("生成失败")
                st.text(logs)

# ================= Footer =================
st.markdown("---")
st.caption(
    "支持多 LoRA 线性融合（权重滑条）、可选 AutoPrompt 与 Sampler。"
)