import streamlit as st
import subprocess
import os
from pathlib import Path
import shutil
from PIL import Image
import time

# --- 配置 ---
GENERATE_SCRIPT = "generate.py"  # 指向你的脚本路径
TEMP_INPUT_DIR = "temp_web_inputs"
TEMP_OUTPUT_DIR = "temp_web_outputs"

# 确保临时目录存在
os.makedirs(TEMP_INPUT_DIR, exist_ok=True)
os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="SD LoRA Generator UI", layout="wide")

# --- 侧边栏：模型配置 ---
with st.sidebar:
    st.header("⚙️ 模型设置")
    
    base_model = st.text_input(
        "Base Model Path / ID", 
        value="./hub/models--runwayml--stable-diffusion-v1-5",
        help="指向 diffusers 格式的模型目录或 HuggingFace ID"
    )
    
    device = st.selectbox("Device", ["cuda", "cpu", "mps"], index=0)
    
    st.markdown("---")
    st.subheader("LoRA 配置")
    
    # 动态 LoRA 输入
    lora_paths_input = st.text_area(
        "LoRA Paths (每行一个)", 
        value="./lora/lora_final.safetensors",
        help="输入 .safetensors 文件的路径"
    )
    
    lora_weights_input = st.text_input(
        "LoRA Weights (空格分隔)", 
        value="1.0",
        help="对应上面的 LoRA，例如: 0.8 0.5"
    )
    
    apply_lora_cnet = st.checkbox("Apply LoRA to ControlNet", value=True)
    
    st.markdown("---")
    st.subheader("ControlNet 配置")
    controlnet_paths = st.text_area(
        "ControlNet Paths (每行一个)", 
        value="lllyasviel/sd-controlnet-canny",
        help="HuggingFace ID 或本地路径"
    )

# --- 主界面 ---
st.title("🎨 Stable Diffusion Generator")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. 图片输入 & 提示词")
    
    uploaded_file = st.file_uploader("上传原图 (img2img)", type=["png", "jpg", "jpeg", "webp"])
    
    control_image_file = st.file_uploader("上传 ControlNet 参考图 (可选)", type=["png", "jpg", "jpeg", "webp"], help="如果不传，默认使用原图")

    prompt = st.text_area("Positive Prompt", height=100, value="portrait of a girl, masterpiece, best quality")
    negative_prompt = st.text_area("Negative Prompt", height=80, value="lowres, bad anatomy, blurry, worst quality")

    with st.expander("高级参数设置 (Steps, CFG, Size)", expanded=True):
        c1, c2 = st.columns(2)
        width = c1.number_input("Width", value=512, step=64)
        height = c2.number_input("Height", value=768, step=64)
        
        steps = c1.slider("Steps", 10, 100, 28)
        guidance = c2.slider("Guidance Scale", 1.0, 20.0, 7.5)
        strength = st.slider("Denoising Strength", 0.0, 1.0, 0.6, help="重绘幅度，越大变化越大")

    generate_btn = st.button("🚀 生成图片", type="primary", use_container_width=True)

# --- 生成逻辑 ---
with col2:
    st.subheader("2. 生成结果")
    
    result_placeholder = st.empty()
    logs_placeholder = st.empty()

    if generate_btn:
        if not uploaded_file:
            st.error("❌ 请先上传一张图片！")
        else:
            # 1. 保存上传的图片到临时目录
            input_path = Path(TEMP_INPUT_DIR) / uploaded_file.name
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 处理 Control Image
            control_arg = []
            if control_image_file:
                c_path = Path(TEMP_INPUT_DIR) / f"control_{control_image_file.name}"
                with open(c_path, "wb") as f:
                    f.write(control_image_file.getbuffer())
                control_arg = ["--control-input", str(c_path)]

            # 2. 清理旧的输出
            if os.path.exists(TEMP_OUTPUT_DIR):
                shutil.rmtree(TEMP_OUTPUT_DIR)
            os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

            # 3. 构建命令行参数
            # 解析多行输入
            lora_list = [l.strip() for l in lora_paths_input.split('\n') if l.strip()]
            cnet_list = [c.strip() for c in controlnet_paths.split('\n') if c.strip()]
            weights_list = lora_weights_input.strip().split()

            cmd = [
                "python", GENERATE_SCRIPT,
                "--model", base_model,
                "--input", str(input_path),
                "--output", TEMP_OUTPUT_DIR,
                "--prompt", prompt,
                "--negative", negative_prompt,
                "--device", device,
                "--width", str(width),
                "--height", str(height),
                "--steps", str(steps),
                "--guidance", str(guidance),
                "--strength", str(strength),
                "--mode", "single" # 强制单图模式
            ]

            if lora_list:
                cmd.append("--lora")
                cmd.extend(lora_list)
            
            if weights_list:
                cmd.append("--lora-weights")
                cmd.extend(weights_list)

            if cnet_list:
                cmd.append("--controlnets")
                cmd.extend(cnet_list)

            if apply_lora_cnet:
                cmd.append("--apply-lora-to-controlnet")
            
            cmd.extend(control_arg)

            # 4. 执行命令并流式显示日志
            result_placeholder.info("正在初始化模型并生成...")
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1, 
                universal_newlines=True
            )
            
            logs = ""
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logs += output
                    # 简单显示最后几行日志
                    logs_placeholder.code("\n".join(logs.split('\n')[-5:]), language="bash")
            
            rc = process.poll()
            
            if rc == 0:
                # 5. 寻找并展示生成的图片
                generated_images = list(Path(TEMP_OUTPUT_DIR).glob("*.png"))
                if generated_images:
                    # 找到最新的图片
                    latest_img = max(generated_images, key=os.path.getctime)
                    image = Image.open(latest_img)
                    result_placeholder.image(image, caption="生成结果", use_container_width=True)
                    st.success(f"生成成功！耗时: {logs.split('it/s')[-1] if 'it/s' in logs else 'N/A'}")
                else:
                    result_placeholder.error("脚本运行成功，但未找到输出图片。")
            else:
                result_placeholder.error("生成失败，请检查参数或日志。")
                with st.expander("查看完整错误日志"):
                    st.text(logs)

# --- 页脚说明 ---
st.markdown("---")
st.markdown("*此界面是 generate.py 的前端封装，确保所有路径（模型、LoRA）相对于脚本运行位置是正确的。*")