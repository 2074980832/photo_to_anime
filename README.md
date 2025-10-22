# photo_to_anime

## 🎯 项目目标
将现实人物照片转换为具有**国风特色（Chinese traditional / Hanfu-like）**的动漫风格图像（Photo → Anime Style Transfer）。
技术栈：Stable Diffusion (Animagine-XL) + ControlNet（结构/姿态约束）+ LoRA（轻量风格微调）。目标是在保留人物内容（轮廓、姿态、五官）前提下，让生成结果呈现明显且稳定的国风动漫视觉特征。

---

## 👥 团队成员
- 组长：czh  
- 模型加载与优化组：fyh、yzy  
- 推理展示与交互组：czh、lrb  
- 风格评估与报告组：lpz、bky  

---

## 🧩 数据策略

人像（Photo）：CelebAMask-HQ（使用其裁剪/对齐后的高质量人像作为 photo 域）

动漫（Anime）：Anime Face Dataset（Kaggle）作为基础二次元数据域

国风补充集（anime_guofeng）：组内额外收集数百至一千张国风/汉服/国画风格动漫或插画，用于 LoRA 微调（增强辨识度）

数据策略：采用**非成对（unpaired）**数据 — 使用大量通用动漫图建立基础生成能力，再用少量国风样本训练 LoRA 以强化特色风格。

### 🧩 数据目录规范
data/
├── raw/
│   ├── photos/             # CelebAMask-HQ 原始/裁剪人像
│   ├── anime/              # Anime Face Dataset 原始动漫图
│   └── anime_guofeng/      # 收集的国风风格图（少量）
└── processed/
    ├── photos/             # 预处理后人像（512×512 等）
    ├── anime/              # 预处理后动漫图
    └── edges/              # ControlNet 所需边缘/线稿（与 photos 对应或独立）

---

## 🧠 技术路线

### 🔹 总体流程

输入：现实人物照片
↓
图像预处理（边缘/姿态提取）
↓
Stable Diffusion + ControlNet/LoRA 推理
↓
输出：动漫风格人物图像


### 🔹 技术要点
| 模块 | 方法 / 工具 | 功能 |
|------|---------------|------|
| 基础模型 | Animagine-XL-4.0（Stable Diffusion 系列） | 生成动漫风格图像 |
| 结构保持 | ControlNet | 保持姿态、轮廓、构图一致 |
| 风格微调 | LoRA | 增强特定动漫风格特征 |
| 推理接口 | Diffusers Pipeline + Torch | 实现本地加载与批量生成 |
| 展示交互 | Gradio / Streamlit | 用户界面输入 prompt、查看结果 |
| 性能评估 | SSIM / CLIPScore / 耗时统计 | 分析保真度与风格质量 |

---

## ⚙️ 项目结构

photo_to_anime/
├── data/ # 数据目录（raw / processed）
├── models/ # Animagine 模型文件与权重
├── outputs/ # 生成图像结果
├── logs/ # 日志记录
├── scripts/ # 核心脚本
├── requirements.txt # 依赖库
└── README.md # 项目说明

## 🧰 环境依赖

python >= 3.10
torch >= 2.0
diffusers
transformers
safetensors
gradio
opencv-python