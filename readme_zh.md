<div align="center">

<img src="asset/teaser.png" alt="CutClaw teaser" width="50%" />

# 基于音乐同步的智能长视频剪辑系统

**🎬 你的个人剪辑师——将数小时素材一键打造成电影级蒙太奇。**

[![arXiv](https://img.shields.io/badge/arXiv-论文-b31b1b.svg)](https://arxiv.org/abs/2506.08412)
[![GitHub Stars](https://img.shields.io/github/stars/GVCLab/CutClaw?style=social)](https://github.com/GVCLab/CutClaw)

<p align="center">
  <img src="https://img.shields.io/badge/🎞️_长视频处理-1f6feb?style=flat-square" alt="长视频处理" />
  <img src="https://img.shields.io/badge/🎵_音乐节拍同步-00b894?style=flat-square" alt="音乐节拍同步" />
  <img src="https://img.shields.io/badge/✍️_指令跟随-f59f00?style=flat-square" alt="指令跟随" />
  <img src="https://img.shields.io/badge/🖱️_一键剪辑-e17055?style=flat-square" alt="一键剪辑" />
  <img src="https://img.shields.io/badge/🔌_LiteLLM驱动-6c5ce7?style=flat-square" alt="LiteLLM驱动" />
</p>

<p>
  <a href="readme.md"><img src="https://img.shields.io/badge/English-1a1a2e?style=for-the-badge"></a>
    <a href="readme_zh.md"><img src="https://img.shields.io/badge/中文版-1a1a2e?style=for-the-badge"></a>
</p>

[概述](#-概述) • [核心功能](#-核心功能) • [效果展示](#️-效果展示) • [快速开始](#-快速开始) • [常见问题](#️-常见问题) • [引用](#-引用)

</div>

---

## 💡 概述

CutClaw 是一个面向长视频素材与音乐的端到端自动剪辑系统。

它首先将原始视频和音频解析为结构化描述，再通过多智能体流水线完成镜头规划（`shot_plan`）、片段时间戳选取（`shot_point`）及质量验证，最终渲染输出成片。

![CutClaw Pipeline](asset/method.png)

---

## ✨ 核心功能

<table align="center" width="100%" style="border: none; table-layout: fixed;">
<tr>
<td width="25%" align="center" style="vertical-align: top; padding: 16px;">

### 🎬 **一键素材解析**

<img src="https://img.shields.io/badge/长视频处理-4c6ef5?style=for-the-badge" alt="长视频处理" />

只需一键，即可将数小时的原始视频和音频转化为结构化、可检索的素材库。

</td>
<td width="25%" align="center" style="vertical-align: top; padding: 16px;">

### 🎯 **自然语言指令控制**

<img src="https://img.shields.io/badge/指令跟随-f59f00?style=for-the-badge" alt="指令跟随" />

只需一句文字指令即可主导剪辑风格——既能生成快节奏人物混剪，也能输出慢节奏情感叙事。

</td>
<td width="25%" align="center" style="vertical-align: top; padding: 16px;">

### 📱 **智能自动裁剪**

<img src="https://img.shields.io/badge/智能适配-12b886?style=for-the-badge" alt="智能适配" />

内容感知裁剪自动识别画面主体，并按各平台比例进行智能调整。

</td>
<td width="25%" align="center" style="vertical-align: top; padding: 16px;">

### 🎵 **音乐感知同步**

<img src="https://img.shields.io/badge/音频同步-e64980?style=for-the-badge" alt="音频同步" />

提取音乐节拍与能量信号，构建与音乐节奏完美契合的剪切点。

</td>
</tr>
</table>

---


## 🖼️ 效果展示

<table width="100%">
<tr>
<td align="center" width="33%">
  <video src="https://github.com/user-attachments/assets/0eac0a2c-05ec-4eb9-b540-2752e9c35289" controls width="100%"></video>
</td>
<td align="center" width="33%">
  <video src="https://github.com/user-attachments/assets/0e191afb-aea7-4fcf-98d2-7b4b545f1a89" controls width="100%"></video>
</td>
<td align="center" width="33%">
  <video src="https://github.com/user-attachments/assets/59ccdd42-b4c8-4031-aa2c-3d7523b19024" controls width="100%"></video>
</td>
</tr>
</table>

<table width="100%">
<tr>
<td align="center" width="33%">
  <video src="https://github.com/user-attachments/assets/e41da312-9c20-4796-a600-a9f4534a7cd8" controls width="100%"></video>
</td>
<td align="center" width="33%">
  <video src="https://github.com/user-attachments/assets/c2212275-2a5f-42f5-9841-34e2573c8835" controls width="100%"></video>
</td>
<td align="center" width="33%">
  <video src="https://github.com/user-attachments/assets/1eb71636-f6ee-4a35-9a18-4f3e3f8eb5e7" controls width="100%"></video>
</td>
</tr>
</table>

<table width="100%">
<tr>
<td align="center" width="33.33%">
  <video src="https://github.com/user-attachments/assets/ac86d0c9-b652-4ec0-8527-1ebb0f465e7f" controls width="100%"></video>
</td>
<td align="center" width="33.33%">
  <video src="https://github.com/user-attachments/assets/970fd0c4-38c6-4674-8e5b-acfe4acba6ac" controls width="100%"></video>
</td>
<td align="center" width="33.33%">
  <video src="https://github.com/user-attachments/assets/8b02d26f-9b15-4961-b17a-74f915329021" controls width="100%"></video>
</td>
</tr>
</table>

----

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/GVCLab/CutClaw.git
cd CutClaw
conda create -n CutClaw python=3.12
conda activate CutClaw
pip install -r requirements.txt
```

> 强烈推荐使用支持 GPU 加速的 Decord/NVDEC 版本以加快视频解码速度，请参考[源码编译指南](https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source)。

### 2. 放入素材文件

```
resource/
├── video/      ← 放入 .mp4 / .mkv 视频文件
├── audio/      ← 放入 .mp3 / .wav 音频文件
└── subtitle/   ← 可选 .srt 字幕文件（跳过 ASR，节省时间）
```

### 3. 运行

**UI 界面（推荐）**

```bash
streamlit run app.py
```

在浏览器中打开 `http://localhost:8501`。（如无法访问，请尝试 `http://127.0.0.1:8501`）

![CutClaw UI demo](asset/UI.png)

> 将素材放入上述路径后，可直接在 UI 中选择对应文件。

模型选择建议：

- **视频模型**
  - **用途**：镜头/场景理解与视觉描述生成。
  - **推荐**：Gemini-3、Qwen3.5、GPT-5.3

- **音频模型**
  - **用途**：语音识别（ASR）及音乐结构分析（节拍/强拍、音高、能量），用于节拍感知分割。
  - **推荐**：Gemini-3

- **智能体模型**
  - **用途**：驱动编剧 + 剪辑 + 审阅智能体循环，生成 `shot_plan` 和 `shot_point`。
  - **推荐**：MiniMax-2.7、Kimi-2.5、Claude-4.5

系统使用 `LiteLLM` 作为 API 统一网关，模型名称格式如 `openai/MiniMax-2.7`，表示通过 OpenAI 协议调用该模型。更多信息请参阅 [LiteLLM 文档](https://github.com/BerriAI/litellm)。


<details>
<summary><strong>命令行模式（进阶）</strong></summary>

```bash
python local_run.py \
  --Video_Path "resource/video/xxxx.mp4" \
  --Audio_Path "resource/audio/xxxx.mp3" \
  --Instruction "xxxx"
```

<details>
<summary>常用配置覆盖参数</summary>

所有 `src/config.py` 中的参数均可通过 `--config.PARAM_NAME VALUE` 在运行时覆盖。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `VIDEO_PATH` | `"resource/video/The_Dark_Knight.mkv"` | 默认视频路径（UI 记忆输入） |
| `AUDIO_PATH` | `"resource/audio/Way_Down_We_Go.mp3"` | 默认音频路径（UI 记忆输入） |
| `INSTRUCTION` | `"Joker's crazy that want to change the world."` | 默认剪辑指令 |
| `ASR_BACKEND` | `"litellm"` | ASR 引擎（`litellm` 云端或 `whisper_cpp` 本地） |
| `VIDEO_FPS` | `2` | 预处理采样帧率 |
| `MAIN_CHARACTER_NAME` | `"Joker"` | 主角名称（角色聚焦剪辑） |
| `AUDIO_MIN_SEGMENT_DURATION` | `3.0` | 节拍片段最短时长（秒） |
| `AUDIO_MAX_SEGMENT_DURATION` | `5.0` | 节拍片段最长时长（秒） |
| `AUDIO_DETECTION_METHODS` | `["downbeat", "pitch", "mel_energy"]` | 音频关键点检测方法 |
| `PARALLEL_SHOT_MAX_WORKERS` | `4` | 并行镜头选择线程数 |

示例：

```bash
python local_run.py \
  --Video_Path "resource/video/xxxx.mp4" \
  --Audio_Path "resource/audio/xxxx.mp3" \
  --Instruction "xxxx" \
  --config.MAIN_CHARACTER_NAME "Batman" \
  --config.VIDEO_FPS 2 \
  --config.AUDIO_TOTAL_SHOTS 50
```

</details>

手动渲染：

```bash
python render/render_video.py \
  --shot-plan  "Output/<video_audio>/shot_plan_*.json" \
  --shot-json  "Output/<video_audio>/shot_point_*.json" \
  --video  "resource/video/xxxx.mp4" \
  --audio  "resource/audio/xxxx.mp3" \
  --output "output/final.mp4" \
  --crop-ratio "9:16" \
  --no-labels --render-hook-dialogue
```

</details>

---



## 🛠️ 常见问题

**运行速度很慢**

1. **API 延迟** —— 流水线会向视觉/语言 API 发送大量并发请求，速度很大程度上取决于 API 提供商的响应时间和速率限制。
2. **首次素材解析耗时长** —— 第一次处理某段视频时，镜头检测、描述生成、ASR 和场景分析均从头运行，这是每段视频的一次性开销。后续使用相同素材时会直接复用缓存，速度大幅提升。
3. **GPU 加速** —— 支持 CUDA 的 GPU 能显著加快视频解码和编码速度。推荐参考安装章节，使用支持 NVDEC 的 Decord 版本。
4. **视频编码兼容性** —— 若流水线在视频处理环节卡住，可能是源视频编码格式导致的。经测试，使用 `libx264` 编码的视频运行最稳定。

---

## ⭐ 引用

如果 CutClaw 对您的研究有所帮助，欢迎引用我们的工作：

<!-- ```bibtex
@article{li2025personalive,
  title={PersonaLive! Expressive Portrait Image Animation for Live Streaming},
  author={Li, Zhiyuan and Pun, Chi-Man and Fang, Chen and Wang, Jue and Cun, Xiaodong},
  journal={arXiv preprint arXiv:2512.11253},
  year={2025}
}
``` -->
