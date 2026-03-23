<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="asset/CutClaw_dark.png" />
  <source media="(prefers-color-scheme: light)" srcset="asset/Cutclaw_light.png" />
  <img src="asset/Cutclaw_light.png" alt="CutClaw teaser" width="50%" />
</picture>

## 🦞CutClaw: Agentic Hours-Long Video Editing via Music Synchronization

**🎬 Your personal editor for turning hours of footage into cinematic montages.**

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2506.08412)
[![GitHub Stars](https://img.shields.io/github/stars/GVCLab/CutClaw?style=social)](https://github.com/GVCLab/CutClaw)

<p align="center">
  <img src="https://img.shields.io/badge/🎞️_Hours--Long_Footage-1f6feb?style=flat-square" alt="Hours-Long Footage" />
  <img src="https://img.shields.io/badge/🎵_Music--Beat_Sync-00b894?style=flat-square" alt="Music Beat Sync" />
  <img src="https://img.shields.io/badge/✍️_Instruction--Following-f59f00?style=flat-square" alt="Instruction Following" />
  <img src="https://img.shields.io/badge/🖱️_One--Click_Editing-e17055?style=flat-square" alt="One-Click Editing" />
  <img src="https://img.shields.io/badge/🔌_LiteLLM--Powered-6c5ce7?style=flat-square" alt="LiteLLM Powered" />
</p>

<p>
	<a href="readme.md"><img src="https://img.shields.io/badge/English-1a1a2e?style=for-the-badge"></a>
    <a href="readme_zh.md"><img src="https://img.shields.io/badge/中文版-1a1a2e?style=for-the-badge"></a>
</p>

[Overview](#-overview) • [Features](#-key-features) • [Gallery](#️-gallery) • [Quick Start](#-quick-start) • [Troubleshooting](#️-troubleshooting) • [Citation](#-citation)

</div>

---

## 💡 Overview

CutClaw is an end-to-end editing system for long-form footage + music.

It first deconstructs raw video/audio into structured captions, then uses a multi-agent pipeline to plan shots (`shot_plan`), select clip timestamps (`shot_point`), and validate final quality before rendering.

![CutClaw Pipeline](asset/method.png)

---

## ✨ Key Features

<table align="center" width="100%" style="border: none; table-layout: fixed;">
<tr>
<td width="25%" align="center" valign="top" style="padding: 16px;">

### 🎬 **One-Click Deconstruction**

<img src="https://img.shields.io/badge/LONG--FORM%20PROCESSING-4c6ef5?style=for-the-badge" alt="Long-Form Processing" />

Effortlessly transforms hours-long raw video and audio into structured, searchable assets with a single click.

</td>
<td width="25%" align="center" valign="top" style="padding: 16px;">

### 🎯 **Instruction Control**

<img src="https://img.shields.io/badge/TEXT%20TO%20EDIT-f59f00?style=for-the-badge" alt="Text to Edit" />

Requires only one text instruction to steer the editing style—easily generating fast-paced character montages or slow-paced emotional narratives.

</td>
<td width="25%" align="center" valign="top" style="padding: 16px;">

### 📱 **Smart Auto-Cropping**

<img src="https://img.shields.io/badge/SMART%20ADAPTATION-12b886?style=for-the-badge" alt="Smart Adaptation" />

Content-aware cropping automatically identifies core subjects and adjusts aspect ratios to fit various social platforms.

</td>
<td width="25%" align="center" valign="top" style="padding: 16px;">

### 🎵 **Music-Aware Sync**

<img src="https://img.shields.io/badge/AUDIO%20SYNC-e64980?style=for-the-badge" alt="Audio Sync" />

Extracts musical beats and energy signals to build rhythm-aware cuts that perfectly match the music's pacing.

</td>
</tr>
</table>

---


## 🖼️ Gallery（remember to turn on the audio）

<table width="100%">
<tr>
<td align="center" width="33%">
  <video src="https://github.com/user-attachments/assets/6e5d6ce8-2fd6-4acf-92a4-620784d56bca" controls width="100%"></video>
</td>
<td align="center" width="33%">
  <video src="https://github.com/user-attachments/assets/5fa41312-786b-4f63-afe3-abedf7e03e05" controls width="100%"></video>
</td>
<td align="center" width="33%">
  <video src="https://github.com/user-attachments/assets/024e6fad-b154-4864-80fe-601e9e9b56c0" controls width="100%"></video>
</td>
</tr>
</table>

<table width="100%">
<tr>
<td align="center" width="33%">
  <video src="https://github.com/user-attachments/assets/f87c7755-f777-4802-9f59-ab851a4b7881" controls width="100%"></video>
</td>
<td align="center" width="33%">
  <video src="https://github.com/user-attachments/assets/0dde3dc0-440b-4e18-82b2-970a1ee11fa5" controls width="100%"></video>
</td>
<td align="center" width="33%">
  <video src="https://github.com/user-attachments/assets/68f635d7-446e-4f3c-b8ad-a0d0baed9e7b" controls width="100%"></video>
</td>
</tr>
</table>

<table width="100%">
<tr>
<td align="center" width="33%">
  <video src="https://github.com/user-attachments/assets/1c55d0df-5811-432b-a6e8-9458e102dd96" controls width="100%"></video>
</td>
<td align="center" width="33%">
  <video src="https://github.com/user-attachments/assets/05183151-c4c5-455d-97bf-3cf6f4c6de72" controls width="100%"></video>
</td>
<td align="center" width="33%">
  <video src="https://github.com/user-attachments/assets/427ecd8b-c3ff-471c-bd39-d64fd76dfc79" controls width="100%"></video>
</td>
</tr>
</table>

----

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/GVCLab/CutClaw.git
cd CutClaw
conda create -n CutClaw python=3.12
conda activate CutClaw
pip install -r requirements.txt
```

> We strongly recommend the GPU-accelerated Decord/NVDEC build for faster video decoding. Build from [source](https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source).

### 2. Add your files

```
resource/
├── video/      ← put your .mp4 / .mkv here
├── audio/      ← put your .mp3 / .wav here
└── subtitle/   ← optional .srt (skips ASR, saves time)
```

### 3. Run

**UI (recommended)**

```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser. (*If `http://localhost:8501` does not work well, try `http://127.0.0.1:8501`)

![CutClaw UI demo](asset/UI.png)

> Place your footage in the paths above, then you can directly select those files in the UI.

Model selection guidance:

- **Video model**
  - **Role**: shot/scene understanding and visual captioning.
  - **Recommended**: Gemini-3, Qwen3.5, GPT-5.3

- **Audio model**
  - **Role**: ASR plus music-structure parsing (beat/downbeat, pitch, energy) for music-aware segmentation.
  - **Recommended**: Gemini-3

- **Agent model**
  - **Role**: drives the Screenwriter + Editor + Reviewer loop to generate `shot_plan` and `shot_point`.
  - **Recommended**: MiniMax-2.7, Kimi-2.5, Claude-4.5

We leverage `LiteLLM` as the api manager gateway, the typical Model name is e.g. 'openai/MiniMax-2.7' which means using openai protocol to call the given model, more information see [LiteLLM documents](https://github.com/BerriAI/litellm).


<details>
<summary><strong>CLI (advanced)</strong></summary>

```bash
python local_run.py \
  --Video_Path "resource/video/xxxx.mp4" \
  --Audio_Path "resource/audio/xxxx.mp3" \
  --Instruction "xxxx"
```

<details>
<summary>Common config overrides</summary>

Any `src/config.py` parameter can be overridden with `--config.PARAM_NAME VALUE`.

| Parameter | Default | Effect |
|---|---|---|
| `VIDEO_PATH` | `"resource/video/The_Dark_Knight.mkv"` | Default input video path used by UI remembered inputs |
| `AUDIO_PATH` | `"resource/audio/Way_Down_We_Go.mp3"` | Default input audio path used by UI remembered inputs |
| `INSTRUCTION` | `"Joker's crazy that want to change the world."` | Default editing instruction prompt |
| `ASR_BACKEND` | `"litellm"` | ASR engine (`litellm` cloud or `whisper_cpp` local) |
| `VIDEO_FPS` | `2` | Sampling FPS for preprocessing |
| `MAIN_CHARACTER_NAME` | `"Joker"` | Protagonist name for character-focused edits |
| `AUDIO_MIN_SEGMENT_DURATION` | `3.0` | Minimum beat segment duration (seconds) |
| `AUDIO_MAX_SEGMENT_DURATION` | `5.0` | Maximum beat segment duration (seconds) |
| `AUDIO_DETECTION_METHODS` | `["downbeat", "pitch", "mel_energy"]` | Audio keypoint detection methods |
| `PARALLEL_SHOT_MAX_WORKERS` | `4` | Parallel shot selection workers |

Example:

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



Then render manually:

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



## 🛠️ Troubleshooting

**Very slow runtime**

1. **API latency** — the pipeline sends a large number of concurrent requests to vision/language APIs. Speed is heavily dependent on your API provider's response time and rate limits.
2. **First-run Footage Deconstruction** — the first time you process a video, shot detection, captioning, ASR, and scene analysis all run from scratch. This is a one-time cost per video; subsequent edits with the same footage reuse the cached results and are much faster.
3. **GPU acceleration** — a CUDA-capable GPU significantly speeds up video decoding and encoding. We recommend building Decord with NVDEC support (see Install section).
4. **Video codec compatibility** — if the pipeline appears to hang during video-related steps, the source video's encoding may be the cause. In our testing, videos encoded with `libx264` worked reliably.





## ⭐ Citation
If you find PersonaLive useful for your research, welcome to cite our work using the following BibTeX:
<!-- ```bibtex
@article{li2025personalive,
  title={PersonaLive! Expressive Portrait Image Animation for Live Streaming},
  author={Li, Zhiyuan and Pun, Chi-Man and Fang, Chen and Wang, Jue and Cun, Xiaodong},
  journal={arXiv preprint arXiv:2512.11253},
  year={2025}
}
``` -->

