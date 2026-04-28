import os

# ==============================================================================
# CutClaw 全局配置（初学者友好版本）
# ------------------------------------------------------------------------------
# 这个文件控制整个剪辑流水线：
# 1) 视频预处理（抽帧、镜头检测、场景分析）
# 2) 音频分析（节拍/能量/结构）
# 3) 智能体生成（编剧 + 编辑 + 审核）
#
# 推荐用法：
# - 实验时优先使用 CLI 覆盖，而不是直接修改默认值：
#   python local_run.py ... --config.PARAM_NAME VALUE
# - 示例：--config.VIDEO_FPS 1 --config.AUDIO_TOTAL_SHOTS 80
# ==============================================================================

# ------------------ UI 记住的输入 ------------------ #
# 当您在侧边栏更改字段时，这些会自动由 app 保存。
# Java 类比：相当于 public static final String，但 Python 中可以在运行时修改

VIDEO_PATH = "resource/video/IMG_0931.MOV"
# 默认视频路径（用于 UI 记住上次选择）
# Java 类比：public static final String VIDEO_PATH = "...";
AUDIO_PATH = ""
# 默认音频路径
INSTRUCTION = ""
# 默认剪辑指令
SRT_PATH = ""
# 默认字幕路径（可选，提供后可跳过 ASR）


# ------------------ 视频预处理 ------------------ #
# 这些参数控制视频的采样方式、输出位置和处理的內容量。
# Java 类比：这些是全局配置常量，影响整个项目的行为


VIDEO_DATABASE_FOLDER = "./Output/"
# 所有中间产物和输出 JSON 文件的根文件夹。
# 修改此值会影响整个项目的读写路径。
# Java 类比：相当于一个全局常量配置路径，类似 Spring 的 application.properties

VIDEO_RESOLUTION = 240
# 帧提取时的目标短边分辨率（保持宽高比）。
# - 更小：更快且内存占用更低，但细节较少
# - 更大：更好的细节，但更慢
# Java 类比：public static final int VIDEO_RESOLUTION = 240;

VIDEO_FPS = 2 
# 预处理期间的帧采样率（每秒帧数）。
# 典型范围：1~3。更高的 FPS 提供更精细的分析，但增加成本/时间。
# Java 类比：类似于视频处理的配置参数，决定采样密度

VIDEO_MAX_MINUTES = None
# 要处理的最大视频时长（分钟）。None 表示完整视频。
# 对于快速调试，设置为 3~10 通常很有帮助。
# Python 特有：None 相当于 Java 的 null

VIDEO_MAX_FRAMES = None if VIDEO_MAX_MINUTES is None else int(VIDEO_MAX_MINUTES * 60 * VIDEO_FPS)
# 自动计算的帧数上限（当设置了 VIDEO_MAX_MINUTES 时）。
# Java 等价写法：
# Integer VIDEO_MAX_FRAMES = (VIDEO_MAX_MINUTES == null) ? null : 
#     (int)(VIDEO_MAX_MINUTES * 60 * VIDEO_FPS);

VIDEO_SAVE_DEBUG_FRAMES = False
# 如果为 True，采样的调试帧会写入磁盘以便排查问题。
# 这会增加磁盘 I/O。
# Python 布尔值：True/False（首字母大写），Java 是 true/false


# ------------------ 镜头检测 ------------------ #
# 这些参数决定在哪里检测镜头边界。
# Java 类比：类似于视频处理库的配置参数

SHOT_DETECTION_FPS = 2.0
# 专门用于镜头检测的采样 FPS（独立于 VIDEO_FPS）。

VIDEO_TYPE = "film"
# 全局视频类型："film"（电影）或 "vlog"（博客）。
# 注意：local_run.py 会用 CLI 的 --type 覆盖这个值。

# Python 的条件赋值（类似 Java 的 if-else）
if VIDEO_TYPE == "film":
    SHOT_DETECTION_THRESHOLD = 3.0
    SHOT_DETECTION_MIN_SCENE_LEN = 3
elif VIDEO_TYPE == "vlog":
    SHOT_DETECTION_THRESHOLD = 1.5
    SHOT_DETECTION_MIN_SCENE_LEN = 45
else:
    # 回退默认值：使用电影设置。
    SHOT_DETECTION_THRESHOLD = 3.0
    # 对于 scenedetect：越低 = 更多切分，越高 = 更保守。
    SHOT_DETECTION_MIN_SCENE_LEN = 3
    # 最小镜头长度（以帧为单位），某些检测器使用。

# Java 等价写法：
# double SHOT_DETECTION_THRESHOLD;
# int SHOT_DETECTION_MIN_SCENE_LEN;
# if ("film".equals(VIDEO_TYPE)) {
#     SHOT_DETECTION_THRESHOLD = 3.0;
#     SHOT_DETECTION_MIN_SCENE_LEN = 3;
# } else if ("vlog".equals(VIDEO_TYPE)) {
#     SHOT_DETECTION_THRESHOLD = 1.5;
#     SHOT_DETECTION_MIN_SCENE_LEN = 45;
# } else {
#     SHOT_DETECTION_THRESHOLD = 3.0;
#     SHOT_DETECTION_MIN_SCENE_LEN = 3;
# }

SHOT_DETECTION_SCENES_PATH = "shot_scenes.txt"
# 镜头边界结果的输出文件名（通常无需更改）。

SHOT_DETECTION_MODEL = "scenedetect"
# 选项："autoshot", "transnetv2", "Qwen3VL", "scenedetect"。
# 初学者应从默认值开始：scenedetect。



CLIP_SECS = 30
# 单个候选片段的最大长度（秒）。

MERGE_SHORT_SCENES = True
# 如果为 True，连续的短场景会被合并以减少碎片化。

SCENE_MERGE_METHOD = "min_length"
# 场景合并策略：
# - min_length：优先避免过短的场景
# - max_length：优先不超过最大场景持续时间

SCENE_MIN_LENGTH_SECS = 3
# 当 SCENE_MERGE_METHOD="min_length" 时的最小场景长度（秒）。

SCENE_SIMILARITY_THRESHOLD = 0.5
# 场景分割的相似度阈值：
# - 较低：更难分割（更长的场景）
# - 较高：更容易分割（更多碎片化场景）

MAX_SCENE_DURATION_SECS = 300
# 强制分割前的最大允许场景持续时间。

MIN_SCENE_DURATION_SECS = 30.0
# 短于此值的场景会被合并到相邻场景中。

WHOLE_VIDEO_SUMMARY_BATCH_SIZE = 50
# 全视频摘要每批的片段数量。
# 影响并行性和吞吐量。


# ═══════════════════════════════════════════════════════════════════════════════
# ASR (Speech Recognition)
# Converts dialogue to subtitles and can optionally run speaker diarization.
# Recommendation: use one backend at a time (local whisper_cpp or cloud litellm).
# ═══════════════════════════════════════════════════════════════════════════════

ASR_BACKEND = "litellm"
# 选项："whisper_cpp"（本地）| "litellm"（云端）。
# - 本地：成本更低/离线，速度取决于硬件
# - 云端：设置更简单，但产生 API 费用

ASR_LANGUAGE = "English"
# 识别语言。示例值："English", "Chinese", "en", "zh"。
# 设置为 None 进行自动检测。

# ──────────────────────────────────────────────────────────────────────────────
# 选项 1：whisper.cpp（本地）
# ──────────────────────────────────────────────────────────────────────────────

ASR_DEVICE = "cuda:0" if __import__('torch').cuda.is_available() else "cpu"
# 本地 ASR 的设备。可用时使用 cuda:0，否则使用 cpu。
# Java 等价写法：
# String ASR_DEVICE = torch.cuda().isAvailable() ? "cuda:0" : "cpu";
# 但 Python 可以动态导入模块

ASR_WHISPER_CPP_MODEL = "base.en"
# whisper.cpp 模型名称或本地 ggml 模型路径（例如 "base.en", "large-v3"）。

ASR_WHISPER_CPP_N_THREADS = 8
# CPU 推理线程数（主要与 CPU 运行相关）。

ASR_ENABLE_DIARIZATION = True
# 启用说话人分离（谁在说话）。
# 改善电影的对话理解，但增加运行时间。

ASR_DIARIZATION_MODEL_PATH = "pyannote/speaker-diarization-community-1"
# HuggingFace 模型名称/路径用于说话人分离。

ASR_MERGE_SAME_SPEAKER = True
# 合并来自同一说话人的相邻字幕片段。

ASR_MERGE_GAP = 1.0
# 合并同一说话人相邻片段的最大间隔（秒）。

# ───────────────────────────────────────────────────────────────────────────────
# 选项 2：LiteLLM（云端 ASR）
# Option 2: LiteLLM (cloud ASR)
# ───────────────────────────────────────────────────────────────────────────────

ASR_LITELLM_MAX_SEGMENT_MB = 1.0
# Max size (MB) per uploaded audio segment for cloud ASR.
# Helps avoid oversized requests.
# 云端 ASR 每次上传音频片段的最大大小（MB）。
# 有助于避免过大的请求。

ASR_LITELLM_BATCH_SIZE = 128
# Number of audio segments per request batch.
# Larger batches improve throughput but may increase rate-limit risk.
# 每个请求批次的音频片段数量。
# 更大的批次提高吞吐量，但可能增加速率限制风险。

# ------------------ Video Understanding Model ------------------ #
# ------------------ 视频理解模型 ------------------ #

SCENE_PROMPT_TYPE = VIDEO_TYPE
# Prompt style for scene analysis. Usually kept consistent with VIDEO_TYPE.
# 场景分析的提示样式。通常与 VIDEO_TYPE 保持一致。

VIDEO_ANALYSIS_MODEL_MAX_TOKEN = 16384 
# Max output token count for the video analysis model.
# 视频分析模型的最大输出 token 数。

VIDEO_ANALYSIS_MODEL = ""
# Video semantic analysis model name (called via OpenAI-compatible endpoint).
# 视频语义分析模型名称（通过 OpenAI 兼容端点调用）。

VIDEO_ANALYSIS_ENDPOINT = ""  
# API base URL for the video analysis model.
# 视频分析模型的 API 基础 URL。

VIDEO_ANALYSIS_API_KEY = ""
# API key for the video analysis model.
# 视频分析模型的 API 密钥。

CAPTION_BATCH_SIZE = 64
# Batch size for parallel clip captioning/analysis.
# 并行片段字幕生成/分析的批量大小。

SCENE_ANALYSIS_MIN_FRAMES = 6
# Minimum number of sampled frames per scene.
# Higher values may improve stability but increase runtime.
# 每个场景的最小采样帧数。
# 更高的值可能提高稳定性但增加运行时间。


# ------------------ Audio Model ------------------ #
# Analyzes musical beat/energy/structure and outputs editing keypoints.
# ------------------ 音频模型 ------------------ #
# 分析音乐节拍/能量/结构并输出剪辑关键点。

AUDIO_LITELLM_MODEL = ""
# Cloud model used for audio captioning and structure analysis.
# 用于音频字幕生成和结构分析的云端模型。

AUDIO_LITELLM_API_KEY = ""
# API key for the audio model.
# 音频模型的 API 密钥。

AUDIO_LITELLM_BASE_URL = ""
# API base URL for the audio model.
# 音频模型的 API 基础 URL。

AUDIO_DETECTION_METHODS = ["downbeat", "pitch", "mel_energy"]
# Keypoint detection methods (single or combined):
# - downbeat: rhythm/beat structure
# - pitch: melodic variation
# - mel_energy: energy peaks
# 关键点检测方法（单一或组合）：
# - downbeat：节奏/节拍结构
# - pitch：旋律变化
# - mel_energy：能量峰值

# ----- Downbeat（强拍）-----
# ----- Downbeat -----
AUDIO_BEATS_PER_BAR = 4
# Beats per bar (e.g., 4 for 4/4 time).
# 每小节的拍数（例如 4 表示 4/4 拍）。

AUDIO_DBN_THRESHOLD = 0.05
# Activation threshold for downbeat tracking.
# 强拍跟踪的激活阈值。

AUDIO_MIN_BPM = 55.0
AUDIO_MAX_BPM = 215.0
# BPM search range preset (may be weakly used in some paths).
# BPM 搜索范围预设（在某些路径中可能弱使用）。

# ----- Pitch（音高）-----
# ----- Pitch -----
AUDIO_PITCH_TOLERANCE = 0.8
# Pitch matching tolerance.
# 音高匹配容差。

AUDIO_PITCH_THRESHOLD = 0.8
# Confidence threshold for keeping pitch keypoint candidates.
# Higher value = stricter filtering.
# 保留音高关键点候选者的置信度阈值。
# 值越高 = 过滤越严格。

AUDIO_PITCH_MIN_DISTANCE = 0.3
# Minimum spacing (seconds) between pitch keypoints.
# 音高关键点之间的最小间距（秒）。

AUDIO_PITCH_NMS_METHOD = "basic"
# Non-maximum suppression strategy: "basic" / "adaptive" / "window".
# 非极大值抑制策略："basic" / "adaptive" / "window"。

AUDIO_PITCH_MAX_POINTS = 50
# Maximum number of retained pitch keypoints.
# 保留的音高关键点最大数量。

# ----- Mel Energy（梅尔能量）-----
# ----- Mel Energy -----
AUDIO_MEL_WIN_S = 512
# FFT window size.
# FFT 窗口大小。

AUDIO_MEL_N_FILTERS = 40
# Number of mel filters.
# 梅尔滤波器数量。

AUDIO_MEL_THRESHOLD_RATIO = 0.3
# Threshold ratio for energy peak detection.
# 能量峰值检测的阈值比率。

AUDIO_MEL_MIN_DISTANCE = 0.3
# Minimum spacing (seconds) between mel-energy keypoints.
# 梅尔能量关键点之间的最小间距（秒）。

AUDIO_MEL_NMS_METHOD = "basic"
# Non-maximum suppression strategy: "basic" / "adaptive" / "window".
# 非极大值抑制策略："basic" / "adaptive" / "window"。

AUDIO_MEL_MAX_POINTS = 50
# Maximum number of retained mel-energy keypoints.
# 保留的梅尔能量关键点最大数量。

# ----- Keypoint post-processing (denoising/sparsification) -----
# ----- 关键点后处理（去噪/稀疏化）-----
AUDIO_MERGE_CLOSE = 0
# [May be unused in main path] Merge very close keypoints.
# [主路径中可能未使用] 合并非常接近的关键点。

AUDIO_MIN_INTERVAL = 0
# Global minimum keypoint interval (seconds).
# 全局最小关键点间隔（秒）。

AUDIO_TOP_K = 0
# Keep only top-K strongest keypoints. 0 means unlimited.
# 仅保留前 K 个最强关键点。0 表示无限制。

AUDIO_ENERGY_PERCENTILE = 0
# Keep only keypoints above this energy percentile (0~100).
# 仅保留高于此能量百分位数（0~100）的关键点。

AUDIO_SILENCE_THRESHOLD_DB = -45.0
# Silence filtering threshold (dB).
# Segments below this level are treated as too quiet and filtered.
# 静音过滤阈值（分贝）。
# 低于此水平的片段被视为太安静并被过滤。

# ----- Audio segment duration constraints (frequently tuned) -----
# ----- 音频片段持续时间约束（经常调整）-----
AUDIO_MIN_SEGMENT_DURATION = 0.1
# Minimum segment duration (seconds). Smaller values create faster cuts.
# 最小片段持续时间（秒）。较小的值创建更快的剪辑。

AUDIO_MAX_SEGMENT_DURATION = 2.0
# Maximum segment duration (seconds). Larger values create slower pacing.
# 最大片段持续时间（秒）。较大的值创建更慢的节奏。

# ----- Music structure analysis (Level-1) -----
# ----- 音乐结构分析（第 1 级）-----
AUDIO_STRUCTURE_TEMPERATURE = 0.7
AUDIO_STRUCTURE_TOP_P = 0.95
AUDIO_STRUCTURE_MAX_TOKENS = 4096
# Controls generation style and token limit for structure analysis.
# 控制结构分析的生成风格和 token 限制。

# ----- Section-aware shot allocation -----
# ----- 基于段落的镜头分配 -----
AUDIO_USE_STAGE1_SECTIONS = True
# Use Level-1 structural sections to guide keypoint filtering.
# 使用第 1 级结构段落来指导关键点过滤。

AUDIO_SECTION_MIN_INTERVAL = AUDIO_MIN_SEGMENT_DURATION
# Global minimum keypoint interval across sections.
# 跨段落的全局最小关键点间隔。

AUDIO_TOTAL_SHOTS = 200
# Target total shot count, allocated proportionally by sections.
# For quick debugging, try reducing this to 30~80.
# 目标总镜头数，按段落比例分配。
# 对于快速调试，尝试将其减少到 30~80。

# ----- Multi-feature fusion weights -----
# ----- 多特征融合权重 -----
AUDIO_WEIGHT_DOWNBEAT = 1.0
AUDIO_WEIGHT_PITCH = 1.0
AUDIO_WEIGHT_MEL_ENERGY = 1.0
# Fusion weights for downbeat/pitch/mel-energy signals.
# Defaults use equal weighting.
# 强拍/音高/梅尔能量信号的融合权重。
# 默认使用等权重。

# ----- Keypoint caption analysis (Level-2) -----
# ----- 关键点字幕分析（第 2 级）-----
AUDIO_BATCH_SIZE = 8
# Batch size for parallel audio-segment inference.
# Larger values increase throughput but use more resources.
# 音频片段并行推理的批量大小。
# 较大的值增加吞吐量但使用更多资源。

AUDIO_KEYPOINT_TEMPERATURE = 0.7
AUDIO_KEYPOINT_TOP_P = 0.95
AUDIO_KEYPOINT_MAX_TOKENS = 4096
# Controls style and maximum length for keypoint caption generation.
# 控制关键点字幕生成的风格和最大长度。

# ------------------ Agent Runtime ------------------ #
# ------------------ 智能体运行时 ------------------ #

AGENT_MODEL_MAX_TOKEN = 8192
# Maximum generated tokens per agent response (not total context size).
# 每个智能体响应的最大生成 token 数（不是总上下文大小）。

AGENT_MODEL_MAX_RETRIES = 4
# Max retries per agent step when model calls fail.
# 模型调用失败时每个智能体步骤的最大重试次数。

AGENT_RATE_LIMIT_BACKOFF_BASE = 1.0
AGENT_RATE_LIMIT_MAX_BACKOFF = 8.0
# Backoff timing (seconds) when rate limits occur.
# 发生速率限制时的退避计时（秒）。

AUDIO_SEGMENT_MIN_DURATION_SEC = 5.0
AUDIO_SEGMENT_MAX_DURATION_SEC = 15.0
# Allowed music-span duration range for short-video planning.
# 短视频规划允许的音乐跨度持续时间范围。

AUDIO_SEGMENT_SELECTION_MAX_RETRIES = 3
# Retry count when selected music span is invalid.
# 选择的音乐跨度无效时的重试次数。

AUDIO_SEGMENT_TIME_TOLERANCE_SEC = 0.25
# Allowed timestamp drift (seconds) when validating selected spans.
# 验证所选跨度时允许的时间戳漂移（秒）。

ENABLE_TRIM_SHOT_CHARACTER_ANALYSIS = True
# Enable VLM character analysis during trim_shot.
# 在 trim_shot 期间启用 VLM 角色分析。

CORE_MAX_FRAMES = 60
# Maximum sampled frames per clip for core + reviewer analysis.
# 核心 + 审核器分析的每个片段的最大采样帧数。

AGENT_LITELLM_URL = ""
# API base URL for the agent LLM.
# 智能体 LLM 的 API 基础 URL。

AGENT_LITELLM_API_KEY = ""
# API key for the agent LLM.
# 智能体 LLM 的 API 密钥。

AGENT_LITELLM_MODEL = ""
# Primary model for the agent.
# 智能体的主要模型。

PARALLEL_SHOT_ENABLED = True
# Whether to enable parallel shot selection (ParallelShotOrchestrator) in film mode.
# 是否在电影模式下启用并行镜头选择（ParallelShotOrchestrator）。

PARALLEL_SHOT_MAX_WORKERS = 4
# Number of parallel workers.
# 并行工作线程数量。
# Java 类比：相当于 ThreadPoolExecutor 的核心线程数

PARALLEL_SHOT_MAX_RERUNS = 2
# Maximum rerun rounds for conflicted shots.
# 冲突镜头的最大重跑轮数。



# ------------------ Reviewer (Quality Checks) ------------------ #
# ------------------ 审核器（质量检查）------------------ #

ENABLE_REVIEWER = True
# Master switch for the Reviewer agent. Set to False to skip all review checks (face quality, duplicate detection, etc.).
# 审核器智能体的总开关。设置为 False 跳过所有审核检查（面部质量、重复检测等）。

ENABLE_FACE_QUALITY_CHECK = True
# Enable face/protagonist quality checks before finalizing shot selection.
# 在最终确定镜头选择之前启用人脸/主角质量检查。

VLM_FACE_LOG_EACH_FRAME = False
# Print per-frame protagonist detection logs (very verbose; debug only).
# 打印每帧的主角检测日志（非常详细；仅用于调试）。

ENABLE_AESTHETIC_QUALITY_CHECK = True
# Toggle aesthetic quality checks for vlog mode (may not be fully implemented).
# 切换 vlog 模式的美学质量检查（可能未完全实现）。

FACE_QUALITY_CHECK_METHOD = "vlm"
# Quality check method (currently only "vlm" is supported).
# 质量检查方法（目前仅支持 "vlm"）。

# ------------------ Protagonist Presence Constraints ------------------ #
# ------------------ 主角存在约束 ------------------ #

MAIN_CHARACTER_NAME = ""
# Main character / target subject name (comma-separated for multiple roles).
# This is one of the highest-impact parameters in object mode.
# 主要角色/目标主题名称（多个角色用逗号分隔）。
# 这是对象模式中影响最大的参数之一。

MIN_PROTAGONIST_RATIO = 0.7
# Minimum ratio (0~1) of frames where the protagonist should be the main focus.
# 主角应作为主要焦点的帧的最小比例（0~1）。

VLM_MIN_BOX_SIZE = 100
# Minimum protagonist bounding-box size in pixels. Smaller detections are ignored.
# 主角边界框的最小尺寸（像素）。较小的检测被忽略。


# ------------------ Shot Selection Constraints ------------------ #
# These parameters define fallback behavior when perfect matches are unavailable.
# ------------------ 镜头选择约束 ------------------ #
# 这些参数定义了在无法获得完美匹配时的回退行为。

MIN_ACCEPTABLE_SHOT_DURATION = 2.0
# Minimum acceptable final shot duration (seconds).
# Smaller values increase match rate but may produce more fragmented edits.
# 最小可接受的最终镜头持续时间（秒）。
# 较小的值增加匹配率但可能产生更多碎片化剪辑。

ALLOW_DURATION_TOLERANCE = 1.0
# Allow duration deviation of ±N seconds from target.
# 允许偏离目标时长 ±N 秒。

ALLOW_CONTENT_MISMATCH = True
# Allow semantically similar (not exact) content matches.
# 允许语义相似（不完全相同）的内容匹配。

ENABLE_FALLBACK_STRATEGY = True
# Enable multi-level fallback strategy to reduce hard failures.
# 启用多级回退策略以减少硬性失败。

SCENE_EXPLORATION_RANGE = 3
# Extra exploration range around recommended scenes (±N scenes).
# Example: if recommended scene is 8 and range=3, search scene 5~11.
# Set to 0 to strictly limit selection to recommended scenes only.
# 推荐场景周围的额外探索范围（±N 个场景）。
# 示例：如果推荐场景是 8 且 range=3，则搜索场景 5~11。
# 设置为 0 以严格限制选择仅限推荐场景。


