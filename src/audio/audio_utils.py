"""
Audio processing utilities without librosa dependency.
不使用 librosa 依赖的音频处理工具。

This module provides audio processing functions using soundfile and scipy
to replace librosa, avoiding the numba JIT compilation issues.
此模块使用 soundfile 和 scipy 提供音频处理函数，
替代 librosa，避免 numba JIT 编译问题。

Java 类比：类似 Java 中的音频处理工具类，使用第三方库进行音频加载和重采样。
"""

import base64
from io import BytesIO
import numpy as np
import soundfile as sf
from scipy import signal
import audioread
import av


SAMPLE_RATE = 16000
# 默认采样率（每秒样本数）
# Java 类比：public static final int SAMPLE_RATE = 16000;


def load_audio_no_librosa(path_or_buffer, sr=16000, offset=0.0, duration=None):
    """
    Load audio file without using librosa.
    不使用 librosa 加载音频文件。
    
    This function:
    此函数：
    1. Tries soundfile first (fastest for standard formats)
    1. 首先尝试 soundfile（标准格式最快）
    2. Falls back to audioread for non-standard formats
    2. 对非标准格式回退到 audioread
    3. Converts stereo to mono if needed
    3. 如果需要，将立体声转换为单声道
    4. Applies offset and duration trimming
    4. 应用偏移量和时长裁切
    5. Resamples to target sample rate
    5. 重采样到目标采样率
    
    Args:
        path_or_buffer: 文件路径、URL 或缓冲区 (File path, URL, or buffer)
        sr: 目标采样率（默认 16000） (Target sample rate, default 16000)
        offset: 起始时间（秒，默认 0.0） (Start time in seconds, default 0.0)
        duration: 持续时间（秒，默认 None 表示加载整个文件） (Duration in seconds, default None for entire file)
    Returns:
        单声道音频样本的 numpy 数组 (numpy array of mono audio samples)
    
    Java 类比：类似 AudioSystem.getAudioInputStream() + 自定义重采样逻辑。
    """
    try:
        # Try using soundfile first (fastest for standard formats)
        # 首先尝试使用 soundfile（标准格式最快）
        audio, orig_sr = sf.read(path_or_buffer, dtype='float32')
        
        # Convert stereo to mono if needed
        # 如果需要，将立体声转换为单声道（取左右声道的平均值）
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Apply offset
        # 应用起始偏移量（裁切掉前面的部分）
        if offset > 0:
            start_sample = int(offset * orig_sr)  # 计算起始样本索引
            audio = audio[start_sample:]
        
        # Apply duration
        # 应用持续时间限制（裁切后面的部分）
        if duration is not None:
            duration_samples = int(duration * orig_sr)  # 计算持续时间的样本数
            audio = audio[:duration_samples]
        
        # Resample if needed
        # 如果需要，重采样到目标采样率
        if orig_sr != sr:
            audio = resample_audio(audio, orig_sr, sr)
        
        return audio
        
    except Exception as e:
        # Fallback to audioread for non-standard formats
        # 对非标准格式回退到 audioread
        print(f"soundfile failed, using audioread fallback: {e}")
        return load_audio_with_audioread(path_or_buffer, sr, offset, duration)


def resample_audio(audio, orig_sr, target_sr):
    """
    Resample audio to target sample rate using scipy.
    使用 scipy 将音频重采样到目标采样率。
    
    This function uses scipy.signal.resample for high-quality resampling
    with anti-aliasing filtering.
    此函数使用 scipy.signal.resample 进行高质量重采样，
    包含抗混叠滤波。
    
    Args:
        audio: 音频样本的 numpy 数组 (numpy array of audio samples)
        orig_sr: 原始采样率 (Original sample rate)
        target_sr: 目标采样率 (Target sample rate)
    Returns:
        重采样后的音频数组（float32 类型） (Resampled audio array in float32)
    
    Java 类比：类似 javax.sound.sampled.AudioFormat 的重采样逻辑。
    """
    # 如果采样率相同，直接返回
    if orig_sr == target_sr:
        return audio
    
    # Calculate new length
    # 计算新的样本数量
    num_samples = int(len(audio) * target_sr / orig_sr)
    
    # Use scipy's resample for high-quality resampling
    # 使用 scipy 的 resample 进行高质量重采样
    resampled = signal.resample(audio, num_samples)
    
    return resampled.astype(np.float32)  # 转换为 float32 类型


def load_audio_with_audioread(path_or_buffer, sr=16000, offset=0.0, duration=None):
    """
    Load audio using audioread (ffmpeg backend) as fallback.
    使用 audioread（ffmpeg 后端）作为回退方案加载音频。
    
    This function is called when soundfile fails to load the audio file.
    It supports more formats through ffmpeg but is slower.
    此函数在 soundfile 无法加载音频文件时被调用。
    通过 ffmpeg 支持更多格式，但速度较慢。
    
    Args:
        path_or_buffer: 文件路径、URL 或缓冲区 (File path, URL, or buffer)
        sr: 目标采样率 (Target sample rate)
        offset: 起始时间（秒） (Start time in seconds)
        duration: 持续时间（秒） (Duration in seconds)
    Returns:
        音频样本的 numpy 数组 (numpy array of audio samples)
    
    Java 类比：类似使用 FFmpeg 命令行工具提取音频并转换为 PCM 格式。
    """
    # 打开音频文件（使用 audioread/ffmpeg）
    with audioread.audio_open(path_or_buffer) as audio_file:
        orig_sr = audio_file.samplerate  # 原始采样率
        channels = audio_file.channels  # 声道数
        
        # Calculate offset and duration in samples
        # 计算偏移量和持续时间的样本数
        offset_samples = int(offset * orig_sr)
        if duration is not None:
            duration_samples = int(duration * orig_sr)
        else:
            duration_samples = None
        
        # Read all audio data
        # 读取所有音频数据（分块读取）
        audio_data = []
        for buf in audio_file:
            audio_data.append(buf)
        
        # Convert to numpy array
        # 转换为 numpy 数组
        audio_bytes = b''.join(audio_data)  # 合并所有字节数据
        # 从字节缓冲区创建 int16 数组，然后转换为 float32 并归一化到 [-1, 1]
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Reshape for multi-channel audio
        # 多声道音频重塑
        if channels > 1:
            audio = audio.reshape(-1, channels)  # 重塑为 (样本数, 声道数)
            # Convert to mono
            # 转换为单声道（取各声道的平均值）
            audio = audio.mean(axis=1)
        
        # Apply offset
        # 应用起始偏移量
        if offset > 0:
            audio = audio[offset_samples:]
        
        # Apply duration
        # 应用持续时间限制
        if duration_samples is not None:
            audio = audio[:duration_samples]
        
        # Resample if needed
        # 如果需要，重采样到目标采样率
        if orig_sr != sr:
            audio = resample_audio(audio, orig_sr, sr)
        
        return audio


def _check_if_video_has_audio(video_path):
    """
    Check if video file has audio track.
    检查视频文件是否有音频轨道。
    
    Args:
        video_path: 视频文件路径 (Path to video file)
    Returns:
        True 如果有音频轨道，否则 False (True if has audio track, else False)
    
    Java 类比：类似检查 MediaFormat 中是否有音频轨道。
    """
    # 打开视频容器
    container = av.open(video_path)
    # 过滤出所有音频流（列表推导式）
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        return False
    return True


def process_audio_info(conversations, use_audio_in_video):
    """
    Read and process audio info without using librosa.
    不使用 librosa 读取和处理音频信息。
    
    This function processes audio data from conversations, supporting:
    此函数处理对话中的音频数据，支持：
    1. Direct numpy array input
    1. 直接的 numpy 数组输入
    2. Base64 encoded audio
    2. Base64 编码的音频
    3. Audio file paths
    3. 音频文件路径
    4. Video files with audio tracks
    4. 带音频轨道的视频文件
    
    Support dict keys:
    支持的字典键：
    
    type = audio
    - audio: 音频数据或路径 (audio data or path)
    - audio_start: 起始时间（秒） (start time in seconds)
    - audio_end: 结束时间（秒） (end time in seconds)
    
    type = video
    - video: 视频路径 (video path)
    - video_start: 起始时间（秒） (start time in seconds)
    - video_end: 结束时间（秒） (end time in seconds)
    
    Args:
        conversations: 对话列表，包含音频/视频信息 (List of conversations with audio/video info)
        use_audio_in_video: 是否从视频中提取音频 (Whether to extract audio from video)
    Returns:
        处理后的音频列表 (List of processed audio arrays)
    
    Java 类比：类似一个音频数据处理管道，支持多种输入源和格式转换。
    """
    audios = []  # 存储处理后的音频列表
    # 如果 conversations 是单个对话字典，包装成列表
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    
    # 遍历所有对话
    for conversation in conversations:
        for message in conversation:
            # 跳过没有 content 列表的消息
            if not isinstance(message["content"], list):
                continue
            
            # 遍历消息内容中的每个元素
            for ele in message["content"]:
                if ele["type"] == "audio":
                    # 处理音频类型
                    if "audio" in ele or "audio_url" in ele:
                        path = ele.get("audio", ele.get("audio_url"))  # 获取音频路径或数据
                        audio_start = ele.get("audio_start", 0.0)  # 起始时间
                        audio_end = ele.get("audio_end", None)  # 结束时间
                        
                        # Handle numpy array input
                        # 处理 numpy 数组输入（直接传入音频数据）
                        if isinstance(path, np.ndarray):
                            if path.ndim > 1:
                                raise ValueError("Support only mono audio")  # 只支持单声道
                            # 根据起始和结束时间裁切音频
                            audios.append(
                                path[int(SAMPLE_RATE * audio_start) : None if audio_end is None else int(SAMPLE_RATE * audio_end)]
                            )
                            continue
                        
                        # Handle base64 encoded audio
                        # 处理 Base64 编码的音频
                        elif path.startswith("data:audio"):
                            _, base64_data = path.split("base64,", 1)
                            data = BytesIO(base64.b64decode(base64_data))  # 解码为字节流
                        
                        # Handle HTTP(S) URLs
                        # 处理 HTTP/HTTPS URL
                        elif path.startswith("http://") or path.startswith("https://"):
                            data = path
                        
                        # Handle file:// protocol
                        # 处理 file:// 协议
                        elif path.startswith("file://"):
                            data = path[len("file://") :]  # 移除 file:// 前缀
                        
                        # Handle regular file paths
                        # 处理普通文件路径
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown audio {}".format(ele))
                
                elif use_audio_in_video and ele["type"] == "video":
                    # 处理视频类型（从视频中提取音频）
                    if "video" in ele or "video_url" in ele:
                        path = ele.get("video", ele.get("video_url"))  # 获取视频路径
                        audio_start = ele.get("video_start", 0.0)  # 起始时间
                        audio_end = ele.get("video_end", None)  # 结束时间
                        
                        # 断言：视频必须有音频轨道
                        assert _check_if_video_has_audio(
                            path
                        ), "Video must has audio track when use_audio_in_video=True"
                        
                        # 处理不同格式的路径
                        if path.startswith("http://") or path.startswith("https://"):
                            data = path
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown video {}".format(ele))
                else:
                    # 跳过其他类型
                    continue
                
                # Load audio using our custom function (no librosa!)
                # 使用自定义函数加载音频（不使用 librosa！）
                duration = (audio_end - audio_start) if audio_end is not None else None  # 计算持续时间
                audio = load_audio_no_librosa(
                    data,
                    sr=SAMPLE_RATE,  # 目标采样率
                    offset=audio_start,  # 起始偏移
                    duration=duration  # 持续时间
                )
                audios.append(audio)  # 添加到结果列表
    
    # 如果没有音频，返回 None
    if len(audios) == 0:
        audios = None
    
    return audios


def process_mm_info_no_librosa(conversations, use_audio_in_video):
    """
    Process multimodal information without using librosa.
    不使用 librosa 处理多模态信息。
    
    This is a replacement for qwen_omni_utils.process_mm_info that doesn't
    depend on librosa, avoiding numba JIT compilation issues.
    这是 qwen_omni_utils.process_mm_info 的替代品，
    不依赖 librosa，避免 numba JIT 编译问题。
    
    This function:
    此函数：
    1. Processes audio data using custom functions (no librosa)
    1. 使用自定义函数处理音频数据（不使用 librosa）
    2. Processes vision data using qwen_omni_utils (if available)
    2. 使用 qwen_omni_utils 处理视觉数据（如果可用）
    3. Returns combined multimodal data
    3. 返回组合的多模态数据
    
    Args:
        conversations: 对话消息列表 (List of conversation messages)
        use_audio_in_video: 是否从视频文件中提取音频 (Whether to extract audio from video files)
    Returns:
        (audios, images, videos) 元组 (Tuple of audios, images, videos)
    
    Java 类比：类似一个多模态数据处理器，分别处理音频和视觉数据并合并结果。
    """
    # Import vision processing from qwen_omni_utils (doesn't use librosa)
    # 从 qwen_omni_utils 导入视觉处理（不使用 librosa）
    try:
        from qwen_omni_utils.v2_5.vision_process import process_vision_info
    except ImportError:
        # Fallback: return None for images and videos
        # 回退方案：图像和视频返回 None
        print("Warning: qwen_omni_utils not found, returning None for vision data")
        audios = process_audio_info(conversations, use_audio_in_video)
        return audios, None, None
    
    # Process audio without librosa
    # 不使用 librosa 处理音频
    audios = process_audio_info(conversations, use_audio_in_video)
    
    # Process vision using qwen_omni_utils (doesn't depend on librosa)
    # 使用 qwen_omni_utils 处理视觉数据（不依赖 librosa）
    vision = process_vision_info(conversations, return_video_kwargs=False)
    
    # 返回组合结果：(audios,) + vision 相当于 (audios, images, videos)
    return (audios,) + vision
