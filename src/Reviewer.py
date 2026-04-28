"""
Reviewer Agent - Shot Selection Validation and Quality Review.
审核器智能体 - 镜头选择验证和质量审核。

This module implements the Reviewer agent responsible for:
此模块实现负责以下任务的审核器智能体：
1. Validating time range overlaps to prevent duplicate footage
1. 验证时间范围重叠以防止重复片段
2. Checking shot duration matches target length
2. 检查镜头时长是否匹配目标长度
3. Validating multi-shot stitching continuity
3. 验证多镜头拼接的连续性
4. VLM-based aesthetic analysis and protagonist detection
4. 基于 VLM 的美学分析和主角检测
5. Thread-safe video frame extraction for parallel processing
5. 线程安全的视频帧提取用于并行处理

Java 类比：类似一个质量检查器，结合规则验证和 AI 视觉分析。
"""
import os
import json
import copy
import re
import numpy as np
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Annotated as A, Optional, Tuple, List
import cv2
import litellm
from src.utils.media_utils import seconds_to_hhmmss as convert_seconds_to_hhmmss, array_to_base64
from src.utils.time_format_convert import hhmmss_to_seconds, seconds_to_hhmmss
from src import config
from src.prompt import VLM_AESTHETIC_ANALYSIS_PROMPT, VLM_PROTAGONIST_DETECTION_PROMPT
from src.func_call_shema import doc as D
from src.video.preprocess.video_utils import _create_decord_reader

class StopException(Exception):
    """
    Stop Execution by raising this exception (Signal that the task is Finished).
    通过抛出此异常停止执行（信号表示任务已完成）。
    
    Java 类比：类似自定义的 InterruptedException 或 CancellationException。
    """


# Thread-local storage for video readers
# 线程本地存储，用于保存视频阅读器
_THREAD_VIDEO_READERS = threading.local()


def _get_thread_video_reader(video_path: str):
    """
    Get or create a thread-local video reader for the given video path.
    获取或创建给定视频路径的线程本地视频阅读器。
    
    This function ensures each thread has its own video reader instance,
    preventing race conditions when multiple threads read from the same video.
    此函数确保每个线程都有自己的视频阅读器实例，
    防止多个线程从同一视频读取时出现竞态条件。
    
    Args:
        video_path: 视频文件路径 (Path to video file)
    Returns:
        Decord 视频阅读器实例，如果 video_path 为空则返回 None (Decord video reader instance, None if video_path is empty)
    
    Java 类比：类似 ThreadLocal<VideoReader>，每个线程维护自己的阅读器副本。
    """
    if not video_path:
        return None
    # 获取当前线程的阅读器
    reader = getattr(_THREAD_VIDEO_READERS, "reader", None)
    reader_path = getattr(_THREAD_VIDEO_READERS, "video_path", None)
    # 如果阅读器不存在或路径不匹配，创建新的阅读器
    if reader is None or reader_path != video_path:
        target_resolution = getattr(config, "VIDEO_RESOLUTION", None)
        reader = _create_decord_reader(video_path, target_resolution)
        # 存储到线程本地变量
        _THREAD_VIDEO_READERS.reader = reader
        _THREAD_VIDEO_READERS.video_path = video_path
    return reader


def _clear_thread_video_reader():
    """
    Clear the thread-local video reader to free resources.
    清除线程本地视频阅读器以释放资源。
    
    Should be called when a thread finishes processing to prevent memory leaks.
    应在线程完成处理时调用以防止内存泄漏。
    
    Java 类比：类似在 finally 块中关闭资源，或调用 ThreadLocal.remove()。
    """
    if hasattr(_THREAD_VIDEO_READERS, "reader"):
        _THREAD_VIDEO_READERS.reader = None
    if hasattr(_THREAD_VIDEO_READERS, "video_path"):
        _THREAD_VIDEO_READERS.video_path = None

def Review_timeline(timeline, ):
    """
    Review Execution by raising this exception (Signal that the task is Finished).
    """

def Review_audio_video_alignment(alignment, ):
    """
    Review Execution by raising this exception (Signal that the task is Finished).
    """


def review_clip(
    time_range: A[str, D("The time range to check (e.g., '00:13:28 to 00:13:40').")],
    used_time_ranges: A[list, D("List of already used time ranges. Auto-injected.")] = None
) -> str:
    """
    Check if the proposed time range overlaps with any previously used clips.
    检查提议的时间范围是否与之前使用过的片段重叠。
    
    You MUST call this tool BEFORE calling finish to ensure no duplicate footage.
    在调用 finish 之前必须调用此工具以确保没有重复片段。
    
    This function:
    此函数：
    1. Parses the time range string into seconds
    1. 将时间范围字符串解析为秒数
    2. Compares against all previously used ranges
    2. 与所有之前使用的范围进行比较
    3. Detects overlaps and reports overlap details
    3. 检测重叠并报告重叠详情
    
    Args:
        time_range: 要检查的时间范围，格式如 "HH:MM:SS to HH:MM:SS" (Time range to check)
        used_time_ranges: 已使用的时间范围列表，自动注入 (List of used time ranges, auto-injected)
    Returns:
        验证结果消息，包含是否重叠及详细信息 (Validation result message with overlap details)
    
    Example:
    示例：
    >>> review_clip("00:10:00 to 00:10:05", [(0, 3), (6, 9)])
    '✅ OK: Time range 00:10:00 to 00:10:05 does not overlap...'
    """
    if used_time_ranges is None:
        used_time_ranges = []

    # Parse the time range
    # 解析时间范围
    match = re.search(r'([\d:.]+)\s+to\s+([\d:.]+)', time_range, re.IGNORECASE)
    if not match:
        return f"Error: Could not parse time range '{time_range}'. Please use format 'HH:MM:SS to HH:MM:SS'."

    try:
        fps = getattr(config, "VIDEO_FPS", 24) or 24
        start_sec = hhmmss_to_seconds(match.group(1), fps=fps)
        end_sec = hhmmss_to_seconds(match.group(2), fps=fps)
    except Exception as e:
        return f"Error parsing time range: {e}"

    if not used_time_ranges:
        return f"✅ OK: Time range {time_range} is available. No previous clips have been used yet. You can proceed with finish."

    # Check for overlaps
    # 检查重叠
    overlapping_clips = []
    for idx, (used_start, used_end) in enumerate(used_time_ranges):
        # 判断是否重叠：新片段的开始 < 已用片段的结束 AND 新片段的结束 > 已用片段的开始
        # Java 类比：if (startSec < usedEnd && endSec > usedStart)
        if start_sec < used_end and end_sec > used_start:
            # 计算重叠部分
            overlap_start = max(start_sec, used_start)
            overlap_end = min(end_sec, used_end)
            overlapping_clips.append({
                "clip_idx": idx + 1,
                "used_range": f"{convert_seconds_to_hhmmss(used_start)} to {convert_seconds_to_hhmmss(used_end)}",
                "overlap": f"{convert_seconds_to_hhmmss(overlap_start)} to {convert_seconds_to_hhmmss(overlap_end)}"
            })

    if overlapping_clips:
        result = f"❌ OVERLAP DETECTED: Time range {time_range} overlaps with {len(overlapping_clips)} previously used clip(s):\n"
        for clip in overlapping_clips:
            result += f"  - Clip {clip['clip_idx']}: {clip['used_range']} (overlap: {clip['overlap']})\n"
        result += "\n⚠️ Please select a DIFFERENT time range to avoid duplicate footage. Do NOT call finish with this range."
        return result
    else:
        return f"✅ OK: Time range {time_range} does not overlap with any previously used clips. You can proceed with finish."


def review_finish(
    answer: A[str, D("Output the final shot time range. Must be exactly ONE continuous clip.")],
    target_length_sec: A[float, D("Expected total length in seconds")] = 0.0,
) -> str:
    """
    Review and validate the proposed shot selection before finishing.
    在完成前审核并验证提议的镜头选择。
    
    Validates that exactly ONE shot is provided and its duration matches the target.
    验证是否只提供了一个镜头且其时长与目标匹配。
    
    You MUST call this tool BEFORE calling finish to ensure the shot is valid.
    在调用 finish 之前必须调用此工具以确保镜头有效。
    
    IMPORTANT: Only accepts ONE continuous time range. Multiple shots will be rejected.
    重要：只接受一个连续的时间范围。多个镜头将被拒绝。
    Example: [shot: 00:10:00 to 00:10:03.4]
    示例：[shot: 00:10:00 to 00:10:03.4]
    
    This function performs comprehensive validation:
    此函数执行全面验证：
    1. Parses shot time ranges from LLM output
    1. 从 LLM 输出中解析镜头时间范围
    2. Validates shot count (max configurable limit)
    2. 验证镜头数量（最大可配置限制）
    3. Checks for overlapping or gapped shots
    3. 检查重叠或有间隙的镜头
    4. Validates total duration against target
    4. 验证总时长与目标的匹配
    5. Provides actionable feedback for adjustments
    5. 提供可操作的调整建议
    
    Args:
        answer: LLM 的输出，包含镜头时间范围 (LLM output with shot time ranges)
        target_length_sec: 目标时长（秒） (Target duration in seconds)
    Returns:
        验证结果消息，成功或失败及详细信息 (Validation result message with details)
    """
    # Parse the answer to extract shot time ranges
    # 解析答案以提取镜头时间范围
    # Expected formats: "[shot: 00:10:00 to 00:10:05]" or "shot 1: 00:10:00 to 00:10:05"
    shot_pattern = re.compile(r'\[?shot[\s_]*\d*:\s*([0-9:.]+)\s+to\s+([0-9:.]+)\]?', re.IGNORECASE)
    matches = shot_pattern.findall(answer)

    if not matches:
        return "❌ Error: Could not parse shot time ranges from the answer. Please provide time range(s) in the format: [shot: HH:MM:SS to HH:MM:SS]"

    # Allow multiple shots for stitching (with reasonable limit)
    # 允许多个镜头进行拼接（有合理限制）
    from src import config
    max_shots_allowed = getattr(config, 'MAX_SHOTS_PER_CLIP', 3)
    if len(matches) > max_shots_allowed:
        return (
            f"❌ Error: You provided {len(matches)} shots, but maximum allowed is {max_shots_allowed}. "
            f"Please reduce the number of stitched shots or combine them into fewer segments."
        )

    # Calculate total duration and collect clips
    # 计算总时长并收集片段
    clips = []
    total_duration = 0

    fps = getattr(config, "VIDEO_FPS", 24) or 24
    for i, (start_time, end_time) in enumerate(matches, 1):
        try:
            start_sec = hhmmss_to_seconds(start_time, fps=fps)
            end_sec = hhmmss_to_seconds(end_time, fps=fps)
            duration = end_sec - start_sec

            if duration <= 0:
                return f"❌ Error: Shot {i} has invalid duration (start: {start_time}, end: {end_time}). End time must be greater than start time."

            clips.append({
                'start': start_time,
                'end': end_time,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'duration': duration
            })
            total_duration += duration
        except Exception as e:
            return f"❌ Error parsing shot {i} time range ({start_time} to {end_time}): {str(e)}"

    # Validate continuity for multi-shot stitching
    # 验证多镜头拼接的连续性
    if len(clips) > 1:
        max_gap = getattr(config, 'MAX_STITCH_GAP_SEC', 2.0)
        for i in range(len(clips) - 1):
            gap = clips[i+1]['start_sec'] - clips[i]['end_sec']
            if gap < 0:
                return f"❌ Error: Overlapping shots. Shot {i+1} ends at {clips[i]['end']}, but shot {i+2} starts at {clips[i+1]['start']}"
            if gap > max_gap:
                return (
                    f"❌ Error: Time gap ({gap:.2f}s) between shot {i+1} and {i+2} exceeds maximum ({max_gap}s).\n"
                    f"Stitched shots must maintain visual continuity. Please select closer shots or use a single continuous clip."
                )

    # Check if total duration matches target length (allow tolerance)
    # 检查总时长是否与目标长度匹配（允许容差）
    duration_diff = total_duration - target_length_sec

    # Prepare duration summary
    # 准备时长摘要
    if len(clips) == 1:
        duration_line = f"shot: {clips[0]['start']} to {clips[0]['end']} ({clips[0]['duration']:.2f}s)"
    else:
        duration_line = f"{len(clips)} stitched shots (total {total_duration:.2f}s):\n"
        for i, clip in enumerate(clips, 1):
            duration_line += f"  Shot {i}: {clip['start']} to {clip['end']} ({clip['duration']:.2f}s)\n"

    # Check for very short clips
    # 检查过短的片段
    min_acceptable = getattr(config, 'MIN_ACCEPTABLE_SHOT_DURATION', 2.0)
    short_clips = [c for c in clips if c['duration'] < min_acceptable]
    short_warning = ""
    if short_clips:
        short_warning = f"\n⚠️ Warning: {len(short_clips)} shot(s) shorter than {min_acceptable}s - consider using longer clips if possible."

    # Allow flexible tolerance
    # 允许灵活的容差
    tolerance = getattr(config, 'ALLOW_DURATION_TOLERANCE', 1.0)
    if abs(duration_diff) > tolerance:
        if duration_diff > 0:
            action = "shorten"
            suggestion = f"Try trimming {duration_diff:.2f}s from the end."
        else:
            action = "extend"
            suggestion = f"Try adding {abs(duration_diff):.2f}s more footage."

        return (
            f"❌ Error: Duration mismatch! Your total duration is {total_duration:.2f}s but target is {target_length_sec:.2f}s.\n"
            f"Current selection:\n{duration_line}"
            f"Difference: {abs(duration_diff):.2f}s ({action} needed)\n"
            f"Suggestion: {suggestion}{short_warning}\n"
            f"⚠️ Please adjust your shot selection before calling finish."
        )

    # If duration exceeds target by small amount, provide trimming suggestion
    # 如果时长略微超过目标，提供修剪建议
    tolerance = getattr(config, 'ALLOW_DURATION_TOLERANCE', 1.0)
    if 0 < duration_diff <= tolerance:
        new_end_sec = clips[-1]['end_sec'] - duration_diff
        new_end = seconds_to_hhmmss(new_end_sec)
        return (
            f"✅ OK: Shot validation passed (will auto-trim {duration_diff:.2f}s from last clip).\n"
            f"Current selection:\n{duration_line}"
            f"Target duration: {target_length_sec:.2f}s\n"
            f"Auto-adjusted end time: {new_end}\n"
            f"You can proceed with finish.{short_warning}"
        )

    # Validation passed
    # 验证通过
    status_msg = "✅ OK: Shot validation passed.\n"
    if len(clips) > 1:
        status_msg += f"✓ {len(clips)} shots stitched successfully with proper continuity\n"

    return (
        f"{status_msg}"
        f"Current selection:\n{duration_line}"
        f"Target duration: {target_length_sec:.2f}s\n"
        f"Duration match: ✓{short_warning}\n"
        f"You can proceed with finish."
    )



class ReviewerAgent:
    """
    ReviewerAgent reviews shot selections generated by DVDCoreAgent.
    审核器智能体 - 审核由 DVDCoreAgent 生成的镜头选择。
    
    The Core should pass review before calling finish.
    核心智能体在调用 finish 之前应该通过审核。
    
    This agent performs:
    此智能体执行：
    1. Video frame extraction from specified time ranges
    1. 从指定时间范围提取视频帧
    2. VLM-based aesthetic quality analysis
    2. 基于 VLM 的美学质量分析
    3. Protagonist detection and tracking
    3. 主角检测和跟踪
    4. Parallel processing for efficiency
    4. 并行处理以提高效率
    
    Java 类比：类似一个质检服务，结合计算机视觉和 AI 模型进行多维度评估。
    """

    def __init__(self, frame_folder_path=None, video_path=None):
        """
        Initialize ReviewerAgent.
        初始化审核器智能体。

        Args:
            frame_folder_path: 提取的视频帧文件夹路径（可选） (Path to extracted video frames, optional)
            video_path: 视频文件路径 (Path to the video file)
        """
        self.frame_folder_path = frame_folder_path
        self.video_path = video_path

    def cleanup(self):
        """
        Clean up resources to prevent memory leaks.
        清理资源以防止内存泄漏。
        
        Clears thread-local video readers and triggers garbage collection.
        清除线程本地视频阅读器并触发垃圾回收。
        
        Java 类比：类似 Closeable 接口的 close() 方法或 try-with-resources。
        """
        _clear_thread_video_reader()
        gc.collect()

    def _compute_frame_indices(self, start_sec: float, end_sec: float, fps: float, max_frames: Optional[int] = None) -> list:
        """
        Compute frame indices for a time range using native fps with a hard cap.
        使用原生帧率计算时间范围内的帧索引，并有硬性上限。
        
        This function:
        此函数：
        1. Converts time range to frame indices based on FPS
        1. 根据 FPS 将时间范围转换为帧索引
        2. Applies downsampling if frame count exceeds max_frames
        2. 如果帧数超过 max_frames 则应用下采样
        3. Ensures the last frame is always included
        3. 确保始终包含最后一帧
        
        Args:
            start_sec: 开始时间（秒） (Start time in seconds)
            end_sec: 结束时间（秒） (End time in seconds)
            fps: 视频帧率 (Video frames per second)
            max_frames: 最大帧数限制（可选） (Maximum frame limit, optional)
        Returns:
            帧索引列表 (List of frame indices)
        
        Example:
        示例：
        >>> agent._compute_frame_indices(0.0, 5.0, 24, max_frames=10)
        [0, 12, 24, 36, 48, 60, 72, 84, 96, 120]  # 下采样后保留 10 帧
        """
        if fps <= 0:
            return []
        # 计算起始和结束帧索引
        start_f = max(0, int(start_sec * fps))
        end_f = max(0, int(end_sec * fps))
        if end_f < start_f:
            return []
        # 生成所有帧索引
        indices = list(range(start_f, end_f + 1))
        # 如果帧数超过限制，进行下采样
        if max_frames and len(indices) > max_frames:
            import math
            # 计算步长：总帧数 / 最大帧数
            stride = max(1, math.ceil(len(indices) / max_frames))
            # 按步长采样
            indices = indices[::stride]
            # 确保最后一帧被包含
            if indices and indices[-1] != end_f:
                indices.append(end_f)
            # 如果仍然超过限制，截断并保留最后一帧
            if len(indices) > max_frames:
                indices = indices[:max_frames - 1] + [end_f]
        return indices

    def _call_video_analysis_model(self, messages: list) -> Optional[str]:
        """
        Call VIDEO_ANALYSIS_MODEL using the same settings as fine_grained_shot_trimming.
        调用 VIDEO_ANALYSIS_MODEL，使用与 fine_grained_shot_trimming 相同的设置。
        
        This method:
        此方法：
        1. Makes API calls to the video analysis model
        1. 调用视频分析模型的 API
        2. Retries up to 3 times on failure
        2. 失败时最多重试 3 次
        3. Returns the model's response content
        3. 返回模型的响应内容
        
        Args:
            messages: LLM 消息列表，包含角色和内容 (List of LLM messages with role and content)
        Returns:
            模型响应的文本内容，失败时返回 None (Model response text, None on failure)
        
        Java 类比：类似带有重试机制的 HTTP 客户端调用。
        """
        tries = 3
        while tries > 0:
            tries -= 1
            try:
                # 构建 API 调用参数
                kwargs = dict(
                    model=config.VIDEO_ANALYSIS_MODEL,  # 模型名称
                    messages=messages,  # 消息列表
                    max_tokens=getattr(config, "VIDEO_ANALYSIS_MODEL_MAX_TOKEN", 2048),  # 最大 token 数
                    temperature=0.0,  # 温度参数（0.0 表示确定性输出）
                )
                # 如果配置了自定义端点，添加到参数中
                if config.VIDEO_ANALYSIS_ENDPOINT:
                    kwargs["api_base"] = config.VIDEO_ANALYSIS_ENDPOINT
                # 如果配置了 API 密钥，添加到参数中
                if config.VIDEO_ANALYSIS_API_KEY:
                    kwargs["api_key"] = config.VIDEO_ANALYSIS_API_KEY
                # 调用 LiteLLM 完成 API
                raw = litellm.completion(**kwargs)
                # 返回第一个选择的响应内容
                return raw.choices[0].message.content
            except Exception as e:
                print(f"❌ [Reviewer] video analysis call failed: {e}")
                # 如果没有重试次数了，返回 None
                if tries == 0:
                    return None
        return None


    def check_face_quality_vlm(
        self,
        video_path: A[str, D("Path to the video file.")],
        time_range: A[str, D("The time range to check (e.g., '00:13:28 to 00:13:40').")],
        main_character_name: A[str, D("Name of the main character/protagonist to look for. Default: 'the main character'")] = "the main character",
        min_protagonist_ratio: A[float, D("Minimum required ratio of frames where protagonist is the main focus (0.0-1.0). Default: 0.7 (70%)")] = 0.7,
        min_box_size: A[int, D("Minimum bounding box size in pixels. Default: 50")] = 50,
        return_frame_data: A[bool, D("Whether to return frame-level protagonist data from the same VLM call.")] = False,
    ) -> str | tuple[str, list]:
        """
        Check face quality using VLM frame-by-frame detection.
        使用 VLM 逐帧检测检查面部质量。
        
        This function loops through frames in the time range, calls VLM to detect the protagonist
        in each frame and get bounding box coordinates, then calculates break_ratio based on detection results.
        此函数遍历时间范围内的帧，调用 VLM 检测每帧中的主角并获取边界框坐标，
        然后根据检测结果计算中断比例。
        
        The workflow:
        工作流程：
        1. Parse time range and validate inputs
        1. 解析时间范围并验证输入
        2. Extract frames from video with downsampling
        2. 从视频中提取帧并进行下采样
        3. Process frames in parallel batches using ThreadPoolExecutor
        3. 使用 ThreadPoolExecutor 并行批处理帧
        4. Call VLM to detect protagonist in each frame
        4. 调用 VLM 检测每帧中的主角
        5. Calculate protagonist ratio and compare against threshold
        5. 计算主角比例并与阈值比较
        6. Return detailed result message
        6. 返回详细的结果消息
        
        Args:
            video_path: 视频文件路径 (Path to the video file)
            time_range: 时间范围，格式如 "HH:MM:SS to HH:MM:SS" (Time range string)
            main_character_name: 要检测的主要角色名称（默认 "the main character"） (Main character name)
            min_protagonist_ratio: 最小主角比例阈值（默认 0.7 = 70%） (Minimum protagonist ratio threshold)
            min_box_size: 最小边界框尺寸（像素，默认 50） (Minimum bounding box size in pixels)
            return_frame_data: 是否返回逐帧数据 (Whether to return frame-level data)
        Returns:
            成功或失败的消息，如果 return_frame_data=True 则返回 (message, frame_data) 元组
            (Success/failure message, or (message, frame_data) tuple if return_frame_data=True)
        
        Example:
        示例：
            >>> check_face_quality_vlm("/path/to/video.mp4", "00:10:00 to 00:10:10", "Bruce Wayne", min_protagonist_ratio=0.7)
        """
        # Parse time range
        # 解析时间范围
        match = re.search(r'([\d:.]+)\s+to\s+([\d:.]+)', time_range, re.IGNORECASE)
        if not match:
            return f"❌ Error: Could not parse time range '{time_range}'. Please use format 'HH:MM:SS to HH:MM:SS'."

        try:
            fps = getattr(config, "VIDEO_FPS", 24) or 24
            start_sec = hhmmss_to_seconds(match.group(1), fps=fps)
            end_sec = hhmmss_to_seconds(match.group(2), fps=fps)
            duration_sec = end_sec - start_sec

            if duration_sec <= 0:
                return f"❌ Error: Invalid time range. End time must be greater than start time."
        except Exception as e:
            return f"❌ Error parsing time range: {e}"

        if not os.path.exists(video_path):
            return f"❌ Error: Video file not found: {video_path}"

        print(f"🔍 [Reviewer: VLM] Analyzing {time_range} ({duration_sec:.2f}s)...")

        try:
            break_frames = 0  # 中断帧计数（未检测到主角的帧）
            total_sampled_frames = 0  # 总采样帧数
            detection_details = []  # 检测详情列表
            frame_results = []  # 逐帧结果列表
            verbose_frame_log = bool(getattr(config, "VLM_FACE_LOG_EACH_FRAME", False))  # 是否记录每帧日志

            # 获取最大帧数限制
            max_frames = int(getattr(config, "CORE_MAX_FRAMES", getattr(config, "TRIM_SHOT_MAX_FRAMES", 240)))
            # 获取线程本地视频阅读器
            vr = _get_thread_video_reader(video_path)
            if vr is None:
                return "❌ Error: Unable to initialize video reader."
            video_fps = float(vr.get_avg_fps())  # 获取视频实际帧率
            # 计算要处理的帧索引（带下采样）
            frame_indices = self._compute_frame_indices(start_sec, end_sec, video_fps, max_frames=max_frames)
            if not frame_indices:
                return f"❌ Error: No frames to process in the specified time range."

            if verbose_frame_log:
                print(f"🎞️  [Reviewer: VLM] Decoding video; processing {len(frame_indices)} frames...")

            # 批量获取帧数据
            frames = vr.get_batch(frame_indices).asnumpy()
            frame_items = list(zip(frame_indices, frames))
            # 获取批处理配置
            batch_size = int(getattr(config, "VLM_FACE_BATCH_SIZE", 8))  # 每批帧数
            batch_concurrency = int(getattr(config, "VLM_FACE_BATCH_CONCURRENCY", 16))  # 并发批次数

            # 将帧分成多个批次
            batches = [
                frame_items[batch_start:batch_start + batch_size]
                for batch_start in range(0, len(frame_items), batch_size)
            ]
            batch_results_list = [None] * len(batches)  # 存储每批的结果

            # 使用线程池并行处理批次
            with ThreadPoolExecutor(max_workers=min(batch_concurrency, len(batches) or 1)) as executor:
                future_to_idx = {}  # Future 到批次索引的映射
                for idx, batch in enumerate(batches):
                    batch_indices = [frame_idx for frame_idx, _ in batch]
                    batch_frames = [frame for _, frame in batch]
                    # 提交异步任务
                    future = executor.submit(
                        self._detect_protagonist_in_frames_vlm,
                        frame_arrays=batch_frames,
                        frame_indices=batch_indices,
                        main_character_name=main_character_name,
                        min_box_size=min_box_size,
                    )
                    future_to_idx[future] = idx

                # 收集结果（按完成顺序）
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        batch_results_list[idx] = future.result()
                    except Exception:
                        batch_results_list[idx] = None

            # 处理所有批次的结果
            for batch, batch_results in zip(batches, batch_results_list):
                if not batch_results or len(batch_results) != len(batch):
                    return "❌ Error: VLM batch detection failed or returned mismatched results."

                for (frame_idx, _), detection_result in zip(batch, batch_results):
                    total_sampled_frames += 1
                    is_break_frame = detection_result["is_break"]  # 是否为中断帧
                    reason = detection_result["reason"]  # 检测原因
                    status = "❌" if is_break_frame else "✅"

                    time_at_frame = frame_idx / video_fps  # 计算帧的时间戳
                    if verbose_frame_log:
                        print(f"  Frame {frame_idx:6d} | Time: {time_at_frame:7.2f}s | {status:15s} | {reason}")

                    # 记录帧结果
                    frame_results.append({
                        "frame_idx": frame_idx,
                        "time_sec": time_at_frame,
                        "protagonist_detected": not is_break_frame,
                        "bounding_box": detection_result.get("bounding_box"),
                        "confidence": detection_result.get("confidence", 0.0),
                        "reason": reason
                    })

                    if is_break_frame:
                        break_frames += 1
                        detection_details.append({
                            "frame": frame_idx,
                            "time": time_at_frame,
                            "reason": reason
                        })

            # Calculate break ratio
            # 计算中断比例
            if total_sampled_frames == 0:
                return f"❌ Error: No frames were successfully processed."

            break_ratio = break_frames / total_sampled_frames
            non_break_ratio = 1.0 - break_ratio  # 非中断比例（主角存在比例）

            # Prepare result message
            # 准备结果消息
            result_msg = f"\n[VLM Face Quality Check Results (Frame-by-Frame)]\n"
            result_msg += f"Time range: {time_range} ({duration_sec:.2f}s)\n"
            result_msg += f"Character: {main_character_name}\n"
            result_msg += f"Sampled frames: {total_sampled_frames}\n"
            result_msg += f"Break frames: {break_frames}/{total_sampled_frames}\n"
            result_msg += f"Protagonist ratio: {non_break_ratio * 100:.1f}%\n"
            result_msg += f"Required ratio: {min_protagonist_ratio * 100:.1f}%\n"

            # Check if ratio meets threshold
            # 检查比例是否达到阈值
            if non_break_ratio < min_protagonist_ratio:
                # 未达到阈值，返回失败消息
                result_msg += f"\n❌ FAILED: Protagonist ratio ({non_break_ratio * 100:.1f}%) is below minimum threshold ({min_protagonist_ratio * 100:.1f}%)\n"

                if detection_details:
                    result_msg += f"\nBreak frame examples (first 5):\n"
                    for detail in detection_details[:5]:
                        result_msg += f"  - Frame {detail['frame']} ({detail['time']:.2f}s): {detail['reason']}\n"

                result_msg += f"\n⚠️ This shot does not maintain sufficient focus on {main_character_name}. Please select a different shot."
                if return_frame_data:
                    return result_msg, frame_results
                return result_msg
            else:
                # 达到阈值，返回成功消息
                result_msg += f"\n✅ PASSED: Protagonist ratio ({non_break_ratio * 100:.1f}%) meets the minimum threshold.\n"
                result_msg += f"Shot maintains good focus on {main_character_name}. You can proceed with this shot."
                if return_frame_data:
                    return result_msg, frame_results
                return result_msg

        except Exception as e:
            import traceback
            traceback.print_exc()
            if return_frame_data:
                return f"❌ Error during VLM frame-by-frame detection: {str(e)}", []
            return f"❌ Error during VLM frame-by-frame detection: {str(e)}"
        finally:
            # 清理资源，释放内存
            frames = None
            frame_items = None
            batches = None
            batch_results_list = None
            gc.collect()


    def _evaluate_protagonist_detection(self, detection: dict, min_box_size: int) -> dict:
        """
        Evaluate a raw VLM detection result and apply size/role rules.
        评估原始 VLM 检测结果并应用尺寸/角色规则。
        
        This function applies business rules to determine if a frame is a "break" frame:
        此函数应用业务规则来判断帧是否为“中断”帧：
        1. Minor character detected → break
        1. 检测到配角 → 中断
        2. No protagonist detected → break
        2. 未检测到主角 → 中断
        3. No bounding box → break
        3. 无边界框 → 中断
        4. Protagonist too small → break
        4. 主角太小 → 中断
        5. Otherwise → valid frame
        5. 否则 → 有效帧
        
        Args:
            detection: VLM 的原始检测结果字典 (Raw VLM detection result dict)
            min_box_size: 最小边界框尺寸（像素） (Minimum bounding box size in pixels)
        Returns:
            评估后的结果字典，包含 is_break、reason、bounding_box、confidence
            (Evaluated result dict with is_break, reason, bounding_box, confidence)
        
        Java 类比：类似一个 Validator 或 Rule Engine，按优先级应用多个验证规则。
        """
        # 提取检测结果的各个字段
        protagonist_detected = detection.get("protagonist_detected", False)
        is_minor_character = detection.get("is_minor_character", False)
        bounding_box = detection.get("bounding_box", None)
        confidence = detection.get("confidence", 0.0)
        reason_text = detection.get("reason", "")

        # 规则 1：检测到配角 → 中断帧
        if is_minor_character:
            return {
                "is_break": True,
                "reason": f"minor_character_detected ({reason_text})",
                "bounding_box": None,
                "confidence": confidence
            }

        # 规则 2：未检测到主角 → 中断帧
        if not protagonist_detected:
            return {
                "is_break": True,
                "reason": f"no_protagonist ({reason_text})",
                "bounding_box": None,
                "confidence": confidence
            }

        # 规则 3：无边界框 → 中断帧
        if bounding_box is None:
            return {
                "is_break": True,
                "reason": "no_bounding_box",
                "bounding_box": None,
                "confidence": confidence
            }

        # 规则 4：检查边界框尺寸
        box_width = bounding_box.get("width", 0)
        box_height = bounding_box.get("height", 0)
        box_size = min(box_width, box_height)  # 取宽高中的较小值
        relaxed_min_size = max(30, min_box_size // 2)  # 放宽的最小尺寸（原尺寸的一半，最小 30）

        if box_size < relaxed_min_size:
            return {
                "is_break": True,
                "reason": f"protagonist_too_small ({box_size}px < {relaxed_min_size}px)",
                "bounding_box": bounding_box,
                "confidence": confidence
            }

        # 规则 5：所有检查通过 → 有效帧
        return {
            "is_break": False,
            "reason": f"protagonist_ok (size={box_size}px, conf={confidence:.2f})",
            "bounding_box": bounding_box,
            "confidence": confidence
        }


    def _detect_protagonist_in_frames_vlm(
        self,
        frame_arrays: List[np.ndarray],
        frame_indices: List[int],
        main_character_name: str,
        min_box_size: int
    ) -> Optional[List[dict]]:
        """
        Call VLM once for multiple frames and return per-frame results in order.
        一次性调用 VLM 处理多帧并按顺序返回每帧的结果。
        
        This method:
        此方法：
        1. Builds a multi-modal prompt with text + images
        1. 构建包含文本和图像的多模态提示词
        2. Converts numpy arrays to base64 encoded images
        2. 将 numpy 数组转换为 base64 编码的图像
        3. Calls VLM API with all frames in one request
        3. 在一次请求中调用 VLM API 处理所有帧
        4. Parses JSON response and validates structure
        4. 解析 JSON 响应并验证结构
        5. Evaluates each detection result using business rules
        5. 使用业务规则评估每个检测结果
        
        Args:
            frame_arrays: 帧图像数组列表 (List of frame image arrays)
            frame_indices: 对应的帧索引列表 (Corresponding frame indices)
            main_character_name: 主要角色名称 (Main character name)
            min_box_size: 最小边界框尺寸（像素） (Minimum bounding box size in pixels)
        Returns:
            评估后的检测结果列表，失败时返回 None
            (List of evaluated detection results, None on failure)
        
        Java 类比：类似批量图片处理的 REST API 调用，一次请求处理多个资源。
        """
        # 验证输入参数
        if not frame_arrays or not frame_indices or len(frame_arrays) != len(frame_indices):
            return None

        # 构建提示词模板
        prompt = VLM_PROTAGONIST_DETECTION_PROMPT.format(
            main_character_name=main_character_name,
            frame_count=len(frame_indices),
            frame_indices=frame_indices,
        )

        # 初始化变量用于 finally 清理
        user_content = None
        messages = None
        content_str = None
        content = None
        detections = None
        try:
            # 构建用户内容：文本提示 + 图像列表
            user_content = [{"type": "text", "text": prompt}]
            for frame in frame_arrays:
                # 将 numpy 数组转换为 base64 编码
                b64 = array_to_base64(frame)
                if b64:
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

            # 构建消息列表（系统消息 + 用户消息）
            messages = [
                {"role": "system", "content": "You are an expert at character detection and localization in video frames."},
                {"role": "user", "content": user_content}
            ]

            # 调用 VLM API
            content_str = self._call_video_analysis_model(messages)
            if not content_str:
                return None

            # 清理响应内容
            content = content_str.strip()
            # 提取 JSON 代码块（如果存在）
            json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if json_block_match:
                content = json_block_match.group(1).strip()

            # 解析 JSON
            detections = json.loads(content)
            if not isinstance(detections, list):
                return None
            # 验证返回数量与帧数匹配
            if len(detections) != len(frame_indices):
                return None

            # 评估每个检测结果
            results = []
            for detection in detections:
                results.append(self._evaluate_protagonist_detection(detection, min_box_size))
            return results
        except Exception:
            # 任何异常都返回 None
            return None
        finally:
            # 清理大型对象，释放内存
            user_content = None
            messages = None
            content_str = None
            content = None
            detections = None
            gc.collect()


    def get_protagonist_frame_data(
        self,
        video_path: str,
        time_range: str,
        main_character_name: str = "the main character",
        min_box_size: int = 50,
    ) -> list:
        """
        Get frame-level protagonist detection data for a time range.
        获取时间范围内的逐帧主角检测数据。
            
        Returns structured data instead of a summary string.
        返回结构化数据而非摘要字符串。
            
        This method:
        此方法：
        1. Parses and validates the time range
        1. 解析并验证时间范围
        2. Extracts frames with downsampling
        2. 提取帧并进行下采样
        3. Processes frames in parallel batches
        3. 并行批处理帧
        4. Calls VLM for protagonist detection
        4. 调用 VLM 进行主角检测
        5. Returns detailed per-frame results
        5. 返回详细的逐帧结果
            
        Args:
            video_path: 视频文件路径 (Path to the video file)
            time_range: 时间范围，格式如 "HH:MM:SS to HH:MM:SS" (Time range string)
            main_character_name: 主要角色名称（默认 "the main character"） (Main character name)
            min_box_size: 最小边界框尺寸（像素，默认 50） (Minimum bounding box size in pixels)
        Returns:
            逐帧检测结果列表，每个元素包含 frame_idx、time_sec、protagonist_detected 等字段
            (List of frame detection results with detailed fields)
            
        Example return structure:
        示例返回结构：
        [
            {
                "frame_idx": 120,
                "time_sec": 5.0,
                "protagonist_detected": True,
                "bounding_box": {"x": 100, "y": 50, "width": 200, "height": 250},
                "confidence": 0.95,
                "reason": "protagonist_ok (size=200px, conf=0.95)"
            },
            ...
        ]
        """
        # Parse time range
        # 解析时间范围
        match = re.search(r'([\d:.]+)\s+to\s+([\d:.]+)', time_range, re.IGNORECASE)
        if not match:
            print(f"❌ [Reviewer] Error: Could not parse time range '{time_range}'")
            return []
    
        try:
            fps = getattr(config, "VIDEO_FPS", 24) or 24
            start_sec = hhmmss_to_seconds(match.group(1), fps=fps)
            end_sec = hhmmss_to_seconds(match.group(2), fps=fps)
            duration_sec = end_sec - start_sec
    
            if duration_sec <= 0:
                print(f"❌ [Reviewer] Error: Invalid time range")
                return []
        except Exception as e:
            print(f"❌ [Reviewer] Error parsing time range: {e}")
            return []
    
        frame_results = []  # 存储逐帧结果
    
        # 初始化变量用于 finally 清理
        vr = None
        frames = None
        frame_items = None
        batches = None
        batch_results_list = None
        try:
            # 检查视频文件是否存在
            if not os.path.exists(video_path):
                print(f"❌ [Reviewer] Error: Video file not found: {video_path}")
                return []
    
            # 获取线程本地视频阅读器
            vr = _get_thread_video_reader(video_path)
            if vr is None:
                print("❌ [Reviewer] Error: Unable to initialize video reader.")
                return []
            video_fps = float(vr.get_avg_fps())  # 获取视频实际帧率
            # 计算帧索引（带下采样）
            max_frames = int(getattr(config, "CORE_MAX_FRAMES", getattr(config, "TRIM_SHOT_MAX_FRAMES", 240)))
            frame_indices = self._compute_frame_indices(start_sec, end_sec, video_fps, max_frames=max_frames)
            if not frame_indices:
                return []
    
            # 批量获取帧数据
            frames = vr.get_batch(frame_indices).asnumpy()
            frame_items = list(zip(frame_indices, frames))
            # 获取批处理配置
            batch_size = int(getattr(config, "VLM_FACE_BATCH_SIZE", 8))
            batch_concurrency = int(getattr(config, "VLM_FACE_BATCH_CONCURRENCY", 16))
    
            # 将帧分成多个批次
            batches = [
                frame_items[batch_start:batch_start + batch_size]
                for batch_start in range(0, len(frame_items), batch_size)
            ]
            batch_results_list = [None] * len(batches)
    
            # 使用线程池并行处理批次
            with ThreadPoolExecutor(max_workers=min(batch_concurrency, len(batches) or 1)) as executor:
                future_to_idx = {}
                for idx, batch in enumerate(batches):
                    batch_indices = [frame_idx for frame_idx, _ in batch]
                    batch_frames = [frame for _, frame in batch]
                    # 提交异步任务
                    future = executor.submit(
                        self._detect_protagonist_in_frames_vlm,
                        frame_arrays=batch_frames,
                        frame_indices=batch_indices,
                        main_character_name=main_character_name,
                        min_box_size=min_box_size,
                    )
                    future_to_idx[future] = idx
    
                # 收集结果（按完成顺序）
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        batch_results_list[idx] = future.result()
                    except Exception:
                        batch_results_list[idx] = None
    
            # 处理所有批次的结果
            for batch, batch_results in zip(batches, batch_results_list):
                if not batch_results or len(batch_results) != len(batch):
                    print("❌ [Reviewer] Error: VLM batch detection failed or returned mismatched results.")
                    return []
    
                # 构建逐帧结果
                for (frame_idx, _), detection_result in zip(batch, batch_results):
                    time_at_frame = frame_idx / video_fps  # 计算帧的时间戳
                    frame_data = {
                        "frame_idx": frame_idx,  # 帧索引
                        "time_sec": time_at_frame,  # 时间戳（秒）
                        "protagonist_detected": not detection_result["is_break"],  # 是否检测到主角
                        "bounding_box": detection_result.get("bounding_box"),  # 边界框
                        "confidence": detection_result.get("confidence", 0.0),  # 置信度
                        "reason": detection_result["reason"]  # 检测原因
                    }
                    frame_results.append(frame_data)
    
        except Exception as e:
            print(f"❌ [Reviewer] Error during protagonist detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            # 清理资源，释放内存
            frames = None
            frame_items = None
            batches = None
            batch_results_list = None
            gc.collect()
    
        return frame_results


    def check_aesthetic_quality(
        self,
        video_path: A[str, D("Path to the video file.")],
        time_range: A[str, D("The time range to check (e.g., '00:13:28 to 00:13:40').")],
        min_aesthetic_score: A[float, D("Minimum required aesthetic score (1-5 scale). Default: 3.0")] = 3.0,
        sample_fps: A[float, D("Sampling frame rate for analysis. Default: 2.0")] = 2.0,
    ) -> str:
        """
        Check aesthetic quality of a video clip using VLM analysis.
        使用 VLM 分析检查视频片段的美学质量。
        
        Analyzes visual appeal, lighting, composition, colors, and cinematography.
        分析视觉吸引力、光线、构图、色彩和摄影技术。
        
        This method evaluates multiple aesthetic dimensions:
        此方法评估多个美学维度：
        1. Lighting quality (natural vs artificial, exposure)
        1. 光线质量（自然光 vs 人造光，曝光）
        2. Color grading and vibrancy
        2. 色彩分级和饱和度
        3. Composition and framing
        3. 构图和取景
        4. Camera work (stability, movement)
        4. 摄影技术（稳定性、运动）
        5. Visual interest and engagement
        5. 视觉吸引力和参与度
        
        Args:
            video_path: 视频文件路径 (Path to the video file)
            time_range: 时间范围，格式如 "HH:MM:SS to HH:MM:SS" (Time range string)
            min_aesthetic_score: 最小美学分数阈值（1-5 分制，默认 3.0） (Minimum aesthetic score threshold)
            sample_fps: 分析用的帧采样率（默认 2.0 fps） (Frame sampling rate for analysis)
        Returns:
            成功或失败的消息，包含详细的美学分析结果
            (Success/failure message with detailed aesthetic analysis)
        
        Example:
        示例：
            >>> check_aesthetic_quality("/path/to/video.mp4", "00:10:00 to 00:10:10", min_aesthetic_score=3.5)
        """
        # 解析时间范围
        match = re.search(r'([\d:.]+)\s+to\s+([\d:.]+)', time_range, re.IGNORECASE)
        if not match:
            return f"❌ Error: Could not parse time range '{time_range}'. Please use format 'HH:MM:SS to HH:MM:SS'."

        try:
            fps = getattr(config, "VIDEO_FPS", 24) or 24
            start_sec = hhmmss_to_seconds(match.group(1), fps=fps)
            end_sec = hhmmss_to_seconds(match.group(2), fps=fps)
            duration_sec = end_sec - start_sec

            if duration_sec <= 0:
                return "❌ Error: Invalid time range. End time must be greater than start time."
        except Exception as e:
            return f"❌ Error parsing time range: {e}"

        if not os.path.exists(video_path):
            return f"❌ Error: Video file not found: {video_path}"

        print(f"✨ [Reviewer: Aesthetics] Analyzing {time_range} ({duration_sec:.2f}s)...")

        # 初始化变量用于 finally 清理
        vr = None
        b64_frames = None
        user_content = None
        litellm_messages = None
        content_str = None
        response = None
        try:
            # 获取线程本地视频阅读器
            vr = _get_thread_video_reader(video_path)
            if vr is None:
                return "❌ Error: Unable to initialize video reader."
            video_native_fps = vr.get_avg_fps()  # 获取视频实际帧率
            # 计算帧索引（带下采样）
            max_frames = int(getattr(config, "CORE_MAX_FRAMES", getattr(config, "TRIM_SHOT_MAX_FRAMES", 240)))
            indices = self._compute_frame_indices(start_sec, end_sec, video_native_fps, max_frames=max_frames)
            if indices:
                # 确保索引不超出视频长度
                indices = [i for i in indices if i < len(vr)]
            # 提取帧并转换为 base64
            b64_frames = [array_to_base64(vr.get_batch([i]).asnumpy()[0]) for i in indices] if indices else []
            
            # 构建多模态提示词：文本 + 图像
            user_content = [{"type": "text", "text": VLM_AESTHETIC_ANALYSIS_PROMPT}]
            for b64 in b64_frames:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            
            # 构建消息列表
            litellm_messages = [
                {"role": "system", "content": "You are an expert cinematographer and visual aesthetics analyst."},
                {"role": "user", "content": user_content},
            ]
            
            # 调用 VLM API 进行美学分析
            content_str = self._call_video_analysis_model(litellm_messages)
            response = {"content": content_str} if content_str else None

            if response is None or response.get("content") is None:
                return "⚠️ WARNING: VLM returned no response for aesthetic analysis. Proceeding without validation."

            # 解析响应内容
            content = response["content"].strip()
            # 提取 JSON 代码块（如果存在）
            json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if json_block_match:
                content = json_block_match.group(1).strip()

            # 解析 JSON 响应
            analysis = json.loads(content)

            # 提取各个维度的评分
            overall_score = analysis.get("overall_aesthetic_score", 0.0)  # 总体美学分数
            lighting_score = analysis.get("lighting_score", 0.0)  # 光线分数
            color_score = analysis.get("color_score", 0.0)  # 色彩分数
            composition_score = analysis.get("composition_score", 0.0)  # 构图分数
            camera_work_score = analysis.get("camera_work_score", 0.0)  # 摄影技术分数
            visual_interest_score = analysis.get("visual_interest_score", 0.0)  # 视觉吸引力分数
            strengths = analysis.get("strengths", [])  # 优点列表
            weaknesses = analysis.get("weaknesses", [])  # 缺点列表
            recommendation = analysis.get("recommendation", "UNKNOWN")  # 推荐意见
            detailed_analysis = analysis.get("detailed_analysis", "")  # 详细分析

            # 构建结果消息
            result_msg = "\n[Aesthetic Quality Check Results]\n"
            result_msg += f"Time range: {time_range} ({duration_sec:.2f}s)\n"
            result_msg += f"Overall Aesthetic Score: {overall_score:.2f}/5.0\n"
            result_msg += f"  • Lighting: {lighting_score:.2f}/5.0\n"
            result_msg += f"  • Color: {color_score:.2f}/5.0\n"
            result_msg += f"  • Composition: {composition_score:.2f}/5.0\n"
            result_msg += f"  • Camera Work: {camera_work_score:.2f}/5.0\n"
            result_msg += f"  • Visual Interest: {visual_interest_score:.2f}/5.0\n"
            result_msg += f"Recommendation: {recommendation}\n"
            result_msg += f"Minimum Required: {min_aesthetic_score:.2f}/5.0\n"

            # 添加优点（最多 3 个）
            if strengths:
                result_msg += "\nStrengths:\n"
                for strength in strengths[:3]:
                    result_msg += f"  ✓ {strength}\n"

            # 添加缺点（最多 3 个）
            if weaknesses:
                result_msg += "\nWeaknesses:\n"
                for weakness in weaknesses[:3]:
                    result_msg += f"  ✗ {weakness}\n"

            # 添加详细分析
            if detailed_analysis:
                result_msg += f"\nAnalysis: {detailed_analysis}\n"

            # 检查是否达到阈值
            if overall_score < min_aesthetic_score:
                # 未达到阈值，返回失败消息
                result_msg += (
                    f"\n❌ FAILED: Aesthetic score ({overall_score:.2f}/5.0) is below minimum "
                    f"threshold ({min_aesthetic_score:.2f}/5.0)\n"
                )
                result_msg += "\n⚠️ This shot does not meet the aesthetic quality requirements. Please select a shot with:\n"
                result_msg += "  • Better lighting (natural light preferred)\n"
                result_msg += "  • Improved composition (well-framed, balanced)\n"
                result_msg += "  • More vibrant colors and good contrast\n"
                result_msg += "  • Stable camera work\n"
                result_msg += "  • More visually interesting content\n"
                return result_msg

            # 达到阈值，返回成功消息
            result_msg += (
                f"\n✅ PASSED: Aesthetic score ({overall_score:.2f}/5.0) meets the minimum threshold.\n"
            )
            if overall_score >= 4.0:
                result_msg += "⭐ Excellent visual quality! This shot is highly recommended for the final edit.\n"
            result_msg += "You can proceed with this shot."
            return result_msg

        except json.JSONDecodeError as e:
            # JSON 解析失败，返回警告但允许继续
            return f"⚠️ WARNING: Could not parse VLM response for aesthetic analysis: {e}\n\nProceeding without validation."
        except Exception as e:
            # 其他异常，打印堆栈并返回警告
            import traceback
            traceback.print_exc()
            return f"⚠️ WARNING: Error during aesthetic analysis: {str(e)}\n\nProceeding without validation."
        finally:
            # 清理资源，释放内存
            b64_frames = None
            user_content = None
            litellm_messages = None
            content_str = None
            response = None
            gc.collect()


    def review(self, shot_proposal: dict, context: dict, used_time_ranges: list = None) -> dict:
        """
        Review whether the shot selection meets requirements.
        审核镜头选择是否符合要求。
        
        This is the main entry point for the reviewer agent.
        这是审核器智能体的主要入口点。
        
        The review process:
        审核流程：
        1. Validate time range format and duration
        1. 验证时间范围格式和时长
        2. Check for overlap with already used clips
        2. 检查与已使用片段的重叠
        3. (TODO) Content match checks
        3. （待实现）内容匹配检查
        4. Build comprehensive feedback
        4. 构建全面的反馈
        
        Args:
            shot_proposal: 镜头选择信息字典 (Shot selection info dict)
                - answer: str, 选择的时间范围（如 "[shot: 00:10:00 to 00:10:07]"）
                - target_length_sec: float, 目标时长
            context: 当前镜头上下文字典 (Current shot context dict)
                - content: str, 目标内容描述
                - emotion: str, 目标情绪
                - section_idx: int, 当前段落索引
                - shot_idx: int, 当前镜头索引
            used_time_ranges: 已使用的时间范围列表 [(start_sec, end_sec), ...]
        Returns:
            审核结果字典，包含 approved、feedback、issues、suggestions
            (Review result dict with approved, feedback, issues, suggestions)
        
        Example return structure:
        示例返回结构：
        {
            "approved": True/False,
            "feedback": "详细反馈消息",
            "issues": ["问题列表"],
            "suggestions": ["改进建议列表"]
        }
        
        Java 类比：类似一个 Service 层的 validate() 方法，执行多个验证规则并返回 ValidationResult。
        """
        # 初始化已使用时间范围列表
        if used_time_ranges is None:
            used_time_ranges = []

        # 提取镜头提议的关键字段
        answer = shot_proposal.get("answer", "")  # 选择的时间范围字符串
        target_length_sec = shot_proposal.get("target_length_sec", 0.0)  # 目标时长

        issues = []  # 发现的问题列表
        suggestions = []  # 改进建议列表

        # 1. 验证时间范围格式和时长
        finish_review = review_finish(answer, target_length_sec)
        if "❌" in finish_review:
            issues.append(finish_review)

        # 2. 检查与已使用片段的重叠
        # 从 answer 中提取时间范围
        match = re.search(r'\[?shot[\s_]*\d*:\s*([0-9:.]+)\s+to\s+([0-9:.]+)\]?', answer, re.IGNORECASE)
        if match:
            time_range = f"{match.group(1)} to {match.group(2)}"
            overlap_review = review_clip(time_range, used_time_ranges)
            if "❌" in overlap_review:
                issues.append(overlap_review)

        # 3. 内容匹配检查（可以使用 LLM 进行更深入的审核）
        # TODO: 在此添加更多检查，例如：
        # - 选择的片段是否匹配目标内容/情绪
        # - 视觉质量检查
        # - 叙事连贯性检查

        # 构建反馈消息
        if issues:
            # 发现问题，返回失败结果
            feedback = "❌ Review failed with the following issues:\n"
            for i, issue in enumerate(issues, 1):
                feedback += f"\nIssue {i}:\n{issue}\n"

            # 生成通用建议
            suggestions.append("Adjust your shot selection based on the issues above.")
            # 根据具体问题类型生成针对性建议
            if "Duration mismatch" in str(issues) or "duration" in str(issues).lower():
                suggestions.append("Adjust start/end times to match the target duration.")
            if "OVERLAP" in str(issues) or "overlap" in str(issues).lower():
                suggestions.append("Choose a time range that does not overlap with previously used clips.")

            feedback += "\nSuggestions:\n" + "\n".join(f"- {s}" for s in suggestions)

            return {
                "approved": False,  # 审核未通过
                "feedback": feedback,  # 详细反馈消息
                "issues": issues,  # 问题列表
                "suggestions": suggestions  # 改进建议
            }
        else:
            # 没有问题，返回成功结果
            return {
                "approved": True,  # 审核通过
                "feedback": f"✅ Review passed! Shot selection meets requirements.\n{finish_review}",  # 成功消息
                "issues": [],  # 无问题
                "suggestions": []  # 无建议
            }
