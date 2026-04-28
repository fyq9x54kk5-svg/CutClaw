"""
Video preprocessing utilities - frame extraction, shot detection, and scene analysis.
视频预处理工具 - 帧提取、镜头检测和场景分析。

This module provides functions for:
此模块提供以下功能：
1. Video reader creation with GPU/CPU fallback
1. 创建带 GPU/CPU 回退的视频阅读器
2. Shot boundary detection using scene detection algorithms
2. 使用场景检测算法进行镜头边界检测
3. Frame extraction and sampling
3. 帧提取和采样
4. Scene boundary adjustment and validation
4. 场景边界调整和验证

Java 类比：类似一个视频处理工具类，结合 decord（视频读取）和 scenedetect（镜头检测）。
"""
import os
from concurrent.futures import ProcessPoolExecutor  # 进程池，类似 Java 的 ForkJoinPool
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm  # 进度条库
from decord import VideoReader, cpu, gpu  # 视频读取库，支持 GPU 加速
from decord._ffi.base import DECORDError  # decord 错误类型
from PIL import Image  # 图像处理库
from scenedetect import AdaptiveDetector, SceneManager, open_video  # 场景检测库
from scenedetect.backends.pyav import VideoStreamAv  # PyAV 后端


def _ensure_dir(path: str) -> None:
    """
    Ensure directory exists, create if not.
    确保目录存在，不存在则创建。
    
    Java 类比：类似 Files.createDirectories()（Java 7+）
    """
    os.makedirs(path, exist_ok=True)


_decord_ctx = None  # 全局变量：缓存 decord 上下文（GPU 或 CPU）


def _get_decord_ctx():
    """
    Return GPU context if decord was compiled with CUDA support, otherwise CPU.
    如果 decord 编译时支持 CUDA，返回 GPU 上下文，否则返回 CPU。
    
    The check is cached after the first call via _create_decord_reader, which
    has the video path needed to trigger decord's CUDA validation.
    首次调用后通过 _create_decord_reader 缓存检查结果，
    该函数有触发 decord CUDA 验证所需的视频路径。
    
    Returns:
        GPU 或 CPU 上下文对象 (GPU or CPU context object)
    
    Java 类比：类似单例模式，首次初始化后缓存结果。
    """
    global _decord_ctx
    if _decord_ctx is None:
        return gpu(0)  # 尝试使用 GPU（_create_decord_reader 会确认是否可用）
    return _decord_ctx


def _adjust_scene_boundaries(scenes: List[List[int]]) -> List[List[int]]:
    """
    Adjust scene boundaries to ensure no gaps or overlaps.
    调整场景边界以确保没有间隙或重叠。
    
    This function:
    此函数：
    1. Ensures consecutive scenes are contiguous (no gaps)
    1. 确保连续场景是连续的（无间隙）
    2. Merges zero-duration shots into previous scene
    2. 将零时长的镜头合并到前一个场景
    
    Args:
        scenes: 场景列表，每个场景是 [start_frame, end_frame] (List of scenes, each [start_frame, end_frame])
    Returns:
        调整后的场景列表 (Adjusted list of scenes)
    
    Java 类比：类似合并重叠区间的算法。
    """
    if not scenes or len(scenes) <= 1:
        return scenes

    adjusted = [scenes[0]]  # 初始化结果列表
    for i in range(1, len(scenes)):
        start_frame = adjusted[-1][1] + 1  # 当前场景的起始帧 = 前一场景结束帧 + 1
        end_frame = scenes[i][1]  # 当前场景的结束帧
        if start_frame >= end_frame:
            # zero-duration shot: merge into previous
            # 零时长镜头：合并到前一个场景
            adjusted[-1][1] = end_frame
        else:
            adjusted.append([start_frame, end_frame])
    return adjusted


def _timecode_to_seconds(timecode: str) -> float:
    """
    Convert timecode string (HH:MM:SS.mmm) to seconds.
    将时间码字符串（HH:MM:SS.mmm）转换为秒数。
    
    Args:
        timecode: 时间码字符串，格式如 "00:01:23.456" (Timecode string, e.g., "00:01:23.456")
    Returns:
        秒数（浮点数） (Seconds as float)
    
    Example:
        >>> _timecode_to_seconds("00:01:30.500")
        90.5
    
    Java 类比：类似解析时间字符串并计算总秒数。
    """
    # 分割时间码：小时、分钟、秒.毫秒
    hours, minutes, seconds_milliseconds = timecode.split(":")
    seconds, milliseconds = seconds_milliseconds.split(".")
    # 计算总秒数
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds) / 1000.0
    )


def _create_decord_reader(
    video_path: str,
    target_resolution: Optional[Tuple[int, int]] = None,
) -> VideoReader:
    """
    Create a decord VideoReader with GPU/CPU fallback and optional resolution scaling.
    创建带 GPU/CPU 回退和可选分辨率缩放的 decord VideoReader。
    
    This function:
    此函数：
    1. Tries to use GPU acceleration if available
    1. 如果可用，尝试使用 GPU 加速
    2. Falls back to CPU if CUDA is not supported
    2. 如果不支持 CUDA，回退到 CPU
    3. Optionally scales video to target resolution
    3. 可选地将视频缩放到目标分辨率
    4. Maintains aspect ratio when using short-side mode
    4. 使用短边模式时保持宽高比
    
    Args:
        video_path: 视频文件路径 (Path to video file)
        target_resolution: 目标分辨率，可以是整数（短边）或 (height, width) 元组
                          (Target resolution: int for short-side or (height, width) tuple)
    Returns:
        配置好的 VideoReader 对象 (Configured VideoReader object)
    
    Java 类比：类似创建一个带硬件加速的视频解码器，支持自动降级。
    """
    global _decord_ctx

    def _make_reader(path, ctx, **kwargs):
        """
        Try to open VideoReader; fall back to CPU if decord lacks CUDA support.
        尝试打开 VideoReader；如果 decord 不支持 CUDA，则回退到 CPU。
        
        Java 类比：类似 try-catch 中的降级策略。
        """
        global _decord_ctx
        try:
            vr = VideoReader(path, ctx=ctx, **kwargs)
            _decord_ctx = ctx  # confirmed working - 确认可以使用
            return vr
        except DECORDError as e:
            if 'CUDA not enabled' in str(e):
                # CUDA 不可用，切换到 CPU
                _decord_ctx = cpu(0)
                return VideoReader(path, ctx=_decord_ctx, **kwargs)
            raise

    # 获取上下文（GPU 或 CPU）
    ctx = _get_decord_ctx()
    if target_resolution is None:
        # 无分辨率要求，直接创建阅读器
        return _make_reader(video_path, ctx, num_threads=8)

    # Parse target_resolution as a short-side constraint (int) or explicit (H, W)
    # 解析目标分辨率：短边约束（整数）或明确的 (H, W)
    if isinstance(target_resolution, (tuple, list)) and len(target_resolution) == 2:
        # Explicit (H, W) — use as-is
        # 明确的 (高度, 宽度) - 直接使用
        return _make_reader(video_path, ctx,
                            height=int(target_resolution[0]),
                            width=int(target_resolution[1]),
                            num_threads=16)

    # Short-side mode: read native resolution first, then scale proportionally
    # 短边模式：先读取原始分辨率，然后按比例缩放
    short_side = int(target_resolution[0]) if isinstance(target_resolution, (tuple, list)) else int(target_resolution)
    # 探测视频：获取原始分辨率
    probe = _make_reader(video_path, ctx)
    native_h, native_w = probe[0].shape[:2]  # 获取第一帧的高度和宽度
    del probe  # 删除探测器，释放内存
    ctx = _decord_ctx or cpu(0)  # 使用确认后的上下文

    if native_h <= native_w:
        # Height is the short side
        # 高度是短边
        target_h = short_side
        # 计算目标宽度，保持宽高比，并确保为偶数（视频编码要求）
        target_w = int(round(native_w * short_side / native_h / 2) * 2)  # keep even
    else:
        # Width is the short side
        # 宽度是短边
        target_w = short_side
        # 计算目标高度，保持宽高比，并确保为偶数
        target_h = int(round(native_h * short_side / native_w / 2) * 2)  # keep even

    return _make_reader(video_path, ctx, height=target_h, width=target_w)


def _save_sampled_frames_to_disk(
    video_reader: VideoReader,
    frame_indices: List[int],
    frames_dir: str,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
) -> List[str]:
    """
    Save sampled video frames to disk as image files.
    将采样的视频帧保存为磁盘上的图像文件。
    
    This function:
    此函数：
    1. Clears existing frame files in the directory
    1. 清除目录中现有的帧文件
    2. Extracts frames at specified indices
    2. 提取指定索引的帧
    3. Saves each frame as JPG/PNG with progress bar
    3. 将每帧保存为 JPG/PNG，带进度条
    
    Args:
        video_reader: decord 视频阅读器 (decord VideoReader)
        frame_indices: 要提取的帧索引列表 (List of frame indices to extract)
        frames_dir: 保存帧的目录路径 (Directory path to save frames)
        image_format: 图像格式，'jpg' 或 'png' (Image format: 'jpg' or 'png')
        jpeg_quality: JPEG 质量（1-100，默认 95） (JPEG quality 1-100, default 95)
    Returns:
        保存的图像文件路径列表 (List of saved image file paths)
    
    Java 类比：类似批量将 BufferedImage 保存为文件。
    """
    # 确定文件扩展名
    file_ext = image_format.lower()
    if file_ext not in {"jpg", "jpeg", "png"}:
        raise ValueError(f"Unsupported image_format: {image_format}")

    # 确保目录存在
    _ensure_dir(frames_dir)

    # 清除现有的帧文件
    for filename in os.listdir(frames_dir):
        if filename.startswith("frame_") and filename.lower().endswith((".jpg", ".jpeg", ".png")):
            os.remove(os.path.join(frames_dir, filename))

    # 如果没有帧索引，返回空列表
    if not frame_indices:
        return []

    # 批量获取帧数据
    sampled = video_reader.get_batch(frame_indices).asnumpy()
    saved_paths: List[str] = []  # 保存的路径列表
    # 限制 JPEG 质量在 1-100 范围内
    quality = max(1, min(100, int(jpeg_quality)))

    # 遍历并保存每一帧
    for i, frame in enumerate(tqdm(sampled, desc="Saving frames", unit="frame")):
        out_path = os.path.join(frames_dir, f"frame_{i:06d}.{file_ext}")  # 输出路径
        image = Image.fromarray(frame)  # 从 numpy 数组创建 PIL 图像
        if file_ext in {"jpg", "jpeg"}:
            image.save(out_path, quality=quality)  # 保存 JPEG，指定质量
        else:
            image.save(out_path)  # 保存 PNG
        saved_paths.append(out_path)

    return saved_paths


def _run_scenedetect(
    video_path: str,
    threshold: float,
    min_scene_len: int,
    end_frame: Optional[int],
    frame_skip: int = 0,
    start_frame: int = 0,
    warmup_start_frame: int = 0,
) -> List:
    """
    Run scenedetect on [warmup_start_frame, end_frame), but only return scenes >= start_frame.
    在 [warmup_start_frame, end_frame) 范围内运行场景检测，但只返回 >= start_frame 的场景。
    
    This function uses AdaptiveDetector to find scene boundaries based on visual changes.
    此函数使用 AdaptiveDetector 根据视觉变化查找场景边界。
    
    Args:
        video_path: 视频文件路径 (Path to video file)
        threshold: 自适应阈值（越高越敏感） (Adaptive threshold, higher = more sensitive)
        min_scene_len: 最小场景长度（帧数） (Minimum scene length in frames)
        end_frame: 结束帧索引，None 表示到视频末尾 (End frame index, None for end of video)
        frame_skip: 跳帧数（加速检测） (Frame skip count for faster detection)
        start_frame: 逻辑起始帧 (Logical start frame)
        warmup_start_frame: 预热起始帧（用于初始化检测器） (Warmup start frame for detector initialization)
    Returns:
        场景列表，每个场景是 (start_timecode, end_timecode) 元组
        (List of scenes, each (start_timecode, end_timecode) tuple)
    
    Java 类比：类似视频分析库中的场景分割算法。
    """
    from scenedetect.frame_timecode import FrameTimecode

    # 打开视频流
    video = VideoStreamAv(video_path)
    # 创建场景管理器
    manager = SceneManager()
    # 添加自适应检测器
    manager.add_detector(
        AdaptiveDetector(
            adaptive_threshold=threshold,  # 自适应阈值
            min_scene_len=min_scene_len,  # 最小场景长度
        )
    )

    # 如果有预热起始帧，跳转到该位置
    if warmup_start_frame > 0:
        video.seek(warmup_start_frame)

    # 构建检测参数
    detect_kwargs = {}
    if frame_skip > 0:
        detect_kwargs["frame_skip"] = frame_skip  # 设置跳帧数
    if end_frame is not None:
        # 设置结束时间（转换为 FrameTimecode）
        detect_kwargs["end_time"] = FrameTimecode(end_frame, fps=video.frame_rate)

    # 执行场景检测
    manager.detect_scenes(video, **detect_kwargs)

    # 获取场景列表
    scene_list = manager.get_scene_list()

    # Filter out scenes that belong to the warmup region (before start_frame)
    # 过滤掉属于预热区域的场景（start_frame 之前的场景）
    if warmup_start_frame < start_frame:
        scene_list = [s for s in scene_list if s[0].get_frames() >= start_frame]

    return scene_list


def _run_scenedetect_segment(args: tuple) -> List:
    """
    Worker function for parallel scenedetect. Returns scene list for one segment.
    并行场景检测的工作函数。返回一个片段的场景列表。
    
    This function is designed to be called by ProcessPoolExecutor.
    此函数设计为由 ProcessPoolExecutor 调用。
    
    Args:
        args: 元组，包含 (video_path, threshold, min_scene_len, start_frame, end_frame, frame_skip, warmup_frames)
    Returns:
        该片段的场景列表 (Scene list for this segment)
    
    Java 类比：类似 Callable 任务，用于并行处理。
    """
    # 解包参数
    video_path, threshold, min_scene_len, start_frame, end_frame, frame_skip, warmup_frames = args
    # 计算预热起始帧（确保不小于 0）
    warmup_start_frame = max(0, start_frame - warmup_frames)
    return _run_scenedetect(
        video_path, threshold, min_scene_len, end_frame, frame_skip,
        start_frame=start_frame, warmup_start_frame=warmup_start_frame,
    )


def _run_scenedetect_parallel(
    video_path: str,
    threshold: float,
    min_scene_len: int,
    total_frames: int,
    frame_skip: int = 0,
    num_workers: int = 8,
) -> List:
    """
    Split video into segments and run scenedetect in parallel across processes.
    将视频分成多个片段并在多进程中并行运行场景检测。
    
    Each segment starts decoding `warmup_frames` before the logical segment boundary
    so that AdaptiveDetector's sliding window is warmed up before the region of interest.
    Scenes detected in the warmup region are discarded.
    每个片段在逻辑边界之前解码 `warmup_frames` 帧，
    以便 AdaptiveDetector 的滑动窗口在感兴趣区域之前预热。
    预热区域中检测到的场景会被丢弃。
    
    This function:
    此函数：
    1. Divides video into equal segments for parallel processing
    1. 将视频分成相等的片段进行并行处理
    2. Calculates warmup frames to ensure detector accuracy
    2. 计算预热帧数以确保检测器准确性
    3. Runs detection in parallel using ProcessPoolExecutor
    3. 使用 ProcessPoolExecutor 并行运行检测
    4. Merges and sorts results from all segments
    4. 合并并排序所有片段的结果
    
    Args:
        video_path: 视频文件路径 (Path to video file)
        threshold: 自适应阈值 (Adaptive threshold)
        min_scene_len: 最小场景长度（帧数） (Minimum scene length in frames)
        total_frames: 视频总帧数 (Total number of frames in video)
        frame_skip: 跳帧数 (Frame skip count)
        num_workers: 工作进程数 (Number of worker processes)
    Returns:
        排序后的场景列表 (Sorted list of scenes)
    
    Java 类比：类似 ForkJoinPool 并行处理大任务，然后合并结果。
    """
    # warmup must cover: AdaptiveDetector window (window_width=2 → 5 frames) + min_scene_len,
    # all scaled by (frame_skip+1) to account for skipped frames.
    # 预热必须覆盖：AdaptiveDetector 窗口（window_width=2 → 5 帧）+ min_scene_len，
    # 全部按 (frame_skip+1) 缩放以考虑跳帧。
    warmup_frames = (min_scene_len + 5) * (frame_skip + 1)

    # 计算每个片段的大小
    segment_size = total_frames // num_workers
    segments = []
    # 创建片段列表
    for i in range(num_workers):
        seg_start = i * segment_size  # 片段起始帧
        # 最后一个片段包含剩余的所有帧
        seg_end = total_frames if i == num_workers - 1 else (i + 1) * segment_size
        segments.append((video_path, threshold, min_scene_len, seg_start, seg_end, frame_skip, warmup_frames))

    # 使用进程池并行执行
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # map 保持顺序，返回每个片段的场景列表
        results = list(executor.map(_run_scenedetect_segment, segments))

    # 合并所有片段的场景
    all_scenes = []
    for scene_list in results:
        all_scenes.extend(scene_list)

    # 按起始帧排序
    all_scenes.sort(key=lambda s: s[0].get_frames())
    return all_scenes


def scenedetect_extract_and_detect(
    video_path: str,
    frames_dir: str,
    target_fps: float = 2.0,
    target_resolution: Optional[int] = None,
    threshold: float = 3.0,
    min_scene_len: int = 15,
    save_frames_to_disk: bool = False,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
    max_minutes: Optional[float] = None,
    num_workers: int = 1,
) -> dict:
    """
    Extract frames and detect scenes using PySceneDetect.
    使用 PySceneDetect 提取帧并检测场景。
    
    This is the main entry point for video preprocessing.
    这是视频预处理的主要入口点。
    
    The workflow:
    工作流程：
    1. Opens video and calculates sampling parameters
    1. 打开视频并计算采样参数
    2. Generates frame indices at target FPS
    2. 生成目标 FPS 的帧索引
    3. Runs scene detection (parallel or single-process)
    3. 运行场景检测（并行或单进程）
    4. Optionally saves sampled frames to disk
    4. 可选地将采样帧保存到磁盘
    5. Converts scene timecodes to frame indices
    5. 将场景时间码转换为帧索引
    6. Returns comprehensive metadata dictionary
    6. 返回全面的元数据字典
    
    Args:
        video_path: 视频文件路径 (Path to video file)
        frames_dir: 保存帧的目录 (Directory to save frames)
        target_fps: 目标采样 FPS（默认 2.0） (Target sampling FPS, default 2.0)
        target_resolution: 目标分辨率（短边像素数） (Target resolution as short-side pixels)
        threshold: 场景检测阈值（默认 3.0） (Scene detection threshold, default 3.0)
        min_scene_len: 最小场景长度（帧数，默认 15） (Minimum scene length in frames, default 15)
        save_frames_to_disk: 是否保存帧到磁盘 (Whether to save frames to disk)
        image_format: 图像格式（'jpg' 或 'png'） (Image format: 'jpg' or 'png')
        jpeg_quality: JPEG 质量（1-100，默认 95） (JPEG quality 1-100, default 95)
        max_minutes: 最大处理分钟数，None 表示全部 (Max minutes to process, None for all)
        num_workers: 工作进程数（1 表示单进程） (Number of worker processes, 1 for single-process)
    Returns:
        包含帧信息、场景信息、视频元数据的字典
        (Dict with frame info, scene info, and video metadata)
    
    Java 类比：类似一个视频预处理的 Facade，整合多个步骤并返回结果。
    """
    # 确保目录存在
    _ensure_dir(frames_dir)

    # 验证目标 FPS
    if target_fps <= 0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")

    # 打开视频获取元数据
    meta_video = open_video(video_path)
    video_fps = float(meta_video.frame_rate)  # 视频原始 FPS
    total_frames = int(meta_video.duration.get_frames())  # 总帧数

    # 计算结束帧（如果限制了最大分钟数）
    if max_minutes is not None:
        end_frame = min(int(max_minutes * 60 * video_fps), total_frames)
    else:
        end_frame = total_frames

    # 与旧版等间隔采样逻辑保持一致，返回采样索引而不落盘抽帧
    # Generate frame indices at target FPS
    # 生成目标 FPS 的帧索引
    frame_indices: List[int] = []
    sample_interval = video_fps / target_fps  # 采样间隔（帧）
    current_frame = 0.0
    while int(current_frame) < end_frame:
        frame_indices.append(int(current_frame))
        current_frame += sample_interval

    # Compute frame_skip so scenedetect processes at roughly target_fps
    # 计算跳帧数，使 scenedetect 以大约 target_fps 处理
    frame_skip = max(0, round(video_fps / target_fps) - 1)

    shot_scenes_path = os.path.join(frames_dir, "shot_scenes.txt")  # 场景文件路径
    # 创建视频阅读器（带分辨率缩放）
    video_reader = _create_decord_reader(video_path, target_resolution)

    # 检查是否存在已有的场景文件
    if os.path.exists(shot_scenes_path):
        print(f"[SceneDetect] Found existing shot_scenes.txt, skipping detection")
        scene_list = None  # will load from file below - 将从文件加载
    else:
        # 运行场景检测
        print(f"[SceneDetect] Running PySceneDetect (frame_skip={frame_skip}, num_workers={num_workers})")
        if num_workers > 1:
            # 并行模式
            scene_list = _run_scenedetect_parallel(
                video_path,
                threshold,
                min_scene_len,
                end_frame,
                frame_skip=frame_skip,
                num_workers=num_workers,
            )
        else:
            # 单进程模式
            scene_list = _run_scenedetect(
                video_path,
                threshold,
                min_scene_len,
                end_frame if max_minutes is not None else None,
                frame_skip=frame_skip,
            )
    sample_fps = float(target_fps)  # 采样 FPS

    # 获取视频尺寸
    if len(video_reader) > 0:
        first_frame = video_reader[0].asnumpy()
        height, width = int(first_frame.shape[0]), int(first_frame.shape[1])
    else:
        height, width = 0, 0

    # 过滤超出视频长度的帧索引
    if len(video_reader) > 0:
        frame_indices = [idx for idx in frame_indices if idx < len(video_reader)]

    # 可选：保存帧到磁盘
    frame_paths: List[str] = []
    if save_frames_to_disk:
        print(f"[SceneDetect] Saving sampled frames to disk in {frames_dir}")
        frame_paths = _save_sampled_frames_to_disk(
            video_reader=video_reader,
            frame_indices=frame_indices,
            frames_dir=frames_dir,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
        )

    # 转换场景时间码为帧索引
    scenes: List[List[int]] = []
    if scene_list is not None:
        for scene in scene_list:
            # 将时间码转换为秒数
            start_sec = _timecode_to_seconds(scene[0].get_timecode())
            end_sec_scene = _timecode_to_seconds(scene[1].get_timecode())
            # 转换为采样帧索引
            start_frame = int(start_sec * sample_fps)
            end_frame_i = int(end_sec_scene * sample_fps)
            scenes.append([start_frame, end_frame_i])

        # 调整场景边界（消除间隙和重叠）
        scenes = _adjust_scene_boundaries(scenes)

        # 保存场景到文件
        if scenes:
            np.savetxt(shot_scenes_path, np.array(scenes), fmt="%d")
        else:
            np.savetxt(shot_scenes_path, np.array([]).reshape(0, 2), fmt="%d")
    else:
        # Load existing scenes from file
        # 从文件加载已有场景
        raw = np.loadtxt(shot_scenes_path, dtype=int)
        if raw.ndim == 1 and len(raw) == 2:
            scenes = [raw.tolist()]  # 单个场景
        elif raw.ndim == 2:
            scenes = raw.tolist()  # 多个场景
        else:
            scenes = []  # 无场景

    print(f"[SceneDetect] Completed: {len(frame_indices)} sampled indices, {len(scenes)} scenes")

    # 返回综合元数据
    return {
        "num_frames": len(frame_indices),  # 采样帧数
        "sample_fps": float(sample_fps),  # 采样 FPS
        "height": int(height),  # 视频高度
        "width": int(width),  # 视频宽度
        "video_reader": video_reader,  # 视频阅读器对象
        "frame_indices": frame_indices,  # 帧索引列表
        "frame_paths": frame_paths,  # 保存的帧文件路径
        "save_frames_to_disk": bool(save_frames_to_disk),  # 是否保存到磁盘
        "shot_scenes_path": shot_scenes_path,  # 场景文件路径
        "scenes": scenes,  # 场景列表
        "shot_detection_fps": float(sample_fps),  # 场景检测 FPS
        "shot_detection_model": "scenedetect",  # 使用的检测模型
    }


def decode_video_to_frames(
    video_path: str,
    frames_dir: str,
    target_fps: Optional[float] = None,
    target_resolution: Optional[Tuple[int, int]] = None,
    max_minutes: Optional[float] = None,
    shot_detection_threshold: float = 3.0,
    shot_detection_min_scene_len: int = 15,
    save_frames_to_disk: bool = False,
    image_format: str = "jpg",
    jpeg_quality: int = 80,
    num_workers: int = 16,
) -> Dict[str, Any]:
    """
    High-level API to decode video to frames with scene detection.
    带场景检测的视频解码到帧的高级 API。
    
    This is a convenience wrapper around scenedetect_extract_and_detect
    with sensible defaults.
    这是 scenedetect_extract_and_detect 的便捷包装器，
    带有合理的默认值。
    
    Args:
        video_path: 视频文件路径 (Path to video file)
        frames_dir: 保存帧的目录 (Directory to save frames)
        target_fps: 目标 FPS，默认 2.0 (Target FPS, default 2.0)
        target_resolution: 目标分辨率 (Target resolution)
        max_minutes: 最大处理分钟数 (Max minutes to process)
        shot_detection_threshold: 场景检测阈值（默认 3.0） (Scene detection threshold, default 3.0)
        shot_detection_min_scene_len: 最小场景长度（默认 15） (Minimum scene length, default 15)
        save_frames_to_disk: 是否保存帧到磁盘 (Whether to save frames to disk)
        image_format: 图像格式（默认 'jpg'） (Image format, default 'jpg')
        jpeg_quality: JPEG 质量（默认 80） (JPEG quality, default 80)
        num_workers: 工作进程数（默认 16） (Number of workers, default 16)
    Returns:
        与 scenedetect_extract_and_detect 相同的字典
        (Same dict as scenedetect_extract_and_detect)
    
    Java 类比：类似一个 Facade 方法，简化底层复杂 API 的调用。
    """
    # 设置默认 FPS
    fps = float(target_fps) if target_fps is not None else 2.0
    if fps <= 0:
        fps = 2.0

    # 处理分辨率参数
    resolution = target_resolution
    if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
        resolution = (int(resolution[0]), int(resolution[1]))
    elif isinstance(resolution, (tuple, list)) and len(resolution) == 1:
        resolution = int(resolution[0])

    # 调用主函数
    return scenedetect_extract_and_detect(
        video_path=video_path,
        frames_dir=frames_dir,
        target_fps=fps,
        target_resolution=resolution,
        threshold=float(shot_detection_threshold),
        min_scene_len=int(shot_detection_min_scene_len),
        save_frames_to_disk=bool(save_frames_to_disk),
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        max_minutes=max_minutes,
        num_workers=num_workers,
    )
