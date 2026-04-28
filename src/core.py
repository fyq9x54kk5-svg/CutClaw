import os
import json  # JSON 处理模块，类似 Java 的 com.google.gson
import copy  # 深拷贝模块，类似 Java 的 clone()
import re  # 正则表达式模块，类似 Java 的 java.util.regex
import time  # 时间模块，类似 Java 的 System.currentTimeMillis()
import gc  # 垃圾回收模块，Python 自动管理内存，但可手动触发
import threading  # 线程模块，类似 Java 的 java.lang.Thread
from concurrent.futures import ThreadPoolExecutor, as_completed  # 线程池，类似 Java 的 ExecutorService
from typing import Annotated as A  # 类型注解
from src.video.deconstruction.video_caption import (
    SYSTEM_PROMPT,
    messages as caption_messages,
)
from src.video.preprocess.video_utils import _create_decord_reader
import litellm
from src.utils.media_utils import seconds_to_hhmmss as convert_seconds_to_hhmmss, hhmmss_to_seconds as _hhmmss_to_seconds, parse_srt_to_dict, array_to_base64
from src import config
from src.func_call_shema import as_json_schema
from src.func_call_shema import doc as D
from src.Reviewer import ReviewerAgent
from src.prompt import (
    DENSE_CAPTION_PROMPT_FILM,
    EDITOR_SYSTEM_PROMPT,
    EDITOR_USER_PROMPT,
    EDITOR_FINISH_PROMPT,
    EDITOR_USE_TOOL_PROMPT,
)


class StopException(Exception):
    """
    Stop Execution by raising this exception (Signal that the task is Finished).
    通过抛出此异常停止执行（表示任务已完成）。
    
    Java 类比：类似于抛出一个自定义异常来中断流程
    """


# 工具名称别名映射（用于兼容不同命名风格）
TOOL_NAME_ALIASES = {
    "semantic_neighborhood_retrieval": "Semantic Neighborhood Retrieval",
    "fine_grained_shot_trimming": "Fine-Grained Shot Trimming",
    "commit": "Commit",
}

# 旧版工具名称映射（向后兼容）
LEGACY_TOOL_NAME_MAP = {
    "get_related_shot": "semantic_neighborhood_retrieval",
    "trim_shot": "fine_grained_shot_trimming",
    "finish": "commit",
}


# 线程局部存储，用于保存每个线程的视频读取器
# Java 类比：ThreadLocal<VideoReader>
_THREAD_VIDEO_READERS = threading.local()


def _get_thread_video_reader(video_path: str):
    """
    Get or create a video reader for the current thread.
    获取或创建当前线程的视频读取器。
    
    In parallel mode, each worker thread keeps its own decord reader to avoid
    并行模式下，每个线程持有自己的 decord reader，
    cross-thread reader conflicts and repeated init overhead.
    避免跨线程冲突与重复初始化开销。
    
    Java 类比：类似 ThreadLocal 模式，每个线程有自己的资源副本
    """
    if not video_path:
        return None
    
    # getattr: 获取属性，带默认值
    reader = getattr(_THREAD_VIDEO_READERS, "reader", None)
    reader_path = getattr(_THREAD_VIDEO_READERS, "video_path", None)
    
    # 如果 reader 不存在或路径不匹配，则创建新的
    if reader is None or reader_path != video_path:
        target_resolution = getattr(config, "VIDEO_RESOLUTION", None)
        reader = _create_decord_reader(video_path, target_resolution)
        _THREAD_VIDEO_READERS.reader = reader
        _THREAD_VIDEO_READERS.video_path = video_path
    return reader


def _clear_thread_video_reader():
    """
    Clear the video reader for the current thread.
    清除当前线程的视频读取器。
    
    Java 类比：类似 ThreadLocal.remove()
    """
    if hasattr(_THREAD_VIDEO_READERS, "reader"):
        _THREAD_VIDEO_READERS.reader = None
    if hasattr(_THREAD_VIDEO_READERS, "video_path"):
        _THREAD_VIDEO_READERS.video_path = None


def _normalize_video_reader(video_reader):
    """
    Normalize video reader to ensure it's a valid reader object.
    标准化视频读取器，确保它是有效的读取器对象。
    """
    # isinstance: 类型检查，类似 Java 的 instanceof
    if isinstance(video_reader, dict):
        video_reader = video_reader.get("video_reader")
    if video_reader is not None and not hasattr(video_reader, "get_avg_fps"):
        return None
    return video_reader


def _canonical_tool_name(name: str) -> str:
    """
    Convert tool name to canonical form (handles legacy names and aliases).
    将工具名称转换为标准形式（处理旧版名称和别名）。
    """
    # 如果是旧版名称，转换为新名称
    if name in LEGACY_TOOL_NAME_MAP:
        return LEGACY_TOOL_NAME_MAP[name]
    
    # 如果是别名，转换为原始名称
    for original_name, alias_name in TOOL_NAME_ALIASES.items():
        if name == original_name or name == alias_name:
            return original_name
    return name


def _parse_shot_time_ranges(answer: str) -> list[tuple[float, float]]:
    """
    Parse shot ranges in the format: [shot: HH:MM:SS to HH:MM:SS].
    解析镜头时间范围，格式：[shot: HH:MM:SS to HH:MM:SS]。
    
    Args:
        answer: 模型输出的文本回答
    Returns:
        时间范围列表 [(start_sec, end_sec), ...]
    """
    # re.compile: 编译正则表达式，类似 Java 的 Pattern.compile()
    shot_pattern = re.compile(r'\[?shot[\s_]*\d*:\s*([0-9:.]+)\s+to\s+([0-9:.]+)\]?', re.IGNORECASE)
    matches = shot_pattern.findall(answer or "")
    if not matches:
        return []

    ranges = []
    for start_time, end_time in matches:
        # _hhmmss_to_seconds: 将 HH:MM:SS 格式转换为秒数
        start_sec = _hhmmss_to_seconds(start_time, fps=getattr(config, 'VIDEO_FPS', 24) or 24)
        end_sec = _hhmmss_to_seconds(end_time, fps=getattr(config, 'VIDEO_FPS', 24) or 24)
        ranges.append((start_sec, end_sec))
    return ranges


def _ranges_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """
    Check if two time ranges overlap.
    检查两个时间范围是否重叠。
    
    Java 类比：类似判断两个区间是否有交集
    """
    return a_start < b_end and a_end > b_start


def _parse_retry_after_seconds(error_text: str, default_seconds: float = 1.0) -> float:
    """
    Parse retry-after seconds from rate limit error message.
    从速率限制错误消息中解析重试等待秒数。
    """
    # re.search: 搜索正则匹配，类似 Java 的 Matcher.find()
    match = re.search(r'after\s+(\d+(?:\.\d+)?)\s*seconds?', error_text or "", re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return default_seconds
    return default_seconds


def _compact_json_str_for_log(s: str, max_len: int = 500) -> str:
    """
    Truncate JSON string for logging to avoid overly long logs.
    截断 JSON 字符串用于日志记录，避免过长的日志。
    
    Args:
        s: JSON 字符串
        max_len: 最大长度，默认500字符
    Returns:
        截断后的字符串
    """
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"... [truncated {len(s) - max_len} chars]"


def commit(
    answer: A[str, D("Output the final shot time range. Must be exactly ONE continuous clip.")],
    video_path: A[str, D("Path to the source video file")] = "",
    output_path: A[str, D("Path to save the edited video")] = "",
    target_length_sec: A[float, D("Expected total length in seconds")] = 0.0,
    section_idx: A[int, D("Current section index. Auto-injected.")] = -1,
    shot_idx: A[int, D("Current shot index. Auto-injected.")] = -1,
    protagonist_frame_data: A[list, D("Frame-by-frame protagonist detection data. Auto-injected.")] = None
) -> str:
    """
    Call this function to finalize the shot selection and save the result.
    调用此函数以最终确定镜头选择并保存结果。
    
    NOTE: You MUST call commit first to validate the shot before calling this function.
    注意：必须先调用 commit 验证镜头，然后才能调用此函数。

    IMPORTANT: Only accepts ONE continuous time range.
    重要：只接受一个连续的时间范围。
    Example: [shot: 00:10:00 to 00:10:06.4]

    Returns:
        str: Success message with saved result, or error message if parsing fails.
        成功消息（包含保存的结果），或解析失败的错误消息。
    """
    # 内部辅助函数：将时间字符串转换为秒数
    def hhmmss_to_seconds(time_str: str) -> float:
        return _hhmmss_to_seconds(time_str, fps=getattr(config, 'VIDEO_FPS', 24) or 24)

    seconds_to_hhmmss = convert_seconds_to_hhmmss

    # Parse the model's textual answer into one or more clip ranges.
    # 将模型文本回答解析为一个或多个片段时间范围。
    shot_pattern = re.compile(r'\[?shot[\s_]*\d*:\s*([0-9:.]+)\s+to\s+([0-9:.]+)\]?', re.IGNORECASE)
    matches = shot_pattern.findall(answer)

    if not matches:
        return "Error: Could not parse shot time ranges. Please provide format: [shot: HH:MM:SS to HH:MM:SS]"

    # Support controlled multi-shot stitching (max count + continuity checks).
    # 支持受控的多镜头拼接（最大数量 + 连续性校验）。
    max_shots_allowed = getattr(config, 'MAX_SHOTS_PER_CLIP', 3)
    if len(matches) > max_shots_allowed:
        return f"Error: Too many shots detected ({len(matches)}). Maximum allowed: {max_shots_allowed}"

    # Parse all time ranges and validate
    # 解析所有时间范围并验证
    clips = []
    total_duration = 0
    for i, (start_time, end_time) in enumerate(matches):
        try:
            start_sec = hhmmss_to_seconds(start_time)
            end_sec = hhmmss_to_seconds(end_time)
            duration = end_sec - start_sec

            if duration <= 0:
                return f"Error: Shot {i+1} has invalid duration (start: {start_time}, end: {end_time})"

            clips.append({
                'start_sec': start_sec,
                'end_sec': end_sec,
                'duration': duration,
                'start_time': start_time,
                'end_time': end_time
            })
            total_duration += duration
        except Exception as e:
            return f"Error parsing shot {i+1} time range: {str(e)}"

    # Validate temporal continuity for stitched clips:
    # 校验拼接片段的时间连续性：
    # no overlap, and no large jump between adjacent clips.
    # 不允许重叠，也不允许相邻片段出现过大时间跳跃。
    if len(clips) > 1:
        max_gap = getattr(config, 'MAX_STITCH_GAP_SEC', 2.0)
        for i in range(len(clips) - 1):
            gap = clips[i+1]['start_sec'] - clips[i]['end_sec']
            if gap < 0:
                return f"Error: Overlapping shots detected. Shot {i+1} ends at {clips[i]['end_time']}, but shot {i+2} starts at {clips[i+1]['start_time']}"
            if gap > max_gap:
                return f"Error: Time gap ({gap:.2f}s) between shot {i+1} and {i+2} exceeds maximum ({max_gap}s). Shots must maintain visual continuity."

    # Use total duration for validation
    # 使用总持续时间进行验证
    duration = total_duration

    # For result building, we'll use the first and last timestamps
    # 构建结果时，使用第一个和最后一个时间戳
    start_sec = clips[0]['start_sec']
    end_sec = clips[-1]['end_sec']
    start_time = clips[0]['start_time']
    end_time = clips[-1]['end_time']

    # Auto-trim tiny overrun to improve robustness instead of failing.
    # 对轻微超时自动裁切，提升鲁棒性而不是直接失败。
    duration_diff = duration - target_length_sec
    if 0 < duration_diff <= 1.0:
        # Only auto-trim the last clip's end time
        # 只自动裁切最后一个片段的结束时间
        clips[-1]['end_sec'] = clips[-1]['end_sec'] - duration_diff
        clips[-1]['duration'] = clips[-1]['end_sec'] - clips[-1]['start_sec']
        clips[-1]['end_time'] = seconds_to_hhmmss(clips[-1]['end_sec'])

        # Recalculate total duration
        # 重新计算总持续时间
        duration = sum(c['duration'] for c in clips)

        end_sec = clips[-1]['end_sec']
        end_time = clips[-1]['end_time']
        print(f"✂️  [Trim] Auto-trimmed by {duration_diff:.2f}s. New end: {end_time}")

    # Build result data with all clips
    # 构建包含所有片段的结果数据
    result_clips = []
    for i, clip in enumerate(clips):
        result_clips.append({
            "shot": i + 1,
            "start": seconds_to_hhmmss(clip['start_sec']),
            "end": seconds_to_hhmmss(clip['end_sec']),
            "duration": round(clip['duration'], 2)
        })

    result_data = {
        "status": "success",
        "section_idx": section_idx,
        "shot_idx": shot_idx,
        "total_duration": round(duration, 2),
        "target_duration": target_length_sec,
        "num_clips": len(clips),
        "is_stitched": len(clips) > 1,  # 是否为拼接片段
        "clips": result_clips
    }

    # Add protagonist frame detection data if available
    # 如果有主角帧检测数据，添加到结果中
    if protagonist_frame_data:
        result_data["protagonist_detection"] = {
            "method": "vlm",
            "total_frames_analyzed": len(protagonist_frame_data),
            "frames_with_protagonist": sum(1 for f in protagonist_frame_data if f.get("protagonist_detected", False)),
            "protagonist_ratio": round(
                sum(1 for f in protagonist_frame_data if f.get("protagonist_detected", False)) / len(protagonist_frame_data),
                3
            ) if protagonist_frame_data else 0.0,
            "frame_detections": protagonist_frame_data
        }

    # Save result to output_path if provided
    # 如果提供了输出路径，保存结果到文件
    if output_path:
        # 检查文件是否存在，如果存在则加载已有数据
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                try:
                    all_results = json.load(f)
                except json.JSONDecodeError:
                    all_results = []
        else:
            all_results = []

        # 添加新结果到列表
        all_results.append(result_data)

        # 保存所有结果到 JSON 文件
        # json.dump: 将 Python 对象序列化为 JSON，类似 Java 的 Gson.toJson()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"💾 [Output] Result saved to {output_path}")

    success_msg = f"Successfully created edited video: {seconds_to_hhmmss(start_sec)} to {seconds_to_hhmmss(end_sec)} ({duration:.2f}s)"
    print(f"✅ [Success] {success_msg}")

    # 返回 JSON 字符串格式的result_data
    # json.dumps: 将 Python 对象转换为 JSON 字符串，类似 Java 的 Gson.toJson()
    return json.dumps(result_data, ensure_ascii=False)


def semantic_neighborhood_retrieval(
        related_scenes: A[list, D("List of scene indices to search. Optional - you can specify nearby scenes within allowed range.")] = None,
        scene_folder_path: A[str, D("Path to the folder containing scene JSON files. Auto-injected.")] = None,
        recommended_scenes: A[list, D("Recommended scene indices from shot_plan. Auto-injected.")] = None
) -> str:
    """
    Retrieves shot information from specified scenes.

    You can optionally specify which scenes to search by passing a 'related_scenes' list.
    However, you can only explore scenes within ±SCENE_EXPLORATION_RANGE of the recommended scenes.

    Example:
    - If recommended scenes are [8, 12] and SCENE_EXPLORATION_RANGE=3
    - You can search scenes 5-11 (around 8) and 9-15 (around 12)
    - Searching scene 50 would be REJECTED

    If you don't specify scenes, the system will use the recommended scenes automatically.

    Returns:
        str: A formatted string containing the shot information from the requested scenes.
        IMPORTANT: Select segments within shot boundaries to avoid visual discontinuities.

    Notes:
        - If you can't find suitable shots in recommended scenes, try nearby scenes
        - Going too far from recommended scenes may result in mismatched content
    """

    # Guardrails: agent can explore only near recommended scenes to keep
    # 约束机制：智能体只能在推荐场景附近探索，
    # semantic consistency and prevent drifting to unrelated content.
    # 以保持语义一致，避免漂移到无关内容。
    if related_scenes and recommended_scenes:
        from src import config
        allowed_range = getattr(config, 'SCENE_EXPLORATION_RANGE', 3)

        # Get total scene count by checking available scene files
        max_scene_idx = 0
        if scene_folder_path and os.path.isdir(scene_folder_path):
            import glob
            scene_files = glob.glob(os.path.join(scene_folder_path, "scene_*.json"))
            if scene_files:
                # Extract scene numbers from filenames
                scene_numbers = []
                for f in scene_files:
                    basename = os.path.basename(f)  # e.g., "scene_42.json"
                    try:
                        num = int(basename.replace("scene_", "").replace(".json", ""))
                        scene_numbers.append(num)
                    except ValueError:
                        continue
                max_scene_idx = max(scene_numbers) if scene_numbers else 0

        # Build allowed scene set with boundary constraints
        allowed_scenes = set()
        for rec_scene in recommended_scenes:
            for offset in range(-allowed_range, allowed_range + 1):
                scene_idx = rec_scene + offset
                # Ensure scene index is within valid range [0, max_scene_idx]
                if scene_idx >= 0 and (max_scene_idx == 0 or scene_idx <= max_scene_idx):
                    allowed_scenes.add(scene_idx)

        # Check if all requested scenes are within allowed range
        invalid_scenes = [s for s in related_scenes if s not in allowed_scenes]
        if invalid_scenes:
            return (
                f"❌ Error: Cannot search scenes {invalid_scenes} - they are outside the allowed range.\n"
                f"Recommended scenes: {recommended_scenes}\n"
                f"Allowed exploration range: ±{allowed_range} scenes\n"
                f"Valid scenes you can search: {sorted(list(allowed_scenes))}\n"
                f"Please select scenes within the allowed range or omit the 'related_scenes' parameter to use defaults."
            )

        print(f"🗺️  [Explore] Agent exploring nearby scenes: {related_scenes} (recommended: {recommended_scenes})")
    elif not related_scenes and recommended_scenes:
        # Use recommended scenes if agent didn't specify
        related_scenes = recommended_scenes
        print(f"🗺️  [Explore] Using recommended scenes: {related_scenes}")

    if not related_scenes:
        return "Error: No scenes specified and no recommended scenes available."

    all_shots_info = []

    for scene_idx in related_scenes:
        scene_file = os.path.join(scene_folder_path, f"scene_{scene_idx}.json")
        if os.path.exists(scene_file):
            try:
                with open(scene_file, 'r', encoding='utf-8') as f:
                    scene_data = json.load(f)

                scene_time_range = scene_data.get('time_range', {})
                scene_start = scene_time_range.get('start_seconds', '00:00:00')
                scene_end = scene_time_range.get('end_seconds', '00:00:00')

                all_shots_info.append(f"\n=== Scene {scene_idx} ({scene_start} - {scene_end}) ===")

                shots_data = scene_data.get('shots_data', [])
                for shot in shots_data:
                    duration = shot.get('duration', {})
                    start_time = duration.get('clip_start_time', '')
                    end_time = duration.get('clip_end_time', '')

                    action = shot.get('action_atoms', {})
                    event_summary = action.get('event_summary', '')

                    narrative = shot.get('narrative_analysis', {})
                    mood = narrative.get('mood', '')

                    shot_info = f"[{start_time} - {end_time}] {event_summary}"
                    if mood:
                        shot_info += f" (Mood: {mood})"

                    all_shots_info.append(shot_info)

            except Exception as e:
                print(f"⚠️  [Warning] Failed to load scene {scene_idx}: {e}")
                all_shots_info.append(f"Scene {scene_idx}: Failed to load - {e}")
        else:
            all_shots_info.append(f"Scene {scene_idx}: File not found")

    result = "\n".join(all_shots_info)
    return f"Here are the available shots from related scenes {related_scenes}:\n{result}"


def review_clip(
    time_range: A[str, D("The time range to check (e.g., '00:13:28 to 00:13:40').")],
    used_time_ranges: A[list, D("List of already used time ranges. Auto-injected.")] = None
) -> str:
    """
    Check if the proposed time range overlaps with any previously used clips.
    You MUST call this tool BEFORE calling finish to ensure no duplicate footage.

    Returns:
        str: A message indicating whether the time range is available or overlaps with used clips.
             If overlap is detected, you should select a different time range.
    """
    def hhmmss_to_seconds(time_str: str) -> float:
        return _hhmmss_to_seconds(time_str, fps=getattr(config, 'VIDEO_FPS', 24) or 24)

    if used_time_ranges is None:
        used_time_ranges = []

    # Parse the time range
    match = re.search(r'([\d:.]+)\s+to\s+([\d:.]+)', time_range, re.IGNORECASE)
    if not match:
        return f"Error: Could not parse time range '{time_range}'. Please use format 'HH:MM:SS to HH:MM:SS'."

    try:
        start_sec = hhmmss_to_seconds(match.group(1))
        end_sec = hhmmss_to_seconds(match.group(2))
    except Exception as e:
        return f"Error parsing time range: {e}"

    if not used_time_ranges:
        return f"✅ OK: Time range {time_range} is available. No previous clips have been used yet. You can proceed with finish."

    # Prevent duplicate footage across generated shots.
    # 防止不同镜头选择中出现素材重复。
    # This tool is a lightweight temporal conflict checker.
    # 该工具是轻量级时间冲突检测器。
    overlapping_clips = []
    for idx, (used_start, used_end) in enumerate(used_time_ranges):
        if start_sec < used_end and end_sec > used_start:
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


def fine_grained_shot_trimming(
    time_range: A[str, D("The time range to analyze ('HH:MM:SS to HH:MM:SS'). This tool will analyze the ENTIRE range and provide scene breakdowns within it.")],
    frame_path: A[str, D("The path to the video frames file.")] = "",
    transcript_path: A[str, D("Optional path to an .srt transcript file; subtitles in this range will be injected into the prompt.")] = "",
    original_shot_boundaries: A[list, D("List of original shot boundaries from source material. Auto-injected.")] = None,
) -> str:
    """
    Analyze a video clip time range and return detailed scene information and usability assessment.
        
    Returns:
        A JSON string with structure:
        {
            "analyzed_range": "HH:MM:SS to HH:MM:SS",  # The full range you requested, must longer that 3.0s
            "total_duration_sec": float,                # Total duration
            "usability_assessment": "...",              # Overall evaluation
            "recommended_usage": "...",                 # How to use this clip
            "internal_scenes": [...]                    # Scene breakdowns (for reference)
        }
        
        The "internal_scenes" are fine-grained descriptions to help you understand what's  happening INSIDE the analyzed range.
        Use them to decide whether to use the full range, a subset, or refine with another call.

    
    Args:
        time_range: String in format 'HH:MM:SS to HH:MM:SS' - the range to analyze
        frame_path: Path to the video frames directory
    """
    def hhmmss_to_seconds(time_str: str) -> float:
        return _hhmmss_to_seconds(time_str, fps=getattr(config, 'VIDEO_FPS', 24) or 24)

    def _extract_subtitles_in_range(srt_path: str, start_s: float, end_s: float, max_chars: int = 1500) -> str:
        """Reuse video_caption.parse_srt_to_dict() and only do range filtering + formatting here."""
        if not srt_path:
            return ""
        if not os.path.exists(srt_path):
            return ""
        try:
            subtitle_map = parse_srt_to_dict(srt_path)
            if not subtitle_map:
                return ""

            # parse_srt_to_dict() truncates timestamps to int seconds for keys.
            # Align the filtering to the same granularity to avoid boundary misses.
            import math
            start_i = int(start_s)
            end_i = int(math.ceil(end_s))
            if end_i <= start_i:
                end_i = start_i + 1


            picked = []  # list[tuple[int, str]]
            for key, text in subtitle_map.items():
                try:
                    s_sec, e_sec = map(int, key.split("_"))
                except Exception:
                    continue

                # Overlap check in integer-second domain (half-open interval)
                if s_sec >= end_i or e_sec <= start_i:
                    continue
                t = re.sub(r"\s+", " ", (text or "")).strip()
                if t:
                    picked.append((s_sec, t))

            if not picked:
                return ""

            picked.sort(key=lambda x: x[0])
            joined = " ".join(t for _, t in picked)
            joined = re.sub(r"\s+", " ", joined).strip()
            if len(joined) > max_chars:
                joined = joined[:max_chars].rsplit(' ', 1)[0] + "…"
            return joined
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ""

    
    # Parse the time range string: 'HH:MM:SS to HH:MM:SS' or 'HH:MM:SS.s to HH:MM:SS.s'
    match = re.search(r'([\d:.]+)\s+to\s+([\d:.]+)', time_range, re.IGNORECASE)
    
    if not match:
        return f"Error: Could not parse time range '{time_range}'."
    
    start_time_str = match.group(1)
    end_time_str = match.group(2)
    
    # Convert to seconds
    start_sec = hhmmss_to_seconds(start_time_str)
    end_sec = hhmmss_to_seconds(end_time_str)
    
    # Convert seconds to HH:MM:SS format for display
    clip_start_time = convert_seconds_to_hhmmss(start_sec)
    clip_end_time = convert_seconds_to_hhmmss(end_sec)

    subtitles_context = _extract_subtitles_in_range(transcript_path, start_sec, end_sec)
    
    # Reuse the same vision prompt family as video deconstruction, but scope it
    # 复用视频拆解阶段的视觉提示词体系，但限定在当前局部区间，
    # to the requested local range for precise trim-time reasoning.
    # 用于更精确的裁剪时段判断。
    send_messages = copy.deepcopy(caption_messages)
    send_messages[0]["content"] = SYSTEM_PROMPT

    dense_caption_prompt = DENSE_CAPTION_PROMPT_FILM.replace(
        "MAIN_CHARACTER_NAME_PLACEHOLDER",
        getattr(config, 'MAIN_CHARACTER_NAME', 'the main character')
    ).replace(
        "MIN_SEGMENT_DURATION_PLACEHOLDER",
        str(getattr(config, 'AUDIO_MIN_SEGMENT_DURATION', 3.0))
    )
    requested_duration = max(0.0, end_sec - start_sec)
    requested_end_rel = convert_seconds_to_hhmmss(requested_duration)
    dense_caption_prompt += (
        "\n\n[Clip Timing Constraints]\n"
        f"- Requested clip duration: {requested_duration:.2f}s\n"
        f"- Relative timeline MUST start at 00:00:00 and end at {requested_end_rel}\n"
        "- Segments must collectively cover the full duration without truncation\n"
        "- Keep timestamps monotonic and contiguous\n"
    )
    if subtitles_context:
        dense_caption_prompt += f"\n\n[Subtitles in this range]\n{subtitles_context}\n"
    send_messages[1]["content"] = dense_caption_prompt

    # Extract frames from only the requested range and encode into data-URI
    # 仅提取请求区间内帧，并编码为 data-URI 图片，
    # images for multimodal LLM input.
    # 供多模态 LLM 输入使用。
    def _extract_clip_frames(video_path, start_s, end_s, video_reader=None):
        vr = _normalize_video_reader(video_reader)
        if vr is None:
            if not video_path:
                return []
            vr = _get_thread_video_reader(video_path)
            if vr is None:
                return []
        video_fps = float(vr.get_avg_fps())
        start_f = max(0, int(start_s * video_fps))
        end_f = min(int(end_s * video_fps), len(vr) - 1)
        if end_f < start_f:
            return []
        indices = list(range(start_f, end_f + 1))
        # Safety cap to avoid provider limit: max data-uri per request (e.g., 250 on OpenAI).
        # Keep a margin for robustness.
        max_frames = int(getattr(config, 'CORE_MAX_FRAMES', getattr(config, 'TRIM_SHOT_MAX_FRAMES', 240)))
        if max_frames > 0 and len(indices) > max_frames:
            import math
            stride = max(1, math.ceil(len(indices) / max_frames))
            indices = indices[::stride]
            # Ensure last frame is included so ending timestamp context is preserved.
            if indices[-1] != end_f:
                indices.append(end_f)
            # Hard cap after end-frame append.
            if len(indices) > max_frames:
                indices = indices[:max_frames - 1] + [end_f]

        if not indices:
            return []
        frames = vr.get_batch(indices).asnumpy()
        return [array_to_base64(frames[i]) for i in range(len(frames))]

    # Build litellm messages with base64 frames injected
    def _build_litellm_messages(base_messages, b64_frames):
        result = []
        for msg in base_messages:
            if msg["role"] == "user" and b64_frames:
                content = [{"type": "text", "text": msg["content"]}] if msg["content"] else []
                for b64 in b64_frames:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                result.append({"role": "user", "content": content})
            else:
                result.append({"role": msg["role"], "content": msg["content"]})
        return result

    active_reader = _get_thread_video_reader(frame_path) if frame_path else None
    b64_frames = _extract_clip_frames(frame_path, start_sec, end_sec, video_reader=active_reader)
    litellm_messages = _build_litellm_messages(send_messages, b64_frames)

    try:
        # Call VIDEO_ANALYSIS_MODEL via litellm with retry logic for provider
        # 通过 litellm 调用 VIDEO_ANALYSIS_MODEL，并加入重试逻辑以应对
        # instability / malformed responses.
        # 服务不稳定或返回格式异常。
        tries = 3
        while tries > 0:
            tries -= 1
            try:
                kwargs = dict(
                    model=config.VIDEO_ANALYSIS_MODEL,
                    messages=litellm_messages,
                    max_tokens=config.VIDEO_ANALYSIS_MODEL_MAX_TOKEN,
                    temperature=0.0,
                )
                if config.VIDEO_ANALYSIS_ENDPOINT:
                    kwargs["api_base"] = config.VIDEO_ANALYSIS_ENDPOINT
                if config.VIDEO_ANALYSIS_API_KEY:
                    kwargs["api_key"] = config.VIDEO_ANALYSIS_API_KEY
                raw = litellm.completion(**kwargs)
                content_str = raw.choices[0].message.content
            except Exception as e:
                print(f"❌ [Error] [trim_shot] litellm call failed: {e}")
                content_str = None

            if not content_str:
                if tries == 0:
                    return f"Error: Failed to generate caption for time range {time_range}."
                continue

            try:
                content = content_str.strip()
                
                # Debug: print the raw content to help diagnose issues
                if not content:
                    print(f"⚠️  [Warning] Empty content from model for time range {time_range}")
                    if tries == 0:
                        return f"Error: Empty response from model for time range {time_range}."
                    continue
                
                # Try to extract JSON from markdown code blocks if present
                # Pattern: ```json ... ``` or ``` ... ```
                json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
                if json_block_match:
                    content = json_block_match.group(1).strip()
                
                # Try to parse JSON
                parsed = json.loads(content)

                # DEBUG: Print first segment to diagnose timestamp issues
                # Handle "segments" format (from DENSE_CAPTION_PROMPT)
                if isinstance(parsed, dict) and "segments" in parsed:
                    result = {
                        "analyzed_range": f"{clip_start_time} to {clip_end_time}",
                        "total_duration_sec": end_sec - start_sec,
                        "usability_assessment": "See segment details with quality scores and emotions.",
                        "internal_scenes": []
                    }

                    for seg in parsed["segments"]:
                        # Build comprehensive description from new format
                        desc_parts = []

                        # Cut type
                        if seg.get("cut_type"):
                            desc_parts.append(f"[{seg['cut_type'].upper()}]")

                        # Content description
                        if seg.get("content_description"):
                            desc_parts.append(seg["content_description"])

                        # Visual quality info
                        visual_quality = seg.get("visual_quality", {})
                        quality_score = visual_quality.get("score", "N/A")
                        quality_notes = visual_quality.get("notes", "")

                        # Emotion info
                        emotion = seg.get("emotion", {})
                        mood = emotion.get("mood", "")
                        intensity = emotion.get("intensity", "")
                        narrative_func = emotion.get("narrative_function", "")

                        # Editor recommendation
                        editor_rec = seg.get("editor_recommendation", "")

                        # Get base character_presence from VLM scene analysis
                        character_presence = seg.get("character_presence", {})

                        scene = {
                            "scene_time": seg.get("timestamp", ""),
                            "description": " ".join(desc_parts),
                            "cut_type": seg.get("cut_type", ""),
                            "visual_quality": {
                                "score": quality_score,
                                "notes": quality_notes
                            },
                            "emotion": {
                                "mood": mood,
                                "intensity": intensity,
                                "narrative_function": narrative_func
                            },
                            "character_presence": character_presence,
                            "editor_recommendation": editor_rec,
                            "duration_sec": 0
                        }

                        # Calculate absolute timestamps and duration
                        seg_start_sec = None
                        seg_end_sec = None
                        if "timestamp" in seg:
                            range_match = re.search(r'([0-9:.]+)\s+to\s+([0-9:.]+)', seg["timestamp"], re.IGNORECASE)
                            if range_match:
                                try:
                                    # Timestamps from model are relative to the clip start (00:00:00)
                                    # We need to convert them to absolute timestamps
                                    s_rel = hhmmss_to_seconds(range_match.group(1))
                                    e_rel = hhmmss_to_seconds(range_match.group(2))

                                    s_abs = start_sec + s_rel
                                    e_abs = start_sec + e_rel

                                    seg_start_sec = s_abs
                                    seg_end_sec = e_abs

                                    scene["scene_time"] = f"{convert_seconds_to_hhmmss(s_abs)} to {convert_seconds_to_hhmmss(e_abs)}"
                                    scene["duration_sec"] = round(e_abs - s_abs, 2)
                                except ValueError:
                                    pass

                        result["internal_scenes"].append(scene)


                    # Validate that internal_scenes cover the requested time range
                    total_requested_duration = end_sec - start_sec
                    covered_duration = sum(scene.get("duration_sec", 0) for scene in result["internal_scenes"])
                    coverage_ratio = covered_duration / total_requested_duration if total_requested_duration > 0 else 0

                    min_coverage_ratio = 0.5  # Require at least 50% coverage
                    if coverage_ratio < min_coverage_ratio:
                        print(f"⚠️ trim_shot output validation failed:")
                        print(f"   Requested: {total_requested_duration:.2f}s ({clip_start_time} to {clip_end_time})")
                        print(f"   Covered: {covered_duration:.2f}s (ratio: {coverage_ratio:.1%})")
                        print(f"   Scenes returned: {len(result['internal_scenes'])}")

                        # Print scene details for debugging
                        for i, scene in enumerate(result["internal_scenes"]):
                            scene_time = scene.get("scene_time", "unknown")
                            scene_dur = scene.get("duration_sec", 0)
                            print(f"   Scene {i+1}: {scene_time} ({scene_dur:.2f}s)")

                        if tries > 0:
                            print(f"   Retrying... ({tries} attempts remaining)")
                            continue
                        else:
                            print(f"   ⚠️ Max retries reached. Returning partial result.")
                            # Add warning to result
                            result["usability_assessment"] = (
                                f"⚠️ WARNING: Model only provided {coverage_ratio:.0%} coverage of requested range. "
                                f"Scenes may be incomplete or improperly segmented. Consider calling trim_shot with a different time range."
                            )

                    return json.dumps(result, indent=4, ensure_ascii=False)
                
            except json.JSONDecodeError as e:
                print(f"❌ [Error] JSON decode error for time range {time_range}: {e}")
                print(f"📄 [Data] Raw content (first 500 chars): {content_str[:500]}")
                if tries == 0:
                    return f"Error: Failed to parse model response for time range {time_range}. Content: {content_str[:200]}"
                continue
            except Exception as e:
                print(f"❌ [Error] Unexpected error processing response for time range {time_range}: {e}")
                if tries == 0:
                    return f"Error: Unexpected error processing response: {str(e)}"
                continue

        return f"Error: Failed to generate caption for time range {time_range} after multiple attempts."
    finally:
        b64_frames = None
        litellm_messages = None
        gc.collect()


class EditorCoreAgent:
    """
    Core ReAct-style agent for video editing.
    核心 ReAct 风格的视频剪辑智能体。
    
    This agent follows the ReAct (Reasoning + Acting) pattern:
    该智能体遵循 ReAct（推理 + 行动）模式：
    - reads one shot requirement from shot_plan
    - 从 shot_plan 读取单个镜头需求
    - iteratively calls tools to retrieve/trim/review
    - 迭代调用工具进行检索/细裁/审查
    - commits final timestamp range into shot_point output
    - 将最终时间戳写入 shot_point 输出
    
    Java 类比：这是一个状态机，通过循环调用工具来完成任务
    """
    def __init__(self, video_caption_path, video_scene_path, audio_caption_path, output_path, max_iterations, video_path=None, video_reader=None, frame_folder_path=None, transcript_path: str = None):
        # 初始化可用工具列表
        self.tools = [semantic_neighborhood_retrieval, fine_grained_shot_trimming, review_clip, commit]
        
        # 创建工具名称到函数的映射字典
        # Java 类比：Map<String, Function> toolMap
        self.name_to_function_map = {tool.__name__: tool for tool in self.tools}
        for original_name, alias_name in TOOL_NAME_ALIASES.items():
            if original_name in self.name_to_function_map:
                self.name_to_function_map[alias_name] = self.name_to_function_map[original_name]

        # 构建工具的 JSON Schema（用于 LLM 函数调用）
        self.function_schemas = []
        for func in self.tools:
            schema = as_json_schema(func)
            schema["name"] = func.__name__
            display_name = TOOL_NAME_ALIASES.get(func.__name__)
            if display_name:
                original_desc = schema.get("description", "")
                schema["description"] = f"Display name: {display_name}.\n{original_desc}".strip()
            self.function_schemas.append({"function": schema, "type": "function"})
        # 保存路径和配置
        self.video_caption_path = video_caption_path
        self.video_scene_path = video_scene_path
        
        # 加载音频数据库（JSON 格式）
        # json.load: 从文件读取 JSON，类似 Java 的 Gson.fromJson()
        self.audio_db = json.load(open(audio_caption_path, 'r', encoding='utf-8'))
        self.max_iterations = max_iterations  # 最大迭代次数
        self.frame_folder_path = frame_folder_path
        self.video_path = video_path
        self.video_reader = _normalize_video_reader(video_reader)
        self.transcript_path = transcript_path
        self.output_path = output_path
        self.current_target_length = None  # Will be set during run()  # 当前目标时长
        self.messages = self._construct_messages()  # 构建初始消息列表
        
        # Track used time ranges to avoid duplicate clip selection
        # 跟踪已使用的时间范围，避免重复选择片段
        self.used_time_ranges = []  # List of (start_sec, end_sec) tuples
        self.current_section_idx = None
        self.current_shot_idx = None
        self.current_related_scenes = []  # Will be set during run() for each shot
        
        # 跟踪已尝试的 trim_shot 时间范围，避免重复调用
        self.attempted_time_ranges = set()  # Track attempted trim_shot time ranges to avoid duplicate calls
        self.duplicate_call_count = 0  # Count consecutive duplicate calls  # 连续重复调用计数
        self.max_duplicate_calls = 3  # Max duplicates before restart  # 重启前的最大重复次数
        self.forbidden_time_ranges = []  # Global avoid ranges injected by orchestrator  # 全局禁止时间范围
        self.guidance_text = None
        self.last_commit_result = None
        self.last_commit_raw = None

        # Initialize ReviewerAgent for finish validation
        # 初始化审核器智能体，用于完成验证
        self.reviewer = ReviewerAgent(
            frame_folder_path=frame_folder_path,
            video_path=video_path
        )
        # Current shot context for reviewer
        # 当前镜头上下文，供审核器使用
        self.current_shot_context = {}

    def _load_progress(self):
        """
        Load progress from existing output file to support resume functionality.
        从现有输出文件加载进度以支持断点续传功能。
        
        Returns a set of completed (section_idx, shot_idx) tuples.
        返回已完成的 (section_idx, shot_idx) 元组集合。
        
        Java 类比：类似从数据库或文件中恢复任务状态
        """
        # 检查输出路径是否存在
        if not self.output_path or not os.path.exists(self.output_path):
            return set()

        try:
            # with open: 上下文管理器，自动关闭文件，类似 Java 的 try-with-resources
            with open(self.output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)

            # 创建已完成任务的集合（set 是无序不重复集合）
            # Java 类比：Set<Tuple<Integer, Integer>> completed = new HashSet<>();
            completed = set()
            for result in results:
                # dict.get(): 获取字典值，如果键不存在返回 None
                # Java 类比：map.get(key)
                if result.get('status') == 'success':
                    sec_idx = result.get('section_idx')
                    shot_idx = result.get('shot_idx')
                    if sec_idx is not None and shot_idx is not None:
                        # 将元组添加到集合中
                        completed.add((sec_idx, shot_idx))

            if completed:
                print(f"📋 Found {len(completed)} completed shots in existing output file")
                print(f"   Completed: {sorted(completed)}")

            return completed
        except Exception as e:
            # 捕获所有异常，避免程序崩溃
            print(f"⚠️ Error loading progress: {e}")
            return set()

    def _construct_messages(self):
        """
        Construct initial messages for the LLM agent.
        构建 LLM 智能体的初始消息。
        
        Returns:
            List of message dictionaries with system and user prompts.
            包含系统提示和用户提示的消息列表。
        """
        # 替换提示模板中的占位符为实际配置值
        # str.replace(): 字符串替换，类似 Java 的 String.replace()
        user_prompt = (
            EDITOR_USER_PROMPT
            .replace("SCENE_EXPLORATION_RANGE_PLACEHOLDER", str(getattr(config, 'SCENE_EXPLORATION_RANGE', 3)))
            .replace("MIN_PROTAGONIST_RATIO_PLACEHOLDER", f"{config.MIN_PROTAGONIST_RATIO * 100:.0f}")
            .replace("MIN_ACCEPTABLE_SHOT_DURATION_PLACEHOLDER", str(getattr(config, 'MIN_ACCEPTABLE_SHOT_DURATION', 2.0)))
            .replace("ALLOW_DURATION_TOLERANCE_PLACEHOLDER", str(getattr(config, 'ALLOW_DURATION_TOLERANCE', 1.0)))
        )
        
        # 构建消息列表（system + user）
        # Java 类比：List<Map<String, String>> messages = new ArrayList<>();
        messages = [
            {"role": "system", "content": EDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return messages

    def _build_audio_section_info(self, audio_section, shot_idx):
        """
        Build audio section information string for the agent.
        构建音频段落信息字符串供智能体使用。
        
        Args:
            audio_section: 音频段落数据字典
            shot_idx: 镜头索引
        Returns:
            格式化的音频信息字符串
        """
        # 获取详细分析数据，默认为空字典
        detailed_analysis = audio_section.get('detailed_analysis', {})
        audio_info_parts = []

        # 提取音频段落的名称和描述
        if 'name' in audio_section:
            audio_info_parts.append(f"Section: {audio_section['name']}")
        if 'description' in audio_section:
            audio_info_parts.append(f"Description: {audio_section['description']}")

        # 提取摘要信息
        if isinstance(detailed_analysis, dict) and 'summary' in detailed_analysis:
            audio_info_parts.append(f"Summary: {detailed_analysis['summary']}")

        # 提取镜头字幕（可能是列表或字典格式）
        if isinstance(detailed_analysis, dict) and 'sections' in detailed_analysis:
            sections_list = detailed_analysis['sections']
            # 如果是列表，直接通过索引访问
            if isinstance(sections_list, list) and shot_idx < len(sections_list):
                audio_info_parts.append(f"Shot caption: {sections_list[shot_idx]}")
            # 如果是字典，通过字符串键访问
            elif isinstance(sections_list, dict) and str(shot_idx) in sections_list:
                audio_info_parts.append(f"Shot caption: {sections_list[str(shot_idx)]}")

        # 用换行符连接所有部分，如果没有内容则返回默认消息
        # "\n".join(): 将列表元素用换行符连接，类似 Java 的 String.join("\n", list)
        return "\n".join(audio_info_parts) if audio_info_parts else "No audio information available"

    def _prepare_shot_messages(self, shot, audio_section_info, related_scene_value, guidance_text=None, forbidden_time_ranges=None):
        """
        Prepare messages for a specific shot by replacing placeholders.
        为特定镜头准备消息，替换占位符。
        
        Args:
            shot: 镜头数据字典
            audio_section_info: 音频段落信息字符串
            related_scene_value: 相关场景值
            guidance_text: 指导文本（可选）
            forbidden_time_ranges: 禁止的时间范围列表（可选）
        Returns:
            准备好的消息列表
        """
        # copy.deepcopy: 深拷贝，创建完全独立的副本
        # Java 类比：使用序列化/反序列化或手动克隆实现深拷贝
        msgs = copy.deepcopy(self.messages)
        
        # 替换用户消息中的占位符为实际值
        # msgs[-1]: 访问列表最后一个元素，类似 Java 的 list.get(list.size() - 1)
        msgs[-1]["content"] = msgs[-1]["content"].replace("VIDEO_LENGTH_PLACEHOLDER", str(shot['time_duration']))
        msgs[-1]["content"] = msgs[-1]["content"].replace("CURRENT_VIDEO_CONTENT_PLACEHOLDER", shot['content']).replace("CURRENT_VIDEO_EMOTION_PLACEHOLDER", shot['emotion'])
        msgs[-1]["content"] = msgs[-1]["content"].replace("BACKGROUND_MUSIC_PLACEHOLDER", audio_section_info)

        recommended_scenes_str = str(related_scene_value) if related_scene_value else "None specified"
        msgs[-1]["content"] = msgs[-1]["content"].replace("RECOMMENDED_SCENES_PLACEHOLDER", recommended_scenes_str)

        # 如果有指导文本或禁止时间范围，添加额外的用户消息
        if guidance_text or forbidden_time_ranges:
            avoid_msg = []
            if forbidden_time_ranges:
                formatted = []
                # 将秒数转换为 HH:MM:SS 格式
                for start_sec, end_sec in forbidden_time_ranges:
                    formatted.append(f"{convert_seconds_to_hhmmss(start_sec)} to {convert_seconds_to_hhmmss(end_sec)}")
                avoid_msg.append("Avoid time ranges: " + "; ".join(formatted))
            if guidance_text:
                avoid_msg.append("Guidance: " + guidance_text)
            
            # 添加新的用户消息到消息列表
            msgs.append({
                "role": "user",
                "content": "\n".join(avoid_msg)
            })

        return msgs

    def _run_shot_loop(self, msgs, max_iterations=None):
        """
        Run the ReAct loop for a single shot.
        为单个镜头运行 ReAct 循环。
        
        This is the core reasoning + acting loop where the agent:
        这是核心的推理+行动循环，智能体在此：
        1. Calls LLM to decide next action
        1. 调用 LLM 决定下一步行动
        2. Executes the chosen tool
        2. 执行选择的工具
        3. Observes the result and continues
        3. 观察结果并继续
        
        Args:
            msgs: 消息列表
            max_iterations: 最大迭代次数
        Returns:
            (should_restart, section_completed) 元组
        """
        if max_iterations is None:
            max_iterations = self.max_iterations

        should_restart = False
        section_completed = False

        # One "shot loop" = one target shot from shot_plan.
        # 一个“shot loop”对应 shot_plan 中一个目标镜头。
        # It can restart when agent degenerates (e.g., repeated duplicate trim calls).
        # 当智能体行为退化（如反复重复 trim 调用）时会触发重启。
        for i in range(max_iterations):
            # 如果是最后一次迭代，添加强制完成提示
            if i == max_iterations - 1:
                msgs.append(
                    {
                        "role": "user",
                        "content": EDITOR_FINISH_PROMPT,
                    }
                )

            # 获取最大重试次数配置
            max_model_retries = getattr(config, "AGENT_MODEL_MAX_RETRIES", 2)
            max_tool_retries = 2
            tool_execution_success = False

            # tool_retry handles cases where a tool call/response was unusable.
            # tool_retry 用于处理工具调用或结果不可用的情况。
            for tool_retry in range(max_tool_retries):
                # 记录消息长度，用于回滚而不是深拷贝（性能优化）
                msgs_snapshot_len = len(msgs)  # track length for rollback instead of deepcopy

                response = None
                context_length_error = False
                
                # model_retry handles transient model/provider errors.
                # model_retry 用于处理模型/服务商的瞬时错误。
                for model_retry in range(max_model_retries):
                    tool_calls_raw = None
                    try:
                        # 构建 LLM 调用参数
                        # dict(): 创建字典，类似 Java 的 HashMap
                        kwargs = dict(
                            model=config.AGENT_LITELLM_MODEL,
                            messages=msgs,
                            temperature=1.0,  # 温度参数，控制随机性
                            max_tokens=config.AGENT_MODEL_MAX_TOKEN,
                            tools=self.function_schemas,  # 可用工具列表
                            tool_choice="auto",  # 自动选择是否调用工具
                        )
                        if config.AGENT_LITELLM_URL:
                            kwargs["api_base"] = config.AGENT_LITELLM_URL
                        if config.AGENT_LITELLM_API_KEY:
                            kwargs["api_key"] = config.AGENT_LITELLM_API_KEY
                        
                        # litellm.completion: 调用 LLM API
                        # 类似 Java 的 HTTP 客户端调用 REST API
                        raw = litellm.completion(**kwargs)
                        
                        # 获取 LLM 响应的第一个选择
                        # raw.choices[0]: 列表访问，类似 Java 的 list.get(0)
                        msg = raw.choices[0].message
                        
                        # getattr: 获取属性，带默认值
                        # 尝试获取 tool_calls 属性，如果不存在返回 None
                        tool_calls_raw = getattr(msg, "tool_calls", None)
                        
                        # 构建响应字典
                        # Java 类比：Map<String, Object> response = new HashMap<>();
                        response = {
                            "role": msg.role or "assistant",
                            "content": msg.content,
                            # 列表推导式：将工具调用转换为字典列表
                            # Java Stream 等价：
                            # List<Map<String, Object>> toolCalls = toolCallsRaw.stream()
                            #     .map(tc -> convertToMap(tc))
                            #     .collect(Collectors.toList());
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in (tool_calls_raw or [])
                            ] or None,
                        }
                        
                        # 获取推理内容（如果模型支持）
                        reasoning = getattr(msg, "reasoning_content", None)
                        if reasoning:
                            response["reasoning_content"] = reasoning
                        
                        # 如果成功获取响应，跳出重试循环
                        if response is not None:
                            break
                        else:
                            print(f"🔄 [Retry] Model returned None, retrying ({model_retry + 1}/{max_model_retries})...")
                    except Exception as e:
                        # 捕获 LLM 调用异常
                        error_msg = str(e).lower()
                        
                        # 检查是否是上下文长度超限错误
                        if "context length" in error_msg or "too large" in error_msg or "max_tokens" in error_msg:
                            print(f"❌ [Error] Context length exceeded: {e}")
                            context_length_error = True
                            break
                        
                        # 检查是否是速率限制错误
                        is_rate_limited = (
                            "ratelimit" in error_msg
                            or "max organization concurrency" in error_msg
                            or "too many requests" in error_msg
                            or " 429" in error_msg  # HTTP 429 Too Many Requests
                        )
                        
                        if is_rate_limited:
                            # 解析重试等待时间
                            base_wait = _parse_retry_after_seconds(
                                str(e),
                                default_seconds=getattr(config, "AGENT_RATE_LIMIT_BACKOFF_BASE", 1.0),
                            )
                            max_backoff = getattr(config, "AGENT_RATE_LIMIT_MAX_BACKOFF", 8.0)
                            
                            # 指数退避：wait = base * 2^retry
                            # Java 类比：Math.min(maxBackoff, baseWait * Math.pow(2, retry))
                            wait_seconds = min(max_backoff, base_wait * (2 ** model_retry))
                            print(
                                f"Rate limit encountered, sleeping {wait_seconds:.1f}s "
                                f"before retry ({model_retry + 1}/{max_model_retries})..."
                            )
                            # time.sleep: 暂停执行，类似 Java 的 Thread.sleep()
                            time.sleep(wait_seconds)
                        
                        print(f"🔄 [Retry] Model call failed: {e}, retrying ({model_retry + 1}/{max_model_retries})...")
                        
                        # 如果是最后一次重试，抛出异常
                        if model_retry == max_model_retries - 1:
                            raise

                if context_length_error:
                    # 上下文长度超限，触发重启
                    print("🔄 [Restart] Triggering restart due to context overflow...")
                    should_restart = True
                    break

                if response is None:
                    # LLM 调用失败，回滚消息列表
                    print(f"❌ [Error] Model call failed after {max_model_retries} retries.")
                    # msgs[:]: 切片赋值，修改原列表
                    # Java 类比：list.subList(0, snapshotLen).clear()
                    msgs[:] = msgs_snapshot
                    break

                # 设置默认角色为 assistant
                response.setdefault("role", "assistant")
                
                # 将响应添加到消息历史
                msgs.append(response)
                print("#### Iteration: ", i, f"(Tool retry: {tool_retry + 1}/{max_tool_retries})" if tool_retry > 0 else "")
                print(response)

                tool_execution_failed = False

                try:
                    # 获取工具调用列表
                    # dict.get(): 获取字典值，如果键不存在返回默认值 []
                    tool_calls = response.get("tool_calls", [])
                    if tool_calls is None:
                        tool_calls = []

                        if not tool_calls:
                            # Encourage tool-using behavior. Raw free-text answers
                            # 引导模型优先调用工具。仅自由文本回答通常证据不足，
                            # are usually under-grounded before final commit.
                            # 不适合直接进入最终 commit。
                            content = response.get("content", "")
                            
                            # 检查是否是最终的镜头时间范围回答
                            # re.search: 搜索正则匹配，类似 Java 的 Matcher.find()
                            final_shot_pattern = re.search(r'\[shot:\s*[\d:.]+\s+to\s+[\d:.]+\s*\]', content, re.IGNORECASE)
                            is_short_response = len(content) < 500
                            is_final_answer = final_shot_pattern and (is_short_response or content.strip().endswith(']'))

                            if is_final_answer:
                                print("✅ [Agent] Model returned final answer. Task completed.")
                                section_completed = True
                                tool_execution_success = True
                                break
                            else:
                                print("⚠️  [Agent] Model did not call any tool. Adding prompt to use tools...")
                                msgs.append({
                                    "role": "user",
                                    "content": EDITOR_USE_TOOL_PROMPT
                                })

                    # 遍历所有工具调用并执行
                    for tool_call in tool_calls:
                        # _exec_tool: 执行工具，返回是否完成或需要重启
                        is_finished = self._exec_tool(tool_call, msgs)
                        if is_finished == "RESTART":
                            should_restart = True
                            break
                        if is_finished:
                            section_completed = True
                            break

                    if should_restart:
                        print("🔄 [Restart] Restarting conversation for current shot...")
                        break

                    tool_execution_success = True

                except StopException:
                    # 捕获停止异常，表示任务已完成
                    # Java 类比：throw new CustomException("Task completed")
                    return True, False
                except Exception as e:
                    # 捕获工具执行异常
                    print(f"❌ [Error] Error executing tool calls: {e}")
                    import traceback
                    traceback.print_exc()  # 打印完整堆栈跟踪
                    
                    # 为每个工具调用添加错误消息
                    for tc in (response.get("tool_calls") or []):
                        self._append_tool_msg(
                            tc["id"],
                            tc["function"]["name"],
                            f"Tool execution error: {e}",
                            msgs,
                        )
                    tool_execution_success = True
                    tool_execution_failed = False

                # 如果工具执行成功或达到最大重试次数，跳出重试循环
                if tool_execution_success or tool_retry == max_tool_retries - 1:
                    break

                if tool_execution_failed:
                    # 回滚消息列表到快照位置
                    # del msgs[start:end]: 删除列表切片，类似 Java 的 list.subList(start, end).clear()
                    print("🔄 [Retry] Rolling back messages and retrying...")
                    del msgs[msgs_snapshot_len:]
                    continue

            if should_restart:
                break

            if section_completed:
                print(f"⏭️  [Progress] Shot {self.current_shot_idx + 1} completed. Moving to next shot...")
                break

        # 返回 (section_completed, should_restart) 元组
        # Python 可以返回多个值，自动打包为元组
        # Java 等价：return new Pair<Boolean, Boolean>(sectionCompleted, shouldRestart);
        return section_completed, should_restart

    # ------------------------------------------------------------------ #
    # Helper methods
    # 辅助方法
    # ------------------------------------------------------------------ #
    def _append_tool_msg(self, tool_call_id, name, content, msgs):
        """
        Append tool execution result to message history.
        将工具执行结果添加到消息历史。
        
        Args:
            tool_call_id: 工具调用 ID
            name: 工具名称
            content: 工具返回内容
            msgs: 消息列表（会被修改）
        """
        # 添加工具响应消息到对话历史
        # Java 类比：messages.add(Map.of("tool_call_id", id, "role", "tool", ...))
        msgs.append(
            {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": name,
                "content": content,
            }
        )

    def _exec_tool(self, tool_call, msgs):
        """
        Execute a tool call and append result to message history.
        执行工具调用并将结果添加到消息历史。
        
        Args:
            tool_call: 工具调用字典
            msgs: 消息列表（会被修改）
        Returns:
            True/False/"RESTART" - 是否完成或需要重启
        """
        # 获取工具名称并转换为标准形式
        name = tool_call["function"]["name"]
        canonical_name = _canonical_tool_name(name)
        
        # 检查工具是否存在
        if canonical_name not in self.name_to_function_map:
            self._append_tool_msg(tool_call["id"], name, f"Invalid function name: {name!r}", msgs)
            return False

        # Parse arguments
        # 解析工具参数（JSON 字符串）
        try:
            # json.loads: 将 JSON 字符串解析为 Python 对象
            # Java 类比：Gson.fromJson(jsonString, Map.class)
            args = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError as exc:
            raise StopException(f"Error decoding arguments: {exc!s}")

        # Inject runtime context that the model does not need to guess:
        # 注入模型无需猜测的运行时上下文：
        # scene paths, transcript paths, used ranges, target duration, etc.
        # 场景路径、字幕路径、已用范围、目标时长等。
        if "topk" in args:
            if config.OVERWRITE_CLIP_SEARCH_TOPK > 0:
                args["topk"] = config.OVERWRITE_CLIP_SEARCH_TOPK

        # For semantic_neighborhood_retrieval, inject scene_folder_path and recommended_scenes
        # 对于语义邻域检索工具，注入场景路径和推荐场景
        if canonical_name == "semantic_neighborhood_retrieval":
            agent_requested_scenes = args.get("related_scenes", [])
            if agent_requested_scenes and isinstance(agent_requested_scenes, list):
                # Agent explicitly requested specific scenes - will be validated in function
                print(f"📍 Agent requested scenes: {agent_requested_scenes}")
            else:
                # No agent request, will use default recommended related scenes in function
                print(f"📍 No specific scenes requested, will use recommended: {self.current_related_scenes}")

            # Inject both scene_folder_path and recommended_scenes for validation
            args["scene_folder_path"] = self.video_scene_path
            args["recommended_scenes"] = self.current_related_scenes

        # For fine_grained_shot_trimming, inject video/transcript parameters and check for duplicate calls
        # 对于细粒度镜头裁剪工具，注入视频/字幕参数并检查重复调用
        if canonical_name == "fine_grained_shot_trimming":
            # 注入视频路径或读取器
            if self.video_path:
                args["frame_path"] = self.video_path
            elif self.video_reader is None:
                # 如果既没有视频路径也没有读取器，返回错误
                self._append_tool_msg(
                    tool_call["id"],
                    name,
                    "Error: neither video_reader nor video_path is configured in agent.",
                    msgs
                )
                return False

            # 注入字幕路径（如果存在）
            if self.transcript_path:
                args["transcript_path"] = self.transcript_path

            # Check for duplicate time range calls to prevent infinite loops
            # 检查重复的时间范围调用，防止无限循环
            time_range = args.get("time_range", "")
            # Normalize the time range for comparison (remove extra spaces)
            # 标准化时间范围以便比较（移除多余空格）
            # str.split(): 分割字符串，类似 Java 的 String.split()
            # " ".join(): 用空格连接，类似 Java 的 String.join(" ", array)
            normalized_range = " ".join(time_range.split())

            # 检查是否已经尝试过这个时间范围
            if normalized_range in self.attempted_time_ranges:
                self.duplicate_call_count += 1
                print(f"⚠️ Duplicate call detected ({self.duplicate_call_count}/{self.max_duplicate_calls}): {normalized_range}")

                # 如果达到最大重复次数，触发重启
                if self.duplicate_call_count >= self.max_duplicate_calls:
                    print(f"🔄 Max duplicate calls reached. Restarting conversation for this shot...")
                    return "RESTART"  # Signal to restart the conversation

                # Return a helpful message instead of calling the tool again
                # 返回提示消息而不是再次调用工具
                self._append_tool_msg(
                    tool_call["id"],
                    name,
                    f"Warning: You have already analyzed '{time_range}'. "
                    f"Duplicate call {self.duplicate_call_count}/{self.max_duplicate_calls}. "
                    f"Call 'Commit' NOW with your best selection, or conversation will restart.",
                    msgs
                )
                return False

            # Reset duplicate counter on new time range
            # 新的时间范围，重置重复计数器
            self.duplicate_call_count = 0
            # Record this time range as attempted
            # 记录这个时间范围为已尝试
            # set.add(): 添加到集合，类似 Java 的 Set.add()
            self.attempted_time_ranges.add(normalized_range)
        
        # For review_clip, inject used_time_ranges
        # 对于 review_clip 工具，注入已使用的时间范围
        if canonical_name == "review_clip":
            # 合并已使用的时间范围和禁止的时间范围
            # list + list: 列表拼接，类似 Java 的 List.addAll()
            args["used_time_ranges"] = self.used_time_ranges + (self.forbidden_time_ranges or [])
            print(f"📍 Checking overlap against {len(self.used_time_ranges)} used clips")

        # commit is protected by Reviewer checks so low-quality or duplicated
        # commit 前有 Reviewer 保护，低质量或重复片段提案会被拦截，
        # proposals can be rejected before writing final output.
        # 避免写入最终输出。
        if canonical_name == "commit":
            # 注入视频路径、输出路径、目标时长等参数
            args["video_path"] = self.video_path or ""
            args["output_path"] = self.output_path or ""
            args["target_length_sec"] = self.current_target_length or 0.0
            args["section_idx"] = self.current_section_idx if self.current_section_idx is not None else -1
            args["shot_idx"] = self.current_shot_idx if self.current_shot_idx is not None else -1
            # Note: protagonist_frame_data will be set after face quality check

            # Enforce forbidden time ranges (from parallel orchestrator guidance)
            # 强制执行禁止的时间范围（来自并行协调器的指导）
            if self.forbidden_time_ranges:
                # 解析提议的时间范围
                proposed_ranges = _parse_shot_time_ranges(args.get("answer", ""))
                if not proposed_ranges:
                    self._append_tool_msg(
                        tool_call["id"],
                        name,
                        "Error: Could not parse shot time range for overlap checks. Please use format: [shot: HH:MM:SS to HH:MM:SS]",
                        msgs
                    )
                    return False
                
                # 检查是否与禁止范围重叠
                for p_start, p_end in proposed_ranges:
                    for f_start, f_end in self.forbidden_time_ranges:
                        # _ranges_overlap: 检查两个时间范围是否重叠
                        if _ranges_overlap(p_start, p_end, f_start, f_end):
                            self._append_tool_msg(
                                tool_call["id"],
                                name,
                                "Overlap detected with forbidden ranges. Please select a different time range.",
                                msgs
                            )
                            return False

            # Call ReviewerAgent to validate before executing finish
            # 调用 ReviewerAgent 在执行完成前进行验证
            if config.ENABLE_REVIEWER:
                # 构建镜头提案字典
                shot_proposal = {
                    "answer": args.get("answer", ""),
                    "target_length_sec": self.current_target_length or 0.0
                }

                # Face quality check (optional, controlled by config.ENABLE_FACE_QUALITY_CHECK)
                # 面部质量检查（可选，由配置控制）
                if config.ENABLE_FACE_QUALITY_CHECK:
                    # 从回答中提取时间范围
                    # re.search: 搜索正则匹配，group(1) 和 group(2) 是捕获组
                    time_match = re.search(r'\[?shot[\s_]*\d*:\s*([0-9:.]+)\s+to\s+([0-9:.]+)\]?', shot_proposal["answer"], re.IGNORECASE)
                    if time_match:
                        time_range = f"{time_match.group(1)} to {time_match.group(2)}"

                        # 获取面部检查方法配置
                        face_check_method = getattr(config, 'FACE_QUALITY_CHECK_METHOD', 'vlm')
                        if face_check_method != 'vlm':
                            print("⚠️  FACE_QUALITY_CHECK_METHOD is not 'vlm'; falling back to 'vlm' (face_recognition removed).")
                            face_check_method = 'vlm'

                        print("🎬 Using VLM face quality check method...")
                        
                        # 调用审核器的面部质量检查方法
                        # check_face_quality_vlm: 使用 VLM 检查面部质量
                        face_check, protagonist_frame_data = self.reviewer.check_face_quality_vlm(
                            video_path=args.get("video_path", ""),
                            time_range=time_range,
                            main_character_name=getattr(config, 'MAIN_CHARACTER_NAME', 'the main character'),
                            min_protagonist_ratio=getattr(config, 'MIN_PROTAGONIST_RATIO', 0.7),
                            min_box_size=getattr(config, 'VLM_MIN_BOX_SIZE', 50),
                            return_frame_data=True,  # 返回每帧的检测数据
                        )

                        # Store for debugging/trace
                        # 存储用于调试/跟踪
                        self.current_shot_context["face_quality"] = face_check
                        self.current_shot_context["face_quality_method"] = face_check_method

                        # 如果面部检查失败，返回错误消息
                        if "❌" in face_check or "FAILED" in face_check:
                            self._append_tool_msg(
                                tool_call["id"],
                                name,
                                f"Review Failed - Face quality check ({face_check_method}) did not pass:\n{face_check}",
                                msgs
                            )
                            return False

                        # 保存主角帧检测数据
                        if not protagonist_frame_data:
                            self.current_shot_context["protagonist_frame_data"] = None
                            args["protagonist_frame_data"] = None
                        else:
                            self.current_shot_context["protagonist_frame_data"] = protagonist_frame_data
                            args["protagonist_frame_data"] = protagonist_frame_data

                # Set protagonist_frame_data if not already set
                # 如果尚未设置，则设置主角帧数据
                if "protagonist_frame_data" not in args or args.get("protagonist_frame_data") is None:
                    context_data = self.current_shot_context.get("protagonist_frame_data", None)
                    if context_data:
                        args["protagonist_frame_data"] = context_data
                        print(f"✅ Set protagonist_frame_data from context: {len(context_data)} detections")
                    else:
                        args["protagonist_frame_data"] = None

                # 调用审核器进行综合审查
                review_result = self.reviewer.review(
                    shot_proposal=shot_proposal,
                    context=self.current_shot_context,
                    used_time_ranges=self.used_time_ranges
                )

                self.current_shot_context["review_result"] = review_result

                # 如果审核不通过，返回反馈消息
                if not review_result["approved"]:
                    self._append_tool_msg(
                        tool_call["id"],
                        name,
                        f"Review Failed - Please adjust your selection:\n{review_result['feedback']}",
                        msgs
                    )
                    return False  # Continue iteration  # 继续迭代

        # Call the tool
        # 调用工具函数
        try:
            # **args: 字典解包，将字典作为关键字参数传递
            # Java 类比：无法直接等价，需要通过反射或 Map 传参
            result = self.name_to_function_map[canonical_name](**args)
            
            # 准备用于消息的工具结果（可能需要截断）
            tool_result_for_msg = result
            
            if canonical_name == "commit":
                # commit payload can include large frame-level detections; avoid flooding logs/context
                # commit 载荷可能包含大量帧级检测数据；避免淹没日志/上下文
                # _compact_json_str_for_log: 截断 JSON 字符串用于日志
                tool_result_for_msg = _compact_json_str_for_log(result)
            elif canonical_name == "fine_grained_shot_trimming":
                # Truncate internal_scenes list to avoid msgs growing unboundedly across iterations
                # 截断 internal_scenes 列表，避免消息在迭代中无限增长
                try:
                    parsed = json.loads(result)
                    scenes = parsed.get("internal_scenes", [])
                    max_scenes = getattr(config, "TRIM_SHOT_MAX_SCENES_IN_HISTORY", 8)
                    if len(scenes) > max_scenes:
                        # 截断场景列表
                        parsed["internal_scenes"] = scenes[:max_scenes]
                        parsed["_scenes_truncated"] = f"{len(scenes) - max_scenes} more scenes omitted from context"
                    # json.dumps: 将 Python 对象转换为 JSON 字符串
                    tool_result_for_msg = json.dumps(parsed, ensure_ascii=False)
                except Exception:
                    pass
            
            # 将工具结果添加到消息历史
            self._append_tool_msg(tool_call["id"], name, tool_result_for_msg, msgs)

            # Check if commit was successful
            # 检查 commit 是否成功
            if canonical_name == "commit":
                # Parse result as JSON and check status
                # 解析结果为 JSON 并检查状态
                try:
                    result_data = json.loads(result)
                    if result_data.get("status") == "success":
                        # 保存最后一次成功的 commit 结果
                        self.last_commit_result = result_data
                        self.last_commit_raw = result
                        
                        # Record used time ranges to prevent duplicate selection
                        # 记录已使用的时间范围，防止重复选择
                        clips = result_data.get("clips", [])
                        for clip in clips:
                            start_str = clip.get("start", "")
                            end_str = clip.get("end", "")
                            if start_str and end_str:
                                # Convert to seconds for comparison
                                # 转换为秒数以便比较
                                def hhmmss_to_sec(t):
                                    """将 HH:MM:SS 格式转换为秒数"""
                                    parts = t.strip().split(':')
                                    if len(parts) == 3:
                                        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                                    elif len(parts) == 2:
                                        return int(parts[0]) * 60 + float(parts[1])
                                    return float(parts[0])
                                
                                start_sec = hhmmss_to_sec(start_str)
                                end_sec = hhmmss_to_sec(end_str)
                                # 添加到已使用时间范围列表
                                self.used_time_ranges.append((start_sec, end_sec))
                        
                        return True  # Signal to break the current section loop  # 信号：跳出当前段落循环
                except json.JSONDecodeError:
                    # If not JSON, check for success message in string
                    # 如果不是 JSON，检查字符串中是否有成功消息
                    if "Successfully validated shot selection" in result:
                        return True
            
            # 工具执行完成但未 commit，返回 False 继续迭代
            return False
        except StopException as exc:  # graceful stop
            # 捕获停止异常，优雅地停止
            # Java 类比：catch (CustomStopException e) { throw e; }
            raise

    # ------------------------------------------------------------------ #
    # Main loop
    # 主循环
    # ------------------------------------------------------------------ #
    def run(self, shot_plan_path: str) -> list[dict]:
        """
        Run the ReAct-style loop with OpenAI Function Calling.
        运行带有 OpenAI 函数调用的 ReAct 风格循环。

        Args:
            shot_plan_path: Path to a pre-generated shot_plan.json file.
            shot_plan_path: 预生成的 shot_plan.json 文件路径。
        
        Returns:
            所有镜头的结果列表
        """

        # Load shot plan from file
        # 从文件加载镜头计划
        print(f"📂 [Init] Loading shot plan from: {shot_plan_path}")
        with open(shot_plan_path, 'r', encoding='utf-8') as f:
            structure_proposal = json.load(f)

        # Load progress from existing output file (for resume functionality)
        # 从现有输出文件加载进度（用于断点续传）
        completed_shots = self._load_progress()

        # Store original output path and create section-specific paths
        # 保存原始输出路径并创建段落特定路径
        original_output_path = self.output_path
        print("📄 [Data] structure_proposal: ", structure_proposal)
        
        # 遍历视频结构中的每个段落
        # enumerate: 枚举，返回 (index, item) 元组
        # Java 类比：for (int i = 0; i < list.size(); i++) { Item item = list.get(i); }
        for sec_idx, sec_cur in enumerate(structure_proposal['video_structure']):
            print(f"\n{'='*60}")
            print(f"Processing Section {sec_idx + 1}/{len(structure_proposal['video_structure'])}")
            print(f"{'='*60}\n")
            
            # Set current section for reporting
            # 设置当前段落索引用于报告
            self.current_section_idx = sec_idx
            
            # Load shot_plan from sec_cur (loaded from file)
            # 从段落数据中加载镜头计划
            shot_plan = sec_cur.get('shot_plan')
            if not shot_plan:
                print(f"❌ [Error] No shot_plan found for section {sec_idx}")
                continue
            print("Using shot plan from file")
            
            # 遍历段落中的每个镜头
            for idx, shot in enumerate(shot_plan['shots']):
                # Check if this shot is already completed (resume functionality)
                # 检查此镜头是否已完成（断点续传功能）
                if (sec_idx, idx) in completed_shots:
                    print(f"\n⏭️  Skipping Shot {idx + 1}/{len(shot_plan['shots'])} - Already completed")
                    continue

                # 最大重启次数
                max_shot_restarts = 3  # Max restarts per shot
                for restart_attempt in range(max_shot_restarts):
                    if restart_attempt > 0:
                        print(f"\nRestart attempt {restart_attempt + 1}/{max_shot_restarts} for Shot {idx + 1}")

                    print(f"\n{'='*60}")
                    print(f"Processing Shot {idx + 1}/{len(shot_plan['shots'])}")
                    print(f"{'='*60}\n")
                    print("shot plan: ", shot)

                    # 重置输出路径和状态
                    self.output_path = original_output_path
                    print(f"Output path: {self.output_path}")
                    self.current_shot_idx = idx
                    
                    # 重置尝试过的时间范围和重复调用计数
                    self.attempted_time_ranges = set()
                    self.duplicate_call_count = 0

                    # 获取音频段落信息
                    audio_section = self.audio_db['sections'][sec_idx]
                    audio_section_info = self._build_audio_section_info(audio_section, idx)

                    # 设置当前目标时长
                    self.current_target_length = shot['time_duration']
                    
                    # 构建当前镜头上下文
                    self.current_shot_context = {
                        "content": shot.get('content', ''),
                        "emotion": shot.get('emotion', ''),
                        "section_idx": sec_idx,
                        "shot_idx": idx,
                        "time_duration": shot.get('time_duration', 0)
                    }

                    # 获取相关场景值
                    related_scene_value = shot.get('related_scene', [])
                    if isinstance(related_scene_value, int):
                        self.current_related_scenes = [related_scene_value]
                    elif isinstance(related_scene_value, list):
                        self.current_related_scenes = related_scene_value
                    else:
                        self.current_related_scenes = []

                    # 准备镜头消息
                    msgs = self._prepare_shot_messages(
                        shot=shot,
                        audio_section_info=audio_section_info,
                        related_scene_value=related_scene_value,
                    )

                    # 运行 ReAct 循环处理当前镜头
                    # _run_shot_loop: 返回 (section_completed, should_restart)
                    section_completed, should_restart = self._run_shot_loop(msgs, max_iterations=self.max_iterations)

                    # 如果段落完成，跳出重启循环
                    if section_completed:
                        break
                    
                    # 如果不需要重启，说明达到最大迭代次数
                    if not should_restart:
                        print(f"Max iterations reached for Shot {idx + 1}. Moving to next shot.")
                        break

                # End restart loop  # 重启循环结束

            # End of shot loop  # 镜头循环结束
            print(f"\nSection {sec_idx + 1} completed. All shots processed.")

        # 返回消息列表（包含所有交互历史）
        return msgs

    def run_single_shot(self, shot, sec_idx: int, shot_idx: int, guidance_text: str = None, forbidden_time_ranges: list = None, max_shot_restarts: int = 2, max_iterations: int = None):
        """
        Run a single shot selection loop and return the commit result dict on success.
        运行单个镜头选择循环，成功时返回 commit 结果字典。
        
        This method is used by ParallelShotOrchestrator for parallel processing.
        此方法被 ParallelShotOrchestrator 用于并行处理。
        
        Args:
            shot: 镜头数据
            sec_idx: 段落索引
            shot_idx: 镜头索引
            guidance_text: 指导文本（可选）
            forbidden_time_ranges: 禁止的时间范围列表（可选）
            max_shot_restarts: 最大重启次数
            max_iterations: 最大迭代次数
        Returns:
            commit 结果字典或 None
        """
        if max_iterations is None:
            max_iterations = self.max_iterations

        # 重置输出路径和状态
        self.output_path = ""
        self.current_section_idx = sec_idx
        self.current_shot_idx = shot_idx
        self.forbidden_time_ranges = forbidden_time_ranges or []
        self.guidance_text = guidance_text
        self.last_commit_result = None
        self.last_commit_raw = None

        # 获取音频段落信息
        audio_section = self.audio_db['sections'][sec_idx]
        audio_section_info = self._build_audio_section_info(audio_section, shot_idx)

        # 设置当前目标时长和上下文
        self.current_target_length = shot['time_duration']
        self.current_shot_context = {
            "content": shot.get('content', ''),
            "emotion": shot.get('emotion', ''),
            "section_idx": sec_idx,
            "shot_idx": shot_idx,
            "time_duration": shot.get('time_duration', 0)
        }

        # 获取相关场景值
        related_scene_value = shot.get('related_scene', [])
        if isinstance(related_scene_value, int):
            self.current_related_scenes = [related_scene_value]
        elif isinstance(related_scene_value, list):
            self.current_related_scenes = related_scene_value
        else:
            self.current_related_scenes = []

        # 重启循环：尝试多次处理单个镜头
        for restart_attempt in range(max_shot_restarts):
            if restart_attempt > 0:
                print(f"Restart attempt {restart_attempt + 1}/{max_shot_restarts} for Shot {shot_idx + 1}")

            # 重置状态
            self.attempted_time_ranges = set()
            self.duplicate_call_count = 0

            # 准备镜头消息（包含指导和禁止范围）
            msgs = self._prepare_shot_messages(
                shot=shot,
                audio_section_info=audio_section_info,
                related_scene_value=related_scene_value,
                guidance_text=guidance_text,
                forbidden_time_ranges=forbidden_time_ranges,
            )

            # 运行 ReAct 循环
            section_completed, should_restart = self._run_shot_loop(msgs, max_iterations=max_iterations)

            # 如果段落完成且有 commit 结果，返回结果
            if section_completed and self.last_commit_result:
                return self.last_commit_result
            
            # 如果不需要重启，跳出循环
            if not should_restart:
                break

        # 所有重试都失败，返回 None
        return None

    def cleanup(self):
        """
        Release large references to help GC after each subagent run.
        释放大型引用以帮助垃圾回收，在每个子智能体运行后调用。
        
        This is important for parallel processing to avoid memory leaks.
        这对于并行处理避免内存泄漏很重要。
        
        Java 类比：类似手动设置对象为 null 并调用 System.gc()
        """
        # 清空大型数据结构，释放内存
        self.messages = []
        self.audio_db = {}
        self.used_time_ranges = []
        self.current_related_scenes = []
        self.attempted_time_ranges = set()
        self.current_shot_context = {}
        self.last_commit_result = None
        self.last_commit_raw = None
        
        # 清理审核器
        self.reviewer = None
        
        # 清理视频读取器
        self.video_reader = None
        _clear_thread_video_reader()
        _clear_thread_video_reader()
        
        # gc.collect(): 手动触发垃圾回收
        # Java 类比：System.gc()（但不保证立即执行）
        gc.collect()


class ParallelShotOrchestrator:
    """
    Orchestrates parallel processing of video shots using multiple worker threads.
    编排器：使用多个工作线程并行处理视频镜头。
    
    This class manages:
    此类管理：
    - Thread pool for concurrent shot processing
    - 用于并发镜头处理的线程池
    - Conflict detection between parallel results
    - 并行结果之间的冲突检测
    - Quality-based winner selection
    - 基于质量的优胜者选择
    - Rerun mechanism for conflicting shots
    - 冲突镜头的重跑机制
    - Checkpoint saving for fault tolerance
    - 检查点保存以实现容错
    
    Java 类比：类似 ExecutorService + Future 模式的协调器，负责管理多个 Runnable 任务的结果合并和冲突解决。
    """
    def __init__(self, video_caption_path, video_scene_path, audio_caption_path, output_path, max_iterations, video_path=None, frame_folder_path=None, transcript_path: str = None, max_workers: int = None, max_reruns: int = None):
        """
        Initialize the parallel orchestrator.
        初始化并行编排器。
        
        Args:
            video_caption_path: 视频描述文件路径 (Video caption file path)
            video_scene_path: 视频场景划分文件路径 (Video scene segmentation file path)
            audio_caption_path: 音频描述文件路径 (Audio caption file path)
            output_path: 输出结果文件路径 (Output result file path)
            max_iterations: 每个镜头的最大迭代次数 (Max iterations per shot)
            video_path: 视频文件路径 (Video file path)
            frame_folder_path: 帧图像文件夹路径 (Frame images folder path)
            transcript_path: ASR 转录文本路径 (ASR transcript text path)
            max_workers: 最大工作线程数，默认从配置读取 (Max worker threads, default from config)
            max_reruns: 冲突镜头的最大重跑次数，默认从配置读取 (Max reruns for conflicting shots, default from config)
        """
        self.video_caption_path = video_caption_path
        self.video_scene_path = video_scene_path
        self.audio_caption_path = audio_caption_path
        self.output_path = output_path
        self.max_iterations = max_iterations
        self.video_path = video_path
        self.frame_folder_path = frame_folder_path
        self.transcript_path = transcript_path

        # getattr(config, 'PARALLEL_SHOT_MAX_WORKERS', 4): 从配置模块获取参数，默认为 4
        # Java 类比：config.PARALLEL_SHOT_MAX_WORKERS != null ? config.PARALLEL_SHOT_MAX_WORKERS : 4
        self.max_workers = max_workers or getattr(config, 'PARALLEL_SHOT_MAX_WORKERS', 4)
        self.max_reruns = max_reruns if max_reruns is not None else getattr(config, 'PARALLEL_SHOT_MAX_RERUNS', 2)
        
        # threading.Lock(): 线程锁，用于保护共享资源（输出文件）的并发写入
        # Java 类比：ReentrantLock 或 synchronized 块
        self._output_lock = threading.Lock()

    def _compute_quality_score(self, result_data: dict) -> float:
        if not result_data:
            return 0.0
        protagonist_ratio = 0.0
        if "protagonist_detection" in result_data:
            protagonist_ratio = result_data["protagonist_detection"].get("protagonist_ratio", 0.0)
        total_duration = result_data.get("total_duration", 0.0)
        target_duration = result_data.get("target_duration", 0.0)
        if target_duration <= 0:
            duration_score = 0.0
        else:
            duration_score = 1.0 - min(1.0, abs(total_duration - target_duration) / max(target_duration, 1.0))
        return 0.6 * protagonist_ratio + 0.4 * duration_score

    def _result_ranges(self, result_data: dict) -> list[tuple[float, float]]:
        """
        Extract time ranges from result data as (start_sec, end_sec) tuples.
        从结果数据中提取时间范围，返回 (起始秒数, 结束秒数) 元组列表。
        
        Args:
            result_data: 镜头结果数据 (Shot result data)
        Returns:
            时间范围列表，每个元素为 (start, end) 元组 (List of time ranges as (start, end) tuples)
        """
        ranges = []
        if not result_data:
            return ranges
        
        # 获取所有剪辑片段
        clips = result_data.get("clips", [])
        for clip in clips:
            start = clip.get("start")
            end = clip.get("end")
            if start and end:
                # 将 HH:MM:SS 格式转换为秒数
                start_sec = _hhmmss_to_seconds(start, fps=getattr(config, 'VIDEO_FPS', 24) or 24)
                end_sec = _hhmmss_to_seconds(end, fps=getattr(config, 'VIDEO_FPS', 24) or 24)
                ranges.append((start_sec, end_sec))
        return ranges

    def _detect_conflicts(self, results: dict, keep_ranges: list) -> dict:
        """
        Detect temporal conflicts among parallel shot results and identify losers.
        检测并行镜头结果之间的时间冲突，并识别失败者。
        
        This method performs two types of conflict detection:
        此方法执行两种类型的冲突检测：
        1. Check against already kept ranges (from previous sections or winners)
        1. 检查与已保留范围的冲突（来自之前的段落或优胜者）
        2. Pairwise conflicts within the current batch of results
        2. 当前批次结果内的成对冲突
        
        When conflicts are found, the lower quality result becomes a "loser" and will be rerun.
        发现冲突时，质量较低的结果成为“失败者”，将被重跑。
        
        Args:
            results: 字典，键为 (sec_idx, shot_idx)，值为结果数据 (Dict keyed by (sec_idx, shot_idx) with result data)
            keep_ranges: 已保留的时间范围列表 (List of already kept time ranges)
        Returns:
            失败者字典，键为 (sec_idx, shot_idx)，值为指导文本 (Losers dict keyed by (sec_idx, shot_idx) with guidance text)
        """
        losers = {}
        items = list(results.items())

        # 第一类冲突：与已保留范围的冲突（来自之前的段落或优胜者）
        # Conflicts with already kept ranges (from prior sections or winners)
        for key, result in items:
            ranges = self._result_ranges(result)
            for r_start, r_end in ranges:
                for k_start, k_end in keep_ranges:
                    if _ranges_overlap(r_start, r_end, k_start, k_end):
                        losers[key] = "Overlap with already selected clips. Please choose a different time range."
                        break
                if key in losers:
                    break

        # 第二类冲突：当前批次内的成对冲突
        # Pairwise conflicts in the current batch
        for i in range(len(items)):
            key_i, res_i = items[i]
            if key_i in losers:
                continue
            ranges_i = self._result_ranges(res_i)
            for j in range(i + 1, len(items)):
                key_j, res_j = items[j]
                if key_j in losers:
                    continue
                ranges_j = self._result_ranges(res_j)
                overlap = False
                # 检查两个结果的所有时间范围是否有重叠
                for a_start, a_end in ranges_i:
                    for b_start, b_end in ranges_j:
                        if _ranges_overlap(a_start, a_end, b_start, b_end):
                            overlap = True
                            break
                    if overlap:
                        break
                if overlap:
                    # 计算两个结果的质量分数
                    score_i = self._compute_quality_score(res_i)
                    score_j = self._compute_quality_score(res_j)
                    # 质量较低的成为失败者，需要重跑
                    if score_i >= score_j:
                        losers[key_j] = f"Overlap with shot {key_i[1] + 1}. Please choose a different time range."
                    else:
                        losers[key_i] = f"Overlap with shot {key_j[1] + 1}. Please choose a different time range."
                        break

        return losers

    def _run_worker(self, shot, sec_idx, shot_idx, guidance_text=None, forbidden_time_ranges=None):
        """
        Worker function executed in a thread to process a single shot.
        在工作线程中执行的工作函数，用于处理单个镜头。
        
        This function:
        此函数：
        1. Creates a new EditorCoreAgent instance for this shot
        1. 为此镜头创建新的 EditorCoreAgent 实例
        2. Runs the shot processing (initial or rerun with guidance)
        2. 运行镜头处理（初始或带指导的重跑）
        3. Saves result to output file if successful
        3. 如果成功，将结果保存到输出文件
        4. Cleans up resources to prevent memory leaks
        4. 清理资源以防止内存泄漏
        
        Args:
            shot: 镜头计划数据 (Shot plan data)
            sec_idx: 段落索引 (Section index)
            shot_idx: 镜头索引 (Shot index)
            guidance_text: 重跑时的指导文本，None 表示初始运行 (Guidance text for rerun, None for initial run)
            forbidden_time_ranges: 禁止使用的时间范围列表 (List of forbidden time ranges)
        Returns:
            镜头结果数据，失败时返回 None (Shot result data, None on failure)
        """
        # 判断是重跑还是初始运行
        mode = "rerun" if guidance_text else "initial"
        print(f"[SubAgent S{sec_idx + 1} Shot{shot_idx + 1}] start ({mode})")
        
        # 创建新的智能体实例（每个线程独立）
        agent = EditorCoreAgent(
            self.video_caption_path,
            self.video_scene_path,
            self.audio_caption_path,
            output_path="",
            max_iterations=self.max_iterations,
            video_path=self.video_path,
            frame_folder_path=self.frame_folder_path,
            transcript_path=self.transcript_path
        )
        try:
            # 运行单个镜头处理
            result = agent.run_single_shot(
                shot=shot,
                sec_idx=sec_idx,
                shot_idx=shot_idx,
                guidance_text=guidance_text,
                forbidden_time_ranges=forbidden_time_ranges
            )
            if result:
                print(f"[SubAgent S{sec_idx + 1} Shot{shot_idx + 1}] success")
                # 将结果追加到输出文件
                self._append_result_to_output((sec_idx, shot_idx), result)
            else:
                print(f"[SubAgent S{sec_idx + 1} Shot{shot_idx + 1}] no-result")
            return result
        finally:
            # 清理智能体资源
            agent.cleanup()
            # 清除当前线程的线程局部视频读取器
            # Explicitly clear thread-local video reader in this worker thread
            _clear_thread_video_reader()
            # 手动触发垃圾回收
            gc.collect()

    def _merge_results(self, existing_list: list, new_results: dict) -> list:
        """
        Merge existing results with new results, avoiding duplicates.
        合并现有结果和新结果，避免重复。
        
        Args:
            existing_list: 现有结果列表 (List of existing results)
            new_results: 新结果字典，键为 (sec_idx, shot_idx) (New results dict keyed by (sec_idx, shot_idx))
        Returns:
            合并后的结果列表，按段落和镜头索引排序 (Merged result list sorted by section and shot index)
        """
        # 使用字典去重：相同 (section_idx, shot_idx) 的结果会被覆盖
        result_map = {}
        for item in existing_list or []:
            if item.get("status") == "success":
                key = (item.get("section_idx"), item.get("shot_idx"))
                result_map[key] = item

        # 添加新结果（会覆盖同键的旧结果）
        for key, result in new_results.items():
            if result:
                result_map[key] = result

        # 转换为列表并按 (section_idx, shot_idx) 排序
        merged = list(result_map.values())
        merged.sort(key=lambda x: (x.get("section_idx", 0), x.get("shot_idx", 0)))
        return merged

    def _save_checkpoint(self, existing_list: list, new_results: dict):
        """
        Save merged results to output file as a checkpoint.
        将合并后的结果保存到输出文件作为检查点。
        
        This provides fault tolerance - if processing is interrupted, it can resume from this checkpoint.
        这提供了容错能力 - 如果处理中断，可以从此检查点恢复。
        
        Args:
            existing_list: 现有结果列表 (List of existing results)
            new_results: 新结果字典 (New results dict)
        """
        if not self.output_path:
            return
        # 合并结果
        merged = self._merge_results(existing_list, new_results)
        # 写入 JSON 文件
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"[Parallel] checkpoint saved: {len(merged)} shots")

    def _append_result_to_output(self, key: tuple, result: dict):
        """
        Thread-safe method to append a single result to the output file.
        线程安全的方法，将单个结果追加到输出文件。
        
        This uses a lock to prevent concurrent writes from multiple worker threads.
        使用锁来防止多个工作线程的并发写入。
        
        Args:
            key: (sec_idx, shot_idx) 元组 ((sec_idx, shot_idx) tuple)
            result: 镜头结果数据 (Shot result data)
        """
        if not self.output_path or not result:
            return
        # 使用线程锁保护共享资源（输出文件）
        # Java 类比：synchronized 块或 ReentrantLock.lock()
        with self._output_lock:
            existing = []
            if os.path.exists(self.output_path):
                try:
                    with open(self.output_path, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                except Exception:
                    existing = []
            # 合并现有结果和新结果
            merged = self._merge_results(existing, {key: result})
            # 写回文件
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            print(f"[Parallel] subagent saved: section={key[0]} shot={key[1]}")

    def run_parallel(self, shot_plan_path: str):
        """
        Main method to orchestrate parallel processing of all shots.
        主方法：编排所有镜头的并行处理。
        
        Parallel strategy:
        并行策略：
        1) Run each shot as a sub-agent worker in thread pool
        1）在线程池中将每个镜头作为子智能体工作器运行
        2) Detect temporal conflicts among worker outputs
        2）检测工作器输出之间的时间冲突
        3) Keep better winners based on quality score, rerun losers with guidance
        3）基于质量分数保留更优的优胜者，带指导重跑失败者
        4) Continuously checkpoint merged results for fault tolerance
        4）持续保存合并结果作为检查点以实现容错
        
        Args:
            shot_plan_path: 镜头计划文件路径 (Shot plan file path)
        Returns:
            合并后的所有镜头结果列表 (Merged list of all shot results)
        """
        # Parallel strategy:
        # 并行策略：
        # 1) run each shot as a sub-agent worker
        # 1）每个镜头由一个子 worker 并行处理
        # 2) detect temporal conflicts among worker outputs
        # 2）检测各 worker 输出之间的时间冲突
        # 3) keep better winners, rerun losers with guidance
        # 3）保留更优结果，并带指导重跑冲突项
        # 4) checkpoint merged results continuously
        # 4）持续保存合并后的检查点结果
        
        # 加载镜头计划文件
        with open(shot_plan_path, 'r', encoding='utf-8') as f:
            structure_proposal = json.load(f)

        # global_keep_ranges: 全局已保留的时间范围（跨段落）
        global_keep_ranges = []
        # final_results: 最终结果字典，键为 (sec_idx, shot_idx)
        final_results = {}
        existing = []
        # completed_shots: 已完成的镜头集合，用于断点续传
        completed_shots = set()

        # 如果输出文件存在，加载已有结果以支持断点续传
        if self.output_path and os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        # 从现有结果中提取已完成的镜头和时间范围
        for item in existing:
            if item.get("status") != "success":
                continue
            sec_idx = item.get("section_idx")
            shot_idx = item.get("shot_idx")
            if sec_idx is None or shot_idx is None:
                continue
            completed_shots.add((sec_idx, shot_idx))
            for r_start, r_end in self._result_ranges(item):
                global_keep_ranges.append((r_start, r_end))

        if completed_shots:
            print(f"📋 [Parallel] Found {len(completed_shots)} completed shots in existing output file")
            print(f"   Completed: {sorted(completed_shots)}")

        # 遍历视频结构中的每个段落
        for sec_idx, sec_cur in enumerate(structure_proposal['video_structure']):
            shot_plan = sec_cur.get('shot_plan')
            if not shot_plan:
                print(f"❌ [Error] No shot_plan found for section {sec_idx}")
                continue
            shots = shot_plan['shots']
            print(f"\n[Parallel] Processing Section {sec_idx + 1}/{len(structure_proposal['video_structure'])}")

            # pending: 当前段落中待处理的镜头字典
            pending = {}
            skipped_in_section = 0
            for idx, shot in enumerate(shots):
                key = (sec_idx, idx)
                # 跳过已完成的镜头（断点续传）
                if key in completed_shots:
                    skipped_in_section += 1
                    continue
                pending[key] = shot

            if skipped_in_section:
                print(f"[Parallel][Section {sec_idx + 1}] skipped {skipped_in_section} completed shots")
            if not pending:
                print(f"[Parallel][Section {sec_idx + 1}] all shots already completed")
                continue

            # pending_guidance: 每个待处理镜头的指导文本（重跑时使用）
            pending_guidance = {key: None for key in pending}
            # section_keep_ranges: 当前段落内已保留的时间范围
            section_keep_ranges = []
            rerun_count = 0
            round_idx = 0

            # 重跑循环：直到没有待处理的镜头或达到最大重跑次数
            while pending:
                round_idx += 1
                results = {}
                # combined_keep_ranges: 合并全局和当前段落的已保留范围
                combined_keep_ranges = global_keep_ranges + section_keep_ranges
                print(
                    f"[Parallel][Section {sec_idx + 1}][Round {round_idx}] "
                    f"pending={len(pending)} rerun_count={rerun_count}/{self.max_reruns}"
                )

                # 使用线程池并行执行所有待处理镜头
                # Java 类比：ExecutorService executor = Executors.newFixedThreadPool(max_workers);
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {}
                    # 提交所有任务到线程池
                    for (s_idx, shot_idx), shot in pending.items():
                        key = (s_idx, shot_idx)
                        # executor.submit(): 提交任务并返回 Future 对象
                        # Java 类比：Future<Result> future = executor.submit(task);
                        futures[executor.submit(
                            self._run_worker,
                            shot,
                            s_idx,
                            shot_idx,
                            guidance_text=pending_guidance.get(key),
                            forbidden_time_ranges=combined_keep_ranges
                        )] = key

                    # 等待所有任务完成并收集结果
                    # as_completed(): 按完成顺序迭代 Future 对象
                    # Java 类比：for (Future<Result> future : futures) { Result r = future.get(); }
                    for future in as_completed(futures):
                        key = futures[future]
                        try:
                            results[key] = future.result()
                        except Exception as e:
                            print(f"Worker failed for shot {key}: {e}")
                            results[key] = None

                # 检测冲突，识别失败者
                losers = self._detect_conflicts(results, combined_keep_ranges)
                print(
                    f"[Parallel][Section {sec_idx + 1}][Round {round_idx}] "
                    f"conflicts={len(losers)} winners={len(results) - len(losers)}"
                )

                # Keep winners
                # 保留优胜者（非失败者的结果）
                round_has_updates = False
                for key, result in results.items():
                    if key in losers:
                        continue
                    if result:
                        final_results[key] = result
                        round_has_updates = True
                        # 将优胜者的时间范围添加到当前段落的已保留列表
                        for r_start, r_end in self._result_ranges(result):
                            section_keep_ranges.append((r_start, r_end))

                # 如果本轮有更新，保存检查点
                if round_has_updates:
                    self._save_checkpoint(existing, final_results)

                # 如果没有冲突，退出重跑循环
                if not losers:
                    print(f"[Parallel][Section {sec_idx + 1}] no conflicts remaining")
                    break

                # 如果达到最大重跑次数，停止重跑
                if rerun_count >= self.max_reruns:
                    print(
                        f"[Parallel][Section {sec_idx + 1}] reached max reruns "
                        f"({self.max_reruns}), stop rerunning unresolved shots"
                    )
                    break

                # 准备下一轮：只重跑失败者
                pending = {key: shots[key[1]] for key in losers}
                # 将冲突信息作为指导文本传递给下一轮
                pending_guidance = losers
                rerun_count += 1

            # 将当前段落的已保留范围合并到全局列表
            global_keep_ranges.extend(section_keep_ranges)

        # 保存最终合并结果
        # Save merged results
        merged = self._merge_results(existing, final_results)
        if self.output_path:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
        return merged
