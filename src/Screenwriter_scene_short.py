"""
Screenwriter Agent - Scene and Shot Planning for Short Video Editing.
编剧智能体 - 短视频编辑的场景和镜头规划。

This module implements the Screenwriter agent responsible for:
此模块实现负责以下任务的编剧智能体：
1. Selecting appropriate audio segments based on user instructions
1. 根据用户指令选择合适的音频片段
2. Generating video structure proposals (scene selection and segmentation)
2. 生成视频结构提案（场景选择和分段）
3. Creating detailed shot plans for each video section
3. 为每个视频段落创建详细的镜头计划
4. Validating scene distribution to ensure narrative balance
4. 验证场景分布以确保叙事平衡

Java 类比：类似一个智能的内容策划器，结合 LLM 进行创意决策和结构化输出。
"""

import os
import json
import re
import time
import argparse
import random
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated as A
from src import config
from src.func_call_shema import doc as D
from src.prompt import GENERATE_STRUCTURE_PROPOSAL_PROMPT, GENERATE_SHOT_PLAN_PROMPT, SELECT_AUDIO_SEGMENT_PROMPT, SELECT_HOOK_DIALOGUE_PROMPT
from src.utils.media_utils import (
    hhmmss_to_seconds,
    load_scene_summaries,
    parse_srt_file,
    parse_structure_proposal_output,
    parse_shot_plan_output,
)
import litellm



# 钩子对话的最大字幕字符数限制
HOOK_DIALOGUE_MAX_SUBTITLE_CHARS = 20000


class HookDialogueSelectionError(RuntimeError):
    """
    Raised when hook dialogue selection should fail the pipeline.
    当钩子对话选择失败时抛出此异常，用于中断处理流程。
    
    Java 类比：自定义异常类，类似 extends RuntimeException
    """


def _has_meaningful_value(value) -> bool:
    """
    Check whether a JSON field is present with usable content.
    检查 JSON 字段是否存在且具有可用内容。
    
    Args:
        value: 要检查的值 (Value to check)
    Returns:
        True 如果值有意义，False 否则 (True if value is meaningful, False otherwise)
    """
    if value is None:
        return False
    # 字符串：检查是否非空（去除空白后）
    if isinstance(value, str):
        return bool(value.strip())
    # 集合类型：检查是否非空
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    # 其他类型（数字、布尔等）：认为有意义
    return True


def get_missing_shot_plan_parts(output_data: dict) -> list[str]:
    """
    Return operationally important shot-plan parts that are missing.
    返回缺失的关键镜头计划部分。
    
    This function validates the structure of the shot plan output and identifies
    which required fields are missing or empty. Used for retry feedback.
    此函数验证镜头计划输出的结构，并识别哪些必需字段缺失或为空。用于重跑反馈。
    
    Args:
        output_data: LLM 输出的镜头计划数据 (Shot plan output data from LLM)
    Returns:
        缺失字段的列表，如 ["metadata", "video_structure[0].shot_plan.shots"] (List of missing field paths)
    """
    if not isinstance(output_data, dict):
        return ["root"]

    missing_parts: list[str] = []

    # 检查顶层必需字段
    for key in ("instruction", "overall_theme", "narrative_logic"):
        if not _has_meaningful_value(output_data.get(key)):
            missing_parts.append(key)

    # 检查 metadata 及其子字段
    metadata = output_data.get("metadata")
    if not isinstance(metadata, dict):
        missing_parts.append("metadata")
    else:
        for key in ("selected_audio_start", "selected_audio_end"):
            if not _has_meaningful_value(metadata.get(key)):
                missing_parts.append(f"metadata.{key}")

    # 检查 video_structure 数组
    video_structure = output_data.get("video_structure")
    if not isinstance(video_structure, list) or not video_structure:
        missing_parts.append("video_structure")
        return missing_parts

    # 检查第一个段落的必需字段
    first_section = video_structure[0]
    if not isinstance(first_section, dict):
        missing_parts.append("video_structure[0]")
        return missing_parts

    for key in ("overall_theme", "narrative_logic", "start_time", "end_time"):
        if not _has_meaningful_value(first_section.get(key)):
            missing_parts.append(f"video_structure[0].{key}")

    # 检查镜头计划
    section_shot_plan = first_section.get("shot_plan")
    if not isinstance(section_shot_plan, dict):
        missing_parts.append("video_structure[0].shot_plan")
    elif not isinstance(section_shot_plan.get("shots"), list) or not section_shot_plan.get("shots"):
        missing_parts.append("video_structure[0].shot_plan.shots")

    return missing_parts


def _call_agent_litellm(messages: list, max_tokens: int = None) -> str | None:
    """
    Call the agent LLM via litellm. Returns content string or None on failure.
    通过 litellm 调用智能体 LLM。返回内容字符串，失败时返回 None。
    
    This is a wrapper around litellm.completion() with error handling and
    content extraction logic for different response formats.
    这是 litellm.completion() 的包装器，包含错误处理和针对不同响应格式的内容提取逻辑。
    
    Args:
        messages: 消息列表，格式为 [{"role": "user", "content": "..."}] (List of messages)
        max_tokens: 最大生成 token 数，默认使用配置值 (Max tokens to generate)
    Returns:
        LLM 响应的文本内容，失败时返回 None (Text content from LLM, None on failure)
    """
    kwargs = dict(
        model=config.AGENT_LITELLM_MODEL,
        messages=messages,
        max_tokens=max_tokens or config.AGENT_MODEL_MAX_TOKEN,
        api_key=config.AGENT_LITELLM_API_KEY,
        timeout=60,
    )
    if config.AGENT_LITELLM_URL:
        kwargs["api_base"] = config.AGENT_LITELLM_URL
    try:
        resp = litellm.completion(**kwargs)
        content = resp.choices[0].message.content
        if content is None:
            return None
        # 处理字符串类型的内容
        if isinstance(content, str):
            content = content.strip()
            return content or None
        # Some providers may return structured content blocks.
        # 某些提供商可能返回结构化的内容块（如多模态模型）
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")).strip())
                elif isinstance(item, str):
                    text_parts.append(item.strip())
            merged = "\n".join([p for p in text_parts if p]).strip()
            return merged or None
        return str(content).strip() or None
    except Exception as e:
        # 捕获所有异常并返回 None，调用方需要处理重试逻辑
        return None


def _to_audio_seconds(value) -> float:
    """
    Normalize section timestamps into seconds.
    将时间段的时间戳标准化为秒数。
    
    Handles both numeric values and HH:MM:SS string formats.
    处理数字值和 HH:MM:SS 字符串格式。
    
    Args:
        value: 时间值，可以是数字或字符串 (Time value, can be number or string)
    Returns:
        转换为秒数的浮点数 (Time in seconds as float)
    """
    if isinstance(value, (int, float)):
        return float(value)
    return hhmmss_to_seconds(str(value))


def _seconds_to_mmss(seconds: float) -> str:
    """
    Convert seconds to MM:SS.f format (e.g. 90.5 → '1:30.5').
    将秒数转换为 MM:SS.f 格式（例如 90.5 → '1:30.5'）。
    
    Args:
        seconds: 秒数 (Seconds)
    Returns:
        格式化的时间字符串，如 "1:30.5" (Formatted time string like "1:30.5")
    """
    seconds = max(0.0, seconds)
    # 转换为十分之一秒的整数，用于保留一位小数
    total_tenths = int(round(seconds * 10))
    tenths = total_tenths % 10
    total_secs = total_tenths // 10
    mm = total_secs // 60
    ss = total_secs % 60
    return f"{mm}:{ss:02d}.{tenths}"


def _parse_audio_segment_selection_response(content: str) -> dict | None:
    """
    Parse JSON response for audio section selection.
    解析音频段落选择的 JSON 响应。
    
    Handles responses with or without markdown code block wrappers.
    处理带或不带 Markdown 代码块包装的响应。
    
    Args:
        content: LLM 返回的原始文本内容 (Raw text content from LLM)
    Returns:
        解析后的字典，失败时返回 None (Parsed dict, None on failure)
    """
    if not content:
        return None
    clean = content.strip()
    # 移除 Markdown 代码块标记（如 ```json ... ```）
    if clean.startswith("```"):
        clean = re.sub(r"^```[a-z]*\n?", "", clean)
        clean = re.sub(r"\n?```$", "", clean)
    try:
        parsed = json.loads(clean)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def select_audio_segment(audio_db: dict, instruction: str) -> tuple[str, str]:
    """
    Use LLM to select the best music section, then trim to target duration if needed.
    使用 LLM 选择最佳音乐段落，如果需要则裁剪到目标时长。
    
    This function:
    此函数：
    1. Analyzes audio sections and their characteristics
    1. 分析音频段落及其特征
    2. Calls LLM to select the most appropriate section based on user instruction
    2. 调用 LLM 根据用户指令选择最合适的段落
    3. Validates the selection and retries with feedback if needed
    3. 验证选择，如果需要则带反馈重跑
    4. Trims the selected section to fit within duration constraints
    4. 裁剪选定的段落以适应时长限制
    5. Falls back to heuristic selection if LLM fails after max retries
    5. 如果 LLM 在最大重试次数后仍失败，则回退到启发式选择
    
    Args:
        audio_db: 音频数据库字典，包含 'sections' 和 'overall_analysis' (Audio database dict)
        instruction: 用户的编辑指令 (User's editing instruction)
    Returns:
        (start_time, end_time) 元组，表示选定的音频时间段 ((start_time, end_time) tuple)
    """
    # 获取音频段落列表和总体分析摘要
    sections = audio_db.get('sections', [])
    summary = audio_db.get('overall_analysis', {}).get('summary', '')
    min_dur = config.AUDIO_SEGMENT_MIN_DURATION_SEC
    max_dur = config.AUDIO_SEGMENT_MAX_DURATION_SEC
    # 目标时长：最小和最大时长的平均值
    target_dur = (min_dur + max_dur) / 2

    # 如果没有段落，返回默认值
    if not sections:
        return '0:00', _seconds_to_mmss(target_dur)

    # 构建供 LLM 使用的段落信息列表
    # Build sections_info for LLM
    sections_info = []
    for i, sec in enumerate(sections):
        sec_start = _to_audio_seconds(sec.get('Start_Time', 0))
        sec_end = _to_audio_seconds(sec.get('End_Time', 0))
        dur = round(max(0.0, sec_end - sec_start), 1)
        sections_info.append({
            "section_index": i,
            "name": sec.get('name', ''),
            "description": sec.get('description', ''),
            "Start_Time": sec.get('Start_Time', ''),
            "End_Time": sec.get('End_Time', ''),
            "duration_seconds": dur,
            # 标记时长是否合格
            "duration_ok": "✓" if dur >= min_dur else "✗ too short",
        })

    def _apply_section(idx: int) -> tuple[str, str]:
        """
        Apply the selected section index, trimming if necessary.
        应用选定的段落索引，必要时进行裁剪。
        
        If the section duration is within [min_dur, max_dur], use it as-is.
        如果段落时长在 [min_dur, max_dur] 范围内，则原样使用。
        Otherwise, trim from start to target_dur (but never exceed section end).
        否则，从开始裁剪到 target_dur（但不超过段落结束）。
        """
        sec = sections[idx]
        sec_start = _to_audio_seconds(sec.get('Start_Time', 0))
        sec_end = _to_audio_seconds(sec.get('End_Time', 0))
        sec_dur = max(0.0, sec_end - sec_start)
        # 如果时长在允许范围内，直接返回原始时间
        if min_dur <= sec_dur <= max_dur:
            return str(sec.get('Start_Time', _seconds_to_mmss(sec_start))), str(sec.get('End_Time', _seconds_to_mmss(sec_end)))
        # Trim from section start to target_dur, but never exceed section end
        # 从段落开始裁剪到目标时长，但不超过段落结束
        trim_end = min(sec_start + target_dur, sec_end)
        return _seconds_to_mmss(sec_start), _seconds_to_mmss(trim_end)

    # 重试循环：最多尝试 AUDIO_SEGMENT_SELECTION_MAX_RETRIES 次
    feedback = None
    for attempt in range(1, config.AUDIO_SEGMENT_SELECTION_MAX_RETRIES + 1):
        # 构建提示词，包含之前的反馈（如果有）
        prompt = SELECT_AUDIO_SEGMENT_PROMPT.format(
            summary=summary,
            sections_json=json.dumps(sections_info, indent=2, ensure_ascii=False),
            instruction=instruction,
            min_duration_sec=min_dur,
            max_duration_sec=max_dur,
            feedback_block=(
                f"\nValidation feedback from previous attempt: {feedback}\n"
                if feedback else ""
            ),
        )
        # 调用 LLM 进行选择
        content = _call_agent_litellm([{"role": "user", "content": prompt}], max_tokens=512)
        if not content:
            feedback = "No response returned. Return valid JSON with section_index."
            continue

        # 解析 LLM 的 JSON 响应
        result = _parse_audio_segment_selection_response(content)
        if not isinstance(result, dict):
            feedback = "Response is not a JSON object. Return {\"section_index\": N, \"reason\": \"...\"}"
            continue

        # 验证 section_index 的有效性
        raw_idx = result.get('section_index')
        if not isinstance(raw_idx, int) or raw_idx < 0 or raw_idx >= len(sections):
            feedback = (
                f"Invalid section_index: {raw_idx!r}. "
                f"Must be an integer between 0 and {len(sections) - 1}."
            )
            continue

        # 验证通过，应用选定的段落
        return _apply_section(raw_idx)

    # Fallback: pick section with duration closest to target_dur
    # 回退策略：选择时长最接近目标时长的段落
    print(f"[Audio Selection] LLM failed after {config.AUDIO_SEGMENT_SELECTION_MAX_RETRIES} attempts, using fallback")
    best_idx = 0
    best_diff = float('inf')
    for i, sec in enumerate(sections):
        sec_start = _to_audio_seconds(sec.get('Start_Time', 0))
        sec_end = _to_audio_seconds(sec.get('End_Time', 0))
        diff = abs((sec_end - sec_start) - target_dur)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return _apply_section(best_idx)


def filter_sub_segments_by_range(
    sections: list, start_time_str: str, end_time_str: str
) -> list:
    """
    Collect all sub-segments whose time range overlaps with [start_time_str, end_time_str].
    收集所有与给定时间范围 [start_time_str, end_time_str] 重叠的子段落。
    
    Returns a flat list of sub-segment dicts with absolute timestamps.
    返回包含绝对时间戳的子段落字典的扁平列表。
    
    This function:
    此函数：
    1. Converts time strings to seconds for comparison
    1. 将时间字符串转换为秒数进行比较
    2. Iterates through sections and their sub-segments
    2. 遍历段落及其子段落
    3. Filters sub-segments that overlap with the target range
    3. 过滤与目标范围重叠的子段落
    4. Fills gaps between consecutive sub-segments
    4. 填充连续子段落之间的间隙
    
    Args:
        sections: 段落列表，每个段落包含 'detailed_analysis' (List of sections with detailed_analysis)
        start_time_str: 起始时间，可以是 HH:MM:SS 或秒数 (Start time string or seconds)
        end_time_str: 结束时间，可以是 HH:MM:SS 或秒数 (End time string or seconds)
    Returns:
        重叠的子段落列表，每个元素包含绝对时间戳 (List of overlapping sub-segments with absolute timestamps)
    """
    def _to_sec(t):
        """
        Convert various time formats to seconds.
        将各种时间格式转换为秒数。
        
        Supports: numeric values, HH:MM:SS, MM:SS, or plain seconds.
        支持：数字值、HH:MM:SS、MM:SS 或纯秒数。
        """
        if isinstance(t, (int, float)):
            return float(t)
        parts = str(t).split(':')
        if len(parts) == 3:
            # HH:MM:SS 格式
            h, m, s = [float(x) for x in parts]
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            # MM:SS 格式
            m, s = [float(x) for x in parts]
            return m * 60 + s
        else:
            # 纯秒数
            try:
                return float(parts[0])
            except ValueError:
                return 0.0

    # 转换目标范围为秒数
    range_start = _to_sec(start_time_str)
    range_end = _to_sec(end_time_str)

    result = []
    # 遍历所有段落
    for section in sections:
        # 获取段落的起始时间（作为基准）
        section_start = _to_sec(section.get('Start_Time', section.get('start_time', 0)))
        # 遍历段落的详细分析中的子段落
        for sub in section.get('detailed_analysis', {}).get('sections', []):
            # 计算子段落的绝对时间（相对于视频开头）
            sub_start = section_start + _to_sec(sub.get('Start_Time', sub.get('start_time', 0)))
            sub_end = section_start + _to_sec(sub.get('End_Time', sub.get('end_time', 0)))
            # 检查是否与目标范围重叠
            # Java 类比：if (subEnd > rangeStart && subStart < rangeEnd)
            if sub_end > range_start and sub_start < range_end:
                # 创建子段落的副本并添加绝对时间戳
                sub_abs = dict(sub)
                sub_abs['Start_Time'] = sub_start
                sub_abs['End_Time'] = sub_end
                result.append(sub_abs)

    # Fill gaps: extend each section's End_Time to the next section's Start_Time
    # 填充间隙：将每个子段落的结束时间扩展到下一个子段落的开始时间
    for i in range(len(result) - 1):
        gap = result[i + 1]['Start_Time'] - result[i]['End_Time']
        if gap > 0:
            result[i]['End_Time'] = result[i + 1]['Start_Time']

    return result


def check_scene_distribution(
    structure_proposal: dict,
    total_scene_count: int,
) -> tuple[bool, str]:
    """
    Validate basic structure of the scene proposal (flat format).
    验证场景提案的基本结构（扁平格式）。
    
    This function checks:
    此函数检查：
    1. Proposal format validity
    1. 提案格式的有效性
    2. Minimum number of scenes (at least 8 or total count, whichever is smaller)
    2. 最少场景数量（至少 8 个或总场景数，取较小值）
    3. Scene index validity (non-negative, within bounds)
    3. 场景索引的有效性（非负、在范围内）
    4. Distribution across video timeline (early/middle/late thirds)
    4. 在视频时间轴上的分布（前/中/后三段）
    
    Args:
        structure_proposal: 结构提案字典，包含 'related_scenes' 列表 (Structure proposal dict with 'related_scenes')
        total_scene_count: 视频中的总场景数 (Total number of scenes in the video)
    Returns:
        (passed, feedback_message) 元组，passed 为布尔值，feedback_message 为反馈信息 ((passed, feedback_message) tuple)
    """
    # 验证提案格式
    if not structure_proposal or not isinstance(structure_proposal, dict):
        return False, "Invalid structure proposal format."

    # 获取相关场景列表
    related_scenes = structure_proposal.get('related_scenes', [])
    if not related_scenes:
        return False, "No related_scenes found in proposal."

    # 检查最少场景数量
    min_scenes = min(8, total_scene_count)
    if len(related_scenes) < min_scenes:
        return False, (
            f"Too few scenes selected: {len(related_scenes)}. "
            f"Need at least {min_scenes} scenes (out of {total_scene_count} available). "
            f"Please select more diverse scenes."
        )

    # 最多 15 个场景是允许的，不警告
    if len(related_scenes) > 15:
        pass  # allow but don't warn

    # 验证每个场景索引的有效性
    for scene_id in related_scenes:
        if not isinstance(scene_id, int):
            return False, f"Invalid scene index (not an integer): {scene_id}"
        if scene_id < 0:
            return False, f"Invalid scene index (negative): {scene_id}"
        if scene_id >= total_scene_count:
            return False, f"Scene index {scene_id} exceeds total scene count ({total_scene_count})"

    # Distribution check: all three thirds must have at least one scene
    # 分布检查：视频的三段（前/中/后）都必须至少有一个场景
    third = max(1, total_scene_count // 3)
    # 使用列表推导式过滤各段的场景
    # Java 类比：relatedScenes.stream().filter(s -> s < third).collect(Collectors.toList())
    early  = [s for s in related_scenes if s < third]
    middle = [s for s in related_scenes if third <= s < 2 * third]
    late   = [s for s in related_scenes if s >= 2 * third]
    missing = []
    if not early:
        missing.append(f"early section (scenes 0–{third - 1})")
    if not middle:
        missing.append(f"middle section (scenes {third}–{2 * third - 1})")
    if not late:
        missing.append(f"late section (scenes {2 * third}–{total_scene_count - 1})")
    if missing:
        return False, (
            f"Scene distribution is too concentrated. Missing coverage in: {', '.join(missing)}. "
            f"Current selection: early={len(early)}, middle={len(middle)}, late={len(late)}. "
            f"Please add scenes from the missing section(s)."
        )

    print(
        f"[Scene Check] {len(related_scenes)} scenes selected "
        f"(early={len(early)}, middle={len(middle)}, late={len(late)}). "
        f"Indices: {related_scenes}"
    )
    return True, f"Scene selection looks good - {len(related_scenes)} scenes selected."


def generate_structure_proposal(
    video_scene_path: A[str, D("Path to scene_summaries_video folder containing scene JSON files.")],
    audio_caption_path: A[str, D("Path to captions.json describing the audio segments.")],
    user_instruction: A[str, D("Editing brief provided by the user.")],
    selected_start_str: A[str | None, D("Start time of the selected audio segment.")] = None,
    selected_end_str: A[str | None, D("End time of the selected audio segment.")] = None,
    feedback: A[str | None, D("Validation feedback from previous attempt, injected to guide retry.")] = None,
    main_character: A[str | None, D("Name of the main character to focus on.")] = None,
) -> str | None:
    """
    Generate a structure proposal for the video editing based on scene summaries.
    基于场景摘要生成视频编辑的结构提案。
    
    This function:
    此函数：
    1. Loads video scene summaries and audio captions
    1. 加载视频场景摘要和音频字幕
    2. Filters audio sections if a time range is specified
    2. 如果指定了时间范围，则过滤音频段落
    3. Constructs a prompt with all necessary context
    3. 构建包含所有必要上下文的提示词
    4. Calls LLM to generate the structure proposal
    4. 调用 LLM 生成结构提案
    
    The structure proposal includes:
    结构提案包括：
    - Related scenes selection (which scenes to use)
    - 相关场景选择（使用哪些场景）
    - Video structure segmentation (how to divide the video)
    - 视频结构分段（如何划分视频）
    
    Args:
        video_scene_path: 视频场景摘要文件夹路径 (Path to video scene summaries)
        audio_caption_path: 音频字幕文件路径或数据 (Path to audio captions file or data)
        user_instruction: 用户的编辑指令 (User's editing instruction)
        selected_start_str: 选定音频段落的起始时间（可选） (Selected audio start time, optional)
        selected_end_str: 选定音频段落的结束时间（可选） (Selected audio end time, optional)
        feedback: 之前尝试的验证反馈，用于指导重跑 (Validation feedback from previous attempt)
        main_character: 要聚焦的主要角色名称（可选） (Main character name to focus on, optional)
    Returns:
        LLM 生成的结构提案文本，失败时返回 None (Generated structure proposal text, None on failure)
    """
    # 加载视频场景摘要和场景数量
    video_summary, scene_count = load_scene_summaries(video_scene_path)
    max_scene_index = scene_count - 1 if scene_count > 0 else 0

    # 加载音频字幕数据
    if isinstance(audio_caption_path, str):
        with open(audio_caption_path, 'r', encoding='utf-8') as f:
            audio_caption_data = json.load(f)
    else:
        audio_caption_data = audio_caption_path

    # 提取音频摘要和段落列表
    audio_summary = audio_caption_data.get('overall_analysis', {}).get('summary', '')
    sections = audio_caption_data.get('sections', [])

    # If a selected audio range is provided, filter sections to only those overlapping it
    # 如果提供了选定的音频范围，只保留与该范围重叠的段落
    if selected_start_str and selected_end_str:
        def _to_sec(t):
            """Convert time string to seconds."""
            if isinstance(t, (int, float)):
                return float(t)
            parts = str(t).split(':')
            if len(parts) == 3:
                h, m, s = [float(x) for x in parts]
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m, s = [float(x) for x in parts]
                return m * 60 + s
            else:
                try:
                    return float(parts[0])
                except ValueError:
                    return 0.0

        range_start = _to_sec(selected_start_str)
        range_end = _to_sec(selected_end_str)
        # 过滤出与选定范围重叠的段落
        sections = [
            s for s in sections
            if _to_sec(s.get('End_Time', 0)) > range_start and _to_sec(s.get('Start_Time', 0)) < range_end
        ]

    # 构建供 LLM 使用的音频段落信息列表（只保留关键字段）
    filtered_sections = [
        {
            'name': s.get('name', ''),
            'description': s.get('description', ''),
            'Start_Time': s.get('Start_Time', ''),
            'End_Time': s.get('End_Time', ''),
        }
        for s in sections
    ]
    # 将音频结构转换为 JSON 字符串
    audio_structure = json.dumps(filtered_sections, indent=2, ensure_ascii=False)

    # 构建提示词，替换所有占位符
    prompt = GENERATE_STRUCTURE_PROPOSAL_PROMPT
    prompt = prompt.replace("TOTAL_SCENE_COUNT_PLACEHOLDER", str(scene_count))
    prompt = prompt.replace("MAX_SCENE_INDEX_PLACEHOLDER", str(max_scene_index))
    prompt = prompt.replace("VIDEO_SUMMARY_PLACEHOLDER", video_summary)
    prompt = prompt.replace("AUDIO_SUMMARY_PLACEHOLDER", audio_summary)
    prompt = prompt.replace("AUDIO_STRUCTURE_PLACEHOLDER", audio_structure)
    prompt = prompt.replace("INSTRUCTION_PLACEHOLDER", user_instruction)
    prompt = prompt.replace("MAIN_CHARACTER_PLACEHOLDER", main_character or "the main character")

    # 如果有之前的反馈，添加到提示词中
    if feedback:
        prompt += f"\n\n**IMPORTANT - PREVIOUS ATTEMPT FAILED:**\n{feedback}\nPlease fix this issue in your response."

    # 调用 LLM 生成结构提案
    return _call_agent_litellm([{"role": "user", "content": prompt}], max_tokens=config.AGENT_MODEL_MAX_TOKEN)


def generate_structure_proposal_with_retry(
    video_scene_path: str,
    audio_caption_path: str,
    user_instruction: str,
    max_retries: int = 2,
    selected_start_str: str | None = None,
    selected_end_str: str | None = None,
    main_character: str | None = None,
) -> str | None:
    """
    Generate structure proposal with basic validation and retry.
    生成结构提案，包含基本验证和重试机制。
    
    This function wraps generate_structure_proposal() with:
    此函数包装 generate_structure_proposal()，添加：
    1. JSON parsing validation
    1. JSON 解析验证
    2. Scene distribution checking
    2. 场景分布检查
    3. Automatic retry with feedback on failure
    3. 失败时自动带反馈重跑
    
    Args:
        video_scene_path: 视频场景摘要路径 (Path to video scene summaries)
        audio_caption_path: 音频字幕路径 (Path to audio captions)
        user_instruction: 用户编辑指令 (User instruction)
        max_retries: 最大重试次数，默认 2 (Max retries, default 2)
        selected_start_str: 选定音频起始时间（可选） (Selected audio start time, optional)
        selected_end_str: 选定音频结束时间（可选） (Selected audio end time, optional)
        main_character: 主要角色名称（可选） (Main character name, optional)
    Returns:
        验证通过的结构提案文本，或最后一次尝试的结果 (Validated proposal text, or last attempt result)
    """
    # 获取场景总数用于验证
    _, scene_count = load_scene_summaries(video_scene_path)
    # 首次调用生成结构提案
    content = generate_structure_proposal(
        video_scene_path, audio_caption_path, user_instruction,
        selected_start_str, selected_end_str, main_character=main_character,
    )
    if content is None:
        return None

    last_feedback = None
    # 重试循环
    for retry in range(max_retries):
        try:
            # 尝试解析 LLM 输出
            parsed = parse_structure_proposal_output(content)
            if parsed is None:
                # 解析失败，重跑并传入反馈
                content = generate_structure_proposal(
                    video_scene_path, audio_caption_path, user_instruction,
                    selected_start_str, selected_end_str, last_feedback, main_character=main_character,
                )
                continue

            # 检查场景分布
            passed, last_feedback = check_scene_distribution(parsed, scene_count)
            if passed:
                # 验证通过，返回结果
                return content

            # 如果还有重试次数，则重跑
            if retry < max_retries - 1:
                content = generate_structure_proposal(
                    video_scene_path, audio_caption_path, user_instruction,
                    selected_start_str, selected_end_str, last_feedback, main_character=main_character,
                )
            else:
                # 达到最大重试次数，返回最后一次结果
                return content

        except Exception as e:
            # 捕获异常，如果还有重试次数则重跑
            if retry < max_retries - 1:
                content = generate_structure_proposal(
                    video_scene_path, audio_caption_path, user_instruction,
                    selected_start_str, selected_end_str, last_feedback, main_character=main_character,
                )
            else:
                return content

    return content


def generate_shot_plan(
    music_detailed_structure: A[list | dict | str, D("Detailed per-segment music analysis for current section.")],
    video_section_proposal: A[dict, D("Section brief extracted from structure proposal.")],
    scene_folder_path: A[str | None, D("Path to scene summaries folder.")] = None,
    user_instruction: A[str, D("User's editing instruction.")] = "",
    main_character: str | None = None,
) -> str | None:
    """
    Generate a one-to-one shot mapping for each music segment.
    为每个音乐段落生成一对一的镜头映射。
    
    This function creates a detailed shot plan that maps visual content to music segments.
    It considers:
    此函数创建将视觉内容映射到音乐段落的详细镜头计划。它考虑：
    - Music structure (beats, mood changes, intensity)
    - 音乐结构（节拍、情绪变化、强度）
    - Video section context (theme, narrative logic)
    - 视频段落上下文（主题、叙事逻辑）
    - Related scene descriptions for better matching
    - 相关场景描述以更好地匹配
    
    Args:
        music_detailed_structure: 音乐的详细结构分析，可以是列表、字典或 JSON 字符串 (Music structure analysis)
        video_section_proposal: 视频段落提案，包含主题、相关场景等信息 (Video section proposal)
        scene_folder_path: 场景摘要文件夹路径（可选） (Path to scene summaries folder, optional)
        user_instruction: 用户编辑指令 (User instruction)
        main_character: 主要角色名称（可选） (Main character name, optional)
    Returns:
        LLM 生成的镜头计划 JSON 文本，失败时返回 None (Generated shot plan JSON text, None on failure)
    """
    # 将音乐结构转换为 JSON 字符串
    if isinstance(music_detailed_structure, (dict, list)):
        music_json = json.dumps(music_detailed_structure, ensure_ascii=False, indent=2)
    else:
        music_json = str(music_detailed_structure or '')

    # 构建提示词，替换占位符
    prompt = GENERATE_SHOT_PLAN_PROMPT
    prompt = prompt.replace("AUDIO_SUMMARY_PLACEHOLDER", music_json)
    prompt = prompt.replace("VIDEO_SECTION_INFO_PLACEHOLDER", str(video_section_proposal))
    prompt = prompt.replace("INSTRUCTION_PLACEHOLDER", user_instruction)
    prompt = prompt.replace("MAIN_CHARACTER_PLACEHOLDER", main_character or "the main character")

    # 加载相关场景的描述，为 LLM 提供更多上下文
    related_video_context = ""
    related_scenes = video_section_proposal.get("related_scenes", []) if isinstance(video_section_proposal, dict) else []
    if related_scenes and scene_folder_path:
        scene_descriptions = []
        # 遍历相关场景索引，加载每个场景的摘要
        for scene_idx in related_scenes:
            scene_file = os.path.join(scene_folder_path, f"scene_{scene_idx}.json")
            if os.path.exists(scene_file):
                try:
                    with open(scene_file, 'r', encoding='utf-8') as f:
                        scene_data = json.load(f)
                    # 提取场景摘要
                    scene_summary = (
                        scene_data.get('video_analysis', {})
                        .get('scene_caption', {})
                        .get('scene_summary', '')
                    )
                    if scene_summary:
                        scene_descriptions.append(f"Scene {scene_idx}: {scene_summary}")
                except Exception:
                    # 忽略单个场景加载失败，继续处理其他场景
                    pass
        # 将所有场景描述合并为一个字符串
        related_video_context = "\n".join(scene_descriptions)

    # 将相关视频上下文添加到提示词中
    prompt = prompt.replace("RELATED_VIDEO_PLACEHOLDER", related_video_context)

    # 调用 LLM 生成镜头计划
    return _call_agent_litellm([{"role": "user", "content": prompt}], max_tokens=config.AGENT_MODEL_MAX_TOKEN)


def _validate_shot_plan_result(shot_plan: dict | None, expect_non_empty: bool = True) -> tuple[bool, str]:
    """
    Validate parsed shot plan structure.
    验证解析后的镜头计划结构。
    
    This function checks:
    此函数检查：
    1. shot_plan is a dictionary
    1. shot_plan 是字典类型
    2. 'shots' field exists and is a list
    2. 'shots' 字段存在且是列表
    3. Each shot in the list is a dictionary
    3. 列表中的每个镜头都是字典
    4. Optionally check if shots list is non-empty
    4. 可选地检查镜头列表是否非空
    
    Args:
        shot_plan: 要验证的镜头计划字典 (Shot plan dict to validate)
        expect_non_empty: 是否要求镜头列表非空，默认 True (Whether to require non-empty shots list)
    Returns:
        (passed, message) 元组，passed 为布尔值，message 为验证信息 ((passed, message) tuple)
    """
    # 检查是否为字典类型
    if not isinstance(shot_plan, dict):
        return False, "shot_plan is not a dict"

    # 检查 'shots' 字段
    shots = shot_plan.get("shots")
    if not isinstance(shots, list):
        return False, "missing or invalid 'shots' list"

    # 检查镜头列表是否非空（如果需要）
    if expect_non_empty and len(shots) == 0:
        return False, "'shots' is empty"

    # 检查每个镜头是否为字典
    for idx, shot in enumerate(shots):
        if not isinstance(shot, dict):
            return False, f"shot at index {idx} is not an object"

    return True, "ok"


def generate_shot_plan_with_retry(
    music_detailed_structure: list | dict | str,
    video_section_proposal: dict,
    scene_folder_path: str | None = None,
    user_instruction: str = "",
    max_retries: int | None = None,
    main_character: str | None = None,
) -> dict | None:
    """
    Generate and parse shot plan with validation + retry.
    生成并解析镜头计划，包含验证和重试机制。
    
    This function wraps generate_shot_plan() with:
    此函数包装 generate_shot_plan()，添加：
    1. JSON parsing of LLM response
    1. LLM 响应的 JSON 解析
    2. Structure validation using _validate_shot_plan_result()
    2. 使用 _validate_shot_plan_result() 进行结构验证
    3. Exponential backoff retry on failure
    3. 失败时指数退避重试
    
    Args:
        music_detailed_structure: 音乐详细结构分析 (Music structure analysis)
        video_section_proposal: 视频段落提案 (Video section proposal)
        scene_folder_path: 场景摘要文件夹路径（可选） (Scene folder path, optional)
        user_instruction: 用户编辑指令 (User instruction)
        max_retries: 最大重试次数，默认从配置读取 (Max retries, default from config)
        main_character: 主要角色名称（可选） (Main character name, optional)
    Returns:
        解析后的镜头计划字典，失败时返回 None (Parsed shot plan dict, None on failure)
    """
    # 从配置中获取重试次数和退避参数
    retries = max(1, int(max_retries or getattr(config, "AGENT_MODEL_MAX_RETRIES", 3)))
    base_backoff = float(getattr(config, "AGENT_RATE_LIMIT_BACKOFF_BASE", 1.0))
    max_backoff = float(getattr(config, "AGENT_RATE_LIMIT_MAX_BACKOFF", 8.0))
    # 判断是否期望非空镜头列表
    expected_non_empty = bool(music_detailed_structure) if isinstance(music_detailed_structure, list) else True
    last_error = "unknown_error"

    # 重试循环
    for attempt in range(1, retries + 1):
        # 调用 LLM 生成镜头计划
        raw_shot_plan = generate_shot_plan(
            music_detailed_structure,
            video_section_proposal,
            scene_folder_path,
            user_instruction,
            main_character=main_character,
        )
        if not raw_shot_plan:
            last_error = "empty response from shot plan request"
        else:
            # 解析 LLM 输出的 JSON
            parsed_shot_plan = parse_shot_plan_output(raw_shot_plan)
            # 验证解析结果的结构
            is_valid, reason = _validate_shot_plan_result(
                parsed_shot_plan,
                expect_non_empty=expected_non_empty,
            )
            if is_valid:
                # 验证通过，返回结果
                return parsed_shot_plan
            last_error = f"invalid shot plan format: {reason}"

        # 如果还有重试次数，等待后重试
        if attempt < retries:
            # 指数退避：wait = base * 2^(attempt-1)
            # Java 类比：Math.min(maxBackoff, baseWait * Math.pow(2, attempt - 1))
            wait_seconds = min(max_backoff, base_backoff * (2 ** (attempt - 1)))
            print(f"🔄 [Screenwriter: Shot Plan] Retrying in {wait_seconds:.1f}s...")
            time.sleep(wait_seconds)

    # 达到最大重试次数，返回 None
    return None


def _seconds_to_srt_time(seconds: float) -> str:
    """
    Convert seconds to SRT subtitle time format (HH:MM:SS,mmm).
    将秒数转换为 SRT 字幕时间格式（HH:MM:SS,mmm）。
    
    SRT format requires comma-separated milliseconds.
    SRT 格式要求使用逗号分隔的毫秒数。
    
    Args:
        seconds: 秒数 (Seconds)
    Returns:
        SRT 格式的时间字符串，如 "00:01:23,456" (SRT formatted time string)
    
    Example:
    示例：
    >>> _seconds_to_srt_time(83.456)
    '00:01:23,456'
    """
    # 转换为毫秒数
    total_ms = max(0, int(round(seconds * 1000)))
    # 计算小时、分钟、秒、毫秒
    hh = total_ms // 3600000
    mm = (total_ms % 3600000) // 60000
    ss = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    # 格式化为 HH:MM:SS,mmm
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _subtitle_line_text(sub: dict) -> str:
    """
    Extract text from a subtitle entry, including speaker label if present.
    从字幕条目中提取文本，如果存在则包含说话者标签。
    
    Args:
        sub: 字幕字典，包含 'text' 和可选的 'speaker' 字段 (Subtitle dict with 'text' and optional 'speaker')
    Returns:
        格式化的字幕文本，如 "[Speaker] Text" 或 "Text" (Formatted subtitle text)
    """
    text = (sub.get('text') or '').strip()
    speaker = (sub.get('speaker') or '').strip()
    if speaker:
        return f"[{speaker}] {text}"
    return text


def _normalize_dialogue_text(text: str) -> str:
    """
    Normalize dialogue text for robust subtitle matching.
    标准化对话文本以实现鲁棒的字幕匹配。
    
    This function:
    此函数：
    1. Converts to lowercase
    1. 转换为小写
    2. Removes speaker labels in brackets [...] and angle brackets <...>
    2. 移除方括号 [...] 和尖括号 <...> 中的说话者标签
    3. Removes punctuation and special characters
    3. 移除标点符号和特殊字符
    4. Collapses multiple spaces into one
    4. 将多个空格合并为一个
    
    Args:
        text: 原始对话文本 (Raw dialogue text)
    Returns:
        标准化后的文本，用于相似度比较 (Normalized text for similarity comparison)
    
    Example:
    示例：
    >>> _normalize_dialogue_text("[John] Hello, World!")
    'john hello world'
    """
    if not text:
        return ""
    # 转换为小写并去除首尾空白
    clean = str(text).lower().strip()
    # 移除方括号中的内容（如说话者标签）
    clean = re.sub(r"\[[^\]]+\]", " ", clean)
    # 移除尖括号中的内容
    clean = re.sub(r"<[^>]+>", " ", clean)
    # 移除非单词字符（保留字母、数字、下划线）
    clean = re.sub(r"[^\w]+", " ", clean)
    # 将多个空格合并为一个
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _dialogue_similarity(a: str, b: str) -> float:
    """
    Compute fuzzy similarity between two normalized subtitle lines.
    计算两个标准化字幕行之间的模糊相似度。
    
    This function combines:
    此函数结合：
    1. Sequence matching (SequenceMatcher ratio)
    1. 序列匹配（SequenceMatcher 比率）
    2. Substring containment check
    2. 子串包含检查
    3. Jaccard similarity on word tokens
    3. 词元上的 Jaccard 相似度
    
    Args:
        a: 第一个标准化文本 (First normalized text)
        b: 第二个标准化文本 (Second normalized text)
    Returns:
        相似度分数，范围 0.0-1.0 (Similarity score, range 0.0-1.0)
    """
    if not a or not b:
        return 0.0
    # 完全相同，返回最高分
    if a == b:
        return 1.0
    
    # 使用 SequenceMatcher 计算序列相似度
    # Java 类比：类似 Apache Commons Text 的 LevenshteinDistance
    seq_score = SequenceMatcher(None, a, b).ratio()
    # 如果一个字符串包含另一个，提高分数
    if a in b or b in a:
        seq_score = max(seq_score, 0.9)

    # 计算 Jaccard 相似度（基于词元集合）
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return seq_score
    # Jaccard = |A ∩ B| / |A ∪ B|
    jaccard = len(a_tokens & b_tokens) / len(a_tokens | b_tokens)
    # 综合评分：65% 序列相似度 + 35% Jaccard 相似度
    return max(seq_score, 0.65 * seq_score + 0.35 * jaccard)


def _match_dialogue_lines_to_subtitles(
    lines: list[str],
    subtitles: list[dict],
    min_score: float = 0.55,
) -> list[dict]:
    """
    Match model-selected lines back to original SRT subtitle entries.
    将模型选择的对话行匹配回原始 SRT 字幕条目。
    
    This function:
    此函数：
    1. Normalizes both model output and SRT subtitles
    1. 标准化模型输出和 SRT 字幕
    2. Uses fuzzy matching to find best matches
    2. 使用模糊匹配找到最佳匹配
    3. Ensures monotonic ordering (matches go forward in time)
    3. 确保单调顺序（匹配按时间向前推进）
    4. Filters by minimum similarity score
    4. 按最小相似度分数过滤
    
    Args:
        lines: 模型选择的对话行列表 (Model-selected dialogue lines)
        subtitles: 原始 SRT 字幕列表 (Original SRT subtitles)
        min_score: 最小相似度阈值，默认 0.55 (Minimum similarity threshold, default 0.55)
    Returns:
        匹配的字幕条目列表 (List of matched subtitle entries)
    """
    if not lines or not subtitles:
        return []

    # 预计算所有字幕的标准化文本
    # Java 类比：List<String> normalized = subtitles.stream().map(...).collect(Collectors.toList())
    subtitle_norm = [_normalize_dialogue_text(_subtitle_line_text(s)) for s in subtitles]
    matched_indices = []
    last_idx = -1

    # 遍历模型选择的每一行
    for raw_line in lines:
        norm_line = _normalize_dialogue_text(str(raw_line))
        if not norm_line:
            continue

        # 在剩余的字幕中寻找最佳匹配
        best_idx = None
        best_score = 0.0
        # 从上一个匹配的位置之后开始搜索（确保单调性）
        for idx in range(last_idx + 1, len(subtitles)):
            score = _dialogue_similarity(norm_line, subtitle_norm[idx])
            if score > best_score:
                best_score = score
                best_idx = idx

        # 如果找到足够相似的匹配，记录下来
        if best_idx is not None and best_score >= min_score:
            matched_indices.append(best_idx)
            last_idx = best_idx

    if not matched_indices:
        return []

    # 去重并排序，然后返回对应的字幕条目
    unique_sorted = sorted(set(matched_indices))
    return [subtitles[i] for i in unique_sorted]


def _build_timed_lines(subtitles: list[dict], clip_start_sec: float) -> list[dict]:
    """
    Build per-line absolute and relative timing records.
    构建每行字幕的绝对和相对时间记录。
    
    This function converts subtitle timestamps to both:
    此函数将字幕时间戳转换为：
    - Absolute time (original video timeline)
    - 绝对时间（原始视频时间轴）
    - Relative time (relative to clip start)
    - 相对时间（相对于剪辑开始）
    
    Args:
        subtitles: 字幕列表，每个元素包含 'start_sec' 和 'end_sec' (List of subtitles with timing)
        clip_start_sec: 剪辑的起始时间（秒） (Clip start time in seconds)
    Returns:
        包含文本和时间的字典列表 (List of dicts with text and timing info)
    
    Example output:
    示例输出：
    [
        {
            "text": "Hello World",
            "start": "00:00:01,500",  # 相对时间
            "end": "00:00:03,200",    # 相对时间
            "source_start": "00:01:01,500",  # 绝对时间
            "source_end": "00:01:03,200"     # 绝对时间
        }
    ]
    """
    timed_lines = []
    for sub in subtitles:
        # 获取绝对时间（原始视频中的时间）
        abs_start = float(sub.get('start_sec', 0.0))
        abs_end = float(sub.get('end_sec', 0.0))
        # 计算相对时间（相对于剪辑开始）
        rel_start = max(0.0, abs_start - clip_start_sec)
        rel_end = max(rel_start, abs_end - clip_start_sec)
        timed_lines.append({
            "text": _subtitle_line_text(sub),
            "start": _seconds_to_srt_time(rel_start),      # 相对开始时间
            "end": _seconds_to_srt_time(rel_end),          # 相对结束时间
            "source_start": _seconds_to_srt_time(abs_start),  # 绝对开始时间
            "source_end": _seconds_to_srt_time(abs_end),      # 绝对结束时间
        })
    return timed_lines



def _format_subtitles_for_prompt(
    subtitles: list[dict],
    max_chars: int = HOOK_DIALOGUE_MAX_SUBTITLE_CHARS,
    window_mode: str = "tail",
    start_index: int | None = None,
) -> tuple[str, int]:
    """
    Format subtitles for inclusion in LLM prompt with character limit.
    格式化字幕以包含在 LLM 提示词中，并限制字符数。
    
    This function:
    此函数：
    1. Formats each subtitle with index, timing, and text
    1. 为每个字幕格式化索引、时间和文本
    2. Selects a subset based on window_mode (head/tail/random)
    2. 根据 window_mode（头部/尾部/随机）选择子集
    3. Ensures total length stays within max_chars limit
    3. 确保总长度不超过 max_chars 限制
    
    Args:
        subtitles: 字幕列表 (List of subtitle dicts)
        max_chars: 最大字符数限制，默认 20000 (Max character limit, default 20000)
        window_mode: 窗口模式 - "head"(开头), "tail"(结尾), "random_window"(随机) (Window mode)
        start_index: 随机窗口的起始索引（仅用于 random_window 模式） (Start index for random_window mode)
    Returns:
        (formatted_text, count) 元组，formatted_text 为格式化的字幕文本，count 为字幕数量 ((formatted_text, count) tuple)
    
    Example output format:
    示例输出格式：
    1
    00:00:01,500 --> 00:00:03,200 [1.7s]
    Hello World
    
    2
    00:00:04,000 --> 00:00:06,500 [2.5s]
    How are you?
    """
    # 构建所有字幕块
    all_blocks = []
    for idx, sub in enumerate(subtitles, start=1):
        text = _subtitle_line_text(sub).strip()
        if not text:
            continue
        # 计算持续时间
        dur = max(0.0, sub.get('end_sec', 0.0) - sub.get('start_sec', 0.0))
        # 格式化字幕块：索引 + 时间范围 + 文本
        block = (
            f"{idx}\n"
            f"{_seconds_to_srt_time(sub.get('start_sec', 0.0))} --> {_seconds_to_srt_time(sub.get('end_sec', 0.0))} [{dur:.1f}s]\n"
            f"{text}"
        )
        all_blocks.append(block)

    if not all_blocks:
        return "", 0

    used = 0
    selected = []

    # 根据窗口模式选择迭代顺序
    if window_mode == "random_window":
        # 随机窗口：从随机位置开始
        if start_index is None:
            start_index = random.randrange(len(all_blocks))
        start_index = max(0, min(start_index, len(all_blocks) - 1))
        iterable = all_blocks[start_index:]
    elif window_mode == "head":
        # 头部窗口：从头开始
        iterable = all_blocks
    else:
        # 尾部窗口（默认）：从尾开始
        iterable = reversed(all_blocks)

    # 选择不超过 max_chars 的字幕块
    for block in iterable:
        # 检查添加当前块是否会超出限制（+2 是为了 \n\n 分隔符）
        if selected and used + len(block) + 2 > max_chars:
            break
        selected.append(block)
        used += len(block) + 2

    # 如果是尾部窗口，需要反转回原始顺序
    if window_mode == "tail":
        selected.reverse()

    # 用双换行符连接所有选中的块
    return "\n\n".join(selected), len(selected)


def _extract_first_balanced_json_object(text: str) -> str | None:
    """
    Extract the first balanced {...} JSON object from mixed text.
    从混合文本中提取第一个平衡的 {...} JSON 对象。
    
    This function uses a state machine to track:
    此函数使用状态机跟踪：
    - Brace depth (to find matching closing brace)
    - 大括号深度（找到匹配的闭合大括号）
    - String boundaries (to ignore braces inside strings)
    - 字符串边界（忽略字符串内的大括号）
    - Escape sequences (to handle escaped quotes)
    - 转义序列（处理转义的引号）
    
    Args:
        text: 可能包含 JSON 对象的混合文本 (Mixed text potentially containing JSON)
    Returns:
        第一个完整的 JSON 对象字符串，未找到则返回 None (First complete JSON object string, None if not found)
    
    Example:
    示例：
    >>> _extract_first_balanced_json_object('Some text {"key": "value"} more text')
    '{"key": "value"}'
    """
    # 查找第一个左大括号
    start = text.find("{")
    if start < 0:
        return None

    depth = 0  # 大括号嵌套深度
    in_str = False  # 是否在字符串内
    escaped = False  # 是否遇到转义字符
    
    # 遍历文本，使用状态机跟踪
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            # 在字符串内部
            if escaped:
                # 前一个字符是转义符，当前字符被转义
                escaped = False
            elif ch == "\\":
                # 遇到转义符
                escaped = True
            elif ch == '"':
                # 遇到未转义的引号，退出字符串
                in_str = False
            continue

        # 在字符串外部
        if ch == '"':
            # 进入字符串
            in_str = True
        elif ch == "{":
            # 增加嵌套深度
            depth += 1
        elif ch == "}":
            # 减少嵌套深度
            depth -= 1
            if depth == 0:
                # 找到匹配的闭合大括号，返回完整的 JSON 对象
                return text[start:idx + 1]
    
    # 未找到平衡的 JSON 对象
    return None


def _parse_llm_json_object(raw_content: str) -> tuple[dict | None, Exception | None]:
    """
    Parse LLM output into a JSON object with light normalization.
    将 LLM 输出解析为 JSON 对象，进行轻度标准化处理。
    
    This function handles common LLM output issues:
    此函数处理常见的 LLM 输出问题：
    1. Markdown code block wrappers (```json ... ```)
    1. Markdown 代码块包装（```json ... ```）
    2. BOM characters and control characters
    2. BOM 字符和控制字符
    3. Non-standard escape sequences (\')
    3. 非标准转义序列（\'）
    4. Mixed text with embedded JSON objects
    4. 包含嵌入 JSON 对象的混合文本
    
    Args:
        raw_content: LLM 的原始输出文本 (Raw LLM output text)
    Returns:
        (parsed_dict, error) 元组，成功时 error 为 None ((parsed_dict, error) tuple, error is None on success)
    """
    if not raw_content:
        return None, ValueError("empty_response")

    # 清理文本：去除首尾空白
    clean = raw_content.strip()
    # 移除 Markdown 代码块标记
    if clean.startswith("```"):
        clean = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", clean)
        clean = re.sub(r"\n?```$", "", clean)
    # 移除 BOM 字符
    clean = clean.replace("\ufeff", "").strip()
    # 移除控制字符（ASCII 0-31，保留换行和制表符）
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", clean)

    # 构建候选列表：尝试多种解析方式
    candidates: list[str] = [clean]
    # 尝试提取平衡的 JSON 对象
    extracted = _extract_first_balanced_json_object(clean)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    # Some providers output non-standard \' escapes; normalize them.
    # 某些提供商输出非标准的 \' 转义；对其进行标准化
    for c in list(candidates):
        fixed = c.replace("\\'", "'")
        if fixed not in candidates:
            candidates.append(fixed)

    last_error: Exception | None = None
    # 尝试解析每个候选
    for c in candidates:
        try:
            parsed = json.loads(c)
            # 确保解析结果是字典类型
            if isinstance(parsed, dict):
                return parsed, None
            last_error = TypeError(f"json_root_not_object: {type(parsed).__name__}")
        except Exception as e:
            last_error = e

    # 所有候选都失败，返回最后一个错误
    return None, last_error


def select_hook_dialogue(
    subtitle_path: str,
    shot_plan: dict,
    instruction: str,
    target_duration_sec: float = 10.0,
    main_character: str | None = None,
    prompt_window_mode: str = "tail_then_head",
    random_window_attempts: int = 3,
) -> dict:
    """
    Select ONE opening dialogue clip for the whole video (target ~10s).
    为整个视频选择一个开场对话片段（目标约 10 秒）。
    
    This function uses LLM to select an engaging hook dialogue from subtitles.
    It tries multiple strategies:
    此函数使用 LLM 从字幕中选择引人入胜的钩子对话。它尝试多种策略：
    1. Tail-first approach (end of video)
    1. 尾部优先策略（视频结尾）
    2. Head retry (beginning of video)
    2. 头部重试（视频开头）
    3. Random windows (multiple attempts)
    3. 随机窗口（多次尝试）
    
    The selection process:
    选择流程：
    - Formats subtitles with timing information
    - 格式化带时间信息的字幕
    - Calls LLM to select best hook based on shot plan context
    - 调用 LLM 根据镜头计划上下文选择最佳钩子
    - Matches selected lines back to SRT entries
    - 将选定的行匹配回 SRT 条目
    - Validates duration constraints
    - 验证时长限制
    
    Args:
        subtitle_path: SRT 字幕文件路径 (Path to SRT subtitle file)
        shot_plan: 镜头计划字典，包含 'video_structure' (Shot plan dict with 'video_structure')
        instruction: 用户编辑指令 (User editing instruction)
        target_duration_sec: 目标时长（秒），默认 10.0 (Target duration in seconds, default 10.0)
        main_character: 主要角色名称（可选） (Main character name, optional)
        prompt_window_mode: 提示词窗口模式，默认 "tail_then_head" (Prompt window mode, default "tail_then_head")
        random_window_attempts: 随机窗口尝试次数，默认 3 (Random window attempts, default 3)
    Returns:
        包含选定对话的字典，包括文本、时间、原因等 (Dict with selected dialogue including text, timing, reason)
    Raises:
        HookDialogueSelectionError: 如果所有尝试都失败 (If all attempts fail)
    """
    # 解析 SRT 字幕文件
    subtitles_all = parse_srt_file(subtitle_path)
    if not subtitles_all:
        raise HookDialogueSelectionError(
            f"No subtitle entries found in subtitle file: {subtitle_path}"
        )

    # 构建镜头计划摘要，供 LLM 参考
    shots_info = []
    for section in shot_plan.get('video_structure', []):
        for shot in section.get('shot_plan', {}).get('shots', []):
            content = shot.get('content', '')
            emotion = shot.get('emotion', '')
            if content or emotion:
                shots_info.append(f"- {content} [{emotion}]")
    # 最多取前 10 个镜头的信息
    shot_plan_summary = "\n".join(shots_info[:10]) if shots_info else instruction

    # 字幕候选列表
    subtitle_candidates = subtitles_all
    # 计算时长范围：目标时长 ±5 秒，最小 6 秒
    min_duration_sec = max(6.0, target_duration_sec - 5.0)
    max_duration_sec = target_duration_sec + 5.0
    failure_reasons: list[str] = []  # 记录失败原因

    def _try_select(window_mode: str, attempt_side: str, start_index: int | None = None) -> dict | None:
        """
        Try to select hook dialogue using a specific window mode.
        尝试使用特定的窗口模式选择钩子对话。
        
        Args:
            window_mode: 窗口模式（tail/head/random_window） (Window mode)
            attempt_side: 尝试来源标识，用于错误报告 (Attempt source identifier for error reporting)
            start_index: 随机窗口的起始索引（可选） (Start index for random_window, optional)
        Returns:
            选定的对话字典，失败时返回 None (Selected dialogue dict, None on failure)
        """
        # 格式化字幕上下文
        subtitle_context, _ = _format_subtitles_for_prompt(
            subtitle_candidates,
            max_chars=HOOK_DIALOGUE_MAX_SUBTITLE_CHARS,
            window_mode=window_mode,
            start_index=start_index,
        )
        if not subtitle_context:
            failure_reasons.append(
                f"from_{attempt_side}: subtitle context is empty after formatting"
            )
            return None

        # 构建提示词
        prompt = SELECT_HOOK_DIALOGUE_PROMPT.format(
            instruction=instruction,
            main_character=main_character or "the main character",
            shot_plan_summary=shot_plan_summary,
            subtitles=subtitle_context,
            target_duration_sec=int(round(target_duration_sec)),
            min_duration_sec=int(round(min_duration_sec)),
            max_duration_sec=int(round(max_duration_sec)),
        )

        # 调用 LLM，最多尝试 2 次
        llm_result = None
        for attempt in range(2):
            attempt_prompt = prompt
            # 第二次尝试时添加强制 JSON 格式的提示
            if attempt == 1:
                attempt_prompt += (
                    "\n\nIMPORTANT: Your previous answer was invalid. "
                    "Return ONLY a valid JSON object with keys lines,start,end,reason."
                )
            raw_content = _call_agent_litellm([{"role": "user", "content": attempt_prompt}], max_tokens=16000)
            if not raw_content:
                continue
            # 解析 LLM 输出
            parsed, _ = _parse_llm_json_object(raw_content)
            if parsed is not None:
                llm_result = parsed
                break

        if llm_result is None:
            failure_reasons.append(
                f"from_{attempt_side}: LLM did not return a valid JSON selection"
            )
            return None

        # 提取时间信息
        start_sec = hhmmss_to_seconds(str(llm_result.get('start', '')).strip())
        end_sec = hhmmss_to_seconds(str(llm_result.get('end', '')).strip())
        lines = llm_result.get('lines') if isinstance(llm_result.get('lines'), list) else []

        # 将 LLM 选择的行匹配回字幕条目
        matched_subtitles = _match_dialogue_lines_to_subtitles(lines, subtitle_candidates)
        # 如果匹配失败但提供了时间范围，则按时间范围过滤
        if not matched_subtitles and not (start_sec <= 0 and end_sec <= 0):
            matched_subtitles = [
                s for s in subtitle_candidates
                if s.get('end_sec', 0.0) >= start_sec and s.get('start_sec', 0.0) <= end_sec
            ]
        if not matched_subtitles:
            failure_reasons.append(
                f"from_{attempt_side}: no subtitles matched selected dialogue lines={lines!r}, "
                f"start={llm_result.get('start')!r}, end={llm_result.get('end')!r}"
            )
            return None

        # 按开始时间排序匹配的字幕
        matched_subtitles = sorted(matched_subtitles, key=lambda s: float(s.get('start_sec', 0.0)))
        # 获取第一个和最后一个字幕的时间
        source_start_sec = float(matched_subtitles[0].get('start_sec', 0.0))
        source_end_sec = float(matched_subtitles[-1].get('end_sec', 0.0))
        duration_sec = max(0.0, source_end_sec - source_start_sec)

        # 检查时长是否在允许范围内
        if duration_sec < min_duration_sec or duration_sec > max_duration_sec:
            reason = (
                f"from_{attempt_side}: duration {duration_sec:.2f}s out of range "
                f"[{min_duration_sec:.0f}, {max_duration_sec:.0f}]s"
            )
            print(f"[Hook Dialogue] Rejected: {reason}")
            failure_reasons.append(reason)
            return None

        # 构建带时间的行（相对时间和绝对时间）
        timed_lines = _build_timed_lines(matched_subtitles, source_start_sec)
        # 返回选定的对话信息
        return {
            "lines": [item["text"] for item in timed_lines if item.get("text")],  # 对话文本列表
            "timed_lines": timed_lines,  # 带时间的行
            "start": _seconds_to_srt_time(0.0),  # 相对开始时间（从 0 开始）
            "end": _seconds_to_srt_time(duration_sec),  # 相对结束时间
            "source_start": _seconds_to_srt_time(source_start_sec),  # 原始视频中的开始时间
            "source_end": _seconds_to_srt_time(source_end_sec),  # 原始视频中的结束时间
            "reason": llm_result.get('reason', ''),  # LLM 选择的原因
            "duration_seconds": round(duration_sec, 3),  # 持续时间（秒）
            "selection_method": "llm_srt_matched",  # 选择方法标识
        }

    # 主选择逻辑：尝试不同的窗口模式
    result = None
    if prompt_window_mode == "random_window":
        # 纯随机窗口模式：从多个随机位置尝试
        total_subtitles = len(subtitle_candidates)
        attempt_count = max(1, min(int(random_window_attempts), total_subtitles))
        # 随机选择不重复的起始索引
        random_start_indices = random.sample(range(total_subtitles), k=attempt_count)
        for attempt_number, start_index in enumerate(random_start_indices, start=1):
            attempt_side = f"random_window_{attempt_number}_start_{start_index + 1}"
            print(
                f"[Hook Dialogue] Trying random subtitle window "
                f"{attempt_number}/{attempt_count} starting at subtitle #{start_index + 1}..."
            )
            result = _try_select(
                window_mode="random_window",
                attempt_side=attempt_side,
                start_index=start_index,
            )
            if result is not None:
                break  # 成功则退出
    else:
        # 默认模式：先从尾部尝试，失败后从头部重试，最后尝试随机窗口
        # First attempt from end; retry from beginning if rejected
        result = _try_select(window_mode="tail", attempt_side="end")
        if result is None:
            print("[Hook Dialogue] Retrying from beginning of subtitles...")
            result = _try_select(window_mode="head", attempt_side="beginning")
        # 如果前两次都失败，且允许随机尝试，则进行随机窗口尝试
        if result is None and random_window_attempts > 0:
            total_subtitles = len(subtitle_candidates)
            attempt_count = max(1, min(int(random_window_attempts), total_subtitles))
            random_start_indices = random.sample(range(total_subtitles), k=attempt_count)
            for attempt_number, start_index in enumerate(random_start_indices, start=1):
                attempt_side = f"fallback_random_window_{attempt_number}_start_{start_index + 1}"
                print(
                    f"[Hook Dialogue] Retrying with random subtitle window "
                    f"{attempt_number}/{attempt_count} starting at subtitle #{start_index + 1}..."
                )
                result = _try_select(
                    window_mode="random_window",
                    attempt_side=attempt_side,
                    start_index=start_index,
                )
                if result is not None:
                    break  # 成功则退出

    # 如果所有尝试都失败，抛出异常
    if result is None:
        failure_detail = " | ".join(failure_reasons) if failure_reasons else "unknown reason"
        raise HookDialogueSelectionError(
            "Failed to select hook dialogue from subtitles. "
            f"subtitle_path={subtitle_path}. Details: {failure_detail}"
        )

    # 打印选定的钩子对话信息
    print(f"\n[Hook Dialogue Selected]")
    print(f"  Lines: {result.get('lines')}")
    print(f"  Relative Time: {result.get('start')} --> {result.get('end')} ({result.get('duration_seconds')}s)")
    print(f"  Source Time: {result.get('source_start')} --> {result.get('source_end')}")
    print(f"  Reason: {result.get('reason')}\n")
    return result


def refresh_hook_dialogue_in_shot_plan(
    shot_plan_path: str,
    subtitle_path: str,
    instruction: str | None = None,
    main_character: str | None = None,
    target_duration_sec: float = 10.0,
    prompt_window_mode: str = "tail_then_head",
    random_window_attempts: int = 3,
) -> dict:
    """
    Refresh hook dialogue in an existing shot-plan JSON file and save it in place.
    刷新现有镜头计划 JSON 文件中的钩子对话，并原地保存。
    
    This function:
    此函数：
    1. Loads the existing shot plan from disk
    1. 从磁盘加载现有的镜头计划
    2. Extracts or uses provided instruction
    2. 提取或使用提供的指令
    3. Calls select_hook_dialogue() to generate new hook dialogue
    3. 调用 select_hook_dialogue() 生成新的钩子对话
    4. Updates the shot plan data with new hook dialogue
    4. 用新的钩子对话更新镜头计划数据
    5. Saves the updated shot plan back to disk
    5. 将更新后的镜头计划保存回磁盘
    
    Args:
        shot_plan_path: 镜头计划 JSON 文件路径 (Path to shot plan JSON file)
        subtitle_path: SRT 字幕文件路径 (Path to SRT subtitle file)
        instruction: 用户编辑指令（可选，默认从镜头计划中读取） (User instruction, optional, defaults to reading from shot plan)
        main_character: 主要角色名称（可选） (Main character name, optional)
        target_duration_sec: 目标时长（秒），默认 10.0 (Target duration in seconds, default 10.0)
        prompt_window_mode: 提示词窗口模式，默认 "tail_then_head" (Prompt window mode, default "tail_then_head")
        random_window_attempts: 随机窗口尝试次数，默认 3 (Random window attempts, default 3)
    Returns:
        更新后的镜头计划字典 (Updated shot plan dict)
    Raises:
        FileNotFoundError: 如果文件或字幕文件不存在 (If files not found)
        ValueError: 如果缺少指令且无法从镜头计划中提取 (If instruction is missing)
    """
    # 检查文件是否存在
    if not os.path.exists(shot_plan_path):
        raise FileNotFoundError(f"Shot plan file not found: {shot_plan_path}")
    if not subtitle_path or not os.path.exists(subtitle_path):
        raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")

    # 加载现有的镜头计划
    with open(shot_plan_path, 'r', encoding='utf-8') as f:
        shot_plan_data = json.load(f)

    # 确定使用的指令：优先使用参数，其次从镜头计划中提取
    instruction_to_use = (
        instruction
        or shot_plan_data.get("instruction")
        or shot_plan_data.get("narrative_logic")
    )
    if not instruction_to_use:
        raise ValueError(
            f"Cannot refresh hook dialogue because instruction is missing in {shot_plan_path}"
        )

    # 调用 select_hook_dialogue 生成新的钩子对话
    shot_plan_data["hook_dialogue"] = select_hook_dialogue(
        subtitle_path,
        shot_plan_data,
        instruction_to_use,
        target_duration_sec=target_duration_sec,
        main_character=main_character,
        prompt_window_mode=prompt_window_mode,
        random_window_attempts=random_window_attempts,
    )
    # 确保指令字段存在
    shot_plan_data["instruction"] = instruction_to_use

    # 保存更新后的镜头计划到磁盘（原地更新）
    with open(shot_plan_path, 'w', encoding='utf-8') as f:
        json.dump(shot_plan_data, f, indent=2, ensure_ascii=False)

    return shot_plan_data


class Screenwriter:
    """
    Screenwriter Agent - Main orchestrator for video shot planning.
    编剧智能体 - 视频镜头计划的主编排器。
    
    This class coordinates the entire screenwriting pipeline:
    此类协调整个编剧流程：
    1. Audio segment selection based on user instruction
    1. 根据用户指令选择音频段落
    2. Video structure proposal generation
    2. 生成视频结构提案
    3. Shot plan generation for each section
    3. 为每个段落生成镜头计划
    4. Hook dialogue selection (if subtitles available)
    4. 钩子对话选择（如果有字幕）
    
    Java 类比：类似一个工作流引擎，按顺序执行多个步骤并聚合结果。
    """
    
    def __init__(self, video_scene_path, audio_caption_path, output_path, video_path=None, subtitle_path=None, main_character=None, **kwargs):
        """
        Initialize the Screenwriter agent.
        初始化编剧智能体。
        
        Args:
            video_scene_path: 视频场景摘要文件夹路径 (Path to video scene summaries folder)
            audio_caption_path: 音频标注 JSON 文件路径 (Path to audio caption JSON file)
            output_path: 输出镜头计划 JSON 文件路径 (Path to output shot plan JSON file)
            video_path: 原始视频文件路径（可选） (Path to original video file, optional)
            subtitle_path: SRT 字幕文件路径（可选） (Path to SRT subtitle file, optional)
            main_character: 主要角色名称（可选） (Main character name, optional)
            **kwargs: 其他参数（保留用于扩展） (Other parameters for extensibility)
        """
        self.video_scene_path = video_scene_path
        self.audio_caption_path = audio_caption_path
        # 加载音频数据库
        self.audio_db = json.load(open(audio_caption_path, 'r', encoding='utf-8'))
        self.video_path = video_path
        self.subtitle_path = subtitle_path
        self.output_path = output_path
        self.main_character = main_character

    def run(self, instruction) -> dict:
        """
        Run the screenwriter pipeline to generate a shot plan.
        运行编剧流程以生成镜头计划。
        
        This method executes the complete pipeline:
        此方法执行完整的流程：
        1. Check for existing output and reuse if valid
        1. 检查现有输出，如果有效则重用
        2. Select audio segment based on instruction
        2. 根据指令选择音频段落
        3. Generate video structure proposal
        3. 生成视频结构提案
        4. Filter audio sub-segments within selected range
        4. 过滤选定范围内的音频子段落
        5. Generate detailed shot plan
        5. 生成详细的镜头计划
        6. Select hook dialogue (if subtitles available)
        6. 选择钩子对话（如果有字幕）
        7. Save result to output file
        7. 保存结果到输出文件
        
        Args:
            instruction: 用户编辑指令 (User editing instruction)
        Returns:
            完整的镜头计划字典 (Complete shot plan dict)
        Raises:
            RuntimeError: 如果关键步骤失败 (If critical steps fail)
        """
        # 检查是否存在有效的输出文件，避免重复计算
        if self.output_path and os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    existing_output = json.load(f)
            except Exception as exc:
                print(f"⚠️  [Screenwriter] Warning: failed to load existing output {self.output_path}: {exc}. Regenerating...")
            else:
                # 检查现有输出是否完整
                missing_parts = get_missing_shot_plan_parts(existing_output)
                if missing_parts:
                    print(
                        f"⚠️  [Screenwriter] Existing shot plan is incomplete. "
                        f"Missing parts: {', '.join(missing_parts)}. Regenerating..."
                    )
                elif self.subtitle_path and os.path.exists(self.subtitle_path) and not existing_output.get("hook_dialogue"):
                    # 如果缺少钩子对话但有字幕，尝试补充
                    print(
                        f"⚠️  [Screenwriter] Existing shot plan is missing hook_dialogue. "
                        f"Retrying hook dialogue selection for {self.output_path}..."
                    )
                    existing_output = refresh_hook_dialogue_in_shot_plan(
                        self.output_path,
                        self.subtitle_path,
                        instruction=instruction,
                        main_character=self.main_character,
                        target_duration_sec=10.0,
                    )
                    print(f"💾 [Screenwriter] Updated hook dialogue saved to {self.output_path}")
                    return existing_output
                else:
                    # 输出完整，直接返回
                    return existing_output

        # Step 1: Select the audio segment first
        # 步骤 1：首先选择音频段落
        selected_start_str, selected_end_str = select_audio_segment(self.audio_db, instruction)

        print(
            f"\n🎵 [Screenwriter] Audio segment selected: "
            f"{selected_start_str} → {selected_end_str}\n"
        )

        # Step 2: Generate structure proposal scoped to the selected audio segment
        # 步骤 2：生成限定在选定音频段落的结构提案
        structure_proposal = generate_structure_proposal_with_retry(
            self.video_scene_path, self.audio_caption_path, instruction,
            selected_start_str=selected_start_str,
            selected_end_str=selected_end_str,
            main_character=self.main_character,
        )
        if structure_proposal is None:
            raise RuntimeError("generate_structure_proposal_with_retry returned None — check API connectivity and model config")
        # 解析结构提案 JSON
        structure_proposal = parse_structure_proposal_output(structure_proposal)

        # 获取音频段落列表
        audio_sections = self.audio_db.get('sections', [])

        # 过滤出选定范围内的子段落
        selected_sub_segments = filter_sub_segments_by_range(
            audio_sections, selected_start_str, selected_end_str
        )

        # 内部辅助函数：将时间字符串转换为秒数
        def _to_sec(t):
            if isinstance(t, (int, float)):
                return float(t)
            parts = str(t).split(':')
            if len(parts) == 3:
                h, m, s = [float(x) for x in parts]
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m, s = [float(x) for x in parts]
                return m * 60 + s
            else:
                try:
                    return float(parts[0])
                except ValueError:
                    return 0

        # 计算选定段落的起始、结束时间和持续时间
        start_time = _to_sec(selected_start_str)
        end_time = _to_sec(selected_end_str)
        duration = end_time - start_time

        # Determine display name from first overlapping top-level section
        # 从第一个重叠的顶层段落确定显示名称
        segment_name = "audio segment"
        for sec in audio_sections:
            sec_start = _to_sec(sec.get('Start_Time', 0))
            sec_end = _to_sec(sec.get('End_Time', 0))
            if sec_end > start_time and sec_start < end_time:
                segment_name = sec.get('name', segment_name)
                break

        # Step 3: Generate detailed shot plan
        # 步骤 3：生成详细的镜头计划
        shot_plan = generate_shot_plan_with_retry(
            selected_sub_segments,
            structure_proposal,
            self.video_scene_path,
            instruction,
            main_character=self.main_character,
        )
        if shot_plan is None:
            raise RuntimeError(
                "Failed to generate a valid shot plan after retries — "
                "check API connectivity / model availability / prompt output format."
            )
        
        # Step 4: Select hook dialogue (if subtitles available)
        # 步骤 4：选择钩子对话（如果有字幕）
        hook_dialogue = None
        if self.subtitle_path and os.path.exists(self.subtitle_path):
            # 构建部分输出，用于钩子对话选择
            partial_output = {
                "video_structure": [{**structure_proposal, "shot_plan": shot_plan}]
            }
            # 调用钩子对话选择函数
            hook_dialogue = select_hook_dialogue(
                self.subtitle_path,
                partial_output,
                instruction,
                target_duration_sec=15.0,  # 钩子对话目标时长 15 秒
                main_character=self.main_character,
            )
        else:
            hook_dialogue = None

        # Step 5: Assemble final output
        # 步骤 5：组装最终输出
        import datetime
        output_data = {
            "instruction": instruction,  # 用户指令
            "metadata": {  # 元数据
                "created_at": datetime.datetime.now().isoformat(),  # 创建时间
                "video_path": self.video_path,  # 视频路径
                "audio_path": self.audio_caption_path,  # 音频路径
                "video_scene_path": self.video_scene_path,  # 场景摘要路径
                "selected_audio_start": selected_start_str,  # 选定音频开始时间
                "selected_audio_end": selected_end_str,  # 选定音频结束时间
            },
            "overall_theme": f"Short video for {segment_name}",  # 整体主题
            "narrative_logic": instruction,  # 叙事逻辑
            "hook_dialogue": hook_dialogue,  # 钩子对话
            "video_structure": [{  # 视频结构
                **structure_proposal,  # 展开结构提案
                "start_time": start_time,  # 开始时间（秒）
                "end_time": end_time,  # 结束时间（秒）
                "shot_plan": shot_plan,  # 镜头计划
            }],
        }

        print("\n✅ [Screenwriter] Short video shot plan generated successfully!")

        # Step 6: Save to disk
        # 步骤 6：保存到磁盘
        if self.output_path:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 [Screenwriter] Complete shot plan saved to {self.output_path}")

        return output_data


def main():
    def _norm_name(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    def _resolve_video_assets(
        video_path: str,
        video_scene_path: str | None,
    ) -> str:
        """Resolve video scene path from a raw video path."""
        if video_scene_path:
            return video_scene_path

        if not video_path:
            raise ValueError("--video-path is required")

        repo_root = Path(__file__).resolve().parents[1]
        video_db_root = repo_root / "video_database" / "Video"
        if not video_db_root.exists():
            raise FileNotFoundError(
                f"Cannot find video database root at: {video_db_root}. "
                "Run from the repo workspace or pass --video-scene-path manually."
            )

        stem = Path(video_path).stem
        target_norm = _norm_name(stem)

        match_dir: Path | None = None
        if (video_db_root / stem).is_dir():
            match_dir = video_db_root / stem
        else:
            for child in video_db_root.iterdir():
                if child.is_dir() and _norm_name(child.name) == target_norm:
                    match_dir = child
                    break

        if match_dir is None:
            raise FileNotFoundError(
                f"Cannot resolve video database folder for '{stem}'. "
                "Please pass --video-scene-path manually."
            )

        captions_dir = match_dir / "captions"
        for candidate in ("scene_summaries_video", "scene_summaries"):
            cand = captions_dir / candidate
            if cand.is_dir():
                return str(cand)

        raise FileNotFoundError(
            f"Cannot find scene summaries folder under: {captions_dir}. "
            "Please pass --video-scene-path manually."
        )

    parser = argparse.ArgumentParser(
        description="Generate a short-video shot plan from video scene summaries and audio captions."
    )
    parser.add_argument("--video-scene-path", default=None,
                        help="Path to scene summaries folder. If omitted, inferred from --video-path.")
    parser.add_argument("--audio-caption-path", required=True,
                        help="Path to captions.json describing the audio segments.")
    parser.add_argument("--video-path", required=True,
                        help="Path to the source video file.")
    parser.add_argument("--output-path", required=True,
                        help="Output path to save the generated shot plan JSON.")
    parser.add_argument("--instruction", default="A dynamic montage.",
                        help="User instruction / creative brief.")
    args = parser.parse_args()

    resolved_video_scene_path = _resolve_video_assets(args.video_path, args.video_scene_path)

    agent = Screenwriter(
        video_scene_path=resolved_video_scene_path,
        audio_caption_path=args.audio_caption_path,
        output_path=args.output_path,
        video_path=args.video_path,
    )
    agent.run(args.instruction)


if __name__ == "__main__":
    main()
