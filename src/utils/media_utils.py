"""
Shared media utilities for the VideoCuttingAgent pipeline.
VideoCuttingAgent 流水线的共享媒体工具。

Covers: JSON parsing, SRT parsing, time conversion, image encoding, shot scene I/O.
涵盖：JSON 解析、SRT 解析、时间转换、图像编码、镜头场景 I/O。

Java 类比：类似一个工具类集合，包含各种静态辅助方法。
"""

import base64
import json
import os
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def parse_json_safely(text: Optional[str]) -> Optional[Dict]:
    """
    Robustly parse a JSON string, stripping Markdown code fences if present.
    稳健地解析 JSON 字符串，如果存在则去除 Markdown 代码围栏。
    
    This function:
    此函数：
    1. Strips ```json ... ``` markdown wrappers
    1. 去除 ```json ... ``` Markdown 包装
    2. Falls back to regex extraction if direct parsing fails
    2. 如果直接解析失败，回退到正则表达式提取
    3. Returns None on complete failure
    3. 完全失败时返回 None
    
    Args:
        text: 可能包含 JSON 的文本字符串 (Text string that may contain JSON)
    Returns:
        解析后的字典，失败时返回 None (Parsed dict, None on failure)
    
    Java 类比：类似 Jackson 的 ObjectMapper 加上容错处理。
    """
    if text is None:
        return None
    text = text.strip()
    # 去除 Markdown 代码围栏
    if text.startswith("```"):
        text = re.sub(r"^```json\s*|^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)  # 尝试直接解析
    except json.JSONDecodeError:
        # 回退：使用正则表达式提取 JSON 对象
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Time conversion
# ---------------------------------------------------------------------------

def seconds_to_hhmmss(seconds: float) -> str:
    """
    Convert seconds to 'HH:MM:SS.s' string (one decimal place).
    将秒数转换为 'HH:MM:SS.s' 字符串（一位小数）。
    
    Args:
        seconds: 秒数（浮点数） (Seconds as float)
    Returns:
        格式化的时间字符串，如 "00:01:30.5" (Formatted time string, e.g., "00:01:30.5")
    
    Example:
        >>> seconds_to_hhmmss(90.5)
        '00:01:30.5'
    
    Java 类比：类似 String.format("%02d:%02d:%04.1f", h, m, s)。
    """
    h = int(seconds // 3600)  # 计算小时
    seconds %= 3600  # 剩余秒数
    m = int(seconds // 60)  # 计算分钟
    s = seconds % 60  # 剩余秒数
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def hhmmss_to_seconds(time_str: str, fps: float = 24.0) -> float:
    """
    Convert time string to seconds.
    将时间字符串转换为秒数。
    
    Supported formats:
    支持的格式：
    - HH:MM:SS or HH:MM:SS.mmm  (standard)
    - HH:MM:SS or HH:MM:SS.mmm  （标准格式）
    - HH:MM:SS:FF               (with frame number, uses fps parameter)
    - HH:MM:SS:FF               （带帧号，使用 fps 参数）
    - MM:SS or MM:SS.mmm
    - MM:SS or MM:SS.mmm
    - plain seconds as string
    - 纯秒数字符串
    
    Args:
        time_str: 时间字符串 (Time string)
        fps: 帧率，用于解析帧号（默认 24.0） (Frame rate for parsing frame numbers, default 24.0)
    Returns:
        秒数（浮点数） (Seconds as float)
    
    Example:
        >>> hhmmss_to_seconds("00:01:30.5")
        90.5
        >>> hhmmss_to_seconds("00:01:30:12", fps=24)
        90.5  # 30秒 + 12/24秒
    
    Java 类比：类似解析时间字符串并计算总秒数的工具方法。
    """
    if not time_str:
        return 0.0
    time_str = time_str.strip().replace(',', '.')  # 标准化：替换逗号为点号
    parts = time_str.split(':')  # 按冒号分割
    try:
        if len(parts) == 4:
            # HH:MM:SS:FF 格式（带帧号）
            h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            return h * 3600 + m * 60 + s + (f / fps)
        if len(parts) == 3:
            # HH:MM:SS 或 HH:MM:SS.mmm 格式
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            # MM:SS 或 MM:SS.mmm 格式
            return int(parts[0]) * 60 + float(parts[1])
        # 纯秒数
        return float(parts[0])
    except (ValueError, IndexError):
        return 0.0


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

def parse_srt_file(srt_path: str) -> List[Dict]:
    """
    Parse an SRT file into a list of subtitle dicts.
    解析 SRT 文件为字幕字典列表。
    
    Each dict has keys: start_sec, end_sec, speaker (or None), text.
    每个字典包含键：start_sec（起始秒数）、end_sec（结束秒数）、speaker（说话者，可能为 None）、text（文本）。
    
    This function:
    此函数：
    1. Reads the SRT file content
    1. 读取 SRT 文件内容
    2. Splits into blocks by blank lines
    2. 按空行分割成块
    3. Parses timecodes and extracts speaker info
    3. 解析时间码并提取说话者信息
    4. Converts timecodes to seconds
    4. 将时间码转换为秒数
    
    Args:
        srt_path: SRT 文件路径 (Path to SRT file)
    Returns:
        字幕字典列表 (List of subtitle dicts)
    
    Example SRT format:
    SRT 格式示例：
    ```
    1
    00:00:01,000 --> 00:00:04,000
    [Bruce Wayne] Hello, Alfred.
    ```
    
    Java 类比：类似解析字幕文件的工具方法，返回 List<Subtitle>。
    """
    # 检查文件是否存在
    if not os.path.exists(srt_path):
        return []

    # 读取文件内容
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    subtitles = []  # 字幕列表
    # 按空行分割成块
    for block in re.split(r'\n\s*\n', content.strip()):
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        try:
            # 解析时间码行
            time_match = re.match(
                r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})',
                lines[1]
            )
            if not time_match:
                continue
            # 合并所有文本行
            text = ' '.join(lines[2:]).strip()
            speaker = None  # 说话者
            # 尝试提取说话者信息（格式：[Name] Text）
            speaker_match = re.match(r'\[([^\]]+)\]\s*(.*)', text)
            if speaker_match:
                speaker = speaker_match.group(1)  # 提取说话者名称
                text = speaker_match.group(2).strip()  # 提取纯文本
            subtitles.append({
                'start_sec': hhmmss_to_seconds(time_match.group(1)),  # 起始秒数
                'end_sec': hhmmss_to_seconds(time_match.group(2)),  # 结束秒数
                'speaker': speaker,  # 说话者
                'text': text,  # 文本内容
            })
        except Exception:
            # 跳过解析失败的块
            continue
    return subtitles


def parse_srt_to_dict(srt_path: str) -> Dict[str, str]:
    """
    Parse an SRT file and return a mapping '{startSec}_{endSec}' -> 'subtitle text'.
    解析 SRT 文件并返回映射 '{startSec}_{endSec}' -> '字幕文本'。
    
    Timestamps are truncated to integer seconds.
    时间戳被截断为整数秒。
    
    This function:
    此函数：
    1. Reads the SRT file line by line
    1. 逐行读取 SRT 文件
    2. Parses timecodes and converts to integer seconds
    2. 解析时间码并转换为整数秒
    3. Builds a dictionary with time range keys
    3. 构建以时间范围为键的字典
    4. Merges duplicate time ranges
    4. 合并重复的时间范围
    
    Args:
        srt_path: SRT 文件路径 (Path to SRT file)
    Returns:
        时间范围到文本的映射字典 (Dict mapping time ranges to text)
    
    Example output:
    示例输出：
    ```python
    {
        "0_4": "Hello, Alfred.",
        "5_8": "Good evening, Master Wayne."
    }
    ```
    
    Java 类比：类似解析文件并构建 Map<String, String>。
    """
    # 检查文件是否存在
    if not os.path.isfile(srt_path):
        return {}

    result: Dict[str, str] = {}  # 结果字典
    # 读取所有行
    with open(srt_path, 'r', encoding='utf-8') as fh:
        lines = [l.rstrip('\n') for l in fh]

    idx = 0  # 当前行索引
    n = len(lines)  # 总行数
    # 逐行解析
    while idx < n:
        # 跳过序号行
        if lines[idx].strip().isdigit():
            idx += 1
        if idx >= n:
            break
        # 查找时间码行
        if '-->' not in lines[idx]:
            idx += 1
            continue
        # 解析起始和结束时间
        start_ts, end_ts = [t.strip() for t in lines[idx].split('-->')]
        start_sec = int(hhmmss_to_seconds(start_ts))  # 转换为整数秒
        end_sec = int(hhmmss_to_seconds(end_ts))  # 转换为整数秒
        idx += 1
        # 收集所有字幕文本行
        subtitle_lines: List[str] = []
        while idx < n and lines[idx].strip():
            subtitle_lines.append(lines[idx].strip())
            idx += 1
        # 合并文本行
        subtitle = ' '.join(subtitle_lines)
        key = f'{start_sec}_{end_sec}'  # 构建键
        # 如果键已存在，追加文本；否则创建新条目
        result[key] = result[key] + ' ' + subtitle if key in result else subtitle
        idx += 1
    return result


def get_subtitles_in_range(subtitles: List[Dict], start: float, end: float) -> List[Dict]:
    """
    Return subtitle entries that overlap [start, end].
    返回与 [start, end] 时间范围重叠的字幕条目。
    
    Args:
        subtitles: 字幕列表 (List of subtitle dicts)
        start: 起始时间（秒） (Start time in seconds)
        end: 结束时间（秒） (End time in seconds)
    Returns:
        重叠的字幕列表 (List of overlapping subtitles)
    
    Java 类比：类似过滤列表中满足时间区间条件的元素。
    """
    # 使用列表推导式过滤重叠的字幕
    return [s for s in subtitles if s['end_sec'] >= start and s['start_sec'] <= end]


def format_subtitles(subtitles: List[Dict]) -> str:
    """
    Format subtitle list as dialogue lines.
    将字幕列表格式化为对话行。
    
    Args:
        subtitles: 字幕列表 (List of subtitle dicts)
    Returns:
        格式化的对话字符串 (Formatted dialogue string)
    
    Example output:
    示例输出：
    ```
    [Bruce Wayne]: "Hello, Alfred."
    [Alfred]: "Good evening, Master Wayne."
    ```
    
    Java 类比：类似将对象列表转换为格式化字符串。
    """
    if not subtitles:
        return 'No dialogue.'  # 无对话
    # 格式化每一行：[说话者]: "文本"
    lines = [
        f"[{s.get('speaker', 'Unknown')}]: \"{s['text']}\""
        for s in subtitles if s.get('text')  # 只包含有文本的字幕
    ]
    return '\n'.join(lines) if lines else 'No dialogue.'


# ---------------------------------------------------------------------------
# Image encoding
# 图像编码
# ---------------------------------------------------------------------------

def pil_to_base64(img: Image.Image, quality: int = 85) -> str:
    """
    Encode a PIL Image to a base64 JPEG string.
    将 PIL 图像编码为 base64 JPEG 字符串。
    
    Args:
        img: PIL 图像对象 (PIL Image object)
        quality: JPEG 质量（1-100，默认 85） (JPEG quality 1-100, default 85)
    Returns:
        Base64 编码的 JPEG 字符串 (Base64 encoded JPEG string)
    
    Java 类比：类似将 BufferedImage 转换为 Base64 字符串。
    """
    buf = BytesIO()  # 内存缓冲区
    img.save(buf, format='JPEG', quality=quality)  # 保存为 JPEG 到缓冲区
    return base64.b64encode(buf.getvalue()).decode('utf-8')  # Base64 编码


def array_to_base64(frame: np.ndarray, quality: int = 80) -> str:
    """
    Encode a numpy uint8 RGB array (H, W, C) to a base64 JPEG string.
    将 numpy uint8 RGB 数组（H, W, C）编码为 base64 JPEG 字符串。
    
    Args:
        frame: NumPy 数组，形状为 (高度, 宽度, 通道) (NumPy array with shape (H, W, C))
        quality: JPEG 质量（1-100，默认 80） (JPEG quality 1-100, default 80)
    Returns:
        Base64 编码的 JPEG 字符串 (Base64 encoded JPEG string)
    
    Java 类比：类似将 byte[] 像素数据转换为 Base64 字符串。
    """
    # 从 NumPy 数组创建 PIL 图像，然后编码为 Base64
    return pil_to_base64(Image.fromarray(frame), quality=quality)


# ---------------------------------------------------------------------------
# Shot scene file I/O
# 镜头场景文件 I/O
# ---------------------------------------------------------------------------

def parse_shot_scenes(shot_scenes_path: str) -> List[Tuple[int, int]]:
    """
    Parse a shot_scenes.txt file into a list of (start_frame, end_frame) tuples.
    解析 shot_scenes.txt 文件为（起始帧，结束帧）元组列表。
    
    Args:
        shot_scenes_path: 场景文件路径 (Path to shot scenes file)
    Returns:
        场景元组列表 (List of (start_frame, end_frame) tuples)
    
    Example file format:
    文件格式示例：
    ```
    0 24
    25 50
    51 80
    ```
    
    Java 类比：类似读取文本文件并解析为 List<Pair<Integer, Integer>>。
    """
    scenes: List[Tuple[int, int]] = []  # 场景列表
    if not os.path.isfile(shot_scenes_path):
        return scenes
    # 逐行读取文件
    with open(shot_scenes_path, 'r') as f:
        for line in f:
            parts = line.strip().split()  # 分割每行
            if len(parts) >= 2:
                scenes.append((int(parts[0]), int(parts[1])))  # 添加场景元组
    return scenes


# ---------------------------------------------------------------------------
# Sorting
# 排序
# ---------------------------------------------------------------------------

def natural_sort_key(s: str) -> List:
    """
    Key function for natural (human) sort order (e.g. clip_2 before clip_10).
    自然（人类）排序顺序的键函数（例如 clip_2 在 clip_10 之前）。
    
    This function splits the string into numeric and non-numeric parts,
    converting numbers to integers for proper sorting.
    此函数将字符串分割为数字和非数字部分，
    将数字转换为整数以实现正确的排序。
    
    Args:
        s: 要排序的字符串 (String to sort)
    Returns:
        排序键列表 (List of sort keys)
    
    Example:
        >>> sorted(['clip_10', 'clip_2', 'clip_1'], key=natural_sort_key)
        ['clip_1', 'clip_2', 'clip_10']
    
    Java 类比：类似自定义 Comparator 实现自然排序。
    """
    # 按数字分割字符串，数字部分转换为整数，非数字部分转为小写
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


# ---------------------------------------------------------------------------
# Screenwriter helpers
# 编剧助手函数
# ---------------------------------------------------------------------------

def load_scene_summaries(scene_folder_path: str) -> tuple[str, int]:
    """
    Load scene_caption.scene_summary from all scene JSON files in a folder.
    从文件夹中的所有场景 JSON 文件加载 scene_caption.scene_summary。
    
    Skips non-usable scenes and scenes with importance_score < 3.
    跳过不可用的场景和重要性分数 < 3 的场景。
    
    This function:
    此函数：
    1. Finds all scene_*.json files in the folder
    1. 查找文件夹中所有的 scene_*.json 文件
    2. Sorts them by scene number
    2. 按场景编号排序
    3. Filters out unusable or low-importance scenes
    3. 过滤掉不可用或低重要性的场景
    4. Extracts and formats scene summaries
    4. 提取并格式化场景摘要
    5. Returns concatenated summaries and count
    5. 返回拼接的摘要和数量
    
    Args:
        scene_folder_path: 场景文件夹路径 (Path to scene folder)
    Returns:
        (拼接的场景摘要字符串, 加载的场景数量)
        (Concatenated scene summaries string, number of loaded scenes)
    
    Java 类比：类似读取多个 JSON 文件并聚合结果的批处理操作。
    """
    scene_summaries = []  # 场景摘要列表

    # 查找所有场景文件
    scene_files = [f for f in os.listdir(scene_folder_path)
                   if f.startswith('scene_') and f.endswith('.json')]

    def _scene_number(filename: str) -> int:
        """
        从文件名提取场景编号。
        Extract scene number from filename.
        """
        try:
            return int(filename.replace('scene_', '').replace('.json', ''))
        except ValueError:
            return float('inf')  # 无法解析的文件排到最后

    # 按场景编号排序
    scene_files.sort(key=_scene_number)

    # 遍历每个场景文件
    for filename in scene_files:
        filepath = os.path.join(scene_folder_path, filename)
        try:
            # 读取 JSON 文件
            with open(filepath, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)

            # 提取视频分析数据
            video_analysis = scene_data.get('video_analysis', {})
            scene_caption = video_analysis.get('scene_caption', {})
            scene_classification = scene_caption.get('scene_classification', {})

            # 检查场景是否可用
            if not scene_classification.get('is_usable', True):
                print(f"Skipping {filename}: not usable ({scene_classification.get('unusable_reason', 'unknown')})")
                continue

            # 检查重要性分数
            importance_score = scene_classification.get('importance_score', 5)
            if importance_score < 3:
                print(f"Skipping {filename}: importance_score ({importance_score}) below threshold (3)")
                continue

            # 提取场景摘要
            scene_summary = scene_caption.get('scene_summary', {})
            if not scene_summary:
                continue

            # 提取场景元数据
            scene_id = scene_data.get('scene_id', 'Unknown')
            time_range = scene_data.get('time_range', {})
            start_time = time_range.get('start_seconds', 'N/A')
            end_time = time_range.get('end_seconds', 'N/A')

            # 提取摘要字段
            narrative = scene_summary.get('narrative', '')  # 叙事
            key_event = scene_summary.get('key_event', '')  # 关键事件
            location = scene_summary.get('location', '')  # 地点
            time_state = scene_summary.get('time', '')  # 时间状态

            # 格式化场景摘要文本
            summary_text = (
                f"[Scene {scene_id}] ({start_time} - {end_time})\n"
                f"Location: {location}, Time: {time_state}\n"
                f"Key Event: {key_event}\n"
                f"Narrative: {narrative}\n"
            )
            scene_summaries.append(summary_text)

        except Exception as e:
            # 捕获异常，继续处理下一个文件
            print(f"Warning: Failed to read {filename}: {e}")
            continue

    total_scene_files = len(scene_files)  # 总文件数
    print(f"Loaded {len(scene_summaries)} scene summaries (out of {total_scene_files} files) from {scene_folder_path}")
    # 返回拼接的摘要和场景数量
    return "\n".join(scene_summaries), total_scene_files


def parse_structure_proposal_output(output: str) -> Optional[Dict]:
    """
    Parse structure proposal JSON from LLM output.
    从 LLM 输出中解析结构提案 JSON。
    
    Expected format::
    预期格式：
    ```json
    {
        "overall_theme": "...",
        "narrative_logic": "...",
        "emotion": "...",
        "related_scenes": [list of int scene indices]
    }
    ```
    
    This function tries multiple strategies:
    此函数尝试多种策略：
    1. Direct JSON parsing
    1. 直接 JSON 解析
    2. Strip markdown code fences
    2. 去除 Markdown 代码围栏
    3. Find first JSON object/array
    3. 查找第一个 JSON 对象/数组
    4. Regex extraction of {...} blocks
    4. 正则表达式提取 {...} 块
    
    Args:
        output: LLM 输出的文本 (LLM output text)
    Returns:
        解析后的字典，失败时返回 None (Parsed dict or None on failure)
    
    Java 类比：类似带有多种回退策略的 JSON 解析器。
    """
    def _validate(data) -> bool:
        """
        验证数据结构。
        Validate data structure.
        """
        if not isinstance(data, dict):
            return False
        # 检查必需字段
        for field in ('overall_theme', 'narrative_logic', 'emotion', 'related_scenes'):
            if field not in data:
                print(f"Warning: Missing required field '{field}'")
                return False
        # 检查 related_scenes 是否为列表
        if not isinstance(data['related_scenes'], list):
            print(f"Warning: 'related_scenes' must be a list")
            return False
        # 检查列表元素是否为整数
        for idx, scene_id in enumerate(data['related_scenes']):
            if not isinstance(scene_id, int):
                print(f"Warning: Scene index at position {idx} is not an integer: {scene_id}")
                return False
        return True

    # Direct parse - 直接解析
    try:
        result = json.loads(output)
        if _validate(result):
            return result
    except Exception:
        pass

    # Strip ```json ... ``` fences - 去除 Markdown 代码围栏
    m = re.compile(r"```(?:json)?\n(.*?)```", re.DOTALL | re.IGNORECASE).search(output)
    if m:
        try:
            result = json.loads(m.group(1))
            if _validate(result):
                return result
        except Exception:
            pass

    # Find first '{' or '[' - 查找第一个 JSON 起始符
    json_start = min((i for i in (output.find("{"), output.find("[")) if i != -1), default=None)
    if json_start is not None:
        try:
            result = json.loads(output[json_start:])
            if _validate(result):
                return result
        except Exception:
            pass

    # Last resort: any {...} block - 最后手段：任意 {...} 块
    for b in re.findall(r'({.*})', output, re.DOTALL):
        try:
            result = json.loads(b)
            if _validate(result):
                return result
        except Exception:
            continue

    print("parse_structure_proposal_output: all attempts failed.")
    print(output[:500])  # 打印前 500 个字符用于调试
    return None


def parse_shot_plan_output(output: str) -> Optional[Dict]:
    """
    Parse shot plan JSON from LLM output, stripping markdown fences if present.
    从 LLM 输出中解析镜头计划 JSON，如果存在则去除 Markdown 围栏。
    
    This function:
    此函数：
    1. Strips ```json ... ``` markdown wrappers
    1. 去除 ```json ... ``` Markdown 包装
    2. Parses the cleaned JSON
    2. 解析清理后的 JSON
    3. Returns None on failure
    3. 失败时返回 None
    
    Args:
        output: LLM 输出的文本 (LLM output text)
    Returns:
        解析后的字典，失败时返回 None (Parsed dict or None on failure)
    
    Java 类比：类似简单的 JSON 解析工具方法。
    """
    if not output:
        return None
    text = output.strip()  # 去除首尾空白
    # 尝试去除 Markdown 代码围栏
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()  # 提取代码块内容
    try:
        return json.loads(text)  # 解析 JSON
    except Exception:
        return None  # 解析失败
