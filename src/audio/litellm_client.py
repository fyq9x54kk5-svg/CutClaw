"""
Lightweight litellm wrapper for audio analysis via cloud API.
使用 litellm 封装的音频分析云 API 客户端。

Reads configuration from src/audio/.env:
从 src/audio/.env 读取配置：
  AUDIO_MODEL    - LiteLLM model string (e.g. openai/Qwen3-Omni-30B-A3B-Instruct)
  AUDIO_MODEL    - LiteLLM 模型字符串（例如 openai/Qwen3-Omni-30B-A3B-Instruct）
  AUDIO_API_KEY  - API key (use EMPTY for no auth)
  AUDIO_API_KEY  - API 密钥（无认证时使用 EMPTY）
  AUDIO_BASE_URL - Base URL for OpenAI-compatible endpoints
  AUDIO_BASE_URL - OpenAI 兼容端点的基础 URL

Java 类比：类似一个 HTTP 客户端封装类，用于调用远程 AI API。
"""

import os
import asyncio
import base64
import subprocess
import tempfile
from pathlib import Path
from typing import List

import litellm
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    from .. import config as project_config
except Exception:
    project_config = None

# Load .env from the same directory as this file
load_dotenv(Path(__file__).parent / ".env")


def _get_setting(config_key: str, env_key: str, default=None):
    """
    Read setting from src/config.py first, then fallback to environment/.env.
    首先从 src/config.py 读取配置，然后回退到环境变量/.env。
    
    This function implements a configuration priority chain:
    此函数实现配置优先级链：
    1. Project config (src/config.py)
    1. 项目配置（src/config.py）
    2. Environment variable
    2. 环境变量
    3. Default value
    3. 默认值
    
    Args:
        config_key: 项目配置中的键名 (Key name in project config)
        env_key: 环境变量名 (Environment variable name)
        default: 默认值 (Default value)
    Returns:
        配置值 (Configuration value)
    
    Java 类比：类似 ConfigurationProvider 模式，支持多层配置源。
    """
    # 首先尝试从项目配置获取
    if project_config is not None and hasattr(project_config, config_key):
        value = getattr(project_config, config_key)
        if value is not None and value != "":
            return value
    # 回退到环境变量
    env_value = os.getenv(env_key)
    if env_value is not None and env_value != "":
        return env_value
    # 返回默认值
    return default


AUDIO_MODEL = _get_setting(
    config_key="AUDIO_LITELLM_MODEL",
    env_key="AUDIO_MODEL",
    default="openai/Qwen3-Omni-30B-A3B-Instruct",
)
AUDIO_API_KEY = _get_setting(
    config_key="AUDIO_LITELLM_API_KEY",
    env_key="AUDIO_API_KEY",
    default="EMPTY",
) or "EMPTY"
AUDIO_BASE_URL = _get_setting(
    config_key="AUDIO_LITELLM_BASE_URL",
    env_key="AUDIO_BASE_URL",
    default=None,
)


def _audio_to_base64_mp3(audio_path: str) -> str:
    """
    Convert audio file to base64-encoded MP3 for cloud API submission.
    将音频文件转换为 base64 编码的 MP3，用于云 API 提交。
    
    This function:
    此函数：
    1. Creates a temporary MP3 file
    1. 创建临时 MP3 文件
    2. Uses ffmpeg to convert and compress audio
    2. 使用 ffmpeg 转换和压缩音频
    3. Reads and encodes to base64
    3. 读取并编码为 base64
    4. Cleans up temporary file
    4. 清理临时文件
    
    Args:
        audio_path: 音频文件路径 (Path to audio file)
    Returns:
        Base64 编码的 MP3 数据 (Base64 encoded MP3 data)
    
    Java 类比：类似使用 ProcessBuilder 调用 ffmpeg 进行音频转换。
    """
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        # 使用 ffmpeg 转换音频：单声道、16kHz 采样率、32kbps 比特率
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", "-ab", "32k", tmp_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # 读取并编码为 base64
        with open(tmp_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@retry(
    reraise=True,
    stop=stop_after_attempt(4),  # 1 次正常 + 最多 3 次重试
    wait=wait_exponential(multiplier=2, min=1, max=30),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda rs: print(
        f"[litellm_client] Retry {rs.attempt_number}/3 "
        f"after {rs.outcome.exception()} — sleeping {rs.next_action.sleep:.1f}s"
    ),
)
async def acall_audio_api(
    audio_path: str,
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
) -> str:
    """
    Async: Call the cloud audio API with a prompt about the given audio file.
    异步：使用提示词调用云音频 API 分析给定的音频文件。
    
    Audio is converted to MP3 once (not retried); the API call is retried up to 3
    times with exponential backoff (1s → 2s → 4s, capped at 30s) on any error.
    音频只转换一次为 MP3（不重试）；API 调用在任何错误时最多重试 3 次，
    使用指数退避（1s → 2s → 4s，上限 30s）。
    
    This function:
    此函数：
    1. Converts audio to base64 MP3 in a thread pool
    1. 在线程池中将音频转换为 base64 MP3
    2. Calls litellm.acompletion with audio + text
    2. 使用音频 + 文本调用 litellm.acompletion
    3. Returns model response text
    3. 返回模型响应文本
    
    Args:
        audio_path: 音频文件路径 (Path to audio file)
        prompt: 分析提示词 (Text prompt for analysis)
        temperature: 采样温度（0-1，默认 0.7） (Sampling temperature 0-1, default 0.7)
        top_p: Top-p 采样参数（默认 0.95） (Top-p sampling parameter, default 0.95)
        max_tokens: 最大生成 token 数（默认 4096） (Max tokens to generate, default 4096)
    Returns:
        模型的响应文本 (Response text from the model)
    
    Java 类比：类似 CompletableFuture + RetryTemplate 的异步 API 调用。
    """
    loop = asyncio.get_running_loop()  # 获取当前事件循环
    # ffmpeg conversion only runs once — not included in the retry scope
    # ffmpeg 转换只运行一次 - 不包含在重试范围内
    audio_b64 = await loop.run_in_executor(None, _audio_to_base64_mp3, audio_path)

    # 调用 LiteLLM API
    response = await litellm.acompletion(
        model=AUDIO_MODEL,  # 模型名称
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},  # 文本提示
                {"type": "image_url", "image_url": {"url": f"data:audio/mp3;base64,{audio_b64}"}},  # 音频数据
            ],
        }],
        temperature=temperature,  # 温度参数
        top_p=top_p,  # Top-p 参数
        max_tokens=max_tokens,  # 最大 token 数
        timeout=300,  # 超时时间（秒）
        api_key=AUDIO_API_KEY,  # API 密钥
        **({"api_base": AUDIO_BASE_URL} if AUDIO_BASE_URL else {}),  # 可选的基础 URL
    )

    return response.choices[0].message.content  # 返回响应内容


def call_audio_api(
    audio_path: str,
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
) -> str:
    """
    Sync wrapper around acall_audio_api.
    acall_audio_api 的同步包装器。
    
    This function creates a new event loop and runs the async function.
    此函数创建一个新的事件循环并运行异步函数。
    
    Args:
        audio_path: 音频文件路径 (Path to audio file)
        prompt: 分析提示词 (Text prompt for analysis)
        temperature: 采样温度（默认 0.7） (Sampling temperature, default 0.7)
        top_p: Top-p 采样参数（默认 0.95） (Top-p sampling parameter, default 0.95)
        max_tokens: 最大生成 token 数（默认 4096） (Max tokens to generate, default 4096)
    Returns:
        模型的响应文本 (Response text from the model)
    
    Java 类比：类似 CompletableFuture.get() 阻塞等待异步结果。
    """
    # 创建事件循环并运行异步函数
    return asyncio.run(acall_audio_api(audio_path, prompt, temperature, top_p, max_tokens))


async def acall_audio_api_batch(
    audio_paths: List[str],
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    max_concurrent: int = 5,
) -> List[str]:
    """
    Async: Call the cloud audio API concurrently for multiple audio files.
    异步：并发调用云音频 API 处理多个音频文件。
    
    Uses asyncio.gather with a semaphore to limit concurrent requests.
    Each request is retried automatically via the @retry decorator on acall_audio_api.
    使用 asyncio.gather 和信号量限制并发请求数。
    每个请求通过 acall_audio_api 上的 @retry 装饰器自动重试。
    
    This function:
    此函数：
    1. Creates a semaphore to control concurrency
    1. 创建信号量控制并发数
    2. Wraps each call with semaphore acquisition
    2. 用信号量获取包装每个调用
    3. Runs all calls concurrently with gather
    3. 使用 gather 并发运行所有调用
    4. Handles exceptions gracefully (returns empty string)
    4. 优雅地处理异常（返回空字符串）
    
    Args:
        audio_paths: 音频文件路径列表 (List of paths to audio files)
        prompt: 分析提示词 (Text prompt for analysis)
        temperature: 采样温度（默认 0.7） (Sampling temperature, default 0.7)
        top_p: Top-p 采样参数（默认 0.95） (Top-p sampling parameter, default 0.95)
        max_tokens: 最大生成 token 数（默认 4096） (Max tokens to generate, default 4096)
        max_concurrent: 最大并发请求数（默认 5） (Max simultaneous API requests, default 5)
    Returns:
        响应文本列表（与 audio_paths 顺序相同，失败时为空字符串）
        (List of response texts, same order as audio_paths, "" on error)
    
    Java 类比：类似 Semaphore + CompletableFuture.allOf() 的并发控制。
    """
    if not audio_paths:
        return []

    # 创建信号量限制并发数
    sem = asyncio.Semaphore(max_concurrent)

    async def _limited(path: str) -> str:
        """
        带信号量限制的包装函数。
        Wrapper function with semaphore limiting.
        """
        async with sem:  # 获取信号量
            return await acall_audio_api(path, prompt, temperature, top_p, max_tokens)

    # 并发执行所有请求
    results = await asyncio.gather(
        *[_limited(p) for p in audio_paths],
        return_exceptions=True,  # 返回异常而非抛出
    )

    # 处理结果
    processed = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # 记录失败的请求
            print(f"[litellm_client] Failed after retries: {audio_paths[i]}: {result}")
            processed.append("")  # 失败时返回空字符串
        else:
            processed.append(result)  # 成功时返回结果
    return processed


def call_audio_api_batch(
    audio_paths: List[str],
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    max_workers: int = 5,
) -> List[str]:
    """
    Sync wrapper around acall_audio_api_batch.
    acall_audio_api_batch 的同步包装器。
    
    This function creates a new event loop and runs the async batch function.
    此函数创建一个新的事件循环并运行异步批量函数。
    
    Args:
        audio_paths: 音频文件路径列表 (List of paths to audio files)
        prompt: 分析提示词 (Text prompt for analysis)
        temperature: 采样温度（默认 0.7） (Sampling temperature, default 0.7)
        top_p: Top-p 采样参数（默认 0.95） (Top-p sampling parameter, default 0.95)
        max_tokens: 最大生成 token 数（默认 4096） (Max tokens to generate, default 4096)
        max_workers: 最大并发请求数（默认 5） (Max simultaneous API requests, default 5)
    Returns:
        响应文本列表（与 audio_paths 顺序相同，失败时为空字符串）
        (List of response texts, same order as audio_paths, "" on error)
    
    Java 类比：类似 CompletableFuture.allOf().get() 阻塞等待批量异步结果。
    """
    # 创建事件循环并运行异步批量函数
    return asyncio.run(
        acall_audio_api_batch(audio_paths, prompt, temperature, top_p, max_tokens, max_workers)
    )
