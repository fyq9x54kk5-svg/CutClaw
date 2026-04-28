import src.config as config  # 导入配置模块，类似 Java 的 import static
import os  # 操作系统接口模块，类似 Java 的 java.io.File
import argparse  # 命令行参数解析器，类似 Java 的 Apache Commons CLI
import time  # 时间模块，用于性能测量
import threading  # 多线程模块，类似 Java 的 java.lang.Thread

# 从子模块导入函数和类
# Python 的 import 类似 Java 的 import，但可以导入具体函数
from src.video.preprocess import decode_video_to_frames
from src.video.preprocess.asr import run_asr, assign_speakers_to_srt
from src.video.deconstruction.get_character import analyze_subtitles

from src.video.deconstruction.video_caption import process_video
from src.video.deconstruction.scene_merge import OptimizedSceneSegmenter, load_shots, save_scenes
from src.video.deconstruction.scene_analysis_video import SceneVideoAnalyzer

from src.audio.audio_caption_madmom import caption_audio_with_madmom_segments

def parse_config_overrides(unknown_args):
    """
    解析配置覆盖参数，格式为 --config.PARAM_NAME value
    
    参数:
        unknown_args: argparse 捕获的未知参数列表
    
    返回:
        None（直接修改 config 模块）
    
    Java 类比：类似于 Spring 的 @Value 注解或 Properties 覆盖
    """
    # 解析 CLI 参数如：
    # --config.VIDEO_FPS 2 --config.AUDIO_TOTAL_SHOTS 80
    # 并在运行时更新 src.config 的值（无需编辑 config.py）。
    # 这允许在不修改源代码的情况下调整配置。
    i = 0
    while i < len(unknown_args):  # 遍历所有未知参数
        arg = unknown_args[i]
        if arg.startswith('--config.'):  # 检查是否为配置覆盖参数
            param_name = arg[9:]  # 移除 '--config.' 前缀，获取参数名
            # 类似 Java 的 substring(9)
            
            # 检查是否有下一个参数且不是以 '--' 开头（即不是另一个选项）
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                value_str = unknown_args[i + 1]  # 获取值字符串

                # 根据现有配置值或从字符串推断自动检测类型
                # hasattr: 检查对象是否有某属性，类似 Java 的反射
                if hasattr(config, param_name):
                    original_value = getattr(config, param_name)  # 获取属性值
                    # Preserve original type（保留原始类型）
                    # isinstance: 类型检查，类似 Java 的 instanceof
                    if isinstance(original_value, bool):
                        # Python 布尔转换：将字符串转为布尔值
                        value = value_str.lower() in ('true', '1', 'yes')
                    elif isinstance(original_value, int):
                        value = int(value_str)  # 转为整数
                    elif isinstance(original_value, float):
                        value = float(value_str)  # 转为浮点数
                    else:
                        value = value_str  # 保持字符串
                else:
                    # 从字符串推断类型
                    try:
                        if '.' in value_str:  # 包含小数点，视为浮点数
                            value = float(value_str)
                        else:
                            value = int(value_str)  # 否则视为整数
                    except ValueError:  # 如果转换失败
                        if value_str.lower() in ('true', 'false'):
                            value = value_str.lower() == 'true'  # 布尔值
                        else:
                            value = value_str  # 保持字符串

                # setattr: 设置对象属性，类似 Java 反射的 field.set()
                setattr(config, param_name, value)
                print(f"✅ Config override: {param_name} = {value} (type: {type(value).__name__})")
                i += 2  # 跳过已处理的参数和值
            else:
                print(f"⚠️ Warning: --config.{param_name} specified but no value provided")
                i += 1
        else:
            print(f"⚠️ Warning: Unknown argument '{arg}' ignored")
            i += 1

def main():
    """
    主函数 - 端到端的 CLI 流水线
    
    Java 类比：public static void main(String[] args)
    但 Python 中需要 if __name__ == "__main__" 来保护
    """
    # local_run.py 是端到端的命令行主流水线：
    # 1）视频预处理（抽帧 + 镜头切分）
    # 2）并行执行 ASR / 视频拆解 / 音频分析
    # 3）通过 Screenwriter 生成 shot_plan
    # 4）通过 Editor 智能体生成 shot_point
    
    # argparse: 命令行参数解析器
    # 类似 Java 的 Apache Commons CLI 或 JCommander
    parser = argparse.ArgumentParser(description="Run VideoCaptioningAgent on a video.")
    
    # 添加命令行参数
    # add_argument: 定义一个参数，类似 Java CLI 库的 Option
    parser.add_argument("--Video_Path", help="The URL of the video to process.", default="Dataset/Video/Movie/La_La_Land.mkv")
    parser.add_argument("--Audio_Path", help="The URL of the video to process.", default="Dataset/Audio/Norman_fucking_rockwell.mp3")
    parser.add_argument("--Instruction", help="The Instruction to cutting the video.", default="Mia and Sebastian's relationship evolves through sweet to break moments.")
    parser.add_argument("--instruction_type", help="Type of instruction: 'object' for Object-centric or 'narrative' for Narrative-driven", default="object", choices=["object", "narrative"])
    parser.add_argument("--type", help="film or vlog", default="film")
    parser.add_argument("--SRT_Path", type=str,
                        help="Path to existing SRT file. Skips ASR transcription; diarization still runs to assign speakers.")

    # Parse known args and capture unknown args for config overrides
    # 解析已知参数并捕获未知参数用于配置覆盖
    # parse_known_args(): 返回 (known_args, unknown_args) 元组
    # Java 类比：类似 Apache CLI 的 parser.parse(args)
    args, unknown = parser.parse_known_args()

    # Apply config overrides
    # 应用配置覆盖
    parse_config_overrides(unknown)

    config.VIDEO_TYPE = args.type

    Video_Path = args.Video_Path
    Audio_Path = args.Audio_Path
    Instruction = args.Instruction
    instruction_type = args.instruction_type

    # 生成视频和音频的唯一 ID（用于文件命名）
    # os.path.basename: 获取文件名，类似 Java 的 new File(path).getName()
    # os.path.splitext: 分割文件名和扩展名，返回 (name, ext) 元组
    video_id = os.path.splitext(os.path.basename(Video_Path))[0].replace('.', '_').replace(' ', '_')
    audio_id = os.path.splitext(os.path.basename(Audio_Path))[0].replace('.', '_').replace(' ', '_')

    # Generate a safe filename from instruction
    # 从指令生成安全的文件名
    import re  # 正则表达式模块，类似 Java 的 java.util.regex
    import hashlib  # 哈希算法模块，类似 Java 的 MessageDigest
    
    # Create a short hash of the instruction for uniqueness
    # 创建指令的短哈希以确保唯一性
    # hashlib.md5(): MD5 哈希，hexdigest() 返回十六进制字符串
    # [:8] 取前8个字符，类似 Java 的 substring(0, 8)
    instruction_hash = hashlib.md5(Instruction.encode('utf-8')).hexdigest()[:8]
    
    # Create a more readable version (up to 50 characters, sanitized)
    # 创建更易读的版本（最多50个字符，已清理）
    # re.sub(): 正则替换，移除特殊字符
    instruction_safe = re.sub(r'[^\w\s-]', '', Instruction)[:50].strip().replace(' ', '_')
    
    # If instruction is too long or empty, use a more informative format
    # 如果指令太长或为空，使用更信息化的格式
    if len(instruction_safe) > 0:
        instruction_id = f"{instruction_safe}_{instruction_hash}"
    else:
        instruction_id = f"instruction_{instruction_hash}"

    # ===== All Path Definitions =====
    # ===== 统一路径定义 =====
    # Keep all intermediate and final artifact paths centralized so every stage
    # 将所有中间产物与最终输出路径集中定义，便于各阶段复用缓存
    # can reuse cache files (existing files are skipped in later checks).
    # （后续阶段会检测并跳过已存在文件）。
    
    # Raw video output
    # 原始视频输出路径
    output_path = os.path.join(config.VIDEO_DATABASE_FOLDER, "raw", f"{video_id}.mp4")

    # Video-related paths
    # 视频相关路径
    frames_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "frames")
    video_captions_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "captions")
    video_db_path = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "database.json")
    srt_path = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "subtitles.srt")
    srt_with_characters = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "subtitles_with_characters.srt")
    character_info_path = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "character_info.json")

    # Shot and scene paths
    # 镜头和场景路径
    shot_scenes_file = os.path.join(frames_dir, "shot_scenes.txt")
    caption_file = os.path.join(video_captions_dir, "captions.json")
    shots_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "captions", "ckpt")
    scenes_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "captions", "scenes")
    scenes_output = os.path.join(scenes_dir, "scene_0.json")
    scene_summaries_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "captions", "scene_summaries_video")

    # Audio-related paths
    # 音频相关路径
    audio_captions_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Audio', audio_id, "captions")
    audio_caption_file = os.path.join(audio_captions_dir, "captions.json")

    # Output paths (include instruction type and instruction ID for different editing tasks)
    # 输出路径（包含指令类型和指令ID，用于不同的剪辑任务）
    shot_plan_output_path = os.path.join(
        config.VIDEO_DATABASE_FOLDER,
        'Output',
        f"{video_id}_{audio_id}",
        f"shot_plan_{instruction_id}.json"
    )
    shot_point_output_path = os.path.join(
        config.VIDEO_DATABASE_FOLDER,
        'Output',
        f"{video_id}_{audio_id}",
        f"shot_point_{instruction_id}.json"
    )
    
    # 记录开始时间和各阶段耗时
    start_time = time.time()
    stage_times = {}

    print(f"\n{'='*80}")
    print(f"🎬 Starting VideoCuttingAgent Pipeline")
    print(f"📽️  Video: {Video_Path}")
    print(f"🎵 Audio: {Audio_Path}")
    print(f"📝 Instruction: {Instruction}")
    print(f"{'='*80}\n")

    # Step 1: Decode video to frames and perform shot detection.
    # 第 1 步：视频解码抽帧并执行镜头切分。
    # This is the base index for all downstream steps.
    # 这是后续所有阶段依赖的基础索引。
    print(f"🎞️ [Step 1] Extracting video frames format in {frames_dir}...")
    t0 = time.time()  # 记录开始时间，用于性能测量
    
    # decode_video_to_frames: 解码视频为帧并检测镜头边界
    # 返回 vr (video reader) 对象，用于后续读取视频帧
    vr = decode_video_to_frames(
        Video_Path,
        frames_dir,
        config.VIDEO_FPS,
        config.VIDEO_RESOLUTION,
        max_minutes=getattr(config, 'VIDEO_MAX_MINUTES', None),  # getattr: 获取属性，带默认值
        shot_detection_threshold=config.SHOT_DETECTION_THRESHOLD,
        shot_detection_min_scene_len=config.SHOT_DETECTION_MIN_SCENE_LEN,
        save_frames_to_disk=getattr(config, 'VIDEO_SAVE_DEBUG_FRAMES', False),
        image_format='jpg',
        jpeg_quality=80,
    )
    stage_times['shot_detection'] = time.time() - t0  # 计算耗时
    print(f"✅ [Step 1] Shot detection completed in {stage_times['shot_detection']:.1f}s")


    # Run heavy preprocessing in parallel. Each thread records its own error;
    # 重计算预处理任务并行执行。每个线程独立记录错误；
    # any thread failure aborts the pipeline after all joins.
    # 任一线程失败会在 join 完成后使流水线整体失败。
    
    # thread_errors: 字典，用于存储各线程的错误信息
    # Java 类比：Map<String, Exception> threadErrors = new ConcurrentHashMap<>();
    thread_errors = {}
    
    # threading.Event(): 线程同步事件，类似 Java 的 CountDownLatch(1)
    # used only for film type（仅用于电影类型）
    asr_done_event = threading.Event()

    def run_asr_and_character_id():
        """
        Thread A: ASR + Character ID (film only). Sets asr_done_event when complete.
        线程 A：ASR + 角色识别（仅电影模式）。完成后设置 asr_done_event。
        
        Java 类比：这是一个 Runnable 任务，在单独的线程中执行
        """
        # Film mode depends on subtitle/character information for better scene
        # understanding and later shot selection constraints.
        # 电影模式依赖字幕/角色信息以更好地理解场景和后续的镜头选择约束。
        try:
            t0 = time.time()
            if args.type != "vlog":
                if args.SRT_Path is not None:
                    print(f"🔤 [Thread A: ASR] External SRT provided, skipping ASR transcription: {args.SRT_Path}")
                    if not os.path.exists(srt_path):
                        enable_diarization = getattr(config, 'ASR_ENABLE_DIARIZATION', False)
                        if enable_diarization:
                            from src.video.preprocess.asr import extract_audio_mp3_16k
                            extracted_audio_path = os.path.join(frames_dir, "audio_16k_mono.mp3")
                            if not os.path.exists(extracted_audio_path):
                                print("[Thread A: ASR] 🔊 Extracting audio for diarization...")
                                extract_audio_mp3_16k(Video_Path, extracted_audio_path)
                            assign_speakers_to_srt(
                                srt_path=args.SRT_Path,
                                audio_path=extracted_audio_path,
                                output_srt_path=srt_path,
                                device=config.ASR_DEVICE,
                            )
                        else:
                            import shutil
                            shutil.copy(args.SRT_Path, srt_path)
                            print(f"[Thread A: ASR] 📋 Diarization disabled, copied SRT to {srt_path}")
                    else:
                        print(f"[Thread A: ASR] ⏭️ SRT already exists at {srt_path}, skipping.")
                else:
                    print("[Thread A: ASR] 🎙️ Running ASR to generate subtitles...")
                    run_asr(
                        video_path=Video_Path,
                        output_dir=frames_dir,
                        srt_path=srt_path,
                        backend=config.ASR_BACKEND,
                        asr_device=config.ASR_DEVICE,
                        asr_language=config.ASR_LANGUAGE,
                        whisper_cpp_model_name=getattr(config, 'ASR_WHISPER_CPP_MODEL', 'base.en'),
                        whisper_cpp_n_threads=getattr(config, 'ASR_WHISPER_CPP_N_THREADS', 4),
                        litellm_model=getattr(config, 'ASR_LITELLM_MODEL', None),
                        litellm_api_key=getattr(config, 'ASR_LITELLM_API_KEY', None),
                        litellm_api_base=getattr(config, 'ASR_LITELLM_API_BASE', None),
                        litellm_max_segment_mb=getattr(config, 'ASR_LITELLM_MAX_SEGMENT_MB', 25.0),
                        litellm_batch_size=getattr(config, 'ASR_LITELLM_BATCH_SIZE', 8),
                        litellm_debug_dir=os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "subtitles_segments"),
                    )
                print("[Thread A: ASR] ✅ ASR/SRT step completed.")
                if os.path.exists(srt_path) and not os.path.exists(character_info_path):
                    print("[Thread A: CharID] 👥 Analyzing subtitles to identify characters...")
                    video_name = video_id.replace('_', ' ')  # 将下划线替换为空格，恢复原始名称
                    
                    # analyze_subtitles: 分析字幕识别角色
                    # 返回 speaker_mapping（说话人映射）和 _character_info（角色信息）
                    speaker_mapping, _character_info = analyze_subtitles(
                        srt_path=srt_path,
                        movie_name=video_name,
                        output_dir=os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id),
                        use_full_subtitles=True,
                        model=config.VIDEO_ANALYSIS_MODEL,
                        api_base=config.VIDEO_ANALYSIS_ENDPOINT,
                        api_key=config.VIDEO_ANALYSIS_API_KEY,
                        max_tokens=config.VIDEO_ANALYSIS_MODEL_MAX_TOKEN,
                    )
                    print(f"[Thread A: CharID] ✅ Character identification completed. Found {len(speaker_mapping)} characters.")
                elif os.path.exists(character_info_path):
                    print(f"[Thread A: CharID] ⏭️ Character info already exists at {character_info_path}.")
                else:
                    print(f"[Thread A: CharID] ⚠️ Subtitle file not found at {srt_path}, skipping character identification.")
            else:
                print("[Thread A] ⏭️ Skipping ASR/character ID for vlog type.")
            
            # 记录 ASR 和角色识别的耗时
            stage_times['asr_character_id'] = time.time() - t0
            print(f"[Thread A] ✨ Completed in {stage_times['asr_character_id']:.1f}s")
        except Exception as e:
            thread_errors['asr'] = e  # 存储错误到字典
            print(f"[Thread A] ❌ ERROR: {e}")
        finally:
            # always signal, even on error, so Thread B doesn't hang
            # 总是发送信号，即使出错，避免线程 B 永久等待
            # Java 类比：countDownLatch.countDown()
            asr_done_event.set()

    def run_video_captioning():
        """
        Thread B: Video Captioning → Scene Merge → Scene Analysis.
        线程 B：视频字幕生成 → 场景合并 → 场景分析。
        
        For film: waits for ASR to complete first. For vlog: starts immediately.
        电影模式：等待 ASR 完成后开始。Vlog 模式：立即开始。
        
        Java 类比：另一个 Runnable 任务，依赖于 Thread A 的完成
        """
        try:
            t0 = time.time()
            if args.type != "vlog":
                print("[Thread B: Video] ⏳ Waiting for ASR/Character ID to complete...")
                # asr_done_event.wait(): 阻塞等待，直到事件被设置
                # Java 类比：countDownLatch.await()
                asr_done_event.wait()
                if 'asr' in thread_errors:
                    raise RuntimeError("ASR failed, cannot proceed with video captioning")

            # 检查字幕文件是否已存在，避免重复处理（缓存机制）
            if not os.path.exists(caption_file):
                print("[Thread B: Video] 🎬 Processing video to get captions...")
                if args.type == "vlog":
                    subtitle_to_use = None
                    print("[Thread B: Video] 🎬 Processing vlog without subtitles.")
                else:
                    # 优先使用带角色信息的字幕，否则使用普通字幕
                    subtitle_to_use = srt_with_characters if os.path.exists(srt_with_characters) else srt_path
                    print(f"[Thread B: Video] 🎬 Processing video with subtitle file: {subtitle_to_use}")
                
                # process_video: 处理视频生成字幕
                process_video(
                    video=vr,
                    output_caption_folder=video_captions_dir,
                    subtitle_file_path=subtitle_to_use,
                    long_shots_path=shot_scenes_file if os.path.exists(shot_scenes_file) else None,
                    video_type=args.type,
                    frames_dir=frames_dir,
                )
            else:
                print(f"[Thread B: Video] ⏭️ Captions already exist at {caption_file}.")

            # Scene Merge: merge fine-grained shots into scene units, which are
            # 场景合并：将细粒度镜头合并为场景单元，
            # later referenced by Screenwriter/Editor as semantic neighborhoods.
            # 供后续 Screenwriter/Editor 作为语义邻域检索。
            
            # 检查是否需要执行场景合并（缓存机制）
            if os.path.exists(shots_dir) and not os.path.exists(scenes_output):
                print("[Thread B: Scene] 🧩 Merging shots into scenes...")
                
                # load_shots: 加载所有镜头数据
                shots = load_shots(shots_dir)
                print(f"[Thread B: Scene] 📄 Loaded {len(shots)} shots")
                
                if shots:
                    # OptimizedSceneSegmenter: 优化的场景分割器
                    segmenter = OptimizedSceneSegmenter()
                    
                    # segment(): 执行场景合并算法
                    # threshold: 相似度阈值，低于此值的镜头会被分到不同场景
                    # max_scene_duration_secs: 单个场景的最大持续时间
                    merged_scenes = segmenter.segment(
                        shots,
                        threshold=config.SCENE_SIMILARITY_THRESHOLD if hasattr(config, 'SCENE_SIMILARITY_THRESHOLD') else 0.5,
                        max_scene_duration_secs=config.MAX_SCENE_DURATION_SECS if hasattr(config, 'MAX_SCENE_DURATION_SECS') else 300
                    )
                    print(f"[Thread B: Scene] ✅ Merged {len(shots)} shots into {len(merged_scenes)} scenes")
                    
                    # save_scenes: 保存合并后的场景到文件
                    save_scenes(merged_scenes, scenes_dir)
                    print(f"[Thread B: Scene] 💾 Scenes saved to {scenes_dir}")
                else:
                    print("[Thread B: Scene] ⚠️ No shots found to merge")
            elif os.path.exists(scenes_output):
                print(f"[Thread B: Scene] ⏭️ Scenes already exist at {scenes_dir}")
            else:
                print(f"[Thread B: Scene] ⚠️ Shots directory not found at {shots_dir}, skipping scene merge")

            # Scene Analysis: summarize each merged scene for retrieval and
            # 场景分析：为每个合并后的场景生成摘要，
            # planning in the downstream agent loop.
            # 支持后续智能体循环中的检索与规划。
            
            # 检查是否需要执行场景分析（缓存机制）
            if os.path.exists(scenes_dir) and os.path.exists(scenes_output):
                if args.type == "vlog":
                    subtitle_to_use = None
                    print("[Thread B: Analysis] 🔍 Analyzing scenes without subtitles for vlog.")
                else:
                    # 优先使用带角色信息的字幕
                    subtitle_to_use = srt_with_characters if os.path.exists(srt_with_characters) else srt_path
                
                # SceneVideoAnalyzer: 场景视频分析器
                analyzer = SceneVideoAnalyzer(vr=vr, subtitle_file=subtitle_to_use)
                
                # analyze_scenes_dir: 并行分析所有场景
                # max_workers: 并行工作线程数，类似 Java 的 ThreadPoolExecutor
                result = analyzer.analyze_scenes_dir(
                    scenes_dir=scenes_dir,
                    output_dir=scene_summaries_dir,
                    max_workers=config.CAPTION_BATCH_SIZE,
                    overwrite=False,  # 不覆盖已有结果
                )
                if result["status"] == "invalid":
                    print(f"[Thread B: Analysis] ❌ {result['errors'][0]}")
                elif result["status"] == "skipped":
                    # 如果所有场景都已分析，跳过
                    print(f"[Thread B: Analysis] ⏭️ Scene summaries already exist ({result['already_analyzed']} files)")
                else:
                    print(f"[Thread B: Analysis] ✅ Scene analysis completed: {result['success']} success, {result['skipped']} skipped")
                    if result["errors"]:
                        print(f"[Thread B: Analysis] ⚠️ Errors: {len(result['errors'])}")
                        for e in result["errors"][:3]:  # 只显示前3个错误
                            print(f"   > {e}")
            else:
                print(f"[Thread B: Analysis] ⚠️ Scenes directory not found or empty at {scenes_dir}, skipping scene analysis")
            
            # 记录视频处理的总耗时
            stage_times['video_captioning'] = time.time() - t0
            print(f"[Thread B] ✨ Completed in {stage_times['video_captioning']:.1f}s")
        except Exception as e:
            thread_errors['video'] = e  # 存储错误
            print(f"[Thread B] ❌ ERROR: {e}")

    def run_audio_analysis():
        """
        Thread C: Audio Analysis — fully independent of video pipeline.
        线程 C：音频分析 - 完全独立于视频流水线。
        
        Produces music structure + keypoints that define target pacing and
        该阶段产出音乐结构与关键点，用于定义目标节奏与
        timing constraints for shot planning/selection.
        镜头规划/选择时长约束。
        
        Java 类比：独立的 Runnable 任务，不依赖其他线程
        """
        try:
            t0 = time.time()
            
            # 检查音频字幕是否已存在（缓存机制）
            if not os.path.exists(audio_caption_file):
                print("[Thread C: Audio] 🎵 Processing audio to get captions...")
                
                # caption_audio_with_madmom_segments: 使用 madmom 分析音频并生成字幕
                # 这是一个复杂的函数，包含多个步骤：
                # 1. 音乐结构分析（Level-1）
                # 2. 关键点检测（downbeat/pitch/mel_energy）
                # 3. 关键点后处理（去噪/稀疏化）
                # 4. 关键点字幕生成（Level-2）
                caption_audio_with_madmom_segments(
                    audio_path=Audio_Path,
                    output_path=audio_caption_file,
                    max_tokens=config.AUDIO_KEYPOINT_MAX_TOKENS,
                    temperature=config.AUDIO_KEYPOINT_TEMPERATURE,
                    top_p=config.AUDIO_KEYPOINT_TOP_P,
                    max_workers=config.AUDIO_BATCH_SIZE,  # 并行工作线程数
                    detection_methods=config.AUDIO_DETECTION_METHODS,  # 检测方法列表
                    beats_per_bar=[config.AUDIO_BEATS_PER_BAR],  # 每小节拍数
                    min_bpm=config.AUDIO_MIN_BPM,
                    max_bpm=config.AUDIO_MAX_BPM,
                    pitch_tolerance=config.AUDIO_PITCH_TOLERANCE,
                    pitch_threshold=config.AUDIO_PITCH_THRESHOLD,
                    pitch_min_distance=config.AUDIO_PITCH_MIN_DISTANCE,
                    pitch_nms_method=config.AUDIO_PITCH_NMS_METHOD,
                    pitch_max_points=config.AUDIO_PITCH_MAX_POINTS,
                    mel_win_s=config.AUDIO_MEL_WIN_S,
                    mel_n_filters=config.AUDIO_MEL_N_FILTERS,
                    mel_threshold_ratio=config.AUDIO_MEL_THRESHOLD_RATIO,
                    mel_min_distance=config.AUDIO_MEL_MIN_DISTANCE,
                    mel_nms_method=config.AUDIO_MEL_NMS_METHOD,
                    mel_max_points=config.AUDIO_MEL_MAX_POINTS,
                    merge_close=config.AUDIO_MERGE_CLOSE,
                    min_interval=config.AUDIO_MIN_INTERVAL,
                    top_k_keypoints=config.AUDIO_TOP_K,
                    energy_percentile=config.AUDIO_ENERGY_PERCENTILE,
                    min_segment_duration=config.AUDIO_MIN_SEGMENT_DURATION,
                    max_segment_duration=config.AUDIO_MAX_SEGMENT_DURATION,
                    use_stage1_sections=config.AUDIO_USE_STAGE1_SECTIONS,
                    section_min_interval=config.AUDIO_SECTION_MIN_INTERVAL,
                )
            else:
                print(f"[Thread C: Audio] ⏭️ Audio captions already exist at {audio_caption_file}.")
            
            # 记录音频分析的耗时
            stage_times['audio_analysis'] = time.time() - t0
            print(f"[Thread C] ✨ Completed in {stage_times['audio_analysis']:.1f}s")
        except Exception as e:
            thread_errors['audio'] = e  # 存储错误
            print(f"[Thread C] ❌ ERROR: {e}")

    # Launch three preprocessing branches concurrently to reduce wall-clock time.
    # 并发启动三个预处理分支，以降低整体耗时。
    
    # threading.Thread: 创建线程对象
    # target: 线程执行的函数，类似 Java 的 Runnable
    # name: 线程名称，用于调试
    # daemon=False: 非守护线程，主线程会等待这些线程完成
    # Java 类比：new Thread(runnable, "ThreadName")
    thread_a = threading.Thread(target=run_asr_and_character_id, name="ASR-CharID",    daemon=False)
    thread_b = threading.Thread(target=run_video_captioning,     name="VideoCaptions", daemon=False)
    thread_c = threading.Thread(target=run_audio_analysis,       name="AudioAnalysis", daemon=False)

    # 启动所有线程（并行执行）
    # start(): 启动线程，类似 Java 的 thread.start()
    thread_a.start()
    thread_b.start()
    thread_c.start()

    # 等待所有线程完成（阻塞）
    # join(): 等待线程结束，类似 Java 的 thread.join()
    thread_a.join()
    thread_b.join()
    thread_c.join()

    # 检查是否有线程失败
    if thread_errors:
        for name, err in thread_errors.items():
            print(f"❌ Pipeline stage '{name}' failed: {err}")
        raise RuntimeError(f"Pipeline failed in stages: {list(thread_errors.keys())}")

    print("\n🚀 All parallel stages completed.")

    end_time = time.time()
    print(f"\n{'='*60}")
    print(f"⏱️  Stage Timing Summary:")
    for stage, elapsed in stage_times.items():
        print(f"  {stage:<30} {elapsed:>8.1f}s")
    print(f"  {'total (wall clock)':<30} {end_time - start_time:>8.1f}s")
    print(f"{'='*60}\n")

    



    # Step 5: Run Screenwriter to generate shot_plan.
    # 第 5 步：运行 Screenwriter 生成 shot_plan。
    # shot_plan answers "what to cut" (semantics/emotion/target durations),
    # shot_plan 解决“剪什么”（语义/情绪/目标时长），
    # not exact frame ranges yet.
    # 还不包含精确时间戳范围。
    
    # 检查场景摘要和音频字幕是否都已生成
    if os.path.exists(scene_summaries_dir) and os.path.exists(audio_caption_file):
        print("\n" + "="*80)
        if os.path.exists(shot_plan_output_path):
            print("✍️  Running Screenwriter to validate/complete existing shot plan...")
            print(f"📄 Existing shot plan detected: {shot_plan_output_path}")
        else:
            print("✍️  Running Screenwriter to generate shot plan...")
        print("="*80)

        # 导入 Screenwriter 智能体
        from src.Screenwriter_scene_short import Screenwriter

        # Create output directory
        # os.makedirs: 创建目录，exist_ok=True 表示如果目录已存在不报错
        # Java 类比：new File(dir).mkdirs()
        os.makedirs(os.path.dirname(shot_plan_output_path), exist_ok=True)

        # Initialize Screenwriter agent
        # Screenwriter: 编剧智能体，负责规划剪辑内容
        screenwriter = Screenwriter(
            video_scene_path=scene_summaries_dir,
            audio_caption_path=audio_caption_file,
            output_path=shot_plan_output_path,
            video_path=Video_Path,
            subtitle_path=srt_with_characters if config.VIDEO_TYPE == "film" and os.path.exists(srt_with_characters) else None,
            main_character=config.MAIN_CHARACTER_NAME if config.MAIN_CHARACTER_NAME else None,
            max_iterations=20,  # 最大迭代次数
        )

        # Run the screenwriter with the Instruction
        print(f"📝 Instruction: '{Instruction}'")
        t0 = time.time()
        _shot_plan = screenwriter.run(Instruction)  # 执行编剧智能体
        stage_times['screenwriter'] = time.time() - t0

        print(f"\n{'='*80}")
        print(f"✅ Shot plan generated successfully in {stage_times['screenwriter']:.1f}s!")
        print(f"💾 Output saved to: {shot_plan_output_path}")
        print(f"{'='*80}\n")
w
    # Step 6: Run EditorCoreAgent to select exact clip ranges based on shot_plan.
    # 第 6 步：基于 shot_plan 运行 EditorCoreAgent 选择精确片段范围。
    # shot_point answers "where exactly to cut" (timestamps in source video).
    # shot_point 解决“具体从哪里剪”（源视频中的时间戳）。
    
    # Check if we have all required files for core agent
    # 检查是否有所需的文件来运行核心智能体
    if os.path.exists(scene_summaries_dir) and os.path.exists(audio_caption_file) and os.path.exists(shot_plan_output_path):
        print("\n" + "="*80)
        print("✂️  Running EditorCoreAgent to select video clips...")
        print("="*80)

        # 根据视频类型导入不同的 EditorCoreAgent
        if config.VIDEO_TYPE == "film":
            from src.core import EditorCoreAgent, ParallelShotOrchestrator
        elif config.VIDEO_TYPE == "vlog":
            from src.core_vlog import EditorCoreAgent

        # Create output directory
        os.makedirs(os.path.dirname(shot_point_output_path), exist_ok=True)

        # 获取最大迭代次数配置
        max_iterations = config.AGENT_MAX_ITERATIONS if hasattr(config, 'AGENT_MAX_ITERATIONS') else 20
        
        # 判断是否启用并行镜头选择（仅电影模式）
        use_parallel_shot = (
            config.VIDEO_TYPE == "film" and
            getattr(config, "PARALLEL_SHOT_ENABLED", True)
        )

        print(f"🚀 Running editor agent with instruction: '{Instruction}'")
        print(f"📂 Using shot plan from: {shot_plan_output_path}")

        if use_parallel_shot:
            # 并行模式：使用 ParallelShotOrchestrator
            max_workers = getattr(config, "PARALLEL_SHOT_MAX_WORKERS", 4)
            max_reruns = getattr(config, "PARALLEL_SHOT_MAX_RERUNS", 2)
            print(f"⚡ Parallel mode enabled (workers: {max_workers}, max_reruns: {max_reruns})")
            
            # ParallelShotOrchestrator: 并行镜头协调器
            orchestrator = ParallelShotOrchestrator(
                video_caption_path=caption_file,
                video_scene_path=scene_summaries_dir,
                audio_caption_path=audio_caption_file,
                output_path=shot_point_output_path,
                max_iterations=max_iterations,
                video_path=Video_Path,
                frame_folder_path=frames_dir,
                transcript_path=srt_with_characters if os.path.exists(srt_with_characters) else srt_path,
                max_workers=max_workers,  # 并行工作线程数
                max_reruns=max_reruns,    # 最大重跑次数
            )
            _results = orchestrator.run_parallel(shot_plan_path=shot_plan_output_path)
            print(f"✅ Parallel mode completed, selected {len(_results)} shots.")
        else:
            # 串行模式：使用 EditorCoreAgent
            print("🚶 Sequential mode enabled (EditorCoreAgent.run).")
            editor_agent = EditorCoreAgent(
                video_caption_path=caption_file,
                video_scene_path=scene_summaries_dir,
                audio_caption_path=audio_caption_file,
                output_path=shot_point_output_path,
                max_iterations=max_iterations,
                video_path=Video_Path,
                video_reader=vr.get("video_reader") if isinstance(vr, dict) else vr,
                frame_folder_path=frames_dir,
                transcript_path=srt_with_characters if os.path.exists(srt_with_characters) else srt_path
            )
            _messages = editor_agent.run(shot_plan_path=shot_plan_output_path)

        print(f"\n{'='*80}")
        print(f"🎉 Video clip selection completed!")
        print(f"💾 Output saved to: {shot_point_output_path}")
        print(f"{'='*80}\n")
    else:
        print("\n" + "="*80)
        print("❌ Cannot run EditorCoreAgent - missing required files:")
        if not os.path.exists(scene_summaries_dir):
            print(f"  ❌ Scene summaries directory not found at {scene_summaries_dir}")
        if not os.path.exists(audio_caption_file):
            print(f"  ❌ Audio caption file not found at {audio_caption_file}")
        if not os.path.exists(shot_plan_output_path):
            print(f"  ❌ Shot plan file not found at {shot_plan_output_path}")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
