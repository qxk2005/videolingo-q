import os
import sys
import shutil
import tempfile
import re
import random
import soundfile as sf
import numpy as np
import gc
from pathlib import Path
from pydub import AudioSegment, effects

# --- 1. 核心库检查 ---
try:
    from f5_tts_mlx.generate import generate
except ImportError:
    print("❌ Error: f5-tts-mlx not found.")
    raise ImportError("必须安装 f5-tts-mlx 才能运行此脚本")

# 尝试导入 Whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: openai-whisper not found. 将默认使用通用参考文本。")
    WHISPER_AVAILABLE = False

# --- 2. 资源管理类 ---
class ProjectResourceManager:
    def __init__(self, start_dir):
        self.start_path = Path(start_dir).resolve()
        self.project_root = None
        self.output_root = None
        self._locate_roots()
        
        if self.output_root:
            self.refers_dir = self.output_root / "audio" / "refers"
            self.ref_audio_save_path = self.output_root / "audio" / "master_ref_soothing.wav"
            self.ref_text_save_path = self.output_root / "audio" / "master_ref_soothing.txt"
        else:
            self.refers_dir = None
            self.ref_audio_save_path = Path("master_ref_soothing.wav")
            self.ref_text_save_path = Path("master_ref_soothing.txt")

    def _locate_roots(self):
        current = self.start_path
        for _ in range(6):
            if (current / "log" / "cleaned_chunks.json").exists():
                self.output_root = current; self.project_root = current.parent; return
            if current.name == "output":
                self.output_root = current; self.project_root = current.parent; return
            if current == current.parent: break
            current = current.parent

class AudioPreprocessor:
    def __init__(self, resource_manager):
        self.mgr = resource_manager

    def prepare_master_reference(self):
        """
        构建参考音频：5s-9s，带静音间隔，确保基调舒缓。
        """
        if self.mgr.ref_audio_save_path.exists() and self.mgr.ref_audio_save_path.stat().st_size > 50000:
            return True

        print("🔄 [F5-TTS] 正在构建'自然韵律'主参考音频...")

        if not self.mgr.refers_dir or not self.mgr.refers_dir.exists():
            return self._fallback_to_global_vocal()

        wav_files = sorted(list(self.mgr.refers_dir.glob("*.wav")), key=lambda x: str(x))
        if not wav_files: return self._fallback_to_global_vocal()

        try:
            combined_audio = AudioSegment.empty()
            silence_gap = AudioSegment.silent(duration=300) 
            combined_audio += AudioSegment.silent(duration=200)

            current_duration = 0
            MIN_DURATION = 5000 
            MAX_DURATION = 9000

            start_idx = max(0, len(wav_files) // 3)
            
            for i in range(start_idx, len(wav_files)):
                f_path = wav_files[i]
                segment = AudioSegment.from_file(str(f_path))
                
                if len(segment) > MAX_DURATION: segment = segment[:MAX_DURATION]
                if current_duration + len(segment) > MAX_DURATION + 1000: break
                
                combined_audio += segment + silence_gap
                current_duration += len(segment) + 300
                if current_duration > MIN_DURATION: break
            
            if current_duration < 2000:
                while len(combined_audio) < MIN_DURATION:
                    combined_audio += combined_audio + silence_gap

            combined_audio += AudioSegment.silent(duration=200)
            combined_audio = effects.normalize(combined_audio)
            combined_audio = combined_audio.set_channels(1).set_frame_rate(24000)
            
            self.mgr.ref_audio_save_path.parent.mkdir(parents=True, exist_ok=True)
            combined_audio.export(self.mgr.ref_audio_save_path, format="wav")
            
            del combined_audio, wav_files
            gc.collect()
            return True

        except Exception as e:
            print(f"⚠️ 构建参考音频失败: {e}")
            return self._fallback_to_global_vocal()

    def _fallback_to_global_vocal(self):
        p = self.mgr.output_root / "audio" / "vocal.mp3"
        if not p.exists(): p = self.mgr.output_root / "audio" / "vocal.wav"
        
        if p.exists():
            try:
                audio = AudioSegment.from_file(str(p))
                start = len(audio) // 3
                audio = audio[start : start + 6000]
                audio = effects.normalize(audio)
                audio = audio.set_channels(1).set_frame_rate(24000)
                audio.export(self.mgr.ref_audio_save_path, format="wav")
                del audio
                gc.collect()
                return True
            except: pass
        return False

    def get_reference_text(self):
        if self.mgr.ref_text_save_path.exists():
            with open(self.mgr.ref_text_save_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        
        fallback_text = "The voice in this audio will be used as the reference for synthesis."
        if not WHISPER_AVAILABLE: return fallback_text

        print("🤖 [Whisper] 正在识别主参考音频内容...")
        try:
            model = whisper.load_model("small")
            result = model.transcribe(str(self.mgr.ref_audio_save_path))
            text = result["text"].strip()
            del model, result
            gc.collect()
            
            if len(text) < 5: return fallback_text
            with open(self.mgr.ref_text_save_path, 'w', encoding='utf-8') as f:
                f.write(text)
            return text
        except: return fallback_text

# --- 3. 智能中文数字处理类 ---
class ZhNumberNormalizer:
    def __init__(self):
        self.digits = list("零一二三四五六七八九")
        self.units = ["", "十", "百", "千"]
        self.big_units = ["", "万", "亿"]
        self.roman_map = {
            'I': '一', 'II': '二', 'III': '三', 'IV': '四', 'V': '五',
            'VI': '六', 'VII': '七', 'VIII': '八', 'IX': '九', 'X': '十'
        }
        self.math_map = {
            '+': '加', '-': '减', '×': '乘', '÷': '除以', '=': '等于', '%': '百分之'
        }

    def _num2han(self, num_str):
        if not num_str.isdigit(): return num_str
        num = int(num_str)
        if num == 0: return "零"
        result = []
        num_str = str(num)
        parts = []
        while len(num_str) > 0:
            parts.append(num_str[-4:])
            num_str = num_str[:-4]
        parts = parts[::-1]
        for i, part in enumerate(parts):
            part_len = len(part)
            part_int = int(part)
            if part_int == 0:
                if i != len(parts) - 1 and len(parts) > 1: result.append("零")
                continue
            part_res = ""
            has_zero = False 
            for j, digit in enumerate(part):
                d = int(digit)
                pos = part_len - 1 - j 
                if d == 0:
                    has_zero = True
                else:
                    if has_zero:
                        part_res += "零"
                        has_zero = False
                    if not (part_len == 2 and j == 0 and d == 1 and i == 0 and len(parts) == 1):
                        part_res += self.digits[d]
                    part_res += self.units[pos]
            result.append(part_res)
            big_unit_idx = len(parts) - 1 - i
            if big_unit_idx > 0 and big_unit_idx < len(self.big_units):
                result.append(self.big_units[big_unit_idx])
        final_str = "".join(result)
        final_str = re.sub(r'零+', '零', final_str)
        final_str = final_str.strip('零')
        if final_str == "": return "零"
        return final_str

    def _replace_range(self, match):
        return f"{self._num2han(match.group(1))}到{self._num2han(match.group(2))}"

    def _replace_percent(self, match):
        num = match.group(1)
        if '.' in num:
            parts = num.split('.')
            return f"百分之{self._num2han(parts[0])}点{self._digits_only(parts[1])}"
        else:
            return f"百分之{self._num2han(num)}"

    def _replace_decimal(self, match):
        parts = match.group(0).split('.')
        return f"{self._num2han(parts[0])}点{self._digits_only(parts[1])}"

    def _digits_only(self, text):
        return text.translate(str.maketrans("0123456789", "零一二三四五六七八九"))

    def normalize(self, text):
        if not text: return ""
        def replace_roman(match): return self.roman_map.get(match.group(0), match.group(0))
        text = re.sub(r'\b(I|II|III|IV|V|VI|VII|VIII|IX|X)\b', replace_roman, text)
        for symbol, char in self.math_map.items():
            if symbol == '%': continue
            pattern = r'(?<=\d)\s*\\' + symbol + r'\s*(?=\d)'
            if symbol in ['+', '-', '=']: pattern = r'(?<=\d)\s*\\' + symbol + r'\s*(?=\d)'
            text = re.sub(pattern, f" {char} ", text)
        text = re.sub(r'(\d+)\s*[-~]\s*(\d+)', self._replace_range, text)
        text = re.sub(r'(\d+(?:\.\d+)?)%', self._replace_percent, text)
        def replace_id_year(match): return self._digits_only(match.group(0))
        text = re.sub(r'\b(19|20)\d{2}\b', replace_id_year, text)
        text = re.sub(r'\b\d{5,}\b', replace_id_year, text)
        text = re.sub(r'\d+\.\d+', self._replace_decimal, text)
        def replace_int(match): return self._num2han(match.group(0))
        text = re.sub(r'\d+', replace_int, text)
        return text

normalizer = ZhNumberNormalizer()

# --- 4. 增强版文本清洗 ---
def clean_text_natural(text):
    """
    🌿 终极清洗：数字标准化 + 字符白名单过滤
    防止 F5-TTS 因为特殊符号（如 emoji, 书名号, 省略号）生成静音或报错
    """
    if not text: return ""
    
    # 1. 基础清理
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[[^\]]+\]', '', text)
    text = text.replace(" Ub ", " Ubuntu ").replace("Ub ", "Ubuntu ")
    
    # 2. 数字/符号强制中文归一化
    text = normalizer.normalize(text)
    
    # 3. 标点归一化（将不安全的标点转为安全的）
    # 将中文书名号、括号、引号替换为空格或去掉，因为F5通常不会读这些，反而可能报错
    text = text.replace("《", " ").replace("》", " ")
    text = text.replace("【", " ").replace("】", " ")
    text = text.replace("“", " ").replace("”", " ")
    text = text.replace("‘", " ").replace("’", " ")
    text = text.replace("…", "，") # 省略号转逗号，表示停顿
    text = text.replace("—", "，") # 破折号转逗号
    text = text.replace(";", "，").replace("；", "，")
    text = text.replace(":", "：")

    # 4. 【关键】白名单过滤
    # 只保留：汉字(\u4e00-\u9fa5)、英文字母(a-zA-Z)、数字(0-9，虽然前面已转汉字，防漏)、基础标点
    # 这一步能彻底剔除 F5 模型“不认可”的奇怪字符
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？,.\?!：]', '', text)
    
    # 5. 最终标点优化
    text = text.replace(",", "，").replace(".", "。").replace("!", "！").replace("?", "？")
    text = text.replace("，", "， ").replace("。", "。 ").replace("！", "！ ").replace("？", "？ ")
    
    # 6. 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- 5. 主生成函数 ---
def custom_tts(text_list, output_file_path):
    output_path = Path(output_file_path).resolve()
    os.makedirs(output_path.parent, exist_ok=True)
    
    # 清理旧文件
    if output_path.exists():
        try: os.remove(output_path)
        except: pass

    mgr = ProjectResourceManager(output_path.parent.parent.parent)
    preprocessor = AudioPreprocessor(mgr)

    # 准备参考音频
    if not preprocessor.prepare_master_reference():
        # 如果连参考音频都做不出来，生成一个静音文件防止崩溃
        print("❌ 无法构建参考音频，生成静音占位。")
        sf.write(str(output_path), np.zeros(24000), 24000)
        return [str(output_path)]

    ref_text_content = preprocessor.get_reference_text()
    
    # 合并文本并清洗
    full_text = "".join(text_list)
    cleaned_text = clean_text_natural(full_text)
    
    # 安全检查：如果清洗后没字了（比如输入全是emoji），给一个默认文本
    if not cleaned_text or len(cleaned_text.strip()) == 0:
        print("⚠️ 警告：文本清洗后为空，使用静音占位。")
        sf.write(str(output_path), np.zeros(24000), 24000)
        return [str(output_path)]

    # 打印清洗后的文本，用于调试
    print(f"🚀 [F5] 输入文本: '{cleaned_text[:50]}...'")

    GENERIC_REF_TEXT = "This audio is used for voice cloning reference."

    try:
        # 第一次尝试生成
        generate(
            generation_text=cleaned_text,
            ref_audio_path=str(mgr.ref_audio_save_path),
            ref_audio_text=ref_text_content, 
            output_path=str(output_path),
            steps=32,
            speed=0.95 
        )
        
        # 质量检查
        is_failed = False
        if not output_path.exists(): 
            print("❌ 生成失败：文件未创建")
            is_failed = True
        elif output_path.stat().st_size < 10000: 
            # 文件过小（小于10KB），极大概率是生成了静音或头文件损坏
            print(f"⚠️ 生成异常：文件过小 ({output_path.stat().st_size} bytes)，尝试回退方案...")
            is_failed = True

        # 如果失败，使用通用参考文本重试（有时 Whisper 识别的参考文本太乱会导致生成失败）
        if is_failed:
            if output_path.exists(): os.remove(output_path)
            print("🔄 正在重试 (使用通用参考文本)...")
            generate(
                generation_text=cleaned_text,
                ref_audio_path=str(mgr.ref_audio_save_path),
                ref_audio_text=GENERIC_REF_TEXT, 
                output_path=str(output_path),
                steps=32,
                speed=0.95
            )
            
            # 最终检查
            if not output_path.exists() or output_path.stat().st_size < 4000:
                print("❌ 重试失败，写入静音。")
                sf.write(str(output_path), np.zeros(24000), 24000)

    except Exception as e:
        print(f"❌ F5-TTS 致命错误: {e}")
        # 发生异常时不中断流程，写入静音文件让后续步骤继续
        sf.write(str(output_path), np.zeros(24000), 24000)
    
    finally:
        gc.collect()

    return [str(output_path)]  