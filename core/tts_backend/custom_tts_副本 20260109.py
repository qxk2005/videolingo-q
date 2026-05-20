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
            # 这里的命名保持不变
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

def normalize_zh_numbers(text):
    """
    🔢 数字归一化：将可能被误读为英文的数字符号强制转为中文汉字
    """
    # 映射表
    digits_map = str.maketrans("0123456789", "零一二三四五六七八九")
    
    # 1. 处理百分比: 10% -> 百分之十 / 1.5% -> 百分之一点五
    def replace_percent(match):
        num_str = match.group(1) # 获取数字部分
        # 如果是纯整数且小于100，尝试转成更自然的读法 (10 -> 十, 12 -> 十二)
        if "." not in num_str and len(num_str) <= 2:
            try:
                val = int(num_str)
                if val == 10: cn_str = "十"
                elif 10 < val < 20: cn_str = "十" + str(val % 10).translate(digits_map)
                elif val % 10 == 0: cn_str = str(val // 10).translate(digits_map) + "十"
                else: # 普通数字直译，为了稳妥可以直接用直译
                    cn_str = num_str.translate(digits_map)
            except:
                cn_str = num_str.translate(digits_map)
        else:
            # 小数或大数字，直接按位转 (1.2 -> 一点二)
            cn_str = num_str.translate(digits_map).replace(".", "点")
            
        return f"百分之{cn_str}"

    text = re.sub(r'(\d+(?:\.\d+)?)%', replace_percent, text)

    # 2. 处理小数: 1.2 -> 一点二
    def replace_decimal(match):
        # 将小数点替换为“点”，数字转为汉字
        return match.group(0).translate(digits_map).replace(".", "点")
    
    text = re.sub(r'\d+\.\d+', replace_decimal, text)

    return text

def clean_text_natural(text):
    """
    🌿 自然韵律 + 数字修正
    """
    if not text: return ""
    
    # 1. 基础清理
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[[^\]]+\]', '', text)
    
    # 2. 修复单词截断
    text = text.replace(" Ub ", " Ubuntu ").replace("Ub ", "Ubuntu ")
    
    # 3. 【新增】数字/符号强制中文归一化 (解决 10% -> ten percent 问题)
    text = normalize_zh_numbers(text)
    
    # 4. 标点标准化
    text = text.replace(",", "，").replace(".", "。").replace("!", "！").replace("?", "？")
    text = text.replace("，", "， ").replace("。", "。 ").replace("！", "！ ").replace("？", "？ ")
    
    return text.strip()

# --- 3. 主生成函数 ---
def custom_tts(text_list, output_file_path):
    output_path = Path(output_file_path).resolve()
    os.makedirs(output_path.parent, exist_ok=True)
    
    if output_path.exists():
        try: os.remove(output_path)
        except: pass

    mgr = ProjectResourceManager(output_path.parent.parent.parent)
    preprocessor = AudioPreprocessor(mgr)

    if not preprocessor.prepare_master_reference():
        raise RuntimeError("❌ 无法构建参考音频")

    ref_text_content = preprocessor.get_reference_text()

    full_text = "".join(text_list)
    # 使用包含数字修正的清洗函数
    cleaned_text = clean_text_natural(full_text)
    
    if not cleaned_text:
        sf.write(str(output_path), np.zeros(24000), 24000)
        return [str(output_path)]

    print(f"🚀 [F5] 生成: '{cleaned_text[:20]}...'")

    GENERIC_REF_TEXT = "This audio is used for voice cloning reference."

    try:
        generate(
            generation_text=cleaned_text,
            ref_audio_path=str(mgr.ref_audio_save_path),
            ref_audio_text=ref_text_content, 
            output_path=str(output_path),
            steps=32,
            speed=0.95 
        )
        
        is_failed = False
        if not output_path.exists(): is_failed = True
        elif output_path.stat().st_size < 20000: 
            is_failed = True
            print(f"⚠️ 生成异常，回退...")

        if is_failed:
            if output_path.exists(): os.remove(output_path)
            generate(
                generation_text=cleaned_text,
                ref_audio_path=str(mgr.ref_audio_save_path),
                ref_audio_text=GENERIC_REF_TEXT, 
                output_path=str(output_path),
                steps=32,
                speed=0.95
            )
            if not output_path.exists() or output_path.stat().st_size < 4000:
                raise RuntimeError("生成失败")

    except Exception as e:
        print(f"❌ 生成出错: {e}")
        raise e
    
    finally:
        gc.collect()

    return [str(output_path)]