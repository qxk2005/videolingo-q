import numpy as np
import soundfile as sf
import os
import sys
import tempfile
import traceback
import warnings

warnings.filterwarnings("ignore")

try:
    import librosa
    import whisper
    from f5_tts_mlx.generate import generate
except ImportError as e:
    sys.exit(f"❌ 缺少库: {e}")

def ensure_24k_audio(input_path):
    print(f"🔄 [预处理] 读取并重采样: {os.path.basename(input_path)}")
    try:
        audio, _ = librosa.load(input_path, sr=24000, mono=True)
        # 裁剪前 5 秒
        duration = 5
        if len(audio) > 24000 * duration:
            audio = audio[:24000 * duration]
            print(f"✂️ 已自动裁剪至前 {duration} 秒")
        
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, audio, 24000)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        print(f"❌ 音频处理出错: {e}")
        return None

def auto_transcribe(audio_path):
    print("🤖 正在调用 Whisper 识别参考音频内容...")
    try:
        model = whisper.load_model("small")
        result = model.transcribe(audio_path)
        return result["text"].strip()
    except Exception as e:
        print(f"❌ Whisper 识别失败: {e}")
        return None

def test_generation():
    real_vocal_path = r"/Users/niuzhidao/Library/CloudStorage/OneDrive-qiuyang.ai/PC/Program/videolingo/VideoLingo/output/audio/vocal.mp3"
    
    # 定义明确的输出文件路径（在当前目录下）
    output_filename = "success_result.wav"
    output_abs_path = os.path.abspath(output_filename)
    
    # 如果旧文件存在，先删除，避免干扰
    if os.path.exists(output_abs_path):
        os.remove(output_abs_path)

    print("="*60)
    print("🧪 F5-TTS 最终决胜验证 (文件输出模式)")
    print("="*60)

    if not os.path.exists(real_vocal_path):
        print("❌ 找不到 vocal.mp3")
        return

    # 1. 预处理
    ref_path = ensure_24k_audio(real_vocal_path)
    if not ref_path: return

    # 2. 识别内容
    ref_text = auto_transcribe(ref_path)
    if not ref_text: return
    print(f"👂 参考音频内容: \"{ref_text}\"")
    
    # 3. 设定目标 (英文 -> 英文)
    target_text = "This represents a successful test of the Video Lingo system."
    print("-" * 60)
    print(f"🚀 开始生成到文件: {output_filename}")
    print("-" * 60)

    try:
        # 🟢 关键修改：不再传 None，而是传入具体的文件路径
        generate(
            generation_text=target_text,
            ref_audio_path=ref_path,
            ref_audio_text=ref_text,
            output_path=output_abs_path,  # 👈 强制写入硬盘
            steps=32,
            speed=1.0
        )
        
        # 4. 验证文件是否存在
        print("-" * 60)
        if os.path.exists(output_abs_path):
            file_size = os.path.getsize(output_abs_path)
            print(f"📄 检查输出文件...")
            print(f"   路径: {output_abs_path}")
            print(f"   大小: {file_size} bytes")
            
            if file_size > 1000:
                print("\n🎉🎉🎉 大功告成！")
                print("✅ F5-TTS 模型运行完美，音频已成功生成！")
                print(f"👉 请立即打开这个文件听一下效果: {output_filename}")
            else:
                print("❌ 文件已生成，但是是空的 (0KB)。")
        else:
            print("❌ 生成函数运行结束，但没有找到输出文件。")

    except Exception as e:
        print(f"❌ 运行报错: {e}")
        traceback.print_exc()
    finally:
        if os.path.exists(ref_path): os.remove(ref_path)

if __name__ == "__main__":
    test_generation()
