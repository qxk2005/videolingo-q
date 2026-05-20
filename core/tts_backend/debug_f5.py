import numpy as np
import soundfile as sf
from f5_tts_mlx.generate import generate
import tempfile
import os

def test_generation():
    # 1. 复现报错的参数
    text = "我在 iPad 上部署了 Ubuntu 22"
    ref_text = "这段声音将作为语音合成的参考音频。"
    
    # 制作 3.5s 白噪音参考
    sr = 24000
    ref_audio = np.random.uniform(-0.1, 0.1, int(sr * 3.5)).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        ref_path = tmp.name
        sf.write(ref_path, ref_audio, sr)

    print(f"🧪 测试启动: '{text}'")
    
    try:
        # 2. 生成
        result = generate(
            generation_text=text,
            ref_audio_path=ref_path,
            ref_audio_text=ref_text,
            output_path=None, # 内存模式
            steps=32,
            speed=1.0
        )
        
        # 3. 深度分析返回数据
        audio_data = result
        if hasattr(result, "audio"): audio_data = result.audio
        
        # 转换为 Numpy
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)

        print(f"📊 数据形状 (Shape): {audio_data.shape}")
        print(f"📏 直接 len(): {len(audio_data)}")
        
        # 扁平化处理 (这是修复的关键)
        flat_data = audio_data.flatten()
        print(f"📏 Flatten后 len(): {len(flat_data)}")
        print(f"⏱️ 实际时长: {len(flat_data)/sr:.4f}秒")

        # 保存听听看
        out_file = "debug_ubuntu.wav"
        sf.write(out_file, flat_data, sr)
        print(f"💾 已保存: {out_file} (请打开听一下内容是否完整)")

    except Exception as e:
        print(f"❌ 报错: {e}")
    finally:
        try: os.remove(ref_path)
        except: pass

if __name__ == "__main__":
    test_generation()
