#!/usr/bin/env python3
"""
VideoLingo Q - 离线独立版启动器
=================================
此脚本用于启动打包好的完全离线版 VideoLingo Q。
它会寻找内建的 Python 和 FFmpeg 目录，并将它们加入环境变量，
然后拉起 Streamlit 启动应用。
"""

import os
import sys
import platform
import subprocess

APP_NAME = "VideoLingo Q"
APP_VERSION = "1.0.0"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.join(SCRIPT_DIR, "runtime")
PYTHON_DIR = os.path.join(RUNTIME_DIR, "python")
FFMPEG_DIR = os.path.join(RUNTIME_DIR, "ffmpeg")

SYSTEM = platform.system()

def print_banner():
    banner = f"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     🎬  {APP_NAME} v{APP_VERSION}                           ║
║     专业视频本地化工作站 (完全离线版)                       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)

def get_python_exe():
    if SYSTEM == "Windows":
        return os.path.join(PYTHON_DIR, "python.exe")
    else:
        # macOS portable python 路径 (python-build-standalone)
        return os.path.join(PYTHON_DIR, "bin", "python3")

def main():
    print_banner()
    
    python_exe = get_python_exe()
    
    # 兼容开发环境或非打包环境运行
    if not os.path.exists(python_exe):
        print("⚠️  未检测到内建的离线 Python 环境。尝试使用系统当前 Python。")
        python_exe = sys.executable

    print(f"✅ 使用 Python: {python_exe}")
    
    # 设置 PATH 环境变量，优先使用内建的 FFmpeg 和 Python 脚本目录
    env = os.environ.copy()
    if SYSTEM == "Windows":
        scripts_dir = os.path.join(PYTHON_DIR, "Scripts")
        path_prepends = [FFMPEG_DIR, PYTHON_DIR, scripts_dir]
    else:
        scripts_dir = os.path.join(PYTHON_DIR, "bin")
        path_prepends = [FFMPEG_DIR, scripts_dir]
        
    env["PATH"] = os.pathsep.join(path_prepends) + os.pathsep + env.get("PATH", "")
    
    if os.path.exists(FFMPEG_DIR):
        print(f"✅ 使用内建 FFmpeg: {FFMPEG_DIR}")
    else:
        print("⚠️  未检测到内建 FFmpeg，将使用系统自带的 FFmpeg。")

    # GPU 自动检测与配置（仅 Windows）
    if SYSTEM == "Windows":
        try:
            # 确保 VideoLingo 项目根目录在 Python 路径中
            if SCRIPT_DIR not in sys.path:
                sys.path.insert(0, SCRIPT_DIR)
            from core.utils.gpu_utils import auto_configure_gpu
            auto_configure_gpu(verbose=True)
        except Exception as e:
            print(f"⚠️  GPU 自动检测跳过: {e}")

    print("\n🚀 正在启动服务，请稍候...\n")
    
    # 启动 streamlit
    cmd = [
        python_exe, "-m", "streamlit", "run", "st.py",
        "--server.port=8501",
        "--server.address=localhost",
        "--browser.gatherUsageStats=false"
    ]
    
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n⏹️  服务已停止。")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        input("按回车键退出...")

if __name__ == "__main__":
    main()
