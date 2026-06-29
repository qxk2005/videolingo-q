#!/usr/bin/env python3
"""
VideoLingo Q - 跨平台启动引导程序 (Launcher)
=============================================
此脚本是打包发布后用户运行的入口。它负责：
1. 检测或创建便携式 Python 环境 (基于 micromamba)
2. 安装 pip 依赖 (轻量依赖随包分发，ML 大型依赖首次运行时在线安装)
3. 检测和安装 FFmpeg
4. 启动 Streamlit 应用并打开浏览器

用户双击 start_macos.command 或 start_windows.bat 即可触发此脚本。
"""

import os
import sys
import platform
import subprocess
import shutil
import urllib.request
import zipfile
import tarfile
import time
import json
import threading

# ============================================================
# 常量
# ============================================================
APP_NAME = "VideoLingo Q"
APP_VERSION = "1.0.0"

# 相对于本脚本所在目录的路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.join(SCRIPT_DIR, "runtime")
PYTHON_DIR = os.path.join(RUNTIME_DIR, "python")
FFMPEG_DIR = os.path.join(RUNTIME_DIR, "ffmpeg")
FIRST_RUN_MARKER = os.path.join(RUNTIME_DIR, ".installed")

SYSTEM = platform.system()  # 'Darwin' or 'Windows' or 'Linux'
ARCH = platform.machine()   # 'arm64', 'x86_64', 'AMD64', etc.

# Python 版本要求
REQUIRED_PYTHON_MAJOR = 3
REQUIRED_PYTHON_MINOR = 10

# Streamlit 端口
STREAMLIT_PORT = 8501

# ============================================================
# 工具函数
# ============================================================

def print_banner():
    """打印启动横幅"""
    banner = f"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     🎬  {APP_NAME} v{APP_VERSION}                           ║
║     专业视频本地化工作站                                  ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


def log(msg, level="INFO"):
    """简单日志输出"""
    icons = {"INFO": "ℹ️ ", "OK": "✅", "WARN": "⚠️ ", "ERROR": "❌", "STEP": "🔧"}
    prefix = icons.get(level, "  ")
    print(f"  {prefix}  {msg}")


def run_cmd(cmd, desc="", env=None, check=True, capture=False):
    """运行命令行，带友好的错误处理"""
    if desc:
        log(desc, "STEP")
    
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    
    try:
        result = subprocess.run(
            cmd,
            env=merged_env,
            check=check,
            capture_output=capture,
            text=True if capture else None,
            encoding='utf-8' if capture else None,
        )
        return result
    except subprocess.CalledProcessError as e:
        log(f"命令执行失败: {' '.join(cmd) if isinstance(cmd, list) else cmd}", "ERROR")
        if capture and e.stdout:
            print(f"    stdout: {e.stdout[:500]}")
        if capture and e.stderr:
            print(f"    stderr: {e.stderr[:500]}")
        if check:
            raise
        return e
    except FileNotFoundError:
        log(f"命令未找到: {cmd[0] if isinstance(cmd, list) else cmd}", "ERROR")
        if check:
            raise
        return None


def download_file(url, dest, desc=""):
    """下载文件并显示进度"""
    if desc:
        log(f"下载: {desc}", "STEP")
    log(f"  URL: {url}")
    
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 // total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r    进度: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
    
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest, reporthook)
    print()  # 换行
    log("下载完成", "OK")


# ============================================================
# 环境检测
# ============================================================

def get_python_executable():
    """获取可用的 Python 3.10 可执行文件路径。
    
    优先级：
    1. runtime/python 下的便携式 Python
    2. 当前运行此脚本的 Python（如果版本匹配）
    3. 系统 PATH 中的 python3 / python
    """
    # 1. 检查便携式 Python
    if SYSTEM == "Windows":
        portable_python = os.path.join(PYTHON_DIR, "python.exe")
    else:
        portable_python = os.path.join(PYTHON_DIR, "bin", "python3")
    
    if os.path.exists(portable_python):
        log(f"检测到便携式 Python: {portable_python}", "OK")
        return portable_python
    
    # 2. 检查当前 Python
    if sys.version_info.major == REQUIRED_PYTHON_MAJOR and sys.version_info.minor == REQUIRED_PYTHON_MINOR:
        log(f"当前 Python 版本满足要求: {sys.version}", "OK")
        return sys.executable
    
    # 3. 尝试系统 Python
    for cmd in ["python3.10", "python3", "python"]:
        py = shutil.which(cmd)
        if py:
            try:
                result = subprocess.run(
                    [py, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                    capture_output=True, text=True
                )
                version = result.stdout.strip()
                if version == f"{REQUIRED_PYTHON_MAJOR}.{REQUIRED_PYTHON_MINOR}":
                    log(f"找到系统 Python 3.10: {py}", "OK")
                    return py
            except Exception:
                continue
    
    return None


def get_ffmpeg_path():
    """获取 FFmpeg 可执行文件路径。
    
    优先级：
    1. runtime/ffmpeg 下的便携式 FFmpeg
    2. 系统 PATH 中的 ffmpeg
    """
    # 1. 检查便携式 FFmpeg
    if SYSTEM == "Windows":
        portable_ffmpeg = os.path.join(FFMPEG_DIR, "ffmpeg.exe")
    else:
        portable_ffmpeg = os.path.join(FFMPEG_DIR, "ffmpeg")
    
    if os.path.exists(portable_ffmpeg):
        return os.path.dirname(portable_ffmpeg)
    
    # 2. 检查系统 FFmpeg
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return os.path.dirname(system_ffmpeg)
    
    return None


# ============================================================
# FFmpeg 安装
# ============================================================

def install_ffmpeg():
    """下载并安装便携式 FFmpeg"""
    log("正在安装 FFmpeg...", "STEP")
    os.makedirs(FFMPEG_DIR, exist_ok=True)
    
    if SYSTEM == "Darwin":
        # macOS: 从 evermeet.cx 下载静态编译版本 (使用 GitHub 镜像作为替代)
        if ARCH == "arm64":
            ffmpeg_url = "https://github.com/eugeneware/ffmpeg-static/releases/latest/download/darwin-arm64"
            ffprobe_url = "https://github.com/eugeneware/ffmpeg-static/releases/latest/download/darwin-arm64"
        else:
            ffmpeg_url = "https://github.com/eugeneware/ffmpeg-static/releases/latest/download/darwin-x64"
            ffprobe_url = "https://github.com/eugeneware/ffmpeg-static/releases/latest/download/darwin-x64"
        
        # 使用 ffmpeg-static npm 包的 GitHub release
        # 使用更可靠的源: 从 GitHub 下载预编译二进制
        if ARCH == "arm64":
            url = "https://www.osxexperts.net/ffmpeg7arm.zip"
        else:
            url = "https://evermeet.cx/ffmpeg/getrelease/zip"
        
        tmp_file = os.path.join(RUNTIME_DIR, "ffmpeg_tmp.zip")
        try:
            download_file(url, tmp_file, "FFmpeg (macOS)")
            with zipfile.ZipFile(tmp_file, 'r') as zf:
                zf.extractall(FFMPEG_DIR)
            # 设置可执行权限
            ffmpeg_bin = os.path.join(FFMPEG_DIR, "ffmpeg")
            if os.path.exists(ffmpeg_bin):
                os.chmod(ffmpeg_bin, 0o755)
        except Exception as e:
            log(f"FFmpeg 自动下载失败: {e}", "WARN")
            log("请手动安装 FFmpeg: brew install ffmpeg", "WARN")
            return False
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
    
    elif SYSTEM == "Windows":
        # Windows: 从 gyan.dev 或 GitHub 下载
        url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        tmp_file = os.path.join(RUNTIME_DIR, "ffmpeg_tmp.zip")
        
        try:
            download_file(url, tmp_file, "FFmpeg (Windows)")
            with zipfile.ZipFile(tmp_file, 'r') as zf:
                zf.extractall(RUNTIME_DIR)
            
            # 查找解压后的 ffmpeg.exe 并移动到目标目录
            for root, dirs, files in os.walk(RUNTIME_DIR):
                if "ffmpeg.exe" in files and root != FFMPEG_DIR:
                    for f in ["ffmpeg.exe", "ffprobe.exe"]:
                        src = os.path.join(root, f)
                        dst = os.path.join(FFMPEG_DIR, f)
                        if os.path.exists(src):
                            shutil.move(src, dst)
                    # 清理解压的临时目录
                    extracted_dir = os.path.join(RUNTIME_DIR, os.listdir(RUNTIME_DIR)[0])
                    if os.path.isdir(extracted_dir) and extracted_dir != FFMPEG_DIR:
                        shutil.rmtree(extracted_dir, ignore_errors=True)
                    break
        except Exception as e:
            log(f"FFmpeg 自动下载失败: {e}", "WARN")
            log("请手动安装 FFmpeg 并添加到系统 PATH", "WARN")
            return False
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
    
    log("FFmpeg 安装完成", "OK")
    return True


# ============================================================
# pip 依赖安装
# ============================================================

def install_dependencies(python_exe):
    """安装 pip 依赖 (分为核心依赖和 ML 大型依赖两个阶段)"""
    
    requirements_file = os.path.join(SCRIPT_DIR, "requirements.txt")
    if not os.path.exists(requirements_file):
        log("requirements.txt 未找到!", "ERROR")
        return False
    
    # ---- 阶段 1: 升级 pip ----
    log("升级 pip...", "STEP")
    run_cmd([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=False)
    
    # ---- 阶段 2: 安装 PyTorch (平台特定) ----
    log("安装 PyTorch...", "STEP")
    if SYSTEM == "Darwin":
        # macOS: CPU 版 PyTorch
        run_cmd([python_exe, "-m", "pip", "install", "torch==2.1.2", "torchaudio==2.1.2"])
    else:
        # Windows/Linux: 先尝试检测 GPU
        has_gpu = False
        try:
            result = run_cmd(
                [python_exe, "-c", "import subprocess; r=subprocess.run(['nvidia-smi'], capture_output=True); exit(0 if r.returncode==0 else 1)"],
                check=False, capture=True
            )
            has_gpu = (result and result.returncode == 0)
        except Exception:
            pass
        
        if has_gpu:
            log("检测到 NVIDIA GPU，安装 CUDA 加速版 PyTorch...", "OK")
            run_cmd([python_exe, "-m", "pip", "install", "torch==2.0.0", "torchaudio==2.0.0",
                     "--index-url", "https://download.pytorch.org/whl/cu118"])
        else:
            log("未检测到 NVIDIA GPU，安装 CPU 版 PyTorch...", "INFO")
            run_cmd([python_exe, "-m", "pip", "install", "torch==2.1.2", "torchaudio==2.1.2"])
    
    # ---- 阶段 3: 安装常规 pip 依赖 (排除 git 依赖) ----
    log("安装常规 pip 依赖...", "STEP")
    requirements_portable = os.path.join(SCRIPT_DIR, "requirements_portable.txt")
    try:
        with open(requirements_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # 过滤掉 git+ 依赖、注释和空行
        filtered = [l for l in lines if l.strip() and not l.strip().startswith("#") and "git+" not in l]
        with open(requirements_portable, "w", encoding="utf-8") as f:
            f.writelines(filtered)
        run_cmd([python_exe, "-m", "pip", "install", "-r", requirements_portable], check=False)
    except Exception as e:
        log(f"常规依赖安装部分失败: {e}", "WARN")
    finally:
        if os.path.exists(requirements_portable):
            os.remove(requirements_portable)
    
    # ---- 阶段 4: 安装项目自身 (不触发依赖重装) ----
    log("注册项目包 (pip install -e . --no-deps)...", "STEP")
    run_cmd(
        [python_exe, "-m", "pip", "install", "-e", ".", "--no-deps"],
        env={"PIP_NO_CACHE_DIR": "0", "PYTHONIOENCODING": "utf-8"},
        check=False
    )
    
    # ---- 阶段 5: 安装 git 依赖 (demucs, whisperx) ----
    log("安装 Git 依赖 (demucs, whisperx)...", "STEP")
    log("  (这些依赖需要从 GitHub 下载，可能需要几分钟)", "INFO")
    
    git_deps = [
        ("demucs", "demucs[dev] @ git+https://github.com/adefossez/demucs"),
        ("whisperx", "whisperx @ git+https://github.com/m-bain/whisperx.git@7307306a9d8dd0d261e588cc933322454f853853"),
    ]
    
    for name, dep in git_deps:
        result = run_cmd([python_exe, "-m", "pip", "install", dep], check=False)
        if result and hasattr(result, 'returncode') and result.returncode != 0:
            log(f"  {name} 安装失败。请确保已安装 git，并在终端手动运行:", "WARN")
            log(f"    pip install \"{dep}\"", "WARN")
        else:
            log(f"  {name} 安装成功", "OK")
    
    log("所有依赖安装完成!", "OK")
    return True


# ============================================================
# 应用启动
# ============================================================

def start_streamlit(python_exe):
    """启动 Streamlit 应用"""
    log(f"启动 {APP_NAME}...", "STEP")
    
    # 构建环境变量
    env = os.environ.copy()
    
    # 添加 FFmpeg 到 PATH
    ffmpeg_path = get_ffmpeg_path()
    if ffmpeg_path:
        env["PATH"] = ffmpeg_path + os.pathsep + env.get("PATH", "")
        log(f"FFmpeg 路径: {ffmpeg_path}", "OK")
    
    # 添加便携式 Python 的 Scripts/bin 到 PATH
    if SYSTEM == "Windows":
        scripts_dir = os.path.join(os.path.dirname(python_exe), "Scripts")
    else:
        scripts_dir = os.path.join(os.path.dirname(python_exe))
    
    if os.path.exists(scripts_dir):
        env["PATH"] = scripts_dir + os.pathsep + env.get("PATH", "")
    
    # Streamlit 命令
    streamlit_cmd = [
        python_exe, "-m", "streamlit", "run", "st.py",
        "--server.port", str(STREAMLIT_PORT),
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
    ]
    
    log(f"应用地址: http://localhost:{STREAMLIT_PORT}", "OK")
    log("正在启动 Streamlit 服务器 (按 Ctrl+C 停止)...", "INFO")
    print()
    print("=" * 60)
    print()
    
    # 延迟打开浏览器
    def open_browser():
        time.sleep(3)
        import webbrowser
        webbrowser.open(f"http://localhost:{STREAMLIT_PORT}")
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # 启动 Streamlit (阻塞)
    try:
        process = subprocess.Popen(streamlit_cmd, env=env, cwd=SCRIPT_DIR)
        process.wait()
    except KeyboardInterrupt:
        log("收到停止信号，正在关闭...", "INFO")
        process.terminate()
        process.wait(timeout=5)
        log("已关闭", "OK")


# ============================================================
# 首次运行向导
# ============================================================

def first_run_setup(python_exe):
    """首次运行的完整安装流程"""
    print()
    log("=" * 50)
    log(f"首次运行 {APP_NAME}，正在进行初始化配置...")
    log("=" * 50)
    print()
    
    # 1. 检查/安装 FFmpeg
    if not get_ffmpeg_path():
        log("未检测到 FFmpeg，正在自动安装...", "WARN")
        if not install_ffmpeg():
            log("FFmpeg 安装失败。请手动安装后重新运行。", "ERROR")
            if SYSTEM == "Darwin":
                log("推荐: brew install ffmpeg", "INFO")
            else:
                log("推荐: 从 https://ffmpeg.org/download.html 下载", "INFO")
            return False
    else:
        log("FFmpeg 已安装", "OK")
    
    # 2. 安装 pip 依赖
    if not install_dependencies(python_exe):
        log("依赖安装失败，请检查网络连接后重试。", "ERROR")
        return False
    
    # 3. 复制默认配置
    config_path = os.path.join(SCRIPT_DIR, "config.yaml")
    config_example = os.path.join(SCRIPT_DIR, "config.example.yaml")
    if not os.path.exists(config_path) and os.path.exists(config_example):
        shutil.copy(config_example, config_path)
        log("已从模板创建 config.yaml", "OK")
    
    # 4. 创建输出目录
    os.makedirs(os.path.join(SCRIPT_DIR, "output"), exist_ok=True)
    
    # 5. 标记安装完成
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    with open(FIRST_RUN_MARKER, "w") as f:
        f.write(json.dumps({
            "version": APP_VERSION,
            "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": f"{SYSTEM}-{ARCH}",
            "python": python_exe,
        }, indent=2))
    
    print()
    log("=" * 50)
    log("初始化配置完成！即将启动应用...")
    log("=" * 50)
    print()
    return True


# ============================================================
# 主入口
# ============================================================

def main():
    """主入口"""
    print_banner()
    
    # 切换工作目录到脚本所在目录
    os.chdir(SCRIPT_DIR)
    log(f"工作目录: {SCRIPT_DIR}", "INFO")
    log(f"系统平台: {SYSTEM} ({ARCH})", "INFO")
    
    # --check 模式：仅检测环境
    if "--check" in sys.argv:
        python_exe = get_python_executable()
        ffmpeg_path = get_ffmpeg_path()
        log(f"Python: {python_exe or '未找到'}")
        log(f"FFmpeg: {ffmpeg_path or '未找到'}")
        log(f"首次运行标记: {'存在' if os.path.exists(FIRST_RUN_MARKER) else '不存在'}")
        sys.exit(0 if python_exe else 1)
    
    # 1. 获取 Python 可执行文件
    python_exe = get_python_executable()
    
    if not python_exe:
        log("未找到 Python 3.10！", "ERROR")
        print()
        log("请先安装 Python 3.10，推荐方式：")
        if SYSTEM == "Darwin":
            log("  方式 1 (推荐): conda create -n videolingo python=3.10 && conda activate videolingo")
            log("  方式 2: brew install python@3.10")
            log("  方式 3: 从 https://www.python.org/downloads/ 下载")
        else:
            log("  方式 1 (推荐): conda create -n videolingo python=3.10 && conda activate videolingo")
            log("  方式 2: 从 https://www.python.org/downloads/ 下载安装 Python 3.10")
        print()
        log("安装完成后，请重新运行此启动器。")
        input("\n按 Enter 键退出...")
        sys.exit(1)
    
    log(f"使用 Python: {python_exe}", "OK")
    
    # 2. 检查是否首次运行
    if not os.path.exists(FIRST_RUN_MARKER):
        if not first_run_setup(python_exe):
            log("初始化失败，请解决上述问题后重新运行。", "ERROR")
            input("\n按 Enter 键退出...")
            sys.exit(1)
    else:
        log("已检测到安装标记，跳过初始化", "OK")
        # 确保 FFmpeg 可用
        if not get_ffmpeg_path():
            log("FFmpeg 未找到，尝试重新安装...", "WARN")
            install_ffmpeg()
    
    # 3. 启动应用
    start_streamlit(python_exe)


if __name__ == "__main__":
    main()
