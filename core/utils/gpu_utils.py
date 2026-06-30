"""
GPU 检测工具模块
================
统一检测 NVIDIA GPU 可用性（CUDA / NVENC），缓存结果，
并提供自动配置功能，在程序启动时一次性完成所有检测。
"""

import os
import platform
import subprocess
from functools import lru_cache


# ── CUDA 检测 (用于 Whisper / PyTorch) ────────────────────────────────────────

@lru_cache(maxsize=1)
def check_cuda_available() -> bool:
    """检测当前系统是否支持 CUDA (PyTorch GPU 加速)。

    检测策略：
    1. 优先使用 torch.cuda.is_available()
    2. 备选：检测 nvidia-smi 是否可执行
    
    结果会被缓存，后续调用不再重复检测。
    """
    # macOS 不支持 CUDA
    if platform.system() == "Darwin":
        return False

    # 策略1：通过 PyTorch 检测
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass

    # 策略2：通过 nvidia-smi 检测驱动是否存在
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return False


@lru_cache(maxsize=1)
def get_gpu_info() -> dict:
    """获取 NVIDIA GPU 的详细信息。

    返回:
        dict: {
            "available": bool,          # GPU 是否可用
            "device_name": str | None,  # GPU 型号名称
            "vram_gb": float | None,    # 显存大小 (GB)
            "cuda_version": str | None, # CUDA 版本
            "driver_version": str | None, # 驱动版本
        }
    """
    info = {
        "available": False,
        "device_name": None,
        "vram_gb": None,
        "cuda_version": None,
        "driver_version": None,
    }

    if not check_cuda_available():
        return info

    info["available"] = True

    # 通过 PyTorch 获取详细信息
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["device_name"] = props.name
            info["vram_gb"] = round(props.total_mem / (1024 ** 3), 2)
            info["cuda_version"] = torch.version.cuda
    except Exception:
        pass

    # 通过 nvidia-smi 获取驱动版本（如果 PyTorch 未能获取到设备名称则也获取）
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 1:
                info["driver_version"] = parts[0]
            if len(parts) >= 2 and not info["device_name"]:
                info["device_name"] = parts[1]
            if len(parts) >= 3 and not info["vram_gb"]:
                try:
                    info["vram_gb"] = round(float(parts[2]) / 1024, 2)
                except ValueError:
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return info


# ── NVENC 检测 (用于 FFmpeg 硬件编码) ──────────────────────────────────────────

@lru_cache(maxsize=1)
def check_ffmpeg_nvenc_available() -> bool:
    """检测 FFmpeg 是否支持 h264_nvenc 硬件编码。

    检测步骤：
    1. 检查 ffmpeg -encoders 列表中是否包含 h264_nvenc
    2. 运行时尝试实际编码测试

    结果会被缓存。
    """
    if platform.system() == "Darwin":
        return False  # macOS 使用 videotoolbox，不用 nvenc

    # 步骤1：编译支持检测
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True, text=True, timeout=5
        )
        if "h264_nvenc" not in result.stdout:
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False

    # 步骤2：运行时编码测试
    try:
        test_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
            "-c:v", "h264_nvenc",
            "-f", "null", "-"
        ]
        test_result = subprocess.run(
            test_cmd, capture_output=True, text=True, timeout=5
        )
        return test_result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


# ── 自动配置 ──────────────────────────────────────────────────────────────────

def auto_configure_gpu(verbose: bool = True):
    """在程序启动时自动检测 GPU 并配置相关设置。

    - 检测到 NVIDIA GPU：设置 ffmpeg_gpu=true
    - 未检测到：设置 ffmpeg_gpu=false，并在 Windows 下确保 whisper.runtime 不为 mlx
    
    Args:
        verbose: 是否输出检测信息到终端
    """
    # 仅在 Windows 上执行自动配置
    if platform.system() != "Windows":
        return

    try:
        from core.utils.config_utils import load_key, update_key
    except ImportError:
        return

    gpu_info = get_gpu_info()
    has_cuda = gpu_info["available"]
    has_nvenc = check_ffmpeg_nvenc_available()

    if verbose:
        print("=" * 55)
        print("  🔍 NVIDIA GPU 自动检测")
        print("=" * 55)

        if has_cuda:
            name = gpu_info["device_name"] or "未知型号"
            vram = f"{gpu_info['vram_gb']} GB" if gpu_info["vram_gb"] else "未知"
            cuda_ver = gpu_info["cuda_version"] or "未知"
            driver_ver = gpu_info["driver_version"] or "未知"
            print(f"  ✅ CUDA 可用")
            print(f"     GPU: {name}")
            print(f"     显存: {vram}")
            print(f"     CUDA 版本: {cuda_ver}")
            print(f"     驱动版本: {driver_ver}")
        else:
            print("  ❌ 未检测到可用的 NVIDIA GPU / CUDA")
            print("     Whisper 将使用 CPU 模式运行（速度较慢）")

        if has_nvenc:
            print(f"  ✅ FFmpeg NVENC 硬件编码可用")
        else:
            print(f"  ❌ FFmpeg NVENC 不可用，将使用 CPU 软件编码")

        print("=" * 55)

    # ── 自动更新配置 ──
    # FFmpeg GPU 加速
    current_ffmpeg_gpu = load_key("ffmpeg_gpu")
    if has_nvenc and not current_ffmpeg_gpu:
        update_key("ffmpeg_gpu", True)
        if verbose:
            print("  📝 已自动开启 ffmpeg_gpu（检测到 NVENC 支持）")
    elif not has_nvenc and current_ffmpeg_gpu:
        update_key("ffmpeg_gpu", False)
        if verbose:
            print("  📝 已自动关闭 ffmpeg_gpu（未检测到 NVENC 支持）")

    # Whisper runtime 安全检查
    current_runtime = load_key("whisper.runtime")
    if current_runtime == "mlx":
        # mlx 仅支持 Apple Silicon，在 Windows 上自动切换
        new_runtime = "local" if has_cuda else "cloud"
        update_key("whisper.runtime", new_runtime)
        if verbose:
            print(f"  📝 已自动将 whisper.runtime 从 'mlx' 切换为 '{new_runtime}'")
            if new_runtime == "local" and not has_cuda:
                print("     ⚠️ 本地模式将使用 CPU 运行，速度较慢")


def get_gpu_status_text() -> str:
    """获取用于 UI 展示的 GPU 状态文本。"""
    if platform.system() == "Darwin":
        return "🍎 macOS (Apple Silicon / VideoToolbox)"

    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        name = gpu_info["device_name"] or "NVIDIA GPU"
        vram = f" ({gpu_info['vram_gb']} GB)" if gpu_info["vram_gb"] else ""
        return f"✅ {name}{vram}"
    else:
        return "❌ 未检测到 NVIDIA GPU，使用 CPU 模式"
