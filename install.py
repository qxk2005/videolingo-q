import os, sys
import platform

# =========================================================================
# Python 3.10 运行版本门禁硬性检查
# =========================================================================
if sys.version_info.major != 3 or sys.version_info.minor != 10:
    print("=" * 70)
    print(f"❌ 错误: videolingo-q 严格要求使用 Python 3.10 环境!")
    print(f"您当前正在使用的 Python 版本是: {platform.python_version()}")
    print(f"当前 Python 路径: {sys.executable}")
    print("=" * 70)
    print("请安装并切换到 Python 3.10 (例如 3.10.11)，然后重新运行此安装脚本。")
    print("推荐的设置步骤:")
    print("  conda create -n videolingo-q python=3.10")
    print("  conda activate videolingo-q")
    print("  python install.py")
    print("=" * 70)
    sys.exit(1)

import subprocess
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

ascii_logo = """
__     ___     _            _     _                    
\ \   / (_) __| | ___  ___ | |   (_)_ __   __ _  ___  
 \ \ / /| |/ _` |/ _ \/ _ \| |   | | '_ \ / _` |/ _ \ 
  \ V / | | (_| |  __/ (_) | |___| | | | | (_| | (_) |
   \_/  |_|\__,_|\___|\___/|_____|_|_| |_|\__, |\___/ 
                                           |___/        
"""

def install_package(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])

def check_nvidia_gpu():
    install_package("pynvml")
    import pynvml
    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            print("🎮 已检测到 NVIDIA 显卡:")
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                print(f"  GPU {i}: {name}")
            return True
        else:
            print("💻 未检测到 NVIDIA 显卡")
            return False
    except pynvml.NVMLError:
        print("💻 未检测到 NVIDIA 显卡或显卡驱动未正确配置")
        return False
    finally:
        if initialized:
            pynvml.nvmlShutdown()

def check_ffmpeg():
    from rich.console import Console
    from rich.panel import Panel
    console = Console()

    try:
        # 检查 FFmpeg 是否安装
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        console.print(Panel("✅ FFmpeg 已经正确安装并配置环境变量", style="green"))
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        system = platform.system()
        install_cmd = ""
        
        if system == "Windows":
            install_cmd = "choco install ffmpeg"
            extra_note = "请先安装 Chocolatey 包管理器 (https://chocolatey.org/)\n或者直接去 FFmpeg 官网下载并配置环境变量。"
        elif system == "Darwin":
            install_cmd = "brew install ffmpeg"
            extra_note = "请先安装 Homebrew 包管理器 (https://brew.sh/)"
        elif system == "Linux":
            install_cmd = "sudo apt install ffmpeg  # Ubuntu/Debian\nsudo yum install ffmpeg  # CentOS/RHEL"
            extra_note = "使用您系统的包管理器直接进行安装。"
        
        console.print(Panel.fit(
            "❌ 未找到 FFmpeg 核心多媒体依赖库\n\n" +
            "🛠️ 请使用以下推荐命令进行安装:\n" +
            f"[bold cyan]{install_cmd}[/bold cyan]\n\n" +
            f"💡 提示:\n{extra_note}\n\n" +
            "🔄 安装 FFmpeg 完毕后，请重新运行此安装脚本:\n[bold cyan]python install.py[/bold cyan]",
            style="red"
        ))
        raise SystemExit("错误：运行 videolingo-q 必须安装 FFmpeg。请安装后重新运行此安装脚本。")

def main():
    install_package("requests", "rich", "ruamel.yaml", "InquirerPy")
    from rich.console import Console
    from rich.panel import Panel
    from rich.box import DOUBLE
    from InquirerPy import inquirer
    from translations.translations import DISPLAY_LANGUAGES
    from core.utils.config_utils import load_key, update_key
    from core.utils.decorator import except_handler

    console = Console()
    
    width = max(len(line) for line in ascii_logo.splitlines()) + 4
    welcome_panel = Panel(
        ascii_logo,
        width=width,
        box=DOUBLE,
        title="[bold green]🌏 欢迎使用 videolingo-q 一键安装向导 🌏[/bold green]",
        border_style="bright_blue"
    )
    console.print(welcome_panel)
    
    # 选择显示语言
    current_language = load_key("display_language")
    current_display = next((k for k, v in DISPLAY_LANGUAGES.items() if v == current_language), "简体中文")
    selected_language = DISPLAY_LANGUAGES[inquirer.select(
        message="选择显示语言 / Select Language:",
        choices=list(DISPLAY_LANGUAGES.keys()),
        default=current_display
    ).execute()]
    update_key("display_language", selected_language)

    console.print(Panel.fit("🚀 正在启动环境部署程序...", style="bold magenta"))

    # 配置镜像源
    if inquirer.confirm(
        message="您是否需要自动配置国内 PyPI 镜像源? (如果您在中国大陆访问默认源较慢，建议开启)",
        default=True
    ).execute():
        from core.utils.pypi_autochoose import main as choose_mirror
        choose_mirror()

    # 检测系统与 GPU
    has_gpu = platform.system() != 'Darwin' and check_nvidia_gpu()
    if has_gpu:
        console.print(Panel("🎮 已检测到 NVIDIA GPU，正在安装 CUDA 加加速版本的 PyTorch 深度学习库...", style="cyan"))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.0.0", "torchaudio==2.0.0", "--index-url", "https://download.pytorch.org/whl/cu118"])
    else:
        system_name = "🍎 MacOS" if platform.system() == 'Darwin' else "💻 未检测到 NVIDIA 显卡"
        console.print(Panel(f"🖥️ 已检测到 {system_name}，正在安装 CPU 版本的 PyTorch 深度学习库...\n⚠️ 提示: CPU 版本在后续 WhisperX 语音转写时速度可能较慢。", style="cyan"))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.1.2", "torchaudio==2.1.2"])

    @except_handler("videolingo-q 项目依赖安装失败")
    def install_requirements():
        console.print(Panel("📦 正在使用 `pip install -e .` 以开发模式安装项目和全部依赖...", style="cyan"))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."], env={**os.environ, "PIP_NO_CACHE_DIR": "0", "PYTHONIOENCODING": "utf-8"})

    @except_handler("Noto 字体安装失败")
    def install_noto_font():
        if os.path.exists('/etc/debian_version'):
            # Debian/Ubuntu 系统
            cmd = ['sudo', 'apt-get', 'install', '-y', 'fonts-noto']
            pkg_manager = "apt-get"
        elif os.path.exists('/etc/redhat-release'):
            # RHEL/CentOS/Fedora 系统
            cmd = ['sudo', 'yum', 'install', '-y', 'google-noto*']
            pkg_manager = "yum"
        else:
            console.print("⚠️ 警告: 未能识别的 Linux 发行版，请后续手动安装 Noto 字体以防字幕渲染乱码", style="yellow")
            return

        subprocess.run(cmd, check=True)
        console.print(f"✅ 成功通过 {pkg_manager} 自动安装 Noto 字体", style="green")

    if platform.system() == 'Linux':
        install_noto_font()
    
    install_requirements()
    check_ffmpeg()
    
    # 安装完成面板一
    panel1_text = (
        "🎉 videolingo-q 全套环境已成功安装配置完毕！" + "\n\n" +
        "🚀 正在自动为您运行以下命令启动服务:\n" +
        "[bold]streamlit run st.py[/bold]\n\n" +
        "💡 提示: 首次冷启动可能需要最多 1 分钟加载底层模型，请您耐心等待。"
    )
    console.print(Panel(panel1_text, style="bold green"))

    # 故障排查面板二
    panel2_text = (
        "❓ 如果您的浏览器未能自动弹出或启动服务失败，您可以尝试:" + "\n" +
        "1. 确认您的网络配置或 VPN 状态是否顺畅" + "\n" +
        "2. 重新运行此一键修复与安装程序: [bold]python install.py[/bold]"
    )
    console.print(Panel(panel2_text, style="yellow"))

    # 启动应用
    subprocess.Popen(["streamlit", "run", "st.py"])

if __name__ == "__main__":
    main()
