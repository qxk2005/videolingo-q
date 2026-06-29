#!/bin/bash
# ============================================================
# VideoLingo Q - macOS 一键启动脚本
# ============================================================
# 双击此文件即可启动 VideoLingo Q
# 首次运行会自动安装所有依赖
# ============================================================

# 切换到脚本所在目录
cd "$(dirname "$0")"

echo ""
echo "=========================================="
echo "  🎬 VideoLingo Q - macOS 启动器"
echo "=========================================="
echo ""

# 添加常见的 PATH
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# 查找可用的 Python 3.10
PYTHON_CMD=""

# 1. 检查便携式 Python
if [ -f "runtime/python/bin/python3" ]; then
    PYTHON_CMD="runtime/python/bin/python3"
    echo "✅ 使用便携式 Python: $PYTHON_CMD"
fi

# 2. 检查 conda 环境
if [ -z "$PYTHON_CMD" ]; then
    # 尝试激活 conda 环境
    if command -v conda &>/dev/null; then
        # 初始化 conda for this shell
        eval "$(conda shell.bash hook 2>/dev/null)"
        
        # 尝试 videolingo 或 videolingo-q 环境
        for env_name in videolingo-q videolingo; do
            if conda env list 2>/dev/null | grep -q "^$env_name "; then
                echo "🔍 尝试激活 conda 环境: $env_name"
                conda activate "$env_name" 2>/dev/null
                if [ $? -eq 0 ]; then
                    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.minor}')" 2>/dev/null)
                    if [ "$PY_VER" = "10" ]; then
                        PYTHON_CMD="python3"
                        echo "✅ 使用 conda 环境 '$env_name': $(which python3)"
                        break
                    fi
                    conda deactivate 2>/dev/null
                fi
            fi
        done
    fi
fi

# 3. 检查系统 Python
if [ -z "$PYTHON_CMD" ]; then
    for cmd in python3.10 python3 python; do
        if command -v "$cmd" &>/dev/null; then
            PY_VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            if [ "$PY_VER" = "3.10" ]; then
                PYTHON_CMD="$cmd"
                echo "✅ 使用系统 Python 3.10: $(which $cmd)"
                break
            fi
        fi
    done
fi

# 如果找不到 Python 3.10
if [ -z "$PYTHON_CMD" ]; then
    echo ""
    echo "❌ 错误: 未找到 Python 3.10!"
    echo ""
    echo "请先安装 Python 3.10，推荐方式:"
    echo "  方式 1 (推荐): conda create -n videolingo python=3.10"
    echo "                 conda activate videolingo"
    echo "  方式 2: brew install python@3.10"
    echo ""
    echo "安装完成后，请重新双击此文件运行。"
    echo ""
    read -p "按 Enter 键退出..."
    exit 1
fi

# 启动 launcher.py
echo ""
echo "🚀 正在启动 VideoLingo Q..."
echo ""
"$PYTHON_CMD" launcher.py "$@"

# 如果异常退出，保持窗口
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "⚠️ 程序退出，退出码: $EXIT_CODE"
    read -p "按 Enter 键关闭窗口..."
fi
