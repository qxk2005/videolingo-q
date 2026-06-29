#!/usr/bin/env bash
# VideoLingo Q - macOS 启动脚本 (离线版)

# 获取脚本所在绝对路径
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# 解决终端显示中文的问题
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# 执行 launcher
if [ -x "runtime/python/bin/python3" ]; then
    ./runtime/python/bin/python3 launcher.py
else
    # 兼容开发者模式
    python3 launcher.py
fi
