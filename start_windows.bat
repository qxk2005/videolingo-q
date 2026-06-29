@echo off
chcp 65001 >nul
title VideoLingo Q 离线版启动器

:: 切换到脚本所在目录
cd /d "%~dp0"

echo.
echo ===============================================
echo   正在启动 VideoLingo Q
echo ===============================================
echo.

if exist "runtime\python\python.exe" (
    "runtime\python\python.exe" launcher.py
) else (
    :: 兼容开发者模式
    python launcher.py
)

if %errorlevel% neq 0 (
    echo.
    echo ⚠️ 启动异常终止。
    pause
)
