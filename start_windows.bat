@echo off
REM ============================================================
REM VideoLingo Q - Windows 一键启动脚本
REM ============================================================
REM 双击此文件即可启动 VideoLingo Q
REM 首次运行会自动安装所有依赖
REM ============================================================

title VideoLingo Q - 专业视频本地化工作站

echo.
echo ==========================================
echo   VideoLingo Q - Windows 启动器
echo ==========================================
echo.

REM 切换到脚本所在目录
cd /d "%~dp0"

REM 查找可用的 Python 3.10
set "PYTHON_CMD="

REM 1. 检查便携式 Python
if exist "runtime\python\python.exe" (
    set "PYTHON_CMD=runtime\python\python.exe"
    echo [OK] 使用便携式 Python: %PYTHON_CMD%
    goto :found_python
)

REM 2. 检查 conda 环境
where conda >nul 2>&1
if %errorlevel% equ 0 (
    REM 尝试 videolingo-q 环境
    call conda activate videolingo-q >nul 2>&1
    if %errorlevel% equ 0 (
        for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do (
            if "%%i"=="3.10" (
                set "PYTHON_CMD=python"
                echo [OK] 使用 conda 环境 'videolingo-q'
                goto :found_python
            )
        )
        call conda deactivate >nul 2>&1
    )
    REM 尝试 videolingo 环境
    call conda activate videolingo >nul 2>&1
    if %errorlevel% equ 0 (
        for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do (
            if "%%i"=="3.10" (
                set "PYTHON_CMD=python"
                echo [OK] 使用 conda 环境 'videolingo'
                goto :found_python
            )
        )
        call conda deactivate >nul 2>&1
    )
)

REM 3. 检查系统 Python
for %%c in (python3 python py) do (
    where %%c >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=*" %%v in ('%%c -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do (
            if "%%v"=="3.10" (
                set "PYTHON_CMD=%%c"
                echo [OK] 使用系统 Python 3.10
                goto :found_python
            )
        )
    )
)

REM 4. Windows py launcher
where py >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%v in ('py -3.10 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do (
        if "%%v"=="3.10" (
            set "PYTHON_CMD=py -3.10"
            echo [OK] 使用 Python Launcher (py -3.10)
            goto :found_python
        )
    )
)

REM 未找到 Python 3.10
echo.
echo [ERROR] 未找到 Python 3.10!
echo.
echo 请先安装 Python 3.10，推荐方式:
echo   方式 1 (推荐): conda create -n videolingo python=3.10
echo                  conda activate videolingo
echo   方式 2: 从 https://www.python.org/downloads/ 下载安装 Python 3.10
echo.
echo 安装完成后，请重新双击此文件运行。
echo.
pause
exit /b 1

:found_python

REM 将 FFmpeg 添加到 PATH (如果有便携式版本)
if exist "runtime\ffmpeg\ffmpeg.exe" (
    set "PATH=%~dp0runtime\ffmpeg;%PATH%"
)

echo.
echo 正在启动 VideoLingo Q...
echo.
%PYTHON_CMD% launcher.py %*

REM 如果异常退出，保持窗口
if errorlevel 1 (
    echo.
    echo [WARN] 程序异常退出
    pause
)
