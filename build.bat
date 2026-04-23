@echo off
REM ============================================================================
REM PyMeshGen 一键打包工具 (Windows)
REM ============================================================================
REM 功能:
REM   - 自动检查并安装依赖 (PyInstaller)
REM   - 使用 PyInstaller 打包生成可执行文件
REM   - 可选：创建 Inno Setup 安装包
REM   - 可选：创建便携版 ZIP 压缩包
REM
REM 使用方法:
REM   双击运行此脚本，或从命令行执行: build.bat [参数]
REM
REM 参数:
REM   clean       - 清理之前的构建文件
REM   debug       - 启用调试模式
REM   installer   - 创建安装包（需要 Inno Setup）
REM   zip         - 创建便携版 ZIP
REM   all         - 创建所有输出（安装包 + ZIP）
REM   help        - 显示帮助信息
REM ============================================================================

setlocal enabledelayedexpansion

REM 设置标题
title PyMeshGen 一键打包工具

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8+
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ============================================================================
echo PyMeshGen 一键打包工具
echo ============================================================================
echo.

REM 显示 Python 版本
python --version
echo.

REM 解析命令行参数
set ARGS=

for %%a in (%*) do (
    if /i "%%a"=="help" goto :help
    if /i "%%a"=="clean" set ARGS=!ARGS! --clean
    if /i "%%a"=="debug" set ARGS=!ARGS! --debug
    if /i "%%a"=="installer" set ARGS=!ARGS! --installer
    if /i "%%a"=="zip" set ARGS=!ARGS! --zip
    if /i "%%a"=="all" set ARGS=!ARGS! --all
)

REM 运行打包脚本
echo 正在启动打包工具...
echo.

python build_app.py %ARGS%

if errorlevel 1 (
    echo.
    echo ============================================================================
    echo [错误] 打包失败！请检查错误信息。
    echo ============================================================================
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo 打包完成！
echo ============================================================================
echo.
pause

goto :eof

:help
echo ============================================================================
echo PyMeshGen 一键打包工具 - 使用说明
echo ============================================================================
echo.
echo 用法：build.bat [参数]
echo.
echo 参数:
echo   clean       - 清理之前的构建文件后再打包
echo   debug       - 启用调试模式（显示控制台窗口）
echo   installer   - 创建 Inno Setup 安装包
echo   zip         - 创建便携版 ZIP 压缩包
echo   all         - 创建所有输出（安装包 + ZIP）
echo   help        - 显示此帮助信息
echo.
echo 示例:
echo   build.bat              - 基本打包（仅可执行文件）
echo   build.bat clean        - 清理后打包
echo   build.bat installer    - 创建安装包（需要 Inno Setup）
echo   build.bat zip          - 创建 ZIP 便携版
echo   build.bat all          - 创建安装包和 ZIP
echo   build.bat clean all    - 清理并创建所有输出
echo.
echo 输出位置:
echo   可执行文件：dist\PyMeshGen.exe
echo   安装包：installer\PyMeshGen-Setup-*.exe
echo   ZIP 包：releases\PyMeshGen-v*.zip
echo.
echo ============================================================================
goto :eof
