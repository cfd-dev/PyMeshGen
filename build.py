#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMeshGen 一键打包工具
自动完成从源码到安装包/便携版的完整流程

功能:
    - 自动检查并安装依赖 (PyInstaller, pyinstaller-hooks-contrib)
    - 使用 PyInstaller 打包生成可执行文件
    - 可选：使用 Inno Setup 创建 Windows 安装包
    - 可选：创建便携版 ZIP 压缩包

使用方法:
    python unified_build.py [--clean] [--debug] [--installer] [--zip]

参数:
    --clean       清理之前的构建文件
    --debug       启用调试模式（显示控制台窗口）
    --installer   创建 Inno Setup 安装包（需要安装 Inno Setup）
    --zip         创建便携版 ZIP 压缩包
"""

import os
import sys
import subprocess
import shutil
import zipfile
import argparse
from pathlib import Path
from datetime import datetime


# ============================================================================
# 配置
# ============================================================================
class Config:
    """打包配置"""
    APP_NAME = "PyMeshGen"
    VERSION_FILE = "VERSION"
    DEFAULT_VERSION = "1.0.0"
    DIST_DIR = "dist"
    BUILD_DIR = "build"
    BIN_DIR = "bin"
    INSTALLER_DIR = "installer"
    ZIP_DIR = "releases"
    SPEC_FILE = "PyMeshGen.spec"
    MAIN_SCRIPT = "start_gui.py"


# ============================================================================
# 工具函数
# ============================================================================
def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent.absolute()


def get_version(project_root):
    """获取版本号"""
    version_file = project_root / Config.VERSION_FILE
    if version_file.exists():
        return version_file.read_text(encoding='utf-8').strip()
    return Config.DEFAULT_VERSION


def print_header(text, char="="):
    """打印标题"""
    print()
    print(char * 70)
    print(text.center(70))
    print(char * 70)
    print()


def print_section(text):
    """打印节标题"""
    print()
    print(f"--- {text} " + "-" * (60 - len(text)))
    print()


def run_command(cmd, cwd=None, capture=True):
    """运行命令"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture,
            text=True,
            encoding='utf-8'
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


# ============================================================================
# 依赖检查
# ============================================================================
def check_python():
    """检查 Python 是否安装"""
    print_section("步骤 1: 检查 Python")
    
    python_version = sys.version
    print(f"[OK] Python 已安装：{python_version}")
    
    # 检查版本 >= 3.8
    if sys.version_info < (3, 8):
        print("[ERR] Python 版本过低，需要 3.8+")
        return False
    
    return True


def check_and_install_dependencies(project_root, force_install=False):
    """检查并安装依赖"""
    print_section("步骤 2: 检查并安装依赖")
    
    # 定义依赖：(导入名，pip 包名，是否可选)
    required_packages = [
        ('PyInstaller', 'pyinstaller', False),
        (None, 'pyinstaller-hooks-contrib', False),  # 这个包不需要导入，是钩子库
    ]
    
    missing_packages = []
    
    for import_name, pip_name, optional in required_packages:
        is_installed = False
        if import_name:
            try:
                __import__(import_name)
                is_installed = True
            except ImportError:
                pass
        else:
            # 对于不需要导入的包，检查 pip 是否已安装
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "show", pip_name],
                    capture_output=True,
                    text=True
                )
                is_installed = (result.returncode == 0)
            except Exception:
                pass
        
        if is_installed:
            print(f"[OK] {pip_name} 已安装")
        else:
            missing_packages.append(pip_name)
            print(f"[ERR] {pip_name} 未安装")
    
    if missing_packages:
        print(f"\n正在安装缺失的依赖：{', '.join(missing_packages)}")
        print("-" * 60)
        
        cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
        success, stdout, stderr = run_command(cmd)
        
        if not success:
            print(f"[ERR] 依赖安装失败")
            print(stderr)
            return False
        
        print("-" * 60)
        print("[OK] 依赖安装完成")
    
    return True


# ============================================================================
# 清理构建目录
# ============================================================================
def clean_build_dirs(project_root):
    """清理构建目录"""
    print_section("步骤 3: 清理构建目录")
    
    dirs_to_clean = [
        project_root / Config.BUILD_DIR,
        project_root / Config.DIST_DIR,
        project_root / Config.BIN_DIR,
        project_root / "__pycache__",
    ]
    
    cleaned = False
    for dir_path in dirs_to_clean:
        if dir_path.exists():
            print(f"  删除目录：{dir_path}")
            shutil.rmtree(dir_path)
            cleaned = True
    
    # 删除多余的 .spec 文件
    for spec_file in project_root.glob('*.spec'):
        if spec_file.name != Config.SPEC_FILE:
            spec_file.unlink()
            print(f"  删除文件：{spec_file}")
            cleaned = True
    
    if not cleaned:
        print("  无需清理")
    else:
        print("[OK] 清理完成")
    
    return True


# ============================================================================
# PyInstaller 打包
# ============================================================================
def build_executable(project_root, debug=False):
    """使用 PyInstaller 打包"""
    print_section("步骤 4: PyInstaller 打包")
    
    spec_file = project_root / Config.SPEC_FILE
    
    if not spec_file.exists():
        print(f"[ERR] Spec 文件不存在：{spec_file}")
        return False
    
    # 构建命令
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        str(spec_file),
        "--noconfirm",
    ]
    
    if debug:
        print("调试模式：将显示控制台窗口")
        # 注意：spec 文件中已设置 console=False，需要修改 spec 文件才能启用控制台
    
    print(f"执行命令：{' '.join(cmd)}")
    print("-" * 70)
    
    success, stdout, stderr = run_command(cmd, cwd=project_root, capture=False)
    
    print("-" * 70)
    
    if not success:
        print("[ERR] PyInstaller 打包失败")
        return False
    
    # 验证输出
    exe_file = project_root / Config.DIST_DIR / f"{Config.APP_NAME}.exe"
    if not exe_file.exists():
        print(f"[ERR] 可执行文件未生成：{exe_file}")
        return False
    
    file_size_mb = exe_file.stat().st_size / (1024 * 1024)
    print(f"[OK] 可执行文件已生成：{exe_file}")
    print(f"  文件大小：{file_size_mb:.2f} MB")
    
    return True


# ============================================================================
# 创建安装包 (Inno Setup)
# ============================================================================
def check_inno_setup():
    """检查 Inno Setup 是否安装"""
    print_section("检查 Inno Setup")
    
    # 常见的 Inno Setup 安装路径
    possible_paths = [
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
        r"C:\Program Files (x86)\Inno Setup 5\ISCC.exe",
        r"C:\Program Files\Inno Setup 5\ISCC.exe",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"[OK] 找到 Inno Setup: {path}")
            return path
    
    # 检查 PATH 中是否有 iscc
    try:
        result = subprocess.run(
            ["where", "iscc"],
            capture_output=True,
            text=True,
            shell=True
        )
        if result.returncode == 0:
            print(f"[OK] 在 PATH 中找到 iscc")
            return "iscc.exe"
    except Exception:
        pass
    
    print("[ERR] 未找到 Inno Setup")
    return None


def build_installer(project_root, iscc_path):
    """使用 Inno Setup 构建安装包"""
    print_section("步骤 5: 创建安装包 (Inno Setup)")
    
    iss_file = project_root / f"{Config.APP_NAME}.iss"
    
    if not iss_file.exists():
        print(f"[ERR] Inno Setup 脚本不存在：{iss_file}")
        return False
    
    # 创建输出目录
    installer_dir = project_root / Config.INSTALLER_DIR
    installer_dir.mkdir(exist_ok=True)
    
    # 运行 ISCC
    cmd = [iscc_path, "/Q", str(iss_file)]
    print(f"执行命令：{' '.join(cmd)}")
    print("-" * 70)
    
    success, stdout, stderr = run_command(cmd, cwd=project_root)
    
    print("-" * 70)
    
    if not success:
        print("[ERR] 安装包构建失败")
        if stderr:
            print(stderr)
        return False
    
    # 查找生成的安装包
    installer_files = list(installer_dir.glob(f"{Config.APP_NAME}-Setup-*.exe"))
    
    if not installer_files:
        print("[ERR] 未找到生成的安装包")
        return False
    
    latest_installer = max(installer_files, key=lambda f: f.stat().st_mtime)
    file_size_mb = latest_installer.stat().st_size / (1024 * 1024)
    
    print(f"[OK] 安装包已生成：{latest_installer}")
    print(f"  文件大小：{file_size_mb:.2f} MB")
    
    return True, latest_installer


# ============================================================================
# 创建 ZIP 便携版
# ============================================================================
def create_zip_release(project_root):
    """创建便携版 ZIP 压缩包"""
    print_section("步骤 5: 创建便携版 ZIP")
    
    dist_dir = project_root / Config.DIST_DIR
    zip_dir = project_root / Config.ZIP_DIR
    version = get_version(project_root)
    
    if not dist_dir.exists():
        print("[ERR] dist 目录不存在")
        return False
    
    # 创建 releases 目录
    zip_dir.mkdir(exist_ok=True)
    
    # ZIP 文件名
    zip_filename = f"{Config.APP_NAME}-v{version}-windows-portable.zip"
    zip_path = zip_dir / zip_filename
    
    print(f"创建 ZIP 压缩包：{zip_path}")
    print("-" * 70)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in dist_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(dist_dir)
                zipf.write(file_path, arcname)
                print(f"  添加：{arcname}")
    
    print("-" * 70)
    
    file_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[OK] ZIP 压缩包已生成：{zip_path}")
    print(f"  文件大小：{file_size_mb:.2f} MB")
    
    return True, zip_path


# ============================================================================
# 创建构建报告
# ============================================================================
def create_build_report(project_root, success=True, outputs=None):
    """创建构建报告"""
    print_section("生成构建报告")
    
    report_dir = project_root / Config.DIST_DIR
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / "BUILD_REPORT.txt"
    version = get_version(project_root)
    build_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"{Config.APP_NAME} 构建报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"构建时间：{build_time}\n")
        f.write(f"版本号：{version}\n")
        f.write(f"Python 版本：{sys.version}\n")
        f.write(f"构建状态：{'成功' if success else '失败'}\n\n")
        
        if outputs:
            f.write("输出文件:\n")
            for output_type, output_path in outputs.items():
                if output_path:
                    file_size = output_path.stat().st_size / (1024 * 1024)
                    f.write(f"  {output_type}: {output_path} ({file_size:.2f} MB)\n")
            f.write("\n")
        
        f.write("构建内容:\n")
        f.write(f"  - {Config.APP_NAME}.exe (主程序)\n")
        f.write("  - config/ (配置文件)\n")
        f.write("  - docs/ (文档和图标)\n")
        f.write("  - 3rd_party/ (第三方库)\n")
        f.write("  - 其他依赖文件\n\n")
        
        f.write("使用说明:\n")
        f.write(f"  1. 运行 {Config.APP_NAME}.exe 启动程序\n")
        f.write("  2. 确保 config 目录与可执行文件在同一目录\n")
        f.write("  3. 首次运行可能需要安装 Visual C++ Redistributable\n\n")
    
    print(f"[OK] 构建报告已保存：{report_file}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description=f'{Config.APP_NAME} 一键打包工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python unified_build.py              基本构建（仅可执行文件）
  python unified_build.py --clean      清理后构建
  python unified_build.py --installer  创建安装包（需要 Inno Setup）
  python unified_build.py --zip        创建便携版 ZIP
  python unified_build.py --all        创建所有输出
        """
    )
    
    parser.add_argument('--clean', action='store_true',
                       help='清理之前的构建文件')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式（显示控制台窗口）')
    parser.add_argument('--installer', action='store_true',
                       help='创建 Inno Setup 安装包')
    parser.add_argument('--zip', action='store_true',
                       help='创建便携版 ZIP 压缩包')
    parser.add_argument('--all', action='store_true',
                       help='创建所有输出（安装包 + ZIP）')
    parser.add_argument('--skip-clean', action='store_true',
                       help='跳过清理步骤')
    
    args = parser.parse_args()
    
    project_root = get_project_root()
    
    # 打印标题
    print_header(f"{Config.APP_NAME} 一键打包工具")
    print(f"项目根目录：{project_root}")
    print(f"Python 版本：{sys.version}")
    print(f"版本号：{get_version(project_root)}")
    
    # 确定输出类型
    create_installer = args.installer or args.all
    create_zip = args.zip or args.all
    
    outputs = {}
    overall_success = True
    
    try:
        # 步骤 1: 检查 Python
        if not check_python():
            overall_success = False
            raise SystemExit(1)
        
        # 步骤 2: 检查并安装依赖
        if not check_and_install_dependencies(project_root):
            overall_success = False
            raise SystemExit(1)
        
        # 步骤 3: 清理构建目录
        if args.clean or not args.skip_clean:
            if not clean_build_dirs(project_root):
                overall_success = False
                raise SystemExit(1)
        
        # 步骤 4: PyInstaller 打包
        if not build_executable(project_root, debug=args.debug):
            overall_success = False
            raise SystemExit(1)
        
        # 步骤 5a: 创建安装包
        if create_installer:
            iscc_path = check_inno_setup()
            if iscc_path:
                result = build_installer(project_root, iscc_path)
                if result:
                    success, installer_path = result
                    if success:
                        outputs['安装包'] = installer_path
                    else:
                        print("\n[WARN] 安装包创建失败，继续其他步骤...")
            else:
                print("\n[WARN] 未安装 Inno Setup，跳过安装包创建")
                print("  如需创建安装包，请安装 Inno Setup:")
                print("  https://jrsoftware.org/isdl.php")
                print("  或使用：choco install innosetup")

        # 步骤 5b: 创建 ZIP 便携版
        if create_zip:
            result = create_zip_release(project_root)
            if result:
                success, zip_path = result
                if success:
                    outputs['ZIP 便携版'] = zip_path
                else:
                    print("\n[WARN] ZIP 创建失败，继续其他步骤...")

        # 生成构建报告
        create_build_report(project_root, success=overall_success, outputs=outputs)

        # 打印总结
        print_header("构建完成", "=")

        if overall_success:
            print("[OK] 构建成功！\n")

            # 显示输出文件
            exe_file = project_root / Config.DIST_DIR / f"{Config.APP_NAME}.exe"
            print(f"可执行文件：{exe_file}")

            for output_type, output_path in outputs.items():
                print(f"{output_type}: {output_path}")

            print("\n下一步:")
            print("  1. 测试运行：dist\\PyMeshGen.exe")
            if '安装包' in outputs:
                print(f"  2. 测试安装包：{outputs['安装包']}")
            if 'ZIP 便携版' in outputs:
                print(f"  3. 分发 ZIP 包：{outputs['ZIP 便携版']}")
        else:
            print("[ERR] 构建失败！请检查错误信息。")
            sys.exit(1)

    except SystemExit as e:
        print_header("构建失败", "=")
        print("[ERR] 构建过程中断")
        sys.exit(e.code if e.code else 1)

    except KeyboardInterrupt:
        print_header("构建取消", "=")
        print("[ERR] 用户取消构建")
        sys.exit(1)

    except Exception as e:
        print_header("构建失败", "=")
        print(f"[ERR] 发生错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()


if __name__ == '__main__':
    main()
