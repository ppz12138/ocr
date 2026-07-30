#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR引擎自动下载和配置脚本
支持 Umi-OCR (主引擎) 和 PaddleOCR-json (备用引擎)

使用方法: 
  python setup_engines.py              # 下载所有引擎
  python setup_engines.py umi           # 只下载 Umi-OCR
  python setup_engines.py paddle        # 只下载 PaddleOCR-json
  python setup_engines.py --check       # 检查引擎状态

首次使用请先运行: pip install py7zr requests
"""

import os
import sys
import json
import zipfile
import shutil
import urllib.request
import tempfile
import subprocess

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

# 加载配置
config_path = os.path.join(ROOT_DIR, "config", "config.json")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
    """下载进度条"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '░' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    if iteration == total: 
        print()


def download_file(url, save_path, desc="下载中"):
    """下载文件并显示进度"""
    print(f"\n{desc}")
    print(f"  链接: {url[:80]}...")
    print(f"  保存: {save_path}")
    
    # 创建临时文件
    temp_path = save_path + ".downloading"
    
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Python-OCR-Setup')
        
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('content-length', 0))
            
            if total_size == 0:
                print("  ⚠️  无法获取文件大小，下载可能较慢...")
            
            block_size = 8192
            downloaded = 0
            
            with open(temp_path, 'wb') as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        print_progress_bar(downloaded, total_size, prefix='  进度:')
                    else:
                        # 显示已下载大小
                        mb_downloaded = downloaded / (1024 * 1024)
                        print(f'\r  进度: {mb_downloaded:.1f} MB', end='\r')
            
            if total_size == 0:
                print(f"\n  完成！文件大小: {downloaded / (1024 * 1024):.1f} MB")
        
        # 下载完成后重命名
        if os.path.exists(temp_path):
            os.rename(temp_path, save_path)
        return True
    except Exception as e:
        print(f"\n  ❌ 下载失败: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


def get_latest_release_url(repo_name, asset_keyword=None):
    """从 GitHub API 获取最新 release 的下载链接"""
    api_url = f"https://api.github.com/repos/{repo_name}/releases/latest"
    
    try:
        req = urllib.request.Request(api_url)
        req.add_header('User-Agent', 'Python-OCR-Setup')
        
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read())
            
        # 获取所有资产
        assets = data.get('assets', [])
        if not assets:
            print(f"  ⚠️  {repo_name} 没有可用的下载资产")
            return None, None, None
        
        version = data.get('tag_name', 'unknown')
        
        # 优先匹配关键词 (如 windows x64)
        if asset_keyword:
            for asset in assets:
                if asset_keyword.lower() in asset['name'].lower():
                    return asset['browser_download_url'], asset['name'], version
        
        # 否则返回第一个资产
        return assets[0]['browser_download_url'], assets[0]['name'], version
    except Exception as e:
        print(f"  ❌ 获取 {repo_name} 最新版本失败: {str(e)}")
        return None, None, None


def extract_zip(zip_path, extract_to):
    """解压 zip 文件"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取压缩包内的顶层目录
            zip_names = zip_ref.namelist()
            if zip_names:
                # 解压到临时目录
                temp_dir = tempfile.mkdtemp()
                zip_ref.extractall(temp_dir)
                
                # 检查是否有顶层目录
                top_items = os.listdir(temp_dir)
                if len(top_items) == 1 and os.path.isdir(os.path.join(temp_dir, top_items[0])):
                    # 有顶层目录，移动其内容
                    source_dir = os.path.join(temp_dir, top_items[0])
                else:
                    source_dir = temp_dir
                
                # 如果目标目录已存在，删除
                if os.path.exists(extract_to):
                    shutil.rmtree(extract_to)
                
                # 移动解压的内容
                shutil.move(source_dir, extract_to)
                
                # 清理临时目录
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                return True
    except Exception as e:
        print(f"  ❌ ZIP解压失败: {str(e)}")
    return False


def extract_7z(zip_path, extract_to):
    """解压 7z 文件"""
    # 方法1: 尝试使用 py7zr 库
    try:
        import py7zr
        try:
            # 清理旧目录
            if os.path.exists(extract_to):
                shutil.rmtree(extract_to)
            os.makedirs(extract_to, exist_ok=True)
            
            with py7zr.SevenZipFile(zip_path, mode='r') as z:
                z.extractall(path=extract_to)
            
            # 检查是否有顶层目录需要展开
            top_items = os.listdir(extract_to)
            if len(top_items) == 1 and os.path.isdir(os.path.join(extract_to, top_items[0])):
                inner_dir = os.path.join(extract_to, top_items[0])
                for item in os.listdir(inner_dir):
                    shutil.move(os.path.join(inner_dir, item), os.path.join(extract_to, item))
                os.rmdir(inner_dir)
            
            return True
        except Exception as e:
            print(f"  py7zr 解压失败: {str(e)}")
    except ImportError:
        pass
    
    # 方法2: 尝试使用 7z 命令行工具
    try:
        result = subprocess.run(['7z', 'x', zip_path, f'-o{extract_to}', '-y'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            print(f"  7z 命令行解压失败: {result.stderr}")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"  7z 命令行解压错误: {str(e)}")
    
    # 方法3: 尝试使用 PowerShell 的 Expand-Archive (不支持7z)
    print("  ⚠️  需要安装解压工具来处理 .7z 文件")
    print("  建议: pip install py7zr")
    print("  或下载 7-Zip: https://www.7-zip.org/")
    return False


def extract_self_extracting_7z(zip_path, extract_to):
    """处理自解压 .7z.exe 文件（Umi-OCR 使用这种格式）"""
    # 尝试直接运行自解压程序
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 复制文件到临时目录
        temp_exe = os.path.join(temp_dir, os.path.basename(zip_path))
        shutil.copy2(zip_path, temp_exe)
        
        # 运行自解压程序到指定目录
        print("  正在运行自解压程序...")
        result = subprocess.run(
            [temp_exe, f'-o{extract_to}', '-y'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"  ⚠️  自解压失败 (返回码: {result.returncode})")
            # 尝试不带参数运行
            print("  尝试备用方案...")
            return False
    except Exception as e:
        print(f"  ⚠️  自解压出错: {str(e)}")
        return False
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)


def extract_file(file_path, extract_to):
    """根据文件扩展名选择解压方式"""
    print("  解压中...")
    
    if file_path.endswith('.zip'):
        return extract_zip(file_path, extract_to)
    elif file_path.endswith('.7z'):
        return extract_7z(file_path, extract_to)
    elif file_path.endswith('.7z.exe'):
        return extract_self_extracting_7z(file_path, extract_to)
    elif file_path.endswith('.exe'):
        # Umi-OCR 的自解压格式
        return extract_self_extracting_7z(file_path, extract_to)
    else:
        print(f"  ❌ 不支持的文件格式: {os.path.splitext(file_path)[1]}")
        return False


def check_engine_status():
    """检查引擎安装状态"""
    engines = {
        'Umi-OCR': {
            'path': os.path.join(ROOT_DIR, "engines", "Umi-OCR", "Umi-OCR.exe"),
            'url': 'https://github.com/hiroi-sora/Umi-OCR/releases'
        },
        'PaddleOCR-json': {
            'path': os.path.join(ROOT_DIR, "engines", "PaddleOCR", "PaddleOCR-json_v1.4.1", "PaddleOCR-json.exe"),
            'url': 'https://github.com/hiroi-sora/PaddleOCR-json/releases'
        }
    }
    
    print("\n" + "=" * 60)
    print("📋 引擎状态检查")
    print("=" * 60)
    
    for name, info in engines.items():
        if os.path.exists(info['path']):
            print(f"  ✓ {name}: 已安装")
        else:
            print(f"  ✗ {name}: 未安装")
            print(f"    下载: {info['url']}")
    
    print()


def setup_umio_cr():
    """下载和配置 Umi-OCR 引擎"""
    engine_dir = os.path.join(ROOT_DIR, "engines", "Umi-OCR")
    exe_path = os.path.join(engine_dir, "Umi-OCR.exe")
    
    # 检查是否已安装
    if os.path.exists(exe_path):
        print("  ✓ Umi-OCR 引擎已安装")
        return True
    
    # 获取下载链接
    url, filename, version = get_latest_release_url("hiroi-sora/Umi-OCR", "windows")
    if not url:
        print("  ❌ 无法获取 Umi-OCR 下载链接")
        print("  请手动下载: https://github.com/hiroi-sora/Umi-OCR/releases")
        return False
    
    print(f"\n  📌 Umi-OCR 版本: {version}")
    print(f"  📎 文件: {filename}")
    
    # 创建目录
    os.makedirs(engine_dir, exist_ok=True)
    
    # 下载到临时目录
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, filename)
    
    try:
        if not download_file(url, temp_file, "下载 Umi-OCR 引擎..."):
            return False
        
        # 解压
        print("  解压中...")
        success = extract_file(temp_file, engine_dir)
        
        if success and os.path.exists(exe_path):
            print(f"  ✓ Umi-OCR 引擎安装成功！")
            return True
        else:
            # 如果解压失败，保留下载的文件供用户手动处理
            dest_file = os.path.join(engine_dir, filename)
            shutil.copy2(temp_file, dest_file)
            print(f"\n  ⚠️  自动解压失败，但文件已下载到:")
            print(f"     {dest_file}")
            print(f"  请手动解压此文件到:")
            print(f"     {engine_dir}")
            return False
    except Exception as e:
        print(f"  ❌ 安装过程出错: {str(e)}")
        return False
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)


def setup_paddleocr():
    """下载和配置 PaddleOCR-json 引擎"""
    engine_dir = os.path.join(ROOT_DIR, "engines", "PaddleOCR")
    exe_path = os.path.join(engine_dir, "PaddleOCR-json_v1.4.1", "PaddleOCR-json.exe")
    
    # 检查是否已安装
    if os.path.exists(exe_path):
        print("  ✓ PaddleOCR-json 引擎已安装")
        return True
    
    # 获取下载链接
    url, filename, version = get_latest_release_url("hiroi-sora/PaddleOCR-json", "windows")
    if not url:
        print("  ❌ 无法获取 PaddleOCR-json 下载链接")
        print("  请手动下载: https://github.com/hiroi-sora/PaddleOCR-json/releases")
        return False
    
    print(f"\n  📌 PaddleOCR-json 版本: {version}")
    print(f"  📎 文件: {filename}")
    
    # 创建目录
    os.makedirs(engine_dir, exist_ok=True)
    
    # 下载到临时目录
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, filename)
    
    try:
        if not download_file(url, temp_file, "下载 PaddleOCR-json 引擎..."):
            return False
        
        # 解压
        print("  解压中...")
        success = extract_file(temp_file, engine_dir)
        
        if success and os.path.exists(exe_path):
            print(f"  ✓ PaddleOCR-json 引擎安装成功！")
            return True
        else:
            # 如果解压失败，保留下载的文件供用户手动处理
            dest_file = os.path.join(engine_dir, filename)
            shutil.copy2(temp_file, dest_file)
            print(f"\n  ⚠️  自动解压失败，但文件已下载到:")
            print(f"     {dest_file}")
            print(f"  请手动解压此文件到:")
            print(f"     {engine_dir}")
            return False
    except Exception as e:
        print(f"  ❌ 安装过程出错: {str(e)}")
        return False
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    print("\n" + "=" * 60)
    print("🚀 房产OCR项目 - 引擎自动配置工具")
    print("=" * 60)
    print()
    print("此工具将自动下载和配置 OCR 引擎")
    print("需要联网访问 GitHub")
    print()
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == '--check':
            check_engine_status()
            return
        elif arg == 'umi':
            # 只安装 Umi-OCR
            success = setup_umio_cr()
            sys.exit(0 if success else 1)
        elif arg == 'paddle':
            # 只安装 PaddleOCR
            success = setup_paddleocr()
            sys.exit(0 if success else 1)
        else:
            print(f"未知参数: {arg}")
            print("使用方法:")
            print("  python setup_engines.py          # 下载所有引擎")
            print("  python setup_engines.py umi       # 只下载 Umi-OCR")
            print("  python setup_engines.py paddle    # 只下载 PaddleOCR")
            print("  python setup_engines.py --check   # 检查引擎状态")
            sys.exit(1)
    
    # 检查 py7zr 是否可用
    try:
        import py7zr
        print("  ✓ py7zr 库已安装 (支持 .7z 解压)")
    except ImportError:
        print("  ⚠️  建议安装 py7zr 库以支持 .7z 格式解压")
        print("     pip install py7zr")
        print()
    
    # 主引擎：Umi-OCR (必需)
    print("\n" + "-" * 60)
    print("📦 安装 Umi-OCR 引擎 (主引擎)")
    print("-" * 60)
    umi_success = setup_umio_cr()
    
    # 备用引擎：PaddleOCR-json (可选)
    print("\n" + "-" * 60)
    print("📦 安装 PaddleOCR-json 引擎 (可选备用)")
    print("-" * 60)
    paddle_success = setup_paddleocr()
    
    # 汇总
    print("\n" + "=" * 60)
    print("📋 安装结果汇总")
    print("=" * 60)
    
    if umi_success:
        print("  ✓ Umi-OCR 引擎: 就绪")
    else:
        print("  ✗ Umi-OCR 引擎: 未安装 (必需)")
    
    if paddle_success:
        print("  ✓ PaddleOCR-json 引擎: 就绪")
    else:
        print("  ○ PaddleOCR-json 引擎: 未安装 (可选)")
    
    # 检查引擎状态
    check_engine_status()
    
    if not umi_success:
        print("\n💡 手动安装提示:")
        print("  1. 访问 https://github.com/hiroi-sora/Umi-OCR/releases")
        print("  2. 下载最新的 Windows 版本 (.7z.exe)")
        print("  3. 运行自解压程序，将文件解压到 engines/Umi-OCR/ 目录")
        print()
    
    print("✅ 配置完成！")


if __name__ == "__main__":
    main()
