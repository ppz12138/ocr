#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房产数据处理流程3：OCR识别
从图片中提取房产喜报信息
"""

# ==========================================
# 1. 导入模块区
# ==========================================
# 基础模块
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
import requests
import json
import sys
import shutil
import traceback

# 按键读取模块已不需要，直接用input()

# 图像处理库
from PIL import Image

# 全局标志：是否已经检查过OCR引擎可用性
_OCR_ENGINE_CHECKED = False


# ==========================================
# 2. 配置常量区
# ==========================================

# 全局变量
_full_records_with_id = []  # 存储带有ID的完整记录
ocr_engine = None  # OCR引擎实例
known_communities = {}  # 已知小区字典
known_persons = []  # 已知人员列表
dict_changed = False  # 字典是否有变化的标志，只有在用户确认写入后才会变为True

# 配置管理类
class Config:
    def __init__(self):
        # 基础配置
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.join(self.base_dir, "..")
        
        # 加载配置文件
        config_path = os.path.join(self.root_dir, "config", "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config_data = json.load(f)
        
        # PaddleOCR相关配置
        self.ocr_api_dir = os.path.join(self.root_dir, self.config_data["paddle_ocr"]["api_dir"])
        self.ocr_exe_path = os.path.join(self.root_dir, self.config_data["paddle_ocr"]["exe_path"])
        self.ocr_models_path = os.path.join(self.root_dir, self.config_data["paddle_ocr"]["models_path"])
        
        # Umi-OCR相关配置
        self.umi_ocr_path = os.path.join(self.root_dir, self.config_data["umi_ocr"]["exe_path"])
        
        # 输出目录
        self.output_dir = os.path.join(self.root_dir, self.config_data["data"]["output_dir"])
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 字典文件路径
        self.community_dict_file = os.path.join(self.root_dir, self.config_data["data"]["community_dict"])
        self.person_dict_file = os.path.join(self.root_dir, self.config_data["data"]["person_dict"])
        
        # WPS AirScript配置
        self.wps_config = self.config_data["wps"]
        
        # 已知店名列表
        self.known_store_names = self.config_data["known_store_names"]
        
        # 引擎检测
        self._check_engines()
    
    def _check_engines(self):
        """检测OCR引擎是否可用，打印友好提示"""
        umi_available = os.path.exists(self.umi_ocr_path)
        paddle_available = os.path.exists(self.ocr_exe_path) and os.path.isdir(self.ocr_api_dir)
        
        if not umi_available:
            umi_url = self.config_data["umi_ocr"].get("download_url", "")
            print(f"\n⚠️ 警告：Umi-OCR引擎未找到！")
            print(f"   预期路径: {self.umi_ocr_path}")
            print(f"   可运行: python setup_engines.py 自动下载安装")
            print(f"   或手动下载: {umi_url}\n")
        
        if not paddle_available:
            paddle_url = self.config_data["paddle_ocr"].get("download_url", "")
            print(f"⚠️ 提示：PaddleOCR-json备用引擎未找到")
            print(f"   预期路径: {self.ocr_exe_path}")
            print(f"   可运行: python setup_engines.py 自动下载安装")
            print(f"   或手动下载: {paddle_url}\n")
        
        if umi_available:
            print("✓ Umi-OCR 引擎就绪")
        if paddle_available:
            print("✓ PaddleOCR-json 引擎就绪（备用）")
        if not umi_available and not paddle_available:
            print("❌ 没有可用的OCR引擎！")
            print("   快速安装: python setup_engines.py")
            print("   详细说明请查看 config/config.json 中的 _comment 字段\n")

# 创建全局配置实例
config = Config()

# 将API目录添加到系统路径的最前面，确保Python优先搜索这个目录
sys.path.insert(0, config.ocr_api_dir)

# OCR库 - 延迟导入，避免启动时依赖错误
PPOCR_AVAILABLE = True
try:
    from PPOCR_api import GetOcrApi
except ImportError:
    PPOCR_AVAILABLE = False
    print("信息：PPOCR_api模块未找到，PaddleOCR-json引擎将不可用，将使用Umi-OCR作为唯一OCR引擎")


# ==========================================
# 3. 通用工具函数模块
# ==========================================


def get_key_input(prompt=""):
    """读取用户输入，支持输入'e'退出"""
    # 直接用普通input()，支持所有符号输入
    result = input(prompt).strip()
    if result.lower() == 'e':
        return "esc"
    return result


# 相似度计算函数
def calculate_similarity(str1, str2):
    """
    计算两个字符串的相似度（基于Levenshtein距离）
    返回相似度百分比，范围0-100
    """
    # 保留原始字符串，不做过滤
    if not str1 or not str2:
        return 0
    
    # 计算Levenshtein距离
    len1, len2 = len(str1), len(str2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if str1[i-1] == str2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    
    # 计算相似度百分比
    max_len = max(len1, len2)
    similarity = ((max_len - dp[len1][len2]) / max_len) * 100
    
    return similarity


# 通用相似度匹配函数
def match_with_similarity(extracted_name, known_names, threshold=70.0, item_type="名称", is_dict=False):
    """
    通用相似度匹配函数，用于小区名和人名的相似度匹配
    
    参数:
        extracted_name: str - 提取的名称
        known_names: dict or list - 已知名称列表或字典
        threshold: float - 相似度阈值
        item_type: str - 项目类型，用于生成备注
        is_dict: bool - 标识known_names是否为字典
    
    返回:
        tuple - (str, str) 匹配后的名称和备注信息
    """
    final_name = ""
    note = ""
    
    # 1. 精确匹配（简化逻辑，字典和列表的in操作符行为一致）
    if extracted_name in known_names:
        # 如果是小区字典的新格式，需要返回name字段
        if is_dict and item_type == "小区":
            known_item = known_names[extracted_name]
            if isinstance(known_item, dict) and "name" in known_item:
                return known_item["name"], ""
        return extracted_name, ""
    
    # 2. 相似度匹配
    best_match = None
    highest_similarity = 0.0
    
    # 遍历已知名称
    for known_name in known_names:
        if not known_name:
            continue
        
        # 处理小区字典的新格式
        current_name = known_name
        if is_dict and item_type == "小区":
            known_item = known_names[known_name]
            if isinstance(known_item, dict) and "name" in known_item:
                current_name = known_item["name"]
        
        # 计算相似度
        similarity = calculate_similarity(extracted_name, current_name)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = current_name
    
    # 3. 根据相似度阈值决定结果
    if best_match and highest_similarity >= threshold:
        # 相似度匹配成功
        final_name = best_match
        if highest_similarity < 100.0:
            note = f"[{item_type}相似匹配：{extracted_name}→{best_match}，匹配度：{highest_similarity:.1f}%]"
    else:
        # 使用提取的名称
        final_name = extracted_name
    
    return final_name, note


# ==========================================
# 4. 字典管理模块
# ==========================================


# 通用字典管理函数

def _init_dict(dict_name, file_path, dict_obj, expected_type):
    """
    通用字典初始化函数
    
    参数:
        dict_name: str - 字典名称
        file_path: str - 文件路径
        dict_obj: dict/list - 字典对象
        expected_type: type - 期望的数据类型
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, expected_type) and loaded:  # 只接受非空对象
                    if isinstance(dict_obj, dict):
                        dict_obj.clear()
                        dict_obj.update(loaded)
                    elif isinstance(dict_obj, list):
                        dict_obj.clear()
                        dict_obj.extend(loaded)
                    print(f"从文件加载了 {len(loaded)} 个{dict_name}")
        except Exception as e:
            print(f"加载{dict_name}字典失败，保持原有数据不变：{e}")


def _save_dict(dict_name, file_path, dict_obj):
    """
    通用字典保存函数
    
    参数:
        dict_name: str - 字典名称
        file_path: str - 文件路径
        dict_obj: dict/list - 字典对象
    """
    if not dict_obj:
        print(f"{dict_name}字典为空，跳过保存")
        return
    try:
        # 简单备份：只在文件存在时备份
        if os.path.exists(file_path):
            shutil.copy2(file_path, file_path + ".bak")
        # 保存新数据
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dict_obj, f, ensure_ascii=False, indent=4)
        print(f"{dict_name}字典已保存，包含 {len(dict_obj)} 个{dict_name}")
    except Exception as e:
        print(f"保存{dict_name}字典失败：{e}")


# 小区字典管理函数（支持区域分类）

def init_community_dict():
    """
    初始化小区字典
    从文件加载已知小区列表，支持新旧格式兼容（包括混合格式）
    """
    global known_communities
    try:
        if os.path.exists(config.community_dict_file):
            with open(config.community_dict_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict) and loaded:
                    known_communities.clear()
                    converted_count = 0
                    # 逐个检查并转换格式，确保所有条目都是新格式
                    for key, value in loaded.items():
                        if isinstance(value, str):
                            # 旧格式转换为新格式
                            known_communities[key] = {
                                "name": value,
                                "area": "非大岭山"
                            }
                            converted_count += 1
                        elif isinstance(value, dict) and "name" in value:
                            # 已经是新格式，直接使用
                            known_communities[key] = value
                        else:
                            # 其他情况，确保有name字段
                            known_communities[key] = {
                                "name": key,
                                "area": "非大岭山"
                            }
                    
                    if converted_count > 0:
                        print(f"从文件加载了 {len(loaded)} 个小区名（转换了 {converted_count} 个旧格式条目）")
                    else:
                        print(f"从文件加载了 {len(loaded)} 个小区名")
    except Exception as e:
        print(f"加载小区名字典失败，保持原有数据不变：{e}")


def save_community_dict():
    """
    保存小区名字典到文件
    """
    global known_communities
    if not known_communities:
        print("小区名字典为空，跳过保存")
        return
    try:
        # 简单备份
        if os.path.exists(config.community_dict_file):
            shutil.copy2(config.community_dict_file, config.community_dict_file + ".bak")
        with open(config.community_dict_file, "w", encoding="utf-8") as f:
            json.dump(known_communities, f, ensure_ascii=False, indent=4)
        print(f"小区名字典已保存，包含 {len(known_communities)} 个小区名")
    except Exception as e:
        print(f"保存小区名字典失败：{e}")


def get_community_area(community_name):
    """
    获取小区所属区域
    
    参数:
        community_name: str - 小区名称
        
    返回:
        str - "大岭山" 或 "非大岭山"
    """
    global known_communities
    if community_name in known_communities:
        community_info = known_communities[community_name]
        if isinstance(community_info, dict):
            return community_info.get("area", "非大岭山")
    return "非大岭山"


def add_new_community(community_name, area="非大岭山"):
    """
    添加新小区到字典
    
    参数:
        community_name: str - 小区名称
        area: str - 区域，默认为非大岭山
    """
    global known_communities
    if community_name and community_name not in known_communities:
        known_communities[community_name] = {
            "name": community_name,
            "area": area
        }


# 人员字典管理函数（保持原有函数签名）

def init_person_dict():
    """
    初始化人员字典
    从文件加载已知人员列表，失败则保持原有数据不变
    """
    global known_persons
    _init_dict("人员名", config.person_dict_file, known_persons, list)


def save_person_dict():
    """
    保存人员字典到文件
    防止保存空列表，添加简单备份
    """
    global known_persons
    _save_dict("人员名", config.person_dict_file, known_persons)


# ==========================================
# 5. OCR核心模块
# ==========================================

# OCR引擎初始化和关闭
def init_ocr_engine():
    """
    检查并初始化OCR引擎
    现在Umi-OCR是主引擎，PaddleOCR-json是备用引擎
    """
    global ocr_engine, _OCR_ENGINE_CHECKED
    
    # 检查并初始化PaddleOCR-json作为备用引擎
    if not ocr_engine:
        if PPOCR_AVAILABLE:
            try:
                # 初始化PaddleOCR-json引擎
                ocr_engine = GetOcrApi(
                    exePath=config.ocr_exe_path,
                    modelsPath=config.ocr_models_path,
                    argument={
                        'enable_mkldnn': True,  # 启用MKLDNN加速
                        'limit_side_len': 2880,  # 图像边长限制，提高分辨率以提升识别精度
                        'det': True,  # 启用det目标识别
                        'cls': False,  # 禁用cls方向分类
                        'use_angle_cls': False,  # 禁用方向分类
                        'det_model_dir': os.path.join(config.ocr_models_path, 'ch_PP-OCRv3_det_infer'),
                        'rec_model_dir': os.path.join(config.ocr_models_path, 'ch_PP-OCRv3_rec_infer'),
                        'cls_model_dir': os.path.join(config.ocr_models_path, 'ch_ppocr_mobile_v2.0_cls_infer'),
                        'rec_char_dict_path': os.path.join(config.ocr_models_path, 'dict_chinese.txt')
                    },
                    ipcMode='pipe'  # 使用管道模式
                )
                print("PaddleOCR-json引擎初始化成功（备用引擎）")
                _OCR_ENGINE_CHECKED = True
                return True
            except Exception as e:
                print(f"PaddleOCR-json引擎初始化失败：{str(e)}")
                _OCR_ENGINE_CHECKED = True
                return False
        else:
            # 只在首次检查时打印一次提示
            if not _OCR_ENGINE_CHECKED:
                print("信息：PPOCR_api模块未找到，PaddleOCR-json引擎将不可用，将使用Umi-OCR作为唯一OCR引擎")
                _OCR_ENGINE_CHECKED = True
            ocr_engine = True  # 设置一个标记，表示已经检查过并确认不可用
            return True
    return True


def close_ocr_engine():
    """
    关闭OCR引擎，释放资源
    """
    global ocr_engine
    
    # 只处理PaddleOCR-json引擎，Umi-OCR作为命令行工具不需要关闭
    if PPOCR_AVAILABLE and ocr_engine and ocr_engine is not True:
        try:
            # 调用引擎的关闭方法（如果存在）
            if hasattr(ocr_engine, 'exit'):
                ocr_engine.exit()
            ocr_engine = None
            print("PaddleOCR-json引擎已关闭")
        except Exception as e:
            print(f"关闭PaddleOCR-json引擎时发生错误：{str(e)}")
    else:
        # 对于Umi-OCR或未初始化的引擎，不需要关闭操作
        ocr_engine = None

# 辅助函数：清理OCR输出文本
def clean_ocr_output(text):
    """
    清理OCR输出文本，过滤掉命令行提示和空行
    
    参数:
        text: str - 原始OCR输出文本
    
    返回:
        str - 清理后的文本
    """
    if not text:
        return ""
    
    # 定义需要过滤的前缀
    filter_prefixes = ('usage:', 'optional arguments:', 'positional arguments:', 
                      'Python modules:', 'Qml modules:', 'Tips:', '[OCR引擎]', 'Umi-OCR')
    
    # 清理文本，过滤掉空行和特定前缀的行
    clean_lines = [line.strip() for line in text.split('\n') 
                  if line.strip() and not line.strip().startswith(filter_prefixes)]
    
    return '\n'.join(clean_lines)


# 辅助函数：打印带分隔线的标题
def print_section(title, width=80):
    """
    打印带分隔线的标题，用于分割不同的功能模块
    
    参数:
        title: str - 标题文本
        width: int - 分隔线宽度
    """
    print(f'\n{"="*width}')
    print(title)
    print(f'={"="*width}')


# 辅助函数：获取文件夹中的图片文件
def get_image_files(folder):
    """
    获取文件夹中的所有图片文件，支持多种图片格式
    
    参数:
        folder: str - 文件夹路径
    
    返回:
        list - 图片文件路径列表
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    return [os.path.join(folder, f) for f in os.listdir(folder) 
            if f.lower().endswith(image_extensions)]


# 文本提取核心函数
def extract_text_from_image(image_path, original_image_path=None, use_backup_engine=False):
    """
    使用OCR从图片中提取文本，提高中文识别精度
    优先使用Umi-OCR，失败后回退到PaddleOCR-json
    
    参数:
        image_path: str - 图片文件路径
        original_image_path: str - 原始图片文件路径（用于显示）
        use_backup_engine: bool - 是否直接使用备用引擎PaddleOCR-json
    
    返回:
        tuple - (str, bool) 提取的文本和是否使用了Umi-OCR的标志
    """
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        return "", False
    
    # 优先使用Umi-OCR，按照对照试验中的方式调用
    umi_ocr_path = os.path.join(config.root_dir, config.config_data["umi_ocr"]["exe_path"])
    
    # 确保OCR引擎已初始化（只有当PPOCR可用时才需要初始化）
    global ocr_engine
    if PPOCR_AVAILABLE and not init_ocr_engine():
        return "", False
    
    # 单次尝试，重试逻辑由调用方process_property_image统一管理
    temp_file_path = None
    try:
        # 创建临时文件用于保存Umi-OCR输出
        with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', suffix='.txt', delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        # 构建命令，使用--path和--output参数，默认使用single_para模式
        command = f'"{umi_ocr_path}" --path "{image_path}" --output "{temp_file_path}"'
        
        # 执行命令
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        
        # 读取输出文件
        try:
            with open(temp_file_path, 'r', encoding='utf-8') as f_temp:
                text = f_temp.read().strip()
            
            # 清理输出，只保留OCR识别的文本
            clean_text = clean_ocr_output(text)
            if clean_text:
                return clean_text, True
        except Exception as e:
            print(f"读取Umi-OCR输出文件失败：{str(e)}")
        
        # 如果临时文件方式失败，尝试直接从输出中提取
        combined_output = result.stdout + result.stderr
        clean_text = clean_ocr_output(combined_output)
        if clean_text:
            return clean_text, True
        
        print(f"Umi-OCR调用失败")
    except Exception as e:
        print(f"Umi-OCR调用失败：{str(e)}")
    finally:
        # 确保临时文件被删除
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
    
    # 如果Umi-OCR失败，返回空文本，由调用方决定是否重试或切换引擎
    return "", True


# ==========================================
# 6. 字段提取函数模块
# ==========================================


# 日期提取函数
def extract_date(text, image_path=None):
    """
    从文本中提取日期和精确时间
    
    参数:
        text: str - OCR提取的文本
        image_path: str - 图片文件路径（用于获取文件时间）
    
    返回:
        tuple - (str, str) 日期和精确时间
    """
    # 初始化结果
    current_date = datetime.now().strftime("%Y-%m-%d")
    final_date = current_date
    exact_time = f"{current_date} 00:00:00"
    
    # 合并日期提取模式，支持带时间和不带时间的格式
    date_patterns = [
        # 匹配带时间的日期格式，如"成交时间：2025-12-20 14:30:00"
        r"(?:成交时间|签约时间|认购时间)[:：]?\s*(\d{4}-\d{2}-\d{2})\s*([\d:：]+)",
        # 匹配带时间的YYYYMMDD格式，如"认购时间：20260529 14:30"
        r"(?:成交时间|签约时间|认购时间)[:：]?\s*(\d{4})(\d{2})(\d{2})\s*([\d:：]+)",
        # 匹配日期和时间用点号连接的格式，如"2025-12-26.18:51:01"
        r"(\d{4}-\d{2}-\d{2})\.([\d:：]+)",
        # 匹配日期和时间直接连接的格式
        r"(\d{4}-\d{2}-\d{2})([\d:：]+)",
        # 匹配单独的日期格式，如"2025-12-20"
        r"(?:成交时间|签约时间|认购时间)[:：]?\s*(\d{4}-\d{2}-\d{2})",
        # 匹配单独的YYYYMMDD格式，如"20260529"
        r"(?:成交时间|签约时间|认购时间)[:：]?\s*(\d{4})(\d{2})(\d{2})",
        r"(\d{4}-\d{2}-\d{2})"
    ]
    
    # 提取日期和时间
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                groups = match.groups()
                date_str = ""
                time_part = ""
                
                # 判断是哪种模式
                if len(groups) >= 3 and groups[0] and groups[1] and groups[2] and len(groups[0]) == 4 and len(groups[1]) == 2 and len(groups[2]) == 2:
                    # YYYYMMDD格式
                    year = groups[0]
                    month = groups[1]
                    day = groups[2]
                    date_str = f"{year}-{month}-{day}"
                    
                    # 检查是否有时间
                    if len(groups) >= 4 and groups[3]:
                        time_part = groups[3]
                else:
                    # 普通格式
                    date_str = groups[0]
                    if len(groups) >= 2 and groups[1]:
                        time_part = groups[1]
                
                # 验证日期格式
                datetime.strptime(date_str, "%Y-%m-%d")
                final_date = date_str
                
                # 提取时间（如果有）
                if time_part:
                    cleaned_time = re.sub(r"[^0-9]", "", time_part)
                    
                    # 简化时间规范化逻辑
                    if len(cleaned_time) >= 4:
                        hours = int(cleaned_time[:2])
                        minutes = int(cleaned_time[2:4])
                        seconds = int(cleaned_time[4:6]) if len(cleaned_time) >= 6 else 0
                        
                        if 0 <= hours < 24 and 0 <= minutes < 60 and 0 <= seconds < 60:
                            normalized_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                            exact_time = f"{date_str} {normalized_time}"
                        else:
                            exact_time = f"{date_str} 00:00:00"
                    else:
                        exact_time = f"{date_str} 00:00:00"
                else:
                    exact_time = f"{date_str} 00:00:00"
                break
            except ValueError:
                continue
    
    return final_date, exact_time


# 类型提取函数
def extract_type(text, image_path=None):
    """
    从文本中提取房产类型
    
    参数:
        text: str - OCR提取的文本
        image_path: str - 图片文件路径（用于获取分辨率信息）
    
    返回:
        str - 房产类型（租赁、新房、中盘、二手）
    """
    # 1. 首先通过分辨率判断租赁类型（只有租赁是450*800）
    is_rental = False
    if image_path and os.path.exists(image_path):
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if (width, height) == (450, 800):
                    is_rental = True
                    return "租赁"
        except Exception as e:
            pass
    
    if not is_rental:
        # 2. 解析金额，用于后续判断
        has_amount = False
        amount = 0
        
        # 先简单提取金额进行判断
        amount_patterns = [
            r"(\d+(?:\.\d+)?)(?:\s*万|万元)",  # 99万、216万
            r"合同金额[:：]?\s*(\d+)",  # 合同金额：2800
            r"成交价[:：]?\s*(\d+)",  # 成交价：99万
            r"成交金额[:：]?\s*(\d+)",  # 成交金额：216万
            r"(\d+(?:\.\d+)?)万",  # 匹配单独的数字+万字，如192万（支持后面跟着其他文字）
        ]
        
        for pattern in amount_patterns:
            amount_match = re.search(pattern, text)
            if amount_match:
                amount_str = amount_match.group(1)
                try:
                    amount = int(float(amount_str))
                    has_amount = True
                except ValueError:
                    pass
                break
        
        # 3. 判断其他类型
        # 二手：二手开单关键词优先判断
        if re.search(r"二手开单", text):
            return "二手"
        # 新房：成交项目+成交价为空，或认购时间
        if (re.search(r"成交项目", text) and not has_amount) or re.search(r"认购时间", text):
            return "新房"
        # 中盘：维护人、成交楼盘+成交金额
        elif re.search(r"维护人|成交楼盘.*成交金额", text):
            return "中盘"
        # 二手：成交楼盘+成交价
        elif re.search(r"成交楼盘.*成交价", text):
            return "二手"
        # 默认情况
        else:
            return "二手"
    
    return "二手"


# 通用提取和识别辅助函数
def _extract_and_identify(text, patterns, known_items, item_type, threshold, is_dict, cleaning_func=None, standardization_func=None, special_handling_func=None):
    """
    通用提取和识别辅助函数，用于从文本中提取并识别信息
    
    参数:
        text: str - OCR提取的文本
        patterns: list - 正则表达式模式列表
        known_items: dict or list - 已知项目列表或字典
        item_type: str - 项目类型，用于生成备注
        threshold: float - 相似度阈值
        is_dict: bool - 标识known_items是否为字典
        cleaning_func: callable - 清理函数，用于清理提取的结果
        standardization_func: callable - 标准化函数，用于标准化结果
        special_handling_func: callable - 特殊处理函数，用于处理特殊情况
        
    返回:
        tuple - (str, str) 识别到的项目和备注信息
    """
    final_result = ""
    note = ""
    extracted_item = ""
    
    # 1. 格式匹配：使用正则表达式提取项目
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            extracted_item = match.group(1).strip()
            
            # 执行清理函数（如果提供）
            if cleaning_func:
                extracted_item = cleaning_func(extracted_item, match)
            
            # 先应用标准化函数，过滤无效的提取项
            if standardization_func:
                standardized_item = standardization_func(extracted_item)
            else:
                standardized_item = extracted_item
            
            # 检查提取的内容是否有效
            if standardized_item and not standardized_item.isspace():
                extracted_item = standardized_item
                break
            else:
                extracted_item = ""
    
    # 2. 相似度匹配：如果提取到有效项目，进行相似度匹配
    if extracted_item:
        # 使用通用相似度匹配函数
        final_result, similarity_note = match_with_similarity(
            extracted_item, 
            known_items, 
            threshold=threshold, 
            item_type=item_type, 
            is_dict=is_dict
        )
        
        if similarity_note:
            note = similarity_note
    else:
        final_result = ""
    
    # 4. 特殊处理（如果提供）
    if special_handling_func:
        final_result, special_note = special_handling_func(text, final_result)
        if special_note:
            note = f"{note} {special_note}".strip()
    
    return final_result, note


# 小区提取函数
def extract_and_identify_community(text):
    """
    从OCR文本中提取并识别小区名（精简优化版）
    只保留格式匹配和相似度匹配核心功能，修复识别错误
    
    参数:
        text: str - OCR提取的文本
        
    返回:
        tuple - (str, str) 识别到的小区名和备注信息
    """
    # 小区名清理函数 - 简化版，只保留必要的清理
    def community_cleaning_func(extracted, match):
        # 获取匹配到的原始内容
        raw_content = match.group(0)
        
        # 特殊处理：如果是"楼盘："后面直接跟换行或结束，说明是空楼盘
        if re.search(r"楼盘[:：]\s*$|楼盘[:：]\s*\n", raw_content):
            return ""
        
        # 只做基本清理，移除空白字符和下划线
        extracted = extracted.replace(" ", "").replace("_", "")
        
        # 移除可能的冒号后缀
        if extracted.endswith(":") or extracted.endswith("："):
            extracted = extracted[:-1]
        
        # 基本有效性检查
        if extracted and not extracted.isspace():
            return extracted
        else:
            return ""
    
    # 小区名标准化函数 - 包含后置处理逻辑
    def community_standardization_func(community):
        # 移除不可见字符
        community = re.sub(r"[\x00-\x1F\x7F]", "", community)
        
        # 过滤无效关键词
        invalid_keywords = ["成交时间", "签约时间", "认购时间", "合同金额", "成交金额", "成交价"]
        if any(kw in community for kw in invalid_keywords):
            return ""
        
        # 去除冒号（全角和半角）而不是过滤
        community = community.replace("：", "").replace(":", "")
        
        # 去除常见前缀
        for prefix in ["成交楼盘", "楼盘"]:
            if community.startswith(prefix):
                community = community[len(prefix):]
                break
        
        # 后置处理：
        # 1. 去除第一个汉字之前的所有非汉字
        first_chinese = re.search(r"[\u4e00-\u9fa5]", community)
        if first_chinese:
            # 保留第一个汉字及之后的内容
            community = community[first_chinese.start() :]
        
        # 2. 从非汉字处截断，排除·、英文字母和数字
        # 匹配第一个非汉字且非排除字符的位置
        truncate_pos = re.search(r"[^a-zA-Z0-9\u4e00-\u9fa5·]", community)
        if truncate_pos:
            # 截断到该位置
            community = community[: truncate_pos.start()]
        
        return community
    
    # 小区名特殊处理函数
    def community_special_handling_func(text, community):
        # 中盘喜报小区识别优化（如果还没有识别到小区）
        if not community and "成交楼盘" in text:
            # 专门为中盘喜报优化小区识别
            community_match = re.search(r"成交楼盘[:：]?\s*([^\n]+?)(?:\s*\d+(?:\.\d+)?[m㎡]?|\n|成交金额)", text)
            if community_match:
                community = community_match.group(1).strip()
                # 移除不可见字符
                community = re.sub(r"[\x00-\x1F\x7F]", "", community)
                # 应用标准化处理
                community = community_standardization_func(community)
                # 同时也做基本清理
                community = community.replace(" ", "").replace("_", "")
                # 移除可能的冒号后缀
                if community.endswith(":") or community.endswith("："):
                    community = community[:-1]
        
        return community, ""
    
    # 定义正则表达式模式，优先匹配成交项目成交价格式
    patterns = [
        # 1. 优先匹配：成交项目成交价xxx（支持无数字金额，支持有无空格）
        r"成交项目成交价\s*([^\n]+?)(?:\s*\n|$)",
        # 2. 优先匹配：成交项目成交价xxx（无空格情况，直接匹配小区名）
        r"成交项目成交价([\u4e00-\u9fa5][^\n]+?)(?:\s*\n|$)",
        # 3. 修复：成交楼盘成交价xxx（无空格情况，直接匹配小区名）
        r"成交楼盘成交价([\u4e00-\u9fa5][^\n]+?)(?:\s*\n|$)",
        # 4. 主要模式：楼盘：xxx
        r"楼盘[:：]\s*([^\n]+?)(?:\s*(?:\n|合同金额|成交金额|成交价))",
        # 5. 成交楼盘 xxx 数字 格式（无成交价）
        r"成交楼盘\s+([^\d\s]+?)\s+\d+",
        # 6. 成交楼盘成交价 xxx 数字 格式（有成交价）
        r"成交楼盘成交价\s+([^\d\s]+?)\s+\d+",
        # 7. 成交楼盘 xxx （换行或结束）
        r"成交楼盘\s*([^\n\d]+?)\s*(?:\n|$)",
        # 8. 成交项目/小区：xxx
        r"成交(?:项目|小区)[:：]?\s*([^\n]+?)(?:\s*(?:\n|成交金额|成交价))",
        # 9. 小区名+面积格式（支持斜杠分隔，如"新世纪领居B区/157.0m"）
        r"([\u4e00-\u9fa5]+[花园苑府上院居城期山钻][\u4e00-\u9fa5\d·]*[三二一]?期?[ABCD]?区?)[/\s]+\d+",
        # 10. 小区名+第X单格式（支持中间有面积，如"松湖春天 197.16m第34单"）
        r"(?:^|\n)([\u4e00-\u9fa5]+)(?:\s+[\d.]+[m㎡mlML]+)?第\d+单",
        # 12. 小区名+第X单格式（支持前面有内容，如"认购时间：20260503 珑远首铸森湖翠珑湾第5单"）
        r"([\u4e00-\u9fa5]+)(?:\s+[\d.]+[m㎡mlML]+)?第\d+单",
        # 11. 租赁格式：只匹配楼盘：后的内容，不要求金额
        r"楼盘[:：]\s*([^\n]+?)(?:\s*\n|$)",
    ]
    
    # 使用通用提取和识别函数
    return _extract_and_identify(
        text, 
        patterns, 
        known_communities, 
        "小区", 
        70.0, 
        True, 
        community_cleaning_func, 
        community_standardization_func, 
        community_special_handling_func
    )


# 金额提取函数
def extract_amount(text):
    """
    从文本中提取金额
    
    参数:
        text: str - OCR提取的文本
    
    返回:
        str - 提取的金额字符串
    """
    amount = ""
    
    amount_patterns = [
        r"(\d+(?:\.\d+)?)(?:\s*万|万元)",  # 99万、216万
        r"合同金额[:：]?\s*(\d+(?:\.\d+)?)",  # 合同金额：2800
        r"成交价[:：]?\s*(\d+(?:\.\d+)?)",  # 成交价：99万
        r"成交金额[:：]?\s*(\d+(?:\.\d+)?)",  # 成交金额：216万
        r"成交楼盘成交价\s*[^\d]*?(\d+(?:\.\d+)?)",  # 成交楼盘成交价后面直接跟数字
        r"成交项目成交价\s*[^\d]*?(\d+(?:\.\d+)?)",  # 成交项目成交价后面直接跟数字
        r"(\d+(?:\.\d+)?)万",  # 匹配单独的数字+万字，如192万（支持后面跟着其他文字）
    ]
    
    for pattern in amount_patterns:
        amount_match = re.search(pattern, text)
        if amount_match:
            amount = amount_match.group(1)
            # 如果是万元单位，转换为元（但只有当数字小于1000时才乘，避免重复转换）
            if "万" in text[amount_match.start():amount_match.end()]:
                try:
                    num = float(amount)
                    if num < 1000:  # 只有当数字小于1000时才乘以10000，避免重复转换
                        amount = str(int(num * 10000))
                except ValueError:
                    pass
            break
    
    # 确保金额为整数格式，空值设为0
    if amount:
        # 确保金额为纯数字
        amount = re.sub(r"[^0-9]", "", amount)
        if not amount:
            amount = "0"
    else:
        amount = "0"
    
    return amount


# 面积提取函数
def extract_area(text, is_medium=False):
    """
    从文本中提取面积
    
    参数:
        text: str - OCR提取的文本
        is_medium: bool - 是否是中盘喜报，中盘喜报不修复竖条误识别
    
    返回:
        str - 提取的面积字符串，空值为空字符串
    """
    area = ""
    
    area_patterns = [
        r"(\d+(?:\.\d+)?)\s*[m㎡mlML]",  # 197.16m、184.24ml
        r"(\d+(?:\.\d+)?)\s*(?:平方米|平米)",  # 100平方米、90平米
        r"面积[:：]?\s*(\d+(?:\.\d+)?)",  # 面积：120
        r"[.\s](\d+(?:\.\d+)?)\s*[巾m㎡]",  # .90.55巾（小区名后面直接跟面积）
    ]
    
    for pattern in area_patterns:
        area_match = re.search(pattern, text)
        if area_match:
            area = area_match.group(1)
            break
    
    # 额外检查：如果面积太大（>200），我们再检查一下这个数字是否可能是金额
    # 如果前面有金额相关的关键词，就不应该当成面积
    if area:
        try:
            area_value = float(area)
            if area_value > 200:
                # 检查这个数字前面是否有金额相关关键词
                area_pos = text.find(area)
                if area_pos > 0:
                    check_text = text[max(0, area_pos - 20):area_pos]
                    # 排除"成交楼盘"这种情况，因为它后面通常跟的是小区名和面积
                    if "成交楼盘" in check_text or "成交项目" in check_text:
                        # 成交楼盘/成交项目后面通常跟小区名和面积，不应该清空
                        pass
                    elif any(keyword in check_text for keyword in ["成交金额", "合同金额", "成交价", "金额"]):
                        # 很可能是金额，不是面积，清空
                        area = ""
        except ValueError:
            pass
    
    # 中盘喜报不修复竖条误识别，但需要格式化
    if not is_medium:
        # 修复竖条被误识别成1的情况（只对非中盘喜报）
        # 当面积以1开头，且前面有换行符或空格时，去除第一个1
        # 去除后保证数值>32，否则不去除
        if area and len(area) > 1 and area.startswith("1"):
            # 检查原文中这个面积前面是否有换行符或空格
            # 查找这个面积在原文中的位置
            area_pos = text.find(area)
            if area_pos > 0:
                # 检查前面一个字符
                prev_char = text[area_pos - 1]
                if prev_char in ['\n', ' ']:
                    # 尝试去除第一个1
                    temp_area = area[1:]
                    try:
                        # 检查去除后数值是否>32
                        area_value = float(temp_area)
                        if area_value > 32:
                            area = temp_area
                    except ValueError:
                        # 如果转换失败，保持原状
                        pass
    
    # 格式化为两位小数，不够补0（所有情况都格式化）
    if area:
        try:
            area_value = float(area)
            area = f"{area_value:.2f}"
        except ValueError:
            # 转换失败，保持原状
            pass
    
    return area


# 门店提取函数
def extract_and_identify_store(text):
    """
    从OCR文本中提取并识别门店名（简化版）
    
    参数:
        text: str - OCR提取的文本
        
    返回:
        tuple - (str, str) 识别到的门店名和备注信息
    """
    # 1. 易错项匹配
    if "易晨" in text:
        return "易辰", "[门店映射：易晨→易辰]"
    if "中直" in text:
        return "中宜", "[门店映射：中直→中宜]"
    
    # 2. 特事特办：杰成世家
    if "杰成世家" in text:
        return "杰成", ""
    
    # 3. 中盘喜报特殊处理：优先从维护人相关文本中提取门店
    if "维护中盘" in text or "维护人" in text:
        # 分割文本为行
        lines = text.split('\n')
        for i, line in enumerate(lines):
            # 找到维护人相关的行
            if "维护人" in line:
                # 检查维护人行及其后面3行
                for j in range(i, min(i + 4, len(lines))):
                    check_line = lines[j]
                    # 在这些行中搜索已知店名
                    for store in sorted(config.known_store_names, key=len, reverse=True):
                        if store in check_line:
                            return store, ""
                    # 搜索地产关键字
                    for keyword in ["地产", "加盟店"]:
                        pos = check_line.find(keyword)
                        if pos != -1:
                            prefix_text = check_line[max(0, pos-4):pos]
                            for store in sorted(config.known_store_names, key=len, reverse=True):
                                if store in prefix_text:
                                    return store, ""
    
    # 4. 简化逻辑：搜索地产和加盟店关键字
    keywords = ["地产", "加盟店"]
    for keyword in keywords:
        pos = text.find(keyword)
        if pos != -1:
            # 取关键字前4个字
            prefix_text = text[max(0, pos-4):pos]
            # 从这四个字里匹配已知店名
            for store in sorted(config.known_store_names, key=len, reverse=True):
                if store in prefix_text:
                    return store, ""
            # 如果匹配不上，取后两个字
            if len(prefix_text) >= 2:
                return prefix_text[-2:], ""
    
    # 5. 直接匹配已知店名
    for store in sorted(config.known_store_names, key=len, reverse=True):
        if store in text:
            return store, ""
    
    # 6. 未识别到任何门店
    return "", ""


# 人员提取函数 - 返回 (维护人, 成交人) 或 (人员, "")
def extract_and_identify_person(text, is_medium=False):
    """
    从OCR文本中提取并识别人名
    简化逻辑：格式匹配 → 精确匹配 → 相似度匹配 → 新内容
    对于中盘喜报，同时提取维护人和成交人
    """
    # 非人名黑名单
    non_person_keywords = {"东莞站", "贝壹", "贝联", "林泽武", "租赁", "相信自己", "再创佳绩", "今日共开", "贝壳开单", "喜报来", "开单套报", "开单喜报", "喜报合作", "合作共赢", "共创辉煌", "地产", "成交楼盘", "成交项目", "莞南大区", "莞北大区", "刘志威", "杨文斌"}
    
    # 通用的人员模式列表
    GENERAL_PERSON_PATTERNS = [
        r"(?:^|\n)\s*([\u4e00-\u9fa5]{2,4})\s*(?:\n|$)",
        r"(?:^|\n)\s*([\u4e00-\u9fa5]{2,4})[！!。.，,、]?(?:贝壹|贝联)",
        r"二手开单([\u4e00-\u9fa5]{2,4})",
        r"租赁([\u4e00-\u9fa5]{2,4})",
        r"热烈\s*祝\s*贺[：:]?\s*([\u4e00-\u9fa5]{2,4})",
        r"[Bb]eike\s*([\u4e00-\u9fa5]{2,4})",
        r"贝壳\s*([\u4e00-\u9fa5]{2,4})",
        r"租赁\s*([\u4e00-\u9fa5]{2,4})\s*莞南大区",
        r"([\u4e00-\u9fa5]{2,4})莞南大区",
        r"[（(]([\u4e00-\u9fa5]{2,4})",
        r"(?:^|\n)\s*([\u4e00-\u9fa5]{2,4})\s*(德佑|乐远|C21|21世纪|住商)",
        r"二手开单喜报([\u4e00-\u9fa5]{2,4})(?:乐远|德佑|C21|21世纪|住商)",
        r"莞南大区-([\u4e00-\u9fa5]{2,4})成交项目",
        r"([\u4e00-\u9fa5]{2,4})[一]+(?:[\u4e00-\u9fa5]{2,4})",  # 匹配"XXX一YYY"格式中的XXX
        r"(?<!莞南大区)[一]+([\u4e00-\u9fa5]{2,4})",  # 排除"莞南大区一"后面的区域经理名
        r"[（(][）)]\s*([\u4e00-\u9fa5]{2,4})",
    ]
    
    # 辅助函数：检查字符串是否是门店名或包含门店名
    def is_store_related(s):
        for store in config.known_store_names:
            if store == s or store in s:
                return True
        return False
    
    # 辅助函数：根据模式列表提取匹配结果
    def extract_matches(patterns):
        matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                matched = match.group(1).strip()
                if len(matched) >= 2 and matched not in non_person_keywords and not any(kw in matched for kw in non_person_keywords) and not is_store_related(matched):
                    matches.append(matched)
        return matches
    
    # 辅助函数：根据单个匹配结果
    def get_best_match(matches):
        if not matches:
            return "", ""
        # 精确匹配
        for matched in matches:
            if matched in known_persons:
                return matched, ""
        # 相似度匹配
        best_match = ""
        highest_similarity = 60.0
        similarity_note = ""
        for matched in matches:
            for known in known_persons:
                similarity = calculate_similarity(matched, known)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = known
                    similarity_note = f"[人员相似匹配：{matched}→{best_match}，匹配度：{similarity:.1f}%]"
        if best_match:
            return best_match, similarity_note
        # 新内容
        return matches[0], ""
    
    # 对于中盘喜报，分别提取维护人和成交人
    if is_medium:
        # 提取维护人
        maintainer_matches = extract_matches([r"维护人[：:]?\s*([\u4e00-\u9fa5]{2,4})"])
        # 提取成交人
        dealer_matches = extract_matches([r"成交人[：:]?\s*([\u4e00-\u9fa5]{2,4})"])
        
        maintainer, maintainer_note = get_best_match(maintainer_matches)
        dealer, dealer_note = get_best_match(dealer_matches)
        
        # 如果没有找到维护人，回退到通用匹配
        if not maintainer:
            general_matches = extract_matches(GENERAL_PERSON_PATTERNS)
            maintainer, maintainer_note = get_best_match(general_matches)
        
        return maintainer, dealer, maintainer_note, dealer_note
    else:
        # 非中盘喜报，按原来的逻辑
        all_patterns = [
            r"维护人[：:]?\s*([\u4e00-\u9fa5]{2,4})",
            r"成交人[：:]?\s*([\u4e00-\u9fa5]{2,4})",
        ] + GENERAL_PERSON_PATTERNS
        all_matches = extract_matches(all_patterns)
        person, note = get_best_match(all_matches)
        return person, note


# ==========================================
# 7. 主解析模块
# ==========================================

def _clean_property_fields(property_info):
    """
    清理和标准化所有字符串字段，确保WPS兼容性
    
    参数:
        property_info: dict - 房产信息字典
        
    返回:
        dict - 清理后的房产信息字典
    """
    # 统一处理所有字符串字段，确保WPS兼容性
    for key in property_info:
        if isinstance(property_info[key], str):
            # 移除不可见字符、控制字符和换行符，确保WPS兼容
            property_info[key] = re.sub(r"[\x00-\x1F\x7F\n\r\t]", " ", property_info[key])
            # 移除多余的空格
            property_info[key] = re.sub(r"\s+", " ", property_info[key]).strip()
    
    # 处理特殊情况：如果小区名包含无效关键字，清空小区名
    if property_info["小区"] and any(keyword in property_info["小区"] for keyword in ["成交时间", "签约时间", "认购时间", "合同金额", "成交价", "成交金额"]):
        property_info["小区"] = ""
    
    return property_info


def _optimize_community_for_medium_disk(property_info, text):
    """
    优化中盘喜报的小区识别
    
    参数:
        property_info: dict - 房产信息字典
        text: str - OCR提取的文本
        
    返回:
        dict - 优化后的房产信息字典
    """
    # 中盘喜报小区识别优化
    if "中盘" in property_info["类型"] and not property_info["小区"] and "成交楼盘" in text:
        # 专门为中盘喜报优化小区识别
        community_match = re.search(r"成交楼盘[:：]?\s*([^\n]+?)(?:\s*\d+(?:\.\d+)?[m㎡]?|\n|成交金额)", text)
        if community_match:
            property_info["小区"] = community_match.group(1).strip()
    
    return property_info


def parse_property_info(text, image_path=None):
    """
    从OCR文本中解析房产喜报信息
    支持四种喜报格式：二手买卖、租赁、中盘、新房成交
    
    参数:
        text: str - OCR提取的文本
        image_path: str - 图片文件路径（用于获取分辨率信息和重试）
    
    返回:
        dict - 解析后的房产信息
    """
    # 初始化结果
    temp_date = datetime.now().strftime("%Y-%m-%d")
    property_info = {
        "日期": temp_date,
        "类型": "",
        "小区": "",
        "金额": "",
        "门店": "",
        "人员": "",
        "面积": "",
        "_exact_time": f"{temp_date} 00:00:00",  # 初始化_exact_time
        "_file_time": f"{temp_date} 00:00:00",  # 初始化_file_time，用于排序
        "_区域": "",  # 添加小区区域标签：大岭山/非大岭山
        "_CA": "",  # 添加CA辅助字段：林泽武/贾梦云
        "_成交人": "",  # 中盘喜报的成交人
        "_原始门店": ""  # 中盘喜报中的其他门店
    }
    
    # 1. 提取日期和精确时间
    property_info["日期"], exact_time = extract_date(text, image_path)
    property_info["_exact_time"] = exact_time
    property_info["_file_time"] = exact_time
    
    # 2. 提取类型
    property_info["类型"] = extract_type(text, image_path)
    
    # 3. 提取金额
    property_info["金额"] = extract_amount(text)
    
    # 3.5. 提取面积（中盘喜报不修复竖条误识别）
    is_medium = (property_info["类型"] == "中盘")
    property_info["面积"] = extract_area(text, is_medium)
    
    # 4-6. 提取小区、门店、人员信息
    # 提取小区
    final_community, community_note = extract_and_identify_community(text)
    property_info["小区"] = final_community
    
    # 添加区域标签
    if final_community:
        property_info["_区域"] = get_community_area(final_community)
    
    # 提取门店
    property_info["门店"], store_note = extract_and_identify_store(text)
    
    # 提取人员（对于中盘喜报，同时提取维护人和成交人）
    if is_medium:
        maintainer, dealer, maintainer_note, dealer_note = extract_and_identify_person(text, is_medium=True)
        property_info["人员"] = maintainer
        property_info["_成交人"] = dealer
        # 合并备注信息
        notes = list(filter(None, [community_note, store_note, maintainer_note, dealer_note]))
        # 同时尝试提取其他门店作为成交人门店
        # 先保存原始门店（维护人门店）
        property_info["_原始门店"] = property_info["门店"]
    else:
        property_info["人员"], person_note = extract_and_identify_person(text)
        notes = list(filter(None, [community_note, store_note, person_note]))
    
    property_info["备注"] = " ".join(notes)
    
    # 7-10. 统一字段清理、标准化和验证
    property_info = _clean_property_fields(property_info)
    
    # 中盘喜报小区识别优化
    property_info = _optimize_community_for_medium_disk(property_info, text)
    
    # CA辅助字段识别：从文本中识别林泽武或贾梦云
    if "林泽武" in text:
        property_info["_CA"] = "林泽武"
    elif "贾梦云" in text:
        property_info["_CA"] = "贾梦云"
    
    return property_info


# ==========================================
# 8. 图片处理主函数
# ==========================================

def process_property_image(image_path):
    """
    处理房产喜报图片，提取并解析信息
    
    参数:
        image_path: str - 图片文件路径
    
    返回:
        tuple - (dict, str, bool) 解析后的房产信息、OCR提取的原文和是否使用了Umi-OCR的标志
    """
    # 0. 前置筛选：检查图片分辨率
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            # 允许的分辨率列表：租赁为450*800，二手、新房和中盘为369*800，新增1078*2279
            allowed_resolutions = [(450, 800), (369, 800),(378, 800)]
            if (width, height) not in allowed_resolutions:
                return None
    except Exception as e:
        print(f"获取图片分辨率失败：{str(e)}")
        # 分辨率获取失败时，直接返回None，不再进行OCR识别
        return None
    
    # 统一OCR识别和重试机制
    # 定义重试策略并初始化最佳结果
    retry_strategies = [(False, "原始识别"), (False, "相同引擎重试"), (True, "备用引擎重试")]
    best_result, best_text, best_is_http, min_empty_count = None, "", False, float('inf')
    
    # 使用重试策略的长度作为最大尝试次数
    for i, (use_backup_engine, attempt_name) in enumerate(retry_strategies):
        print(f"[{attempt_name}] 开始OCR识别")
        
        # 执行OCR识别
        text, is_http = extract_text_from_image(image_path, original_image_path=image_path, use_backup_engine=use_backup_engine)
        
        if not text:
            print(f"[{attempt_name}] OCR识别失败，未返回文本")
            continue
        
        # 前置过滤：检查是否包含需要跳过的关键词
        skip_keywords = ["喜报来", "LIKE"]
        if any(keyword in text for keyword in skip_keywords):
            print(f"[{attempt_name}] 检测到跳过关键词，跳过该图片")
            return "SKIP_FILE"  # 返回特殊标记表示需要跳过
        
        # 仅在第一次识别时打印OCR引擎类型
        if i == 0:
            print(f"[{attempt_name}] [OCR引擎] {'Umi-OCR 命令行' if is_http else 'PaddleOCR-json'}")
        
        # 解析信息并检查结果
        if parse_result := parse_property_info(text, image_path):
            # 直接使用parse_result，不需要类型检查
            property_info = parse_result
            
            # 计算空字段数量（简化逻辑，直接使用布尔值相加）
            empty_count = sum([
                not property_info["小区"] or property_info["小区"] == "未识别小区",
                not property_info["门店"],
                not property_info["人员"] or len(property_info["人员"]) < 2,
                property_info["金额"] == "0" or not property_info["金额"]
            ])
            
            # 更新最佳结果
            if empty_count < min_empty_count:
                # 记录是否是更好的结果
                is_improvement = best_result is not None
                # 合并更新最佳结果
                min_empty_count, best_result, best_text, best_is_http = empty_count, property_info, text, is_http
                
                # 仅在有实际改进且有空字段时打印日志
                if is_improvement and min_empty_count > 0:
                    print(f"[{attempt_name}] 空字段数量: {empty_count} (更新为最佳结果)")
            
            # 空字段数量少于2个，直接成功
            if empty_count < 2:
                if i > 0:  # 仅在重试成功时打印
                    print(f"[{attempt_name}] 识别成功，空字段数量少于2个")
                break
            
            # 不是最后一次尝试，继续重试
            if i < len(retry_strategies) - 1:
                print(f"[{attempt_name}] 检测到 {empty_count} 个空字段，准备进行下一次重试")
        else:
            print(f"[{attempt_name}] 解析结果为空")
    
    # 检查是否有可用结果
    if not best_result:
        print("所有OCR识别尝试都失败了")
        return None
    
    # 合并最佳结果赋值语句
    property_info, text, is_http, empty_count = best_result, best_text, best_is_http, min_empty_count
    
    # 仅在空字段数量大于0时打印最终结果
    if empty_count > 0:
        print(f"最终OCR识别结果：空字段数量 = {empty_count}")

    # 屏蔽缺少多个字段的记录：如果3个或更多关键字段为空，跳过
    if empty_count >= 3:
        print(f"最终检测到 {empty_count} 个空字段，跳过该记录")
        return None
    
    # 中盘喜报过滤规则：维护人门店=优瑞通或益安且成交人门店≠优瑞通或益安，保留
    if property_info["类型"] == "中盘":
        allowed_stores = ["优瑞通", "益安"]
        skip_write_reason = None
        dealer_store = ""  # 保存成交人门店
        
        # 维护人门店不在允许列表
        if property_info["门店"] not in allowed_stores:
            skip_write_reason = f"中盘记录跳过写入：维护人门店 {property_info['门店']} 不在允许列表中"
            print(skip_write_reason)
        else:
            # 提取文本中所有已知门店
            unique_stores = list(set(store for store in config.known_store_names if store in text))
            # 排除维护人门店
            if property_info["门店"] in unique_stores:
                unique_stores.remove(property_info["门店"])
            
            # 保存成交人门店（取第一个非允许列表的门店）
            for store in unique_stores:
                if store not in allowed_stores:
                    dealer_store = store
                    break
            
            # 没有找到其他已知门店，检查是否有其他门店标识（如"店"字）
            if not unique_stores:
                # 检查文本中是否有"店"字，排除维护人门店相关文本
                has_other_store = False
                lines = text.split('\n')
                for line in lines:
                    if '店' in line and property_info["门店"] not in line:
                        # 检查是否是门店相关行（包含"世纪"、"C21"等门店标识）
                        if any(kw in line for kw in ["世纪", "C21", "21世纪", "德佑", "链家", "贝壳", "中原", "乐有家", "Q房"]):
                            has_other_store = True
                            print(f"中盘记录检测到其他门店标识：{line.strip()}")
                            break
                
                if not has_other_store:
                    skip_write_reason = f"中盘记录跳过写入：维护人门店 {property_info['门店']} 与成交人门店相同"
                    print(skip_write_reason)
                else:
                    print(f"中盘记录保留：维护人门店 {property_info['门店']}，检测到其他门店（非已知列表）")
            else:
                # 有成交人门店在允许列表中，跳过
                if any(store in allowed_stores for store in unique_stores):
                    skip_write_reason = f"中盘记录跳过写入：维护人门店 {property_info['门店']} 和成交人门店 {unique_stores} 都在允许列表中"
                    print(skip_write_reason)
                else:
                    print(f"中盘记录保留：维护人门店 {property_info['门店']}，成交人门店 {unique_stores}")
        
        # 保存成交人门店到property_info
        property_info["_成交人门店"] = dealer_store
        
        # 提取成交人所属的大区信息（用于判断是否生成二手记录）
        # 从文本中匹配"莞南大区"、"莞北大区"等
        dealer_district = ""
        district_match = re.search(r"莞([南北])大区", text)
        if district_match:
            dealer_district = f"莞{district_match.group(1)}大区"
        property_info["_成交人大区"] = dealer_district
        
        # 如果需要跳过写入，添加标记但保留记录用于互补
        if skip_write_reason:
            property_info['_skip_write'] = True
            # 添加备注
            if '备注' in property_info and property_info['备注']:
                property_info['备注'] += f" [{skip_write_reason}]"
            else:
                property_info['备注'] = skip_write_reason
    
    # 保存初始字典状态，用于检查是否都不在已知列表中
    initial_person_in_dict = property_info['人员'] in known_persons
    
    # 检查小区是否在字典中（兼容新旧格式）
    initial_community_in_dict = False
    if property_info['小区']:
        if property_info['小区'] in known_communities:
            community_val = known_communities[property_info['小区']]
            if isinstance(community_val, dict) or isinstance(community_val, str):
                initial_community_in_dict = True
    
    initial_store_in_dict = property_info['门店'] in config.known_store_names
    
    # 1. 检查小区、门店、人员中有几个不在已知列表中
    unknown_count = 0
    if not initial_person_in_dict:
        unknown_count += 1
    if not initial_community_in_dict:
        unknown_count += 1
    if not initial_store_in_dict:
        unknown_count += 1
    
    # 先设置默认为不跳过（但保留已经设置为跳过的标记）
    if '_skip_write' not in property_info:
        property_info['_skip_write'] = False
    
    # 只有还没有被跳过的记录，才继续检查其他条件
    if not property_info['_skip_write']:
        if unknown_count >= 2:
            # 添加标记，跳过写入WPS
            property_info['_skip_write'] = True
            # 添加备注
            skip_note = f" [{unknown_count}项不在已知列表，跳过写入WPS]"
            if '备注' in property_info and property_info['备注']:
                property_info['备注'] += skip_note
            else:
                property_info['备注'] = skip_note[1:]
        else:
            # CA辅助字段去留逻辑 + 非大岭山楼盘过滤
            ca = property_info.get('_CA', '')
            area = property_info.get('_区域', '')
            
            # 林泽武CA区域保留所有楼盘，其他CA区域只保留大岭山的
            if ca != '林泽武' and area == '非大岭山':
                property_info['_skip_write'] = True
                skip_note = f" [非林泽武区域非大岭山楼盘，跳过写入]"
                if '备注' in property_info and property_info['备注']:
                    property_info['备注'] += skip_note
                else:
                    property_info['备注'] = skip_note[1:]
    
    return (property_info, text, is_http)


# WPS操作辅助函数

def _send_wps_request(action, payload_data, headers=None):
    """
    发送WPS API请求的通用函数
    
    参数:
        action: str - API操作类型
        payload_data: dict - 请求数据
        headers: dict - 可选的请求头
    
    返回:
        dict - API响应的实际结果
    """
    if headers is None:
        headers = {
            "Content-Type": "application/json",
            "AirScript-Token": config.wps_config["script_token"]
        }
    
    # 构建通用请求数据
    payload = {
        "Context": {
            "argv": {
                "action": action,
                "sheet": config.wps_config["sheet_name"]
            }
        }
    }
    
    # 添加操作特定数据
    payload["Context"]["argv"].update(payload_data)
    
    # 发送请求
    response = requests.post(
        config.wps_config["webhook_url"],
        json=payload,
        headers=headers,
        timeout=60
    )
    response.raise_for_status()
    
    # 解析响应
    result = response.json()
    airscript_response = result.get("data", {})
    actual_result = airscript_response.get("result", {})
    
    return actual_result


def _save_write_count(count):
    """
    保存写入数量到临时文件的通用函数
    
    参数:
        count: int - 写入数量
    """
    try:
        temp_file = os.path.join(config.output_dir, "last_write_count.txt")
        with open(temp_file, "w") as f:
            f.write(str(count))
    except Exception as e:
        print(f"保存写入数量失败：{str(e)}")


def _is_wps_enabled():
    """
    检查WPS功能是否启用的通用函数
    
    返回:
        bool - WPS功能是否启用
    """
    enabled = config.wps_config.get("enabled", True)
    if not enabled:
        print("WPS功能已禁用")
    return enabled

# WPS操作函数
def get_existing_wps_records(sort_records=True, days=14):
    """
    从WPS多维表格获取现有记录（默认只查询最近指定天数的记录）
    
    参数:
        sort_records: bool - 是否对记录进行排序，True表示按日期和ID降序排序
        days: int - 查询最近多少天的记录，默认14天
    
    返回:
        list - 现有记录列表
    """
    global _full_records_with_id
    try:
        # 构建基础请求数据
        today = datetime.now()
        start_date = today - timedelta(days=days)
        base_payload = {
            "sort": {"field": "日期", "order": "desc"},
            "PageSize": 1000,
            "filter": {
                "mode": "AND",
                "criteria": [{
                    "field": "日期",
                    "op": "GreaterEquAndLessEqu",
                    "values": [start_date.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")]
                }]
            }
        }
        
        print("正在获取WPS多维表格中的现有记录...")
        
        # 初始化分页查询变量
        all_flattened_records = []
        offset = None
        page_count = 0
        max_pages = 10
        
        # 内部辅助函数：处理记录解析和去重
        def process_records(data_list, existing_ids):
            """
            处理记录列表，包括展平结构和去重
            
            参数:
                data_list: list - 待处理的记录列表
                existing_ids: set - 已存在的记录ID集合
            
            返回:
                tuple - (new_records, new_offset) 处理后的新记录和下一页偏移量
            """
            flattened_records = []
            new_offset = None
            
            # 展平记录结构
            for item in data_list:
                if isinstance(item, dict):
                    new_offset = item.get("offset")
                    if "records" in item and isinstance(item["records"], list):
                        flattened_records.extend(item["records"])
                    else:
                        flattened_records.append(item)
            
            # 去重处理
            new_records = []
            for record in flattened_records:
                record_id = record.get("id", "")
                if record_id not in existing_ids:
                    new_records.append(record)
                    existing_ids.add(record_id)
            
            return new_records, new_offset
        
        # 使用集合存储已处理的记录ID，用于去重
        processed_record_ids = set()
        
        # 分页查询循环
        while True:
            page_count += 1
            if page_count > max_pages:
                break
            
            # 构建当前页请求
            current_payload = base_payload.copy()
            if offset:
                current_payload["Offset"] = offset
            
            # 发送请求
            actual_result = _send_wps_request("read", current_payload)
            
            # 检查响应
            if not actual_result.get("success", False):
                print(f"获取记录失败：{actual_result.get('message', '未知错误')}")
                break
            
            data_list = actual_result.get("data", [])
            if not isinstance(data_list, list):
                break
            
            # 使用内部辅助函数处理记录
            new_records, new_offset = process_records(data_list, processed_record_ids)
            
            all_flattened_records.extend(new_records)
            
            # 检查终止条件
            if not new_records or offset == new_offset:
                break
            
            offset = new_offset
        
        # 从每条记录中提取fields数据，同时保留记录ID
        simplified_records = []
        full_records_with_id = []
        for record in all_flattened_records:
            if isinstance(record, dict):
                fields = record.get("fields", record)
                if isinstance(fields, dict):
                    simplified_records.append(fields)
                    full_records_with_id.append({
                        "id": record.get("id", ""),
                        "fields": fields
                    })
        
        # 只有当sort_records为True时才进行排序
        if sort_records:
            full_records_with_id.sort(
                key=lambda record: (record.get("fields", {}).get("日期", ""), record.get("id", "").upper()),
                reverse=True
            )
        
        # 打印提取到的记录数量
        total_records = len(simplified_records)
        print(f"成功获取 {total_records} 条现有记录")
        
        # 返回简化记录用于查重，同时保存带有ID的完整记录到全局变量
        global _full_records_with_id
        _full_records_with_id = full_records_with_id
        
        return simplified_records
            
    except Exception as e:
        print(f"获取WPS多维表格现有记录异常：{str(e)}")
        traceback.print_exc()
        
        # 尝试从本地备份文件获取记录作为回退
        print("尝试从本地备份文件获取记录...")
        backup_file = os.path.join(config.output_dir, "wps_records_backup.json")
        try:
            if os.path.exists(backup_file):
                with open(backup_file, "r", encoding="utf-8") as f:
                    backup_records = json.load(f)
                print(f"成功从本地备份文件加载 {len(backup_records)} 条记录")
                
                # 更新全局变量，用于后续查重
                _full_records_with_id = backup_records
                
                return backup_records
        except Exception as backup_e:
            print(f"从本地备份文件加载记录失败：{str(backup_e)}")
        
        return []

def write_data_to_wps_sheet(data_list, days=14, selected_indices=None, force_indices=None):
    """
    将数据写入WPS多维表格（使用AirScript）
    包含模拟写入功能，确保稳定运行
    实现查重功能，避免导出重复记录
    
    参数:
        data_list: list - 要写入的数据列表，每个元素是一个字典
        days: int - 查询最近多少天的记录用于查重，默认14天
        selected_indices: set - 指定处理的记录编号集合（从1开始），None表示全部处理
        force_indices: set - 强制写入的记录编号集合（从1开始），这些记录跳过查重直接新增
    
    返回:
        bool - 写入是否成功
    """
    if not data_list:
        print("没有数据需要写入WPS多维表格")
        return True
    
    # 跳过初始的WPS写入功能检查，保留后续完整流程
    
    print_section("开始WPS多维表格写入流程")
    
    # 初始化变量
    existing_records = []
    existing_dict = {}
    
    print("执行查重流程")
    # 1. 获取现有记录用于查重
    existing_records = get_existing_wps_records(days=days)
    print(f"获取到的现有记录数量：{len(existing_records)}")
    
    # 保存现有记录到本地备份文件，用于API调用失败时的回退
    if existing_records:
        try:
            backup_file = os.path.join(config.output_dir, "wps_records_backup.json")
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(existing_records, f, ensure_ascii=False, indent=2)
            print(f"已将现有记录备份到本地文件：{backup_file}")
        except Exception as backup_e:
            print(f"备份现有记录失败：{str(backup_e)}")
    
    # 2. 转换现有记录为字典，便于查重和更新
    existing_records_map = {}  # 用于存储完整的现有记录，包括id
    existing_records_map_zero_amount = {}  # 专门存储金额为0的记录，用于智能匹配
    if _full_records_with_id:
        for record in _full_records_with_id:
            if isinstance(record, dict):
                fields = record.get("fields", {})
                if isinstance(fields, dict):
                    # 构建唯一标识：日期+类型+小区+金额+门店+人员
                    key = f"{fields.get('日期', '').replace('/', '-')}_{fields.get('类型', '')}_{fields.get('小区', '')}_{fields.get('金额', '')}_{fields.get('门店', '')}_{fields.get('人员', '')}"
                    existing_dict[key] = True
                    # 直接存储完整的记录（包括id）
                    existing_records_map[key] = record
                    
                    # 如果金额是0，也存储到专门的字典中，用于智能匹配
                    amount = fields.get('金额', '')
                    if str(amount) == "0" or str(amount) == "":
                        # 构建不含金额的key
                        key_without_amount = f"{fields.get('日期', '').replace('/', '-')}_{fields.get('类型', '')}_{fields.get('小区', '')}_{fields.get('门店', '')}_{fields.get('人员', '')}"
                        existing_records_map_zero_amount[key_without_amount] = record
    print(f"构建了包含 {len(existing_dict)} 条记录的查重字典")
    
    # 内部辅助函数：构建WPS记录和查重键
    def build_wps_record(result):
        """
        构建WPS记录和对应的查重键
        
        参数:
            result: dict - 原始记录数据
            
        返回:
            tuple - (wps_record, key) 转换后的WPS记录和查重键
        """
        record_type = result["类型"]
        
        # 构建WPS记录（按WPS表格要求的顺序）
        wps_record = {
            "日期": result["日期"].replace("-", "/"),  # 转换为WPS格式
            "类型": record_type,
            "小区": result["小区"],
            "面积": result["面积"],
            "金额": result["金额"],
            "门店": result["门店"],
            "人员": result["人员"]
        }
        
        # 构建唯一标识用于查重
        key = f"{result['日期']}_{record_type}_{result['小区']}_{result['金额']}_{result['门店']}_{result['人员']}"
        
        return wps_record, key
    
    # 内部辅助函数：检测字段是否需要更新
    def _check_field_update_needed(new_value, existing_value):
        """
        检测字段是否需要更新
        
        参数:
            new_value: 新值
            existing_value: 现有值
            
        返回:
            bool - 是否需要更新
        """
        try:
            existing_num = float(existing_value) if existing_value else 0
            new_num = float(new_value) if new_value else 0
            if existing_num > 0 and (not new_value or new_num <= 0):
                return False
            return abs(existing_num - new_num) > 0.001
        except (ValueError, TypeError):
            return str(new_value) != str(existing_value)
    
    print(f"\n=== 写入准备 ===")
    print(f"待写入记录总数：{len(data_list)}")
    
    wps_records = []
    duplicate_count = 0
    records_to_update = []  # 存储需要更新的记录
    
    skip_count = 0
    force_write_records = []  # 存储强制写入的记录
    
    # 预处理：相邻记录除金额外其他字段相同时，跳过金额为0的记录
    skip_zero_amount_indices = set()
    for i in range(1, len(data_list)):
        curr = data_list[i]
        prev = data_list[i - 1]
        # 检查是否除金额外其他字段相同（与查重键一致：日期、类型、小区、门店、人员）
        if (str(curr.get('日期', '')) == str(prev.get('日期', '')) and 
            str(curr.get('类型', '')) == str(prev.get('类型', '')) and 
            str(curr.get('小区', '')) == str(prev.get('小区', '')) and
            str(curr.get('门店', '')) == str(prev.get('门店', '')) and
            str(curr.get('人员', '')) == str(prev.get('人员', ''))):
            # 除金额外其他字段相同，比较金额
            try:
                curr_amount = float(curr.get('金额', 0)) if curr.get('金额') else 0
            except (ValueError, TypeError):
                curr_amount = 0
            try:
                prev_amount = float(prev.get('金额', 0)) if prev.get('金额') else 0
            except (ValueError, TypeError):
                prev_amount = 0
            
            # 跳过金额为0的那条（保留金额非0的）
            if curr_amount == 0 and prev_amount > 0:
                skip_zero_amount_indices.add(i)
            elif prev_amount == 0 and curr_amount > 0:
                skip_zero_amount_indices.add(i - 1)
    
    if skip_zero_amount_indices:
        print(f"  🔄 检测到 {len(skip_zero_amount_indices)} 条相邻重复且金额为0的记录，将自动跳过")
    
    for i, result in enumerate(data_list):
        current_index = i + 1
        
        # 如果指定了编号，只处理在列表中的记录
        if selected_indices is not None and current_index not in selected_indices:
            continue
        
        # 检查是否标记为跳过写入
        if result.get('_skip_write', False):
            skip_count += 1
            print(f"  ⚠️  记录 {current_index}：{result['类型']} - {result['人员']} - {result['小区']} - {result['金额']} (全部不在已知列表，跳过写入)")
            continue
        
        # 检查是否是相邻重复且金额为0的记录（跳过金额为0的，保留金额非0的）
        if i in skip_zero_amount_indices:
            skip_count += 1
            print(f"  ⏭️  记录 {current_index}：{result['类型']} - {result['人员']} - {result['小区']} - {result['金额']} (相邻记录除金额外相同且金额为0，跳过)")
            continue
        
        # 使用内部辅助函数构建记录和查重键
        wps_record, key = build_wps_record(result)
        
        # 检查是否是强制写入记录
        if force_indices is not None and current_index in force_indices:
            force_write_records.append(wps_record)
            print(f"  ⚡ 记录 {current_index}：{result['类型']} - {result['人员']} - {result['小区']} - {result['金额']} (强制写入，跳过查重)")
            continue
        
        matched_record = None
        matched_key = None
        
        # 1. 先尝试精确匹配
        if key in existing_records_map:
            matched_record = existing_records_map[key]
            matched_key = key
        else:
            # 2. 如果没精确匹配，尝试智能匹配（匹配金额为0的记录）
            key_without_amount = f"{result['日期']}_{result['类型']}_{result['小区']}_{result['门店']}_{result['人员']}"
            if key_without_amount in existing_records_map_zero_amount:
                matched_record = existing_records_map_zero_amount[key_without_amount]
                matched_key = key_without_amount  # 这是不含金额的key
        
        # 检查是否找到匹配的记录
        if matched_record is None:
            # 没有匹配，准备新增
            wps_records.append(wps_record)
            existing_dict[key] = True
            print(f"  ✅ 记录 {current_index}：{result['类型']} - {result['人员']} - {result['小区']} - {result['金额']} (准备写入)")
        else:
            duplicate_count += 1
            # 检查是否需要更新
            existing_fields = matched_record.get("fields", {})
            
            # 检查需要更新的字段
            fields_to_update = {}
            new_area = result.get("面积", "")
            existing_area = existing_fields.get("面积", "")
            new_amount = result.get("金额", "")
            existing_amount = existing_fields.get("金额", "")
            
            # 使用通用的更新检测函数
            if _check_field_update_needed(new_area, existing_area):
                fields_to_update["面积"] = new_area
            if _check_field_update_needed(new_amount, existing_amount):
                fields_to_update["金额"] = new_amount
            
            if fields_to_update:
                record_id = matched_record.get("id", "")
                if record_id:
                    records_to_update.append({
                        "id": record_id,
                        "fields": fields_to_update
                    })
                    # 构建更新提示信息
                    update_msg = "，".join([f"{k} → {v}" for k, v in fields_to_update.items()])
                    print(f"  🔄 记录 {current_index}：{result['类型']} - {result['人员']} - {result['小区']} - {result['金额']} (重复记录，将更新 {update_msg})")
                else:
                    print(f"  ❌ 记录 {current_index}：{result['类型']} - {result['人员']} - {result['小区']} - {result['金额']} (重复记录，跳过)")
            else:
                print(f"  ❌ 记录 {current_index}：{result['类型']} - {result['人员']} - {result['小区']} - {result['金额']} (重复记录，跳过)")
    
    # 输出跳过的记录数量
    if skip_count > 0:
        print(f"  ⚠️  共跳过 {skip_count} 条全部不在已知列表的记录")
    
    print(f"\n写入统计：")
    print(f"待写入记录：{len(data_list)}, 重复记录：{duplicate_count}, 全部不在已知列表：{skip_count}, 准备写入：{len(wps_records)}, 准备更新：{len(records_to_update)}, 强制写入：{len(force_write_records)}")
    
    # 如果没有需要写入或更新的记录，直接返回成功
    if not wps_records and not records_to_update and not force_write_records:
        print("所有记录都是重复的，无需写入或更新")
        _save_write_count(0)
        return True
    
    # 检查是否启用调试模式
    if not _is_wps_enabled():
        print(f"\n调试模式：跳过实际写入，共 {len(wps_records)} 条新记录，{len(records_to_update)} 条待更新记录，{len(force_write_records)} 条强制写入记录")
        _save_write_count(len(wps_records) + len(force_write_records))
        return True
    
    # 1. 先执行批量更新操作
    update_count = 0
    if records_to_update:
        print(f"\n开始批量更新 {len(records_to_update)} 条记录...")
        try:
            # 批量更新记录
            update_result = _send_wps_request("batch_update", {
                "records": records_to_update
            })
            if update_result.get("success", False):
                update_count = update_result.get("data", {}).get("count", len(records_to_update))
                print(f"  ✅ 批量更新成功：{update_count} 条记录")
                # 打印更新详情
                for record_update in records_to_update:
                    update_msg = "，".join([f"{k}={v}" for k, v in record_update["fields"].items()])
                    print(f"    ID={record_update['id']}: {update_msg}")
            else:
                print(f"  ❌ 批量更新失败：错误={update_result.get('message', '未知')}")
        except Exception as e:
            print(f"  ❌ 批量更新异常：错误={str(e)}")
        print(f"更新完成：成功 {update_count}/{len(records_to_update)} 条")
    
    # 2. 执行强制写入操作
    force_write_count = 0
    if force_write_records:
        print(f"\n开始强制写入 {len(force_write_records)} 条记录...")
        force_result = _send_wps_request("batch_create", {"records": force_write_records})
        if force_result.get("success", False):
            force_write_count = len(force_write_records)
            print(f"  ✅ 强制写入成功：{force_write_count} 条")
        else:
            print(f"  ❌ 强制写入失败：{force_result.get('message', '未知')}")
    
    # 3. 再执行正常新增操作
    actual_result = None
    if wps_records:
        print(f"\n开始写入 {len(wps_records)} 条新记录...")
        # 使用通用请求函数发送写入请求
        actual_result = _send_wps_request("batch_create", {"records": wps_records})
    
    # 处理新增操作的结果
    actual_write_count = 0
    if actual_result:
        if actual_result.get("success", False):
            # 本地计算的写入数量
            expected_write_count = len(wps_records)
            
            # 从响应中获取实际写入的记录数量（如果有）
            actual_write_count = actual_result.get("data", {}).get("count", 0)
            
            # 如果响应中没有返回实际写入数量，尝试通过两次获取的记录数差异计算
            if actual_write_count <= 0:
                # 再次获取现有记录，计算实际写入数量
                try:
                    time.sleep(1)  # 等待1秒，确保数据已写入
                    final_existing_records = get_existing_wps_records()
                    actual_write_count = len(final_existing_records) - len(existing_records)
                    # 确保实际写入数量为正数
                    actual_write_count = max(actual_write_count, expected_write_count)
                except Exception as e:
                    print(f"计算实际写入数量失败，使用预期数量：{str(e)}")
                    actual_write_count = expected_write_count
            
            print(f"成功写入 {actual_write_count} 条记录到WPS多维表格")
            
            # 检查预期写入数量和实际写入数量是否一致
            if expected_write_count != actual_write_count:
                print(f"⚠️  警告：预期写入 {expected_write_count} 条记录，但实际只写入了 {actual_write_count} 条记录")
                print(f"   可能原因：部分记录可能与现有记录重复，或WPS服务器端有重复检查机制")
        else:
            print(f"写入WPS多维表格失败：{actual_result.get('message', '未知错误')}")
    
    # 保存实际写入数量到临时文件
    _save_write_count(actual_write_count)
    
    # 输出最终总结
    print(f"\n操作完成：新增 {actual_write_count} 条，更新 {update_count} 条")
    return True





def delete_records_from_wps_sheet(record_ids):
    """
    从WPS多维表格中删除指定ID的记录
    
    参数:
        record_ids: list - 要删除的记录ID列表
    
    返回:
        bool - 删除是否成功
    """
    if not record_ids:
        print("没有记录ID需要删除")
        return True
    
    # 检查WPS写入功能是否启用
    if not _is_wps_enabled():
        print("无法执行删除操作")
        return True
    
    print(f"\n开始删除 {len(record_ids)} 条记录")
    
    try:
        # 使用通用请求函数发送删除请求
        actual_result = _send_wps_request("batch_delete", {"recordIds": record_ids})
        
        success = actual_result.get("success", False)
        message = actual_result.get("message", "未知错误")
        
        if success is True or success == "true":
            print(f"成功删除 {len(record_ids)} 条记录：{message}")
            return True
        else:
            print(f"删除记录失败：{message}")
            return False
            
    except Exception as e:
        print(f"删除记录异常：{str(e)}")
        traceback.print_exc()
        return False

def delete_recently_written_records():
    """
    删除最近写入的记录，用于纠错
    """
    print("\n=== 开始删除最近写入的记录 ===")
    
    # 1. 获取现有记录（默认只返回最近15天的记录），不进行排序，这会更新_full_records_with_id全局变量
    existing_records = get_existing_wps_records(sort_records=False)
    
    if not existing_records:
        print("没有找到现有记录")
        return
    
    # 2. 显示最近15天的记录，按API返回的原始顺序（ID增序），显示最后30条
    print("\n完整记录（API原始顺序，ID增序）- 最近30条：")
    # 使用API返回的原始顺序（ID增序），取最后30条记录（即ID最大的30条）
    top_30_records = _full_records_with_id[-30:]
    for i, record in enumerate(top_30_records, 1):
        record_id = record.get("id", "")
        fields = record.get("fields", {})
        print(f"{i}. ID: {record_id}")
        print(f"   日期: {fields.get('日期', '')}")
        print(f"   类型: {fields.get('类型', '')}")
        print(f"   小区: {fields.get('小区', '')}")
        print(f"   金额: {fields.get('金额', '')}")
        print(f"   门店: {fields.get('门店', '')}")
        print(f"   人员: {fields.get('人员', '')}")
        print()
    
    # 3. 让用户选择要删除的记录
    try:
        choice = input("请输入要删除的记录编号（多个编号用逗号分隔，或输入'all'删除最近写入的记录）：")
        
        if choice.strip() == "all":
            # 读取最近写入的记录数量
            recent_write_count = 11  # 默认值
            try:
                temp_file = os.path.join(config.output_dir, "last_write_count.txt")
                if os.path.exists(temp_file):
                    with open(temp_file, "r") as f:
                        count_str = f.read().strip()
                        if count_str.isdigit():
                            recent_write_count = max(int(count_str), 11)
                print(f"\n检测到最近写入了 {recent_write_count} 条记录")
            except Exception as e:
                print(f"读取最近写入数量失败，使用默认值11：{str(e)}")
            
            # 删除最近写入的记录
            record_ids_to_delete = [record.get("id", "") for record in _full_records_with_id[-recent_write_count:] if record.get("id", "")]
            if record_ids_to_delete:
                print(f"\n准备删除最近 {len(record_ids_to_delete)} 条记录")
                return delete_records_from_wps_sheet(record_ids_to_delete)
            else:
                print("没有找到可删除的记录ID")
                return False
        else:
            # 解析用户输入的编号，支持多种格式：
            # 1. 单个序号：1, 2, 3
            # 2. 范围格式：15-25（选择从15到25的记录）
            # 3. 起始到结束格式：14+（选择从14到结束的所有记录）
            # 4. 混合使用：1,3-5,8+（选择1,3-5,8到结束的记录）
            
            # 先将中文逗号替换为英文逗号
            choice = choice.replace("，", ",")
            
            # 初始化最终索引集合
            all_indices = set()
            
            # 分割不同的选择项
            items = choice.split(",")
            for item in items:
                item = item.strip()
                if not item:
                    continue
                
                # 处理范围格式：15-25
                if "-" in item:
                    try:
                        start_str, end_str = item.split("-")
                        start = int(start_str.strip()) - 1  # 转换为0-based索引
                        end = int(end_str.strip()) - 1  # 转换为0-based索引
                        # 添加范围内的所有索引
                        for idx in range(start, end + 1):
                            all_indices.add(idx)
                    except ValueError:
                        print(f"无效的范围格式：{item}")
                        continue
                # 处理起始到结束格式：14+
                elif item.endswith("+"):
                    try:
                        start_str = item[:-1].strip()
                        start = int(start_str) - 1  # 转换为0-based索引
                        # 添加从start到结束的所有索引
                        for idx in range(start, len(top_30_records)):
                            all_indices.add(idx)
                    except ValueError:
                        print(f"无效的起始到结束格式：{item}")
                        continue
                # 处理单个序号格式：5
                else:
                    try:
                        idx = int(item) - 1  # 转换为0-based索引
                        all_indices.add(idx)
                    except ValueError:
                        print(f"无效的序号格式：{item}")
                        continue
            
            # 使用之前排序好的top_30_records
            record_ids_to_delete = []
            
            # 遍历所有有效索引，去重并排序
            for idx in sorted(all_indices):
                if 0 <= idx < len(top_30_records):
                    record_id = top_30_records[idx].get("id", "")
                    if record_id:
                        record_ids_to_delete.append(record_id)
            
            if record_ids_to_delete:
                print(f"\n准备删除 {len(record_ids_to_delete)} 条记录")
                return delete_records_from_wps_sheet(record_ids_to_delete)
            else:
                print("没有找到有效的记录ID")
                return False
                
    except ValueError as e:
        print(f"无效的输入：{e}")
        return False
    except Exception as e:
        print(f"删除记录失败：{str(e)}")
        return False


# 主函数辅助函数：打印汇总报告
def print_summary_report(image_files, processed_count, all_results):
    """打印汇总报告"""
    
    # 先打印缺项记录检查
    if all_results:
        incomplete_records = []
        for result in all_results:
            # 检查是否有缺项（小区、门店、人员中有任意一个为空或为"未知"）
            community = result.get('小区', '')
            store = result.get('门店', '')
            person = result.get('人员', '')
            
            is_incomplete = (
                not community or community == '未知' or
                not store or store == '未知' or
                not person or person == '未知'
            )
            
            if is_incomplete:
                incomplete_records.append(result)
        
        if incomplete_records:
            print("\n" + "=" * 120)
            print("缺项记录检查（供人工核对）")
            print("=" * 120)
            print(f"共发现 {len(incomplete_records)} 条缺项记录\n")
            
            for idx, result in enumerate(incomplete_records, 1):
                print(f"\n【缺项记录 {idx}】")
                print("-" * 80)
                
                # 打印OCR原文
                ocr_text = result.get('_ocr_text', '')
                if ocr_text:
                    print("OCR原文：")
                    print(ocr_text)
                else:
                    print("OCR原文：（未保存）")
                
                print("\n识别结果：")
                print(f"  日期: {result.get('日期', '')}")
                print(f"  类型: {result.get('类型', '')}")
                print(f"  小区: {result.get('小区', '')} {'⚠️ 缺项' if not result.get('小区') or result.get('小区') == '未知' else ''}")
                print(f"  金额: {result.get('金额', '')}")
                print(f"  门店: {result.get('门店', '')} {'⚠️ 缺项' if not result.get('门店') or result.get('门店') == '未知' else ''}")
                print(f"  人员: {result.get('人员', '')} {'⚠️ 缺项' if not result.get('人员') or result.get('人员') == '未知' else ''}")
                if result.get('备注'):
                    print(f"  备注: {result.get('备注')}")
                
                print("-" * 80)
            
            print("\n" + "=" * 120)
    
    print(f"处理完成！共处理 {len(image_files)} 个图片，成功解析 {processed_count} 个")
    
    print("\n" + "=" * 120)
    print("房产数据OCR识别汇总报告")
    print("=" * 120)
    print(f"处理总图片数：{len(image_files)}")
    print(f"成功解析数：{processed_count}")
    print(f"失败数：{len(image_files) - processed_count}")
    print(f"成功率：{(processed_count / len(image_files) * 100):.2f}%")
    
    if all_results:
        # 统计各类型数量
        type_counts = {}
        for result in all_results:
            record_type = result['类型']
            type_counts[record_type] = type_counts.get(record_type, 0) + 1
        
        print("\n各类型分布：")
        for type_name, count in type_counts.items():
            print(f"  {type_name}：{count} 条")
    
    print("=" * 120)


# 主函数辅助函数：处理结果排序和时间填充
def process_results(all_results):
    """处理房产喜报识别结果，包括排序和时间填充"""
    if not all_results:
        return all_results
    
    # 1. 确保结果按原始处理顺序排序
    all_results.sort(key=lambda x: x.get("_original_index", 0))
    
    # 2. 为无精确时间的记录填充上一条有精确时间记录的精确时间
    previous_exact_time = None
    previous_date = None
    
    print("\n时间填充过程：")
    print("=" * 80)
    
    for result in all_results:
        has_date = "日期" in result and result["日期"]
        has_exact_time = "_exact_time" in result and result["_exact_time"] != f"{result['日期']} 00:00:00"
        
        if has_exact_time:
            previous_exact_time = result["_exact_time"]
            previous_date = result["日期"]
            print(f"找到精确时间记录：{result['人员']} - {result['_exact_time']}")
        elif not has_date and previous_exact_time:
            # 只有当没有日期时才填充，有日期但没有具体时间的保持原样
            print(f"为记录 {result['人员']} 填充时间：{previous_exact_time}")
            result["_exact_time"] = previous_exact_time
            result["日期"] = previous_date
        
        # 确保所有记录都有精确时间和日期
        if not result.get("_exact_time"):
            result["_exact_time"] = f"{result['日期']} 00:00:00"
        if not result["日期"]:
            result["日期"] = datetime.now().strftime("%Y-%m-%d")
    
    # 3. 按精确时间排序所有记录
    all_results.sort(key=lambda x: x["_exact_time"])
    
    return all_results


# 主函数辅助函数：打印结果汇总并导出CSV
def print_results_summary(all_results):
    """打印房产喜报识别结果汇总，并保存到CSV文件（Excel可直接打开）"""
    # 定义表头
    headers = ["日期", "类型", "小区", "面积", "金额", "门店", "人员", "备注"]
    
    # 将记录分为两类：正常写入的和被跳过的
    normal_results = []
    skipped_results = []
    for result in all_results:
        if result.get('_skip_write', False):
            skipped_results.append(result)
        else:
            normal_results.append(result)
    
    # 准备输出内容
    output_lines = []
    
    # 打印正常写入的记录
    print("\n房产喜报识别结果汇总（按精确时间排序）")
    print("=" * 120)
    print(f"正常写入记录：{len(normal_results)} 条")
    print("=" * 120)
    print(f"{headers[0]:<10} {headers[1]:<8} {headers[2]:<15} {headers[3]:<10} {headers[4]:<10} {headers[5]:<6} {headers[6]:<8}")
    print("=" * 120)
    
    output_lines.append("房产喜报识别结果汇总（按精确时间排序）")
    output_lines.append("=" * 120)
    output_lines.append(f"正常写入记录：{len(normal_results)} 条")
    output_lines.append("=" * 120)
    output_lines.append(f"{headers[0]:<10} {headers[1]:<8} {headers[2]:<15} {headers[3]:<10} {headers[4]:<10} {headers[5]:<6} {headers[6]:<8}")
    output_lines.append("=" * 120)
    
    for result in normal_results:
        note = result.get("备注", "")
        main_data = f"{result['日期']:<10} {result['类型']:<8} {result['小区']:<15} {result['面积']:<10} {result['金额']:<10} {result['门店']:<6} {result['人员']:<8}"
        lines = [main_data]
        if note:
            lines.append(f"    {note}")
        for line in lines:
            print(line)
            output_lines.append(line)
    
    # 打印被跳过的记录
    if skipped_results:
        print("\n" + "=" * 120)
        print(f"跳过写入记录：{len(skipped_results)} 条（仅供核对，不写入WPS）")
        print("=" * 120)
        print(f"{headers[0]:<10} {headers[1]:<8} {headers[2]:<15} {headers[3]:<10} {headers[4]:<10} {headers[5]:<6} {headers[6]:<8}")
        print("=" * 120)
        
        output_lines.append("")
        output_lines.append("=" * 120)
        output_lines.append(f"跳过写入记录：{len(skipped_results)} 条（仅供核对，不写入WPS）")
        output_lines.append("=" * 120)
        output_lines.append(f"{headers[0]:<10} {headers[1]:<8} {headers[2]:<15} {headers[3]:<10} {headers[4]:<10} {headers[5]:<6} {headers[6]:<8}")
        output_lines.append("=" * 120)
        
        for result in skipped_results:
            note = result.get("备注", "")
            main_data = f"{result['日期']:<10} {result['类型']:<8} {result['小区']:<15} {result['面积']:<10} {result['金额']:<10} {result['门店']:<6} {result['人员']:<8}"
            lines = [main_data]
            if note:
                lines.append(f"    {note}")
            for line in lines:
                print(line)
                output_lines.append(line)
    
    print("=" * 120)
    print(f"共 {len(all_results)} 条有效记录（正常写入 {len(normal_results)} 条，跳过 {len(skipped_results)} 条）")
    
    # 保存到CSV文件（Excel可以直接打开）
    import csv
    output_csv_file = os.path.join(config.output_dir, "完整排序 结果.csv")
    try:
        with open(output_csv_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            # 先写正常记录
            writer.writerow(["正常写入记录"])
            writer.writerow(headers)
            for result in normal_results:
                writer.writerow([
                    result['日期'], result['类型'], result['小区'], result['面积'],
                    result['金额'], result['门店'], result['人员'], result.get("备注", "")
                ])
            # 空一行
            writer.writerow([])
            # 再写跳过记录
            if skipped_results:
                writer.writerow(["跳过记录"])
                writer.writerow(headers)
                for result in skipped_results:
                    writer.writerow([
                        result['日期'], result['类型'], result['小区'], result['面积'],
                        result['金额'], result['门店'], result['人员'], result.get("备注", "")
                    ])
        print(f"\n完整排序结果已保存到：{output_csv_file}")
    except Exception as e:
        print(f"\n保存CSV文件失败：{str(e)}")


# 主函数，用于处理指定文件夹下的所有房产喜报图片
def main():
    global dict_changed
    # 初始化小区名字典和人员字典
    init_community_dict()
    init_person_dict()
    
    # =========================
    # 互动式配置部分
    # =========================
    
    # 默认配置
    default_image_folder = r"C:\Users\007\Documents\WXWork\1688855780630843\Cache\Image\2026-08"
    default_days = 2
    
    # 用户输入
    try:
        user_input = input(f"\n=== 配置选项 ===\n1. 修改默认路径和时间设置\n2. 使用默认设置（路径：{default_image_folder}，时间：近{default_days}天）\n请输入1或2（默认使用2）：")
    except:
        user_input = None
    
    # 处理用户输入
    image_folder = default_image_folder
    days = default_days
    
    if user_input == "1":
        print("\n=== 修改配置 ===")
        # 修改图片文件夹路径
        new_folder = input(f"当前图片路径：{default_image_folder}\n请输入新的图片文件夹路径（直接回车保持默认）：")
        if new_folder.strip():
            image_folder = new_folder.strip()
        
        # 修改时间范围
        new_days = input(f"当前时间范围：近{default_days}天\n请输入新的时间范围（天数，直接回车保持默认，输入-表示无限制）：")
        if new_days.strip():
            if new_days.strip() == "-":
                days = None
                print("无时间范围限制，将处理所有图片")
            else:
                try:
                    days = int(new_days.strip())
                    if days <= 0:
                        days = default_days
                        print(f"天数不能为负数，使用默认值：{default_days}天")
                except ValueError:
                    days = default_days
                    print(f"无效的天数，使用默认值：{default_days}天")
    else:
        print(f"\n使用默认配置：路径={image_folder}，时间=近{days}天")
    
    # 检查文件夹是否存在
    if not os.path.exists(image_folder):
        print(f"错误：文件夹不存在 - {image_folder}")
        return
    
    # 获取文件夹下的所有图片文件，仅保留指定天数内的图片
    current_time = time.time()
    
    # 使用辅助函数获取图片文件
    all_image_files = get_image_files(image_folder)
    
    # 筛选近指定天数修改的文件（或无限制）
    image_files = []
    for file_path in all_image_files:
        # 获取文件修改时间
        mtime = os.path.getmtime(file_path)
        # 如果无时间限制，或者文件在指定天数内
        if days is None or current_time - mtime <= days * 24 * 60 * 60:
            image_files.append((mtime, file_path))
    
    # 按修改时间排序（最早的文件先处理）
    image_files.sort(key=lambda x: x[0])
    # 提取排序后的文件路径
    image_files = [file_path for _, file_path in image_files]
    
    # 加载跳过文件列表
    skip_list_file = os.path.join(config.output_dir, "skip_list.json")
    skip_list = set()
    if os.path.exists(skip_list_file):
        try:
            with open(skip_list_file, "r", encoding="utf-8") as f:
                skip_list = set(json.load(f))
            print(f"已加载跳过列表：{len(skip_list)} 个文件")
        except:
            skip_list = set()
    
    # 过滤掉跳过列表中的文件
    original_count = len(image_files)
    image_files = [f for f in image_files if os.path.basename(f) not in skip_list]
    skipped_count = original_count - len(image_files)
    if skipped_count > 0:
        print(f"跳过列表中过滤掉 {skipped_count} 个文件")
    
    print(f"找到 {len(image_files)} 个近两天的图片文件，开始处理...")
    
    # 初始化OCR引擎（只有当PPOCR可用时才需要初始化）
    if PPOCR_AVAILABLE and not init_ocr_engine():
        print("OCR引擎初始化失败，无法继续处理")
        # 只有在字典有变化时才保存
        if dict_changed:
            save_community_dict()
            save_person_dict()
        return
    
    # 收集所有识别结果
    all_results = []
    processed_count = 0
    
    for i, image_path in enumerate(image_files):
        result = process_property_image(image_path)
        
        # 检查是否是需要跳过的文件
        if result == "SKIP_FILE":
            # 将文件名添加到跳过列表
            skip_list.add(os.path.basename(image_path))
            continue
        
        if result:
            property_info, ocr_text, is_umi_ocr = result
            processed_count += 1
            print(f"\n--- 处理第 {i+1}/{len(image_files)} 个图片 --- ")
            property_info["_original_index"] = i  # 添加原始索引
            property_info["_ocr_text"] = ocr_text  # 保存OCR原文
            property_info["_is_umi_ocr"] = is_umi_ocr  # 保存OCR引擎类型
            all_results.append(property_info)
            
            # 如果使用的是PaddleOCR-json，打印OCR原文
            if not is_umi_ocr:
                print("\nPaddleOCR-json原文：")
                print(ocr_text)
            
            # 打印最终解析结果（内联原print_single_result函数逻辑）
            print("\n最终解析结果：")
            print(f"日期: {property_info['日期']}")
            print(f"类型: {property_info['类型']}")
            print(f"小区: {property_info['小区']}")
            print(f"面积: {property_info['面积']}")
            print(f"金额: {property_info['金额']}")
            print(f"门店: {property_info['门店']}")
            print(f"人员: {property_info['人员']}")
            if property_info.get('备注'):
                print(f"备注: {property_info['备注']}")
            print()
            print("=" * 50)
            print()
    
    # 二手和中盘喜报关联匹配与信息互补
    if all_results:
        secondhand = [r for r in all_results if r["类型"] == "二手"]
        medium = [r for r in all_results if r["类型"] == "中盘"]
        if secondhand and medium:
            print("\n" + "=" * 60)
            print("二手和中盘喜报关联匹配与信息互补")
            print("=" * 60)
            match_count = 0
            
            # 通用互补匹配函数
            def _match_and_complement(source_list, target_list, field_name, 
                                      source_type_name, target_type_name,
                                      always_update=False, skip_zero=True):
                """
                通用互补匹配函数
                
                参数:
                    source_list: 数据源列表
                    target_list: 目标数据列表
                    field_name: 要互补的字段名
                    source_type_name: 源类型名称（用于打印）
                    target_type_name: 目标类型名称（用于打印）
                    always_update: 是否总是更新，不管目标是否已有值
                    skip_zero: 是否跳过源字段值为0或空的情况
                    
                返回:
                    int: 匹配并更新的数量
                """
                count = 0
                for target in target_list:
                    target_val = target.get(field_name, "")
                    if not always_update and not (target_val == "0" or not target_val):
                        continue
                    matched = False
                    target_exact_time = target.get("_exact_time", "")
                    # 优先级1：精确时间匹配
                    for source in source_list:
                        source_exact_time = source.get("_exact_time", "")
                        source_val = source.get(field_name, "")
                        if (target_exact_time and source_exact_time and 
                            target_exact_time == source_exact_time and 
                            source_val and 
                            not (skip_zero and source_val == "0")):
                            print(f"  [互补] {target_type_name} {target['小区']}-{target['人员']} {field_name} → {source_val} (精确时间匹配)")
                            target[field_name] = source_val
                            count += 1
                            matched = True
                            break
                    if matched:
                        continue
                    # 优先级2：小区 + 门店 + 人员匹配
                    for source in source_list:
                        source_val = source.get(field_name, "")
                        if (target["小区"] == source["小区"] and 
                            target["门店"] == source["门店"] and 
                            target["人员"] == source["人员"] and 
                            source_val and 
                            not (skip_zero and source_val == "0")):
                            print(f"  [互补] {target_type_name} {target['小区']}-{target['人员']} {field_name} → {source_val} (小区门店人员匹配)")
                            target[field_name] = source_val
                            count += 1
                            matched = True
                            break
                return count
            
            # 二手面积优先使用中盘喜报的面积（总是更新）
            match_count += _match_and_complement(
                medium, secondhand, "面积", "中盘", "二手",
                always_update=True, skip_zero=False
            )
            
            # 二手缺金额 → 从中盘补
            match_count += _match_and_complement(
                medium, secondhand, "金额", "中盘", "二手",
                always_update=False, skip_zero=True
            )
            
            # 中盘缺金额 → 从二手补
            match_count += _match_and_complement(
                secondhand, medium, "金额", "二手", "中盘",
                always_update=False, skip_zero=True
            )
            
            print(f"完成互补：共 {match_count} 条")
            print("=" * 60)
        
        # 林泽武区域的中盘喜报：如果有成交人且没有对应的二手喜报，则新增一条二手记录
        if medium:
            print("\n" + "=" * 60)
            print("中盘喜报检查：为林泽武区域的成交人生成缺失的二手记录")
            print("=" * 60)
            new_secondhand_count = 0
            
            for m in medium:
                # 检查是否有成交人
                dealer = m.get("_成交人", "")
                if not dealer or len(dealer) < 2:
                    continue
                
                # 检查是否已经有对应的二手记录
                has_matched_secondhand = False
                m_exact_time = m.get("_exact_time", "")
                m_community = m.get("小区", "")
                
                # 尝试匹配：精确时间 + 小区 + 成交人
                for sh in secondhand:
                    sh_exact_time = sh.get("_exact_time", "")
                    sh_community = sh.get("小区", "")
                    sh_person = sh.get("人员", "")
                    if (m_exact_time and sh_exact_time and m_exact_time == sh_exact_time and
                        m_community and sh_community and m_community == sh_community and
                        sh_person == dealer):
                        has_matched_secondhand = True
                        break
                
                # 或者：小区 + 人员（成交人）匹配
                if not has_matched_secondhand:
                    for sh in secondhand:
                        sh_community = sh.get("小区", "")
                        sh_person = sh.get("人员", "")
                        if (m_community and sh_community and m_community == sh_community and
                            sh_person == dealer):
                            has_matched_secondhand = True
                            break
                
                # 如果没有匹配的二手记录，则新增一条
                if not has_matched_secondhand:
                    # 检查成交人是否属于林泽武区域（莞南大区）
                    dealer_district = m.get("_成交人大区", "")
                    if dealer_district != "莞南大区":
                        # 成交人不属于林泽武区域，不生成二手记录
                        print(f"  [跳过] 成交人 {dealer} 所属大区 {dealer_district} 不是林泽武区域（莞南大区）")
                        continue
                    
                    # 获取成交人门店（优先使用提取到的，否则为空）
                    dealer_store = m.get("_成交人门店", "")
                    
                    # 创建新的二手记录
                    new_secondhand = {
                        "日期": m["日期"],
                        "类型": "二手",
                        "小区": m["小区"],
                        "面积": m["面积"],
                        "金额": m["金额"],
                        "门店": dealer_store,
                        "人员": dealer,
                        "_exact_time": m.get("_exact_time", f"{m['日期']} 00:00:00"),
                        "_file_time": m.get("_file_time", f"{m['日期']} 00:00:00"),
                        "_区域": m.get("_区域", ""),
                        "_CA": "林泽武",
                        "_原始索引": len(all_results),
                        "备注": f"[自动生成] 来自中盘喜报 {m['人员']}"
                    }
                    
                    # 添加到 all_results
                    all_results.append(new_secondhand)
                    # 更新 secondhand 列表（用于后续处理）
                    secondhand.append(new_secondhand)
                    new_secondhand_count += 1
                    
                    print(f"  [新增] 二手 {new_secondhand['小区']}-{new_secondhand['人员']} "
                          f"门店：{dealer_store if dealer_store else '未知'} "
                          f"来自中盘 {m['小区']}-{m['人员']}")
            
            if new_secondhand_count > 0:
                print(f"自动生成了 {new_secondhand_count} 条二手记录")
            else:
                print("没有需要自动生成的二手记录")
            print("=" * 60)
    
    # 执行打印
    print_summary_report(image_files, processed_count, all_results)
    
    # 处理结果排序和导出
    if all_results:
        # 使用新的辅助函数处理结果
        all_results = process_results(all_results)
        
        # 打印结果汇总
        print_results_summary(all_results)
    
    # 写入数据到WPS多维表格
    if all_results:
        # 准备写入WPS的数据，跳过被标记为不写入的记录
        wps_data = []
        for result in all_results:
            # 检查是否标记为跳过写入
            if result.get('_skip_write', False):
                continue
            
            wps_record = {
                "日期": result["日期"],
                "类型": result["类型"],
                "小区": result["小区"],
                "金额": result["金额"],
                "门店": result["门店"],
                "人员": result["人员"],
                "面积": result["面积"]
            }
            
            # 保留_skip_write标记（如果存在）
            if '_skip_write' in result:
                wps_record['_skip_write'] = result['_skip_write']
            
            wps_data.append(wps_record)
        
        # 执行预览查重（不实际写入WPS）
        print("\n执行预览查重...")
        wps_days = max(days + 1, 7) if days is not None else 365
        # 临时禁用WPS，只执行查重流程
        original_enabled = config.wps_config.get("enabled", True)
        config.wps_config["enabled"] = False
        # 调用write_data_to_wps_sheet，它会执行查重但不会实际写入
        write_data_to_wps_sheet(wps_data, days=wps_days)
        # 恢复WPS设置
        config.wps_config["enabled"] = original_enabled
        
        print(f"\n完整记录数：{len(all_results)} 条")
        print(f"待写入记录数：{len(wps_data)} 条")
        
        # 保存跳过文件列表
        if skip_list:
            try:
                with open(skip_list_file, "w", encoding="utf-8") as f:
                    json.dump(list(skip_list), f, ensure_ascii=False, indent=2)
                print(f"已保存跳过列表：{len(skip_list)} 个文件")
            except Exception as e:
                print(f"保存跳过列表失败：{e}")
        
        # 询问是否写入
        selected_indices = None
        force_indices = None
        write_confirmed = False
        
        while True:
            try:
                print(f"\n请选择操作（待写入记录共 {len(wps_data)} 条）：")
                print(f"[回车] 确认写入全部  [ESC/e] 退出")
                print(f"或输入编号范围（如 18+、7-17、5,8-10，指定处理这些记录）")
                print(f"或输入引号内的编号（如 1,'2-5',8，引号内强制写入）")
                print(f"或输入/加编号范围（如 /1,11-15，反选模式，除了这些记录外其他处理）：")
                
                user_choice = get_key_input("")
                
                # 检查是否是退出命令
                if user_choice.lower() == "e" or user_choice.lower() == "esc":
                    print("\n退出程序...")
                    # 只有在字典有变化时才保存
                    if dict_changed:
                        save_community_dict()
                        save_person_dict()
                    close_ocr_engine()
                    return
                
                # 回车写入全部
                if user_choice == "":
                    selected_indices = None
                    force_indices = None
                    write_confirmed = True
                    break
                
                # 检查是否是反选模式
                invert_selection = False
                selection_input = user_choice
                if user_choice.startswith("/"):
                    invert_selection = True
                    selection_input = user_choice[1:].strip()
                # 同时也处理可能的多余 /，只需要开头一个 / 即可
                selection_input = selection_input.replace("/", "")
                
                # 尝试解析为编号范围（现在返回两个集合：正常选择和强制选择）
                parsed_indices, parsed_force_indices = parse_index_ranges(selection_input, len(wps_data))
                if parsed_indices is not None or parsed_force_indices is not None:
                    # 特殊情况：只有引号内的内容，而parsed_indices为空，只有parsed_force_indices
                    # 这时应该正常写入所有，只指定parsed_force_indices，其他正常处理
                    final_parsed_indices = None  # None表示全部正常处理
                    if not invert_selection:
                        if not parsed_indices and parsed_force_indices:
                            # 只有引号内的内容：其他正常写入
                            final_parsed_indices = None  # 全部正常处理
                            print(f"\n指定强制写入：{sorted(parsed_force_indices)} 条记录")
                            print(f"  强制写入：{sorted(parsed_force_indices)}")
                        elif parsed_indices or parsed_force_indices:
                            all_parsed_indices = (parsed_indices if parsed_indices else set()) | (parsed_force_indices if parsed_force_indices else set())
                            final_parsed_indices = all_parsed_indices
                            if parsed_force_indices:
                                print(f"\n指定录入模式：已选择 {len(parsed_indices)} 条记录，强制写入 {len(parsed_force_indices)} 条记录")
                                if parsed_indices:
                                    print(f"  正常处理：{sorted(parsed_indices)}")
                                print(f"  强制写入：{sorted(parsed_force_indices)}")
                            else:
                                print(f"\n指定录入模式：已选择 {len(parsed_indices)} 条记录：{sorted(parsed_indices)}")
                    
                    if invert_selection:
                        # 反选模式：计算出需要排除的编号
                        all_parsed_indices = (parsed_indices if parsed_indices else set()) | (parsed_force_indices if parsed_force_indices else set())
                        all_indices = set(range(1, len(wps_data) + 1))
                        excluded_indices = all_parsed_indices
                        final_parsed_indices = all_indices - excluded_indices
                        parsed_force_indices = set()  # 反选模式下不支持强制写入
                        print(f"\n反选模式：排除 {len(excluded_indices)} 条记录，选择 {len(final_parsed_indices)} 条记录")
                        print(f"  排除的记录：{sorted(excluded_indices)}")
                        print(f"  选择的记录：{sorted(final_parsed_indices)}")
                    
                    selected_indices = final_parsed_indices
                    force_indices = parsed_force_indices
                    write_confirmed = True
                    break
                else:
                    print("无效输入，请输入回车、e 或有效的编号范围")
            except KeyboardInterrupt:
                print("\n用户取消操作")
                # 只有在字典有变化时才保存
                if dict_changed:
                    save_community_dict()
                    save_person_dict()
                close_ocr_engine()
                return
        
        if write_confirmed:
            # 过滤选中的记录
            filtered_wps_data = []
            if selected_indices is None:
                # 全部正常处理
                filtered_wps_data = wps_data
            else:
                # 只处理选中的记录
                for i, record in enumerate(wps_data):
                    if (i + 1) in selected_indices:
                        filtered_wps_data.append(record)
                print(f"从 {len(wps_data)} 条待写入记录中筛选出 {len(filtered_wps_data)} 条")
            
            if filtered_wps_data:
                print("\n开始写入WPS多维表格...")
                # 写入WPS多维表格（会自动执行查重）
                success = write_data_to_wps_sheet(filtered_wps_data, days=wps_days, force_indices=force_indices)
                if success:
                    print("\n数据写入WPS多维表格成功")
                else:
                    print("\n数据写入WPS多维表格失败，请检查配置和网络连接")
                
                # 更新全局字典，只添加新增的小区和人员（只添加被选中写入的记录中的）
                print("\n更新全局字典...")
                new_persons_count = 0
                new_communities_count = 0
                
                # 从被选中的记录中收集新增的人员和小区
                # 首先，找到被选中的原始记录（从all_results中）
                selected_records = []
                if selected_indices is None:
                    # 全部选中
                    selected_records = [r for r in all_results if not r.get('_skip_write', False)]
                else:
                    # 只选择部分记录
                    # 注意：wps_data是按顺序排列的，与all_results中的有效记录对应
                    # 我们需要找到被选中的原始记录
                    valid_indices = []  # all_results中有效记录的索引
                    for idx, result in enumerate(all_results):
                        if not result.get('_skip_write', False):
                            valid_indices.append(idx)
                    # 现在，selected_indices是wps_data中的编号（从1开始）
                    # 我们需要把它转换为all_results中的索引
                    for i in selected_indices:
                        if i-1 < len(valid_indices):
                            original_idx = valid_indices[i-1]
                            selected_records.append(all_results[original_idx])
                
                # 添加新增的人员和小区
                for result in selected_records:
                    # 添加新增的人员
                    person = result.get('人员')
                    if person and len(person) >= 2:
                        if person not in known_persons:
                            known_persons.append(person)
                            dict_changed = True
                            print(f"  添加新人员：{person}")
                            new_persons_count += 1
                    
                    # 添加新增的小区
                    community = result.get('小区')
                    if community:
                        if community not in known_communities:
                            add_new_community(community, "非大岭山")
                            dict_changed = True
                            print(f"  添加新小区：{community}")
                            new_communities_count += 1
                
                # 输出统计信息
                print(f"\n新增人员：{new_persons_count} 个，新增小区：{new_communities_count} 个")
            else:
                print("\n没有需要写入的记录")
    else:
        print("\n没有数据需要处理")
    
    # 关闭OCR引擎
    close_ocr_engine()
    
    # 只有在字典有变化时才保存（防止覆盖用户手动修改的字典）
    if dict_changed:
        # 保存小区名字典和人员字典到文件（永久存储）
        save_community_dict()
        save_person_dict()
    else:
        print("\n字典无变化，跳过保存")


def parse_index_ranges(input_str, max_index):
    """
    解析用户输入的编号范围字符串，返回编号集合和强制写入集合
    
    支持格式：
    - 18+ : 从18到末尾
    - 7-17 : 从7到17
    - 5 : 单个编号
    - 5,8-10,18+ : 多个范围用逗号分隔
    - '5', '8-10' : 引号内的表示强制写入
    - '5,8-10' : 整个被引号包裹，全部强制写入
    - 全角半角符号通用
    
    参数:
        input_str: str - 用户输入的范围字符串
        max_index: int - 最大编号（记录总数）
    
    返回:
        tuple - (编号集合, 强制写入编号集合) 从1开始的编号
    """
    result = set()
    force_result = set()
    
    # 统一转换全角符号为半角符号
    input_str = input_str.replace("，", ",").replace("－", "-").replace("—", "-").replace("＋", "+")
    
    # 统一转换中文引号为英文引号
    input_str = input_str.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    
    # 检查是否整个字符串被引号包裹
    all_force = False
    original_input = input_str.strip()
    if (original_input.startswith("'") and original_input.endswith("'")) or \
       (original_input.startswith('"') and original_input.endswith('"')):
        all_force = True
        input_str = original_input[1:-1].strip()
    
    # 按逗号分割
    parts = input_str.split(",")
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # 检查是否是强制模式（引号包裹）
        is_force = all_force  # 如果整个被引号包裹，默认都为强制
        original_part = part
        if (part.startswith("'") and part.endswith("'")) or (part.startswith('"') and part.endswith('"')):
            is_force = True
            part = part[1:-1].strip()
        
        try:
            temp_set = set()
            if "+" in part:
                # 格式: 18+ (从某编号到末尾)
                start_str = part.replace("+", "").strip()
                start = int(start_str)
                if start >= 1 and start <= max_index:
                    temp_set.update(range(start, max_index + 1))
            elif "-" in part:
                # 格式: 7-17 (范围)
                range_parts = part.split("-")
                if len(range_parts) == 2:
                    start = int(range_parts[0].strip())
                    end = int(range_parts[1].strip())
                    if start >= 1 and end <= max_index and start <= end:
                        temp_set.update(range(start, end + 1))
            else:
                # 单个编号
                idx = int(part)
                if idx >= 1 and idx <= max_index:
                    temp_set.add(idx)
            
            # 根据是否强制，添加到不同集合
            if is_force:
                force_result.update(temp_set)
            else:
                result.update(temp_set)
                
        except ValueError:
            continue
    
    return result, force_result


# 主程序入口
if __name__ == "__main__":
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--delete":
            # 执行删除功能
            delete_recently_written_records()
    else:
        # 直接运行主程序
        main()

