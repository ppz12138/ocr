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
        return extracted_name, ""
    
    # 2. 相似度匹配
    best_match = None
    highest_similarity = 0.0
    
    # 遍历已知名称
    for known_name in known_names:
        if not known_name:
            continue
        
        # 直接使用known_name，移除多余的current_name变量
        
        # 计算相似度
        similarity = calculate_similarity(extracted_name, known_name)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = known_name
    
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


# 小区字典管理函数（保持原有函数签名）

def init_community_dict():
    """
    初始化小区字典
    从文件加载已知小区列表，失败则保持原有数据不变
    """
    global known_communities
    _init_dict("小区名", config.community_dict_file, known_communities, dict)


def save_community_dict():
    """
    保存小区名字典到文件
    防止保存空字典，添加简单备份
    """
    global known_communities
    _save_dict("小区名", config.community_dict_file, known_communities)


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
        r"(?:成交时间|签约时间)[:：]?\s*(\d{4}-\d{2}-\d{2})\s*([\d:：]+)",
        # 匹配日期和时间用点号连接的格式，如"2025-12-26.18:51:01"
        r"(\d{4}-\d{2}-\d{2})\.([\d:：]+)",
        # 匹配日期和时间直接连接的格式
        r"(\d{4}-\d{2}-\d{2})([\d:：]+)",
        # 匹配单独的日期格式，如"2025-12-20"
        r"(\d{4}-\d{2}-\d{2})"
    ]
    
    # 提取日期和时间
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                date_str = match.group(1)
                
                # 验证日期格式
                datetime.strptime(date_str, "%Y-%m-%d")
                final_date = date_str
                
                # 提取时间（如果有）
                if len(match.groups()) > 1 and match.group(2):
                    time_part = match.group(2)
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
        
        # 2. 从非汉字处截断，排除·：:和所有英文字母
        # 匹配第一个非汉字且非排除字符的位置
        truncate_pos = re.search(r"[^a-zA-Z\u4e00-\u9fa5·：:]", community)
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
        r"合同金额[:：]?\s*(\d+)",  # 合同金额：2800
        r"成交价[:：]?\s*(\d+)",  # 成交价：99万
        r"成交金额[:：]?\s*(\d+)",  # 成交金额：216万
        r"(\d+(?:\.\d+)?)万",  # 匹配单独的数字+万字，如192万（支持后面跟着其他文字）
    ]
    
    for pattern in amount_patterns:
        amount_match = re.search(pattern, text)
        if amount_match:
            amount = amount_match.group(1)
            # 如果是万元单位，转换为元
            if "万" in text[amount_match.start():amount_match.end()]:
                amount = str(int(float(amount) * 10000))
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
def extract_area(text):
    """
    从文本中提取面积
    
    参数:
        text: str - OCR提取的文本
    
    返回:
        str - 提取的面积字符串，空值为空字符串
    """
    area = ""
    
    area_patterns = [
        r"(\d+(?:\.\d+)?)\s*[m㎡mlML]",  # 197.16m、184.24ml
        r"(\d+(?:\.\d+)?)\s*(?:平方米|平米)",  # 100平方米、90平米
        r"面积[:：]?\s*(\d+(?:\.\d+)?)",  # 面积：120
    ]
    
    for pattern in area_patterns:
        area_match = re.search(pattern, text)
        if area_match:
            area = area_match.group(1)
            break
    
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


# 人员提取函数
def extract_and_identify_person(text):
    """
    从OCR文本中提取并识别人名
    简化逻辑：格式匹配 → 精确匹配 → 相似度匹配 → 新内容
    """
    # 非人名黑名单（包含区域名、品牌名、门店品牌等）
    non_person_keywords = {"东莞站", "贝壹", "贝联", "林泽武", "租赁", "相信自己", "再创佳绩", "今日共开", "贝壳开单", "喜报来", "开单套报", "开单喜报", "喜报合作", "合作共赢", "共创辉煌", "地产", "成交楼盘", "成交项目", "莞南大区", "莞北大区", "刘志威", "杨文斌", "德佑", "乐远", "C21", "21世纪", "住商", "链家", "贝壳", "中原", "乐有家", "Q房"}
    
    # 1. 定义格式匹配模式
    format_patterns = [
        r"二手开单([\u4e00-\u9fa5]{2,4})",  # 二手开单+姓名（优先匹配）
        r"租赁([\u4e00-\u9fa5]{2,4})",  # 租赁+姓名
        r"(?:^|\n)\s*([\u4e00-\u9fa5]{2,4})[！!。.，,、]?(?:贝壹|贝联)",  # 黑名单关键词前的人名（允许中间有标点）
        r"[Bb]eike\s*([\u4e00-\u9fa5]{2,4})",  # Beike+姓名
        r"贝壳\s*([\u4e00-\u9fa5]{2,4})",  # 贝壳+姓名
        r"租赁\s*([\u4e00-\u9fa5]{2,4})\s*莞南大区",
        r"([\u4e00-\u9fa5]{2,4})莞南大区",  # 人名 + 莞南大区（无空格）
        r"[（(]([\u4e00-\u9fa5]{2,4})",  # 左括号后面的人名（不要求右括号）
        r"(?:^|\n)\s*([\u4e00-\u9fa5]{2,4})\s*(德佑|乐远|C21|21世纪|住商)",  # 姓名+品牌名
        r"二手开单喜报([\u4e00-\u9fa5]{2,4})(?:乐远|德佑|C21|21世纪|住商)",  # 二手开单喜报+姓名+品牌名
        r"莞南大区-([\u4e00-\u9fa5]{2,4})成交项目",  # 莞南大区-姓名-成交项目
        r"维护人[：:]?\s*([\u4e00-\u9fa5]{2,4})",
        r"成交人[：:]?\s*([\u4e00-\u9fa5]{2,4})",
        r"热烈\s*祝\s*贺[：:]?\s*([\u4e00-\u9fa5]{2,4})",
        r"([\u4e00-\u9fa5]{2,4})[一]+(?:[\u4e00-\u9fa5]{2,4})",  # 匹配"XXX一YYY"格式中的XXX
        r"(?:^|\n)\s*([\u4e00-\u9fa5]{2,4})\s*(?:\n|$)",  # 独立姓名行（最后匹配）
    ]
    
    # 2. 检查字符串是否是门店名或包含门店名
    def is_store_related(s):
        for store in config.known_store_names:
            if store == s or store in s:
                return True
        return False
    
    # 3. 格式匹配：收集所有匹配结果
    all_matches = []
    for pattern in format_patterns:
        for match in re.finditer(pattern, text):
            matched = match.group(1).strip()
            # 过滤黑名单关键词和门店名
            if len(matched) >= 2 and matched not in non_person_keywords and not is_store_related(matched):
                all_matches.append(matched)
    
    # 4. 无匹配结果，直接返回
    if not all_matches:
        return "", ""
    
    # 5. 精确匹配：查找已知人名
    for matched in all_matches:
        if matched in known_persons:
            return matched, ""
    
    # 6. 相似度匹配
    best_match = ""
    highest_similarity = 60.0
    similarity_note = ""
    
    for matched in all_matches:
        for known in known_persons:
            similarity = calculate_similarity(matched, known)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = known
                similarity_note = f"[人员相似匹配：{matched}→{best_match}，匹配度：{similarity:.1f}%]"
    
    if best_match:
        return best_match, similarity_note
    
    # 6. 输出新内容
    new_person = all_matches[0]
    return new_person, ""


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
        "_file_time": f"{temp_date} 00:00:00"  # 初始化_file_time，用于排序
    }
    
    # 1. 提取日期和精确时间
    property_info["日期"], exact_time = extract_date(text, image_path)
    property_info["_exact_time"] = exact_time
    property_info["_file_time"] = exact_time
    
    # 2. 提取类型
    property_info["类型"] = extract_type(text, image_path)
    
    # 3. 提取金额
    property_info["金额"] = extract_amount(text)
    
    # 3.5. 提取面积
    property_info["面积"] = extract_area(text)
    
    # 4-6. 提取小区、门店、人员信息
    # 提取小区
    final_community, community_note = extract_and_identify_community(text)
    property_info["小区"] = final_community
    
    # 提取门店
    property_info["门店"], store_note = extract_and_identify_store(text)
    
    # 提取人员
    property_info["人员"], person_note = extract_and_identify_person(text)
    
    # 合并备注信息（简化逻辑）
    notes = list(filter(None, [community_note, store_note, person_note]))  # 过滤空备注
    property_info["备注"] = " ".join(notes)
    
    # 7-10. 统一字段清理、标准化和验证
    property_info = _clean_property_fields(property_info)
    
    # 中盘喜报小区识别优化
    property_info = _optimize_community_for_medium_disk(property_info, text)
    
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
        
        # 维护人门店不在允许列表，直接跳过
        if property_info["门店"] not in allowed_stores:
            print(f"中盘记录跳过：维护人门店 {property_info['门店']} 不在允许列表中")
            return None
        
        # 提取文本中所有已知门店
        unique_stores = list(set(store for store in config.known_store_names if store in text))
        # 排除维护人门店
        if property_info["门店"] in unique_stores:
            unique_stores.remove(property_info["门店"])
        
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
                print(f"中盘记录跳过：维护人门店 {property_info['门店']} 与成交人门店相同")
                return None
            else:
                print(f"中盘记录保留：维护人门店 {property_info['门店']}，检测到其他门店（非已知列表）")
        else:
            # 有成交人门店在允许列表中，跳过
            if any(store in allowed_stores for store in unique_stores):
                print(f"中盘记录跳过：维护人门店 {property_info['门店']} 和成交人门店 {unique_stores} 都在允许列表中")
                return None
            
            print(f"中盘记录保留：维护人门店 {property_info['门店']}，成交人门店 {unique_stores}")
    
    # 保存初始字典状态，用于检查是否都不在已知列表中
    initial_person_in_dict = property_info['人员'] in known_persons
    initial_community_in_dict = property_info['小区'] in known_communities
    initial_store_in_dict = property_info['门店'] in config.known_store_names
    
    # 1. 检查小区、门店、人员中有几个不在已知列表中
    unknown_count = 0
    if not initial_person_in_dict:
        unknown_count += 1
    if not initial_community_in_dict:
        unknown_count += 1
    if not initial_store_in_dict:
        unknown_count += 1
    
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
        # 在返回前更新字典：只有保留的记录才会更新字典
        # 内部辅助函数：更新字典并添加备注
        def update_dict_and_add_note(item, dict_obj, item_type):
            """更新字典并添加备注"""
            if isinstance(dict_obj, dict):
                is_new = item not in dict_obj
                if is_new:
                    dict_obj[item] = item
            else:  # 列表类型
                is_new = item not in dict_obj
                if is_new:
                    dict_obj.append(item)
            
            if is_new:
                note_prefix = "新增" if item_type == "小区" else "新增人员"
                note = f" [{note_prefix}：{item}]"
                if '备注' in property_info and property_info['备注']:
                    property_info['备注'] += note
                else:
                    property_info['备注'] = note[1:]
        
        # 1. 更新人员字典
        if property_info['人员'] and len(property_info['人员']) >= 2:
            update_dict_and_add_note(property_info['人员'], known_persons, "人员")
        
        # 2. 更新小区字典
        if property_info['小区']:
            update_dict_and_add_note(property_info['小区'], known_communities, "小区")
    
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
                return backup_records
        except Exception as backup_e:
            print(f"从本地备份文件加载记录失败：{str(backup_e)}")
        
        return []

def write_data_to_wps_sheet(data_list, days=14):
    """
    将数据写入WPS多维表格（使用AirScript）
    包含模拟写入功能，确保稳定运行
    实现查重功能，避免导出重复记录
    
    参数:
        data_list: list - 要写入的数据列表，每个元素是一个字典
        days: int - 查询最近多少天的记录用于查重，默认14天
    
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
    
    # 无论debug模式还是write模式，都执行查重步骤
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
    
    # 2. 转换现有记录为字典，便于查重
    for record in existing_records:
        if isinstance(record, dict):
            # 处理两种可能的记录格式
            fields = record.get("fields", record)
            if isinstance(fields, dict):
                # 构建唯一标识：日期+类型+小区+金额+门店+人员
                key = f"{fields.get('日期', '').replace('/', '-')}_{fields.get('类型', '')}_{fields.get('小区', '')}_{fields.get('金额', '')}_{fields.get('门店', '')}_{fields.get('人员', '')}"
                existing_dict[key] = True
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
        
        # 构建WPS记录
        wps_record = {
            "日期": result["日期"].replace("-", "/"),  # 转换为WPS格式
            "类型": record_type,
            "小区": result["小区"],
            "金额": result["金额"],
            "门店": result["门店"],
            "人员": result["人员"],
            "面积": result.get("面积", "")
        }
        
        # 不将备注写入WPS，但保留在原始记录中用于总结
        
        # 构建唯一标识用于查重
        key = f"{result['日期']}_{record_type}_{result['小区']}_{result['金额']}_{result['门店']}_{result['人员']}"
        
        return wps_record, key
    
    print(f"\n=== 写入准备 ===")
    print(f"待写入记录总数：{len(data_list)}")
    
    wps_records = []
    duplicate_count = 0
    
    # 无论debug模式还是write模式，都执行查重
    skip_count = 0
    for i, result in enumerate(data_list):
        # 检查是否标记为跳过写入
        if result.get('_skip_write', False):
            skip_count += 1
            print(f"  ⚠️  记录 {i+1}：{result['类型']} - {result['人员']} - {result['小区']} - {result['金额']} (全部不在已知列表，跳过写入)")
            continue
        
        # 使用内部辅助函数构建记录和查重键
        wps_record, key = build_wps_record(result)
        
        # 检查是否为重复记录
        if key not in existing_dict:
            wps_records.append(wps_record)
            existing_dict[key] = True
            print(f"  ✅ 记录 {i+1}：{result['类型']} - {result['人员']} - {result['小区']} - {result['金额']} (准备写入)")
        else:
            duplicate_count += 1
            print(f"  ❌ 记录 {i+1}：{result['类型']} - {result['人员']} - {result['小区']} - {result['金额']} (重复记录，跳过)")
    
    # 输出跳过的记录数量
    if skip_count > 0:
        print(f"  ⚠️  共跳过 {skip_count} 条全部不在已知列表的记录")
    
    print(f"\n写入统计：")
    print(f"待写入记录：{len(data_list)}, 重复记录：{duplicate_count}, 全部不在已知列表：{skip_count}, 准备写入：{len(wps_records)}")
    
    # 如果所有记录都是重复的，直接返回成功
    if not wps_records:
        print("所有记录都是重复的，无需写入")
        _save_write_count(0)
        return True
    
    # 检查是否启用调试模式
    if not _is_wps_enabled():
        print(f"\n调试模式：跳过实际写入，共 {len(wps_records)} 条记录")
        _save_write_count(len(wps_records))
        return True
    
    # 使用通用请求函数发送写入请求
    actual_result = _send_wps_request("batch_create", {"records": wps_records})
    
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
        
        # 保存实际写入数量到临时文件
        _save_write_count(actual_write_count)
        return True
    else:
        print(f"写入WPS多维表格失败：{actual_result.get('message', '未知错误')}")
        print("将执行模拟写入作为回退")
        _save_write_count(0)
        
        # 合并模拟写入功能，避免调用外部函数
        print(f"\n模拟写入 {len(wps_records)} 条记录到WPS多维表格：")
        for i, record in enumerate(wps_records, 1):
            print(f"记录 {i}: {record}")
        print(f"✅ 模拟写入完成，共 {len(wps_records)} 条记录")
        print("注意：请确保已在WPS中更新AirScript脚本，以便使用真实写入功能")
        print("AirScript脚本已在 WPS测试脚本.py 中提供")
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
        has_exact_time = "_exact_time" in result and result["_exact_time"] != f"{result['日期']} 00:00:00"
        
        if has_exact_time:
            previous_exact_time = result["_exact_time"]
            previous_date = result["日期"]
            print(f"找到精确时间记录：{result['人员']} - {result['_exact_time']}")
        elif previous_exact_time:
            # 无精确时间，使用上一条有精确时间记录的精确时间和日期
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


# 主函数辅助函数：打印结果汇总
def print_results_summary(all_results):
    """打印房产喜报识别结果汇总，并保存到文件"""
    # 定义表头
    headers = ["日期", "类型", "小区", "金额", "门店", "人员", "备注"]
    
    # 在控制台显示数据列表汇总
    print("\n房产喜报识别结果汇总（按精确时间排序）")
    print("=" * 100)
    print(f"{headers[0]:<10} {headers[1]:<8} {headers[2]:<15} {headers[3]:<10} {headers[4]:<6} {headers[5]:<8}")
    print("=" * 100)
    
    # 准备输出内容，用于同时打印到控制台和文件
    output_lines = []
    for result in all_results:
        # 内联format_result_line函数的逻辑
        note = result.get("备注", "")
        main_data = f"{result['日期']:<10} {result['类型']:<8} {result['小区']:<15} {result['金额']:<10} {result['门店']:<6} {result['人员']:<8}"
        lines = [main_data]
        if note:
            lines.append(f"    {note}")
        
        # 处理输出
        for line in lines:
            print(line)
            output_lines.append(line)
    
    print("=" * 100)
    print(f"共 {len(all_results)} 条有效记录")
    
    # 保存到文件
    output_file = os.path.join(config.output_dir, "完整排序 结果.txt")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("房产喜报识别结果汇总（按精确时间排序）\n")
            f.write("=" * 100 + "\n")
            f.write(f"{headers[0]:<10} {headers[1]:<8} {headers[2]:<15} {headers[3]:<10} {headers[4]:<6} {headers[5]:<8}\n")
            f.write("=" * 100 + "\n")
            for line in output_lines:
                f.write(line + "\n")
            f.write("=" * 100 + "\n")
            f.write(f"共 {len(all_results)} 条有效记录\n")
        print(f"\n完整排序结果已保存到：{output_file}")
    except Exception as e:
        print(f"保存排序结果到文件失败：{str(e)}")


# 主函数，用于处理指定文件夹下的所有房产喜报图片
def main():
    # 初始化小区名字典和人员字典
    init_community_dict()
    init_person_dict()
    
    # 检查配置文件中的debug_mode，如果为true则禁用WPS功能
    if hasattr(config, 'debug_mode') and config.debug_mode:
        print("调试模式：禁用WPS写入功能")
        config.wps_config["enabled"] = False
    
    # =========================
    # 互动式配置部分
    # =========================
    
    # 默认配置
    default_image_folder = r"C:\Users\007\Documents\WXWork\1688855780630843\Cache\Image\2026-05"
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
        new_days = input(f"当前时间范围：近{default_days}天\n请输入新的时间范围（天数，直接回车保持默认）：")
        if new_days.strip():
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
    days_seconds = days * 24 * 60 * 60  # 指定天数的秒数
    
    # 使用辅助函数获取图片文件
    all_image_files = get_image_files(image_folder)
    
    # 筛选近指定天数修改的文件
    image_files = []
    for file_path in all_image_files:
        # 获取文件修改时间
        mtime = os.path.getmtime(file_path)
        # 仅保留近指定天数修改的文件
        if current_time - mtime <= days_seconds:
            image_files.append((mtime, file_path))
    
    # 按修改时间排序（最早的文件先处理）
    image_files.sort(key=lambda x: x[0])
    # 提取排序后的文件路径
    image_files = [file_path for _, file_path in image_files]
    
    # 加载跳过文件列表
    skip_list_file = os.path.join(os.path.dirname(config.cache_file), "skip_list.json")
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
        # 保存字典后退出（仅在非调试模式下）
        if not config.debug_mode:
            save_community_dict()
            save_person_dict()
        return
    
    # 收集所有识别结果
    all_results = []
    processed_count = 0
    
    # 调试模式下，用于记录新增的小区和人员
    new_communities = []
    new_persons = []
    
    # 保存初始字典，用于后续比较新增条目
    initial_communities = set(known_communities.keys())
    initial_persons = set(known_persons)
    
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
            print(f"\n--- 处理第 {processed_count}/{len(image_files)} 个图片 --- ")
            property_info["_original_index"] = i  # 添加原始索引
            property_info["_ocr_text"] = ocr_text  # 保存OCR原文
            property_info["_is_umi_ocr"] = is_umi_ocr  # 保存OCR引擎类型
            all_results.append(property_info)
            
            # 只记录非全部未知记录的新增人员和小区
            if not property_info.get('_skip_write', False):
                # 人员
                person = property_info.get('人员')
                if person and len(person) >= 2:
                    if person not in initial_persons:
                        initial_persons.add(person)  # 更新初始人员集合，避免重复记录
                        new_persons.append(person)
                
                # 小区
                community = property_info.get('小区')
                if community:
                    if community not in initial_communities:
                        initial_communities.add(community)  # 更新初始小区集合，避免重复记录
                        new_communities.append(community)
            
            # 如果使用的是PaddleOCR-json，打印OCR原文
            if not is_umi_ocr:
                print("\nPaddleOCR-json原文：")
                print(ocr_text)
            
            # 打印最终解析结果（内联原print_single_result函数逻辑）
            print("\n最终解析结果：")
            print(f"日期: {property_info['日期']}")
            print(f"类型: {property_info['类型']}")
            print(f"小区: {property_info['小区']}")
            print(f"金额: {property_info['金额']}")
            print(f"门店: {property_info['门店']}")
            print(f"人员: {property_info['人员']}")
            if property_info.get('备注'):
                print(f"备注: {property_info['备注']}")
            print()
            print("=" * 50)
            print()
    
    # 执行打印
    print_summary_report(image_files, processed_count, all_results)
    
    # 处理结果排序和导出
    if all_results:
        # 使用新的辅助函数处理结果
        all_results = process_results(all_results)
        
        # 打印结果汇总
        print_results_summary(all_results)
    
    # 写入数据到WPS多维表格或缓存文件
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
                "人员": result["人员"]
            }
            
            # 保留_skip_write标记（如果存在）
            if '_skip_write' in result:
                wps_record['_skip_write'] = result['_skip_write']
            
            wps_data.append(wps_record)
        
        # 调试模式：执行查重，然后保存完整记录和新增的小区、人员到缓存文件
        if config.debug_mode:
            print("\n调试模式：执行查重并保存完整记录和新增的小区、人员到缓存文件")
            # 执行查重，但不实际写入WPS
            write_data_to_wps_sheet(wps_data, days=max(days+1, 7))
            
            # 保存完整记录、待写入记录和新增的小区、人员到缓存文件
            cache_data = {
                "new_communities": new_communities,
                "new_persons": new_persons,
                "full_records": all_results,  # 保存完整记录用于后续查重
                "wps_data": wps_data  # 保存待写入记录用于指定录入
            }
            
            # 确保缓存目录存在
            cache_dir = os.path.dirname(config.cache_file)
            os.makedirs(cache_dir, exist_ok=True)
            
            # 保存数据到缓存文件
            with open(config.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            print(f"数据已保存到缓存文件：{config.cache_file}")
            print(f"  完整记录数：{len(all_results)} 条")
            print(f"  待写入记录数：{len(wps_data)} 条")
            print(f"  新增小区：{len(new_communities)} 个")
            print(f"  新增人员：{len(new_persons)} 个")
            
            # 保存跳过文件列表（在询问之前保存）
            if skip_list:
                try:
                    with open(skip_list_file, "w", encoding="utf-8") as f:
                        json.dump(list(skip_list), f, ensure_ascii=False, indent=2)
                    print(f"已保存跳过列表：{len(skip_list)} 个文件")
                except Exception as e:
                    print(f"保存跳过列表失败：{e}")
            
            # 询问是否写入
            while True:
                try:
                    user_choice = input(f"\n请选择操作（待写入记录共 {len(wps_data)} 条）：\n[回车/q] 确认写入全部  [w] 重新执行OCR  [e] 退出\n或输入编号范围（如 18+、7-17、5,8-10，可强制写入重复记录）：").strip()
                    if user_choice == "" or user_choice.lower() == "q":
                        print("\n开始写入WPS多维表格...")
                        # 启用WPS功能
                        config.wps_config["enabled"] = True
                        config.debug_mode = False
                        write_cache_data(days=max(days+1, 7))
                        break
                    elif user_choice.lower() == "w":
                        print("\n重新执行OCR程序...")
                        main()  # 重新执行main函数
                        return  # 退出当前执行，避免重复询问
                    elif user_choice.lower() == "e":
                        print("\n退出程序...")
                        return  # 退出程序
                    else:
                        # 尝试解析为编号范围
                        selected_indices = parse_index_ranges(user_choice, len(wps_data))
                        if selected_indices:
                            print(f"\n指定录入模式：已选择 {len(selected_indices)} 条记录：{sorted(selected_indices)}")
                            # 启用WPS功能
                            config.wps_config["enabled"] = True
                            config.debug_mode = False
                            write_cache_data(days=max(days+1, 7), selected_indices=selected_indices, force_write=True)
                            break
                        else:
                            print("无效输入，请输入回车/q/w/e 或有效的编号范围")
                except KeyboardInterrupt:
                    print("\n用户取消操作")
                    break
        else:
            # 写入模式：直接写入WPS和字典，不使用缓存
            print("\n写入模式：直接写入WPS多维表格")
            # 写入WPS多维表格（会自动执行查重）
            success = write_data_to_wps_sheet(wps_data, days=max(days+1, 7))
            if success:
                print("\n数据写入WPS多维表格成功")
            else:
                print("\n数据写入WPS多维表格失败，请检查配置和网络连接")
    else:
        print("\n没有数据需要处理")
    
    # 关闭OCR引擎
    close_ocr_engine()
    
    # 保存小区名字典和人员字典到文件（永久存储）（仅在非调试模式下）
    if not config.debug_mode:
        save_community_dict()
        save_person_dict()
    else:
        print("调试模式：跳过保存小区名字典和人员字典到文件")


# 写入缓存数据到WPS和字典的功能

def parse_index_ranges(input_str, max_index):
    """
    解析用户输入的编号范围字符串，返回编号集合
    
    支持格式：
    - 18+ : 从18到末尾
    - 7-17 : 从7到17
    - 5 : 单个编号
    - 5,8-10,18+ : 多个范围用逗号分隔
    - 全角半角符号通用
    
    参数:
        input_str: str - 用户输入的范围字符串
        max_index: int - 最大编号（记录总数）
    
    返回:
        set - 编号集合（从1开始的编号）
    """
    result = set()
    
    # 统一转换全角符号为半角符号
    input_str = input_str.replace("，", ",").replace("－", "-").replace("—", "-").replace("＋", "+")
    
    # 按逗号分割
    parts = input_str.split(",")
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        try:
            if "+" in part:
                # 格式: 18+ (从某编号到末尾)
                start_str = part.replace("+", "").strip()
                start = int(start_str)
                if start >= 1 and start <= max_index:
                    result.update(range(start, max_index + 1))
            elif "-" in part:
                # 格式: 7-17 (范围)
                range_parts = part.split("-")
                if len(range_parts) == 2:
                    start = int(range_parts[0].strip())
                    end = int(range_parts[1].strip())
                    if start >= 1 and end <= max_index and start <= end:
                        result.update(range(start, end + 1))
            else:
                # 单个编号
                idx = int(part)
                if idx >= 1 and idx <= max_index:
                    result.add(idx)
        except ValueError:
            continue
    
    return result


def write_cache_data(days=30, selected_indices=None, force_write=False):
    """
    将缓存文件中的数据写入到WPS多维表格、小区字典和人员字典
    执行步骤：
    1. 初始化现有字典（从文件加载）
    2. 从缓存文件读取数据（新增小区、新增人员和完整记录）
    3. 执行查重并写入WPS
    4. 更新字典（只添加新增条目，不覆盖原有记录）
    5. 清空缓存文件
    
    参数:
        days: int - 查询最近多少天的记录用于查重，默认30天
        selected_indices: set - 指定录入的记录编号集合（从1开始），None表示全部录入
        force_write: bool - 是否强制写入（跳过查重），默认False
    """
    print("\n=== 开始写入缓存数据 ===")
    
    # 1. 初始化现有字典（从文件加载），确保不丢失原有数据
    print("\n初始化现有字典...")
    init_community_dict()
    init_person_dict()
    print("现有字典初始化完成")
    
    # 检查缓存文件是否存在
    if not os.path.exists(config.cache_file):
        print(f"错误：缓存文件不存在 - {config.cache_file}")
        return
    
    # 读取缓存文件
    with open(config.cache_file, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    
    if not cache_data:
        print("缓存文件中没有数据需要写入")
        return
    
    # 提取完整记录用于查重和写入WPS
    if "full_records" in cache_data:
        all_results = cache_data["full_records"]
        print(f"从缓存文件读取到 {len(all_results)} 条完整记录")
    else:
        # 兼容旧格式
        all_results = cache_data
        print(f"从缓存文件读取到 {len(all_results)} 条完整记录（旧格式）")
    
    # 提取待写入记录
    wps_data_cached = cache_data.get("wps_data", None)
    
    # 如果指定了选择性录入，使用 wps_data 进行筛选
    if selected_indices is not None and wps_data_cached is not None:
        filtered_data = []
        for i, record in enumerate(wps_data_cached):
            if (i + 1) in selected_indices:
                filtered_data.append(record)
        print(f"指定录入模式：从 {len(wps_data_cached)} 条待写入记录中筛选出 {len(filtered_data)} 条")
        wps_data_to_write = filtered_data
    elif wps_data_cached is not None:
        wps_data_to_write = wps_data_cached
    else:
        # 兼容旧格式：从 all_results 构建 wps_data
        wps_data_to_write = []
        for result in all_results:
            if result.get('_skip_write', False):
                continue
            wps_record = {
                "日期": result["日期"],
                "类型": result["类型"],
                "小区": result["小区"],
                "金额": result["金额"],
                "门店": result["门店"],
                "人员": result["人员"]
            }
            if "_skip_write" in result:
                wps_record["_skip_write"] = result["_skip_write"]
            wps_data_to_write.append(wps_record)
    
    # 提取新增的小区和人员
    new_communities = cache_data.get("new_communities", [])
    new_persons = cache_data.get("new_persons", [])
    
    # 2. 执行查重并写入WPS多维表格
    if force_write and selected_indices is not None:
        # 强制写入模式：跳过查重，直接写入
        print("\n强制写入模式：跳过查重，直接写入WPS多维表格...")
        print(f"待写入记录数：{len(wps_data_to_write)}")
        
        # 构建WPS格式的记录
        wps_records = []
        for record in wps_data_to_write:
            wps_record = {
                "日期": record["日期"].replace("-", "/") if "-" in record["日期"] else record["日期"],
                "类型": record["类型"],
                "小区": record["小区"],
                "金额": record["金额"],
                "门店": record["门店"],
                "人员": record["人员"]
            }
            wps_records.append(wps_record)
        
        # 直接调用WPS API写入
        if wps_records:
            actual_result = _send_wps_request("batch_create", {"records": wps_records})
            if actual_result.get("success", False):
                print(f"成功写入 {len(wps_records)} 条记录到WPS多维表格")
            else:
                print(f"写入WPS多维表格失败：{actual_result.get('message', '未知错误')}")
        else:
            print("没有记录需要写入")
        success = True
    else:
        # 正常模式：执行查重后写入
        print("\n写入数据到WPS多维表格...")
        success = write_data_to_wps_sheet(wps_data_to_write, days=days)
        if success:
            print("数据写入WPS多维表格成功")
        else:
            print("数据写入WPS多维表格失败，请检查配置和网络连接")
    
    # 3. 更新全局字典，只添加新增的小区和人员
    print("\n更新全局字典...")
    new_persons_count = 0
    new_communities_count = 0
    
    # 添加新增的人员
    for person in new_persons:
        if person not in known_persons:
            known_persons.append(person)
            print(f"  添加新人员：{person}")
            new_persons_count += 1
    
    # 添加新增的小区
    for community in new_communities:
        if community not in known_communities:
            known_communities[community] = community
            print(f"  添加新小区：{community}")
            new_communities_count += 1
    
    # 兼容旧格式：如果没有明确的新增列表，则遍历完整记录提取
    if not new_communities and not new_persons:
        for result in all_results:
            # 检查是否标记为全部未知（跳过写入）
            if result.get('_skip_write', False):
                print(f"  跳过全部未知记录：{result.get('人员', '')} - {result.get('小区', '')} (不更新字典)")
                continue
            
            # 更新人员字典 - 只添加新增条目
            if result.get('人员') and len(result['人员']) >= 2:
                if result['人员'] not in known_persons:
                    known_persons.append(result['人员'])
                    print(f"  添加新人员：{result['人员']}")
                    new_persons_count += 1
            
            # 更新小区字典 - 只添加新增条目
            if result.get('小区'):
                if result['小区'] not in known_communities:
                    known_communities[result['小区']] = result['小区']
                    print(f"  添加新小区：{result['小区']}")
                    new_communities_count += 1
    
    # 输出统计信息
    print(f"\n新增人员：{new_persons_count} 个，新增小区：{new_communities_count} 个")
    
    # 保存小区字典和人员字典
    print("\n保存小区字典和人员字典...")
    save_community_dict()
    save_person_dict()
    print("小区字典和人员字典保存成功")
    
    # 4. 写入完成后清空缓存文件，避免下次重复处理
    print("\n清空缓存文件...")
    with open(config.cache_file, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=2)
    print(f"缓存文件已清空：{config.cache_file}")
    
    print("\n=== 缓存数据写入完成 ===")

# 包装main函数，确保键鼠操作在异常情况下也能恢复
# 主程序入口
if __name__ == "__main__":
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--delete":
            # 执行删除功能
            delete_recently_written_records()
        elif sys.argv[1] == "--debug":
            # 调试模式：不写入数据到WPS
            print("进入调试模式，不写入数据到WPS多维表格")
            config.wps_config["enabled"] = False
            config.debug_mode = True  # 设置调试模式标志
            main()
        elif sys.argv[1] == "--write-cache":
            # 写入模式：将缓存文件数据写入到WPS和字典
            print("进入写入模式，将缓存文件数据写入到WPS多维表格、小区字典和人员字典")
            write_cache_data()
    else:
        # 默认使用调试模式
        print("默认进入调试模式，不写入数据到WPS多维表格")
        config.wps_config["enabled"] = False
        config.debug_mode = True  # 设置调试模式标志
        main()

