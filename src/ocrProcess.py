# -*- coding: utf-8 -*-
import subprocess
from pathlib import Path
import pytesseract
import os
import cv2
import re
from bs4 import BeautifulSoup
import numpy as np
from PIL import Image
from rtree import index
from tqdm import tqdm
import time
import argparse
from configManager import ConfigManager, log_mgr
from logManager import LogManager, setup_logging
from errors import ProcessingError, InputError, ResourceError, TimeoutError
from util import Util
from dxfProcess import dxfProcess


log_mgr = LogManager().get_instance()
config_manager = ConfigManager.get_instance()
class OCRProcess:
    def __init__(self):
        pass
    
    def validate_ocr_env(self):
        """验证OCR环境配置"""
        tesseract_exe = config_manager.get_tesseract_path()        
        data_dir = config_manager.get_tesseract_data_path()       
        checks = {
            "Tesseract路径": tesseract_exe,
            "语言包": [
                data_dir / 'chi_sim.traineddata',
                data_dir / 'chi_tra.traineddata'
            ]
        }

        missing = []
        # 检查主程序
        if not Path(checks["Tesseract路径"]).exists():
            missing.append("Tesseract主程序")

        # 检查语言包
        for lang in checks["语言包"]:
            if not lang.exists():
                missing.append(f"语言包 {lang.name}")

        if missing:
            raise EnvironmentError(f"缺少必要组件: {', '.join(missing)}")

    @staticmethod
    def ensure_white_background(img_path, output_path):
        img = cv2.imread(img_path)

        # 如果是彩色图像，先转换为灰度
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img  # 已经是灰度图

        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 统计黑色和白色的像素数量
        white_pixels = np.sum(binary == 255)
        black_pixels = np.sum(binary == 0)

        # 如果黑色像素多，说明背景是黑色，需要反转
        if black_pixels > white_pixels:
            img = cv2.bitwise_not(img)

        # 保存处理后的图片
        cv2.imwrite(output_path, img)
        
    def get_ocr_result_tesseract(self, input_path, output_folder, min_confidence=70, max_height_diff=5, verbose=False):
        """
        获取OCR结果
        :param input_path: 输入图片路径
        :param output_path: 输出hOCR文件路径
        :param min_confidence: 最小置信度阈值
        :param max_height_diff: 同一行内允许的最大高度差异（像素）
        :param verbose: 是否显示详细日志
        """
        base_name = Path(input_path).stem
        hocr_path = Path(output_folder) / f"{base_name}_ocr"
        self.get_text_hocr(input_path, str(hocr_path))
        text_positions = self.parse_hocr_optimized(str(hocr_path) + ".hocr",  min_confidence, max_height_diff, verbose)      
        return text_positions
    
    def get_ocr_result_paddle(self, input_image_path):
        try:
            from paddleocr import PaddleOCR, draw_ocr  
        except ImportError:
            raise RuntimeError("PaddleOCR未安装，请先安装PaddleOCR")
        
        # 获取当前文件所在目录
        base_dir = os.path.dirname(__file__)
        
        # 构建模型路径
        # det_model_dir = os.path.join(base_dir, 'ch_PP-OCRv3', 'ch_PP-OCRv3_det')
        # rec_model_dir = os.path.join(base_dir, 'ch_PP-OCRv3', 'ch_PP-OCRv3_rec')
        # cls_model_dir = os.path.join(base_dir, 'ch_PP-OCRv3', 'ch_ppocr_mobile_v2.0_cls')
        
        # 初始化PaddleOCR，使用中文模型
        ocr = PaddleOCR(
            use_angle_cls=True, 
            lang='ch', 
            det_db_box_thresh=0.3, 
            det_db_unclip_ratio=1.6,
            det_db_score_mode='slow'
            # det_model_dir=det_model_dir, 
            # rec_model_dir=rec_model_dir, 
            # cls_model_dir=cls_model_dir
        )        
        result = ocr.ocr(input_image_path, cls=True)          
        image = Image.open(input_image_path)
        original_width, original_height = image.size
        # 解析结果
        parsed_results = self.parse_ocr_result(result[0], original_height)
        for item in parsed_results:
            print(item)
        return parsed_results, original_height
    
    def preprocess_image(self, image_path):
        """
        图像预处理，提高OCR识别效果
        :param image_path: 输入图片路径
        :return: 预处理后的图像
        """
        # 读取图像
        img = cv2.imread(image_path)
        
        # 1. 图像缩放（如果图片太大）
        max_dimension = 4096  # 最大尺寸
        height, width = img.shape[:2]
        scale = 1
        # if max(height, width) > max_dimension:
        #     scale = max_dimension / max(height, width)
        #     new_width = int(width * scale)
        #     new_height = int(height * scale)
        #     img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # 2. 转换为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 3. 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 4. 降噪
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # 5. 自适应二值化
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # 邻域大小
            2    # 常数差值
        )
        
        return binary, scale

    def _process_block(self, row, col, preprocessed_img, scale_factor, temp_dir, 
                       engine, start_x, end_x, start_y, end_y):
        block_path = None
        try:
            # 提取分块
            block = preprocessed_img[start_y:end_y, start_x:end_x]
            
            # 放大处理
            block = cv2.resize(block, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            
            # 保存分块到临时文件
            block_path = Path(temp_dir) / f"temp_block_{row}_{col}.png"
            cv2.imwrite(block_path, block)
            
            # 对分块进行OCR识别
            result = engine(block_path)
            
            # 检查 OCR 结果
            if result is None or not hasattr(result, 'boxes'):
                print(f"No valid OCR result for block at row {row}, col {col}")
                return []  # 返回空结果以继续处理其他分块
            
            # 调整坐标（加上分块的偏移量，并根据放大倍数缩放）
            adjusted_results = []
            if len(result.boxes) > 0:
                for i in range(len(result.boxes)):
                    box = result.boxes[i].astype(float)
                    # 调整坐标
                    box[:, 0] = box[:, 0] / scale_factor + start_x  # x坐标缩放并加上分块的x偏移
                    box[:, 1] = box[:, 1] / scale_factor + start_y  # y坐标缩放并加上分块的y偏移
                    result.boxes[i] = box
                    # print(f"row: {row}, col: {col}, start_x: {start_x}, end_x: {end_x}, start_y: {start_y}, end_y: {end_y}， box: {box}")
                    adjusted_results.append((box, result.txts[i], result.scores[i]))
            
            return adjusted_results

        except Exception as e:
            print(f"Error processing block at row {row}, col {col}: {e}")
            return []  # 返回空结果以继续处理其他分块  
                
    def _is_vertical_text(self, box):
        x0, y0 = min(point[0] for point in box), min(point[1] for point in box)
        x1, y1 = max(point[0] for point in box), max(point[1] for point in box)

        width = x1 - x0
        height = y1 - y0

        # 假设竖向文本的长宽比大于 2
        return height / width > 2
        
    def get_ocr_result_rapidOCR(self, input_image_path, scale_factor=5, max_block_size=512, overlap=20):
        """
        使用 RapidOCR 进行OCR识别
        :param input_image_path: 输入图片路径
        :param scale_factor: 放大倍数
        :param max_block_size: 每个分块的最大尺寸
        :param overlap: 重叠区域大小
        """
        try:
            from rapidocr import RapidOCR  
        except ImportError:
            raise RuntimeError("RapidOCR未安装，请先安装RapidOCR")        
       
        # 图像预处理
        preprocessed_img, _ = self.preprocess_image(input_image_path)        
        
        # 保存预处理后的图像到临时文件
        temp_dir = Path(input_image_path).parent / "temp" / Path(input_image_path).stem    
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = Path(temp_dir) / "temp_preprocessed.png"
        cv2.imwrite(temp_path, preprocessed_img)
        # input_img = cv2.imread(input_image_path)
        # 初始化 RapidOCR
        engine = RapidOCR()
        
        # 对预处理后的图像进行分块处理
        height, width = preprocessed_img.shape[:2]     
        all_results = []
        
        # 计算分块数量
        num_rows = (height + max_block_size - 1) // max_block_size
        num_cols = (width + max_block_size - 1) // max_block_size    

        # 使用 tqdm 显示进度条
        total_blocks = num_rows * num_cols
        with tqdm(total=total_blocks, desc="Processing Blocks") as pbar:
            for row in range(num_rows):
                start_y = max(0, row * max_block_size - overlap)
                end_y = min(height, (row + 1) * max_block_size + overlap)
                for col in range(num_cols):
                    start_x = max(0, col * max_block_size - overlap)
                    end_x = min(width, (col + 1) * max_block_size + overlap)
                    
                    block_result = self._process_block(row, col, preprocessed_img, scale_factor, temp_dir, 
                                        engine, start_x, end_x, start_y, end_y)
                    all_results.extend(block_result)
                    
                    # 更新进度条
                    pbar.update(1)        
        # Util.remove_directory(temp_dir)
        # 合并结果并去重
        merged_results = self._merge_overlapping_results(all_results)
        
        # 转换为标准格式
        image = Image.open(input_image_path)
        original_width, original_height = image.size
        parsed_results = []
        
        for box, text, score in merged_results:
            x0 = float(min(point[0] for point in box))
            y0 = float(min(point[1] for point in box))
            x1 = float(max(point[0] for point in box))
            y1 = float(max(point[1] for point in box))
            
            # 计算宽度和高度
            width = x1 - x0
            height = y1 - y0     
            
            # 转换坐标
            x, y, width, height = self.convert_to_dxf_coords(x0, y0, x1, y1, original_height)
            if self._is_vertical_text(box):
                x = x + width                
                new_width = height
                new_height = width   
                angle = 90
            else:
                new_width = width
                new_height = height   
                angle = 0      
            # 添加到结果列表
            parsed_results.append((text, x, y, new_width, new_height, angle))
        
        return parsed_results, original_height
        
    def _merge_overlapping_results(self, results):
        """
        合并重叠的OCR结果
        :param results: [(box, text, score), ...]
        :return: 合并后的结果
        """
        if not results:
            return []

        def boxes_overlap(box1, box2, threshold=0.5):
            """检查两个框是否重叠"""
            # 计算框的面积
            def box_area(box):
                return (max(p[0] for p in box) - min(p[0] for p in box)) * \
                       (max(p[1] for p in box) - min(p[1] for p in box))
            
            # 计算交集面积
            x1 = max(min(p[0] for p in box1), min(p[0] for p in box2))
            y1 = max(min(p[1] for p in box1), min(p[1] for p in box2))
            x2 = min(max(p[0] for p in box1), max(p[0] for p in box2))
            y2 = min(max(p[1] for p in box1), max(p[1] for p in box2))
            
            if x1 < x2 and y1 < y2:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = box_area(box1)
                area2 = box_area(box2)
                iou = intersection / min(area1, area2)
                return iou > threshold
            return False

        # 按置信度排序
        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
        merged = []
        used = set()

        # 创建 R-tree 索引
        idx = index.Index()
        for i, (box, text, score) in enumerate(sorted_results):
            x0, y0 = min(p[0] for p in box), min(p[1] for p in box)
            x1, y1 = max(p[0] for p in box), max(p[1] for p in box)
            idx.insert(i, (x0, y0, x1, y1))

        for i, (box1, text1, score1) in enumerate(sorted_results):
            if i in used:
                continue

            current_group = [(box1, text1, score1)]
            used.add(i)

            # 使用 R-tree 查找可能重叠的框
            x0, y0 = min(p[0] for p in box1), min(p[1] for p in box1)
            x1, y1 = max(p[0] for p in box1), max(p[1] for p in box1)
            possible_overlaps = list(idx.intersection((x0, y0, x1, y1)))

            for j in possible_overlaps:
                if j != i and j not in used:
                    box2, text2, score2 = sorted_results[j]
                    if boxes_overlap(box1, box2):
                        used.add(j)
                        # 如果文本相似度高，合并文本
                        if self._text_similarity(text1, text2) > 0.7:
                            current_group.append((box2, text2, score2))

            # 如果只有一个结果，直接添加
            if len(current_group) == 1:
                merged.append(current_group[0])
            else:
                # 合并重叠的结果
                merged_box = current_group[0][0]  # 使用置信度最高的框
                merged_text = max(current_group, key=lambda x: len(x[1]))[1]  # 使用最长的文本
                merged_score = max(x[2] for x in current_group)  # 使用最高的置信度
                merged.append((merged_box, merged_text, merged_score))

        return merged
        
    def _text_similarity(self, text1, text2):
        """
        计算两个文本的相似度
        """
        if not text1 or not text2:
            return 0
            
        # 使用最长公共子序列（LCS）计算相似度
        def lcs(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m):
                for j in range(n):
                    if s1[i] == s2[j]:
                        dp[i + 1][j + 1] = dp[i][j] + 1
                    else:
                        dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
            return dp[m][n]
            
        similarity = 2 * lcs(text1, text2) / (len(text1) + len(text2))
        return similarity

    def get_text_hocr(self, input_path, output_path):
        """增强版OCR处理函数"""
        self.validate_ocr_env()

        # 验证输入文件
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        #img_name = input_file.stem
        #img_dir = input_file.parent
        #wb_img = Path(img_dir) / f"{img_name}_wb.png"
        #OCRProcess.ensure_white_background(input_path, wb_img)
        
        config_manager.set_tesseract_mode("best")
        # 获取可执行路径
        tesseract_exe = config_manager.get_tesseract_path()
        tessdata_dir = config_manager.get_tesseract_data_path()  # 语言包目录
        
        # 添加白名单（根据中文需求调整）
        whitelist = r'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ()<>,.+-±:/°"⌀ '
        # 准备命令
        cmd = [
            tesseract_exe,
            str(input_file),
            str(output_path),
            '-c', 'tessedit_char_whitelist=',
            '-l', 'chi_sim+custom',
             # 显式指定语言包路径
            '--tessdata-dir', str(tessdata_dir),
            '--psm', '11',
            '-c', 'tessedit_create_hocr=1',
            '-c', 'preserve_interword_spaces=1',
            '--oem', '1'
        ]

        try:
            # 设置中文环境变量（仅Windows需要）
            env = os.environ.copy()
            if os.name == 'nt':
                env['TESSDATA_PREFIX'] = str(tessdata_dir.parent)
            result = subprocess.run(
                cmd,              
                capture_output=True,
                text=True,
                check=True,
                shell=False,                 
                env=env 
            )
            return result   
        except subprocess.CalledProcessError as e:
            # 增强错误信息
            error_msg = f"""
            OCR失败！
            命令: {e.cmd}
            错误: {e.stderr if e.stderr else '无输出'}
            请检查:
            1. 语言包是否存在: {tessdata_dir}
            2. 是否包含 chi_sim.traineddata 和 chi_tra.traineddata
            3. 系统环境变量 TESSDATA_PREFIX 是否指向: {tessdata_dir.parent}
            """
            raise RuntimeError(error_msg) from e  

    def get_text_with_rotation(self, input_path, conf_threshold=50):
        # 设置 Tesseract-OCR 路径
        self.validate_ocr_env()

        # 验证输入文件
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        # 获取可执行路径
        tesseract_exe = config_manager.get_tesseract_path()

        # 读取图像
        image = cv2.imread(input_path)
        H, W = image.shape[:2]  # 原始图像尺寸

        # 配置 Tesseract 参数
        config = (
            '-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ(),.+-±:/°"⌀ '
            '--psm 11 '  # 改为PSM 11（稀疏文本自动方向）       
            '-c preserve_interword_spaces=1 '  # 保持空格
            '--oem 1'  # 使用 LSTM OCR 引擎
        )

        def process_image(img, rotation):
            ocr_data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
            transformed_data = []
            for i in range(len(ocr_data["text"])):
                text = ocr_data["text"][i].strip()
                conf = int(ocr_data["conf"][i])  # 置信度
                if text and conf >= conf_threshold:  # 过滤低置信度文本
                    x, y = ocr_data["left"][i], ocr_data["top"][i]
                    width, height = ocr_data["width"][i], ocr_data["height"][i]

                    # 坐标转换到原始图像坐标系
                    if rotation == 0:
                        x_new, y_new = x, H - (y + height)  # Y 轴翻转
                    elif rotation == 90:
                        x_new, y_new = y, x  # 旋转 90°（X 对应 Y，Y 对应 X）
                    elif rotation == 270:
                        x_new, y_new = W - (y + height), x  # 修正旋转 270° 坐标
                    else:
                        continue
                    
                    transformed_data.append((text, x_new, y_new, width, height, rotation))
            return transformed_data

        data_0 = process_image(image, 0)
        # data_90 = process_image(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 90)
        # data_270 = process_image(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), 270)

        # 返回 OCR 数据，包含旋转角度信息
        return data_0       
    
    def adjust_hocr_coordinates(self, hocr_data, original_shape, rotated_shape, angle):
        """
        调整 HOCR 数据中的 bbox 坐标，使其适应旋转后的图像
        """
        height, width = original_shape[:2]
        rotated_height, rotated_width = rotated_shape[:2]

        def transform_bbox(match):
            x1, y1, x2, y2 = map(int, match.groups())

            if angle == 90:
                # 旋转 90°: (x, y) → (height - y2, x)
                new_x1, new_y1 = height - y2, x1
                new_x2, new_y2 = height - y1, x2
            elif angle == 270:
                # 旋转 270°: (x, y) → (y1, width - x2)
                new_x1, new_y1 = y1, width - x2
                new_x2, new_y2 = y2, width - x1
            else:
                return match.group(0)  # 不处理其他角度

            return f'bbox {new_x1} {new_y1} {new_x2} {new_y2}'

        # 使用正则表达式匹配 bbox，并调整坐标
        hocr_data_str = hocr_data.decode("utf-8")
        hocr_data_adjusted = re.sub(r'bbox (\d+) (\d+) (\d+) (\d+)', transform_bbox, hocr_data_str)

        return hocr_data_adjusted.encode("utf-8")  # 转回二进制   

    def convert_to_dxf_coords(self, x0, y0, x1, y1, page_height):
        """
        将hOCR坐标转换为DXF坐标系
        :param x0, y0, x1, y1: hOCR边界框坐标
        :param page_height: 页面高度
        :return: x, y, width, height（DXF坐标系）
        """
        # 计算宽度和高度
        width = x1 - x0
        height = y1 - y0
        
        # 在DXF坐标系中，y轴方向是相反的，原点在左下角
        if page_height is not None:
            y = page_height - y1  # 转换y坐标
        else:
            # 如果页面高度未知，保持原样
            y = y0
        
        return x0, y, width, height

    def parse_hocr_optimized(self, hocr_file, min_confidence=70, max_height_diff=5, verbose=False):
        """
        解析 hOCR 文件，提取 ocr_line 级别的文本，并处理高度不一致的情况
        :param hocr_file: hOCR 文件路径
        :param min_confidence: 最小置信度阈值
        :param max_height_diff: 同一行内允许的最大高度差异（像素）
        :param verbose: 是否显示详细日志
        :return: [(文本, x, y, width, height)], page_height
        """
        text_positions = []
        page_height = None
        
        angle = 0
        # 使用更高效的lxml解析器（如果可用）
        parser = 'lxml' if 'lxml' in BeautifulSoup.DEFAULT_BUILDER_FEATURES else 'html.parser'
        
        try:
            with open(hocr_file, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, parser)

            # 解析页面信息
            ocr_page = soup.find('div', class_='ocr_page')
            if ocr_page:
                page_title = ocr_page.get('title', '')
                m_bbox = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', page_title)
                if m_bbox:
                    _, _, _, page_height = map(int, m_bbox.groups())
                    if verbose:
                        print(f"页面高度: {page_height}")

            # 直接处理每个单词，不依赖于行分组
            all_words = []
            
            # 首先收集所有单词
            for word in soup.find_all('span', class_='ocrx_word'):
                try:
                    word_title = word.get('title', '')
                    word_bbox = re.search(r'bbox (\d+) (\d+) (\d+) (\d+)', word_title)
                    if not word_bbox:
                        continue
                        
                    word_x0, word_y0, word_x1, word_y1 = map(int, word_bbox.groups())
                    word_text = word.get_text().strip()
                    
                    # 跳过空文本
                    if not word_text:
                        continue
                    
                    # 提取单词置信度
                    word_conf_match = re.search(r'x_wconf\s+(\d+)', word_title)
                    word_conf = int(word_conf_match.group(1)) if word_conf_match else 0
                    
                    # 如果置信度低于阈值，跳过
                    if word_conf < min_confidence:
                        continue
                    
                    # 计算转换后的坐标（DXF 坐标系调整）
                    x, y, width, height = self.convert_to_dxf_coords(word_x0, word_y0, word_x1, word_y1, page_height)
                    
                    all_words.append({
                        'text': word_text,
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'original_y0': word_y0,
                        'original_height': word_y1 - word_y0,
                        'confidence': word_conf
                    })
                    
                except Exception as e:
                    if verbose:
                        print(f"解析单词失败：{e}")
            
            # 如果没有找到任何单词，尝试直接处理行
            if not all_words:
                if verbose:
                    print("未找到单词，尝试直接处理行...")
                for line in soup.find_all('span', class_='ocr_line'):
                    try:
                        # 提取坐标
                        title = line.get('title', '')
                        bbox_match = re.search(r'bbox (\d+) (\d+) (\d+) (\d+)', title)
                        if not bbox_match:
                            continue
                        
                        x0, y0, x1, y1 = map(int, bbox_match.groups())
                        text = "".join(line.stripped_strings)  # 获取整行文本
                        
                        # 跳过空文本
                        if not text:
                            continue
                        
                        # 计算转换后的坐标（DXF 坐标系调整）
                        x, y, width, height = self.convert_to_dxf_coords(x0, y0, x1, y1, page_height)
                        
                        # 直接添加到结果中
                        text_positions.append((text, x, y, width, height, angle))
                        
                    except Exception as e:
                        if verbose:
                            print(f"解析行失败：{e}")
                
                # 如果仍然没有找到任何文本，尝试解析ocr_carea
                if not text_positions:
                    if verbose:
                        print("未找到行，尝试解析ocr_carea...")
                    for area in soup.find_all('div', class_='ocr_carea'):
                        try:
                            # 提取坐标
                            title = area.get('title', '')
                            bbox_match = re.search(r'bbox (\d+) (\d+) (\d+) (\d+)', title)
                            if not bbox_match:
                                continue
                            
                            x0, y0, x1, y1 = map(int, bbox_match.groups())
                            text = "".join(area.stripped_strings)  # 获取区域内所有文本
                            
                            # 跳过空文本
                            if not text:
                                continue
                            
                            # 计算转换后的坐标（DXF 坐标系调整）
                            x, y, width, height = self.convert_to_dxf_coords(x0, y0, x1, y1, page_height)
                            
                            # 直接添加到结果中
                            text_positions.append((text, x, y, width, height, angle))
                            
                        except Exception as e:
                            if verbose:
                                print(f"解析区域失败：{e}")
                
                return text_positions, page_height
            
            # 按y坐标分组单词（使用更小的分组间隔）
            word_groups = {}
            for word in all_words:
                # 使用y0作为分组键（四舍五入到最近的5像素，便于分组）
                group_key = round(word['original_y0'] / 5) * 5
                if group_key not in word_groups:
                    word_groups[group_key] = []
                word_groups[group_key].append(word)
            
            # 处理每个单词组
            for group_key, words in word_groups.items():
                # 按x坐标排序，从左到右
                words.sort(key=lambda w: w['x'])
                
                # 如果只有一个单词，直接添加
                if len(words) == 1:
                    word = words[0]
                    text_positions.append((word['text'], word['x'], word['y'], word['width'], word['height'], angle))
                    continue
                
                # 分析同一组内的高度差异
                heights = [word['original_height'] for word in words]
                avg_height = sum(heights) / len(heights)
                
                # 检查每个单词
                current_group = []
                for i, word in enumerate(words):
                    # 检查高度差异
                    height_diff = abs(word['original_height'] - avg_height)
                    
                    # 如果高度差异超过阈值或置信度明显不同，单独处理
                    if height_diff > max_height_diff:
                        # 单独处理这个单词
                        text_positions.append((word['text'], word['x'], word['y'], word['width'], word['height'], angle))
                        if verbose:
                            print(f"检测到高度异常的单词: '{word['text']}', 高度: {word['original_height']}, 平均高度: {avg_height}")
                    else:
                        # 添加到当前组
                        current_group.append(word)
                
                # 处理当前组内的单词（高度一致的）
                if current_group:
                    # 检查是否需要合并单词
                    if len(current_group) > 1:
                        # 检查单词是否应该合并（基于x坐标和宽度）
                        merged_groups = self._merge_text_by_position(current_group)
                        
                        for merged_group in merged_groups:
                            if len(merged_group) == 1:
                                # 单个单词
                                word = merged_group[0]
                                text_positions.append((word['text'], word['x'], word['y'], word['width'], word['height'], angle))
                            else:
                                # 合并单词
                                merged_text = "".join([w['text'] for w in merged_group])
                                # 使用第一个单词的位置，但宽度是合并后的
                                first = merged_group[0]
                                last = merged_group[-1]
                                merged_width = (last['x'] + last['width']) - first['x']
                                text_positions.append((merged_text, first['x'], first['y'], merged_width, first['height'], angle))
                    else:
                        # 只有一个单词
                        word = current_group[0]
                        text_positions.append((word['text'], word['x'], word['y'], word['width'], word['height'], angle))
        
        except Exception as e:
            print(f"解析hOCR文件失败: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
        
        if verbose or len(text_positions) == 0:
            print(f"解析到 {len(text_positions)} 个文本元素")
        
        return text_positions, page_height

    def _merge_text_by_position(self, lines):
        """
        根据文本位置合并应该在一起的文本行
        :param lines: 文本行列表
        :return: 合并后的文本组列表
        """
        if not lines:
            return []
        
        # 按x坐标排序
        lines.sort(key=lambda l: l['x'])
        
        merged_groups = []
        current_group = [lines[0]]
        
        for i in range(1, len(lines)):
            current_line = lines[i]
            previous_line = current_group[-1]
            
            # 计算两个文本之间的距离
            gap = current_line['x'] - (previous_line['x'] + previous_line['width'])
            
            # 估计空格宽度（使用平均字符宽度的近似值）
            avg_char_width = previous_line['width'] / max(1, len(previous_line['text']))
            space_threshold = avg_char_width * 3  # 允许的最大间距
            
            # 如果间距小于阈值，认为是同一行文本
            if gap <= space_threshold:
                current_group.append(current_line)
            else:
                # 开始新的组
                merged_groups.append(current_group)
                current_group = [current_line]
        
        # 添加最后一组
        if current_group:
            merged_groups.append(current_group)
        
        return merged_groups
    
    def verify_chinese_recognition(self):
        """生成中文测试图"""
        text_samples = [
            "永和九年，岁在癸丑",  # 书法文本
            "模型准确率: 98.7%",   # 混合文本
            "Hello世界",          # 中英混合
            "深圳市腾讯计算机系统有限公司"  # 长文本
        ]
        
        # 生成测试图像
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (1000, 600), (255,255,255))
        draw = ImageDraw.Draw(img)
        
        # 使用Windows系统字体
        try:
            font = ImageFont.truetype("msyh.ttc", 40)  # 微软雅黑
        except:
            font = ImageFont.load_default()
        
        y = 10
        for text in text_samples:
            draw.text((10, y), text, fill=(0,0,0), font=font)
            y += 60
        
        # 竖向文本
        x = 800  # 竖向文本起始位置
        vertical_texts = [
            "人工智能",
            "深度学习",
            "计算机视觉"
        ]
        for text in vertical_texts:
            # 逐字绘制实现竖向文本
            y = 10 + (len(text) - 1) * 60  # 从下往上绘制，初始位置在底部
            for char in text:
                # 创建临时图像
                text_image = Image.new('RGBA', (50, 100), (255, 255, 255, 0))
                text_draw = ImageDraw.Draw(text_image)
                text_draw.text((0, 0), char, fill=(0, 0, 0), font=font)
                
                # 旋转文字图像90度
                rotated_text = text_image.rotate(90, expand=1)
                
                # 将旋转后的文字粘贴到主图像
                img.paste(rotated_text, (x, y), rotated_text)
                
                y -= 60  # 每个字符上移
            x += 60  # 每列右移
        
        test_img = Path("chinese_test.png")
        img.save(test_img)
        
        # 执行OCR
        output_hocr = test_img.with_name('chinese_test_wb')
        self.get_text_hocr(test_img, output_hocr)
        
        # 解析结果
        with open(output_hocr, 'r', encoding='utf-8') as f:
            content = f.read()
            missing = []
            for text in text_samples:
                if text not in content:
                    missing.append(text)
            if missing:
                raise ValueError(f"未识别文本: {missing}")      
    

    def process_single_file(self, input_path, output_folder, 
                            scale_factor=4,
                            max_block_size=512, 
                            overlap=50):
        """
        处理单个文件的OCR流程

        :param input_path: 输入文件路径
        :param output_folder: 输出目录
        :param scale_factor: 放大倍数
        :param max_block_size: 每个分块的最大尺寸
        :param overlap: 重叠区域大小
        :return: (文件路径, 是否成功, 输出文件路径)
        """
        try:
            log_mgr.log_info(f"开始处理文件: {input_path}")
            Util.validate_image_file(input_path)
            Util.check_system_resources()                

            os.makedirs(output_folder, exist_ok=True)
            if not os.access(output_folder, os.W_OK):
                raise PermissionError(f"输出目录不可写: {output_folder}")

            base_name = Path(input_path).stem      
            start_time = time.time()
            # OCR处理       
            log_mgr.log_info("执行OCR处理...")
            ocr_process = OCRProcess() 
            text_positions = ocr_process.get_ocr_result_rapidOCR(input_path, scale_factor, max_block_size, overlap)      
            log_mgr.log_processing_time("OCR处理", start_time)
            start_time = time.time()      

            # === 结果整合 ===
            log_mgr.log_info("输出结果...")
            final_output = Path(output_folder) / f"output_{base_name}_1.dxf"    
            dxfProcess.save_to_dxf(str(final_output), [], text_positions, input_path)
            log_mgr.log_processing_time("结果输出", start_time)

            log_mgr.log_info(f"成功处理文件: {input_path}")
            log_mgr.log_info(f"结果输出文件: {final_output}")

            return (input_path, True, str(final_output))           
        except InputError as e:
            log_mgr.log_exception(f"输入错误: {e}")
            return (input_path, False, None)
        except ResourceError as e:
            log_mgr.log_exception(f"系统资源错误: {e}")
            raise  # 向上传递严重错误
        except TimeoutError as e:
            log_mgr.log_exception(f"处理超时: {e}")
            return (input_path, False, None)
        except Exception as e:
            log_mgr.log_exception(f"未处理的异常发生: {e}")
            return (input_path, False, None)
        
    def ocr_process(self, input_path, output_folder=None, 
                    scale_factor=4,    
                    max_block_size=512, 
                    overlap=50):
            """
            安全处理单个文件或文件夹的全流程

            :param input_path: 输入文件或文件夹路径
            :param output_folder: 输出目录
            :param scale_factor: 放大倍数
            :param max_block_size: 每个分块的最大尺寸
            :param overlap: 重叠区域大小
            :return: 处理结果列表 [(文件路径, 是否成功, 输出文件路径)]
            """
            results = []
            fn_start_time = time.time()  

            setup_logging(console=True)
            dxfProcess.setup_dxf_logging()
            
            # 显示当前配置参数
            log_mgr.log_info("\n当前OCR识别参数：")
            log_mgr.log_info(f"├─ 输入图片路径：{input_path}")  
            log_mgr.log_info(f"├─ 输出目录：{output_folder}")
            log_mgr.log_info(f"├─ 放大倍数：{scale_factor}")
            log_mgr.log_info(f"├─ 每个分块的最大尺寸：{max_block_size}")
            log_mgr.log_info(f"├─ 重叠区域大小：{overlap}")

            # 检查输入路径是文件还是文件夹
            if os.path.isdir(input_path):
                # 遍历文件夹中的所有文件
                for root, _, files in os.walk(input_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        result = self.process_single_file(file_path, output_folder)
                        results.append(result)
            else:
                # 处理单个文件
                result = self.process_single_file(input_path, output_folder)
                results.append(result)

            log_mgr.log_processing_time(f"总处理时间", fn_start_time)
            return results
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ocr 工具")
    subparsers = parser.add_subparsers(dest='command')

    # 添加 convert 子命令
    convert_parser = subparsers.add_parser('ocr_process', help='ocr 识别')
    convert_parser.add_argument('input_file', type=str, help='输入文件(路径)')
    convert_parser.add_argument('output_path', type=str, help='输出路径')  
    convert_parser.add_argument('--scale_factor', type=int, default=4, help='放大倍数')
    convert_parser.add_argument('--max_block_size', type=int, default=512, help='每个分块的最大尺寸')
    convert_parser.add_argument('--overlap', type=int, default=50, help='重叠区域大小')
    
     # OCR参数组
    ocr_group = parser.add_argument_group('OCR参数')
    ocr_group.add_argument('--lang', default='chi_sim+eng',
                          help="OCR识别语言（默认: chi_sim+eng）")
    ocr_group.add_argument('--no-ocr', action='store_true',
                          help="禁用文字识别功能")  
    # 解析命令行参数
    args = parser.parse_args()

    ocr_process = OCRProcess()
    if args.command == 'ocr_process':
        output_dir = args.output_path or Util.default_output_path(args.input_file, 'ocr')
        ocr_process.ocr_process(args.input_file, output_dir, args.scale_factor, args.max_block_size, args.overlap)
    else:
        print("请输入正确的命令")