# -*- coding: utf-8 -*-
import subprocess
from pathlib import Path
import pytesseract
import os
import cv2
import re
from bs4 import BeautifulSoup
import numpy as np
from configManager import ConfigManager, log_mgr


config_manager = ConfigManager.get_instance()
class OCRProcess:
    def __init__(self):
        pass
    
    def validate_ocr_env(self):
        """验证OCR环境配置"""
        tesseract_exe = config_manager.get_tesseract_path()
        checks = {
            "Tesseract路径": tesseract_exe,
            "语言包": [
                Path(tesseract_exe).parent / 'tessdata/chi_sim.traineddata',
                Path(tesseract_exe).parent / 'tessdata/chi_tra.traineddata'
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
    
    def get_text_hocr(self, input_path, output_path):
        """增强版OCR处理函数"""
        self.validate_ocr_env()

        # 验证输入文件
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        img_name = input_file.stem
        img_dir = input_file.parent
        wb_img = Path(img_dir) / f"{img_name}_wb.png"
        OCRProcess.ensure_white_background(input_path, wb_img)
        
        # 获取可执行路径
        tesseract_exe = config_manager.get_tesseract_path()
        tessdata_dir = Path(tesseract_exe).parent / 'tessdata'  # 语言包目录
        
        # 添加白名单（根据中文需求调整）
        whitelist = r'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ()<>,.+-±:/°"⌀ '
        # 准备命令
        cmd = [
            tesseract_exe,
            str(wb_img),
            str(output_path),
            '-c', 'tessedit_char_whitelist=',
            '-l', 'chi_sim',
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
        根据页面高度，将 HOCR 中的 bbox 坐标转换为 DXF 坐标系。   
        """   
        width = x1 - x0
        height = y1 - y0
        x_new, y_new = x0, page_height - (y0 + height)      
        return x_new, y_new, width, height

    def parse_hocr_optimized(self, hocr_file, min_confidence=70):
        """
        解析 hOCR 文件，仅提取 ocr_line 级别的文本
        :param hocr_file: hOCR 文件路径
        :param min_confidence: 最小置信度阈值
        :return: [(文本, x, y, width, height)]
        """
        text_positions = []
        page_height = None

        with open(hocr_file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')

        # 解析页面信息
        ocr_page = soup.find('div', class_='ocr_page')
        if ocr_page:
            page_title = ocr_page.get('title', '')
            m_bbox = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', page_title)
            if m_bbox:
                _, _, _, page_height = map(int, m_bbox.groups())

        # 遍历 `ocr_line` 级别的文本（忽略单词）
        for line in soup.find_all('span', class_='ocr_line'):
            try:
                # 提取坐标
                title = line.get('title', '')
                bbox_match = re.search(r'bbox (\d+) (\d+) (\d+) (\d+)', title)
                if not bbox_match:
                    continue

                x0, y0, x1, y1 = map(int, bbox_match.groups())
                text = " ".join(line.stripped_strings)  # 获取整行文本

                # 计算转换后的坐标（DXF 坐标系调整）
                x, y, width, height = self.convert_to_dxf_coords(x0, y0, x1, y1, page_height)

                # 记录文本信息
                text_positions.append((text, x, y, width, height))

            except Exception as e:
                print(f"解析行失败：{e}")

        return text_positions, page_height
    
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