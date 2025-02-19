import os
import fitz  # PyMuPDF
from lxml import etree
import xml.etree.ElementTree as ET
import argparse
import subprocess
import cv2
from bs4 import BeautifulSoup
import numpy as np
from scipy.spatial import Voronoi
from shapely.ops import unary_union
from shapely.geometry import LineString, box
from scipy.spatial import Voronoi
import numpy as np
import ezdxf
import pytesseract
from ezdxf import units
from ezdxf.enums import TextEntityAlignment
from Centerline.geometry import Centerline
from shapely.geometry import Polygon, MultiPolygon, MultiLineString, LineString
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from rtree import index
import shutil
import time
import os
import re
import tempfile
import threading
import multiprocessing
     
print_lock = threading.Lock()   
# 将 PNG 转换为 PBM 格式
def convert_png_to_pbm(png_path, pbm_path):
    img = cv2.imread(png_path)  
    if img is None:
        raise ValueError(f"Failed to read the image at {png_path}. Please check the file path or format.")
    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用阈值操作将图像转换为黑白图像
    _, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # skeleton = skeletonize(binary_img)
    # ShowImage.show_image(skeleton, "Skeleton")
    # 保存二值化图像
    
    # 过滤掉过小的区域
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 3 or h < 3:  # 忽略宽度或高度小于 3 的区域
            cv2.drawContours(binary_img, [contour], -1, 0, -1)
    
    cv2.imwrite(pbm_path, binary_img)   

def extract_polygons_from_dxf(file_path):
    """从 DXF 文件中提取多边形数据"""
    try:
        doc = ezdxf.readfile(file_path)
    except IOError:
        print(f"无法读取 DXF 文件: {file_path}")
        return []

    msp = doc.modelspace()
    entities = list(msp.query("POLYLINE LWPOLYLINE"))
    polygons = []

    def process_entity(entity):
        """处理单个实体"""
        if entity.dxftype() == "LWPOLYLINE":
            points = list(entity.vertices())
            if len(points) >= 3:
                return points
        elif entity.dxftype() == "POLYLINE":
            points = [vertex.dxf.location for vertex in entity.vertices]
            if len(points) >= 3:
                return points
        return None
    max_workers = max(1, os.cpu_count() // 2)
    # 使用线程池并行处理实体
    with ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(process_entity, entity) for entity in entities]
        for future in futures:
            result = future.result()
            if result is not None:
                polygons.append(result)

    return polygons

# 合并近似的线段
def merge_lines_with_hough(lines, padding=0):
    """
    使用霍夫变换合并近似的线段，并确保结果与原始线条对齐
    :param lines: 输入的线段列表（MultiLineString 格式）
    :param padding: 图像边界扩展（默认为 0）
    :return: 合并后的线段列表
    """
    if not isinstance(lines, MultiLineString):
        raise ValueError("输入必须是 MultiLineString 类型")

    # 计算输入线段的边界范围
    min_x, min_y, max_x, max_y = lines.bounds

    # 动态调整图像大小并添加 padding
    width = int(max_x - min_x + 2 * padding)
    height = int(max_y - min_y + 2 * padding)
    img = np.zeros((height, width), dtype=np.uint8)

    # 将实际坐标映射到图像坐标
    points = []
    for line in lines.geoms:
        for start, end in zip(line.coords[:-1], line.coords[1:]):
            start_mapped = (int(start[0] - min_x + padding), int(start[1] - min_y + padding))
            end_mapped = (int(end[0] - min_x + padding), int(end[1] - min_y + padding))
            points.append([start_mapped[0], start_mapped[1], end_mapped[0], end_mapped[1]])
            # 在图像上绘制线段
            cv2.line(img, start_mapped, end_mapped, 255, 1)
            
    # 检查输入图像通道数
    if len(img.shape) == 2 or img.shape[2] == 1:
        # 图像已经是灰度图，无需转换
        gray = img
    else:
        # 图像是彩色图，转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    # 二值化
    _, binary = cv2.threshold(blurred, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

    # 边缘检测
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    # 使用霍夫变换检测线段
    lines_detected = cv2.HoughLinesP(
        img,
        rho=1,                          # 距离分辨率
        theta=np.pi / 180,              # 角度分辨率
        threshold=20,                   # 累加器阈值
        minLineLength=10,               # 线段的最小长度
        maxLineGap=20                   # 线段之间的最大间隔
    )

    # 将检测到的线段映射回实际坐标
    merged_lines = []
    if lines_detected is not None:
        for line in lines_detected:
            x1, y1, x2, y2 = line[0]
            # 将图像坐标映射回实际坐标
            x1_actual = x1 + min_x - padding
            y1_actual = y1 + min_y - padding
            x2_actual = x2 + min_x - padding
            y2_actual = y2 + min_y - padding
            merged_lines.append([(x1_actual, y1_actual), (x2_actual, y2_actual)])

    return merged_lines  

def setup_text_styles(doc):
    style = doc.styles.new('工程字体')
    style.font = 'simfang.ttf'  # 仿宋字体
    style.width = 0.7  # 长宽比

def add_text(msp, text, position, height, rotation=0, layer='文本'):
    text_entity = msp.add_text(
        text,
        dxfattribs={
            'height': height,
            "rotation": rotation,
            'color': 2,
            'layer': layer,
            'style': '工程字体',  # 需要提前定义文字样式           
        }
    )
    # 正确设置对齐基准点
    text_entity.set_placement(
        position,  # 基准点位置
        align=TextEntityAlignment.LEFT 
    )
    return text_entity

def add_ridges(msp, ridges):
    """
    批量将 Voronoi 图的边添加到 DXF
    :param msp: DXF ModelSpace
    :param ridges: 可为 LineString、MultiLineString、List（包含 LineString 或 点集）
    """
    line_segments = []  # 缓存所有线段

    def collect_lines(coords):
        """收集线段"""
        for start, end in zip(coords[:-1], coords[1:]):
            line_segments.append((start, end))

    # 处理各种输入类型
    if isinstance(ridges, LineString):
        collect_lines(ridges.coords)

    elif isinstance(ridges, MultiLineString):
        for line in ridges.geoms:
            collect_lines(line.coords)

    elif isinstance(ridges, list):
        for child in ridges:
            if isinstance(child, LineString):
                collect_lines(child.coords)
            elif isinstance(child, MultiLineString):
                for line in child.geoms:
                    collect_lines(line.coords)
            elif isinstance(child, list) and len(child) >= 2:
                # 直接处理点集 (x, y) -> LWPOLYLINE
                msp.add_lwpolyline(child, close=False, dxfattribs={"color": 9, "layer": "脊线"})

    # 批量添加线段
    if line_segments:
        msp.add_lwpolyline(
            [p for segment in line_segments for p in segment],
            close=False,
            dxfattribs={"color": 9, "layer": "脊线"}
        )

def upgrade_dxf(input_file, output_file, target_version="R2010"):
    """正确升级 DXF 版本，并迁移数据"""
    try:
        # 读取旧 DXF 文件
        old_doc = ezdxf.readfile(input_file)
        old_msp = old_doc.modelspace()
    except IOError:
        print(f"无法读取 DXF 文件: {input_file}")
        return False

    # 创建一个新的 DXF 文档（指定版本）
    new_doc = ezdxf.new(target_version)
    new_msp = new_doc.modelspace()

    # 复制所有实体到新 DXF
    entity_count = 0
    for entity in old_msp:
        try:
            new_msp.add_entity(entity.copy())  # 复制实体
            entity_count += 1
        except Exception as e:
            print(f"跳过无法复制的实体: {e}")

    # 保存到新文件
    new_doc.saveas(output_file)
    print(f"DXF 版本已升级到 {target_version}，共复制 {entity_count} 个实体，保存至 {output_file}")
    return True

def is_line_intersect_bbox(x1, y1, x2, y2, bbox):
    """判断线段 (x1, y1, x2, y2) 是否与 bbox 相交"""
    line = LineString([(x1, y1), (x2, y2)])
    text_bbox = box(*bbox)  # 创建矩形
    return line.intersects(text_bbox)  # 是否有交集

def filter_text_by_textbbox(merged_lines, text_data):
    """使用 R-tree 加速过滤文本包围盒范围内的中心线"""
    text_index = index.Index()
    text_bboxes = {}

    words, page_height = text_data   
    # 添加文本（优先使用行级文本）
    for i, (text, x, y, width, height) in enumerate(words):
        bbox = (x, y, x + width, y + height)      
        
        text_index.insert(i, bbox)  # 只插入行级包围盒
        text_bboxes[i] = bbox

    # 过滤被文本覆盖的中心线
    filtered_centerlines = []
    for line in merged_lines:
        if len(line) == 4:
            x1, y1, x2, y2 = line
        elif len(line) == 2:
            (x1, y1), (x2, y2) = line
        else:
            raise ValueError(f"无效的线段格式: {line}")

        # 查询 R-tree，找出可能覆盖该中心线的文本包围盒
        possible_texts = list(text_index.intersection((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))))

        covered = False
        for i in possible_texts:
            bbox = text_bboxes[i]
            if is_line_intersect_bbox(x1, y1, x2, y2, bbox):  # 改进判定逻辑
                covered = True
                break
        if not covered:
            filtered_centerlines.append((x1, y1, x2, y2))
            
    return filtered_centerlines

    
def append_to_dxf(dxf_file, ridges, merged_lines, text_result):
    """
    将 Voronoi 图的边和文本追加到现有的 DXF 文件中
    :param dxf_file: 现有的 DXF 文件路径
    :param ridges: Voronoi ridges 列表（LineString 格式）
    :param merged_lines: 合并后的线段列表，格式为 [(x1, y1, x2, y2), ...] 或 [((x1, y1), (x2, y2)), ...]
    :param text_result: 文本结果列表，格式为 [(text, x0, y0, x1, y1), ...]
    """
    try:
        # 读取现有的 DXF 文件
        doc = ezdxf.readfile(dxf_file)
    except IOError:
        print(f"无法读取 DXF 文件: {dxf_file}")
        return

    msp = doc.modelspace()
    
    # 创建标准图层配置
    layers = {
        '脊线': {'color': 9, 'linetype': 'CONTINUOUS', 'lineweight': 0.15},        
        '中心线': {'color': 3, 'linetype': 'CENTER', 'lineweight': 0.30},
        '文本': {'color': 2, 'linetype': 'HIDDEN', 'lineweight': 0.15}
    }
    
    # 初始化图层
    for layer_name, props in layers.items():
        doc.layers.add(name=layer_name)
        layer = doc.layers.get(layer_name)
        layer.color = props['color']
        layer.linetype = props['linetype']
        layer.lineweight = props['lineweight']
    setup_text_styles(doc)
    
    add_ridges(msp, ridges)
    
    # 批量添加中心线，减少 API 调用
    centerlines = []
    for line in merged_lines:
        if len(line) == 4:  # (x1, y1, x2, y2)
            x1, y1, x2, y2 = line
        elif len(line) == 2:  # ((x1, y1), (x2, y2))
            (x1, y1), (x2, y2) = line
        else:
            raise ValueError(f"无效的线段格式: {line}")
        msp.add_line(start=(x1, y1), end=(x2, y2), dxfattribs={"color": 3, 'layer': '中心线'})
   
    # 批量添加文本    
    # for text, x, y, width, height, rotation in text_data:
    #     if height > 0:            
    #         add_text(msp, text, (x, y), width, height, rotation, layer='文本')     
    words, page_height = text_result
    if page_height is not None:
        seen_lines = set()  # 用于去重

    for text, x, y, width, height in words:  
        if height <= 0:
            continue      
        add_text(msp, text, (x, y), height, 0, layer='文本')  

    # 保存修改后的 DXF 文件
    doc.saveas(dxf_file)  
        
# 使用 Potrace 转换 PBM 为 dxf
def convert_pbm_to_dxf(pbm_path, dxf_path):   
    # 执行 potrace 命令  
    # 定义 Potrace 参数
    params = [
        'D:/Image2CADPy/Image2CAD/potrace',
        pbm_path,
        '-b', 'dxf',
        '-o', dxf_path,
        '-z', 'majority',       # 保持主方向特征
        '-t', '5',              # 严格保留细节
        '-a', '0.15',           # 接近直线模式
        '-O', '0.5',            # 高精度优化    
        '-u', '10',             # 输出量化单位      
        '-n',
    ]
    
    # 执行 Potrace 命令
    try:
        subprocess.run(params, check=True)       
    except subprocess.CalledProcessError as e:
        print(f"Error during Potrace execution: {e}")
    except FileNotFoundError:
        print("Potrace executable not found. Please check the path.")

def preprocess_image(image_path):
    # 读取灰度图
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # OTSU 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学操作去除线条（适用于去除引线标注）
    kernel = np.ones((1, 5), np.uint8)  # 细长核更适用于引线
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary

def get_text_hocr(input_path, output_path):
    cmd = [
        'E:/Program Files/Tesseract-OCR/tesseract.exe',
        input_path,
        output_path,
        '-c', 'tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.+-±:/°"⌀ ',
        '-l', 'chi_sim+chi_tra',  # 语言设置：简体中文 + 繁体中文
        '--psm', '11',  # 改为PSM 11（稀疏文本自动方向）       
        '-c', 'tessedit_create_hocr=1',
        '-c', 'preserve_interword_spaces=1',  # 保持方向校正后的空格
        '--oem', '1'  # 使用LSTM引擎
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )
    
    # 检查输出日志中的方向信息
    if 'Orientation:' in result.stderr:
        print("检测到文本方向调整")
    
    return result
    
def get_text_with_rotation(input_path, conf_threshold=50):
    # 设置 Tesseract-OCR 路径
    pytesseract.pytesseract.tesseract_cmd = r"E:/Program Files/Tesseract-OCR/tesseract.exe"

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

def adjust_hocr_coordinates(hocr_data, original_shape, rotated_shape, angle):
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


def parse_hocr_optimized(hocr_file, min_confidence=70):
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
            x, y, width, height = convert_to_dxf_coords(x0, y0, x1, y1, page_height)

            # 记录文本信息
            text_positions.append((text, x, y, width, height))

        except Exception as e:
            print(f"解析行失败：{e}")

    return text_positions, page_height


def convert_to_dxf_coords(x0, y0, x1, y1, page_height):
    """
    根据页面高度，将 HOCR 中的 bbox 坐标转换为 DXF 坐标系。   
    """   
    width = x1 - x0
    height = y1 - y0
    x_new, y_new = x0, page_height - (y0 + height)      
    return x_new, y_new, width, height

def convert_to_multipolygon(polygons):
    """
    将多边形坐标列表转换为 MultiPolygon。
    :param polygons: 多边形坐标列表，每个多边形为 [(x1, y1), (x2, y2), ...] 的格式。
    :return: shapely.geometry.MultiPolygon 对象。
    """
    valid_polygons = []
    for coords in polygons:
        try:
            polygon = Polygon(coords)
            if polygon.is_valid:
                valid_polygons.append(polygon)
            # else:
            #     print(f"Invalid polygon skipped")
        except Exception as e:
            print(f"Error processing polygon: Error: {e}")
    
    return MultiPolygon(valid_polygons)

# 获取路径的边界框（最小矩形框）
def get_path_bbox(path_data):
    # 这里只是一个简化的版本，实际的路径解析可能更复杂
    # 假设路径数据是一个简单的矩形或直线
    # 这里暂时返回一个假设的边界框
    return (0, 0, 100, 100)  # 示例返回一个固定边界框

# 判断路径边界框和文本边界框是否重叠
def path_bbox_overlaps(path_bbox, text_bbox):
    px1, py1, px2, py2 = path_bbox
    tx1, ty1, tx2, ty2 = text_bbox
    return not (px2 < tx1 or px1 > tx2 or py2 < ty1 or py1 > ty2)

# 从文本块的位置信息中移除与文本重叠的路径
def remove_paths_in_text_area(svg_file, text_positions):
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # 查找所有路径
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}
    paths = root.findall('.//svg:path', namespaces=namespaces)

    for path in paths:
        path_coords = path.attrib.get('d')  # 获取路径数据
        path_bbox = get_path_bbox(path_coords)  # 获取路径的边界框

        # 遍历文本区域，检查路径是否与文本区域重叠
        for text, tx, ty, tw, th in text_positions:
            text_bbox = (tx, ty, tx + tw, ty + th)
            if path_bbox_overlaps(path_bbox, text_bbox):
                root.remove(path)  # 删除重叠的路径

    tree.write('output_no_text_paths.svg')  # 保存修改后的 SVG 
 
def process_page(page, page_num, output_dir, dpi=150):
    """ 处理单个 PDF 页面并保存为 PNG """
    try:
        pix = page.get_pixmap(dpi=dpi)
        image_path = os.path.join(output_dir, f"output_page_{page_num + 1}.png")
        pix.save(image_path)
        with print_lock:
            print(f"Saved: {image_path}")    
    except Exception as e:
        with print_lock:
            print(f"处理第 {page_num + 1} 页时出错：{e}")
        
def pdf_to_images(pdf_path, output_dir=None, dpi=150, max_workers=4):
    """ 并行转换 PDF 为图像 """
    pdf_Dir = os.path.abspath(pdf_path)
    dir = os.path.dirname(pdf_Dir)

    if output_dir is None:
        output_dir = os.path.join(dir, "pdfImages")

    os.makedirs(output_dir, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_page, doc[page_num], page_num, output_dir, dpi)
                       for page_num in range(len(doc))]

            # 等待所有任务完成
            for future in futures:
                future.result()

    except Exception as e:
        print(f"转换 PDF 为图像时出错：{e}")
        
def png_to_svg(input_path, output_folder=None):
    """
    将 PNG 文件或文件夹中的 PNG 转换为 DXF 格式，经过 PBM 格式的中间步骤。
    如果输入是文件夹，则遍历其中的 PNG 文件处理；
    如果输入是文件，则只处理该文件。
    """
    # 如果输入是文件夹
    if os.path.isdir(input_path):
        # 如果没有指定输出文件夹，则默认创建 output_svg 文件夹
        if output_folder is None:
            output_folder = os.path.join(input_path, "output_svg")
        
        # 如果输出文件夹不存在，则创建它
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 遍历文件夹中的所有 PNG 文件
        process_files_in_parallel(input_path, output_folder)             
    # 如果输入是文件
    elif os.path.isfile(input_path) and input_path.endswith(".png"):
        # 如果没有指定输出文件夹，则使用输入文件的目录
        if output_folder is None:
            output_folder = os.path.dirname(input_path)
        
        # 确保输出文件夹存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 处理单个文件
        process_single_file(input_path, output_folder)
    else:
        raise ValueError(f"Invalid input path: {input_path}. Must be a .png file or a folder containing .png files.")
    
def process_files_in_parallel(input_path, output_folder, max_processes=None):
    # 获取所有文件
    filenames = [filename for filename in os.listdir(input_path) if filename.endswith(".png")]
    
    # 如果没有指定最大并发数，则使用默认值（CPU 核心数的一半）
    if max_processes is None:
        max_processes = max(1, cpu_count() // 2)
    
    # 创建进程池，限制并发数
    with Pool(processes=max_processes) as pool:
        # 使用 starmap 函数并行处理每个文件
        pool.starmap(process_single_file, [(os.path.join(input_path, filename), output_folder) for filename in filenames])
 
def repair_multipolygon(multi_poly, max_processes=None):
    """修复 MultiPolygon 几何体，使用并行处理每个子多边形"""
    valid_polys = []
    
    if max_processes is None:
        max_processes = max(1, cpu_count() // 2)
    # 使用线程池并行处理每个子多边形
    with ThreadPoolExecutor(max_processes) as executor:
        # 提交所有任务
        future_to_poly = {executor.submit(repair_single_polygon, poly): poly for poly in multi_poly.geoms}      
        # 等待任务完成
        for future in as_completed(future_to_poly):
            try:
                repaired = future.result()
                if repaired.is_valid and not repaired.is_empty:
                    valid_polys.append(repaired)
            except Exception as e:
                print(f"无法修复子多边形：{str(e)}")
    
    # 返回有效的 MultiPolygon
    # 如果需要进一步过滤确保返回的是 Polygon 类型：
    return MultiPolygon([p for p in valid_polys if isinstance(p, Polygon) and p.is_valid])


def repair_single_polygon(poly):
    """修复单个多边形"""
    # 坐标对齐
    snapped = snap_to_grid(poly, 0.01)
    
    # 缓冲修复
    buffered = snapped.buffer(0.01).buffer(-0.02).buffer(0.01)
    
    # 有效性检查
    if not buffered.is_valid:
        raise Exception("子多边形修复失败")
    
    return buffered


def snap_to_grid(geom, precision=0.01):
    """处理多维坐标的网格对齐"""
    def _round_coord(coord):
        # 处理任意维度坐标 (x, y, [z, ...])
        return tuple(round(c / precision) * precision for c in coord[:2])  # 只处理前两个维度
    
    if geom.geom_type == 'Polygon':
        # 处理外环
        ext_rounded = [_round_coord(c) for c in geom.exterior.coords]
        
        # 处理内环
        int_rounded = [
            [_round_coord(c) for c in interior.coords]
            for interior in geom.interiors
        ]
        
        return Polygon(ext_rounded, int_rounded)
    else:
        return geom


def process_geometry_for_centerline(simplified_polygon):
    """修复并生成Centerline"""
    try:
        if isinstance(simplified_polygon, MultiPolygon):
            # 如果是 MultiPolygon，修复并确保返回有效的 MultiPolygon
            repaired_geom = repair_multipolygon(simplified_polygon)
            return Centerline(repaired_geom, 3) 
        else:
            # 如果是 Polygon，直接处理
            repaired_geom = repair_single_polygon(simplified_polygon)
            return Centerline(repaired_geom, 3)
    except Exception as e:
        print(f"Error calculating centerlines: {e}")
        return 


       
def process_single_file(input_path, output_folder):
    """
    处理单个 PNG 文件：将其转换为 PBM，再转换为 DXF。
    """
    os.makedirs(output_folder, exist_ok=True)
    start_time = time.time()
    # 生成 PBM 文件路径
    filename = os.path.basename(input_path)
    pbm_filename = os.path.splitext(filename)[0] + ".pbm"
    output_pbm_path = os.path.join(output_folder, pbm_filename)  
    
    hocr_filename = os.path.splitext(filename)[0] 
    output_hocrPath = os.path.join(output_folder, hocr_filename)          
    get_text_hocr(input_path, output_hocrPath)
    end_time = time.time()  
    print(f"get_text_hocr Execution time: {end_time - start_time:.2f} seconds")
    start_time = end_time
    # 调用函数将 PNG 转换为 PBM
    convert_png_to_pbm(input_path, output_pbm_path)
    end_time = time.time()  
    print(f"convert_png_to_pbm Execution time: {end_time - start_time:.2f} seconds")
    start_time = end_time
    
    # 生成 DXF 文件路径
    dxf_filename = os.path.splitext(filename)[0] + ".dxf"
    output_dxf_path = os.path.join(output_folder, dxf_filename)
    convert_pbm_to_dxf(output_pbm_path, output_dxf_path)
    end_time = time.time()  
    print(f"convert_pbm_to_dxf Execution time: {end_time - start_time:.2f} seconds")
    start_time = end_time
    
    # 提取多边形
    polygons = extract_polygons_from_dxf(output_dxf_path)
    end_time = time.time()  
    print(f"extract_polygons_from_dxf Execution time: {end_time - start_time:.2f} seconds")
    start_time = end_time
  
    max_workers = max(1, os.cpu_count() // 2)
    with ThreadPoolExecutor(max_workers) as executor:
        multi_polygon = convert_to_multipolygon(polygons)      
         # 简化 multi_polygon
        simplified_polygon = multi_polygon.simplify(tolerance=0.1, preserve_topology=True)  
    end_time = time.time()  
    print(f"convert_to_multipolygon Execution time: {end_time - start_time:.2f} seconds")
    start_time = end_time
    try:
        centerlines = process_geometry_for_centerline(simplified_polygon)
        if not centerlines:
            print(f"Error calculating centerlines: is Empty")
            return 
    except Exception as e:
        print(f"Error calculating centerlines: {e}")
        return
    end_time = time.time()  
    print(f"Centerline Execution time: {end_time - start_time:.2f} seconds")
    start_time = end_time
    # longest_paths = process_multilinestring(centerlines.geometry)
   
    # centerlines = get_centerline(multi_polygon)       
    merged_lines = merge_lines_with_hough(centerlines.geometry, 0) 
    end_time = time.time()  
    print(f"merge_lines_with_hough Execution time: {end_time - start_time:.2f} seconds")
    start_time = end_time
    
    # text_data = get_text_with_rotation(input_path)
    # end_time = time.time()  
    # print(f"OCR get text Execution time: {end_time - start_time:.2f} seconds")
    # start_time = end_time     
    
    # 追加到原始 DXF 并保存
    newdxf_filename = os.path.splitext(filename)[0] + "_newPy.dxf"    
    output_newdxf_path = os.path.join(output_folder, newdxf_filename)           
    shutil.copy2(output_dxf_path, output_newdxf_path)   
    
    text_positions = parse_hocr_optimized(output_hocrPath + ".hocr")
    filtered_lines = filter_text_by_textbbox(merged_lines, text_positions)
    append_to_dxf(output_newdxf_path, [], filtered_lines, text_positions)
    end_time = time.time()  
    print(f"append dxf Execution time: {end_time - start_time:.2f} seconds")
    start_time = end_time

    print(f"Voronoi ridges have been added to {output_newdxf_path}.")
            
    # os.remove(output_pbmPath)
    
    # text_positions = parse_hocr_optimized(output_hocrPath + ".hocr")
    # svgText_fileName = os.path.splitext(filename)[0] + "_text.svg"
    # output_svgTextPath = os.path.join(output_folder, svgText_fileName) 
    # insert_text_into_svg(output_svgPath, text_positions, output_svgTextPath)    


def main():    
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Process PDF and PNG files.")
    parser.add_argument('action', choices=['pdf2images', 'png2svg'], help="Choose the action to perform: 'pdf2images' or 'png2svg'")
    parser.add_argument('input_path', help="Input file or folder path.")
    parser.add_argument('output_path', nargs='?', help="Output file or folder path. If not provided, it will be auto-generated based on input_path.")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据选择的 action 执行相应的函数
    if args.action == 'pdf2images':
        pdf_to_images(args.input_path, args.output_path, 200, 4)
    elif args.action == 'png2svg':
        png_to_svg(args.input_path, args.output_path)

if __name__ == "__main__":
    main()

