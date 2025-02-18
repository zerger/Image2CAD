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
from scipy.spatial import Voronoi
import numpy as np
import ezdxf
from ezdxf import units
from ezdxf.enums import TextEntityAlignment
from Centerline.geometry import Centerline
from shapely.geometry import Polygon, MultiPolygon, MultiLineString, LineString
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import time
import os
import re
import tempfile
import threading
     
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

    # 添加 Voronoi 图的边
    if isinstance(ridges, LineString):
        msp.add_line(start=ridges.coords[0], end=ridges.coords[-1], dxfattribs={"color": 1})
    elif isinstance(ridges, MultiLineString):
        for line in ridges.geoms:
            for start, end in zip(line.coords[:-1], line.coords[1:]):
                msp.add_line(start=start, end=end, dxfattribs={"color": 1})
    elif isinstance(ridges, list):  # 如果 ridges 是一个列表
        for child in ridges:
            if isinstance(child, LineString):  # 如果是 LineString 类型
                msp.add_line(start=child.coords[0], end=child.coords[-1], dxfattribs={"color": 1})
            elif isinstance(child, MultiLineString):  # 如果是 MultiLineString 类型
                for line in child.geoms:
                    for start, end in zip(line.coords[:-1], line.coords[1:]):
                        msp.add_line(start=start, end=end, dxfattribs={"color": 1})
            else:
                for contour in ridges:
                    edge_coords = []
                    for pt in contour:
                        edge_coords.append(pt)
                    if edge_coords:
                        msp.add_lwpolyline(edge_coords, close=False, dxfattribs={"color": 1})

    # 添加合并后的线段
    for line in merged_lines:
        if len(line) == 4:  # 如果 line 是 (x1, y1, x2, y2) 格式
            x1, y1, x2, y2 = line
        elif len(line) == 2:  # 如果 line 是 ((x1, y1), (x2, y2)) 格式
            (x1, y1), (x2, y2) = line
        else:
            raise ValueError(f"无效的线段格式: {line}")
        msp.add_line(start=(x1, y1), end=(x2, y2), dxfattribs={"color": 2})

    words, page_height = text_result
    if page_height is not None:
        # 添加文本
        for text, x0, y0, x1, y1 in words:
            center_x, center_y, height = convert_to_dxf_coords(x0, y0, x1, y1, page_height)
            if height <= 0:
                continue
            msp.add_text(text, dxfattribs={'height': height, 'color': 5}).set_placement(
                (center_x, center_y), align=TextEntityAlignment.MIDDLE_CENTER
            )

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
        print(f"DXF file successfully generated")
    except subprocess.CalledProcessError as e:
        print(f"Error during Potrace execution: {e}")
    except FileNotFoundError:
        print("Potrace executable not found. Please check the path.")
    
def get_text_hocr(input_path, output_path):
    cmd = [
        'E:/Program Files/Tesseract-OCR/tesseract.exe',
        input_path,
        output_path,
        '-c', 'tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
        '--psm', '11',  # 改为PSM 11（稀疏文本自动方向）
        '-c', 'textord_orientation_fix=1',  # 启用方向修正
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
    
# 解析 hocr 文件，提取文本块的位置
def parse_hocr_optimized(hocr_file):
    """
    解析 hOCR 文件，提取每个单词的文本和 bbox 坐标，
    同时从 ocr_page 元素中提取页面高度（依据 bbox）和 DPI 信息（可扩展）。
    
    返回：
      text_positions: 每个元素为 (text, x0, y0, x1, y1)
      page_height: 页面高度（像素），用于坐标转换
    """
    with open(hocr_file, 'r', encoding='utf-8') as file:
        hocr_content = file.read()

    text_positions = []
    page_height = None  # 页面高度：将从 ocr_page 的 bbox 中提取
    try:
        soup = BeautifulSoup(hocr_content, 'html.parser')
        
        # 从 ocr_page 标签中解析页面 bbox 信息
        ocr_page = soup.find('div', class_='ocr_page')
        if ocr_page:
            page_title = ocr_page.get('title', '')
            # 例如: 'image "D:/..."; bbox 0 0 377 262; ppageno 0; scan_res 96 96'
            m_bbox = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', page_title)
            if m_bbox:
                page_x0, page_y0, page_x1, page_y1 = map(int, m_bbox.groups())
                page_height = page_y1 - page_y0  # 例如 262 - 0 = 262

        # 遍历所有包含文本的 ocrx_word 标签
        for word_element in soup.find_all('span', class_='ocrx_word'):
            coords = word_element.get('title', '')
            text = word_element.text.strip() if word_element.text else ''
            if coords and 'bbox' in coords:
                # 提取 bbox 坐标部分，如 "bbox 124 18 142 31; ..."
                bbox_part = coords.split('bbox ')[1].split(';')[0]
                bbox_coords = bbox_part.split()[:4]
                x0, y0, x1, y1 = [int(i) for i in bbox_coords]
                text_positions.append((text, x0, y0, x1, y1))
    except Exception as e:
        print(f"Error parsing hOCR file: {e}")

    return text_positions, page_height

def convert_to_dxf_coords(x0, y0, x1, y1, page_height):
    """
    根据页面高度，将 HOCR 中的 bbox 坐标转换为 DXF 坐标系。
    
    计算：
      - 中心坐标：center_x, center_y
      - 文字高度：height = y1 - y0（这里单位仍为像素，后续可能需要换算为工程单位）
    
    对 Y 坐标转换公式： center_y = page_height - ((y0 + y1) / 2)
    """
    center_x = (x0 + x1) / 2
    center_y_image = (y0 + y1) / 2
    center_y = page_height - center_y_image  # 翻转 Y 轴
    height = y1 - y0
    return center_x, center_y, height

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

    # 计算边界框
    # bounds = calculate_bounds(polygons)

    # 生成 Voronoi 图
    # ridges = perform_voronoi_analysis(polygons)
    
     # 初始化提取器（设置最大间距为 2.5）
    # extractor = OptimizedRidgeExtractor(polygons, max_distance=10)
    attributes = {"id": 1, "name": "polygon", "valid": True}
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

    # 追加到原始 DXF 并保存
    newdxf_filename = os.path.splitext(filename)[0] + "_newPy.dxf"
    output_newdxf_path = os.path.join(output_folder, newdxf_filename)
    # processed = process_ridges(centerlines.geometry, 0.1, 0.5, 0.1)
    # processed = merge_with_dbscan(centerlines.geometry)
    # ocr_text_result = get_text(input_path)
    shutil.copy2(output_dxf_path, output_newdxf_path)
    text_positions = parse_hocr_optimized(output_hocrPath + ".hocr")
    append_to_dxf(output_newdxf_path, [], merged_lines, text_positions)
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

