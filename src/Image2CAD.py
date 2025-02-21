# -*- coding: utf-8 -*-
import os
import fitz  # PyMuPDF
import argparse
import subprocess
import cv2
from bs4 import BeautifulSoup
import numpy as np
from scipy.spatial import Voronoi
from shapely.ops import unary_union
from shapely.geometry import LineString, box
from scipy.spatial import Voronoi
import xml.etree.ElementTree as ET
import numpy as np

from shapely.geometry import Polygon, MultiPolygon, MultiLineString, LineString
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from rtree import index
import os
import time
import tempfile
from functools import wraps
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from func_timeout import func_timeout, FunctionTimedOut
import shutil
import time
import os
import sys
import tempfile
import threading
import OCRProcess
from Centerline.geometry import Centerline
from OCRProcess import OCRProcess, config_manager
from dxfProcess import dxfProcess
from Error import ProcessingError, InputError, ResourceError, TimeoutError
from Util import Util
from LogManager import LogManager, setup_logging

     
print_lock = threading.Lock()   
log_mgr = LogManager()

def retry(max_attempts: int = 3, delay: int = 1):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts+1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        log_mgr.log_error(f"操作重试{max_attempts}次后失败")
                        raise
                    log_mgr.log_warn(f"重试 {attempt}/{max_attempts}，原因：{str(e)}")
                    time.sleep(delay * attempt)
        return wrapper
    return decorator

def validate_input_file(input_path: str) -> None:
    """验证输入文件有效性"""
    path = Path(input_path)
    if not path.exists():
        raise InputError(f"输入文件不存在: {input_path}")
    if not path.is_file():
        raise InputError(f"输入路径不是文件: {input_path}")
    if path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
        raise InputError(f"不支持的文件格式: {path.suffix}")
    if path.stat().st_size > 100 * 1024 * 1024:  # 100MB限制
        raise InputError("文件大小超过100MB限制")

def check_system_resources() -> None:
    """检查系统资源是否充足"""
    # 示例：检查磁盘空间
    free_space, _ = Util.get_disk_space_psutil('/')
    if free_space < 500 * 1024 * 1024:  # 500MB
        raise ResourceError("磁盘空间不足（需要至少500MB空闲空间）")

@retry(max_attempts=3, delay=2)
def convert_pbm_with_retry(pbm_path: str, dxf_path: str) -> None:
    """带重试的PBM转换"""
    try:
        func_timeout(300, convert_pbm_to_dxf, args=(pbm_path, dxf_path))
    except FunctionTimedOut as e:
        log_mgr.log_error("DXF转换超时")
        raise TimeoutError("转换操作超时") from e

def process_single_file(input_path: str, output_folder: str) -> Tuple[bool, Optional[str]]:
    """
    安全处理单个文件的全流程
    
    :param input_path: 输入文件路径
    :param output_folder: 输出目录
    :return: (是否成功, 输出文件路径)
    """
    fn_start_time = time.time()
    temp_files = []
    
    setup_logging(console=True)
    try:
        # === 阶段1：输入验证 ===
        log_mgr.log_info(f"开始处理文件: {input_path}")
        validate_input_file(input_path)
        check_system_resources()
        
        # === 阶段2：准备输出 ===
        os.makedirs(output_folder, exist_ok=True)
        if not os.access(output_folder, os.W_OK):
            raise PermissionError(f"输出目录不可写: {output_folder}")
            
        base_name = Path(input_path).stem
        output_dxf = Path(output_folder) / f"{base_name}.dxf"
        
        # === 阶段3：创建临时工作区 ===
        # with tempfile.TemporaryDirectory(prefix="img2cad_") as tmp_dir:
        start_time = time.time()
        # OCR处理
        hocr_path = Path(output_folder) / f"{base_name}_ocr"
        log_mgr.log_info("执行OCR处理...")
        ocr_process = OCRProcess()
        ocr_process.get_text_hocr(input_path, str(hocr_path))
        log_mgr.log_processing_time("OCR处理", start_time)
        start_time = time.time()
        
        # 转换PBM
        pbm_path = Path(output_folder) / f"{base_name}.pbm"
        log_mgr.log_info("转换图像格式...")
        convert_png_to_pbm(input_path, str(pbm_path))
        log_mgr.log_processing_time("图像格式转换", start_time)
        start_time = time.time()
        
        # 转换DXF（带重试和超时）
        log_mgr.log_info("转换DXF格式...")
        convert_pbm_with_retry(str(pbm_path), str(output_dxf))
        log_mgr.log_processing_time("DXF转换", start_time)
        start_time = time.time()
        
        # === 阶段4：后处理 ===
        log_mgr.log_info("提取多边形...")
        polygons = dxfProcess.extract_polygons_from_dxf(str(output_dxf))
        log_mgr.log_processing_time("多边形提取", start_time)
        start_time = time.time()
        
        log_mgr.log_info("生成中心线...")
        with ThreadPoolExecutor() as executor:
            multi_polygon = convert_to_multipolygon(polygons)
            multipolygon_to_txt(multi_polygon, filename=output_folder + "/output.txt")
            simplified = multi_polygon.simplify(tolerance=1)
            centerlines = process_geometry_for_centerline(simplified)
            merged_lines = merge_lines_with_hough(centerlines.geometry, 0) 
            log_mgr.log_processing_time("中心线生成", start_time)
            start_time = time.time()
            
            # === 阶段6：ocr整合 ===
        log_mgr.log_info("获取ocr结果...")
        text_positions = ocr_process.parse_hocr_optimized(str(hocr_path) + ".hocr")
        filtered_lines = filter_text_by_textbbox(merged_lines, text_positions)    
        log_mgr.log_processing_time("ocr结果获取", start_time)
        start_time = time.time()
        
        # === 阶段6：结果整合 ===
        log_mgr.log_info("输出结果...")
        final_output = Path(output_folder) / f"output_{base_name}.dxf"
        # shutil.copy2(output_dxf, final_output)
        dxfProcess.upgrade_dxf(output_dxf, final_output, "R2010")
        dxfProcess.append_to_dxf(str(final_output), simplified, [], filtered_lines, text_positions)
        log_mgr.log_processing_time("结果输出", start_time)
        start_time = time.time()
        
        log_mgr.log_info(f"成功处理文件: {input_path}")
        log_mgr.log_info(f"结果输出文件: {str(final_output)}")
        
        return True, str(final_output)
            
    except InputError as e:
        log_mgr.log_error(f"输入错误: {e}")
    except ResourceError as e:
        log_mgr.log_error(f"系统资源错误: {e}")
        raise  # 向上传递严重错误
    except TimeoutError as e:
        log_mgr.log_error(f"处理超时: {e}")
    except Exception as e:
        log_mgr.log_error(f"未处理的异常发生: {e}")
    finally:
        # 清理临时资源
        for f in temp_files:
            try:
                os.remove(f)
            except:
                pass
        log_mgr.log_processing_time("处理结束", fn_start_time)
    
    return False, None

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

def multipolygon_to_txt(multipolygon, filename="output.txt"):
    with open(filename, "w") as f:
        for i, polygon in enumerate(multipolygon.geoms):
            f.write(f"Polygon {i+1}:\n")  # 标注多边形编号
            
            # 处理外边界
            if polygon.exterior:
                f.write("  Exterior:\n")
                for coord in polygon.exterior.coords:
                    x, y = coord[:2]  # 兼容 2D 和 3D
                    f.write(f"    {x}, {y}\n")
            else:
                f.write("  Empty Polygon\n")

            # 处理内部孔洞
            for j, interior in enumerate(polygon.interiors):
                f.write(f"  Hole {j+1}:\n")
                for coord in interior.coords:
                    x, y = coord[:2]  # 兼容 2D 和 3D
                    f.write(f"    {x}, {y}\n")
            
            f.write("\n")  # 分隔多边形
    print(f"TXT 文件已保存为 {filename}")
    
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

        
# 使用 Potrace 转换 PBM 为 dxf
def convert_pbm_to_dxf(pbm_path, dxf_path):   
    # 执行 potrace 命令  
    # 定义 Potrace 参数
    params = [
        'D:/Image2CADPy/src/potrace',
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
        
def png_to_dxf(input_path, output_folder=None):
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

def main():    
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Process PDF and PNG files.")
    parser.add_argument('action', choices=['pdf2images', 'png2dxf', 'set-tesseract'],
                        help="Action: 'pdf2images', 'png2dxf' or 'set-tesseract'")
    parser.add_argument('input_path', nargs='?', 
                        help="Input path (file/folder) or tesseract path for set-tesseract")
    parser.add_argument('output_path', nargs='?',
                        help="Output path (auto-generated if omitted)")    
    # 添加可选参数
    parser.add_argument('--tesseract', '-t',
                        help="Specify Tesseract path (overrides config)")
    parser.add_argument('--log-file', help='指定日志文件路径')
    parser.add_argument('--no-console', action='store_true', help='禁用控制台输出')
    
    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(f"参数错误: {e}")
        parser.print_help()
        sys.exit(1)
    
    # 根据选择的 action 执行相应的函数
    if args.action == 'pdf2images':
        pdf_to_images(args.input_path, args.output_path, 200, 4)
    elif args.action == 'png2dxf':
        # 优先使用命令行参数
        tesseract_path = args.tesseract or config_manager.get_tesseract_path()
        try:
            config_manager.set_tesseract_path(tesseract_path)
        except FileNotFoundError as e:
            print(f"Tesseract路径无效: {e}")
            return
        png_to_dxf(args.input_path, args.output_path)
    elif args.action == 'set-tesseract':
        if not args.input_path:
            print("Error: Missing Tesseract path")
            return
        try:
            config_manager.set_tesseract_path(args.input_path)
            print(f"Tesseract路径已设置为: {config_manager.get_tesseract_path()}")
        except Exception as e:
            print(f"配置失败: {e}")
        return

if __name__ == "__main__":
    main()

