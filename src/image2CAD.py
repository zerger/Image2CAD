# -*- coding: utf-8 -*-
import os
import pymupdf
import argparse
import subprocess
import cv2
from bs4 import BeautifulSoup
import numpy as np
from scipy.spatial import Voronoi
from shapely.ops import unary_union
from shapely.geometry import LineString, box
from shapely.prepared import prep
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
from PIL import Image, ImageEnhance, ImageOps
from func_timeout import func_timeout, FunctionTimedOut
import shutil
import time
import sys
import threading
from Centerline.geometry import Centerline
from ocrProcess import OCRProcess
from dxfProcess import dxfProcess
from configManager import ConfigManager
from errors import ProcessingError, InputError, ResourceError, TimeoutError
from util import Util
from logManager import LogManager, setup_logging
from train_shx import TrainSHX_data
from shapely.validation import make_valid
import tqdm    

 
print_lock = threading.Lock()  
log_mgr = LogManager().get_instance()
config_manager = ConfigManager.get_instance()
allow_imgExt = ConfigManager.get_allow_imgExt()

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

@retry(max_attempts=3, delay=2)
def convert_pbm_with_retry(pbm_path: str, dxf_path: str) -> None:
    """带重试的PBM转换"""
    try:
        func_timeout(300, convert_pbm_to_dxf, args=(pbm_path, dxf_path))
    except FunctionTimedOut as e:
        log_mgr.log_error("DXF转换超时")
        raise TimeoutError("转换操作超时") from e
  
def simplify_preserve_topology(line, tolerance):
    """保持拓扑结构的简化"""
    prepared = prep(line)
    simplified = line.simplify(tolerance)
    
    # 确保简化后的线段与原始图形相交
    if prepared.intersects(simplified):
        return simplified
    return line  # 拓扑改变时返回原线
   
def parallel_simplify(lines, tolerance=0.5):
    """并行简化线段"""
    with ThreadPoolExecutor() as executor:
        futures = []
        for line in lines.geoms:
            futures.append(executor.submit(
                simplify_preserve_topology, 
                line, 
                tolerance
            ))
        
    return MultiLineString([f.result() for f in as_completed(futures)])

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
    dxfProcess.setup_dxf_logging()
    try:
        # === 阶段1：输入验证 ===
        log_mgr.log_info(f"开始处理文件: {input_path}")
        Util.validate_image_file(input_path)
        Util.check_system_resources()
        
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
        log_mgr.log_info("执行OCR处理...")
        ocr_process = OCRProcess()
        # ocr_process.verify_chinese_recognition()  
        # ocr_process.get_ocr_result_paddle(input_path)
        text_positions = ocr_process.get_ocr_result_tesseract(input_path, output_folder, min_confidence=70, max_height_diff=10)
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
        polygons = None
        # polygons = dxfProcess.extract_polygons_from_dxf(str(output_dxf), show_progress=True)
        log_mgr.log_processing_time("多边形提取", start_time)
        start_time = time.time()     
        
         # === 阶段5：生成中心线 ===
        log_mgr.log_info("生成中心线...")
        centerline = None
        with ThreadPoolExecutor() as executor:
            multi_polygon = convert_to_multipolygon(polygons)
            if multi_polygon:                   
                # filtered_multiPolygon = filter_polygons_by_textbbox(multi_polygon, text_positions, buffer_ratio=0.15, show_progress=True)   
                inter_dist = float(config_manager.get_setting('interpolation_distance', fallback=3))
                # 确保输入几何体有效
                if not multi_polygon.is_valid:
                    multi_polygon = make_valid(multi_polygon)
                
                # 使用buffer(0)修复可能的拓扑问题
                multi_polygon = multi_polygon.buffer(0)
                
                centerline = Centerline(
                        multi_polygon,
                        interpolation_distance=inter_dist,
                        simplify_tolerance=0.5,                       
                        use_multiprocessing=True,
                        show_progress=True
                        )            
                log_mgr.log_processing_time("中心线生成", start_time)
                start_time = time.time()   
        
        # === 阶段6：中心线分析 ===       
        log_mgr.log_info("中心线分析...")        
        try:
            # 确保使用centerline的geometry属性
            if hasattr(centerline, 'geometry'):
                merged_lines = merge_lines_with_hough(centerline.geometry, 0)
            else:
                merged_lines = merge_lines_with_hough(centerline, 0)
        except Exception as e:
            print(f"合并线段时出错: {e}")
            # 回退到直接使用centerline对象
            if hasattr(centerline, 'geoms'):
                # 如果centerline是几何集合，直接提取坐标
                merged_lines = []
                for geom in centerline.geoms:
                    if hasattr(geom, 'coords'):
                        coords = list(geom.coords)
                        if len(coords) >= 2:
                            merged_lines.append(coords)
            else:
                print("无法处理centerline对象，跳过线段合并")
                merged_lines = []
        
        filtered_lines = filter_line_by_textbbox(merged_lines, text_positions)   
        log_mgr.log_processing_time("中心线分析", start_time)
        start_time = time.time()   
                 
        # === 阶段7：结果整合 ===
        log_mgr.log_info("输出结果...")
        final_output = Path(output_folder) / f"output_{base_name}.dxf"
        # shutil.copy2(output_dxf, final_output)
        # dxfProcess.upgrade_dxf(output_dxf, final_output, "R2010")         
        dxfProcess.save_to_dxf(str(final_output), filtered_lines, text_positions, input_path)
        log_mgr.log_processing_time("结果输出", start_time)
        start_time = time.time()
        
        log_mgr.log_info(f"成功处理文件: {input_path}")
        log_mgr.log_info(f"结果输出文件: {final_output}")
        
        return True, str(final_output)
            
    except InputError as e:
        log_mgr.log_exception(f"输入错误: {e}")
    except ResourceError as e:
        log_mgr.log_exception(f"系统资源错误: {e}")
        raise  # 向上传递严重错误
    except TimeoutError as e:
        log_mgr.log_exception(f"处理超时: {e}")
    except Exception as e:
        log_mgr.log_exception(f"未处理的异常发生: {e}")
    finally:        
        log_mgr.log_processing_time(f"{base_name} 结束", fn_start_time)
    
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
    
# 合并近似的线段
def merge_lines_with_hough(lines, padding=0):
    """
    使用霍夫变换合并近似的线段，并确保结果与原始线条对齐
    :param lines: 输入的线段（MultiLineString、LineString、Centerline对象或其他几何体）
    :param padding: 图像边界扩展（默认为 0）
    :return: 合并后的线段列表
    """
    # 处理None或空输入
    if lines is None:
        return []
    
    # 处理Centerline对象
    if hasattr(lines, 'geometry'):
        lines = lines.geometry
    
    # 确保输入是MultiLineString或LineString类型
    if isinstance(lines, LineString):
        lines = MultiLineString([lines])
    elif not isinstance(lines, MultiLineString):
        # 尝试从各种几何类型中提取LineString
        try:
            if hasattr(lines, 'geoms'):
                # 从GeometryCollection中提取LineString
                line_geoms = [g for g in lines.geoms if isinstance(g, LineString)]
                if line_geoms:
                    lines = MultiLineString(line_geoms)
                else:
                    # 尝试从每个几何体中提取坐标
                    coords_list = []
                    for geom in lines.geoms:
                        if hasattr(geom, 'coords'):
                            coords = list(geom.coords)
                            if len(coords) >= 2:
                                coords_list.append(coords)
                    
                    if coords_list:
                        # 创建新的LineString对象
                        line_geoms = [LineString(coords) for coords in coords_list]
                        lines = MultiLineString(line_geoms)
                    else:
                        print("警告: 无法从输入几何体提取有效的线段")
                        return []
            elif hasattr(lines, 'coords'):
                # 单个几何体，尝试提取坐标
                coords = list(lines.coords)
                if len(coords) >= 2:
                    lines = MultiLineString([LineString(coords)])
                else:
                    print("警告: 输入几何体不包含足够的坐标点")
                    return []
            else:
                # 尝试将输入转换为字符串并检查WKT格式
                try:
                    from shapely import wkt
                    lines_str = str(lines)
                    if lines_str.startswith(('LINESTRING', 'MULTILINESTRING')):
                        geom = wkt.loads(lines_str)
                        if isinstance(geom, LineString):
                            lines = MultiLineString([geom])
                        elif isinstance(geom, MultiLineString):
                            lines = geom
                        else:
                            raise ValueError("WKT解析结果不是LineString或MultiLineString")
                    else:
                        raise ValueError("输入不是有效的WKT格式")
                except Exception as e:
                    print(f"尝试WKT解析失败: {e}")
                    # 最后的尝试：检查是否有__geo_interface__属性
                    if hasattr(lines, '__geo_interface__'):
                        from shapely.geometry import shape
                        try:
                            geom = shape(lines.__geo_interface__)
                            if isinstance(geom, LineString):
                                lines = MultiLineString([geom])
                            elif isinstance(geom, MultiLineString):
                                lines = geom
                            else:
                                raise ValueError("__geo_interface__转换结果不是LineString或MultiLineString")
                        except Exception as e2:
                            print(f"尝试__geo_interface__转换失败: {e2}")
                            raise ValueError("输入无法转换为MultiLineString或LineString类型")
                    else:
                        # 打印更多调试信息
                        print(f"输入类型: {type(lines)}")
                        print(f"输入属性: {dir(lines)}")
                        raise ValueError("输入必须是MultiLineString或LineString类型，或可转换为这些类型的对象")
        except Exception as e:
            print(f"处理输入几何体时出错: {e}")
            # 尝试直接访问centerline对象的内部结构
            if hasattr(lines, '_centerline') and lines._centerline is not None:
                return merge_lines_with_hough(lines._centerline, padding)
            elif hasattr(lines, 'centerline') and lines.centerline is not None:
                return merge_lines_with_hough(lines.centerline, padding)
            else:
                print(f"无法处理的输入类型: {type(lines)}")
                return []
    
    # 如果是空的MultiLineString，直接返回空列表
    if len(lines.geoms) == 0:
        return []

    # 计算输入线段的边界范围
    min_x, min_y, max_x, max_y = lines.bounds

    # 动态调整图像大小并添加 padding
    width = int(max_x - min_x + 2 * padding) + 1  # 加1确保至少有1个像素
    height = int(max_y - min_y + 2 * padding) + 1
    
    # 确保图像尺寸合理
    if width <= 0 or height <= 0 or width > 10000 or height > 10000:
        print(f"警告: 图像尺寸异常 ({width}x{height})，使用默认尺寸")
        width = max(1, min(width, 10000))
        height = max(1, min(height, 10000))
    
    img = np.zeros((height, width), dtype=np.uint8)

    # 将实际坐标映射到图像坐标
    points = []
    for line in lines.geoms:
        coords = list(line.coords)
        if len(coords) < 2:
            continue  # 跳过无效线段
            
        for start, end in zip(coords[:-1], coords[1:]):
            try:
                start_mapped = (int(start[0] - min_x + padding), int(start[1] - min_y + padding))
                end_mapped = (int(end[0] - min_x + padding), int(end[1] - min_y + padding))
                
                # 确保坐标在图像范围内
                if (0 <= start_mapped[0] < width and 0 <= start_mapped[1] < height and
                    0 <= end_mapped[0] < width and 0 <= end_mapped[1] < height):
                    points.append([start_mapped[0], start_mapped[1], end_mapped[0], end_mapped[1]])
                    # 在图像上绘制线段
                    cv2.line(img, start_mapped, end_mapped, 255, 1)
            except Exception as e:
                print(f"绘制线段时出错: {e}")
    
    # 如果没有有效点，返回空列表
    if not points:
        return []
    
    try:
        # 使用霍夫变换检测线段
        lines_detected = cv2.HoughLinesP(
            img,
            rho=1,                          # 距离分辨率
            theta=np.pi / 180,              # 角度分辨率
            threshold=20,                   # 累加器阈值
            minLineLength=10,               # 线段的最小长度
            maxLineGap=20                   # 线段之间的最大间隔
        )
    except Exception as e:
        print(f"霍夫变换检测失败: {e}")
        return []

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

def filter_line_by_textbbox(merged_lines, text_data):
    """使用 R-tree 加速过滤文本包围盒范围内的中心线"""
    if merged_lines is None or not merged_lines:
        return []
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

def filter_polygons_by_textbbox(multi_polygon, text_data, buffer_ratio=0.1, show_progress=True) -> MultiPolygon:
    """
    过滤掉与文本区域重叠的多边形
    :param multi_polygon: 输入的多边形集合（MultiPolygon）
    :param text_data: OCR识别结果 (words, page_height)
    :param buffer_ratio: 文本区域扩展比例（基于原始尺寸）
    :param show_progress: 是否显示进度条
    :return: 过滤后的MultiPolygon
    """
    if not isinstance(multi_polygon, MultiPolygon):
        raise ValueError("输入必须是MultiPolygon类型")
        
    # 如果没有文本数据，直接返回原始多边形
    words, page_height = text_data
    if not words:
        return multi_polygon

    # 显示文本处理进度
    if show_progress:
        print(f"处理 {len(words)} 个文本区域...")
        
    # 创建所有文本区域的并集
    text_polygons = []
    
    # 使用tqdm显示文本处理进度
    text_iter = tqdm.tqdm(words, desc="创建文本区域", disable=not show_progress)
    for text, x, y, w, h in text_iter:
        # 计算动态缓冲尺寸（基于文本区域大小）
        buffer_size = max(w, h) * buffer_ratio
        
        # 创建文本边界框
        text_box = box(
            x - buffer_size,
            y - buffer_size,
            x + w + buffer_size,
            y + h + buffer_size
        )
        text_polygons.append(text_box)
    
    # 合并所有文本区域为一个几何体
    if not text_polygons:
        return multi_polygon
        
    try:
        if show_progress:
            print("合并文本区域...")
            
        # 使用unary_union合并所有文本区域
        text_union = unary_union(text_polygons)
        
        # 确保结果是有效的
        if not text_union.is_valid:
            if show_progress:
                print("修复文本区域几何体...")
            text_union = text_union.buffer(0)
            
        # 准备输入多边形
        poly_geoms = list(multi_polygon.geoms)
        
        if show_progress:
            print(f"处理 {len(poly_geoms)} 个多边形...")
            
        filtered_polygons = []
        
        # 定义检查函数
        def check_polygon(polygon):
            try:
                # 检查多边形是否与文本区域相交
                if polygon.intersects(text_union):
                    # 如果相交，计算差集（从多边形中减去文本区域）
                    difference = polygon.difference(text_union)
                    
                    # 如果差集为空，则完全排除此多边形
                    if difference.is_empty:
                        return None
                        
                    # 如果差集是多边形或多多边形，返回它
                    if isinstance(difference, (Polygon, MultiPolygon)):
                        return difference
                    
                    # 处理GeometryCollection情况
                    if hasattr(difference, 'geoms'):
                        # 提取所有多边形
                        polygons = [g for g in difference.geoms if isinstance(g, Polygon)]
                        if polygons:
                            return MultiPolygon(polygons)
                        return None
                    
                    return None
                else:
                    # 如果不相交，保留原始多边形
                    return polygon
            except Exception as e:
                print(f"处理多边形时出错: {e}")
                # 出错时保留原始多边形
                return polygon
        
        # 并行处理所有多边形，带进度条
        with ThreadPoolExecutor() as executor:
            # 创建future列表
            futures = [executor.submit(check_polygon, poly) for poly in poly_geoms]
            
            # 使用tqdm显示处理进度
            results = []
            for f in tqdm.tqdm(as_completed(futures), total=len(futures), 
                              desc="过滤多边形", disable=not show_progress):
                results.append(f.result())
            
        # 过滤掉None结果并处理返回的几何体
        for result in results:
            if result is None:
                continue
                
            if isinstance(result, Polygon):
                filtered_polygons.append(result)
            elif isinstance(result, MultiPolygon):
                filtered_polygons.extend(list(result.geoms))
        
        if show_progress:
            print(f"过滤后剩余 {len(filtered_polygons)} 个多边形")
            
        # 如果没有剩余多边形，返回空的MultiPolygon
        if not filtered_polygons:
            return MultiPolygon([])
            
        # 返回过滤后的MultiPolygon
        return MultiPolygon(filtered_polygons)
        
    except Exception as e:
        print(f"过滤多边形时出错: {e}")
        # 出错时返回原始多边形
        return multi_polygon
        
# 使用 Potrace 转换 PBM 为 dxf
def convert_pbm_to_dxf(pbm_path, dxf_path):   
    # 执行 potrace 命令  
    # 定义 Potrace 参数
    potrace_path = config_manager.get_potrace_path()
    if not os.path.exists(potrace_path):
        raise FileNotFoundError(f"Potrace可执行文件未找到: {potrace_path}")
    if not os.access(potrace_path, os.X_OK):
        raise PermissionError(f"无执行权限: {potrace_path}")
    params = [
        potrace_path,
        pbm_path,
        '-b', 'dxf',
        '-o', dxf_path,
        '-z', 'majority',       # 追踪方式 (black|white|majority)
        '-t', '10',              # 拐角阈值 (1-100)，值越大线条越平滑
        '-a', '0.15',            # 拐角平滑度 (0-1.4)，值越大拐角越圆滑
        '-O', '1',              # 优化等级 (0-1)，值越大优化程度越高 
        '-u', '3',              # 输出单位 (DPI)  
        '-n',                   # 关闭曲线细分
    ]
    
    # 执行 Potrace 命令
    try:
        result = subprocess.run(
            params,
            check=True,  # 自动检查返回码
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        ) 
          # 二次验证输出文件
        if not os.path.exists(dxf_path):
            raise FileNotFoundError(f"DXF文件生成失败: {dxf_path}")
            
        return True 
    except subprocess.CalledProcessError as e:
        print(f"Error during Potrace execution: {e}")
    except FileNotFoundError:
        print("Potrace executable not found. Please check the path.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log_mgr.log_error(f"DXF转换失败: {str(e)}")
        # 清理不完整文件
        if os.path.exists(dxf_path):
            os.remove(dxf_path)
        return False  # 返回失败标识   

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
    if polygons is None:
        return MultiPolygon([])
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
 
def pdf_page_to_image(page, page_num, image_path, dpi, image_format='PNG'):
    """ 处理单个 PDF 页面并保存为 PNG """
    try:        
        if dpi > 1000:
            dpi = 1000
            print(f"dpi {dpi} 设置过高, 调整为 1000 ")           
        matrix = pymupdf.Matrix(dpi/72, dpi/72)  # 高DPI矩阵 
        pix = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csRGB, alpha=False)  # 使用灰度颜色空间          
        #pix.save(image_path)
        # 将图像转换为 PIL 图像
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)        
        # # 调整图像对比度和饱和度
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(5)  # 增强对比度
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.2)  # 增强饱和度
         # 调整图像亮度
        brightness_enhancer = ImageEnhance.Brightness(image)
        image = brightness_enhancer.enhance(1.5)  # 增强亮度
        gray_image = image.convert("L")        
        # 将 PIL 图像转换为 NumPy 数组
        gray_np = np.array(gray_image)
        # 使用 OpenCV 的自适应阈值
        binary_np = cv2.adaptiveThreshold(
            gray_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        # 将 NumPy 数组转换回 PIL 图像
        binary_image = Image.fromarray(binary_np)
         # 保存图像并设置 DPI
        binary_image.save(image_path, format=image_format.upper(), dpi=(dpi, dpi))
              
        with print_lock:
            print(f"Saved: {image_path}")    
    except Exception as e:
        with print_lock:
            print(f"处理第 {page_num + 1} 页时出错：{e}")
            
def pdf_page_to_svg(page, page_num, image_path):
    try: 
        # 创建一个矩阵用于缩放和旋转
        matrix = pymupdf.Matrix(1, 1)  # 1, 1 表示不缩放    
        # 渲染页面为 SVG
        svg = page.get_svg_image(matrix=matrix)    
        # 将 SVG 内容写入文件
        with open(image_path, 'w', encoding='utf-8') as f:
            f.write(svg)

        print(f"Page {page_num + 1} has been saved as {image_path}")
    except Exception as e:
        with print_lock:
            print(f"处理第 {page_num + 1} 页时出错：{e}")
            
def _process_pdf_page(page, page_num, output_dir, output_type='png', dpi=None):
    """ 处理单个 PDF 页面并根据类型选择保存为 PNG 或 SVG """
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)    
    image_path = os.path.join(output_dir, f"pdf_page_{page_num + 1}.{output_type}")
    if output_type.lower() == 'png':
        # 保存为 PNG       
        pdf_page_to_image(page, page_num, image_path, dpi, 'PNG')
    elif output_type.lower() == 'tiff':
        pdf_page_to_image(page, page_num, image_path, dpi, 'TIFF')
    elif output_type.lower() == 'svg':
        # 保存为 SVG       
        pdf_page_to_svg(page, page_num, image_path)
    else:
        print(f"Unsupported output type: {output_type}")
                
def pdf_to_images(pdf_path, output_dir=None, output_type='png', dpi=None):
    """ 并行转换 PDF 为图像 """
    try:
        import pymupdf  # 延迟导入减少依赖
    except ImportError:
        raise RuntimeError("PDF处理需要PyMuPDF: pip install pymupdf")
    pdf_Dir = os.path.abspath(pdf_path)
    dir = os.path.dirname(pdf_Dir)

    if output_dir is None:      
        output_dir = Util.default_output_path(pdf_path, 'pdf_images')

     # 从配置获取参数 
    config_manager.apply_security_settings()    
    if output_dir is None:
        output_dir = config_manager.get_setting(key='pdf_output_dir', fallback='./pdf_images')
    if dpi is None:
        dpi = int(config_manager.get_setting(key='pdf_export_dpi', fallback=200))
    max_workers = int(config_manager.get_setting(key='max_workers', fallback=os.cpu_count()//2))
    os.makedirs(output_dir, exist_ok=True)
    
    # 显示当前配置参数
    log_mgr.log_info("\n当前PDF转换参数：")
    log_mgr.log_info(f"├─ 输出目录：{os.path.abspath(output_dir)}")
    log_mgr.log_info(f"├─ 解析精度：{dpi} DPI")
    log_mgr.log_info(f"├─ 输出类型：{output_type}")

    try:
        doc = pymupdf.open(pdf_path)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_pdf_page, doc[page_num], page_num, output_dir, output_type, dpi)
                       for page_num in range(len(doc))]

            # 等待所有任务完成
            for future in as_completed(futures):
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
            output_folder = os.path.join(input_path, "cad_output")
        
        # 如果输出文件夹不存在，则创建它
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 遍历文件夹中的所有 PNG 文件
        process_files_in_parallel(input_path, output_folder)             
    # 如果输入是文件
    elif os.path.isfile(input_path) and Util.has_valid_files(input_path, allow_imgExt):
        # 如果没有指定输出文件夹，则使用输入文件的目录
        if output_folder is None:
            output_folder = Util.default_output_path(input_path, 'cad')
        
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


def process_geometry_for_centerline(simplified_polygon, interpolation_distance=0.5):
    """修复并生成Centerline"""
    try:
        if isinstance(simplified_polygon, MultiPolygon):
            # 如果是 MultiPolygon，修复并确保返回有效的 MultiPolygon
            repaired_geom = repair_multipolygon(simplified_polygon)
            return Centerline(repaired_geom, interpolation_distance) 
        else:
            # 如果是 Polygon，直接处理
            repaired_geom = repair_single_polygon(simplified_polygon)
            return Centerline(repaired_geom, interpolation_distance)
    except Exception as e:
        print(f"Error calculating centerlines: {e}")
        return  

def main():    
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""Image2CAD 工程图转换工具
    示例用法:
    转换PDF为图片: python %(prog)s pdf2images input.pdf output_images/
    单PNG转DXF   : python %(prog)s png2dxf input.png --dpi 200 --output output.dxf
    批量转换      : python %(prog)s png2dxf input_folder/ --workers 8"""
    )
    
    # 主命令参数
    parser.add_argument('action', 
                        choices=['pdf2images', 'png2dxf'],
                        help="""操作选项:
    pdf2images   : 将PDF转换为PNG图像
    png2dxf      : 转换PNG图像为CAD格式   
    """)
    
    # 通用参数
    parser.add_argument('input_path', nargs='?', 
                       help="输入文件/目录路径（对set操作为工具路径）")
    parser.add_argument('output_path', nargs='?',
                       help="输出路径（默认根据输入自动生成）")
    parser.add_argument('--config', default='config.ini',
                       help="指定配置文件路径（默认: ./config.ini）")
    
    # 转换参数组
    convert_group = parser.add_argument_group('转换参数')
    convert_group.add_argument('--dpi', type=int, default=200,
                              help="图像处理DPI（默认: 200）")
    convert_group.add_argument('--format', choices=['dxf', 'svg', 'dwg', 'png', 'tiff'], default='dxf',
                              help="输出格式（默认: dxf）")    
    convert_group.add_argument('--overwrite', action='store_true',
                              help="覆盖已存在文件")   
    
    try:
        args = parser.parse_args()
        setup_logging()  # 初始化日志
        action = args.action.lower()
        input_path = args.input_path
        # 参数验证
        if action in ['pdf2images', 'png2dxf'] and not input_path:
            raise InputError("必须指定输入路径")  
            
        # 加载配置文件
        config_manager.load_config(args.config)   
        # 根据选择的 action 执行
        if action == 'pdf2images':
            Util.validate_input_path(input_path, ['.pdf'])
            output_dir = args.output_path or Util.default_output_path(input_path, 'pdf_images')
            pdf_to_images(input_path, output_dir, args.format, args.dpi)
            
        elif action.lower() in {'png2dxf'}:
            validate_input_path(input_path, allow_imgExt)
            output_dir = args.output_path or Util.default_output_path(input_path, 'cad')
            if action == 'png2dxf':
                png_to_dxf(input_path, output_dir)        
        else:
            print("请输入正确的命令")
            
    except argparse.ArgumentError as e:
        log_mgr.log_error(f"参数错误: {e}")
        parser.print_help()
        sys.exit(1)
    except Exception as e:
        log_mgr.log_error(f"运行错误: {str(e)}")
        sys.exit(2)

if __name__ == "__main__":
    main()

