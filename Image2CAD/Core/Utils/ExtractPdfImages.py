import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from lxml import etree
import xml.etree.ElementTree as ET
from PIL import Image
import argparse
import subprocess
import cv2
import re
from bs4 import BeautifulSoup
import numpy as np
from ShowImage import ShowImage
from RidgeExtractor import OptimizedRidgeExtractor
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPoint, LineString
from shapely.ops import unary_union
from scipy.spatial import Voronoi
import numpy as np
import ezdxf
from shapely.geometry import Polygon
from centerline.geometry import Centerline
from shapely.geometry import Polygon, MultiPolygon, MultiLineString, LineString
      
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
    cv2.imwrite(pbm_path, binary_img)
    
def skeletonize(img):
   # 确保是二值图像
    ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    skeleton = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    temp_img = binary.copy()  # 使用副本进行处理
    while True:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        temp_img[:] = eroded
        if cv2.countNonZero(temp_img) == 0:
            break
    return skeleton

def voronoi_skeleton(polygon_points):
    # 计算 Voronoi 图
    vor = Voronoi(polygon_points)
    
    # 创建封闭多边形
    polygon = Polygon(polygon_points)
    skeleton_lines = []
    
    for edge in vor.ridge_vertices:
        if edge[0] != -1 and edge[1] != -1:  # 跳过无穷远点
            p1, p2 = vor.vertices[edge[0]], vor.vertices[edge[1]]
            line = LineString([p1, p2])
            if polygon.contains(line):  # 仅保留多边形内部的线段
                skeleton_lines.append(line)
    
    return skeleton_lines

def extract_polygons_from_dxf(file_path):
    """
    从 DXF 文件中提取多边形
    """
    doc = ezdxf.readfile(file_path)
    polygons = []
    for entity in doc.modelspace():
        if entity.dxftype() == "LWPOLYLINE":
            points = entity.get_points("xy")
            polygons.append(Polygon(points))
        elif entity.dxftype() == "POLYLINE":
            # 获取 POLYLINE 的顶点
            points = []
            for vertex in entity.vertices:
                if hasattr(vertex.dxf, "location"):  # 检查顶点是否包含坐标
                    points.append((vertex.dxf.location.x, vertex.dxf.location.y))
                else:
                    points.append((vertex.dxf.x, vertex.dxf.y))  # 兼容非标准格式
            polygons.append(Polygon(points))
    return polygons

def calculate_bounds(polygons, buffer=5):
    """
    根据多边形计算 Voronoi 图所需的边界框
    :param polygons: shapely.Polygon 列表
    :param buffer: 边界缓冲距离
    :return: [[x_min, x_max], [y_min, y_max]]
    """
    union_polygon = unary_union(polygons)
    minx, miny, maxx, maxy = union_polygon.bounds
    return [[minx - buffer, maxx + buffer], [miny - buffer, maxy + buffer]]


def perform_voronoi_analysis(polygons):
    """
    生成 Voronoi 图
    """
    # 获取所有顶点坐标
    points = [point for polygon in polygons for point in polygon.exterior.coords]
    points = list(set(points))  # 去重

    # 使用 Shapely 处理点
    multi_points = MultiPoint(points)
    vor = Voronoi([p.coords[0] for p in multi_points.geoms])

    # 提取 ridges
    ridges = []
    for ridge in vor.ridge_vertices:
        if -1 not in ridge:  # 过滤掉无限远的边
            start, end = vor.vertices[ridge[0]], vor.vertices[ridge[1]]
            ridges.append(LineString([start, end]))
    return ridges

def efficient_voronoi(polygons, bounds):
    """
    使用 pyvoro 生成 Voronoi 图
    :param polygons: shapely.Polygon 列表
    :param bounds: 边界框 [(x_min, x_max), (y_min, y_max)]
    :return: Voronoi ridges 列表
    """
    # 1. 提取所有点
    # points = [point for polygon in polygons for point in polygon.exterior.coords]
    # points = list(set(points))  # 去重

    # # 2. 将点转换为 pyvoro 格式
    # points = [{"x": p[0], "y": p[1], "z": 0.0} for p in points]  # 2D 点 z 设为 0

    # # 3. 计算 Voronoi 图
    # voronoi = pyvoro.compute_2d_voronoi(
    #     points,
    #     bounds=bounds,
    #     max_radius=10.0  # 最大搜索半径（可调整）
    # )

    # # 4. 提取 ridges
    # ridges = []
    # for cell in voronoi:
    #     for neighbor in cell["faces"]:
    #         if neighbor["adjacent_cell"] is not None:  # 排除无穷远边
    #             edge = neighbor["vertices"]
    #             ridges.append(LineString(edge))
    
    # return ridges

def append_voronoi_to_dxf(dxf_file, ridges):
    """
    将 Voronoi 图的边追加到 DXF 文件中
    :param original_file: 原始 DXF 文件路径
    :param ridges: Voronoi ridges 列表（LineString 格式）
    :param output_file: 输出 DXF 文件路径
    """
    if not ridges:
        print("No ridges to append. Exiting.")
        return
    doc = ezdxf.new()
    msp = doc.modelspace()
    if isinstance(ridges, LineString):
        msp.add_line(start=ridges.coords[0], end=ridges.coords[-1], dxfattribs={"color": 1})
    elif isinstance(ridges, MultiLineString):
        for line in ridges.geoms:
            for start, end in zip(line.coords[:-1], line.coords[1:]):
                msp.add_line(start=start, end=end, dxfattribs={"color": 1})
    doc.saveas(dxf_file)
    
def traverse_geometry(ridges):
    """
    遍历几何对象并处理其元素。
    :param geometry: shapely 几何对象，可以是 Polygon、LineString、MultiPolygon 或 MultiLineString。
    """
    if isinstance(ridges, LineString):
        extractGeo = [ridges]
    elif isinstance(ridges, MultiLineString):
        extractGeo = list(ridges.geoms)
    else:
        extractGeo = []

# 使用 Potrace 转换 PBM 为 dxf
def convert_pbm_to_dxf(pbm_path, dxf_path):   
    # 执行 potrace 命令  
    # 定义 Potrace 参数
    params = [
        'D:/Image2CAD/Image2CAD/potrace',  # potrace 的路径
        pbm_path,                         # 输入文件路径
        '-b', 'dxf',                      # 指定输出格式为 DXF
        '-o', dxf_path,                   # 输出文件路径
        '-z', 'minority',                 # 路径分解策略
        '-t', '1',                        # 忽略小噪点的大小
        '-a', '0.05',                        # 保留清晰的拐角
        '-n',                             # 禁用曲线优化
        '-O', '0.05',                      # 高精度曲线优化容差
        '-u', '20'                        # 输出量化单位
    ]
    
    # 执行 Potrace 命令
    try:
        subprocess.run(params, check=True)
        print(f"DXF file successfully generated: {dxf_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during Potrace execution: {e}")
    except FileNotFoundError:
        print("Potrace executable not found. Please check the path.")
    
def get_text_hocr(input_path, output_path):
    subprocess.run(['E:/Program Files/Tesseract-OCR/tesseract.exe', input_path, output_path, '-c', 'tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', '--psm', '6', '-c', 'tessedit_create_hocr=1'])     
    
# 解析 hocr 文件，提取文本块的位置
def parse_hocr_optimized(hocr_file):
    # 读取 hOCR 文件
    with open(hocr_file, 'r', encoding='utf-8') as file:
        hocr_content = file.read()

    text_positions = []
    try:
        # 使用 BeautifulSoup 解析 hOCR 内容
        soup = BeautifulSoup(hocr_content, 'html.parser')
        
        # 遍历每个包含文本的 ocrx_word 标签
        for word_element in soup.find_all('span', class_='ocrx_word'):
            # 提取坐标和文本
            coords = word_element.get('title', '')
            text = word_element.text.strip() if word_element.text else ''
            
            # 获取 bbox 坐标 (x0, y0, x1, y1)
            if coords:
                # 提取 bbox 坐标，处理包含 ";" 的部分
                bbox_part = coords.split('bbox ')[1].split(';')[0]  # 先提取 bbox 后面的坐标部分，直到遇到 ';'
                bbox_coords = bbox_part.split(' ')[:4]  # 再从中提取前四个坐标
                
                x0, y0, x1, y1 = [int(i) for i in bbox_coords]  # 转换为整数

                # 保存文本和坐标
                text_positions.append((text, x0, y0, x1, y1))
    except Exception as e:
        print(f"Error parsing hOCR file: {e}")

    return text_positions

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
            else:
                print(f"Invalid polygon skipped")
        except Exception as e:
            print(f"Error processing polygon: Error: {e}")
    
    return MultiPolygon(valid_polygons)

# 向 SVG 中插入文本
def insert_text_into_svg(svg_file, text_positions, output_svg_file):
    # 解析现有的 SVG 文件
    tree = etree.parse(svg_file)
    root = tree.getroot()

    # SVG 中的命名空间，通常需要在根元素中处理
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    # 遍历每个提取的文本和坐标信息
    for text, x0, y0, x1, y1 in text_positions:
        # 在 SVG 中创建文本元素
        text_element = etree.Element('text', x=str(x0), y=str(y0), fill="red", font_size="12")
        text_element.text = text

        # 添加文本元素到 SVG 根元素中
        root.append(text_element)

    # 将更新后的 SVG 保存到文件
    tree.write(output_svg_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")

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
    
def pdf_to_images(pdf_path, output_dir=None):  
    pdf_Dir = os.path.abspath(pdf_path)

    dir = os.path.dirname(pdf_Dir)
    # 如果没有指定输出文件夹，则默认创建 output_svg 文件夹
    if output_dir is None:
        output_dir = os.path.join(dir, "pdfImages")
    
      # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
     # 打开 PDF 文件   
    with fitz.open(pdf_path) as doc:    
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # 检查页面是否包含图像资源
            image_list = page.get_images(full=True)

            if image_list:  # 如果页面包含图像
                print(f"页面 {page_num + 1} 是图像，直接提取并保存")
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # 将图像保存为文件
                    image_filename = f"page_{page_num + 1}_image_{img_index + 1}.png"
                    image_path = os.path.join(output_dir, image_filename)
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)

                    print(f"保存图像：{image_path}")
            else:  # 如果页面没有图像，则使用 pdf2image 转换为图像
                print(f"页面 {page_num + 1} 不是图像，转换为图像并保存")
                pages = convert_from_path(pdf_path, 300, first_page=page_num + 1, last_page=page_num + 1)
                image_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
                pages[0].save(image_path, 'PNG')
                print(f"保存图像：{image_path}")
                
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
        for filename in os.listdir(input_path):
            if filename.endswith(".png"):
                file_path = os.path.join(input_path, filename)
                process_single_file(file_path, output_folder)
        print(f"完成png转svg转换，输出到 {output_folder}")
        
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

def process_single_file(input_path, output_folder):
    """
    处理单个 PNG 文件：将其转换为 PBM，再转换为 DXF。
    """
    # 生成 PBM 文件路径
    filename = os.path.basename(input_path)
    pbm_filename = os.path.splitext(filename)[0] + ".pbm"
    output_pbm_path = os.path.join(output_folder, pbm_filename)
    
    # hocr_filename = os.path.splitext(filename)[0] 
    # output_hocrPath = os.path.join(output_folder, hocr_filename)          
    # get_text_hocr(input_path, output_hocrPath)
    # 调用函数将 PNG 转换为 PBM
    convert_png_to_pbm(input_path, output_pbm_path)
    
    # 生成 DXF 文件路径
    dxf_filename = os.path.splitext(filename)[0] + ".dxf"
    output_dxf_path = os.path.join(output_folder, dxf_filename)
    convert_pbm_to_dxf(output_pbm_path, output_dxf_path)
    
   # 提取多边形
    polygons = extract_polygons_from_dxf(output_dxf_path)

    # 计算边界框
    # bounds = calculate_bounds(polygons)

    # 生成 Voronoi 图
    # ridges = perform_voronoi_analysis(polygons)
    
     # 初始化提取器（设置最大间距为 2.5）
    # extractor = OptimizedRidgeExtractor(polygons, max_distance=10)
    attributes = {"id": 1, "name": "polygon", "valid": True}
    multi_polygon = convert_to_multipolygon(polygons)
    centerlines = Centerline(multi_polygon, **attributes)

    # 追加到原始 DXF 并保存
    newdxf_filename = os.path.splitext(filename)[0] + "_new.dxf"
    output_newdxf_path = os.path.join(output_folder, newdxf_filename)
    append_voronoi_to_dxf(output_newdxf_path, centerlines.geometry)


    print(f"Voronoi ridges have been added to {output_dxf_path}.")
            
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
        pdf_to_images(args.input_path, args.output_path)
    elif args.action == 'png2svg':
        png_to_svg(args.input_path, args.output_path)

if __name__ == "__main__":
    main()

