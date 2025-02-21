import ezdxf
import os
from ezdxf import units
from ezdxf.enums import TextEntityAlignment
from lxml import etree
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.geometry import Polygon, MultiPolygon, MultiLineString, LineString, box

class dxfProcess:
    @classmethod
    def extract_polygons_from_dxf(cls, file_path):
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
    
    @classmethod
    def upgrade_dxf(cls, input_file, output_file, target_version="R2010"):
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
    
    @classmethod
    def append_to_dxf(cls, dxf_file, multi_polygon, ridges, merged_lines, text_result):
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
            '轮廓': {'color': 7, 'linetype': 'CONTINUOUS', 'lineweight': 0.15},        
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
        cls.setup_text_styles(doc)

        cls.add_multipolygon(msp, multi_polygon)
        cls.add_ridges(msp, ridges)

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
            cls.dd_text(msp, text, (x, y), height, 0, layer='文本')  

        # 保存修改后的 DXF 文件
        doc.saveas(dxf_file)  
        
    @classmethod
    def setup_text_styles(cls, doc):
        style = doc.styles.new('工程字体')
        style.font = 'simfang.ttf'  # 仿宋字体
        style.width = 0.7  # 长宽比

    @classmethod
    def add_text(cls, msp, text, position, height, rotation=0, layer='文本'):
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
    
    @classmethod
    def add_multipolygon(cls, msp, multipolygon):
       for polygon in multipolygon.geoms:
        for ring in [polygon.exterior] + list(polygon.interiors):  # 处理外环 + 内环
            points = list(ring.coords)  # 获取坐标
            msp.add_lwpolyline(
                points, 
                close=True,
                dxfattribs={"color": 7, "layer": "轮廓"})  # 添加轻量级多段线 
        
    @classmethod
    def add_ridges(cls, msp, ridges):
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