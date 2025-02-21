import ezdxf
import os
import logging
from ezdxf import units, options
from ezdxf.enums import TextEntityAlignment
from matplotlib.font_manager import findfont, FontProperties
import platform
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
    
    @staticmethod
    def setup_dxf_logging():
        """配置DXF日志输出级别"""
        # 关闭ezdxf的调试日志       
        logging.getLogger('ezdxf').setLevel(logging.WARNING)
        
        # 禁用特定模块的日志
        logging.getLogger('ezdxf.entities.dictionary').setLevel(logging.WARNING)
        logging.getLogger('ezdxf.lldxf.tags').setLevel(logging.ERROR)
        
        # 禁用颜色相关日志
        logging.getLogger('ezdxf.colors').setLevel(logging.CRITICAL)
    
    @classmethod  
    def upgrade_dxf(cls, input_file, output_file, target_version="R2010"):
        """升级 DXF 版本，并保留所有数据（Layers, Blocks, Attributes, Layouts）"""
        try:
            # 读取旧 DXF 文件
            old_doc = ezdxf.readfile(input_file)
        except (IOError, ezdxf.DXFStructureError) as e:
            print(f"无法读取 DXF 文件: {input_file}, 错误: {e}")
            return False

        # 创建新的 DXF 文档
        new_doc = ezdxf.new(target_version)

        ### **1. 复制 DXF 头部信息**
        for key in old_doc.header.varnames():
            value = old_doc.header[key]
            if value is not None:  # 只复制有效数据
                new_doc.header[key] = value

        ### **2. 复制 Layers（图层）**
        for layer in old_doc.layers:
            if layer.dxf.name not in new_doc.layers:
                new_doc.layers.new(name=layer.dxf.name, dxfattribs=layer.dxfattribs())

        ### **3. 复制 Blocks（块）**
        for block in old_doc.blocks:
            if block.name not in new_doc.blocks:
                new_block = new_doc.blocks.new(name=block.name)
                for entity in block:
                    try:
                        new_block.add_entity(entity.copy())
                    except Exception as e:
                        print(f"跳过无法复制的块实体: {e}")

        ### **4. 复制所有 Layouts（包括 ModelSpace 和 PaperSpace）**
        for layout_name in old_doc.layout_names():
            old_layout = old_doc.layout(layout_name)
            new_layout = new_doc.layout(layout_name) if layout_name != "Model" else new_doc.modelspace()

            entity_count = 0
            for entity in old_layout:
                try:
                    new_entity = entity.copy() if hasattr(entity, "copy") else None
                    if new_entity:
                        new_layout.add_entity(new_entity)
                    else:
                        new_layout.import_entity(entity)
                    entity_count += 1
                except Exception as e:
                    print(f"跳过无法复制的实体 ({layout_name}): {e}")

            print(f"复制 {layout_name} 的 {entity_count} 个实体")

        ### **5. 保存 DXF**
        new_doc.saveas(output_file)
        print(f"DXF 版本已升级到 {target_version}，保留所有数据，保存至 {output_file}")
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
            cls.add_text(msp, text, (x, y), height, 0, layer='文本')  

        # 保存修改后的 DXF 文件
        doc.saveas(dxf_file)  
    
    @staticmethod
    def find_font(font_name):
        """跨平台字体查找函数"""
        # 先尝试系统字体查找
        try:
            path = findfont(FontProperties(fname=font_name))
            if os.path.exists(path):
                return path
        except:
            pass

        # 自定义字体搜索路径
        search_paths = []
        system = platform.system()

        # 不同系统的字体目录
        if system == "Windows":
            search_paths.append("C:/Windows/Fonts/")
        elif system == "Darwin":  # macOS
            search_paths.extend(["/Library/Fonts/", "/System/Library/Fonts/", os.path.expanduser("~/Library/Fonts/")])
        else:  # Linux
            search_paths.extend(["/usr/share/fonts/", "/usr/local/share/fonts/", os.path.expanduser("~/.fonts/")])

        # 递归搜索字体文件
        for path in search_paths:
            if not os.path.exists(path):
                continue
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower() == font_name.lower():
                        return os.path.join(root, file)
        return None
    
    @classmethod
    def setup_layers(cls, doc):
       # 创建标准图层配置
        layers = {
            '轮廓': {'color': 7, 'linetype': 'CONTINUOUS', 'lineweight': 0.15},        
            '脊线': {'color': 9, 'linetype': 'CONTINUOUS', 'lineweight': 0.15},        
            '中心线': {'color': 3, 'linetype': 'CENTER', 'lineweight': 0.30},
            '文本': {'color': 2, 'linetype': 'HIDDEN', 'lineweight': 0.15}            
        }

        # 初始化图层
        for layer_name, props in layers.items():
            if layer_name not in doc.layers:             
                layer = doc.layers.add(name=layer_name)
            else:               
                layer = doc.layers.get(layer_name)       
            layer.color = props['color']
            layer.linetype = props['linetype']
            layer.lineweight = props['lineweight'] 
            
    @classmethod
    def setup_text_styles(cls, doc):        
        style_name = '工程字体'    
        # 检查并创建/获取文字样式
        if style_name in doc.styles:
            style = doc.styles.get(style_name)           
        else:
            style = doc.styles.new(style_name)       
        # 正确设置字体属性的方式
        font_mapping = {
            # (字体文件, 主字体名, 大字体名)
            'simfang.ttf': ('simfang.shx', 'hztxt.shx'),  # 仿宋体
            'FangSong.ttf': ('FSA_GB.shx', 'hztxt.shx'),  # 方正仿宋
            'simhei.ttf': ('simhei.shx', None),           # 黑体
            'Arial': ('arial.ttf', None)                 # 英文默认
        }
        # 查找可用字体
        for font_file, (shx_font, bigfont) in font_mapping.items():
            if cls.find_font(font_file):
                try:
                    # 正确设置DXF字体属性
                    style.dxf.font = shx_font
                    if bigfont:
                        style.dxf.bigfont = bigfont                    
                    break
                except Exception as e:
                    print(f"字体设置失败: {str(e)}")
                    continue
        else:
            # 回退到默认设置
            style.dxf.font = 'arial.ttf'
            style.dxf.bigfont = None
            print("警告：使用默认字体Arial")

        # 设置其他属性
        style.width = 0.7
        style.last_height = 3.0  # 添加默认字高      

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