import ezdxf
import os
import logging
import cv2
from ezdxf import units, options, DXFAttributeError
from ezdxf.enums import TextEntityAlignment
from matplotlib.font_manager import findfont, FontProperties
from shapely.validation import make_valid
import platform
from lxml import etree
from typing import Tuple
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.geometry import Polygon, MultiPolygon, MultiLineString, LineString, box
from shapely.ops import unary_union

class dxfProcess:
    DXF_VERSION_MAP = {
    'R12': 'AC1009',
    'R2000': 'AC1015',
    'R2004': 'AC1018',
    'R2007': 'AC1021',
    'R2010': 'AC1024',
    'R2013': 'AC1027',
    'R2018': 'AC1032'
    }
    
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
    def save_to_dxf(cls, dxf_file, merged_lines, text_result, 
                    reference_image=None, 
                    simplified_centerlines=None, 
                    polygon=None):
        """
        将 Voronoi 图的边和文本追加到现有的 DXF 文件中
        :param dxf_file: 现有的 DXF 文件路径
        :param ridges: Voronoi ridges 列表（LineString 格式）
        :param merged_lines: 合并后的线段列表，格式为 [(x1, y1, x2, y2), ...] 或 [((x1, y1), (x2, y2)), ...]
        :param text_result: 文本结果列表，格式为 [(text, x0, y0, x1, y1), ...]
        """
        # try:
        #     # 读取现有的 DXF 文件
        #     doc = ezdxf.new(dxf_file)
        # except IOError:
        #     print(f"无法读取 DXF 文件: {dxf_file}")
        #     return
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()        
        cls.setup_text_styles(doc)
        if reference_image:
            cls.add_image(doc, msp, reference_image)
        if simplified_centerlines:
            cls.add_lines(msp, simplified_centerlines, 9, '脊线')
        if polygon:
            cls.add_multipolygon(msp, polygon, 7, '轮廓')     
        if merged_lines:
            for line in merged_lines:
                if len(line) == 4:  # (x1, y1, x2, y2)
                    x1, y1, x2, y2 = line
                elif len(line) == 2:  # ((x1, y1), (x2, y2))
                    (x1, y1), (x2, y2) = line
                else:
                    raise ValueError(f"无效的线段格式: {line}")
                msp.add_line(start=(x1, y1), end=(x2, y2), dxfattribs={"color": 3, 'layer': '中心线'})
                   
        if text_result:
            words, page_height = text_result  
            if page_height is not None and words is not None:
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
            '文本': {'color': 1, 'linetype': 'HIDDEN', 'lineweight': 0.15}            
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
        
    @staticmethod
    def get_image_size(image_path: str) -> Tuple[int, int]:
        """获取图像尺寸（宽, 高）"""
        try:
            from PIL import Image

            # 检查文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在: {image_path}")

            # 打开图像并获取尺寸
            try:
                with Image.open(image_path) as img:
                    return img.size
            except Image.DecompressionBombError:
                # 安全处理超大图像
                print(f"警告：图像尺寸超过安全限制 {image_path}")
                return (0, 0)  # 或抛出特定异常
            except Exception as e:
                print(f"获取图像尺寸失败: {str(e)}")
                raise

        except ImportError:
            # 回退到OpenCV方法
            import cv2
            img = cv2.imread(image_path)
            if img is not None:
                return img.shape[1], img.shape[0]  # (width, height)
            raise ValueError("无法读取图像文件")

        except Exception as e:
            raise RuntimeError(f"获取图像尺寸失败: {str(e)}") from e
    
    @classmethod
    def set_transparency_effect(cls, doc, entity, transparency):
        """
        版本自适应的透明度设置方法
        :param doc: DXF文档对象
        :param entity: 需要设置透明度的实体（图层/图像）
        :param transparency: 0-100（0完全透明，100不透明）
        """
        version = doc.dxfversion
        alpha = int((100 - transparency) * 0.9)  # 转换为DXF的0-90值
    
        # 版本分支处理
        if version >= 'AC1027':  # 2013+
            entity.dxf.transparency = alpha
        elif version >= 'AC1021':  # 2007-2010
            if hasattr(entity.dxf, 'transparency'):
                entity.dxf.transparency = min(alpha, 80)  # 有限支持
            else:
                entity.dxf.color = cls._get_simulated_color(transparency)
        else:  # 2004及更早
            entity.dxf.color = cls._get_simulated_color(transparency)
    
    @classmethod
    def _legacy_add_image(cls, doc, image_def, insert_point, size):
        """旧版本图像添加兼容方案"""
        # 使用原有兼容逻辑
        image = doc.modelspace().add_image(image_def, insert_point, size)
        
        # 设置旧版本可用属性
        try:
            image.dxf.contrast = 0.4
            image.dxf.brightness = 0.6
            image.dxf.fade = 10
        except DXFAttributeError:
            # R12等极旧版本处理
            cls._preprocess_image_effects(image_def.filename)
        
        # 添加透明覆盖层
        cls.add_transparency_overlay(doc, insert_point, size, 50)
        return image

    @staticmethod
    def _preprocess_image_effects(image_path, contrast=1.0, brightness=1.0):
        """为旧版本DXF预处理图像效果"""
        try:
            import cv2
            img = cv2.imread(image_path)
            img = cv2.convertScaleAbs(img, alpha=contrast*2.5, beta=(brightness-0.5)*100)
            cv2.imwrite(image_path, img)
        except ImportError:
            print("警告：需要OpenCV进行图像预处理")
        
    @staticmethod
    def _get_simulated_color(transparency):
        """为旧版本生成模拟透明度的颜色"""
        # 将透明度转换为灰度值（0=黑，100=白）
        gray_value = int(255 * (transparency / 100))
        return ezdxf.colors.rgb2int((gray_value, gray_value, gray_value))
    
    @classmethod
    def add_image_with_compatibility(cls, doc, image_def, insert_point, size):
        """严格遵循 DXF 规格的图像添加方法"""
        image = None
        dxf_version = doc.dxfversion

        try:
            # 基础图像添加（所有版本支持）
            image = doc.modelspace().add_image(image_def, insert_point, size)

            # 版本特性开关
            support_contrast = dxf_version >= 'AC1032'  # 仅2018+支持
            support_fade = dxf_version >= 'AC1032'

            # 安全设置属性
            if support_contrast:
                image.dxf.image_contrast = 70
                image.dxf.image_brightness = 50
            if support_fade:
                image.dxf.image_fade = 20

        except DXFAttributeError as e:
            print(f"跳过不支持的属性: {str(e)}")
            # 回退到基本图像对象
            image = doc.modelspace().add_image(image_def, insert_point, size)

        finally:
            # 设置通用属性
            if image:
                image.dxf.layer = 'REF_IMAGE'               
            return image
    
    @classmethod
    def add_transparency_overlay(cls, doc, insert_point, size, transparency):
        """为旧版本添加透明覆盖层"""
        overlay = doc.modelspace().add_solid(
            points=[
                insert_point,
                (insert_point[0] + size[0], insert_point[1]),
                (insert_point[0] + size[0], insert_point[1] + size[1]),
                insert_point
            ],
            dxfattribs={
                'layer': 'OVERLAY',
                'color': cls._get_simulated_color(transparency),
                'elevation': 0.01  # 确保覆盖在图像上方
            }
        )
        return overlay
            
    @classmethod
    def set_layer_transparency(cls, layer, transparency):
        """
        兼容各DXF版本的透明度设置方法
        :param layer: ezdxf.layer.Layer对象
        :param transparency: 0-100（0=完全透明，100=不透明）
        """
        # 转换为透明度值（0-90，其中90=完全透明）
        alpha = int((100 - transparency) * 0.9)
        
        if cls.doc.dxfversion >= 'AC1027':  # 2013+版本
            layer.dxf.transparency = alpha
        else:
            # 旧版本使用颜色重映射
            remap_color = {
                100: 7,   # 不透明-白色
                70: 8,    # 30%透明-灰色
                50: 9,    # 50%透明-浅灰
                30: 6     # 70%透明-深灰
            }
            closest = min(remap_color.keys(), key=lambda x: abs(x - transparency))
            layer.dxf.color = remap_color[closest]
            print(f"旧版本DXF使用替代颜色: {remap_color[closest]}")

    @classmethod
    def add_text(cls, msp, text, position, height, rotation=0, layer='文本'):
        text_entity = msp.add_text(
            text,
            dxfattribs={
                'height': height,
                "rotation": rotation,
                'color': 1,
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
    def add_image(cls, doc, msp, reference_image, transparency=30):
        try:            
            # 获取图像实际路径（处理Windows路径）
            img_path = os.path.abspath(reference_image).replace('\\', '/')
            if not os.path.exists(img_path):
                return
            # 计算插入参数
            img_width, img_height = cls.get_image_size(img_path)
            scale_factor = 1  # 根据实际需求调整
            
            # 创建图像定义
            image_def = doc.add_image_def(
                filename=img_path,
                size_in_pixel=(img_width, img_height)
            )   
            
            ref_layer_name = 'REF_IMAGE'
            if ref_layer_name not in doc.layers:
                doc.layers.add(name=ref_layer_name, 
                      dxfattribs={
                          'color': 7,  # 白色                        
                          'lineweight': 0  # 无边框
                      })
            layer = doc.layers.get('REF_IMAGE')
            cls.set_transparency_effect(doc, layer, transparency)
            
            # 在图纸左下角插入图像（位置可调）
            insert_point = (0, 0)
            cls.add_image_with_compatibility(doc, image_def, insert_point, (img_width,img_height))
        except Exception as e:
            print(f"底图插入失败: {str(e)}")

    @classmethod
    def add_multipolygon(cls, msp, multipolygon, color, layername):
        if isinstance(multipolygon, MultiPolygon):
            for polygon in multipolygon.geoms:
                for ring in [polygon.exterior] + list(polygon.interiors):  # 处理外环 + 内环
                    points = list(ring.coords)  # 获取坐标
                    msp.add_lwpolyline(
                        points, 
                        close=True,
                        dxfattribs={"color": color, "layer": layername})  # 添加轻量级多段线 
        
    @classmethod
    def add_lines(cls, msp, ridges, color, layername):
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
                    msp.add_lwpolyline(child, close=False, dxfattribs={"color": color, "layer": layername})

        # 批量添加线段
        if line_segments:
            msp.add_lwpolyline(
                [p for segment in line_segments for p in segment],
                close=False,
                dxfattribs={"color": color, "layer": layername}
            )