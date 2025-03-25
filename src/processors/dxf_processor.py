# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Union, Optional, Dict, Any, Tuple
from pathlib import Path
import ezdxf
import os
import logging
import cv2
from ezdxf import units, options, DXFAttributeError
from ezdxf.enums import TextEntityAlignment
from matplotlib.font_manager import findfont, FontProperties
from shapely.validation import make_valid
import platform
import argparse
from lxml import etree
import sys
import os
from dataclasses import dataclass
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, MultiLineString, LineString
from shapely.ops import unary_union

from src.common.utils import Util
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.geometry import Polygon, MultiPolygon, MultiLineString, LineString, box
from shapely.ops import unary_union
from src.common.log_manager import log_mgr

@dataclass
class DXFConfig:
    """DXF配置参数"""
    version: str = 'R2010'
    text_style: str = '工程字体'
    image_transparency: int = 30
    layers: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.layers is None:
            self.layers = {
                '轮廓': {'color': 7, 'linetype': 'CONTINUOUS', 'lineweight': 0.15},
                '脊线': {'color': 9, 'linetype': 'CONTINUOUS', 'lineweight': 0.15},
                '中心线': {'color': 3, 'linetype': 'CENTER', 'lineweight': 0.30},
                '文本': {'color': 1, 'linetype': 'HIDDEN', 'lineweight': 0.15}
            }

class DXFProcessor:
    """DXF处理器"""
    
    def __init__(self, config: Optional[DXFConfig] = None):
        self.config = config or DXFConfig()
        self.doc = None
        self.msp = None
        
    def create_document(self) -> None:
        """创建新的DXF文档"""
        try:
            self.doc = ezdxf.new(self.config.version)
            self.msp = self.doc.modelspace()
            self._setup_document()
        except Exception as e:
            log_mgr.log_error(f"创建DXF文档失败: {str(e)}")
            raise

    def _setup_document(self) -> None:
        """设置文档基本配置"""
        try:
            self._setup_layers()
            DXFProcessor.setup_text_styles(self.doc)
        except Exception as e:
            log_mgr.log_error(f"设置DXF文档配置失败: {str(e)}")
            raise

    def _setup_layers(self) -> None:
        """设置图层"""
        try:
            for name, props in self.config.layers.items():
                if name not in self.doc.layers:
                    layer = self.doc.layers.add(name=name)
                else:
                    layer = self.doc.layers.get(name)
                layer.color = props['color']
                layer.linetype = props['linetype']
                layer.lineweight = props['lineweight']
        except Exception as e:
            log_mgr.log_error(f"设置图层失败: {str(e)}")
            raise

    def add_text(self, text: str, position: Tuple[float, float], 
                height: float, rotation: float = 0.0, 
                layer: str = '文本') -> None:
        """添加文本"""
        try:
            if not self.msp:
                raise ValueError("DXF文档未初始化")
                
            text_entity = self.msp.add_text(
                text,
                dxfattribs={
                    'height': height,
                    'rotation': rotation,
                    'color': self.config.layers[layer]['color'],
                    'layer': layer,
                    'style': self.config.text_style
                }
            )
            
            text_entity.set_placement(
                position,
                align=TextEntityAlignment.LEFT
            )
            
        except Exception as e:
            log_mgr.log_error(f"添加文本失败: {str(e)}")
            raise
            
    def add_image(self, image_path: Union[str, Path]) -> None:
        """添加参考图像"""
        try:
            image_path = Path(image_path).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
                
            # 获取图像尺寸
            width, height = self._get_image_size(image_path)
            
            # 创建图像定义
            image_def = self.doc.add_image_def(
                filename=str(image_path),
                size_in_pixel=(width, height)
            )
            
            # 设置图像图层
            ref_layer_name = 'REF_IMAGE'
            if ref_layer_name not in self.doc.layers:
                layer = self.doc.layers.add(
                    name=ref_layer_name,
                    dxfattribs={
                        'color': 7,
                        'lineweight': 0
                    }
                )
            else:
                layer = self.doc.layers.get(ref_layer_name)
                
            # 设置透明度
            self._set_transparency(layer, self.config.image_transparency)
            
            # 添加图像
            self._add_image_entity(image_def, (0, 0), (width, height))
            
        except Exception as e:
            log_mgr.log_error(f"添加图像失败: {str(e)}")
            raise

    def add_geometry(self, geometry: Union[MultiPolygon, MultiLineString, List[LineString]], 
                    layer: str, color: int) -> None:
        """添加几何图形"""
        try:
            if isinstance(geometry, MultiPolygon):
                self._add_multipolygon(geometry, color, layer)
            elif isinstance(geometry, (MultiLineString, List)):
                self._add_lines(geometry, color, layer)
            else:
                raise ValueError(f"不支持的几何类型: {type(geometry)}")
        except Exception as e:
            log_mgr.log_error(f"添加几何图形失败: {str(e)}")
            raise

    def save(self, output_path: Union[str, Path]) -> None:
        """保存DXF文档"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.doc.saveas(output_path)
        except Exception as e:
            log_mgr.log_error(f"保存DXF文件失败: {str(e)}")
            raise

    @staticmethod
    def _get_image_size(image_path: Path) -> Tuple[int, int]:
        """获取图像尺寸"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
            return img.shape[1], img.shape[0]
        except Exception as e:
            log_mgr.log_error(f"获取图像尺寸失败: {str(e)}")
            raise

    def _set_transparency(self, entity: Any, transparency: int) -> None:
        """设置透明度"""
        try:
            alpha = int((100 - transparency) * 0.9)
            if self.doc.dxfversion >= 'AC1027':  # 2013+
                entity.dxf.transparency = alpha
            else:
                # 旧版本使用颜色模拟透明度
                gray = int(255 * (transparency / 100))
                entity.dxf.color = ezdxf.colors.rgb2int((gray, gray, gray))
        except Exception as e:
            log_mgr.log_error(f"设置透明度失败: {str(e)}")
            raise

    def _add_multipolygon(self, multipolygon: MultiPolygon, 
                         color: int, layer: str) -> None:
        """添加多边形"""
        try:
            for polygon in multipolygon.geoms:
                # 添加外环
                points = list(polygon.exterior.coords)
                self.msp.add_lwpolyline(
                    points,
                    close=True,
                    dxfattribs={
                        "color": color,
                        "layer": layer
                    }
                )
                
                # 添加内环
                for interior in polygon.interiors:
                    points = list(interior.coords)
                    self.msp.add_lwpolyline(
                        points,
                        close=True,
                        dxfattribs={
                            "color": color,
                            "layer": layer
                        }
                    )
        except Exception as e:
            log_mgr.log_error(f"添加多边形失败: {str(e)}")
            raise

    def _add_lines(self, ridges, color, layername):
        """
        批量将线段添加到 DXF
        :param ridges: 可为 LineString、MultiLineString、List（包含 LineString 或 点集）
        """
        try:
            if not ridges:
                return
            
            # 处理单个直线段
            if isinstance(ridges, tuple) and len(ridges) == 2:
                self.msp.add_line(
                    start=ridges[0],
                    end=ridges[1],
                    dxfattribs={"color": color, "layer": layername}
                )
                return
            
            # 处理直线段列表
            if isinstance(ridges, list):
                for line in ridges:
                    if isinstance(line, tuple) and len(line) == 2:
                        # 直线段格式 ((x1,y1), (x2,y2))
                        self.msp.add_line(
                            start=line[0],
                            end=line[1],
                            dxfattribs={"color": color, "layer": layername}
                        )
                    elif len(line) == 4:
                        # 直线段格式 (x1,y1,x2,y2)
                        self.msp.add_line(
                            start=(line[0], line[1]),
                            end=(line[2], line[3]),
                            dxfattribs={"color": color, "layer": layername}
                        )
                    elif isinstance(line, LineString):
                        # 处理 LineString
                        coords = list(line.coords)
                        if len(coords) >= 2:
                            for start, end in zip(coords[:-1], coords[1:]):
                                self.msp.add_line(
                                    start=start,
                                    end=end,
                                    dxfattribs={"color": color, "layer": layername}
                                )
                            
            # 处理 LineString
            elif isinstance(ridges, LineString):
                coords = list(ridges.coords)
                if len(coords) >= 2:
                    for start, end in zip(coords[:-1], coords[1:]):
                        self.msp.add_line(
                            start=start,
                            end=end,
                            dxfattribs={"color": color, "layer": layername}
                        )
                    
            # 处理 MultiLineString
            elif isinstance(ridges, MultiLineString):
                for line in ridges.geoms:
                    coords = list(line.coords)
                    if len(coords) >= 2:
                        for start, end in zip(coords[:-1], coords[1:]):
                            self.msp.add_line(
                                start=start,
                                end=end,
                                dxfattribs={"color": color, "layer": layername}
                            )
                            
        except Exception as e:
            log_mgr.log_error(f"添加线段失败: {str(e)}")
            raise

    @classmethod
    def extract_polygons_from_dxf(cls, file_path, show_progress=True):
        """
        从 DXF 文件中提取多边形和直线数据
        :param file_path: DXF文件路径
        :param show_progress: 是否显示进度条
        :return: 多边形和直线列表
        """
        if show_progress:
            print(f"正在读取DXF文件: {file_path}")
        
        try:
            doc = ezdxf.readfile(file_path)
        except IOError:
            print(f"无法读取 DXF 文件: {file_path}")
            return []

        msp = doc.modelspace()
        # 添加 LINE 实体到查询中
        entities = list(msp.query("POLYLINE LWPOLYLINE LINE"))
        
        if show_progress:
            print(f"找到 {len(entities)} 个实体")
        
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
            elif entity.dxftype() == "LINE":
                # 处理直线段
                start = (entity.dxf.start.x, entity.dxf.start.y)
                end = (entity.dxf.end.x, entity.dxf.end.y)
                # 将直线段转换为两点的多边形
                return [start, end]
            return None
        
        max_workers = max(1, os.cpu_count() // 2)
        
        if show_progress:
            print(f"使用 {max_workers} 个线程并行处理实体...")
        
        # 使用线程池并行处理实体
        with ThreadPoolExecutor(max_workers) as executor:
            futures = [executor.submit(process_entity, entity) for entity in entities]
            
            # 使用tqdm显示处理进度
            if show_progress:
                import tqdm
                for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="处理DXF实体"):
                    result = future.result()
                    if result is not None:
                        polygons.append(result)
            else:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        polygons.append(result)

        if show_progress:
            print(f"成功提取 {len(polygons)} 个几何实体")
        
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
   
    def save_to_dxf(self, dxf_file, merged_lines, text_result, 
                    image_path=None, 
                    simplified_centerlines=None, 
                    polygon=None):
        """将处理结果保存到DXF文件
        
        Args:
            dxf_file: DXF文件保存路径
            merged_lines: 合并后的线段列表，格式为 [(x1,y1,x2,y2), ...] 或 [((x1,y1), (x2,y2)), ...]
            text_result: OCR文本结果，格式为 (words, page_height)
                        其中words为 [(text, x, y, width, height, angle), ...]
            image_path: 可选，参考图像路径
            simplified_centerlines: 可选，简化后的中心线
            polygon: 可选，轮廓多边形
        """
        try:
            # 创建新的DXF文档
            doc = ezdxf.new('R2010')
            self.doc = doc
            self.msp = doc.modelspace()
            
            # 设置图层和文字样式
            self._setup_document()
            
            # 添加参考图像（如果有）
            if image_path:
                self.add_image(image_path)
            
            # 添加简化中心线（如果有）
            if simplified_centerlines:
                self.add_lines(simplified_centerlines, 9, '脊线')
            
            # 添加轮廓（如果有）
            if polygon:
                self.add_geometry(polygon, '轮廓', 7)
            
            # 添加中心线
            if merged_lines:
                for line in merged_lines:
                    try:
                        if len(line) == 4:  # (x1, y1, x2, y2)
                            x1, y1, x2, y2 = line
                            start, end = (x1, y1), (x2, y2)
                        elif len(line) == 2:  # ((x1, y1), (x2, y2))
                            start, end = line
                        else:
                            log_mgr.log_warning(f"跳过无效的线段格式: {line}")
                            continue
                            
                        # 使用 add_line 而不是 add_lwpolyline
                        self.msp.add_line(
                            start=start,
                            end=end,
                            dxfattribs={
                                "color": 3,
                                'layer': '中心线',
                                'linetype': 'CENTER'  # 添加中心线线型
                            }
                        )
                    except Exception as e:
                        log_mgr.log_error(f"添加线段失败: {str(e)}")
            
            # 添加文本
            if text_result:
                words, page_height = text_result
                if words:
                    for text, x, y, width, height, angle in words:
                        try:
                            if height <= 0:
                                continue
                            self.add_text(
                                text=text,
                                position=(x, y),
                                height=height * 0.8,  # 稍微调整文本高度
                                rotation=angle,
                                layer='文本'
                            )
                        except Exception as e:
                            log_mgr.log_error(f"添加文本失败: {text}, {str(e)}")
            
            # 保存DXF文件
            doc.saveas(dxf_file)
            log_mgr.log_info(f"DXF文件已保存: {dxf_file}")
            
        except Exception as e:
            log_mgr.log_error(f"保存DXF文件失败: {str(e)}")
            raise

    @staticmethod
    def geometry_to_txt(geometries, filename="output.txt"):
        """将几何实体（多边形和线段）保存为文本格式
        
        Args:
            geometries: 从 extract_polygons_from_dxf 返回的几何实体列表
            filename: 输出文件路径
        """
        with open(filename, "w") as f:
            for i, geom in enumerate(geometries):
                if isinstance(geom, (Polygon, MultiPolygon)):
                    # 处理 Polygon 或 MultiPolygon 对象
                    if isinstance(geom, MultiPolygon):
                        for j, polygon in enumerate(geom.geoms):
                            f.write(f"Polygon {i+1}-{j+1}:\n")
                            DXFProcessor._write_polygon_to_file(f, polygon)
                    else:
                        f.write(f"Polygon {i+1}:\n")
                        DXFProcessor._write_polygon_to_file(f, geom)
                elif isinstance(geom, list):
                    if len(geom) == 2 and all(isinstance(p, tuple) for p in geom):
                        # 处理线段 [(x1,y1), (x2,y2)]
                        f.write(f"Line {i+1}:\n")
                        start, end = geom
                        f.write(f"  Start: {start[0]}, {start[1]}\n")
                        f.write(f"  End: {end[0]}, {end[1]}\n")
                    elif len(geom) >= 3:
                        # 处理多边形点列表
                        f.write(f"Polygon {i+1}:\n")
                        f.write("  Exterior:\n")
                        for point in geom:
                            if len(point) >= 2:
                                f.write(f"    {point[0]}, {point[1]}\n")
                    else:
                        f.write("  Empty Geometry\n")
                else:
                    print(f"未知的几何数据类型: {type(geom)}")

                f.write("\n")  # 分隔几何实体
                
        print(f"TXT 文件已保存为 {filename}")

    @staticmethod
    def _write_polygon_to_file(f, polygon):
        """将 Polygon 对象写入文件"""
        # 处理外边界
        if polygon.exterior:
            f.write("  Exterior:\n")
            for coord in polygon.exterior.coords:
                x, y = coord[:2]  # 兼容 2D 和 3D
                f.write(f"    {x}, {y}\n")
        else:
            f.write("  Empty Polygon\n")

        # 处理内部孔洞
        for k, interior in enumerate(polygon.interiors):
            f.write(f"  Hole {k+1}:\n")
            for coord in interior.coords:
                x, y = coord[:2]  # 兼容 2D 和 3D
                f.write(f"    {x}, {y}\n")
    
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
            img = Util.opencv_read(image_path)
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
            img = Util.opencv_read(image_path)
            img = cv2.convertScaleAbs(img, alpha=contrast*2.5, beta=(brightness-0.5)*100)
            Util.opencv_write(img, image_path)
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
    def add_multipolygon(cls, multipolygon, color, layername):
        if isinstance(multipolygon, MultiPolygon):
            for polygon in multipolygon.geoms:
                for ring in [polygon.exterior] + list(polygon.interiors):
                    points = list(ring.coords)
                    cls.msp.add_lwpolyline(
                        points, 
                        close=True,
                        dxfattribs={"color": color, "layer": layername})
    
    @classmethod
    def export_dxf_to_txt(cls, input_file, output_file):
        """
        将DXF文件中的几何实体（多边形和线段）导出为文本格式
        :param input_file: 输入DXF文件路径
        :param output_file: 输出文本文件路径
        """
        geometries = cls.extract_polygons_from_dxf(input_file)
        cls.geometry_to_txt(geometries, output_file)
            
    def _add_image_entity(self, image_def, insert_point: Tuple[float, float], size: Tuple[float, float]) -> None:
        """添加图像实体到DXF文档
        
        Args:
            image_def: 图像定义对象
            insert_point: 插入点坐标 (x, y)
            size: 图像尺寸 (width, height)
        """
        try:
            # 根据DXF版本选择合适的添加方法
            if self.doc.dxfversion >= 'AC1027':  # 2013+
                image = self.msp.add_image(
                    image_def=image_def,
                    insert=insert_point,
                    size=size,
                    rotation=0,
                    dxfattribs={
                        'layer': 'REF_IMAGE'
                    }
                )
                
                # 设置图像透明度 (0-100 转换为 0-1)
                if hasattr(image.dxf, 'transparency'):
                    try:
                        # ezdxf中transparency的值应该在0-1之间
                        transparency_value = self.config.image_transparency / 100.0
                        image.dxf.transparency = transparency_value
                    except Exception as e:
                        log_mgr.log_warning(f"设置图像透明度失败: {str(e)}")
            else:
                # 使用兼容模式添加图像
                image = DXFProcessor._legacy_add_image(
                    self.doc,
                    image_def,
                    insert_point,
                    size
                )
            
        except Exception as e:
            log_mgr.log_error(f"添加图像实体失败: {str(e)}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dxf 工具")
    subparsers = parser.add_subparsers(dest='command')

    # 添加 convert 子命令
    convert_parser = subparsers.add_parser('export_dxf_to_txt', help='导出dxf信息到txt')
    convert_parser.add_argument('input_file', type=str, help='输入文件路径')
    convert_parser.add_argument('output_file', type=str, help='输出文件路径')  
    
    # 解析命令行参数
    args = parser.parse_args()

    if args.command == 'export_dxf_to_txt':
        DXFProcessor.export_dxf_to_txt(args.input_file, args.output_file)
    else:
        print("请输入正确的命令")
                