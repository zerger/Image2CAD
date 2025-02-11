import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from lxml import etree
import xml.etree.ElementTree as ET
import argparse
import subprocess
import cv2
from bs4 import BeautifulSoup
import numpy as np
from ShowImage import ShowImage
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
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
from sklearn.cluster import DBSCAN
from multiprocessing import Pool
from multiprocessing import cpu_count
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from rtree import index
import shutil
import time
import os
      
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

    # 使用线程池并行处理实体
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_entity, entity) for entity in entities]
        for future in futures:
            result = future.result()
            if result is not None:
                polygons.append(result)

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
def smooth_line(line_coords, s=0.0):
    if len(line_coords) < 4:
        return line_coords
    # 将坐标转化为数组
    # 移除NaN值，如果有的话
    coords = np.array(line_coords)
    coords = coords[~np.isnan(coords).any(axis=1)]  # 移除包含NaN的行
    # 检查输入数据，确保其不是空的，并且有足够的点
    if len(line_coords) < 2:
        return line_coords
    # 确保数据为二维
    if coords.ndim != 2 or coords.shape[1] != 2:
        return line_coords
    
    # 使用B样条拟合线条，s为平滑参数，值越大平滑效果越强
    tck, u = splprep(coords.T, s=s)
    
    # 生成平滑后的坐标
    smooth_coords = np.array(splev(u, tck)).T
    return smooth_coords

def filter_short_segments(ridges, min_length=0.1):
    if isinstance(ridges, LineString):
        # 计算线段长度，如果短于min_length则不保留
        if ridges.length >= min_length:
            return [ridges]
        else:
            return []
    elif isinstance(ridges, MultiLineString):
        result = []
        for line in ridges.geoms:
            if line.length >= min_length:
                result.append(line)
        return result
    return []

def simplify_line(ridges, tolerance=0.1):
    return ridges.simplify(tolerance, preserve_topology=True)

def process_ridges(ridges, min_length=0.1, smooth_s=0.5, tolerance=0.1):
     # 平滑线条
    if isinstance(ridges, LineString):
        smoothed = smooth_line(ridges.coords, s=smooth_s)
        ridges = LineString(smoothed)
    elif isinstance(ridges, MultiLineString):
        smoothed_lines = []
        for line in ridges.geoms:
            smoothed = smooth_line(line.coords, s=smooth_s)
            smoothed_lines.append(LineString(smoothed))
        ridges = MultiLineString(smoothed_lines)
    
    # 移除短线段
    ridges = filter_short_segments(ridges, min_length)
    if ridges is None:  # 如果是单条短线，过滤掉
        return None
    
    # 简化线条
    if isinstance(ridges, LineString):
        ridges = simplify_line(ridges, tolerance)
    elif isinstance(ridges, MultiLineString):
        ridges = MultiLineString([simplify_line(line, tolerance) for line in ridges.geoms])
    
    return ridges

def merge_nearby_lines_optimized(ridges, tolerance=0.1):
    if isinstance(ridges, LineString):
        # 单一 LineString，暂时不合并
        ridges = [ridges]
    elif isinstance(ridges, MultiLineString):
        # 如果是 MultiLineString，转换为 list
        ridges = list(ridges.geoms)
    # 获取所有线段的中心点
    centers = []
    for line in ridges:
        # 计算线段的中点作为代表
        mid_point = np.mean(np.array(line.coords), axis=0)
        centers.append(mid_point)
    
    # 使用 k-d 树来索引线段的中心点
    tree = KDTree(centers)
    
    merged = []
    merged_indices = set()
    
    for i, line in enumerate(ridges):
        if i in merged_indices:
            continue
        
        # 找到当前线段周围距离小于容差的其他线段
        indices = tree.query_ball_point(centers[i], tolerance)
        to_merge = [ridges[j] for j in indices if j != i and j not in merged_indices]
        
        # 合并这些线段
        if to_merge:
            combined = line
            for line_to_merge in to_merge:
                combined = combined.union(line_to_merge)
                merged_indices.add(ridges.index(line_to_merge))
            
            merged.append(combined)
        else:
            merged.append(line)
            merged_indices.add(i)

    return merged

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

    # 使用霍夫变换检测线段
    lines_detected = cv2.HoughLinesP(img, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

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

def merge_with_dbscan(ridges, tolerance=0.1):
    if isinstance(ridges, LineString):
        # 单一 LineString，暂时不合并
        ridges = [ridges]
    elif isinstance(ridges, MultiLineString):
        # 如果是 MultiLineString，转换为 list
        ridges = list(ridges.geoms)
    # 提取每个线段的中点作为聚类的特征
    centers = np.array([np.mean(np.array(line.coords), axis=0) for line in ridges])
    
    # DBSCAN 聚类，eps 控制容差，min_samples 为最小样本数
    clustering = DBSCAN(eps=tolerance, min_samples=1).fit(centers)
    
    # 根据聚类结果合并线段
    merged = []
    for label in set(clustering.labels_):
        cluster = [ridges[i] for i in range(len(ridges)) if clustering.labels_[i] == label]
        combined = cluster[0]
        for line in cluster[1:]:
            combined = combined.union(line)  # 合并线段
        merged.append(combined)
    
    return merged

def build_graph_from_lines(multi_line_string):
    """
    根据输入的 MultiLineString 构建一个无向图
    :param multi_line_string: 输入的 MultiLineString 数据
    :return: NetworkX 图对象
    """
    graph = nx.Graph()    

    points = []
    for line in multi_line_string.geoms:
        for i in range(len(line.coords) - 1):
            start = line.coords[i]  # 统一精度
            end = line.coords[i + 1]  # 统一精度
            graph.add_edge(start, end, weight=distance(start, end))
            points.append(start)
            points.append(end)                
    # 合并点
    merged_points, point_map = merge_points(points, tolerance=5)
    # 更新图
    updated_graph = update_graph_edges(graph, point_map)
                
    return updated_graph

def merge_points(points, tolerance=5):
    """
    合并误差范围内的点为代表点。
    :param points: 点列表 [(x1, y1), (x2, y2), ...]
    :param tolerance: 合并容差
    :return: 合并后的点列表、新旧点的映射
    """
    tree = KDTree(points) 
    merged_points = []
    point_map = {}
    visited = set()

    for i, point in enumerate(points):
        if i in visited:
            continue
        # 找到当前点的邻近点
        neighbors = tree.query_ball_point(point, r=tolerance)
        cluster = [points[j] for j in neighbors]
        # 计算代表点（取平均值作为代表点）
        representative = tuple(np.mean(cluster, axis=0))
        merged_points.append(representative)
        # 更新映射关系
        for j in neighbors:
            point_map[tuple(points[j])] = representative
            visited.add(j)

    return merged_points, point_map

def update_graph_edges(graph, point_map):
    """
    使用新的点映射更新图的边。
    :param graph: 原始图 (NetworkX)
    :param point_map: 点映射字典 {旧点: 新点}
    :return: 更新后的图
    """
    new_graph = nx.Graph()
    for start, end, data in graph.edges(data=True):
        new_start = point_map.get(start, start)
        new_end = point_map.get(end, end)
        if new_start != new_end:  # 避免自环
            new_graph.add_edge(new_start, new_end, **data)
    return new_graph

def find_longest_paths(graph):
    """
    寻找图中所有连通子图的最长路径
    :param graph: 输入的图对象
    :return: 一个列表，包含每个连通子图的最长路径
    """
    # 获取所有连通组件
    components = list(nx.connected_components(graph))
    
    longest_paths = []
    
    for component in components:
        # 从连通组件创建子图
        subgraph = graph.subgraph(component)
        
        # 用来存储最长路径的列表
        component_longest_paths = []

        # 计算每个节点的最长路径
        for node in subgraph.nodes():
            # 使用 BFS 计算每个节点到其他节点的最短路径
            lengths = nx.single_source_shortest_path_length(subgraph, node)
            
            # 找到最远节点
            farthest_node = max(lengths, key=lengths.get)
            
            # 获取从当前节点到最远节点的路径
            path = nx.shortest_path(subgraph, node, farthest_node)
            path_length = len(path)
            
            # 保存路径
            component_longest_paths.append(path)

        # 如果当前组件有多个最长路径，按路径长度排序
        component_longest_paths = sorted(component_longest_paths, key=len, reverse=True)
        longest_paths.extend(component_longest_paths)

    return longest_paths

def remove_path_from_graph(graph, path):
    """延迟删除路径上的节点和边"""
    nodes_to_remove = []
    edges_to_remove = []

    # 记录需要删除的边
    for i in range(len(path) - 1):
        edges_to_remove.append((path[i], path[i+1]))
    
    # 记录需要删除的节点（不包括起始和目标节点）
    for node in path[1:-1]:
        nodes_to_remove.append(node)
    
    # 在遍历完成后进行删除
    for edge in edges_to_remove:
        graph.remove_edge(*edge)
    for node in nodes_to_remove:
        graph.remove_node(node)

def dijkstra_longest_path(graph, start_node):
    """
    使用 Dijkstra 算法从一个起始节点计算最远节点路径
    :param graph: NetworkX 图对象
    :param start_node: 起始节点
    :return: 从起始节点到最远节点的路径
    """
    # 确保图没有在计算时被修改，可以使用图的副本
    graph_copy = graph.copy()
    
    try:
        lengths = nx.single_source_dijkstra_path_length(graph_copy, start_node)
    except nx.NodeNotFound:
        print(f"Node {start_node} not found in graph")
        return []
    
    # 找到最远的节点
    farthest_node = max(lengths, key=lengths.get)
    
    # 获取最远节点的路径
    path = nx.shortest_path(graph_copy, start_node, farthest_node)
    return path


def find_longest_paths_with_dijkstra(graph):
    longest_paths = []

    # 获取所有连通组件
    components = list(nx.connected_components(graph))

    for component in components:
        subgraph = graph.subgraph(component)

        # 获取节点列表进行迭代
        nodes = list(subgraph.nodes())

        for node in nodes:  # 使用列表来避免修改节点集合时出现错误
            # 计算从节点到最远节点的路径
            path = dijkstra_longest_path(subgraph, node)
            if path:
                longest_paths.append(path)
                
                # 延迟删除已计算的路径
                remove_path_from_graph(graph, path)

    return longest_paths

# 计算线段的方向（单位向量）
def get_direction(line):
    x1, y1 = line.coords[0]
    x2, y2 = line.coords[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = np.sqrt(dx**2 + dy**2)
    return dx / norm, dy / norm  # 返回单位向量

# 计算两个点之间的距离
def distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# 判断两个线段是否可以合并
def can_merge(line1, line2, tolerance=0.1, angle_threshold=np.pi / 12):
    dir1 = get_direction(line1)
    dir2 = get_direction(line2)
    end1 = line1.coords[-1]
    start2 = line2.coords[0]
    if distance(end1, start2) < tolerance:
        angle = np.arccos(np.clip(np.dot(dir1, dir2), -1.0, 1.0))
        if angle < angle_threshold:
            return True
    return False

# 使用空间索引进行线段合并
def merge_lines_with_rtree(lines, tolerance=0.1, angle_threshold=np.pi / 12):
    # 构建 R-tree 空间索引
    idx = index.Index()
    for i, line in enumerate(lines):
        # 每条线段的最小外接矩形 (bounding box)
        minx, miny, maxx, maxy = line.bounds
        idx.insert(i, (minx, miny, maxx, maxy))
    
    merged_lines = []
    merged_flags = [False] * len(lines)

    for i, line in enumerate(lines):
        if merged_flags[i]:
            continue
        merged_flags[i] = True
        current_line = line

        # 查询与当前线段相交的线段
        possible_neighbors = list(idx.intersection(line.bounds))

        for j in possible_neighbors:
            if i == j or merged_flags[j]:
                continue
            neighbor_line = lines[j]
            if can_merge(current_line, neighbor_line, tolerance, angle_threshold):
                current_line = LineString(list(current_line.coords) + list(neighbor_line.coords)[1:])
                merged_flags[j] = True
        
        # 将合并后的线段添加到结果中
        merged_lines.append(current_line)

    return merged_lines

def max_spanning_tree(graph):
    # 使用 Kruskal 算法创建最大生成树
    mst = nx.Graph()
    edges = list(graph.edges(data=True))
    edges.sort(key=lambda x: x[2]['weight'], reverse=True)  # 根据权重从大到小排序
    
    # 使用并查集（Union-Find）来检测是否形成环
    parent = {}
    rank = {}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            if rank[rootX] > rank[rootY]:
                parent[rootY] = rootX
            elif rank[rootX] < rank[rootY]:
                parent[rootX] = rootY
            else:
                parent[rootY] = rootX
                rank[rootX] += 1

    # 初始化父节点和秩
    for edge in edges:
        parent[edge[0]] = edge[0]
        parent[edge[1]] = edge[1]
        rank[edge[0]] = 0
        rank[edge[1]] = 0
    
    # 遍历所有边，并逐一加入最大生成树
    for u, v, data in edges:
        if find(u) != find(v):
            union(u, v)
            mst.add_edge(u, v, weight=data['weight'])
    
    return mst

def extract_contours_from_graph(vertices, edges):
    """
    从图数据结构中提取连通边界（多边形轮廓）。
    :param vertices: 顶点列表 [(x1, y1), (x2, y2), ...]
    :param edges: 边列表 [(start1, end1), (start2, end2), ...]，start 和 end 为顶点索引
    :return: 连通边界列表，每个边界是一个顶点索引的顺序列表
    """
    from collections import defaultdict

    # 构建邻接表
    adj_list = defaultdict(list)
    for start, end in edges:
        adj_list[start].append(end)
        adj_list[end].append(start)  # 假设图是无向的

    visited = set()  # 用于记录访问过的节点
    contours = []    # 存储所有连通边界

    def dfs(node, current_path):
        """
        深度优先搜索，用于找到连通的边界路径
        """
        visited.add(node)
        current_path.append(node)
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                dfs(neighbor, current_path)

    # 遍历所有顶点，寻找连通分量
    for vertex in range(len(vertices)):
        if vertex not in visited:
            contour = []
            dfs(vertex, contour)
            contours.append(contour)

    return contours

def process_multilinestring(multi_line_string):
    """
    处理 MultiLineString 数据，构建图并找出所有连通子图中的最长路径
    :param multi_line_string: 输入的 MultiLineString 数据
    :return: 所有连通子图中的最长路径
    """
    # 1. 构建图
    graph = build_graph_from_lines(multi_line_string)     

    print("Graph adjacency list:")
    for node, neighbors in graph.adjacency():
        print(node, "->", list(neighbors))
    # 2. 找到所有连通子图的最长路径
    # longest_paths = find_longest_paths_with_dijkstra(graph)
    # return longest_paths
    # 获取所有连通组件
    components = list(nx.connected_components(graph))
    
    longest_paths = []
    
    for component in components:
        # 从连通组件创建子图
        subgraph = graph.subgraph(component)
        vertices = subgraph.nodes()
        contours = extract_contours_from_graph(vertices, subgraph.edges())
        contours_coordinates = [[vertices[i] for i in contour] for contour in contours]
        longest_paths.append(contours_coordinates)
    return longest_paths   

def append_ridgesAndText_to_dxf(dxf_file, ridges, merged_lines, text_result):
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

    # 添加文本
    for text, x0, y0, x1, y1 in text_result:
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        height = y1 - y0  # 计算文字的高度
        if height <= 0:
            continue
        msp.add_text(text, dxfattribs={'height': height, 'color': 5}).set_placement(
            (center_x, center_y), align=TextEntityAlignment.MIDDLE_CENTER
        )

    # 保存修改后的 DXF 文件
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
        'D:/Image2CADPy/Image2CAD/potrace',  # potrace 的路径
        pbm_path,                         # 输入文件路径
        '-b', 'dxf',                      # 指定输出格式为 DXF
        '-o', dxf_path,                   # 输出文件路径
        '-z', 'minority',                 # 路径分解策略
        '-t', '5',                        # 忽略小噪点的大小
        '-a', '0.1',                        # 保留清晰的拐角
        # '-n',                             # 禁用曲线优化
        '-O', '0.1',                      # 高精度曲线优化容差
        '-u', '20',                        # 输出量化单位        
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
    
def get_text(img_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='ch') 
    result = ocr.ocr(img_path, cls=True)
    return result
    
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
            # else:
            #     print(f"Invalid polygon skipped")
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
    with ThreadPoolExecutor() as executor:
        multi_polygon = convert_to_multipolygon(polygons)        
    end_time = time.time()  
    print(f"convert_to_multipolygon Execution time: {end_time - start_time:.2f} seconds")
    start_time = end_time
    try:
        # 增加容差，减少计算量
        centerlines = Centerline(multi_polygon, 0.5) 
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
    newdxf_filename = os.path.splitext(filename)[0] + "_new.dxf"
    output_newdxf_path = os.path.join(output_folder, newdxf_filename)
    # processed = process_ridges(centerlines.geometry, 0.1, 0.5, 0.1)
    # processed = merge_with_dbscan(centerlines.geometry)
    # ocr_text_result = get_text(input_path)
    shutil.copy2(output_dxf_path, output_newdxf_path)
    text_positions = parse_hocr_optimized(output_hocrPath + ".hocr")
    append_ridgesAndText_to_dxf(output_newdxf_path, centerlines.geometry, merged_lines, [])
    end_time = time.time()  
    print(f"append dxf Execution time: {end_time - start_time:.2f} seconds")
    start_time = end_time

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

