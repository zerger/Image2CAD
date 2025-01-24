import numpy as np
from shapely.geometry import MultiLineString, LineString
from scipy.spatial import Voronoi
import networkx as nx

def _generate_voronoi_from_lines(multi_line_string):
    """
    从 MultiLineString 数据生成 Voronoi 图。
    :param multi_line_string: 输入的 MultiLineString 数据
    :return: 生成的 Voronoi 图对象
    """
    # 获取所有线条的坐标集合
    all_coords = []
    for line in multi_line_string.geoms:
        all_coords.extend(list(line.coords))  # 合并所有线条的坐标
    
    # 使用所有线条的端点生成 Voronoi 图
    vor = Voronoi(np.array(all_coords))  # 创建 Voronoi 图
    return vor

def _graph_from_voronoi(vor):
    """
    从 Voronoi 图生成图数据结构。
    :param vor: Voronoi 图对象
    :return: 图的 NetworkX 对象
    """
    graph = nx.Graph()
    
    # 遍历 Voronoi 的每一条边
    for ridx, r in enumerate(vor.ridge_vertices):
        # 检查每条边是否有效
        if r[0] >= 0 and r[1] >= 0:
            graph.add_edge(vor.vertices[r[0]], vor.vertices[r[1]], index=ridx)
    
    return graph

def _get_end_nodes(graph):
    """
    获取图中的端点，即只有一个连接边的节点。
    :param graph: 输入的图数据结构
    :return: 端点的列表
    """
    end_nodes = []
    for node, degree in graph.degree():
        if degree == 1:  # 端点是只有一个邻居的节点
            end_nodes.append(node)
    return end_nodes

def _get_longest_paths(end_nodes, graph, max_paths=5):
    """
    从多个端点出发，找到最长的路径。
    :param end_nodes: 端点列表
    :param graph: 图数据结构
    :param max_paths: 返回最多的路径数量
    :return: 长度最长的路径列表
    """
    longest_paths = []
    
    for end_node in end_nodes:
        # 使用 BFS 寻找从端点出发的最长路径
        paths = list(nx.single_source_shortest_path_length(graph, end_node).values())
        
        # 按路径长度排序
        longest_paths.append(sorted(paths, reverse=True)[:max_paths])
    
    return longest_paths

def _get_least_curved_path(longest_paths, vertices):
    """
    从多个最长路径中选出曲率最小的路径。
    :param longest_paths: 各个端点的最长路径
    :param vertices: Voronoi 图的顶点
    :return: 最小曲率的路径
    """
    def curvature(path, vertices):
        """
        计算路径的曲率。可以使用简单的离散曲率计算方法。
        :param path: 路径点列表
        :param vertices: Voronoi 图的顶点
        :return: 曲率
        """
        total_curvature = 0
        for i in range(1, len(path) - 1):
            p0, p1, p2 = vertices[path[i-1]], vertices[path[i]], vertices[path[i+1]]
            # 计算简单的角度差值来估算曲率
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) - np.arctan2(p0[1] - p1[1], p0[0] - p1[0])
            total_curvature += abs(angle)
        return total_curvature
    
    least_curved_path = None
    min_curvature = float('inf')
    
    for path in longest_paths:
        curv = curvature(path, vertices)
        if curv < min_curvature:
            min_curvature = curv
            least_curved_path = path
    
    return least_curved_path

def process_multilinestring(multi_line_string, max_paths=5, tolerance=0.1):
    """
    处理 MultiLineString 数据，提取每个 LineString 的脊线。
    :param multi_line_string: 输入的 MultiLineString 数据
    :param max_paths: 最大路径数量
    :param tolerance: 距离容忍度
    :return: 最小曲率的脊线
    """
    # 生成 Voronoi 图
    vor = _generate_voronoi_from_lines(multi_line_string)
    
    # 从 Voronoi 图中提取图数据结构
    graph = _graph_from_voronoi(vor)
    
    # 获取端点
    end_nodes = _get_end_nodes(graph)
    
    # 获取最长路径
    longest_paths = _get_longest_paths(end_nodes, graph, max_paths)
    
    # 获取最小曲率路径
    least_curved_path = _get_least_curved_path(longest_paths, vor.vertices)
    
    return least_curved_path
