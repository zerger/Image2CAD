import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.strtree import STRtree

class OptimizedRidgeExtractor:
    def __init__(self, polygons, max_distance=None):
        """
        初始化脊线提取器。
        :param polygons: 多边形列表，每个多边形为 shapely.geometry.Polygon 对象。
        :param max_distance: 最大间距（两条平行边之间的距离限制，单位：与多边形坐标一致）。
        """
        self.polygons = polygons
        self.max_distance = max_distance
        self.edges = self._extract_all_edges()
        # 确保空间索引中仅存储有效的 LineString
        self.spatial_index = STRtree([edge for edge in self.edges if isinstance(edge, LineString)])

    def _extract_all_edges(self):
        """
        提取所有多边形的边。
        :return: 所有边的列表，每条边为 shapely.geometry.LineString。
        """
        edges = []
        for polygon in self.polygons:
            coords = list(polygon.exterior.coords)
            edges.extend([LineString([coords[i], coords[i + 1]]) for i in range(len(coords) - 1)])
        return edges

    def is_parallel(self, edge1, edge2, tolerance=5):
        """
        判断两条边是否大致平行。
        :param edge1: 第一个边，shapely.geometry.LineString。
        :param edge2: 第二个边，shapely.geometry.LineString。
        :param tolerance: 平行容差（角度，单位：度）。
        :return: 布尔值，True 表示平行。
        """
        vec1 = np.array(edge1.coords[1]) - np.array(edge1.coords[0])
        vec2 = np.array(edge2.coords[1]) - np.array(edge2.coords[0])
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        cos_theta = np.dot(vec1, vec2)
        angle = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi
        return abs(angle) < tolerance or abs(angle - 180) < tolerance

    def calculate_midline(self, edge1, edge2):
        """
        计算两条平行边的中线。
        :param edge1: 第一个边，shapely.geometry.LineString。
        :param edge2: 第二个边，shapely.geometry.LineString。
        :return: shapely.geometry.LineString 表示的中线。
        """
        mid1 = np.mean(edge1.coords, axis=0)
        mid2 = np.mean(edge2.coords, axis=0)
        return LineString([mid1, mid2])

    def extract_ridges(self):
        """
        提取多边形之间的脊线。
        :return: 脊线列表，每条脊线为 shapely.geometry.LineString 对象。
        """
        ridges = []
        for edge in self.edges:
            # 检查 edge 是否为有效几何对象
            if not isinstance(edge, LineString):
                continue

            # 从空间索引中查询候选边
            candidates = self.spatial_index.query(edge.buffer(self.max_distance))
            for candidate in candidates:
                # 检查候选边是否为有效几何对象
                if not isinstance(candidate, LineString):
                    continue

                # 排除自身
                if edge.equals(candidate):
                    continue

                # 检查是否平行并处理逻辑
                if self.is_parallel(edge, candidate, tolerance=15):
                    distance = edge.distance(candidate)
                    if self.max_distance is None or distance <= self.max_distance:
                        ridge = self.calculate_midline(edge, candidate)
                        ridges.append(ridge)

        return ridges


# 示例使用
if __name__ == "__main__":
    # 示例多边形数据
    poly1 = Polygon([(0, 0), (4, 0), (4, 2), (0, 2)])
    poly2 = Polygon([(5, 0), (9, 0), (9, 2), (5, 2)])
    poly3 = Polygon([(10, 1), (14, 1), (14, 3), (10, 3)])

    # 初始化提取器（设置最大间距为 2.5）
    extractor = OptimizedRidgeExtractor([poly1, poly2, poly3], max_distance=2.5)

    # 提取脊线
    ridges = extractor.extract_ridges()

    # 输出脊线
    for idx, ridge in enumerate(ridges):
        print(f"Ridge {idx + 1}: {ridge}")
