# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from numpy import array
from scipy.spatial import Voronoi
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import exceptions
import os

class Centerline:
    """Create a centerline object.

    The ``attributes`` are copied and set as the centerline's
    attributes.

    :param input_geometry: input geometry
    :type input_geometry: :py:class:`shapely.geometry.Polygon` or
        :py:class:`shapely.geometry.MultiPolygon`
    :param interpolation_distance: densify the input geometry's
        border by placing additional points at this distance,
        defaults to 0.5 [meter]
    :type interpolation_distance: float, optional
    :raises exceptions.InvalidInputTypeError: input geometry is not
        of type :py:class:`shapely.geometry.Polygon` or
        :py:class:`shapely.geometry.MultiPolygon`
    """

    def __init__(self, input_geometry, interpolation_distance=0.5, simplify_tolerance=0.5, **attributes):
        self._input_geometry = input_geometry
        self._interpolation_distance = abs(interpolation_distance)
        self._simplify_tolerance = simplify_tolerance  # 新增简化容差参数

        if not self.input_geometry_is_valid():
            raise exceptions.InvalidInputTypeError
        # 简化多边形
        self._input_geometry = self._simplify_geometry(self._input_geometry)       
        self._min_x, self._min_y = self._get_reduced_coordinates()
        self.assign_attributes_to_instance(attributes)

        self.geometry = MultiLineString(lines=self._construct_centerline())
        
    def _simplify_geometry(self, geometry):
        """简化输入几何体，减少顶点数量"""
        if isinstance(geometry, Polygon):
            return geometry.simplify(self._simplify_tolerance, preserve_topology=True)
        elif isinstance(geometry, MultiPolygon):
            return MultiPolygon([polygon.simplify(self._simplify_tolerance, preserve_topology=True) 
                                for polygon in geometry.geoms])
        return geometry
         
    def input_geometry_is_valid(self):
        """Input geometry is of a :py:class:`shapely.geometry.Polygon`
        or a :py:class:`shapely.geometry.MultiPolygon`.

        :return: geometry is valid
        :rtype: bool
        """
        if isinstance(self._input_geometry, Polygon) or isinstance(
            self._input_geometry, MultiPolygon
        ):
            return True
        else:
            return False

    def _get_reduced_coordinates(self):
        min_x = int(min(self._input_geometry.envelope.exterior.xy[0]))
        min_y = int(min(self._input_geometry.envelope.exterior.xy[1]))
        return min_x, min_y

    def assign_attributes_to_instance(self, attributes):
        """Assign the ``attributes`` to the :py:class:`Centerline` object.

        :param attributes: polygon's attributes
        :type attributes: dict
        """
        for key in attributes:
            setattr(self, key, attributes.get(key))
            
    @staticmethod
    def merge_lines_to_polylines(linestrings):
        """
        将多个小线段合并为多段线（Polyline）
        :param linestrings: 包含多个 LineString 的列表
        :return: 合并后的 MultiLineString
        """
        # 构建图
        graph = defaultdict(list)
        for line in linestrings:
            start, end = line.coords[0], line.coords[-1]
            graph[start].append(end)
            graph[end].append(start)

        # 深度优先搜索（DFS）查找连续路径
        visited = set()
        polylines = []

        def bfs(start_node):
            queue = deque()
            queue.append(start_node)
            path = []
            while queue:
                node = queue.popleft()
                if node not in visited:
                    visited.add(node)
                    path.append(node)
                    for neighbor in graph[node]:
                        queue.append(neighbor)
            return path

        for node in graph:
            if node not in visited:
                path = bfs(node)
                if len(path) >= 2:  # 至少需要两个点才能构成线段
                    polylines.append(LineString(path))

        # 返回合并后的多段线
        return MultiLineString(polylines)

    def _construct_centerline(self):
        vertices, ridges = self._get_voronoi_vertices_and_ridges()
        linestrings = []
        
        # 设置最大并发数为 CPU 核心数的一半
        max_workers = max(1, os.cpu_count() // 2)
        # 并行处理每个 ridge
        with ThreadPoolExecutor(max_workers) as executor:
            futures = [executor.submit(self._process_ridge, ridge, vertices) for ridge in ridges]
            for future in as_completed(futures):
                linestring = future.result()
                if linestring is not None:
                    linestrings.append(linestring)

        str_tree = STRtree(linestrings)
        linestrings_indexes = str_tree.query(
            self._input_geometry, predicate="contains"
        )
        contained_linestrings = [linestrings[i] for i in linestrings_indexes]
        if len(contained_linestrings) < 2:
            raise exceptions.TooFewRidgesError

        return unary_union(contained_linestrings)  
    
    def _process_ridge(self, ridge, vertices):
        """处理单个 ridge"""
        if self._ridge_is_finite(ridge):
            starting_point = self._create_point_with_restored_coordinates(
                x=vertices[ridge[0]][0], y=vertices[ridge[0]][1]
            )
            ending_point = self._create_point_with_restored_coordinates(
                x=vertices[ridge[1]][0], y=vertices[ridge[1]][1]
            )
            return LineString((starting_point, ending_point))
        return None
         
    def _get_voronoi_vertices_and_ridges(self):
        borders = self._get_densified_borders()

        voronoi_diagram = Voronoi(borders)
        vertices = voronoi_diagram.vertices
        ridges = voronoi_diagram.ridge_vertices

        return vertices, ridges

    def _ridge_is_finite(self, ridge):
        return -1 not in ridge

    def _create_point_with_restored_coordinates(self, x, y):
        return (x + self._min_x, y + self._min_y)

    def _linestring_is_within_input_geometry(self, linestring):
        return (
            linestring.within(self._input_geometry)
            and len(linestring.coords[0]) > 1
        )

    def _get_densified_borders(self):
        polygons = self._extract_polygons_from_input_geometry()
        points = []
        for polygon in polygons:
            points += self._get_interpolated_boundary(polygon.exterior)
            if self._polygon_has_interior_rings(polygon):
                for interior in polygon.interiors:
                    points += self._get_interpolated_boundary(interior)

        return array(points)

    def _extract_polygons_from_input_geometry(self):
        if isinstance(self._input_geometry, MultiPolygon):
            return (polygon for polygon in self._input_geometry.geoms)
        else:
            return (self._input_geometry,)

    def _polygon_has_interior_rings(self, polygon):
        return len(polygon.interiors) > 0

    def _get_interpolated_boundary(self, boundary):
        line = LineString(boundary)

        first_point = self._get_coordinates_of_first_point(line)
        last_point = self._get_coordinates_of_last_point(line)

        intermediate_points = self._get_coordinates_of_interpolated_points(
            line
        )

        return [first_point] + intermediate_points + [last_point]

    def _get_coordinates_of_first_point(self, linestring):
        return self._create_point_with_reduced_coordinates(
            x=linestring.xy[0][0], y=linestring.xy[1][0]
        )

    def _get_coordinates_of_last_point(self, linestring):
        return self._create_point_with_reduced_coordinates(
            x=linestring.xy[0][-1], y=linestring.xy[1][-1]
        )

    def _get_coordinates_of_interpolated_points(self, linestring):
        intermediate_points = []
        interpolation_distance = self._interpolation_distance
        line_length = linestring.length
        while interpolation_distance < line_length:
            point = linestring.interpolate(interpolation_distance)
            reduced_point = self._create_point_with_reduced_coordinates(
                x=point.x, y=point.y
            )
            intermediate_points.append(reduced_point)
            interpolation_distance += self._interpolation_distance

        return intermediate_points

    def _create_point_with_reduced_coordinates(self, x, y):
        return (x - self._min_x, y - self._min_y)
