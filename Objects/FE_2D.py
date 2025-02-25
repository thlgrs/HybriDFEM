from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict

import gmsh
import meshio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import KDTree

from Pynite.FEModel3D import FEModel3D
from Pynite.Visualization import Renderer


class Mesh:
    """
    A 2D mesh generator using gmsh.

    Attributes:
        points_list (List[Tuple[float, float]]): List of (x, y) coordinates.
        element_type (str): Element type ("triangle" or "quad").
        element_size (float): Characteristic mesh element size.
        file (str): Output filename.
        name (str): Name of the model.
        generated (int): Counter to indicate if the mesh was generated.
        _mesh_cache (Optional[meshio.Mesh]): Cached mesh read from file.
    """

    def __init__(
        self,
        points: List[Tuple[float, float]],
        element_type: str,
        element_size: float,
        name: str = "myMesh",
    ) -> None:
        """
        Initialize the mesh object.

        Parameters:
            points: List of (x, y) points defining the geometry.
            element_type: Either "triangle" or "quad".
            element_size: Characteristic mesh element size.
            name: Name of the mesh/model.
        """
        if element_type not in ("triangle", "quad"):
            raise ValueError("element_type must be 'triangle' or 'quad'")
        self.points_list: List[Tuple[float, float]] = points
        self.element_type: str = element_type
        self.element_size: float = element_size
        self.file: str = f"{name}.msh"
        self.name: str = name
        self.generated: int = 0
        self._mesh_cache: Optional[meshio.Mesh] = None
        gmsh.initialize()  # initialize gmsh

    def clear_mesh(self) -> None:
        """
        Clear the current gmsh model and reset the generated flag and cache.
        """
        gmsh.model.remove()
        self.generated = 0
        self._mesh_cache = None

    def generate_mesh(self) -> None:
        """
        Generate the 2D mesh from the current points list.
        """
        gmsh.model.add(self.name)
        # add points to gmsh model
        point_ids = [
            gmsh.model.geo.addPoint(x, y, 0, self.element_size, tag)
            for tag, (x, y) in enumerate(self.points_list, start=1)
        ]

        # create lines between successive points (with wrap-around)
        num_points = len(point_ids)
        line_ids = [
            gmsh.model.geo.addLine(point_ids[i], point_ids[(i + 1) % num_points])
            for i in range(num_points)
        ]

        # create a closed curve loop and plane surface
        curve_loop = gmsh.model.geo.addCurveLoop(line_ids)
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])

        gmsh.model.geo.synchronize()
        if self.element_type == "quad":
            gmsh.model.mesh.setRecombine(2, surface)
        gmsh.model.mesh.generate(2)
        gmsh.write(self.file)
        self.generated += 1
        self._mesh_cache = meshio.read(self.file)

    def read_mesh(self) -> meshio.Mesh:
        """
        Read and return the mesh using meshio; uses a cached version if available.
        """
        if self._mesh_cache is None:
            self._mesh_cache = meshio.read(self.file)
        return self._mesh_cache

    def nodes(self) -> np.ndarray:
        """
        Return the 2D coordinates of mesh nodes.
        """
        mesh = self.read_mesh()
        return mesh.points[:, :2]

    def lines(self) -> List[np.ndarray]:
        """
        Return connectivity of line elements.
        """
        mesh = self.read_mesh()
        return mesh.cells_dict.get("line", [])

    def elements(self) -> List[np.ndarray]:
        """
        Return connectivity of elements (triangles or quadrangles).
        """
        mesh = self.read_mesh()
        if self.element_type == "quad":
            return mesh.cells_dict.get("quad", [])
        else:
            return mesh.cells_dict.get("triangle", [])

    def add_point(self, x: float, y: float, regen: bool = False) -> None:
        """
        Append a new point to the geometry. If regen is True, clear and regenerate the mesh.
        """
        if self.generated:
            self.clear_mesh()
        self.points_list.append((x, y))
        if regen:
            self.generate_mesh()

    def change_size(self, new_size: float, regen: bool = False) -> None:
        """
        Change the mesh element size. Clears and regenerates the mesh if already generated.
        """
        if self.generated:
            self.clear_mesh()
        self.element_size = new_size
        if regen:
            self.generate_mesh()

    def change_type(self, new_type: str, regen: bool = False) -> None:
        """
        Change the element type ("triangle" or "quad"). Clears and regenerates the mesh if already generated.
        """
        if new_type not in ("triangle", "quad"):
            raise ValueError("element_type must be 'triangle' or 'quad'")
        if self.generated:
            self.clear_mesh()
        self.element_type = new_type
        if regen:
            self.generate_mesh()

    def plot(self, save_path: Optional[str] = None) -> None:
        """
        Plot the mesh using matplotlib.

        Parameters:
            save_path: If provided, the plot is saved to this file.
        """
        mesh = self.read_mesh()
        points = mesh.points
        lines_list: List[List[np.ndarray]] = []
        for cell_type, elements in mesh.cells_dict.items():
            if cell_type in ("line", "triangle", "quad"):
                for element in elements:
                    n = len(element)
                    for i in range(n):
                        start = element[i]
                        end = element[(i + 1) % n]
                        lines_list.append([points[start][:2], points[end][:2]])
        line_segments = LineCollection(lines_list, linewidths=0.5, colors="black")
        fig, ax = plt.subplots()
        ax.add_collection(line_segments)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title("2D Mesh Visualization")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def _node_tag_to_index(self) -> Dict[int, int]:
        """
        Build a mapping from gmsh node tag to index in the node coordinate array.
        """
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        tags_array = np.array(node_tags)
        return {tag: idx for idx, tag in enumerate(tags_array)}

    def find_points(
        self,
        target_coords: Union[Tuple[float, float], List[Tuple[float, float]]],
        tolerance: float = 1e-6,
    ) -> List[int]:
        """
        Find mesh node tags whose coordinates match target_coords within a tolerance.

        Parameters:
            target_coords: A tuple (x, y) or list of such tuples.
            tolerance: Tolerance for matching coordinates.

        Returns:
            List of matching node tags.
        """
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)[:, :2]
        if isinstance(target_coords[0], (float, int)):
            target_coords = [target_coords]  # type: ignore
        matching_points: List[int] = []
        for target in target_coords:
            for tag, coord in zip(node_tags, node_coords):
                if np.allclose(coord, target, atol=tolerance):
                    matching_points.append(tag)
        return matching_points

    def find_path(
        self,
        coord1: Tuple[float, float],
        coord2: Tuple[float, float],
        tolerance: float = 1e-6,
    ) -> Optional[List[int]]:
        """
        Find a path (sequence of node tags) between two coordinates using BFS on the connectivity graph.

        Parameters:
            coord1: Starting coordinate.
            coord2: Ending coordinate.
            tolerance: Tolerance for searching nearest nodes.

        Returns:
            A list of node tags representing the path, or None if not found.
        """
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)[:, :2]
        kdtree = KDTree(node_coords)
        start_idx = kdtree.query(coord1, k=1)[1]
        end_idx = kdtree.query(coord2, k=1)[1]
        start_node = node_tags[start_idx]
        end_node = node_tags[end_idx]

        # obtain connectivity for 1D elements (lines)
        _, _, element_tags = gmsh.model.mesh.getElements(1)
        if not element_tags:
            return None
        line_connectivity = np.array(element_tags[0]).reshape(-1, 2)
        graph: Dict[int, List[int]] = {}
        for n1, n2 in line_connectivity:
            graph.setdefault(n1, []).append(n2)
            graph.setdefault(n2, []).append(n1)

        def bfs_path(
            graph: Dict[int, List[int]], start: int, end: int
        ) -> Optional[List[int]]:
            queue = deque([(start, [start])])
            visited = set()
            while queue:
                current, path = queue.popleft()
                if current == end:
                    return path
                if current in visited:
                    continue
                visited.add(current)
                for neighbor in graph.get(current, []):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
            return None

        return bfs_path(graph, start_node, end_node)

    def find_elements(
        self, coord: Tuple[float, float], tolerance: float = 1e-6
    ) -> List[int]:
        """
        Find element indices (for triangles and quadrangles) that contain the given coordinate.

        Parameters:
            coord: Coordinate to test.
            tolerance: Tolerance for geometric tests.

        Returns:
            List of element indices (starting at 1) that contain the coordinate.
        """
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)[:, :2]
        mapping = self._node_tag_to_index()
        surface_dim = 2
        _, _, elem_nodes = gmsh.model.mesh.getElements(surface_dim)
        triangle_nodes: np.ndarray = np.array([])
        quad_nodes: np.ndarray = np.array([])
        if elem_nodes:
            if elem_nodes[0]:
                triangle_nodes = np.array(elem_nodes[0]).reshape(-1, 3)
            if len(elem_nodes) > 1 and elem_nodes[1]:
                quad_nodes = np.array(elem_nodes[1]).reshape(-1, 4)

        def is_point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
            v0 = c - a
            v1 = b - a
            v2 = p - a
            dot00 = np.dot(v0, v0)
            dot01 = np.dot(v0, v1)
            dot02 = np.dot(v0, v2)
            dot11 = np.dot(v1, v1)
            dot12 = np.dot(v1, v2)
            denom = dot00 * dot11 - dot01 * dot01
            if abs(denom) < tolerance:
                return False
            inv_denom = 1 / denom
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom
            return (u >= -tolerance) and (v >= -tolerance) and (u + v <= 1 + tolerance)

        def is_point_in_quad(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
            # check by splitting the quad into two triangles
            return is_point_in_triangle(p, a, b, d) or is_point_in_triangle(p, b, c, d)

        matching_elements: List[int] = []
        # check triangles
        for i, nodes in enumerate(triangle_nodes):
            try:
                indices = [mapping[node] for node in nodes]
            except KeyError:
                continue
            a, b, c = node_coords[indices]
            if is_point_in_triangle(np.array(coord), a, b, c):
                matching_elements.append(i + 1)
        offset = len(triangle_nodes)
        # check quadrangles
        for i, nodes in enumerate(quad_nodes):
            try:
                indices = [mapping[node] for node in nodes]
            except KeyError:
                continue
            a, b, c, d = node_coords[indices]
            if is_point_in_quad(np.array(coord), a, b, c, d):
                matching_elements.append(offset + i + 1)
        return matching_elements


@dataclass
class Material:
    E: float   # Young's modulus [Pa]
    nu: float  # Poisson's ratio
    rho: float # Density [kg/m^3]


class FE_2D:
    def __init__(self, mesh: Mesh, material: Material) -> None:
        """
        Initialize the finite element 2D model using the given mesh and material properties.
        """
        self.mesh = meshio.read(mesh.file)
        self.model = FEModel3D()

        # add nodes to the model
        for i, point in enumerate(self.mesh.points):
            x, y, z = point
            self.model.add_node(f'N{i+1}', x, y, z)

        # define material properties (e.g. for steel)
        G = material.E / (2 * (1 + material.nu))  # shear modulus
        self.model.add_material('Material', material.E, G, material.nu, material.rho)

        # define section properties (e.g. rectangular section)
        self.model.add_section('Section', 0.3, 0.5, 1, 1)

        element_count = 1
        for cell in self.mesh.cells:
            if cell.type == 'line':
                for element in cell.data:
                    n1, n2 = element
                    self.model.add_member(f'M{element_count}', f'N{n1+1}', f'N{n2+1}', 'Material', 'Section')
                    element_count += 1
            elif cell.type == 'triangle':
                for element in cell.data:
                    n1, n2, n3 = element
                    self.model.add_plate(f'P{element_count}', f'N{n1+1}', f'N{n2+1}', f'N{n3+1}', None, 0.1, 'Material')
                    element_count += 1
            elif cell.type == 'quad':
                for element in cell.data:
                    n1, n2, n3, n4 = element
                    self.model.add_quad(f'Q{element_count}', f'N{n1+1}', f'N{n2+1}', f'N{n3+1}', f'N{n4+1}', 0.1, 'Material')
                    element_count += 1

    def define_support(self, node_id: Union[int, str], support_conditions: Tuple[bool, bool, bool, bool, bool, bool]) -> None:
        """
        Define support conditions for a given node.

        Parameters:
            node_id: The node identifier (int or str).
            support_conditions: A tuple of six booleans (UX, UY, UZ, RX, RY, RZ).
        """
        self.model.def_support(f'N{node_id}', *support_conditions)

    def add_nodal_load(self, node_id: Union[int, str], load: Tuple[float, float, float, float, float, float]) -> None:
        """
        Add a nodal load to a given node.

        Parameters:
            node_id: The node identifier.
            load: A tuple of forces and moments (FX, FY, FZ, MX, MY, MZ).
        """
        FX, FY, FZ, MX, MY, MZ = load
        if FX:
            self.model.add_node_load(f'N{node_id}', 'FX', FX)
        if FY:
            self.model.add_node_load(f'N{node_id}', 'FY', FY)
        if FZ:
            self.model.add_node_load(f'N{node_id}', 'FZ', FZ)
        if MX:
            self.model.add_node_load(f'N{node_id}', 'MX', MX)
        if MY:
            self.model.add_node_load(f'N{node_id}', 'MY', MY)
        if MZ:
            self.model.add_node_load(f'N{node_id}', 'MZ', MZ)

    def analyze(self) -> None:
        """
        Perform the analysis.
        """
        self.model.analyze()

    def get_node_displacement(self, node_id: Union[int, str]) -> Tuple[float, float, float]:
        """
        Get the displacement of a node.

        Returns:
            A tuple (DX, DY, DZ) representing the nodal displacements.
        """
        node = self.model.nodes[f"N{node_id}"]
        return node.DX, node.DY, node.DZ

    def find_node_id_by_coordinates(self, x: float, y: float, z: float, tolerance: float = 1e-6) -> Optional[str]:
        """
        Find the ID of the node closest to the given coordinates (x, y, z).

        Returns:
            The node ID if found within the specified tolerance; otherwise, None.
        """
        min_distance = float("inf")
        closest_node_id: Optional[str] = None

        for node_id, node in self.model.nodes.items():
            distance = np.sqrt((node.X - x) ** 2 + (node.Y - y) ** 2 + (node.Z - z) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_node_id = node_id

        return closest_node_id if min_distance <= tolerance else None

    def visualize(
        self,
        annotation_size: float = 0.1,
        deformed_shape: bool = True,
        deformed_scale: float = 1000,
        render_loads: bool = True,
        show_nodes: bool = False,
    ) -> None:
        """
        Visualize the finite element model using PyNite's Renderer.

        Parameters:
            annotation_size: Size of the annotations in the plot.
            deformed_shape: Whether to render the deformed shape.
            deformed_scale: Scale factor for the deformed shape.
            render_loads: Whether to render the applied loads.
            show_nodes: Whether to display node names.
        """
        renderer = Renderer(self.model)
        renderer.annotation_size = annotation_size
        renderer.deformed_shape = deformed_shape
        renderer.deformed_scale = deformed_scale
        renderer.render_loads = render_loads
        renderer.show_nodes = show_nodes
        renderer.render_model()
