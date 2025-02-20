from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict

import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial import KDTree
from skfem import ElementTriP1, ElementTriP2, ElementQuad1, ElementQuad2, Basis, FacetBasis, asm, \
    solve
from skfem.helpers import dot
from skfem.io import from_meshio
from skfem.models.elasticity import linear_elasticity


class Mesh:
    """
    A 2d mesh generator using gmsh.

    Attributes:
        points_list (List[Tuple[float, float]]): List of (x, y) coordinates.
        type (str): Element type ("triangle" or "quad").
        element_size (float): Mesh element size.
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
            points: list of (x, y) points defining the geometry.
            element_type: either "triangle" or "quad".
            element_size: characteristic mesh element size.
            name: name of the mesh/model.
        """
        if element_type not in ("triangle", "quad"):
            raise ValueError("element_type must be 'triangle' or 'quad'")
        self.points_list = points
        self.type = element_type
        self.element_size = element_size
        self.file = name + ".msh"
        self.name = name
        self.generated = 0
        self._mesh_cache: Optional[meshio.Mesh] = None
        gmsh.initialize()  # initialize gmsh

    def __enter__(self) -> "Mesh":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        gmsh.finalize()

    def clear_mesh(self) -> None:
        """
        Clear the current gmsh model and reset the generated flag and cache.
        """
        gmsh.model.remove()
        self.generated = 0
        self._mesh_cache = None

    def generate_mesh(self) -> None:
        """
        Generate the 2d mesh from the current points list.
        """
        gmsh.model.add(self.name)
        point_ids = []
        for i, (x, y) in enumerate(self.points_list, start=1):
            pid = gmsh.model.geo.addPoint(x, y, 0, self.element_size, i)
            point_ids.append(pid)

        num_points = len(point_ids)
        line_ids = []
        for i in range(num_points):
            start = point_ids[i]
            end = point_ids[(i + 1) % num_points]  # wrap-around
            lid = gmsh.model.geo.addLine(start, end)
            line_ids.append(lid)

        curve_loop = gmsh.model.geo.addCurveLoop(line_ids)
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])

        gmsh.model.geo.synchronize()
        if self.type == "quad":
            gmsh.model.mesh.setRecombine(2, surface)
        gmsh.model.mesh.generate(2)
        gmsh.write(self.file)
        self.generated += 1
        self._mesh_cache = meshio.read(self.file)

    def read_mesh(self) -> meshio.Mesh:
        """
        Read and return the mesh using meshio; uses cached version if available.
        """
        if self._mesh_cache is None:
            self._mesh_cache = meshio.read(self.file)
        return self._mesh_cache

    def nodes(self) -> np.ndarray:
        """
        Return the 2d coordinates of mesh nodes.
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
        if self.type == "quad":
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
        self.type = new_type
        if regen:
            self.generate_mesh()

    def plot(self, save_path: Optional[str] = None) -> None:
        """
        Plot the mesh using matplotlib.

        Parameters:
            save_path: if provided, the plot is saved to this file.
        """
        mesh = self.read_mesh()
        points = mesh.points
        lines_list = []
        for cell_type, elements in mesh.cells_dict.items():
            if cell_type in ["line", "triangle", "quad"]:
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
        plt.title("2D Mesh Visualization")
        plt.xlabel("X")
        plt.ylabel("Y")
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
            target_coords: a tuple (x, y) or list of such tuples.
            tolerance: tolerance for matching coordinates.

        Returns:
            List of matching node tags.
        """
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)[:, :2]
        if isinstance(target_coords[0], (float, int)):
            target_coords = [target_coords]
        matching_points = []
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
        Find a path (sequence of node tags) between two coordinates using a bfs on the connectivity graph.

        Parameters:
            coord1: starting coordinate.
            coord2: ending coordinate.
            tolerance: tolerance for searching nearest nodes.

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

        # get connectivity for 1d elements (lines)
        _, _, element_tags = gmsh.model.mesh.getElements(1)
        if not element_tags:
            return None
        line_connectivity = np.array(element_tags[0]).reshape(-1, 2)
        graph: Dict[int, List[int]] = {}
        for n1, n2 in line_connectivity:
            graph.setdefault(n1, []).append(n2)
            graph.setdefault(n2, []).append(n1)

        def bfs_path(graph: Dict[int, List[int]], start: int, end: int) -> Optional[List[int]]:
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
            coord: coordinate to test.
            tolerance: tolerance for geometric tests.

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
        if len(elem_nodes) > 0:
            if elem_nodes[0]:
                triangle_nodes = np.array(elem_nodes[0]).reshape(-1, 3)
            if len(elem_nodes) > 1 and elem_nodes[1]:
                quad_nodes = np.array(elem_nodes[1]).reshape(-1, 4)

        def is_point_in_triangle(p, a, b, c):
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

        def is_point_in_quad(p, a, b, c, d):
            # check by splitting quad into two triangles
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
        for i, nodes in enumerate(quad_nodes):
            try:
                indices = [mapping[node] for node in nodes]
            except KeyError:
                continue
            a, b, c, d = node_coords[indices]
            if is_point_in_quad(np.array(coord), a, b, c, d):
                matching_elements.append(offset + i + 1)
        return matching_elements

    def __del__(self) -> None:
        """
        Finalize gmsh. (If __exit__ is used via the context manager, this may be called redundantly.)
        """
        try:
            gmsh.finalize()
        except Exception:
            pass


@dataclass
class Material:
    E: float
    nu: float
    plane_condition: str


class FE_2D:
    def __init__(self, gmsh_file, material, element_type="TriP1"):
        """Initialize finite element model with mesh, material properties, and element type."""
        self.mesh, self.element = self.load_mesh_and_element(gmsh_file, element_type)
        self.basis = Basis(self.mesh, self.element)

        # Material properties
        self.E = material.E
        self.nu = material.nu
        self.lambda_ = material.E * material.nu / ((1 + material.nu) * (1 - 2 * material.nu))  # Lamé parameter λ
        self.mu = material.E / (2 * (1 + material.nu))  # Lamé parameter μ

        # Matrices and load vector
        self.K = None
        self.f = np.zeros((self.basis.N,))
        self.dofs_fixed = {}

        self.assemble_rigidity_matrix()

    def load_mesh_and_element(self, gmsh_file, element_type):
        """Load a 2D triangular or quadrilateral mesh and select the appropriate element."""
        meshio_mesh = meshio.read(gmsh_file)
        mesh = from_meshio(meshio_mesh)

        # Select the finite element type
        if element_type == "TriP1":
            element = ElementTriP1()
        elif element_type == "TriP2":
            element = ElementTriP2()
        elif element_type == "Quad1":
            element = ElementQuad1()
        elif element_type == "Quad2":
            element = ElementQuad2()
        else:
            raise ValueError("Unsupported element type. Choose from: TriP1, TriP2, Quad1, Quad2.")

        return mesh, element

    def assemble_rigidity_matrix(self):
        """Assemble the global stiffness matrix."""
        self.K = asm(linear_elasticity(self.lambda_, self.mu), self.basis)

    def find_nearest_node(self, point):
        """Find the nearest node to a given (x, y) coordinate."""
        distances = np.linalg.norm(self.mesh.p.T - np.array(point), axis=1)
        return np.argmin(distances)  # Return index of the closest node

    def find_nodes_by_axis(self, axis, value, tol=1e-6):
        """Find all nodes where x ≈ value or y ≈ value."""
        if axis == 'x':
            indices = np.where(np.isclose(self.mesh.p[0], value, atol=tol))[0]
        elif axis == 'y':
            indices = np.where(np.isclose(self.mesh.p[1], value, atol=tol))[0]
        else:
            raise ValueError("Axis must be 'x' or 'y'")
        return indices

    def apply_fixed_support(self, node_indices):
        """Applies a fixed (clamped) support at given node indices."""
        self.apply_displacement_bc(node_indices, values=(0.0, 0.0))

    def apply_hinged_support(self, node_indices):
        """Applies a hinged (pinned) support: u_y = 0, u_x is free."""
        self.apply_displacement_bc(node_indices, values=(None, 0.0))

    def apply_roller_support(self, node_indices):
        """Applies a roller support: u_y = 0 (fixed vertically), u_x is free."""
        self.apply_displacement_bc(node_indices, values=(None, 0.0))

    def apply_guided_support(self, node_indices):
        """Applies a guided (sliding) support: u_x = 0, u_y is free."""
        self.apply_displacement_bc(node_indices, values=(0.0, None))

    def apply_displacement_bc(self, node_indices, values):
        """Applies Dirichlet BCs (prescribed displacement)."""
        for node in node_indices:
            self.dofs_fixed[f'u^1_{node}'] = node * 2 if values[0] is not None else None
            self.dofs_fixed[f'u^2_{node}'] = node * 2 + 1 if values[1] is not None else None

            if values[0] is not None:
                self.f[node * 2] = values[0]
            if values[1] is not None:
                self.f[node * 2 + 1] = values[1]

    def apply_force_x(self, node_indices=None, traction_func=None, body_force_func=None, force_value=0.0):
        """
        Apply force only in the X direction.

        Parameters:
        node_indices (list, optional): List of node indices where force is applied.
        traction_func (function, optional): Function defining traction in X direction.
        body_force_func (function, optional): Function defining body force in X direction.
        force_value (float, optional): Constant force value for node-based application.
        """

        # ✅ 1. Apply Point Forces in X
        if node_indices is not None:
            for node in node_indices:
                self.f[node * 2] += force_value  # Apply Fx

        # ✅ 2. Apply Surface Traction Forces in X (Neumann BC)
        if traction_func is not None:
            fbasis = FacetBasis(self.mesh, self.element)
            traction_values = fbasis.interpolate(lambda x: np.array([traction_func(x), np.zeros_like(x[0])])).flatten()

            L = asm(lambda u, v: dot(traction_values, v), fbasis)
            self.f += L

        # ✅ 3. Apply Body Forces in X (Distributed Loads)
        if body_force_func is not None:
            f_ext = self.basis.interpolate(lambda x: np.array([body_force_func(x), np.zeros_like(x[0])])).flatten()
            self.f += f_ext

    def apply_force_y(self, node_indices=None, traction_func=None, body_force_func=None, force_value=0.0):
        """
        Apply force only in the Y direction.

        Parameters:
        node_indices (list, optional): List of node indices where force is applied.
        traction_func (function, optional): Function defining traction in Y direction.
        body_force_func (function, optional): Function defining body force in Y direction.
        force_value (float, optional): Constant force value for node-based application.
        """

        # ✅ 1. Apply Point Forces in Y
        if node_indices is not None:
            for node in node_indices:
                self.f[node * 2 + 1] += force_value  # Apply Fy

        # ✅ 2. Apply Surface Traction Forces in Y (Neumann BC)
        if traction_func is not None:
            fbasis = FacetBasis(self.mesh, self.element)
            traction_values = fbasis.interpolate(lambda x: np.array([np.zeros_like(x[0]), traction_func(x)])).flatten()

            L = asm(lambda u, v: dot(traction_values, v), fbasis)
            self.f += L

        # ✅ 3. Apply Body Forces in Y (Gravity, Pressure)
        if body_force_func is not None:
            f_ext = self.basis.interpolate(lambda x: np.array([np.zeros_like(x[0]), body_force_func(x)])).flatten()
            self.f += f_ext

    def solve_system(self):
        """Solves the finite element system for displacement."""
        I = self.basis.complement_dofs(self.dofs_fixed)
        u = np.zeros_like(self.f)
        u[I] = solve(self.K[I][:, I], self.f[I])
        return u

# class FE_2D:
#    def __init__(self, mesh:Mesh, material):
#        self.mesh = from_meshio(meshio.read(mesh.filename))
#        self._mesh = mesh
#        self.material = material
#        self.basis = None
#        # KU = F linear
#        self.K = None
#        self.F = None
#        self.U = None
#
#    # =================================================================
#    # Core FEM Methods
#    # =================================================================
#    def initialize(self,order=1):
#        element_type = self._mesh.element_type
#        """Initialize basis functions and material properties"""
#        if element_type == "tri":
#            if order == 1:
#                self.basis = skfem.Basis(self.mesh, skfem.ElementTriP1())
#            elif order == 2:
#                self.basis = skfem.Basis(self.mesh, skfem.ElementTriP2())
#            else: raise ValueError("Unsupported order. Use 1 or 2")
#        elif element_type == "quad":
#            if order == 1:
#                self.basis = skfem.Basis(self.mesh, skfem.ElementQuad1())
#            elif order == 2:
#                self.basis = skfem.Basis(self.mesh, skfem.ElementQuad2())
#            else:
#                raise ValueError("Unsupported order. Use 1 or 2")
#        else:
#            raise ValueError("Unsupported element type. Use 'tri' or 'quad'.")
#
#        # Material properties
#        self.D = self._constitutive_matrix()
#
#
#    def assemble_linear_elasticity(self):
#        from skfem.helpers import eye, trace, sym_grad, ddot
#        from skfem.models import lame_parameters
#
#        lam, mu = lame_parameters(self.material.E, self.material.nu)
#
#        def C(T):
#            return 2. * mu * T + lam * eye(trace(T), T.shape[0])
#
#        @skfem.BilinearForm
#        def weakform(u, v, w):
#            return ddot(C(sym_grad(u)), sym_grad(v))
#
#        self.K = weakform.assemble(self.basis)
#
#
#    def apply_dirichlet(self, dirichlet_bc):
#        """Apply Dirichlet boundary conditions"""
#        dirichlet_dofs = []
#        dirichlet_values = []
#        for node, displacement in dirichlet_bc.items():
#            for i, disp in enumerate(displacement):
#                dof = self.basis.get_dofs().nodal[i, node]
#                dirichlet_dofs.append(dof)
#                dirichlet_values.append(disp)
#
#        self.K, self.F = skfem.condense(self.K, self.F, D=dirichlet_dofs, x=dirichlet_values)
#
#
#    def apply_neumann(self, neumann_bc):
#        """Apply Neumann boundary conditions (external forces)."""
#        for node, force in neumann_bc.items():
#            for i, f in enumerate(force):
#                self.F[self.basis.nodal_dofs[i, node]] += f
#
#        self.K, self.F = skfem.condense(self.K, self.F)
#
#    def solve(self):
#        """Solve the system KU = F"""
#        self.U = spsolve(self.K, self.F)
#
#    # =================================================================
#    # Post-Processing Methods
#    # =================================================================
#    def compute_stresses(self):
#        """Compute stresses at integration points"""
#        stress_basis = skfem.Basis(self.mesh, self.basis.elem, intorder=2)
#        stress = np.zeros((3, stress_basis.N))  # σ_xx, σ_yy, σ_xy
#
#        for i in range(3):
#            @skfem.BilinearForm
#            def stress_form(u, v, w):
#                return self.D[i, :] @ skfem.helpers.grad(u) * v
#
#            stress[i] = skfem.asm(stress_form, stress_basis, U=self.U)
#
#        return stress
#
#    def plot_solution(self):
#        """Plot displacement field"""
#        skfem.visuals.matplotlib.plot(self.basis, self.U, shading='gouraud')
#        plt.show()
#
#    # =================================================================
#    # Helper Methods
#    # =================================================================
#    def _constitutive_matrix(self):
#        """Return material D-matrix (plane stress/strain)"""
#        E = self.material.E
#        nu = self.material.nu
#
#        if self.material.plane_condition == "stress":
#            factor = E / (1 - nu**2)
#            return factor * np.array([
#                [1, nu, 0],
#                [nu, 1, 0],
#                [0, 0, (1 - nu) / 2]
#            ])
#        else:  # Plane strain
#            factor = E / ((1 + nu) * (1 - 2 * nu))
#            return factor * np.array([
#                [1 - nu, nu, 0],
#                [nu, 1 - nu, 0],
#                [0, 0, (1 - 2 * nu) / 2]
#            ])
#
