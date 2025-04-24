from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Callable

import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial import KDTree
from sfepy.discrete import (FieldVariable, Integral, Equation, Equations, Problem, Conditions)
from sfepy.discrete import Material as sfepyMaterial
from sfepy.discrete.conditions import EssentialBC
from sfepy.discrete.fem import FEDomain, Field
from sfepy.discrete.fem import Mesh as sfepyMesh
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.terms import Term
from sfepy.terms.terms import Terms


class Mesh:

    def __init__(self, points: List[Tuple[float, float]], element_type: str, element_size: float, name: str = "myMesh"):
        """
        Initialize, modify and generate a mesh object.
        Args:
            points: list of points forming the edge of the mesh
            element_type: "triangle", "quad", "tri", "quadrilateral"
            element_size: size of elements
            name: name of the mesh
        """
        if element_type not in ("triangle", "quad", "tri", "quadrilateral"):
            raise ValueError("element_type must be 'triangle'/'tri' or 'quadrilateral'/'quad'")
        self.points_list: List[Tuple[float, float]] = points
        self.element_type = 'triangle' if element_type in ('triangle', 'tri') else 'quad'
        self.element_size: float = element_size
        self.file: str = f"{name}.msh"
        self.name: str = name
        self.generated: int = 0
        gmsh.initialize()  # initialize gmsh

    def clear_mesh(self) -> None:
        gmsh.model.remove()
        self.generated = 0

    def generate_mesh(self) -> None:
        gmsh.model.add(self.name)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # compatibility with sfepy
        point_tags = [gmsh.model.geo.addPoint(x, y, 0, meshSize=self.element_size) for (x, y) in self.points_list]

        num_points = len(point_tags)
        line_tags = [gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % num_points]) for i in range(num_points)]

        cl = gmsh.model.geo.addCurveLoop(line_tags)
        surface = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()

        # Add physical groups /!\ IMPORTANT FOR BC'S /!\ maybe not
        gmsh.model.add_physical_group(dim=2, tags=[surface], name="domain", tag=1)
        gmsh.model.add_physical_group(dim=1, tags=line_tags, name="boundary", tag=2)
        for i, line_tag in enumerate(line_tags):
            gmsh.model.add_physical_group(dim=1, tags=[line_tag], name="line {}".format(i + 1), tag=10 + i)

        if self.element_type == "quad":
            gmsh.model.mesh.setRecombine(2, surface)
        gmsh.model.mesh.generate(2)
        gmsh.write(self.file)
        self.generated += 1

    def read_mesh(self) -> meshio.Mesh:
        return meshio.read(self.file)

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

    def _node_tag_to_index(self):
        """
        Build a mapping from gmsh node tag to index in the node coordinate array.
        """
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        tags_array = np.array(node_tags)
        dic = {tag: idx for idx, tag in enumerate(tags_array)}
        return dic

    def find_points(self, target_coords: Union[Tuple[float, float], List[Tuple[float, float]]],
                    tolerance: float = 1e-6) -> List[int]:
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

    def find_path( self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> Optional[List[int]]:
        """
        Find a path (sequence of node tags) between two coordinates using BFS on the connectivity graph.

        Parameters:
            coord1: Starting coordinate.
            coord2: Ending coordinate.

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

    def find_elements(self, coord: Tuple[float, float], tolerance: float = 1e-6) -> List[int]:
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
    E: float  # Young's modulus [Pa]
    nu: float  # Poisson's ratio
    rho: float  # Density [kg/m^3]
    plane: str  # stress or strain


class FE_2D: 
    def __init__(self, mesh_file: str, material: Material, order: int = 1):
            # --- Mesh & domain setup ------------------------------------------
            self.mesh_name = mesh_file.removesuffix('.msh')
            mesh = sfepyMesh.from_file(mesh_file)
            if mesh.coors.shape[1] == 3:
                mesh.coors[:, 2] = 0.0
            mesh.write(f'{self.mesh_name}.vtk', io='auto')
            mesh = sfepyMesh.from_file(f'{self.mesh_name}.vtk')

            self.domain = FEDomain('domain', mesh)

            # --- Bounding box & automatic gtol --------------------------------
            bb = self.domain.get_mesh_bounding_box()  # [[xmin,ymin],[xmax,ymax]]
            (xmin, ymin), (xmax, ymax) = bb

            # build KD‐tree of node coords and get typical spacing h_typ
            coords = mesh.coors[:, :2]
            tree = KDTree(coords)
            dists, _ = tree.query(coords, k=2)
            h_typ = float(np.median(dists[:, 1]))

            # gtol = fraction of h_typ, with safe fallback
            self.gtol = max(1e-8, 1e-3 * h_typ)
            tol = self.gtol

            # --- Selector definitions -----------------------------------------
            self.selector_list = {
                'all': lambda: 'all',
                'left': lambda: f'vertices in (x < {xmin + tol:.10e})',
                'right': lambda: f'vertices in (x > {xmax - tol:.10e})',
                'bottom': lambda: f'vertices in (y < {ymin + tol:.10e})',
                'top': lambda: f'vertices in (y > {ymax - tol:.10e})',
                'boundary': lambda: 'vertices of surface',
                'line_at_x': lambda x: f'vertices in (x > {x - tol:.10e}) * (x < {x + tol:.10e})',
                'line_at_y': lambda y: f'vertices in (y > {y - tol:.10e}) * (y < {y + tol:.10e})',
                'box': lambda x0, x1, y0, y1: (
                    f'vertices in (x > {x0 - tol:.10e}) * (x < {x1 + tol:.10e})'
                    f' * (y > {y0 - tol:.10e}) * (y < {y1 + tol:.10e})'
                ),
            }

            # --- Regions -------------------------------------------------------
            self.regions = {}
            # Omega = whole domain cells
            self.regions['Omega'] = self.domain.create_region('Omega', 'all')

            # --- Field & variables --------------------------------------------
            self.field = Field.from_args('u_field', np.float64, 'vector',
                                         self.regions['Omega'], approx_order=order)
            self.u = FieldVariable('u', 'unknown', self.field)
            self.v = FieldVariable('v', 'test', self.field, primary_var_name='u')

            # --- Material & internal term -------------------------------------
            self.D = stiffness_from_youngpoisson(2, material.E, material.nu, plane=material.plane)
            self.mat = sfepyMaterial('mat', D=self.D)

            self.integral = Integral('i', order=2)
            self.t_int = Term.new('dw_lin_elastic(mat.D, v, u)', self.integral, self.regions['Omega'], 
                                  mat=self.mat, v=self.v, u=self.u)

            # --- BC & load containers -----------------------------------------
            self.ebcs = []
            self.load_terms = []
            self.loads = {}

    def selector(self, selector: Union[str, Callable[..., str]], **kwargs) -> str:
        """
        Turn a selector key/callable/raw‐string into a SfePy region expression.

        Args:
            selector:  one of
                         - a key in self.selector_list (e.g. 'left', 'surface', …)
                         - a callable returning a region‐string (must accept **kwargs)
                         - a raw region expression string
            **kwargs:  parameters passed to the above if needed

        Returns:
            A valid SfePy region‐expression string (e.g. 'vertices in (x < 1e-8) *v …').

        Raises:
            KeyError:   if selector is a str but not in self.selector_list and not looking like an expression.
            TypeError:  if selector is neither str nor callable.
            ValueError: if the resulting expression is empty.
        """
        # 1) name → callable
        if callable(selector):
            expr = selector(**kwargs)

        # 2) string → lookup in self.selector_list
        elif isinstance(selector, str) and selector in self.selector_list:
            expr = self.selector_list[selector](**kwargs)

        # 3) raw expression string (heuristic: must contain at least one space or comparison)
        elif isinstance(selector, str) and any(tok in selector for tok in (' ', '<', '>', '*', 'vertices', 'cells')):
            expr = selector

        else:
            raise KeyError(f"Selector '{selector}' not found in predefined selectors "
                           f"and not a valid raw expression.")

        expr = expr.strip()
        if not expr:
            raise ValueError("Built selector expression is empty.")
        return expr

    def combine_selectors(self, *selectors: Union[str, Callable[..., str]], op: str = 'and', **kwargs) -> str:
        """
        Build a combined SfePy selector expression by AND/OR-ing
        any number of existing selectors or raw expressions.

        Args:
            *selectors:  names in self.selector_list, callables, or raw expr strings
            op:          'and' (default) or 'or'
            **kwargs:    keyword args passed to each selector callable

        Returns:
            A single region‐expression string, e.g.
            'vertices in (x < 0.1) * (y > 0.5)' or
            'vertices in (...) *v vertices in (...)'.

        Raises:
            KeyError/TypeError/ValueError from build_selector if something’s wrong.
        """
        # pick the SfePy operator
        if op == 'and':
            joiner = ' * '
        elif op == 'or':
            joiner = ' *v '
        else:
            raise ValueError(f"op must be 'and' or 'or', got {op!r}")

        parts = [self.selector(sel, **kwargs) for sel in selectors]
        # wrap each part in parentheses if it’s more than a single token
        parts = [p if p.isidentifier() else f'({p})' for p in parts]
        expr = joiner.join(parts)
        return expr

    def create_region(self, name: str, selector: Union[str, Callable[..., str]],
                      kind: str = 'cell', override: bool = False, **kwargs ):
        """
        Create and store a region in the model.

        Args:
            name:        the (unique) name of the new region.
            selector:    either
                           - a key of self.selector_list (e.g. 'left', 'surface', etc.),
                           - a raw SfePy region expression string, or
                           - a callable returning such a string.
            kind:        one of 'cell', 'facet' or 'vertex'.
            override:    if False (default), raises if `name` already exists.
            **kwargs:    any parameters passed to the selector callable.

        Returns:
            The new region object.

        Raises:
            ValueError: if the region is empty or name already exists.
            KeyError:   if selector is a string but not in self.selector_list.
        """
        # 1) check name
        if (not override) and (name in self.regions):
            raise ValueError(f"Region '{name}' already exists; use override=True to replace.")

        # 2) build the SfePy‐style expression string
        if callable(selector):
            expr = selector(**kwargs)
        elif isinstance(selector, str) and (selector in self.selector_list):
            expr = self.selector_list[selector](**kwargs)
        elif isinstance(selector, str):
            expr = selector
        else:
            raise TypeError(f"Selector must be a str or callable, got {type(selector)}")

        # 3) create region on the domain
        region = self.domain.create_region(name, expr, kind=kind)

        # 4) sanity check: must have at least one entity
        count = getattr(region, {'cell': 'n_cells',
                                 'facet': 'n_facets',
                                 'vertex': 'n_vertices'}[kind])
        if count == 0:
            # clean up
            del region
            raise ValueError(f"Region '{name}' with expression '{expr}' is empty (kind={kind}).")

        # 5) store and return
        self.regions[name] = region
        return region

    def add_dirichlet(self, name: str, selector: str, comp_dict: dict, **kwargs):
        """
        Fix a displacement component (or all) on a given boundary.
        """
        reg = self.domain.create_region(name, self.selector_list[selector](**kwargs), kind='facet')
        self.regions[name] = reg
        self.ebcs.append(EssentialBC(name, reg, comp_dict))

    def add_functional_dirichlet(self, name: str, selector: str, funcs: List[callable] = None, **kwargs) -> None:
        expr = self.selector_list[selector](**kwargs)
        reg = self.domain.create_region(name, expr, kind='facet')
        self.regions[name] = reg
        if funcs is None:
            self.ebcs.append(EssentialBC(name, reg, {'u.all': 0.0}))
        elif len(funcs) == 2:
            self.ebcs.append(EssentialBC(name + 'x', reg, {'u.0': funcs[0]}))
            self.ebcs.append(EssentialBC(name + 'y', reg, {'u.1': funcs[1]}))
        else:
            self.ebcs.append(EssentialBC(name, reg, {'u.all': funcs[0]}))

    def add_surface_load(self, selector: str, px: float = 0, py: float = 0, **kwargs):
        if px == 0.0 and py == 0.0:
            return
        idx = len(self.load_terms)
        load_name = f'load_{idx}'

        load_val = np.array([[px], [py]], dtype=float)  # shape (2,1)
        surf_load = sfepyMaterial(load_name, val=load_val)

        expr = self.selector_list[selector](**kwargs)
        reg = self.domain.create_region(f'{load_name}_reg', expr, kind='facet')

        term = Term.new(f'dw_surface_ltr({load_name}.val, v)', self.integral, region=reg, **{load_name: surf_load},
                        v=self.v)

        self.regions[f'{load_name}_reg'] = reg
        self.load_terms.append(term)
        self.loads[load_name] = surf_load

    def solve(self, vtk_out: str = None):
        """assemble, solve and store the converged state"""
        if vtk_out is None:
            vtk_out = '{}_solved_lin.vtk'.format(self.mesh_name)
        elif not vtk_out.endswith('.vtk'):
            vtk_out += '.vtk'

        term_list = [self.t_int] + [-lt for lt in self.load_terms]

        rhs = term_list[0] if len(term_list) == 1 else Terms(term_list)
        eq = Equation('balance', rhs)

        pb = Problem('elasticity_2D', equations=Equations([eq]), domain=self.domain)
        pb.set_bcs(ebcs=Conditions(self.ebcs))

        ls = ScipyDirect({})
        nls = Newton({'i_max': 1}, lin_solver=ls)
        pb.set_solver(nls)

        self.state = pb.solve()  # keep for later use
        self.u = pb.get_variables()['u']  # update variable handle
        pb.save_state(vtk_out, self.state)

        return self.state

    def probe(self, coors, quantity='u', return_status=False):
        """
        Interpolate solution‑related data at coordinates *coors*.
        quantity ∈ {'u','ux','uy','strain','stress','sxx','syy','sxy'}.
        """
        if not hasattr(self, 'state'):
            raise RuntimeError('solve() first!')

        pts = np.atleast_2d(coors)
        u_val, _, inside = self.u.evaluate_at(pts, ret_cells=True, ret_status=True)

        # strains from displacement gradient
        if quantity.startswith('strain') or quantity.startswith('stress') or quantity[-2:] in ('xx', 'yy', 'xy'):
            grad = self.u.evaluate_at(pts, mode='grad')
            eps = 0.5 * (grad + np.transpose(grad, (0, 2, 1)))
            strain = np.stack([eps[:, 0, 0], eps[:, 1, 1], 2 * eps[:, 0, 1]], axis=1)  # [ε_xx, ε_yy, γ_xy]

            if quantity == 'strain':
                out = strain
            else:
                stress = strain @ self.D.T  # Voigt
                voigt = dict(sxx=stress[:, 0], syy=stress[:, 1],
                             sxy=stress[:, 2],
                             exx=strain[:, 0], eyy=strain[:, 1], exy=strain[:, 2])
                out = voigt[quantity]
        else:  # displacement components
            comp = dict(u=u_val, ux=u_val[:, 0], uy=u_val[:, 1])
            out = comp[quantity]
        return (out, inside) if return_status else out
