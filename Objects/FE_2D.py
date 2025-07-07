from typing import List, Tuple, Dict, Optional
import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
from matplotlib.collections import LineCollection
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


class FE_Mesh:
    @staticmethod
    def compute_physical_node_groups(mesh: meshio.Mesh) -> Dict[str, List[int]]:
        """
        Extracts mapping from physical group names to node indices for 1D groups (lines).
        Returns:
            {physical_name: [node_idx, ...]}
        """
        groups = {}
        if 'line' in mesh.cells_dict:
            for cell_block in mesh.cells:
                if cell_block.type == "line":
                    if "gmsh:physical" in cell_block.data:
                        phys_names = cell_block.data["gmsh:physical"]
                        for nodes, name in zip(cell_block.data, phys_names):
                            groups.setdefault(name, []).extend(nodes)
            for name, nodes in groups.items():
                groups[name] = sorted(set(nodes))
        return groups

    def __init__(
            self,
            points: Optional[List[Tuple[float, float]]] = None,
            mesh_file: Optional[str] = None,
            element_type: str = 'triangle',
            element_size: float = 0.1,
            order: int = 1,  # New parameter: 1 for linear, 2 for quadratic
            name: str = 'myMesh',
            boundary_groups: Optional[Dict[str, List[int]]] = None,
    ):
        if points is None and mesh_file is None:
            raise ValueError("Provide either `points` or `mesh_file`.")
        self.points_list = points
        self.mesh_file = mesh_file
        self.element_type = 'triangle' if element_type in ('triangle', 'tri') else 'quad'
        self.element_size = element_size
        self.order = order  # Store element order
        self.name = name
        self.boundary_groups = boundary_groups or {}
        self._mesh: Optional[meshio.Mesh] = None
        self.generated = False

        if self.points_list is not None:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1)  # Enable console output

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.points_list is not None:
            gmsh.finalize()

    def generate_mesh(self) -> None:
        if self.points_list is None:
            raise RuntimeError("Cannot generate: no geometry defined.")

        gmsh.model.add(self.name)

        # Create points and lines
        pts = [gmsh.model.geo.addPoint(x, y, 0, self.element_size)
               for x, y in self.points_list]
        lines = [gmsh.model.geo.addLine(pts[i], pts[(i + 1) % len(pts)])
                 for i in range(len(pts))]
        cl = gmsh.model.geo.addCurveLoop(lines)
        surface = gmsh.model.geo.addPlaneSurface([cl])

        gmsh.model.geo.synchronize()

        # Create physical groups
        domain = gmsh.model.addPhysicalGroup(2, [surface])
        gmsh.model.setPhysicalName(2, domain, "domain")

        # Create boundary groups
        for name, line_indices in self.boundary_groups.items():
            actual_lines = [lines[i] for i in line_indices]
            phys = gmsh.model.addPhysicalGroup(1, actual_lines)
            gmsh.model.setPhysicalName(1, phys, name)

        # Set mesh options
        if self.element_type == 'quad':
            gmsh.model.mesh.setRecombine(2, surface)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.ElementOrder", self.order)  # Use specified order

        # Generate 2D mesh
        gmsh.model.mesh.generate(2)

        # Write mesh file
        filename = self.mesh_file or f"{self.name}.msh"
        gmsh.write(filename)
        self.mesh_file = filename
        self.generated = True
        self._mesh = meshio.read(self.mesh_file)

        # Print meshio's interpretation
        print("\nMeshio Physical Groups:")
        for name, (tag, dim) in self._mesh.field_data.items():
            print(f"  '{name}': tag={tag}, dim={dim}")

    def read_mesh(self) -> meshio.Mesh:
        if self._mesh is None:
            if self.mesh_file is None:
                raise RuntimeError("No mesh available.")
            self._mesh = meshio.read(self.mesh_file)
        return self._mesh

    def nodes(self) -> np.ndarray:
        return self.read_mesh().points[:, :2]

    def elements(self) -> np.ndarray:
        """Return elements based on element type and order"""
        if self.element_type == 'triangle':
            key = 'triangle6' if self.order == 2 else 'triangle'
        else:  # quad
            key = 'quad8' if self.order == 2 else 'quad'
        return self.read_mesh().cells_dict.get(key, np.empty((0,)))

    def physical_line_groups(self) -> Dict[str, List[int]]:
        return self.compute_physical_node_groups(self.read_mesh())

    def plot(self, save_path: Optional[str] = None) -> None:
        mesh = self.read_mesh()
        pts = mesh.points
        segments = []

        for cell_block in mesh.cells:
            if cell_block.type == "line":
                for line in cell_block.data:
                    p0 = pts[line[0]][:2]  # Extract only x,y
                    p1 = pts[line[1]][:2]
                    segments.append([p0, p1])
            elif cell_block.type in ["triangle", "triangle6"]:
                for tri in cell_block.data:
                    points = [pts[i][:2] for i in tri]  # Get 2D points
                    n = len(points)
                    for i in range(n):
                        segments.append([points[i], points[(i + 1) % n]])
            elif cell_block.type in ["quad", "quad8"]:
                for quad in cell_block.data:
                    points = [pts[i][:2] for i in quad]  # Get 2D points
                    n = len(points)
                    for i in range(n):
                        segments.append([points[i], points[(i + 1) % n]])

        lc = LineCollection(segments, linewidths=0.5, colors='k')
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect('equal')
        ax.set_title(f"{self.name} ({self.element_type} mesh, order={self.order})")

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


class FE_Material:
    def __init__(self, E: float, nu: float, rho: float, plane: str):
        self.E = E
        self.nu = nu
        self.rho = rho
        self.plane = plane.lower()
        self.D = self._make_D()

    def _make_D(self):
        E, v = self.E, self.nu
        if self.plane == "stress":
            return (E / (1 - v ** 2)) * np.array([
                [1, v, 0],
                [v, 1, 0],
                [0, 0, (1 - v) / 2]
            ])
        elif self.plane == "strain":
            factor = E * (1 - v) / ((1 + v) * (1 - 2 * v))
            return factor * np.array([
                [1, v / (1 - v), 0],
                [v / (1 - v), 1, 0],
                [0, 0, (1 - 2 * v) / (2 * (1 - v))]
            ])
        else:
            raise ValueError("Material.plane must be 'stress' or 'strain'")


class FE:
    @staticmethod
    def element_stiffness_T3(coords: np.ndarray, D: np.ndarray, t: float = 1.0) -> np.ndarray:
        """Stiffness matrix for 3-node triangle element."""
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]

        # Area calculation (absolute value for robustness)
        A = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

        # Shape function derivatives
        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])

        # Strain-displacement matrix
        B = np.zeros((3, 6))
        for i in range(3):
            B[0, 2 * i] = b[i]
            B[1, 2 * i + 1] = c[i]
            B[2, 2 * i] = c[i]
            B[2, 2 * i + 1] = b[i]
        B /= (2 * A)

        # Element stiffness matrix
        return t * A * (B.T @ D @ B)

    @staticmethod
    def element_stiffness_T6(coords: np.ndarray, D: np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        Stiffness matrix for 6-node quadratic triangle element.
        coords: (3,2) array of triangle corner coordinates.
        """
        # Compute full T6 coords: 3 corners + 3 mid-edges
        corner = coords  # shape (3,2)
        m12 = 0.5 * (corner[0] + corner[1])
        m23 = 0.5 * (corner[1] + corner[2])
        m31 = 0.5 * (corner[2] + corner[0])
        full_coords = np.vstack([corner, m12, m23, m31])  # shape (6,2)

        # 3-point Gaussian integration on reference triangle
        gauss = [((2 / 3, 1 / 6), 1 / 3), ((1 / 6, 2 / 3), 1 / 3), ((1 / 6, 1 / 6), 1 / 3)]
        Ke = np.zeros((12, 12))

        for (xi, eta), w in gauss:
            L1, L2 = xi, eta
            L3 = 1 - xi - eta

            # Shape function derivatives in natural (barycentric) coords
            dN_dxi = np.array([4 * L1 - 1, 0, 1 - 4 * L3, 4 * L2, -4 * L2, 4 * (L3 - L1)])
            dN_deta = np.array([0, 4 * L2 - 1, 1 - 4 * L3, 4 * L1, 4 * (L3 - L2), -4 * L1])

            # Jacobian matrix
            J = np.zeros((2, 2))
            for i in range(6):
                J[0, 0] += dN_dxi[i] * full_coords[i, 0]
                J[0, 1] += dN_deta[i] * full_coords[i, 0]
                J[1, 0] += dN_dxi[i] * full_coords[i, 1]
                J[1, 1] += dN_deta[i] * full_coords[i, 1]

            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError(f"Negative Jacobian determinant: {detJ}")

            invJ = np.linalg.inv(J)

            # Strain-displacement matrix
            B = np.zeros((3, 12))
            for i in range(6):
                dN = invJ @ np.array([dN_dxi[i], dN_deta[i]])
                B[0, 2 * i] = dN[0]
                B[1, 2 * i + 1] = dN[1]
                B[2, 2 * i] = dN[1]
                B[2, 2 * i + 1] = dN[0]

            Ke += t * (B.T @ D @ B) * detJ * w

        return Ke

    @staticmethod
    def element_stiffness_Q4(coords: np.ndarray, D: np.ndarray, t: float = 1.0) -> np.ndarray:
        """Stiffness matrix for 4-node quadrilateral element."""
        gp = 1.0 / np.sqrt(3.0)
        points = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
        Ke = np.zeros((8, 8))

        for xi, eta in points:
            # Shape function derivatives
            dN_dxi = 0.25 * np.array([
                -(1 - eta), (1 - eta), (1 + eta), -(1 + eta)
            ])
            dN_deta = 0.25 * np.array([
                -(1 - xi), -(1 + xi), (1 + xi), (1 - xi)
            ])

            # Jacobian matrix
            J = np.zeros((2, 2))
            for i in range(4):
                J[0, 0] += dN_dxi[i] * coords[i, 0]
                J[0, 1] += dN_deta[i] * coords[i, 0]
                J[1, 0] += dN_dxi[i] * coords[i, 1]
                J[1, 1] += dN_deta[i] * coords[i, 1]

            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError(f"Negative Jacobian determinant: {detJ}")

            invJ = np.linalg.inv(J)

            # Strain-displacement matrix
            B = np.zeros((3, 8))
            for i in range(4):
                dN = invJ @ np.array([dN_dxi[i], dN_deta[i]])
                B[0, 2 * i] = dN[0]
                B[1, 2 * i + 1] = dN[1]
                B[2, 2 * i] = dN[1]
                B[2, 2 * i + 1] = dN[0]

            Ke += t * (B.T @ D @ B) * detJ

        return Ke

    @staticmethod
    def element_stiffness_Q8(coords: np.ndarray, D: np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        Stiffness matrix for 8-node serendipity quadrilateral element.
        coords: (4,2) array of quad corner coordinates.
        """
        # Compute full Q8 coords: 4 corners + 4 mid-edges
        corner = coords  # shape (4,2)
        m12 = 0.5 * (corner[0] + corner[1])
        m23 = 0.5 * (corner[1] + corner[2])
        m34 = 0.5 * (corner[2] + corner[3])
        m41 = 0.5 * (corner[3] + corner[0])
        full_coords = np.vstack([corner, m12, m23, m34, m41])  # shape (8,2)

        # 3x3 Gauss rule
        pts = [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]
        wts = [5 / 9, 8 / 9, 5 / 9]
        Ke = np.zeros((16, 16))

        for i, xi in enumerate(pts):
            for j, eta in enumerate(pts):
                # Shape function derivatives
                dN_dxi = np.array([
                    0.25 * (1 - eta) * (2 * xi + eta),
                    0.25 * (1 - eta) * (2 * xi - eta),
                    0.25 * (1 + eta) * (2 * xi + eta),
                    0.25 * (1 + eta) * (2 * xi - eta),
                    -xi * (1 - eta),
                    0.5 * (1 - eta ** 2),
                    -xi * (1 + eta),
                    -0.5 * (1 - eta ** 2)
                ])

                dN_deta = np.array([
                    0.25 * (1 - xi) * (2 * eta + xi),
                    0.25 * (1 + xi) * (2 * eta - xi),
                    0.25 * (1 + xi) * (2 * eta + xi),
                    0.25 * (1 - xi) * (2 * eta - xi),
                    -0.5 * (1 - xi ** 2),
                    -eta * (1 + xi),
                    0.5 * (1 - xi ** 2),
                    -eta * (1 - xi)
                ])

                # Jacobian matrix
                J = np.zeros((2, 2))
                for k in range(8):
                    J[0, 0] += dN_dxi[k] * full_coords[k, 0]
                    J[0, 1] += dN_deta[k] * full_coords[k, 0]
                    J[1, 0] += dN_dxi[k] * full_coords[k, 1]
                    J[1, 1] += dN_deta[k] * full_coords[k, 1]

                detJ = np.linalg.det(J)
                if detJ <= 0:
                    raise ValueError(f"Negative Jacobian determinant: {detJ}")

                invJ = np.linalg.inv(J)

                # Strain-displacement matrix
                B = np.zeros((3, 16))
                for k in range(8):
                    dN = invJ @ np.array([dN_dxi[k], dN_deta[k]])
                    B[0, 2 * k] = dN[0]
                    B[1, 2 * k + 1] = dN[1]
                    B[2, 2 * k] = dN[1]
                    B[2, 2 * k + 1] = dN[0]

                Ke += t * (B.T @ D @ B) * detJ * wts[i] * wts[j]

        return Ke

    def __init__(self, mesh: FE_Mesh, material: FE_Material, thickness: float = 1.0):
        # Read mesh
        self.mesh = mesh.read_mesh()
        self.nodes = mesh.nodes()

        # Extract elements based on element type and order
        self.triangles = []
        self.quads = []

        if mesh.element_type == 'triangle':
            if mesh.order == 1:
                self.triangles = self.mesh.cells_dict.get("triangle", np.empty((0, 3), dtype=int))
            else:  # order=2
                self.triangles = self.mesh.cells_dict.get("triangle6", np.empty((0, 6), dtype=int))
        else:  # quad
            if mesh.order == 1:
                self.quads = self.mesh.cells_dict.get("quad", np.empty((0, 4), dtype=int))
            else:  # order=2
                self.quads = self.mesh.cells_dict.get("quad8", np.empty((0, 8), dtype=int))

        # Physical groups - FIXED ORDER: (tag, dimension)
        self.physical_groups = {}
        print("\nPhysical groups found:")

        # CORRECTED: field_data items are (tag, dimension)
        for name, (tag, dim) in self.mesh.field_data.items():
            group = {"tag": tag, "dim": dim, "cells": {}}

            # Find cells in cell_sets_dict
            if name in self.mesh.cell_sets_dict:
                group["cells"] = self.mesh.cell_sets_dict[name]

            self.physical_groups[name] = group
            print(f"  '{name}': dim={dim}, tag={tag}, cells={len(group['cells'])} blocks")

        # Material properties
        self.material = material
        self.t = thickness
        self.D = material.D

        # Initialize system matrices
        self.n_nodes = self.nodes.shape[0]
        self.ndof = 2 * self.n_nodes
        self.K = lil_matrix((self.ndof, self.ndof))
        self.F = np.zeros(self.ndof)
        self.u = np.zeros(self.ndof)

    def assemble_stiffness_matrix(self):
        """Assemble global stiffness matrix from all elements."""
        # Process triangles
        for conn in self.triangles:
            coords = self.nodes[conn]
            if len(conn) == 3:  # T3 element
                Ke = FE.element_stiffness_T3(coords, self.D, self.t)
            elif len(conn) == 6:  # T6 element
                Ke = FE.element_stiffness_T6(coords, self.D, self.t)
            else:
                raise ValueError(f"Unsupported triangle element with {len(conn)} nodes")
            self.assemble_global(Ke, conn)

        # Process quads
        for conn in self.quads:
            coords = self.nodes[conn]
            if len(conn) == 4:  # Q4 element
                Ke = FE.element_stiffness_Q4(coords, self.D, self.t)
            elif len(conn) == 8:  # Q8 element
                Ke = FE.element_stiffness_Q8(coords, self.D, self.t)
            else:
                raise ValueError(f"Unsupported quad element with {len(conn)} nodes")
            self.assemble_global(Ke, conn)

    def assemble_global(self, Ke: np.ndarray, node_ids: List[int]) -> None:
        """Assemble element stiffness matrix into global system."""
        nd = len(node_ids)
        dofs = []
        for n in node_ids:
            dofs.extend([2 * n, 2 * n + 1])

        # Optimized assembly
        for i in range(2 * nd):
            for j in range(2 * nd):
                self.K[dofs[i], dofs[j]] += Ke[i, j]

    def dirichlet_bc(self, bc_dict: Dict[int, float]) -> None:
        """Apply Dirichlet boundary conditions."""
        for dof, val in bc_dict.items():
            self.K[dof, :] = 0
            self.K[:, dof] = 0
            self.K[dof, dof] = 1.0
            self.F[dof] = val

    def neumann_bc(self, traction_edges: List[Tuple[Tuple[int, int], Tuple[float, float]]]) -> None:
        """Apply Neumann boundary conditions."""
        for (n1, n2), (tx, ty) in traction_edges:
            x1, y1 = self.nodes[n1]
            x2, y2 = self.nodes[n2]
            L = np.hypot(x2 - x1, y2 - y1)
            fe = (self.t * L / 2) * np.array([tx, ty, tx, ty])
            dofs = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
            for idx, dof in enumerate(dofs):
                self.F[dof] += fe[idx]

    def dirichlet_on_group(self, group_name: str, ux: Optional[float] = None, uy: Optional[float] = None) -> None:
        """Apply Dirichlet BCs to a physical group, including mid-side nodes for quadratic elements."""
        if group_name not in self.physical_groups:
            raise KeyError(f"Physical group '{group_name}' not found")

        group = self.physical_groups[group_name]
        nodes = set()

        # Collect nodes from all cell types in the group
        for cell_type, elem_ids in group["cells"].items():
            conn = self.mesh.cells_dict[cell_type]
            for eid in elem_ids:
                # For quadratic elements, include all nodes (not just corners)
                nodes.update(conn[eid])

        # Apply BCs
        bc = {}
        for n in nodes:
            if ux is not None:
                bc[2 * n] = ux
            if uy is not None:
                bc[2 * n + 1] = uy
        self.dirichlet_bc(bc)

    def neumann_on_group(self, group_name: str, tx: float, ty: float) -> None:
        """Apply uniform traction to a line group, supporting both linear and quadratic edges."""
        if group_name not in self.physical_groups:
            available = ", ".join(self.physical_groups.keys())
            raise KeyError(f"Physical group '{group_name}' not found. Available groups: {available}")

        group = self.physical_groups[group_name]
        if group["dim"] != 1:
            # List all groups and their dimensions for debugging
            dims_info = {name: g["dim"] for name, g in self.physical_groups.items()}
            raise ValueError(
                f"Neumann BCs require 1D line groups, but group '{group_name}' has dimension {group['dim']}.\n"
                f"All groups and dimensions: {dims_info}"
            )

        traction_edges = []
        for cell_type, elem_ids in group["cells"].items():
            # Skip if not a line element
            if cell_type not in ["line", "line3"]:
                print(f"Warning: Skipping non-line cell type '{cell_type}' in group '{group_name}'")
                continue

            conn = self.mesh.cells_dict[cell_type]
            for eid in elem_ids:
                nodes = conn[eid]
                n_nodes = len(nodes)

                if n_nodes == 2:  # Linear edge
                    # Equivalent nodal forces for linear element
                    x1, y1 = self.nodes[nodes[0]]
                    x2, y2 = self.nodes[nodes[1]]
                    L = np.hypot(x2 - x1, y2 - y1)
                    fe = (self.t * L / 2) * np.array([tx, ty, tx, ty])
                    traction_edges.append((tuple(nodes), fe))

                elif n_nodes == 3:  # Quadratic edge
                    # Equivalent nodal forces for quadratic element
                    # Using Simpson's rule for integration
                    n1, n2, n3 = nodes
                    x1, y1 = self.nodes[n1]
                    x2, y2 = self.nodes[n2]
                    x3, y3 = self.nodes[n3]

                    # Calculate element length
                    L = np.hypot(x2 - x1, y2 - y1) + np.hypot(x3 - x2, y3 - y2)

                    # Shape functions for quadratic element
                    # Traction is constant over the element
                    # Using Simpson's rule: f = (L/6)*[f1 + 4f2 + f3]
                    f1 = (self.t * L / 6) * np.array([tx, ty])
                    f2 = (4 * self.t * L / 6) * np.array([tx, ty])
                    f3 = (self.t * L / 6) * np.array([tx, ty])

                    # Flattened force vector: [f1x, f1y, f2x, f2y, f3x, f3y]
                    fe = np.array([f1[0], f1[1], f2[0], f2[1], f3[0], f3[1]])
                    traction_edges.append((tuple(nodes), fe))

                else:
                    print(f"Warning: Skipping edge with {n_nodes} nodes (only 2 or 3-node edges supported)")

        if not traction_edges:
            print(f"Warning: No valid edges found in group '{group_name}'")
            return

        # Apply Neumann BCs
        for nodes, fe in traction_edges:
            dofs = []
            for n in nodes:
                dofs.extend([2 * n, 2 * n + 1])

            if len(fe) != len(dofs):
                print(f"Error: Force vector size {len(fe)} doesn't match DOF size {len(dofs)}")
                continue

            for i, dof in enumerate(dofs):
                self.F[dof] += fe[i]

    def solve(self) -> np.ndarray:
        """Solve the system Ku = F."""
        K_csr = self.K.tocsr()
        self.u = spsolve(K_csr, self.F)
        return self.u

    def export_to_vtk(self, filename: str) -> None:
        """Export results to VTK file."""
        # Create 3D points (z=0)
        pts3 = np.hstack([self.nodes, np.zeros((self.nodes.shape[0], 1))])

        # Collect all elements
        cells = []
        if len(self.triangles) > 0:
            if len(self.triangles[0]) == 3:
                cells.append(("triangle", self.triangles))
            else:  # T6
                cells.append(("triangle6", self.triangles))
        if len(self.quads) > 0:
            if len(self.quads[0]) == 4:
                cells.append(("quad", self.quads))
            else:  # Q8
                cells.append(("quad8", self.quads))

        # Displacement data
        U_reshaped = self.u.reshape((-1, 2))
        point_data = {
            "Displacement": np.hstack([U_reshaped, np.zeros((U_reshaped.shape[0], 1))]),
            "Ux": U_reshaped[:, 0],
            "Uy": U_reshaped[:, 1]
        }

        # Create and write mesh
        mesh = meshio.Mesh(
            points=pts3,
            cells=cells,
            point_data=point_data
        )
        meshio.write(filename, mesh)


def plot_mesh(ax, nodes, triangles, quads, color='k', linewidth=0.5, title=""):
    """Plot a mesh given nodes and element connectivity"""
    segments = []

    # Process triangles
    for conn in triangles:
        n_nodes = len(conn)
        if n_nodes == 3:  # Linear triangle
            for i in range(3):
                a, b = conn[i], conn[(i + 1) % 3]
                segments.append([nodes[a], nodes[b]])
        elif n_nodes == 6:  # Quadratic triangle
            # Draw edges: [0-3-1], [1-4-2], [2-5-0]
            segments.append([nodes[conn[0]], nodes[conn[3]]])
            segments.append([nodes[conn[3]], nodes[conn[1]]])
            segments.append([nodes[conn[1]], nodes[conn[4]]])
            segments.append([nodes[conn[4]], nodes[conn[2]]])
            segments.append([nodes[conn[2]], nodes[conn[5]]])
            segments.append([nodes[conn[5]], nodes[conn[0]]])

    # Process quads
    for conn in quads:
        n_nodes = len(conn)
        if n_nodes == 4:  # Linear quad
            for i in range(4):
                a, b = conn[i], conn[(i + 1) % 4]
                segments.append([nodes[a], nodes[b]])
        elif n_nodes == 8:  # Quadratic quad
            # Draw edges: [0-4-1], [1-5-2], [2-6-3], [3-7-0]
            segments.append([nodes[conn[0]], nodes[conn[4]]])
            segments.append([nodes[conn[4]], nodes[conn[1]]])
            segments.append([nodes[conn[1]], nodes[conn[5]]])
            segments.append([nodes[conn[5]], nodes[conn[2]]])
            segments.append([nodes[conn[2]], nodes[conn[6]]])
            segments.append([nodes[conn[6]], nodes[conn[3]]])
            segments.append([nodes[conn[3]], nodes[conn[7]]])
            segments.append([nodes[conn[7]], nodes[conn[0]]])

    lc = LineCollection(segments, linewidths=linewidth, colors=color)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(title)
    return ax

if __name__ == "__main__":
    """
    TODO :
        Singular matrix for Q8 elements
        Stress method in FE
        Verify results order of amplitude
        Verify export file in Paraview
        Define a clear routine to define the creation of the geometry
        Adapt the reading of pRhino files
        Create the gui to import the rhino file
        preview the parameters of the routines
        being able to change those parameters before launching the routine
        defining all possibles routine analysises
        acces K_elem, K_glob, P_r...
        Way to link / influence of the different stiffnesses sources
        Detection of contact edges ifrom rhino file
        Writing of all the theory used in material, and element stiffnesses
        general coupling with hybridfem, ask question over the general implementation of hybridfem as a whole 
    """

    # Define square geometry points
    height = 1
    length = 10
    points = [
        (0.0, 0.0),  # bottom-left
        (length, 0.0),  # bottom-right
        (length, height),  # top-right
        (0.0, height)  # top-left
    ]

    # Define boundary groups by line indices:
    boundary_groups = {
        "left": [3],  # left edge (line index 3)
        "right": [1],  # right edge (line index 1)
        "bottom": [0],  # bottom edge (line index 0)
        "top": [2]  # top edge (line index 2)
    }

    # Create and generate mesh with quadratic triangles
    with FE_Mesh(
            points=points,
            element_type="quad",  # "tri" or "quad" for quadrilaterals
            element_size=0.1,
            order=2,  # 1:Linear elements - 2:Quadratic elements
            boundary_groups=boundary_groups,
            name="SquarePlate"
    ) as mesh:
        mesh.generate_mesh()
        mesh.plot(save_path="mesh.png")

        # Material properties
        material = FE_Material(E=1000, nu=0.3, rho=1, plane="stress")

        # Create FE system
        fe = FE(mesh, material, thickness=0.1)

        # Assemble stiffness matrices
        fe.assemble_stiffness_matrix()

        # Apply boundary conditions:
        fe.dirichlet_on_group("left", ux=0, uy=0)  # Fix left edge
        fe.neumann_on_group("right", tx=10, ty=0)  # Apply traction to right edge

        # Solve system
        u = fe.solve()

        # Export results
        fe.export_to_vtk("results.vtu")

        # Plot deformed mesh
        U = u.reshape((-1, 2))
        scale = 10  # Displacement amplification factor

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Undeformed mesh
        plot_mesh(ax1, fe.nodes, fe.triangles, fe.quads,
                  color='gray', title="Undeformed Mesh")

        # Deformed mesh (scaled)
        xy_def = fe.nodes + scale * fe.u.reshape((-1, 2))
        plot_mesh(ax2, xy_def, fe.triangles, fe.quads,
                  color='red', title="Deformed Mesh (Scaled)")

        plt.tight_layout()
        plt.savefig("deformation.png")
        plt.show()