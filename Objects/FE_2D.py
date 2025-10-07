from typing import List, Tuple, Dict, Optional, Iterable, Callable, Union
import gmsh
import meshio
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, bmat
from scipy.sparse.linalg import spsolve


# =============================
# Mesh wrapper
# =============================
class FE_Mesh:
    """
    Create or read a 2D mesh (triangles or quads, linear or quadratic),
    expose nodes/elements/physical-edge groups, quick plot, and VTK export.
    """

    def __init__(
            self,
            points: Optional[List[Tuple[float, float]]] = None,
            mesh_file: Optional[str] = None,
            element_type: str = "triangle",  # 'triangle'/'tri' or 'quad'
            element_size: float = 0.1,
            order: int = 2,  # 1=linear, 2=quadratic
            name: str = "myMesh",
            edge_groups: Optional[Dict[str, List[int]]] = None,  # indices into boundary edges (CCW)
    ):
        if points is None and mesh_file is None:
            raise ValueError("Provide either `points` or `mesh_file`.")
        self.points_list = points
        self.mesh_file = mesh_file
        self.element_type = "triangle" if element_type in ("tri", "triangle") else "quad"
        self.element_size = float(element_size)
        self.order = int(order)
        self.name = str(name)
        self.edge_groups = edge_groups or {}
        self._mesh: Optional[meshio.Mesh] = None
        self.generated = False

    # -- Mesh generation -----------------------------------------------------
    def generate_mesh(self) -> None:
        """
        Build a polygon from `points_list`, mesh it with Gmsh, create
        physical groups: 'domain' (surface) and named line groups in edge_groups.
        """
        if self.points_list is None:
            raise RuntimeError("Cannot generate: no geometry defined (points_list is None).")

        gmsh_init_here = not gmsh.isInitialized()
        if gmsh_init_here:
            gmsh.initialize()
        try:
            gmsh.model.add(self.name)

            # Points + boundary lines
            pts = [gmsh.model.geo.addPoint(x, y, 0.0, self.element_size) for x, y in self.points_list]
            lines = [gmsh.model.geo.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
            loop = gmsh.model.geo.addCurveLoop(lines)
            surface = gmsh.model.geo.addPlaneSurface([loop])
            gmsh.model.geo.synchronize()

            # Physical groups
            dom_tag = gmsh.model.addPhysicalGroup(2, [surface])
            gmsh.model.setPhysicalName(2, dom_tag, "domain")
            for name, line_indices in (self.edge_groups or {}).items():
                try:
                    phys = gmsh.model.addPhysicalGroup(1, [lines[i] for i in line_indices])
                    gmsh.model.setPhysicalName(1, phys, name)
                except Exception as e:
                    print(f"[warn] failed creating physical group '{name}': {e}")

            # Meshing options
            if self.element_type == "quad":
                gmsh.model.mesh.setRecombine(2, surface)
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
            gmsh.option.setNumber("Mesh.ElementOrder", self.order)

            gmsh.model.mesh.generate(2)

            filename = self.mesh_file or f"{self.name}.msh"
            gmsh.write(filename)
            self.mesh_file = filename
            self.generated = True

            self._mesh = meshio.read(self.mesh_file)

            if self._mesh.field_data:
                print("\nMeshio Physical Groups:")
                for name, (tag, dim) in self._mesh.field_data.items():
                    print(f"  '{name}': tag={tag}, dim={dim}")
        finally:
            if gmsh_init_here:
                gmsh.finalize()

    # -- Accessors -----------------------------------------------------------
    def read_mesh(self) -> meshio.Mesh:
        if self._mesh is None:
            if self.mesh_file is None:
                raise RuntimeError("No mesh available to read.")
            self._mesh = meshio.read(self.mesh_file)
        return self._mesh

    def nodes(self) -> np.ndarray:
        return self.read_mesh().points[:, :2].copy()

    def elements(self) -> np.ndarray:
        """
        Element connectivities for chosen family/order.
        MeshIO names:
          triangle: 'triangle' (3), 'triangle6' (6)
          quad    : 'quad' (4), 'quad8' (8)
        """
        md = self.read_mesh().cells_dict
        if self.element_type == "triangle":
            key = "triangle6" if self.order == 2 else "triangle"
        else:
            key = "quad8" if self.order == 2 else "quad"
        return md.get(key, np.empty((0, 0), dtype=int))

    # -- Physical groups on 1D entities -------------------------------------
    @staticmethod
    def compute_physical_node_groups(mesh: meshio.Mesh) -> Dict[str, List[int]]:
        """
        {physical_name -> sorted unique node ids} for all 1D groups.
        Robust across meshio versions and line/line3 edges.
        """
        groups: Dict[str, set[int]] = {}

        # tag -> name for 1D entities
        tag_to_name = {tag: name for name, (tag, dim) in mesh.field_data.items() if dim == 1}

        # Preferred path: block-aligned lists in mesh.cell_data["gmsh:physical"]
        phys_by_block = None
        if isinstance(mesh.cell_data, dict) and "gmsh:physical" in mesh.cell_data:
            lst = mesh.cell_data["gmsh:physical"]
            if isinstance(lst, list) and len(lst) == len(mesh.cells):
                phys_by_block = lst

        if phys_by_block is not None:
            for ib, cell_block in enumerate(mesh.cells):
                if cell_block.type not in {"line", "line3"}:
                    continue
                tags = phys_by_block[ib]
                if tags is None:
                    continue
                for conn, tag in zip(cell_block.data, tags):
                    name = tag_to_name.get(int(tag))
                    if name:
                        groups.setdefault(name, set()).update(int(i) for i in conn)
        else:
            # Fallback path: aggregated dicts
            phys_dict = getattr(mesh, "cell_data_dict", {}).get("gmsh:physical", {})
            for ctype in ("line", "line3"):
                conns = mesh.cells_dict.get(ctype, [])
                tags = phys_dict.get(ctype, [])
                for conn, tag in zip(conns, tags):
                    name = tag_to_name.get(int(tag))
                    if name:
                        groups.setdefault(name, set()).update(int(i) for i in conn)

        return {name: sorted(nodes) for name, nodes in groups.items()}

    def get_physical_group_nodes(self, group_name: str) -> List[int]:
        mesh = self.read_mesh()
        groups = self.compute_physical_node_groups(mesh)
        return groups.get(group_name, [])

    def physical_line_groups(self) -> Dict[str, List[int]]:
        return self.compute_physical_node_groups(self.read_mesh())

    # -- Quick plot ----------------------------------------------------------
    def plot(self, save_path: Optional[str] = None, title: Optional[str] = None) -> None:
        mesh = self.read_mesh()
        pts = mesh.points[:, :2]
        segs: List[Tuple[np.ndarray, np.ndarray]] = []

        for cb in mesh.cells:
            t = cb.type
            data = cb.data
            if t == "line":
                for e in data:
                    segs.append((pts[e[0]], pts[e[1]]))
            elif t == "line3":
                for e in data:
                    segs.append((pts[e[0]], pts[e[2]]))
                    segs.append((pts[e[2]], pts[e[1]]))
            elif t == "triangle":
                for e in data:
                    cyc = [0, 1, 2, 0]
                    for i in range(3):
                        segs.append((pts[e[cyc[i]]], pts[e[cyc[i + 1]]]))
            elif t == "triangle6":
                for e in data:
                    segs += [(pts[e[0]], pts[e[3]]), (pts[e[3]], pts[e[1]])]
                    segs += [(pts[e[1]], pts[e[4]]), (pts[e[4]], pts[e[2]])]
                    segs += [(pts[e[2]], pts[e[5]]), (pts[e[5]], pts[e[0]])]
            elif t == "quad":
                for e in data:
                    cyc = [0, 1, 2, 3, 0]
                    for i in range(4):
                        segs.append((pts[e[cyc[i]]], pts[e[cyc[i + 1]]]))
            elif t == "quad8":
                for e in data:
                    segs += [(pts[e[0]], pts[e[4]]), (pts[e[4]], pts[e[1]])]
                    segs += [(pts[e[1]], pts[e[5]]), (pts[e[5]], pts[e[2]])]
                    segs += [(pts[e[2]], pts[e[6]]), (pts[e[6]], pts[e[3]])]
                    segs += [(pts[e[3]], pts[e[7]]), (pts[e[7]], pts[e[0]])]

        lc = LineCollection(segs, linewidths=0.5, colors="k")
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title(title or f"{self.name} ({self.element_type}, order={self.order})")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=160)
            plt.close(fig)
        else:
            plt.show()


# =============================
# Material
# =============================
class FE_Material:
    def __init__(self, E: float, nu: float, rho: float, plane: str):
        self.E = float(E)
        self.nu = float(nu)
        self.rho = float(rho)
        self.plane = plane.lower().strip()
        self.D = self._make_D()

    def _make_D(self) -> np.ndarray:
        E, v = self.E, self.nu
        if self.plane == "stress":
            return (E / (1 - v * v)) * np.array(
                [[1, v, 0],
                 [v, 1, 0],
                 [0, 0, (1 - v) / 2]]
            )
        if self.plane == "strain":
            factor = E * (1 - v) / ((1 + v) * (1 - 2 * v))
            return factor * np.array(
                [[1, v / (1 - v), 0],
                 [v / (1 - v), 1, 0],
                 [0, 0, (1 - 2 * v) / (2 * (1 - v))]]
            )
        raise ValueError("Material.plane must be 'stress' or 'strain'")


# =============================
# FE Core
# =============================
class FE:
    # ---- Element stiffness -------------------------------------------------
    @staticmethod
    def element_stiffness_T3(coords: np.ndarray, D: np.ndarray, t: float = 1.0) -> np.ndarray:
        x1, y1 = coords[0];
        x2, y2 = coords[1];
        x3, y3 = coords[2]
        A = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if A <= 1e-16:
            raise ValueError("Degenerate T3 (area ~ 0).")
        b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=float)
        c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=float)
        B = np.zeros((3, 6), dtype=float)
        for i in range(3):
            B[0, 2 * i] = b[i]
            B[1, 2 * i + 1] = c[i]
            B[2, 2 * i] = c[i]
            B[2, 2 * i + 1] = b[i]
        B /= (2 * A)
        return t * A * (B.T @ D @ B)

    @staticmethod
    def element_stiffness_T6(coords: np.ndarray, D: np.ndarray, t: float = 1.0) -> np.ndarray:
        if coords.shape[0] != 6:
            raise ValueError("T6 element stiffness requires 6 node coordinates.")
        # 3-point triangle rule (exact for quadratic)
        gps = [((1 / 6, 1 / 6), 1 / 6), ((2 / 3, 1 / 6), 1 / 6), ((1 / 6, 2 / 3), 1 / 6)]
        Ke = np.zeros((12, 12), dtype=float)

        for (r, s), w in gps:
            L1, L2, L3 = r, s, 1 - r - s
            dN_dr = np.array([4 * L1 - 1, 0, -(4 * L3 - 1), 4 * L2, -4 * L2, 4 * (L3 - L1)], dtype=float)
            dN_ds = np.array([0, 4 * L2 - 1, -(4 * L3 - 1), 4 * L1, 4 * (L3 - L2), -4 * L1], dtype=float)

            dN_dnat = np.vstack([dN_dr, dN_ds])  # (2,6)
            J = dN_dnat @ coords  # (2,2)
            detJ = np.linalg.det(J)
            if abs(detJ) <= 1e-14:
                raise ValueError("Singular Jacobian in T6.")
            invJ = np.linalg.inv(J)

            B = np.zeros((3, 12), dtype=float)
            for i in range(6):
                dN_dx, dN_dy = invJ @ dN_dnat[:, i]
                B[0, 2 * i] = dN_dx
                B[1, 2 * i + 1] = dN_dy
                B[2, 2 * i] = dN_dy
                B[2, 2 * i + 1] = dN_dx

            Ke += (B.T @ D @ B) * abs(detJ) * w * t

        return Ke

    @staticmethod
    def element_stiffness_Q4(coords: np.ndarray, D: np.ndarray, t: float = 1.0) -> np.ndarray:
        gp = 1.0 / np.sqrt(3.0)
        points = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
        Ke = np.zeros((8, 8), dtype=float)

        for xi, eta in points:
            dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)], dtype=float)
            dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)], dtype=float)

            J = np.zeros((2, 2), dtype=float)
            for i in range(4):
                J[0, 0] += dN_dxi[i] * coords[i, 0]
                J[0, 1] += dN_deta[i] * coords[i, 0]
                J[1, 0] += dN_dxi[i] * coords[i, 1]
                J[1, 1] += dN_deta[i] * coords[i, 1]

            detJ = np.linalg.det(J)
            if abs(detJ) <= 1e-14:
                raise ValueError("Singular/negative Jacobian in Q4.")
            invJ = np.linalg.inv(J)

            B = np.zeros((3, 8), dtype=float)
            for i in range(4):
                dN_dx, dN_dy = invJ @ np.array([dN_dxi[i], dN_deta[i]])
                B[0, 2 * i] = dN_dx
                B[1, 2 * i + 1] = dN_dy
                B[2, 2 * i] = dN_dy
                B[2, 2 * i + 1] = dN_dx

            Ke += t * (B.T @ D @ B) * abs(detJ)  # weights are 1

        return Ke

    @staticmethod
    def _gauss_1D_3pt() -> Tuple[np.ndarray, np.ndarray]:
        x = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)], dtype=float)
        w = np.array([5 / 9, 8 / 9, 5 / 9], dtype=float)
        return x, w

    @staticmethod
    def element_stiffness_Q8(coords: np.ndarray, D: np.ndarray, t: float = 1.0) -> np.ndarray:
        if coords.shape[0] != 8:
            raise ValueError("Q8 element stiffness requires 8 node coordinates.")

        xi_list, wx = FE._gauss_1D_3pt()
        eta_list, wy = FE._gauss_1D_3pt()
        Ke = np.zeros((16, 16), dtype=float)

        for i, xi in enumerate(xi_list):
            for j, eta in enumerate(eta_list):
                # Serendipity Q8 shape funcs
                dN_dxi = 0.25 * np.array([
                    (1 - eta) * (2 * xi + eta),
                    (1 - eta) * (2 * xi - eta),
                    (1 + eta) * (2 * xi + eta),
                    (1 + eta) * (2 * xi - eta),
                    -4 * xi * (1 - eta),
                    2 * (1 - eta * eta),
                    -4 * xi * (1 + eta),
                    -2 * (1 - eta * eta)
                ], dtype=float)

                dN_deta = 0.25 * np.array([
                    (1 - xi) * (2 * eta + xi),
                    (1 + xi) * (2 * eta - xi),
                    (1 + xi) * (2 * eta + xi),
                    (1 - xi) * (2 * eta - xi),
                    -2 * (1 - xi * xi),
                    -4 * eta * (1 + xi),
                    2 * (1 - xi * xi),
                    -4 * eta * (1 - xi)
                ], dtype=float)

                J = np.zeros((2, 2), dtype=float)
                for k in range(8):
                    J[0, 0] += dN_dxi[k] * coords[k, 0]
                    J[0, 1] += dN_deta[k] * coords[k, 0]
                    J[1, 0] += dN_dxi[k] * coords[k, 1]
                    J[1, 1] += dN_deta[k] * coords[k, 1]

                detJ = np.linalg.det(J)
                if abs(detJ) <= 1e-14:
                    raise ValueError("Singular Jacobian in Q8.")
                invJ = np.linalg.inv(J)

                # Rebuild N (not strictly needed for stiffness; here for completeness)
                N = 0.25 * np.array([
                    (1 - xi) * (1 - eta) * (-xi - eta - 1),
                    (1 + xi) * (1 - eta) * (xi - eta - 1),
                    (1 + xi) * (1 + eta) * (xi + eta - 1),
                    (1 - xi) * (1 + eta) * (-xi + eta - 1),
                    2 * (1 - xi * xi) * (1 - eta),
                    2 * (1 + xi) * (1 - eta * eta),
                    2 * (1 - xi * xi) * (1 + eta),
                    2 * (1 - xi) * (1 - eta * eta)
                ], dtype=float)
                _ = N  # N might be useful later

                B = np.zeros((3, 16), dtype=float)
                for k in range(8):
                    dN_dx = invJ[0, 0] * dN_dxi[k] + invJ[0, 1] * dN_deta[k]
                    dN_dy = invJ[1, 0] * dN_dxi[k] + invJ[1, 1] * dN_deta[k]
                    B[0, 2 * k] = dN_dx
                    B[1, 2 * k + 1] = dN_dy
                    B[2, 2 * k] = dN_dy
                    B[2, 2 * k + 1] = dN_dx

                Ke += (B.T @ D @ B) * abs(detJ) * (wx[i] * wy[j]) * t

        return Ke

    # ---- FE object ---------------------------------------------------------
    def __init__(self, mesh: FE_Mesh, material: FE_Material, thickness: float = 1.0):
        self._mesh_wrap = mesh
        self.mesh = mesh.read_mesh()
        self.nodes = mesh.nodes()
        self.material = material
        self.D = material.D
        self.t = float(thickness)

        # pick elements based on declared family/order (robust to extra blocks)
        if mesh.element_type == "triangle":
            self.triangles = self.mesh.cells_dict.get("triangle", np.empty((0, 3), int))
            self.tri6 = self.mesh.cells_dict.get("triangle6", np.empty((0, 6), int))
            self.quads = np.empty((0, 4), int)
            self.quad8 = np.empty((0, 8), int)
        else:
            self.triangles = np.empty((0, 3), int)
            self.tri6 = np.empty((0, 6), int)
            self.quads = self.mesh.cells_dict.get("quad", np.empty((0, 4), int))
            self.quad8 = self.mesh.cells_dict.get("quad8", np.empty((0, 8), int))

        self.n_nodes = self.nodes.shape[0]
        self.ndof = 2 * self.n_nodes
        self.K = lil_matrix((self.ndof, self.ndof))
        self.F = np.zeros(self.ndof, dtype=float)
        self.u = np.zeros(self.ndof, dtype=float)

        # optional constraints (for MPC/symmetry using C u = c)
        self._C_rows: List[Tuple[List[int], List[float], float]] = []

    def assemble_stiffness_matrix(self) -> None:
        # Triangles
        for conn in self.triangles:
            Ke = FE.element_stiffness_T3(self.nodes[conn], self.D, self.t)
            self._scatter(Ke, conn)
        for conn in self.tri6:
            Ke = FE.element_stiffness_T6(self.nodes[conn], self.D, self.t)
            self._scatter(Ke, conn)

        # Quads
        for conn in self.quads:
            Ke = FE.element_stiffness_Q4(self.nodes[conn], self.D, self.t)
            self._scatter(Ke, conn)
        for conn in self.quad8:
            Ke = FE.element_stiffness_Q8(self.nodes[conn], self.D, self.t)
            self._scatter(Ke, conn)

    def _scatter(self, Ke: np.ndarray, node_ids: Iterable[int]) -> None:
        node_ids = np.asarray(list(node_ids), dtype=int)
        dofs = np.repeat(node_ids, 2) * 2 + np.tile([0, 1], node_ids.size)
        self.K[np.ix_(dofs, dofs)] += Ke

    # ---- Dirichlet helpers (partitioned elimination) ----------------------
    def build_dirichlet_from_group(
            self, group_name: str,
            ux: Optional[float] = None,
            uy: Optional[float] = None,
            func: Optional[Callable[[float, float], Tuple[Optional[float], Optional[float]]]] = None
    ) -> Dict[int, float]:
        """
        Returns {dof: value} for a physical line group.
        - ux/uy can be floats or None
        - func can be callable f(x,y)->(ux,uy) and overrides ux/uy when provided
        """
        node_ids = self._mesh_wrap.get_physical_group_nodes(group_name)
        if not node_ids:
            print(f"[warn] no nodes in group '{group_name}'")
            return {}
        bc = {}
        XY = self.nodes[node_ids]
        for local, n in enumerate(node_ids):
            if func is not None:
                uxi, uyi = func(*XY[local])
                if uxi is not None: bc[2 * n] = float(uxi)
                if uyi is not None: bc[2 * n + 1] = float(uyi)
            else:
                if ux is not None: bc[2 * n] = float(ux)
                if uy is not None: bc[2 * n + 1] = float(uy)
        return bc

    def solve_with_dirichlet(self, bc: Dict[int, float]) -> np.ndarray:
        """
        Partitioned elimination:
            [Kff Kfc; Kcf Kcc][uf; uc] = [Ff; Fc]
        with uc prescribed → uf = Kff^{-1}(Ff - Kfc*uc)
        """
        if not bc:
            self.u = spsolve(self.K.tocsr(), self.F)
            return self.u

        fixed = np.array(sorted(bc.keys()), dtype=int)
        vals = np.array([bc[d] for d in fixed], dtype=float)

        # de-duplicate (keep first)
        uniq, idx = np.unique(fixed, return_index=True)
        if uniq.size != fixed.size:
            fixed = uniq
            vals = vals[idx]

        all_dofs = np.arange(self.ndof, dtype=int)
        free = np.setdiff1d(all_dofs, fixed, assume_unique=False)

        K = self.K.tocsr()
        Kff = K[free[:, None], free].tocsc()
        Kfc = K[free[:, None], fixed].tocsc()

        rhs = self.F[free] - Kfc @ vals
        uf = spsolve(Kff, rhs)

        u = np.zeros(self.ndof, dtype=float)
        u[free] = uf
        u[fixed] = vals
        self.u = u
        return self.u

    def reactions(self, bc: Dict[int, float]) -> Dict[int, float]:
        """
        Return reactions at constrained DOFs after solve: R = K u - F (only fixed DOFs).
        Call after solve_with_dirichlet(bc).
        """
        if not bc:
            return {}
        R_full = (self.K.tocsr() @ self.u) - self.F
        return {d: float(R_full[d]) for d in bc.keys()}

    # ---- Constraint matrix (optional MPC/symmetry) ------------------------
    def reset_constraints(self):
        self._C_rows = []

    def add_constraint_row(self, dofs: List[int], coeffs: List[float], rhs: float):
        """Append linear constraint: sum_j coeffs[j]*u[dofs[j]] = rhs"""
        assert len(dofs) == len(coeffs)
        self._C_rows.append((dofs, coeffs, rhs))

    def solve_with_constraints(self) -> np.ndarray:
        """
        Solve augmented system:
        [K  C^T][u] = [F]
        [C   0 ][λ]   [c]
        """
        C = lil_matrix((len(self._C_rows), self.ndof), dtype=float)
        c = np.zeros(len(self._C_rows), dtype=float)
        for i, (dofs, coeffs, rhs) in enumerate(self._C_rows):
            for d, a in zip(dofs, coeffs): C[i, d] = a
            c[i] = rhs
        C = C.tocsr()
        K = self.K.tocsr()
        A = bmat([[K, C.transpose()], [C, None]], format="csr")
        rhs = np.concatenate([self.F, c])
        sol = spsolve(A, rhs)
        self.u = sol[:self.ndof]
        return self.u

    # ---- Neumann loads (consistent edge integration) ----------------------
    @staticmethod
    def _N_line2(xi):
        N = np.array([(1 - xi) / 2, (1 + xi) / 2])
        dN_dxi = np.array([-0.5, 0.5])
        return N, dN_dxi

    @staticmethod
    def _N_line3(xi):
        # quadratic on [-1,1] with nodes at (-1, +1, 0) → order [end0, end1, mid]
        N1 = 0.5 * xi * (xi - 1.0)  # at -1 → 1
        N2 = 0.5 * xi * (xi + 1.0)  # at +1 → 1
        N3 = 1.0 - xi * xi  # at  0 → 1
        N = np.array([N1, N2, N3])
        dN1 = xi - 0.5
        dN2 = xi + 0.5
        dN3 = -2.0 * xi
        dN_dxi = np.array([dN1, dN2, dN3])
        return N, dN_dxi

    @staticmethod
    def _gauss_1D(n):
        if n == 2:
            r = 1.0 / np.sqrt(3.0)
            return np.array([-r, r]), np.array([1.0, 1.0])
        if n == 3:
            r = np.sqrt(3.0 / 5.0)
            return np.array([-r, 0.0, r]), np.array([5 / 9, 8 / 9, 5 / 9])
        raise ValueError("Unsupported 1D Gauss count")

    def _edge_consistent_load(self, coords, traction_func, order):
        """
        coords: (m,2) edge node coordinates (m=2 for line, m=3 for line3).
        traction_func: callable (x,y)->(tx,ty) or constant (tx,ty) tuple.
        order: 1 for line (2-pt gauss), 2 for line3 (3-pt gauss).
        Returns the nodal force vector [fx1, fy1, fx2, fy2, ...].
        """
        if callable(traction_func):
            t_fun = traction_func
        else:
            tx, ty = traction_func
            t_fun = lambda x, y: (tx, ty)

        if coords.shape[0] == 2:
            xis, ws = self._gauss_1D(2)
            Nfun = self._N_line2
        elif coords.shape[0] == 3:
            xis, ws = self._gauss_1D(3)
            Nfun = self._N_line3
        else:
            raise ValueError("Edge must have 2 or 3 nodes")

        fe = np.zeros(2 * coords.shape[0], dtype=float)

        for xi, w in zip(xis, ws):
            N, dN_dxi = Nfun(xi)
            # geometry mapping x(ξ) = Σ N_i x_i ; dx/dξ = Σ dN_i/dξ x_i
            dx_dxi = np.array([0.0, 0.0], dtype=float)
            for a in range(coords.shape[0]):
                dx_dxi += dN_dxi[a] * coords[a, :]
            J = np.linalg.norm(dx_dxi)  # ds = |dx/dξ| dξ
            xgp = (N @ coords[:, 0])
            ygp = (N @ coords[:, 1])
            tx, ty = t_fun(xgp, ygp)

            # consistent nodal force: ∫ N^T t ds
            for a in range(coords.shape[0]):
                fe[2 * a] += N[a] * tx * J * w * self.t
                fe[2 * a + 1] += N[a] * ty * J * w * self.t

        return fe

    def neumann_on_group(self, group_name: str, traction: Union[Tuple[float, float], Callable], *,
                         kind: str = "vector"):
        """
        Apply traction along all 'line'/'line3' edges in a physical group.

        Parameters
        ----------
        traction :
            If kind='vector':
                - tuple (tx, ty) in N/m, or
                - callable f(x,y)->(tx,ty) (may vary along the edge).
            If kind='pressure' (scalar normal pressure p>0 along outward normal):
                Pass p or callable p(x,y). Make sure normal orientation matches your convention.
        kind : 'vector' | 'pressure'
        """
        group_nodes = set(self._mesh_wrap.get_physical_group_nodes(group_name))
        if not group_nodes:
            print(f"[warn] no nodes found for group '{group_name}'")
            return

        # Collect boundary edges belonging to the group
        edges = []
        for cb in self.mesh.cells:
            if cb.type in {"line", "line3"}:
                for e in cb.data:
                    e = tuple(int(i) for i in e)
                    if set(e).issubset(group_nodes):
                        edges.append(e)

        if not edges:
            print(f"[warn] no line elements in group '{group_name}'")
            return

        # Adapter for pressure -> vector using local normal derived from end nodes
        def pressure_to_vector_factory(e):
            pts = self.nodes[list(e)]

            def tv(x, y):
                p0, p1 = pts[0], pts[-1]
                t = p1 - p0
                if np.linalg.norm(t) < 1e-14:
                    n = np.array([0.0, 0.0])
                else:
                    t = t / np.linalg.norm(t)
                    n = np.array([t[1], -t[0]])  # +90° rotation; flip p if needed
                p = traction(x, y) if callable(traction) else float(traction)
                return p * n[0], p * n[1]

            return tv

        for e in edges:
            coords = self.nodes[list(e)]
            if kind == "pressure":
                t_fun = pressure_to_vector_factory(e)
            else:
                t_fun = traction

            fe_edge = self._edge_consistent_load(
                coords=coords,
                traction_func=t_fun,
                order=1 if len(e) == 2 else 2
            )

            # scatter-add into global RHS
            dofs = []
            for n in e:
                dofs.extend([2 * n, 2 * n + 1])
            for d, val in zip(dofs, fe_edge):
                self.F[d] += val

    # ---- Solve (no BC) -----------------------------------------------------
    def solve(self) -> np.ndarray:
        self.u = spsolve(self.K.tocsr(), self.F)
        return self.u

    # ---- Export ------------------------------------------------------------
    def export_to_vtk(self, filename: str) -> None:
        pts3 = np.hstack([self.nodes, np.zeros((self.nodes.shape[0], 1))])

        cells = []
        if self.triangles.size:
            cells.append(("triangle", self.triangles))
        if self.tri6.size:
            cells.append(("triangle6", self.tri6))
        if self.quads.size:
            cells.append(("quad", self.quads))
        if self.quad8.size:
            cells.append(("quad8", self.quad8))

        U = self.u.reshape((-1, 2)) if self.u.size else np.zeros((self.n_nodes, 2))
        point_data = {
            "Displacement": np.hstack([U, np.zeros((U.shape[0], 1))]),
            "Ux": U[:, 0],
            "Uy": U[:, 1],
        }

        out = meshio.Mesh(points=pts3, cells=cells, point_data=point_data)
        meshio.write(filename, out)


# =============================
# Tiny self-test
# =============================
if __name__ == "__main__":
    # Square, clamped left, uniform traction on right (triangle6 example)
    pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
    edges = {"left": [3], "right": [1], "bottom": [0], "top": [2]}  # edge indices in CCW order

    mesh = FE_Mesh(points=pts, element_type="triangle", order=2,
                   element_size=0.2, edge_groups=edges, name="square_T6")
    mesh.generate_mesh()
    mesh.plot(title="Generated mesh")

    mat = FE_Material(E=210e9, nu=0.3, rho=7850, plane="stress")
    fe = FE(mesh, mat, thickness=0.01)
    fe.assemble_stiffness_matrix()

    # Dirichlet (partitioned): clamp left edge
    bc = fe.build_dirichlet_from_group("left", ux=0.0, uy=0.0)

    # Neumann: traction on right edge (N/m); integrates to force via thickness
    fe.neumann_on_group("right", traction=(1e6, 0.0), kind="vector")

    # Solve
    u = fe.solve_with_dirichlet(bc)
    print("||u||_inf =", np.linalg.norm(u, ord=np.inf))

    # Reactions at the left edge
    R = fe.reactions(bc)
    r_total = sum(abs(v) for v in R.values())
    print("sum(|reactions|) =", r_total)

    # Export to VTK
    fe.export_to_vtk("square_T6.vtk")
