from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional
from typing import Tuple, List

import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
from matplotlib.collections import LineCollection
from Material import Material_FE
Array = np.ndarray


@dataclass
class Geometry2D:
    """
    Geometry parameters for 2D shell elements.

    Attributes:
        t: Thickness of the shell element [m]
    """
    t: float

    def __post_init__(self):
        if self.t <= 0:
            raise ValueError(f"Thickness must be positive, got {self.t}")

class FE_Mesh:
    """
    Create or read a 2D mesh (triangles or quads, linear or quadratic),
    expose nodes/elements/physical-edge groups, quick plot, and VTK export.
    TODO Need to integrate it into add_fe, loping into the long elements list, create the elements object and append in list
    """

    def __init__(self,
                 points: Optional[List[Tuple[float, float]]] = None,  # Boxing points
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
        self.element_type = (
            "triangle" if element_type in ("tri", "triangle") else "quad"
        )
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
            raise RuntimeError(
                "Cannot generate: no geometry defined (points_list is None)."
            )

        gmsh_init_here = not gmsh.isInitialized()
        if gmsh_init_here:
            gmsh.initialize()
        try:
            gmsh.model.add(self.name)

            # Points + boundary lines
            pts = [
                gmsh.model.geo.addPoint(x, y, 0.0, self.element_size)
                for x, y in self.points_list
            ]
            lines = [
                gmsh.model.geo.addLine(pts[i], pts[(i + 1) % len(pts)])
                for i in range(len(pts))
            ]
            loop = gmsh.model.geo.addCurveLoop(lines)
            surface = gmsh.model.geo.addPlaneSurface([loop])
            gmsh.model.geo.synchronize()

            # Physical groups
            dom_tag = gmsh.model.addPhysicalGroup(2, [surface])
            gmsh.model.setPhysicalName(2, dom_tag, "domain")
            for name, line_indices in (self.edge_groups or {}).items():
                try:
                    phys = gmsh.model.addPhysicalGroup(
                        1, [lines[i] for i in line_indices]
                    )
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

    def read_mesh(self) -> meshio.Mesh:
        if self._mesh is None:
            if self.mesh_file is None:
                raise RuntimeError("No mesh available to read.")
            self._mesh = meshio.read(self.mesh_file)
        return self._mesh

    def nodes(self) -> Array:
        return self.read_mesh().points[:, :2].copy()

    def elements(self) -> Array:
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

    def plot(
            self, save_path: Optional[str] = None, title: Optional[str] = None
    ) -> None:
        mesh = self.read_mesh()
        pts = mesh.points[:, :2]
        segs: List[Tuple[Array, Array]] = []

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

@dataclass
class QuadRule:
    # tensor-product rule on [-1,1]x[-1,1]
    xi: Array
    eta: Array
    w: Array

class FE(ABC):
    def __init__(self, nodes: List[tuple[float, float]], dofs: Array):
        """

        Args:
            nodes: list of 2D nodes coordinates
            dofs: list of dof's index of the structure
        """
        self.nodes = nodes
        self.dofs = dofs

    @abstractmethod
    def make_connect(self, connect, node_number):
        pass

    @abstractmethod
    def get_mass(self):
        pass

    @abstractmethod
    def get_k_glob(self):
        pass

    @abstractmethod
    def get_k_glob0(self):
        pass

    @abstractmethod
    def get_k_glob_LG(self):
        pass

    @abstractmethod
    def get_p_glob(self, q_glob):
        pass

class Timoshenko(FE):
    def __init__(self, nodes, mat: Material_FE, geom):
        super().__init__(nodes, np.zeros(6, dtype=int))
        self.N1 = nodes[0]
        self.N2 = nodes[1]

        self._mat = mat
        self.E = mat.E
        self.nu = mat.nu
        self.rho = mat.rho
        self.d = np.zeros(6)

        self._geom = geom
        self.h = geom.h
        self.A = geom.A
        self.I = geom.I
        self.lin_geom = None

        self.connect = np.zeros(2)

    @property
    def chi(self):
        return (6 + 5 * self.nu) / (5 * (1 + self.nu))

    @property
    def G(self):
        return self.E / (2 * (1 + self.nu))

    @property
    def psi(self):
        return self.E * self.I * self.chi / (self.G * self.A)

    @staticmethod
    def r_C(alpha):
        c = np.cos(alpha)
        s = np.sin(alpha)
        r_C = np.array([
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        return r_C

    @property
    def Lx(self):
        return self.N2[0] - self.N1[0]

    @property
    def Ly(self):
        return self.N2[1] - self.N1[1]

    @property
    def L(self):
        return np.sqrt(self.Lx ** 2 + self.Ly ** 2)

    @property
    def alpha(self):
        return np.arctan2(self.Ly, self.Lx)

    @property
    def nodes(self):
        return [self.N1, self.N2]

    def get_mass(self, no_inertia=False):
        m_node = self.A * self.L / 2 * self.rho
        if no_inertia:
            I_node = 0
        else:
            I_node = ((self.L / 2) ** 2 + self.h ** 2) * (1 / 12) + (self.L / 4) ** 2

        self.mass = np.diag(m_node * np.array([1, 1, I_node, 1, 1, I_node]))

        return self.mass

    def make_connect(self, connect, node_number):
        """
        Set the connection vector between the local and global node index (here Timo and Structure).
        For each element in the fe list.
        Args:
            connect: index of the node in Structure_2D (added in the structure list or already exist based on the coordinates of the element node)
            node_number: node index of the element's node from fe.nodes
        """
        self.connect[node_number] = connect

        if node_number == 0:
            self.dofs[:3] = np.array([0, 1, 2], dtype=int) + 3 * connect * np.ones(3, dtype=int)
        elif node_number == 1:
            self.dofs[3:] = np.array([0, 1, 2], dtype=int) + 3 * connect * np.ones(3, dtype=int)

    def get_k_glob(self):
        self.get_k_loc()

        self.k_glob = np.transpose(self.r_C) @ self.k_loc @ self.r_C

        return self.k_glob

    def get_k_glob0(self):
        self.get_k_loc0()

        self.k_glob0 = np.transpose(self.r_C) @ self.k_loc0 @ self.r_C

        return self.k_glob0

    def get_k_loc(self):
        self.get_k_bsc()

        if self.lin_geom:
            self.gamma_C = np.array(
                [
                    [-1, 0, 0, 1, 0, 0],
                    [0, 1 / self.L, 1, 0, -1 / self.L, 0],
                    [0, 1 / self.L, 0, 0, -1 / self.L, 1],
                ]
            )

        self.k_loc_mat = np.transpose(self.gamma_C) @ self.k_bsc @ self.gamma_C

        if not self.lin_geom:
            self.G1, self.G23 = self.G1_G23(self.l, self.beta)

            self.k_loc_geom = self.G1 * self.p_bsc[0] + self.G23 * (
                    self.p_bsc[1] + self.p_bsc[2]
            )

        if self.lin_geom:
            self.k_loc = deepcopy(self.k_loc_mat)
        else:
            self.k_loc = self.k_loc_mat + self.k_loc_geom

    def get_k_loc0(self):
        self.get_k_bsc()

        self.gamma_C = np.array(
            [
                [-1, 0, 0, 1, 0, 0],
                [0, 1 / self.L, 1, 0, -1 / self.L, 0],
                [0, 1 / self.L, 0, 0, -1 / self.L, 1],
            ]
        )

        self.k_loc_mat0 = np.transpose(self.gamma_C) @ self.k_bsc @ self.gamma_C

        self.k_loc0 = deepcopy(self.k_loc_mat0)

    def get_k_bsc(self):
        l = self.L
        ps = self.psi

        self.k_bsc_ax = (
                self.E * self.A * np.array([[1 / l, 0, 0], [0, 0, 0], [0, 0, 0]])
        )

        self.k_bsc_fl = (
                self.E
                * self.I
                / (l * (l * l + 12 * ps))
                * np.array(
            [
                [0, 0, 0],
                [0, 4 * l * l + 12 * ps, 2 * l * l - 12 * ps],
                [0, 2 * l * l - 12 * ps, 4 * l * l + 12 * ps],
            ]
        )
        )

        self.k_bsc = self.k_bsc_ax + self.k_bsc_fl

    def get_p_glob(self, q_glob):
        self.q_glob = q_glob

        self.p_glob = np.zeros(6)

        self.q_loc = self.r_C @ self.q_glob

        self.get_p_loc()

        self.p_glob = np.transpose(self.r_C) @ self.p_loc

        return self.p_glob

    def get_p_loc(self):
        self.get_p_bsc()

        self.p_loc = np.transpose(self.gamma_C) @ self.p_bsc

    def get_p_bsc(self):
        if self.lin_geom:
            self.gamma_C = np.array(
                [
                    [-1, 0, 0, 1, 0, 0],
                    [0, 1 / self.L, 1, 0, -1 / self.L, 0],
                    [0, 1 / self.L, 0, 0, -1 / self.L, 1],
                ]
            )
            self.q_bsc = self.gamma_C @ self.q_loc

        else:
            self.l = np.sqrt(
                (self.L + self.q_loc[3] - self.q_loc[0]) ** 2
                + (self.q_loc[4] - self.q_loc[1]) ** 2
            )
            self.beta = np.arctan2(
                (self.q_loc[4] - self.q_loc[1]),
                (self.L + self.q_loc[3] - self.q_loc[0]),
            )

            c = np.cos(self.beta)
            s = np.sin(self.beta)

            cl = c / self.l
            sl = s / self.l

            self.gamma_C = np.array(
                [
                    [-c, -s, 0, c, s, 0],
                    [-sl, cl, 1, sl, -cl, 0],
                    [-sl, cl, 0, sl, -cl, 1],
                ]
            )
            self.q_bsc = np.zeros(3)

            self.q_bsc[0] = self.l - self.L
            self.q_bsc[1] = self.q_loc[2] - self.beta
            self.q_bsc[2] = self.q_loc[5] - self.beta

        self.get_k_bsc()

        self.p_bsc = self.k_bsc @ self.q_bsc

    @staticmethod
    def G1_G23(l, beta):
        sb = np.sin(beta)
        cb = np.cos(beta)

        G_1 = (1 / l) * np.array(
            [
                [sb ** 2, -cb * sb, 0, -(sb ** 2), cb * sb, 0],
                [-cb * sb, cb ** 2, 0, cb * sb, -(cb ** 2), 0],
                [0, 0, 0, 0, 0, 0],
                [-(sb ** 2), cb * sb, 0, sb ** 2, -cb * sb, 0],
                [cb * sb, -(cb ** 2), 0, -cb * sb, cb ** 2, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        G_23 = (1 / l ** 2) * np.array(
            [
                [-2 * cb * sb, cb ** 2 - sb ** 2, 0, 2 * cb * sb, sb ** 2 - cb ** 2, 0],
                [cb ** 2 - sb ** 2, 2 * cb * sb, 0, sb ** 2 - cb ** 2, -2 * cb * sb, 0],
                [0, 0, 0, 0, 0, 0],
                [2 * cb * sb, sb ** 2 - cb ** 2, 0, -2 * cb * sb, cb ** 2 - sb ** 2, 0],
                [sb ** 2 - cb ** 2, -2 * cb * sb, 0, cb ** 2 - sb ** 2, 2 * cb * sb, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        return G_1, G_23

    def PlotDefShapeElem(self, defs, scale=1):
        disc = 100

        defs_loc = self.r_C @ defs

        x_loc = np.linspace(0, self.L, disc)
        y_loc = np.zeros(disc)

        phi1 = (
                scale
                * defs_loc[1]
                * (1 - 3 * x_loc ** 2 / self.L ** 2 + 2 * x_loc ** 3 / self.L ** 3)
        )
        phi2 = (
                scale * defs_loc[2] * (x_loc - 2 * x_loc ** 2 / self.L + x_loc ** 3 / self.L ** 2)
        )
        phi3 = (
                scale * defs_loc[4] * (3 * x_loc ** 2 / self.L ** 2 - 2 * x_loc ** 3 / self.L ** 3)
        )
        phi4 = scale * defs_loc[5] * (-(x_loc ** 2) / self.L + x_loc ** 3 / self.L ** 2)

        y_loc += phi1 + phi2 + phi3 + phi4
        x_loc += np.linspace(scale * defs_loc[0], 0, disc) + np.linspace(
            0, scale * defs_loc[3], disc
        )

        # Rotation to align with element in global axes
        x_glob = np.zeros(disc)
        y_glob = np.zeros(disc)

        s, c = self.Ly / self.L, self.Lx / self.L

        r = np.array([[c, -s], [s, c]])

        for i in range(disc):
            x_glob[i], y_glob[i] = r @ np.array([x_loc[i], y_loc[i]])

        # Positioning at correct position

        x_undef = np.linspace(self.N1[0], self.N2[0], disc)
        y_undef = np.linspace(self.N1[1], self.N2[1], disc)

        x_def = x_glob + self.N1[0]
        y_def = y_glob + self.N1[1]

        plt.plot(x_def, y_def, linewidth=1.5, color="black")
        plt.plot(x_def[0], y_def[0], color="black", marker="o", markersize=3)
        plt.plot(x_def[-1], y_def[-1], color="black", marker="o", markersize=3)

    def PlotUndefShapeElem(self):
        disc = 10

        x_loc = np.linspace(0, self.L, disc)
        y_loc = np.zeros(disc)

        # Rotation to align with element in global axes
        x_glob = np.zeros(disc)
        y_glob = np.zeros(disc)

        s, c = self.Ly / self.L, self.Lx / self.L

        r = np.array([[c, -s], [s, c]])

        for i in range(disc):
            x_glob[i], y_glob[i] = r @ np.array([x_loc[i], y_loc[i]])

        # Positioning at correct position

        x_undef = np.linspace(self.N1[0], self.N2[0], disc)
        y_undef = np.linspace(self.N1[1], self.N2[1], disc)

        plt.plot(x_undef, y_undef, linewidth=1.5, color="black")
        plt.plot(x_undef[0], y_undef[0], color="black", marker="o", markersize=3)
        plt.plot(x_undef[-1], y_undef[-1], color="black", marker="o", markersize=3)

        return x_undef, y_undef

class Element2D(FE):
    """
    Isoparametric 2D element shell: subclasses provide N, dN/dxi, dN/deta,
    natural coordinates of nodes, and a quadrature rule.
    """

    def __init__(self, nodes: List[Tuple[float, float]], mat, geom: Geometry2D):
        """
        Initialize 2D finite element.
        """
        self.t = float(geom.t)
        self.mat = mat
        self.nd = len(nodes)
        self.dpn = 2  # DOF per node (u, v only)
        self.edof = self.nd * self.dpn
        self.nodes = [tuple(n) for n in nodes]

        # Initialize connectivity
        self.connect = np.zeros(self.nd, dtype=int)
        self.dofs = np.zeros(self.edof, dtype=int)

        # CRITICAL FIX: Initialize rotation_dofs
        # This was missing and caused AttributeError in Structure_2D.make_nodes()
        self.rotation_dofs = np.array([], dtype=int)

        self.lin_geom = True

    # ----- API each subclass must provide -----
    @abstractmethod
    def N_dN(self, xi: float, eta: float) -> Tuple[Array, Array, Array]:
        """
        Return (N, dN_dxi, dN_deta) at (xi,eta)
        N: (nd,), dN_dxi: (nd,), dN_deta: (nd,)
        """
        pass

    @abstractmethod
    def quad_rule(self) -> Tuple[Array, Array, Array]:
        """Return (XI, ETA, W) of quadrature in natural space."""
        pass

    @staticmethod
    def gauss_1x1() -> QuadRule:
        return QuadRule(np.array([0.0]), np.array([0.0]), np.array([2.0]))  # 1D weights=2

    @staticmethod
    def gauss_2x2() -> QuadRule:
        a = 1 / np.sqrt(3)
        pts = np.array([-a, a])
        w = np.array([1.0, 1.0])
        XI, ETA = np.meshgrid(pts, pts, indexing="xy")
        W = np.outer(w, w)
        return QuadRule(XI.ravel(), ETA.ravel(), W.ravel())

    @staticmethod
    def gauss_3x3() -> QuadRule:
        a = np.sqrt(3 / 5)
        pts = np.array([-a, 0.0, a])
        w1 = 5 / 9
        w2 = 8 / 9
        w = np.array([w1, w2, w1])
        XI, ETA = np.meshgrid(pts, pts, indexing="xy")
        W = np.outer(w, w)
        return QuadRule(XI.ravel(), ETA.ravel(), W.ravel())

    # ----- common machinery -----
    def _xy_arrays(self) -> Tuple[Array, Array]:
        x = np.array([n[0] for n in self.nodes])
        y = np.array([n[1] for n in self.nodes])
        return x, y

    def jacobian(self, dN_dxi: Array, dN_deta: Array) -> Tuple[Array, float, Array]:
        """
        Build Jacobian, detJ, and inverse from natural derivatives.
        """
        x, y = self._xy_arrays()
        J = np.array([[np.dot(dN_dxi, x), np.dot(dN_dxi, y)],
                      [np.dot(dN_deta, x), np.dot(dN_deta, y)]], dtype=float)
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError(f"Non-positive Jacobian determinant: {detJ}")
        Jinv = np.linalg.inv(J)
        return J, detJ, Jinv

    def B_matrix(self, dN_dx: Array, dN_dy: Array) -> Array:
        """
        Construct 3x(2*nd) B-matrix:
        [ dN1/dx  0  dN2/dx  0  pass ]
        [ 0  dN1/dy  0  dN2/dy pass ]
        [ dN1/dy dN1/dx dN2/dy dN2/dx pass ]
        """
        nd = self.nd
        B = np.zeros((3, 2 * nd))
        B[0, 0::2] = dN_dx
        B[1, 1::2] = dN_dy
        B[2, 0::2] = dN_dy
        B[2, 1::2] = dN_dx
        return B

    def Ke(self) -> Array:
        """
        Element stiffness: Ke = ∫ B^T D B t |J| de dn
        """
        D = self.mat.D
        XI, ETA, W = self.quad_rule()
        ndofs = 2 * self.nd
        Ke = np.zeros((ndofs, ndofs))
        for xi, eta, w in zip(XI, ETA, W):
            N, dN_dxi, dN_deta = self.N_dN(xi, eta)
            J, detJ, Jinv = self.jacobian(dN_dxi, dN_deta)
            # chain rule: [dN/dx; dN/dy] = Jinv @ [dN/dxi; dN/deta]
            grads_nat = np.vstack((dN_dxi, dN_deta))  # 2 x nd
            grads_xy = Jinv @ grads_nat  # 2 x nd
            dN_dx, dN_dy = grads_xy[0], grads_xy[1]
            B = self.B_matrix(dN_dx, dN_dy)
            Ke += self.t * (B.T @ D @ B) * detJ * w
        return Ke

    def Me_consistent(self) -> Array:
        """
        Consistent mass: Me = ∫ ρ t (N^T N) dA  (lumped is easy too)
        """
        rho = self.mat.rho
        XI, ETA, W = self.quad_rule()
        ndofs = 2 * self.nd
        Me = np.zeros((ndofs, ndofs))
        for xi, eta, w in zip(XI, ETA, W):
            N, dN_dxi, dN_deta = self.N_dN(xi, eta)
            J, detJ, _ = self.jacobian(dN_dxi, dN_deta)
            # build 2D Nbar for u,v
            Nbar = np.zeros((2, 2 * self.nd))
            Nbar[0, 0::2] = N
            Nbar[1, 1::2] = N
            Me += rho * self.t * (Nbar.T @ Nbar) * detJ * w
        return Me

    @staticmethod
    def tri_area(x: Array, y: Array) -> float:
        # x,y arrays length 3 (or 6 but use first 3 for area)
        return 0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))

    def make_connect(self, connect: int, node_number: int) -> None:
        """
        CORRECTED make_connect() method for Element2D class.

        Maps local element node to global structure node and DOFs.

        CHANGES FROM ORIGINAL:
        - Was just 'pass' - now fully implemented
        - Properly handles 3 DOF/node structure (u, v, θ)
        - Element only uses 2 DOF/node (u, v)
        - Tracks rotation DOFs that need to be fixed

        Args:
            connect: Global node index in Structure_2D.list_nodes
            node_number: Local node index in this element (0 to nd-1)
        """
        # Store global node index
        self.connect[node_number] = connect

        # Map to global DOFs
        # Structure_2D uses 3*node_index + {0:u, 1:v, 2:θ}
        # Element2D only uses u and v
        base_dof = 3 * connect

        # Map element DOFs (just u,v) to global structure DOFs
        self.dofs[2 * node_number] = base_dof  # u component
        self.dofs[2 * node_number + 1] = base_dof + 1  # v component

        # Track rotation DOF (θ) that needs to be fixed/constrained
        rotation_dof = base_dof + 2
        if rotation_dof not in self.rotation_dofs:
            self.rotation_dofs = np.append(self.rotation_dofs, rotation_dof)

    def get_mass(self):
        return self.Me_consistent()

    def get_k_glob(self):
        return self.Ke()

    def get_k_glob0(self):
        pass

    def get_k_glob_LG(self):
        pass

    def get_p_glob(self, q_glob):
        pass

class Q4(Element2D):
    # natural node positions (for reference only)
    NAT = np.array([[-1, -1], [+1, -1], [+1, +1], [-1, +1]], dtype=float)

    def N_dN(self, xi: float, eta: float) -> Tuple[Array, Array, Array]:
        N1 = 0.25 * (1 - xi) * (1 - eta)
        N2 = 0.25 * (1 + xi) * (1 - eta)
        N3 = 0.25 * (1 + xi) * (1 + eta)
        N4 = 0.25 * (1 - xi) * (1 + eta)
        N = np.array([N1, N2, N3, N4])
        dN_dxi = 0.25 * np.array([-(1 - eta), +(1 - eta), +(1 + eta), -(1 + eta)])
        dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), +(1 + xi), +(1 - xi)])
        return N, dN_dxi, dN_deta

    def quad_rule(self):
        return self.gauss_2x2()

class Q9(Element2D):
    # tensor-product 1D quad polynomials
    @staticmethod
    def N1D(xi: float) -> Tuple[float, float, float]:
        # nodes at -1,0,1
        return 0.5 * xi * (xi - 1), 1 - xi ** 2, 0.5 * xi * (xi + 1)

    @staticmethod
    def dN1D(xi: float) -> Tuple[float, float, float]:
        return xi - 0.5, -2 * xi, xi + 0.5

    def N_dN(self, xi: float, eta: float) -> Tuple[Array, Array, Array]:
        Ne = np.array(self.N1D(xi))
        Nn = np.array(self.N1D(eta))
        dNe = np.array(self.dN1D(xi))
        dNn = np.array(self.dN1D(eta))

        # order: corners(1..4), midsides(5..8), center(9)
        # tensor products:
        N = np.array([
            Ne[0] * Nn[0], Ne[2] * Nn[0], Ne[2] * Nn[2], Ne[0] * Nn[2],
            Ne[1] * Nn[0], Ne[2] * Nn[1], Ne[1] * Nn[2], Ne[0] * Nn[1],
            Ne[1] * Nn[1]
        ])

        # derivatives via product rule
        dN_dxi = np.array([
            dNe[0] * Nn[0], dNe[2] * Nn[0], dNe[2] * Nn[2], dNe[0] * Nn[2],
            dNe[1] * Nn[0], dNe[2] * Nn[1], dNe[1] * Nn[2], dNe[0] * Nn[1],
            dNe[1] * Nn[1]
        ])
        dN_deta = np.array([
            Ne[0] * dNn[0], Ne[2] * dNn[0], Ne[2] * dNn[2], Ne[0] * dNn[2],
            Ne[1] * dNn[0], Ne[2] * dNn[1], Ne[1] * dNn[2], Ne[0] * dNn[1],
            Ne[1] * dNn[1]
        ])
        return N, dN_dxi, dN_deta

    def quad_rule(self):
        qr = self.gauss_3x3()
        return qr.xi, qr.eta, qr.w

class T3(Element2D):
    # Constant Strain Triangle T3
    def N_dN(self, xi, eta):
        N = np.array([1 - xi - eta, xi, eta])
        dN_dxi = np.array([-1, 1, 0])
        dN_deta = np.array([-1, 0, 1])
        return N, dN_dxi, dN_deta

    def quad_rule(self):
        # single Gauss point at centroid (1/3, 1/3)
        return np.array([1 / 3]), np.array([1 / 3]), np.array([0.5])

    def Ke(self) -> Array:
        x, y = self._xy_arrays()
        A = self.tri_area(x, y)
        if A <= 0: raise ValueError("Invalid triangle area")
        # coefficients b_i, c_i
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
        B = (1 / (2 * A)) * np.array([
            [b[0], 0, b[1], 0, b[2], 0],
            [0, c[0], 0, c[1], 0, c[2]],
            [c[0], b[0], c[1], b[1], c[2], b[2]],
        ])
        D = self.mat.D
        return self.t * A * (B.T @ D @ B)

    def Me_consistent(self) -> Array:
        x, y = self._xy_arrays()
        A = self.tri_area(x, y)
        rho, t = self.mat.rho, self.t
        # consistent mass for linear triangle (2 dof/node):
        m = rho * t * A / 12.0
        # scalar N^T N integrated gives pattern [[2,1,1],[1,2,1],[1,1,2]]
        Msc = m * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        # expand to u,v
        Me = np.zeros((6, 6))
        Me[0::2, 0::2] = Msc
        Me[1::2, 1::2] = Msc
        return Me

class T6(Element2D):
    # Linear Strain Triangle T6
    """
    6-node quadratic triangle (LST) for plane stress/strain.
    Nodes must be ordered counterclockwise:
        1,2,3 = corners; 4 on edge (1-2), 5 on edge (2-3), 6 on edge (3-1).
    Reference (natural) triangle:
        (xi, eta) with xi >= 0, eta >= 0, xi + eta <= 1
    Barycentric coordinates:
        L1 = 1 - xi - eta,  L2 = xi,  L3 = eta
    Quadratic shape functions:
        N1 = L1(2L1-1), N2 = L2(2L2-1), N3 = L3(2L3-1),
        N4 = 4 L1 L2,   N5 = 4 L2 L3,   N6 = 4 L3 L1
    """

    # 3-point Gaussian rule on the reference triangle:
    # points: (1/6,1/6), (2/3,1/6), (1/6,2/3); weights: 1/6 each (sum = area = 1/2)
    TRI_XI = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=float)
    TRI_ETA = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0], dtype=float)
    TRI_W = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=float)

    def quad_rule(self):
        """Return (XI, ETA, W) for the 3-point degree-2 exact triangular quadrature."""
        return self.TRI_XI, self.TRI_ETA, self.TRI_W

    def N_dN(self, xi: float, eta: float):
        """
        Return (N, dN_dxi, dN_deta) at given (xi, eta) in the reference triangle.

        Chain rule via barycentric coordinates:
            L1 = 1 - xi - eta,  L2 = xi,  L3 = eta
            dL1/dxi=-1, dL1/deta=-1;  dL2/dxi=1, dL2/deta=0;  dL3/dxi=0, dL3/deta=1
        """
        L1 = 1.0 - xi - eta
        L2 = xi
        L3 = eta

        # shape functions (6,)
        N = np.array([
            L1 * (2.0 * L1 - 1.0),
            L2 * (2.0 * L2 - 1.0),
            L3 * (2.0 * L3 - 1.0),
            4.0 * L1 * L2,
            4.0 * L2 * L3,
            4.0 * L3 * L1
        ], dtype=float)

        # partials w.r.t barycentrics (each length-6)
        dN_dL1 = np.array([4.0 * L1 - 1.0, 0.0, 0.0, 4.0 * L2, 0.0, 4.0 * L3], dtype=float)
        dN_dL2 = np.array([0.0, 4.0 * L2 - 1.0, 0.0, 4.0 * L1, 4.0 * L3, 0.0], dtype=float)
        dN_dL3 = np.array([0.0, 0.0, 4.0 * L3 - 1.0, 0.0, 4.0 * L2, 4.0 * L1], dtype=float)

        # chain rule to (xi, eta)
        dN_dxi = (-1.0) * dN_dL1 + (1.0) * dN_dL2 + (0.0) * dN_dL3
        dN_deta = (-1.0) * dN_dL1 + (0.0) * dN_dL2 + (1.0) * dN_dL3

        return N, dN_dxi, dN_deta
