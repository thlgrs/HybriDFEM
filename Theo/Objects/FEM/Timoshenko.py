from copy import deepcopy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from Theo.Objects.Material.Material import TimoshenkoMaterial
from .FE import FE


@dataclass
class GeometryBeam:
    """
    Geometry parameters for Timoshenko beam elements.

    Attributes:
        h: Height of the beam cross-section [m]
        b: Width/thickness of the beam [m]
        A: Cross-sectional area [m²] (computed as h*b if not provided)
        I: Second moment of area [m⁴] (computed as b*h³/12 if not provided)
    """
    h: float
    b: float
    A: float = None
    I: float = None

    def __post_init__(self):
        if self.h <= 0:
            raise ValueError(f"Height must be positive, got {self.h}")
        if self.b <= 0:
            raise ValueError(f"Width must be positive, got {self.b}")

        # Compute A and I if not provided (rectangular cross-section)
        if self.A is None:
            self.A = self.h * self.b
        if self.I is None:
            self.I = self.b * (self.h ** 3) / 12


class Timoshenko(FE):
    """
    Timoshenko beam element for 2D frame analysis.

    Supports both geometric linear and nonlinear analysis.
    Uses 2 nodes with 3 DOFs each: [ux, uy, rotation_z]
    """
    DOFS_PER_NODE = 3  # Timoshenko beams: [ux, uy, rotation_z]

    def __init__(self, nodes, mat: TimoshenkoMaterial, geom: GeometryBeam, lin_geom=True):
        """
        Initialize Timoshenko beam element.

        Args:
            nodes: List of 2 node coordinates [(x1,y1), (x2,y2)]
            mat: TimoshenkoMaterial object
            geom: GeometryBeam object with h, b, A, I
            lin_geom: Use linear geometry (True) or nonlinear (False)
        """
        self.N1 = np.array(nodes[0])
        self.N2 = np.array(nodes[1])

        self._mat = mat
        self.E = mat.stiff['E']
        self.nu = mat.nu
        self.rho = mat.rho
        self.d = np.zeros(6)

        self._geom = geom
        self.h = geom.h
        self.A = geom.A
        self.I = geom.I
        self.lin_geom = lin_geom

        # FE connectivity (2 nodes, 3 DOFs each = 6 total DOFs)
        self.connect = np.zeros(2, dtype=int)
        self.dofs = np.zeros(6, dtype=int)

        # Track rotation DOFs (empty for beams - they use all 3 DOFs)
        self.rotation_dofs = np.array([], dtype=int)

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

    def make_connect(self, connect, node_number, structure=None):
        """
        Set the connection vector between the local and global node index.

        This method now supports variable DOFs per node:
        - If structure provided: uses structure.node_dof_offsets (flexible DOF system)
        - If structure=None: falls back to 3*connect (backward compatibility)

        Args:
            connect: index of the node in Structure_2D
            node_number: node index of the element's node from fe.nodes (0 or 1)
            structure: Structure_2D instance (optional, for flexible DOF support)
        """
        self.connect[node_number] = connect

        # Compute base DOF index for this node
        if structure is not None and hasattr(structure, 'node_dof_offsets') and len(
                structure.node_dof_offsets) > connect:
            # Variable DOF mode: use node_dof_offsets
            base_dof = structure.node_dof_offsets[connect]
        else:
            # Fallback: assume 3 DOFs per node
            base_dof = 3 * connect

        # Map element DOFs (3 per node: ux, uy, rz) to global structure DOFs
        if node_number == 0:
            self.dofs[:3] = np.array([0, 1, 2], dtype=int) + base_dof
        elif node_number == 1:
            self.dofs[3:] = np.array([0, 1, 2], dtype=int) + base_dof

    def get_k_glob(self):
        self.get_k_loc()

        # Compute rotation matrix for current element angle
        r_C = self.r_C(self.alpha)
        self.k_glob = np.transpose(r_C) @ self.k_loc @ r_C

        return self.k_glob

    def get_k_glob0(self):
        self.get_k_loc0()

        # Compute rotation matrix for current element angle
        r_C = self.r_C(self.alpha)
        self.k_glob0 = np.transpose(r_C) @ self.k_loc0 @ r_C

        return self.k_glob0

    def get_k_glob_LG(self):
        """
        Get geometric nonlinearity stiffness matrix (large deformation effects).

        For linear geometry (lin_geom=True), returns same as get_k_glob0().
        For nonlinear geometry, computes full stiffness with geometric terms.
        """
        if self.lin_geom:
            return self.get_k_glob0()
        else:
            return self.get_k_glob()

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
