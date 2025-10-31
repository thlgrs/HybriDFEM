# Standart imports
import warnings
from typing import List, Union, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from matplotlib.patches import Polygon
from scipy.spatial import cKDTree

from Theo.Objects.DFEM.Block import Block_2D
from Theo.Objects.DFEM.ContactFace import CF_2D
from Theo.Structures.Structure_2D import Structure_2D


class Structure_block(Structure_2D):
    """
    Low-level Structure_block class for discrete element assemblies.

    This class provides the core functionality for block assemblies with contact interfaces.
    Use specialized child classes (BeamBlockStructure, ArchBlockStructure, etc.) for
    high-level construction methods.
    """

    def __init__(self, listBlocks: Union[List[Block_2D], None] = None):
        super().__init__()
        self.list_blocks = listBlocks or []
        self.list_cfs: List[CF_2D] = []

    # Construction methods
    def add_block(self, vertices, b=1, material=None, ref_point=None):
        """
        Add a single block to the structure.

        Parameters
        ----------
        vertices : array-like
            Block vertices as 2D array
        b : float, optional
            Out-of-plane thickness [m]
        material : Material, optional
            Material model for the block
        ref_point : array-like, optional
            Reference point for the block (defaults to centroid)
        """
        self.list_blocks.append(
            Block_2D(vertices, b=b, material=material, ref_point=ref_point)
        )

    def add_block(self, ref_point, l, h, b=1, material=None):
        xc, yc = ref_point
        dx, dy = l / 2, h / 2
        vertices = np.array([(xc - dx, yc - dy), (xc + dx, yc - dy), (xc + dx, yc + dy), (xc - dx, yc + dy)])
        block = Block_2D(vertices, b=b, material=material)
        self.list_blocks.append(block)

    # Generation methods
    def _make_nodes_block(self):
        for block in self.list_blocks:
            # Blocks always have 3 DOFs: [ux, uy, rotation_z]
            dof_count = getattr(block, 'DOFS_PER_NODE', 3)
            index = self._add_node_if_new(block.ref_point, dof_count=dof_count)
            block.make_connect(index, structure=self)

    def make_nodes(self):
        self._make_nodes_block()
        # Use flexible DOF calculation
        self.nb_dofs = self.compute_nb_dofs()
        self.U = np.zeros(self.nb_dofs, dtype=float)
        self.P = np.zeros(self.nb_dofs, dtype=float)
        self.P_fixed = np.zeros(self.nb_dofs, dtype=float)
        self.dof_fix = np.array([], dtype=int)
        self.dof_free = np.arange(self.nb_dofs, dtype=int)
        self.nb_dof_fix = 0
        self.nb_dof_free = len(self.dof_free)

    def detect_interfaces(self, eps=1e-9, margin=0.01):
        def overlap_colinear(seg1, seg2, eps=1e-9):
            """
            seg1, seg2: each as np.ndarray shape (2,2), rows are endpoints [[x1,y1],[x2,y2]]
            Return (has_overlap: bool, overlap_endpoints: np.ndarray shape (2,2) or None)
            """
            p1, p2 = np.asarray(seg1, float)
            q1, q2 = np.asarray(seg2, float)

            v = p2 - p1
            Lv = np.linalg.norm(v)
            if Lv <= eps:
                return False, None  # degenerate segment
            u = v / Lv  # unit direction

            # project all endpoints onto u, using p1 as origin
            tp = np.array([0.0, Lv])  # p1->p1, p1->p2
            tq = np.array([np.dot(q1 - p1, u), np.dot(q2 - p1, u)])
            tq.sort()

            t_start = max(tp.min(), tq.min())
            t_end = min(tp.max(), tq.max())

            if t_end - t_start < -eps:
                return False, None

            a = p1 + t_start * u
            b = p1 + t_end * u
            return True, np.vstack([a, b])

        def are_colinear(p1, p2, q1, q2, eps=1e-9):
            v = p2 - p1
            w1 = q1 - p1
            w2 = q2 - p1

            # 2D “cross product” magnitude for (a,b)×(c,d) := a*d - b*c
            def cross(a, b): return a[0] * b[1] - a[1] * b[0]

            return (abs(cross(v, w1)) <= eps) and (abs(cross(v, w2)) <= eps)

        def circles_separated_sq(c1, r1, c2, r2, margin=0.01):
            # return True if centers are farther than (r1+r2)*(1+margin)
            d2 = np.sum((c1 - c2) ** 2)
            thr = (r1 + r2) ** 2 * (1.0 + margin) ** 2
            return d2 >= thr

        # prefetch
        blocks = self.list_blocks
        B = len(blocks)
        triplets = [blk.compute_triplets() for blk in blocks]

        interfaces = []
        self.interf_counter = 0

        for i in range(B):
            cand = blocks[i]
            for j in range(i + 1, B):
                anta = blocks[j]

                # 1) quick prunes
                if cand.connect == anta.connect:
                    continue
                if circles_separated_sq(cand.circle_center, cand.circle_radius,
                                        anta.circle_center, anta.circle_radius,
                                        margin=margin):
                    continue

                # 2) test edges on the same line
                ifaces_ij = []
                for t1 in triplets[i]:
                    A1, B1, C1 = t1["ABC"]
                    P = np.asarray(t1["Vertices"], float)  # shape (2,2)
                    for t2 in triplets[j]:
                        if not np.allclose(t1["ABC"], t2["ABC"], rtol=1e-8, atol=eps):
                            continue
                        Q = np.asarray(t2["Vertices"], float)

                        # now both segments lie on the same infinite line; check finite overlap
                        has, seg = overlap_colinear(P, Q, eps=eps)
                        if not has:
                            continue

                        a, b = seg  # endpoints
                        u = (b - a)
                        Lu = np.linalg.norm(u)
                        if Lu <= eps:  # zero-length overlap
                            continue
                        u /= Lu
                        n = np.array([-u[1], u[0]])  # left-hand normal
                        # Decide block A vs B via normal direction
                        if np.dot(cand.ref_point - a, n) > 0:
                            blA, blB = cand, anta
                        else:
                            blA, blB = anta, cand
                        ifaces_ij.append({
                            "Block A": blA,
                            "Block B": blB,
                            "x_e1": a,
                            "x_e2": b,
                            # (optionally keep unit vectors if useful)
                            # "tangent": u, "normal": n
                        })

                self.interf_counter += 1
                if ifaces_ij:
                    interfaces.extend(ifaces_ij)

        return interfaces

    def make_cfs(self, lin_geom, nb_cps=2, offset=-1, contact=None, surface=None, weights=None, interfaces=None):
        if interfaces is None:
            interfaces = self.detect_interfaces()
        for i, face in enumerate(interfaces):
            cf = CF_2D(face, nb_cps, lin_geom, offset=offset, contact=contact, surface=surface, weights=weights)
            self.list_cfs.append(cf)
            cf.bl_A.cfs.append(i)
            cf.bl_B.cfs.append(i)

    # Solving methods
    def _get_P_r_block(self):
        self.dofs_defined()
        if not hasattr(self, "P_r"):
            self.P_r = np.zeros(self.nb_dofs, dtype=float)

        for CF in self.list_cfs:
            qf_glob = np.zeros(6)
            qf_glob[:3] = self.U[CF.bl_A.dofs]
            qf_glob[3:] = self.U[CF.bl_B.dofs]
            pf_glob = CF.get_pf_glob(qf_glob)
            self.P_r[CF.bl_A.dofs] += pf_glob[:3]
            self.P_r[CF.bl_B.dofs] += pf_glob[3:]

    def get_P_r(self):
        self.P_r = np.zeros(self.nb_dofs, dtype=float)
        self._get_P_r_block()

    def _mass_block(self, no_inertia: bool = False):
        for block in getattr(self, "list_blocks", []):
            # block mass matrix must align with block.dofs length
            M_block = block.get_mass(no_inertia=no_inertia)
            dofs = np.asarray(block.dofs, dtype=int)
            self.M[np.ix_(dofs, dofs)] += M_block

    def get_M_str(self, no_inertia: bool = False):
        self.dofs_defined()
        self.M = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._mass_block(no_inertia=no_inertia)
        return self.M

    def _stiffness_block(self):
        for CF in getattr(self, "list_cfs", []):
            dof1 = CF.bl_A.dofs
            dof2 = CF.bl_B.dofs

            kf_glob = CF.get_kf_glob()

            self.K[np.ix_(dof1, dof1)] += kf_glob[:3, :3]
            self.K[np.ix_(dof1, dof2)] += kf_glob[:3, 3:]
            self.K[np.ix_(dof2, dof1)] += kf_glob[3:, :3]
            self.K[np.ix_(dof2, dof2)] += kf_glob[3:, 3:]

    def _stiffness0_block(self):
        for CF in getattr(self, "list_cfs", []):
            dof1 = CF.bl_A.dofs
            dof2 = CF.bl_B.dofs

            kf_glob0 = CF.get_kf_glob0()

            self.K0[np.ix_(dof1, dof1)] += kf_glob0[:3, :3]
            self.K0[np.ix_(dof1, dof2)] += kf_glob0[:3, 3:]
            self.K0[np.ix_(dof2, dof1)] += kf_glob0[3:, :3]
            self.K0[np.ix_(dof2, dof2)] += kf_glob0[3:, 3:]

    def _stiffness_LG_block(self):
        for CF in getattr(self, "list_cfs", []):
            dof1 = CF.bl_A.dofs
            dof2 = CF.bl_B.dofs

            kf_glob_LG = CF.get_kf_glob_LG()

            self.K_LG[np.ix_(dof1, dof1)] += kf_glob_LG[:3, :3]
            self.K_LG[np.ix_(dof1, dof2)] += kf_glob_LG[:3, 3:]
            self.K_LG[np.ix_(dof2, dof1)] += kf_glob_LG[3:, :3]
            self.K_LG[np.ix_(dof2, dof2)] += kf_glob_LG[3:, 3:]

    def get_K_str(self):
        self.dofs_defined()
        self.K = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness_block()
        return self.K

    def get_K_str0(self):
        self.dofs_defined()
        self.K0 = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness0_block()
        return self.K0

    def get_K_str_LG(self):
        self.dofs_defined()
        self.K_LG = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness_LG_block()
        return self.K_LG

    def set_lin_geom(self, lin_geom=True):
        for cf in self.list_cfs:
            cf.set_lin_geom(lin_geom)

    def get_C_str(self):
        if not (hasattr(self, "K")):
            self.get_K_str()
        # if not (hasattr(self, 'M')): self.get_M_str()

        if not hasattr(self, "damp_coeff"):
            # No damping
            if self.xsi[0] == 0 and self.xsi[1] == 0:
                self.damp_coeff = np.zeros(2)

            elif self.damp_type == "RAYLEIGH":
                try:
                    self.solve_modal(modes=2, save=False, initial=True)
                except Exception:
                    self.solve_modal(save=False, initial=True)

                A = np.array(
                    [
                        [1 / self.eig_vals[0], self.eig_vals[0]],
                        [1 / self.eig_vals[1], self.eig_vals[1]],
                    ]
                )

                if isinstance(self.xsi, float):
                    self.xsi = [self.xsi, self.xsi]
                    self.damp_coeff = 2 * sc.linalg.solve(A, np.array(self.xsi))

                if isinstance(self.xsi, list) and len(self.xsi) == 2:
                    self.damp_coeff = 2 * sc.linalg.solve(A, np.array(self.xsi))

                else:
                    warnings.warn(
                        "Xsi is not a list of two damping ratios for Rayleigh damping"
                    )

            elif self.damp_type == "STIFF":
                if not hasattr(self, "eig_vals"):
                    try:
                        self.solve_modal(modes=1, save=False, initial=True)
                    except Exception:
                        self.solve_modal(save=False, initial=True)
                self.damp_coeff = np.array([0, 2 * self.xsi[0] / self.eig_vals[0]])

            elif self.damp_type == "MASS":
                try:
                    self.solve_modal(modes=1, save=False, initial=True)
                except Exception:
                    self.solve_modal(save=False, initial=True)
                self.damp_coeff = np.array([2 * self.xsi[0] * self.eig_vals[0], 0])
                print(self.damp_coeff)

        if self.stiff_type == "INIT":
            if not (hasattr(self, "C")):
                self.get_K_str0()
                self.C = self.damp_coeff[0] * self.M + self.damp_coeff[1] * self.K0

        elif self.stiff_type == "TAN":
            self.get_K_str()
            self.C = self.damp_coeff[0] * self.M + self.damp_coeff[1] * self.K

        elif self.stiff_type == "TAN_LG":
            self.get_K_str_LG()
            self.C = self.damp_coeff[0] * self.M + self.damp_coeff[1] * self.K_LG

    def commit(self):
        for CF in self.list_cfs:
            CF.commit()

    def revert_commit(self):
        for CF in self.list_cfs:
            CF.revert_commit()

    def plot(
            self,
            ax: Optional[plt.Axes] = None,
            show_nodes: bool = True,
            show_node_labels: bool = False,
            show_blocks: bool = True,
            show_block_labels: bool = False,
            show_contact_faces: bool = True,
            show_deformed: bool = False,
            deformation_scale: float = 1.0,
            node_size: float = 80,
            node_color: str = 'red',
            block_facecolor: str = 'lightblue',
            block_edgecolor: str = 'black',
            block_alpha: float = 0.6,
            contact_color: str = 'green',
            contact_linewidth: float = 2,
            title: Optional[str] = None,
            figsize: Tuple[float, float] = (12, 10),
            **kwargs
    ) -> plt.Figure:
        """
        Plot Structure_block with blocks, nodes, and contact faces.

        Additional Parameters (beyond base class)
        ------------------------------------------
        show_blocks : bool, default=True
            Whether to display block polygons
        show_block_labels : bool, default=False
            Whether to label blocks
        show_contact_faces : bool, default=True
            Whether to display contact interfaces
        node_color : str, default='red'
            Color for node markers
        block_facecolor : str, default='lightblue'
            Fill color for blocks
        block_edgecolor : str, default='black'
            Edge color for blocks
        block_alpha : float, default=0.6
            Transparency of blocks
        contact_color : str, default='green'
            Color for contact faces
        contact_linewidth : float, default=2
            Width of contact face lines
        """
        # Validate
        if not self.list_nodes:
            raise ValueError("No nodes found. Call make_nodes() first.")

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        nodes = np.array(self.list_nodes)

        # Check for deformation data
        displacements = None
        if show_deformed:
            if hasattr(self, 'U') and self.U is not None:
                displacements = self.U
            else:
                warnings.warn("No displacements found. Showing undeformed.")
                show_deformed = False

        # Plot blocks
        if show_blocks:
            self._plot_blocks(
                ax,
                show_deformed,
                deformation_scale,
                displacements,
                block_facecolor,
                block_edgecolor,
                block_alpha,
                show_block_labels
            )

        # Plot contact faces
        if show_contact_faces and self.list_cfs:
            self._plot_contact_faces(
                ax,
                show_deformed,
                deformation_scale,
                displacements,
                contact_color,
                contact_linewidth
            )

        # Plot nodes
        if show_nodes:
            self._plot_nodes(
                ax,
                nodes,
                show_deformed,
                deformation_scale,
                displacements,
                node_size,
                node_color,
                show_node_labels
            )

        # Formatting
        self._format_axes(ax, title, len(self.list_blocks), len(nodes),
                          len(self.list_cfs), show_deformed, deformation_scale)

        return fig

    def _plot_blocks(self, ax, show_deformed, scale, displacements,
                     facecolor, edgecolor, alpha, show_labels):
        """Helper to plot block polygons"""
        for idx, block in enumerate(self.list_blocks):
            vertices = block.v.copy()

            if show_deformed and hasattr(block, 'dofs') and displacements is not None:
                # Apply rigid body transformation
                u = displacements[block.dofs[0]] * scale
                v = displacements[block.dofs[1]] * scale
                theta = displacements[block.dofs[2]] * scale

                # Rotation matrix
                R = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])

                # Transform: rotate about ref_point, then translate
                ref = block.ref_point
                vertices = (vertices - ref) @ R.T + ref + np.array([u, v])

            # Create polygon
            polygon = Polygon(
                vertices,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                linewidth=1.5
            )
            ax.add_patch(polygon)

            # Label
            if show_labels:
                centroid = np.mean(vertices, axis=0)
                ax.text(centroid[0], centroid[1], f'B{idx}',
                        ha='center', va='center', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', alpha=0.7))

    def _plot_contact_faces(self, ax, show_deformed, scale, displacements,
                            color, linewidth):
        """Helper to plot contact faces"""
        for cf in self.list_cfs:
            if not hasattr(cf, 'cps'):
                continue

            for cp in cf.cps:
                x_cp = cp.x_cp.copy()

                # Simple deformation approximation
                if show_deformed and hasattr(cf, 'bl_A') and hasattr(cf, 'bl_B'):
                    if hasattr(cf.bl_A, 'dofs') and hasattr(cf.bl_B, 'dofs'):
                        if displacements is not None:
                            u_A = displacements[cf.bl_A.dofs[:2]] * scale
                            u_B = displacements[cf.bl_B.dofs[:2]] * scale
                            x_cp += (u_A + u_B) / 2

                # Draw interface line
                if hasattr(cp, 'long') and hasattr(cp, 'h'):
                    tran = np.array([-cp.long[1], cp.long[0]])
                    h_half = cp.h / 2
                    p1 = x_cp - h_half * tran
                    p2 = x_cp + h_half * tran

                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                            color=color, linewidth=linewidth, alpha=0.8)

    def _plot_nodes(self, ax, nodes, show_deformed, scale, displacements,
                    size, color, show_labels):
        """Helper to plot nodes"""
        node_coords = nodes.copy()

        if show_deformed and displacements is not None:
            for i in range(len(nodes)):
                u_idx, v_idx = 3 * i, 3 * i + 1
                if u_idx < len(displacements) and v_idx < len(displacements):
                    node_coords[i, 0] += displacements[u_idx] * scale
                    node_coords[i, 1] += displacements[v_idx] * scale

        ax.scatter(node_coords[:, 0], node_coords[:, 1],
                   s=size, c=color, marker='o',
                   edgecolors='darkred', linewidths=1.5,
                   zorder=5, label='Nodes')

        if show_labels:
            for i, coord in enumerate(node_coords):
                ax.annotate(str(i), xy=coord, xytext=(5, 5),
                            textcoords='offset points', fontsize=8,
                            color='darkred', fontweight='bold')

    def _format_axes(self, ax, title, n_blocks, n_nodes, n_cfs, show_deformed, scale):
        """Helper for axis formatting"""
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlabel('x [m]', fontsize=11)
        ax.set_ylabel('y [m]', fontsize=11)

        if title is None:
            title = f'Structure_block: {n_blocks} blocks, {n_nodes} nodes'
            if n_cfs > 0:
                title += f', {n_cfs} contact faces'
            if show_deformed:
                title += f' (deformed, scale={scale})'

        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        plt.tight_layout()


class BeamBlock(Structure_block):
    """Structure_block specialization for beam-like assemblies."""

    def __init__(
            self, N1, N2, n_blocks, h, rho, b=1, material=None, end_1=True, end_2=True
    ):
        blockList = []
        lx = N2[0] - N1[0]
        ly = N2[1] - N1[1]
        L = np.sqrt(lx ** 2 + ly ** 2)
        L_b = L / (n_blocks - 1)

        long = np.array([lx, ly]) / L
        tran = np.array([-ly, lx]) / L

        # Loop to create the blocks
        ref_point = N1.copy()

        for i in np.arange(n_blocks):
            # Initialize array of vertices
            vertices = np.array([ref_point, ref_point, ref_point, ref_point])

            if i == 0:  # First block is half block
                vertices[0] += L_b / 2 * long - h / 2 * tran
                vertices[1] += L_b / 2 * long + h / 2 * tran
                vertices[2] += h / 2 * tran
                vertices[3] += -h / 2 * tran

                if end_1:
                    ref = ref_point
                else:
                    ref = None

            elif i == n_blocks - 1:  # Last block is also a half_block
                vertices[0] += -h / 2 * tran
                vertices[1] += h / 2 * tran
                vertices[2] += h / 2 * tran - L_b / 2 * long
                vertices[3] += -h / 2 * tran - L_b / 2 * long

                if end_2:
                    ref = ref_point
                else:
                    ref = None

            else:
                vertices[0] += -h / 2 * tran + L_b / 2 * long
                vertices[1] += h / 2 * tran + L_b / 2 * long
                vertices[2] += h / 2 * tran - L_b / 2 * long
                vertices[3] += -h / 2 * tran - L_b / 2 * long
                ref = None

            blockList.append(Block_2D(vertices, rho, b=b, material=material, ref_point=ref))
            ref_point += L_b * long

        super().__init__(blockList)


class TaperedBeamBlock(Structure_block):
    def __init__(self, N1, N2, n_blocks, h1, h2, rho, b=1, material=None, contact=None, end_1=True, end_2=True):
        """
        Add a tapered beam (varying height) made of blocks.

        Parameters
        ----------
        N1, N2 : array-like
            Start and end points of beam centerline
        n_blocks : int
            Number of blocks
        h1, h2 : float
            Heights at start and end [m]
        rho : float
            Density [kg/m³]
        b : float, optional
            Out-of-plane thickness [m]
        material : Material, optional
            Material model
        contact : optional
            Contact parameters (deprecated)
        end_1, end_2 : bool, optional
            Whether to include end reference points
        """
        blockList = []
        lx = N2[0] - N1[0]
        ly = N2[1] - N1[1]
        L = np.sqrt(lx ** 2 + ly ** 2)
        L_b = L / (n_blocks - 1)

        heights = np.linspace(h1, h2, n_blocks)
        d_h = (heights[1] - heights[0]) / 2

        long = np.array([lx, ly]) / L
        tran = np.array([-ly, lx]) / L

        # Loop to create the blocks
        ref_point = N1.copy()

        for i in np.arange(n_blocks):
            # Initialize array of vertices
            vertices = np.array([ref_point, ref_point, ref_point, ref_point])

            if i == 0:  # First block is half block
                h1 = heights[i]
                h2 = heights[i] + d_h
                vertices[0] += L_b / 2 * long - h2 / 2 * tran
                vertices[1] += L_b / 2 * long + h2 / 2 * tran
                vertices[2] += h1 / 2 * tran
                vertices[3] += -h1 / 2 * tran

                if end_1:
                    ref = ref_point
                else:
                    ref = None

            elif i == n_blocks - 1:  # Last block is also a half_block
                h2 = heights[i]
                h1 = heights[i] - d_h
                vertices[0] += -h2 / 2 * tran
                vertices[1] += h2 / 2 * tran
                vertices[2] += h1 / 2 * tran - L_b / 2 * long
                vertices[3] += -h1 / 2 * tran - L_b / 2 * long

                if end_2:
                    ref = ref_point
                else:
                    ref = None

            else:
                h1 = heights[i] - d_h
                h2 = heights[i] + d_h
                vertices[0] += -h2 / 2 * tran + L_b / 2 * long
                vertices[1] += h2 / 2 * tran + L_b / 2 * long
                vertices[2] += h1 / 2 * tran - L_b / 2 * long
                vertices[3] += -h1 / 2 * tran - L_b / 2 * long
                ref = None

            blockList.append(Block_2D(vertices, rho, b=b, material=material, ref_point=ref))

            ref_point += L_b * long
        super().__init__(blockList)


class ArchBlock(Structure_block):
    """Structure_block specialization for arch assemblies."""

    def __init__(self, c, a1, a2, R, n_blocks, h, rho, b=1, material=None, contact=None):
        """
        Add an arch made of blocks.

        Parameters
        ----------
        c : array-like
            Center point of the arch
        a1, a2 : float
            Start and end angles [radians]
        R : float
            Mean radius of arch [m]
        n_blocks : int
            Number of blocks
        h : float
            Radial thickness of arch [m]
        rho : float
            Density [kg/m³]
        b : float, optional
            Out-of-plane thickness [m]
        material : Material, optional
            Material model
        contact : optional
            Contact parameters (deprecated)
        """
        blockList = []
        d_a = (a2 - a1) / n_blocks
        angle = a1

        R_int = R - h / 2
        R_out = R + h / 2

        for i in np.arange(n_blocks):
            # Initialize array of vertices
            vertices = np.array([c, c, c, c])

            unit_dir_1 = np.array([np.cos(angle), np.sin(angle)])
            unit_dir_2 = np.array([np.cos(angle + d_a), np.sin(angle + d_a)])
            vertices[0] += R_int * unit_dir_1
            vertices[1] += R_out * unit_dir_1
            vertices[2] += R_out * unit_dir_2
            vertices[3] += R_int * unit_dir_2

            # print(vertices)
            blockList.append(Block_2D(vertices, rho, b=b, material=material))

            angle += d_a
        super().__init__(blockList)


class WallBlock(Structure_block):
    """Structure_block specialization for masonry wall assemblies."""

    def __init__(self, c1, l_block, h_block, pattern, rho, b=1, material=None, orientation=None):
        """
        Add a masonry wall with specified pattern.

        Parameters
        ----------
        c1 : array-like
            Origin point of the wall
        l_block : float
            Standard block length [m]
        h_block : float
            Block height [m]
        pattern : list of lists
            Wall pattern where each row is a list of block length multipliers
            Positive values = full block, negative = gap
        rho : float
            Density [kg/m³]
        b : float, optional
            Out-of-plane thickness [m]
        material : Material, optional
            Material model
        orientation : array-like, optional
            Orientation vector (default is [1, 0])
        """
        blockList = []
        if orientation is not None:
            long = orientation
            tran = np.array([-orientation[1], orientation[0]])
        else:
            long = np.array([1, 0], dtype=float)
            tran = np.array([0, 1], dtype=float)

        for j, line in enumerate(pattern):
            ref_point = (
                    c1 + 0.5 * abs(line[0]) * l_block * long + (j + 0.5) * h_block * tran
            )

            for i, brick in enumerate(line):
                if brick > 0:
                    vertices = np.array([ref_point, ref_point, ref_point, ref_point])
                    vertices[0] += brick * l_block / 2 * long - h_block / 2 * tran
                    vertices[1] += brick * l_block / 2 * long + h_block / 2 * tran
                    vertices[2] += -brick * l_block / 2 * long + h_block / 2 * tran
                    vertices[3] += -brick * l_block / 2 * long - h_block / 2 * tran

                    blockList.append(Block_2D(vertices, rho, b=b, material=material))

                if not i == len(line) - 1:
                    ref_point += 0.5 * l_block * long * (abs(brick) + abs(line[i + 1]))
        super().__init__(blockList)


class VoronoiBlock(Structure_block):
    """Structure_block specialization for Voronoi tessellation assemblies."""

    def __init__(self, surface, list_of_points, rho, b=1, material=None):
        """
        Add blocks using Voronoi tessellation within a surface.

        Parameters
        ----------
        surface : array-like
            List of points defining the boundary surface
        list_of_points : array-like
            Points to use as Voronoi cell centers
        rho : float
            Density [kg/m³]
        b : float, optional
            Out-of-plane thickness [m]
        material : Material, optional
            Material model
        """
        # Surface is a list of points defining the surface to be subdivided into
        # Voronoi cells.

        blockList = []

        def point_in_surface(point, surface):
            # Check if a point lies on the surface
            # Surface is a list of points delimiting the surface
            # Point is a 2D numpy array

            n = len(surface)

            for i in range(n):
                A = surface[i]
                B = surface[(i + 1) % n]
                C = point

                if np.cross(B - A, C - A) < 0:
                    return False

            return True

        for point in list_of_points:
            # Check if all points lie on the surface
            if not point_in_surface(point, surface):
                warnings.warn("Not all points lie on the surface")
                return

        # Create Voronoi cells
        vor = sc.spatial.Voronoi(list_of_points)

        # Create block for each Voronoi region
        # If region is finite, it's easy
        # If region is infinite, delimit it with the edge of the surface
        for region in vor.regions[1:]:
            if not -1 in region:
                vertices = np.array([vor.vertices[i] for i in region])
                blockList.append(Block_2D(vertices, rho, b=b, material=material))

            else:
                vertices = []
                for i in region:
                    if not i == -1:
                        vertices.append(vor.vertices[i])

                # Find the edges of the surface that intersect the infinite cell
                for i in range(len(vertices)):
                    A = vertices[i]
                    B = vertices[(i + 1) % len(vertices)]

                    for j in range(len(surface)):
                        C = surface[j]
                        D = surface[(j + 1) % len(surface)]

                        if np.cross(B - A, C - A) * np.cross(B - A, D - A) < 0:
                            # Intersection between AB and CD
                            if np.cross(D - C, A - C) * np.cross(D - C, B - C) < 0:
                                # Intersection between CD and AB
                                vertices.insert(
                                    i + 1,
                                    C
                                    + np.cross(D - C, A - C)
                                    / np.cross(D - C, B - C)
                                    * (B - A),
                                )
                                vertices.insert(i + 2, D)
                                break
                blockList.append(Block_2D(np.array(vertices), rho, b=b, material=material))
        super().__init__(blockList)
