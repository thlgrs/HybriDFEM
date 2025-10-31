# Standart imports
import warnings
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from Theo.Objects.FEM.FE import FE
from Theo.Structures.Structure_2D import Structure_2D


class Structure_FEM(Structure_2D):
    def __init__(self):
        super().__init__()
        self.list_fes: List[FE] = []

    # Construction methods
    def add_fe(self, nodes, mat,
               geom):  # TODO append Triangle from FE.py and Timosnhenko from Timoshenko.py attention not the same dof per nodes.
        pass

    # Generation methods
    def _make_nodes_fem(self):
        for fe in self.list_fes:
            # Get DOF count from element (2 for Element2D, 3 for Timoshenko)
            dof_count = getattr(fe, 'DOFS_PER_NODE', 3)
            for j, node in enumerate(fe.nodes):
                index = self._add_node_if_new(node,
                                              dof_count=dof_count)  # new or existing index of the node of the element in Structure_2D
                fe.make_connect(index, j, structure=self)  # create the connection vector of the element

    def make_nodes(self):
        self._make_nodes_fem()

        # Use flexible DOF calculation (supports variable DOFs per node)
        self.nb_dofs = self.compute_nb_dofs()
        self.U = np.zeros(self.nb_dofs, dtype=float)
        self.P = np.zeros(self.nb_dofs, dtype=float)
        self.P_fixed = np.zeros(self.nb_dofs, dtype=float)

        # REMOVED: rotation DOF auto-fixing workaround
        # 2D elements now properly use 2 DOFs per node via flexible DOF system
        self.dof_fix = np.array([], dtype=int)
        self.dof_free = np.arange(self.nb_dofs, dtype=int)

        self.nb_dof_fix = len(self.dof_fix)
        self.nb_dof_free = len(self.dof_free)

    # Solving methods
    def _get_P_r_fem(self):
        self.dofs_defined()
        if not hasattr(self, "P_r"):
            self.P_r = np.zeros(self.nb_dofs, dtype=float)

        for fe in self.list_fes:
            q_glob = self.U[fe.dofs]
            p_glob = fe.get_p_glob(q_glob)
            self.P_r[fe.dofs] += p_glob

    def get_P_r(self):
        self.P_r = np.zeros(self.nb_dofs, dtype=float)
        self._get_P_r_fem()

    def _mass_fem(self, no_inertia: bool = False):
        for fe in getattr(self, "list_fes", []):
            mass_fe = fe.get_mass(no_inertia=no_inertia)
            if mass_fe is None:
                continue
            dofs = np.asarray(fe.dofs, dtype=int)
            self.M[np.ix_(dofs, dofs)] += mass_fe

    def get_M_str(self, no_inertia: bool = False):
        self.dofs_defined()
        self.M = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._mass_fem(no_inertia=no_inertia)
        return self.M

    def _stiffness_fem(self):
        for fe in getattr(self, "list_fes", []):
            k_glob = fe.get_k_glob()
            dofs = np.asarray(fe.dofs, dtype=int)
            self.K[np.ix_(dofs, dofs)] += k_glob

    def _stiffness0_fem(self):
        for fe in getattr(self, "list_fes", []):
            k_glob0 = fe.get_k_glob0()
            dofs = np.asarray(fe.dofs, dtype=int)
            self.K0[np.ix_(dofs, dofs)] += k_glob0

    def _stiffness_LG_fem(self):
        for fe in getattr(self, "list_fes", []):
            k_glob_LG = fe.get_k_glob_LG()
            dofs = np.asarray(fe.dofs, dtype=int)
            self.K_LG[np.ix_(dofs, dofs)] += k_glob_LG

    def get_K_str(self):
        self.dofs_defined()
        self.K = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness_fem()
        return self.K

    def get_K_str0(self):
        self.dofs_defined()
        self.K0 = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness0_fem()
        return self.K0

    def get_K_str_LG(self):
        self.dofs_defined()
        self.K_LG = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness_LG_fem()
        return self.K_LG

    def set_lin_geom(self, lin_geom=True):
        for fe in self.list_fes:
            fe.lin_geom = lin_geom

    def plot(
            self,
            ax: Optional[plt.Axes] = None,
            show_nodes: bool = True,
            show_node_labels: bool = False,
            show_elements: bool = True,
            show_element_labels: bool = False,
            show_deformed: bool = False,
            deformation_scale: float = 1.0,
            element_subdivisions: int = 10,
            node_size: float = 60,
            node_color: str = 'blue',
            element_color_undef: str = 'black',
            element_color_def: str = 'red',
            element_linewidth: float = 2,
            mesh_color: str = 'gray',
            mesh_alpha: float = 0.3,
            title: Optional[str] = None,
            figsize: Tuple[float, float] = (14, 10),
            **kwargs
    ) -> plt.Figure:
        """
        Plot Structure_FEM with elements and nodes.

        Additional Parameters (beyond base class)
        ------------------------------------------
        show_elements : bool, default=True
            Whether to display elements
        show_element_labels : bool, default=False
            Whether to label elements
        element_subdivisions : int, default=10
            Points for curved beam visualization
        node_color : str, default='blue'
            Color for node markers
        element_color_undef : str, default='black'
            Color for undeformed elements
        element_color_def : str, default='red'
            Color for deformed elements
        mesh_color : str, default='gray'
            Color for 2D element mesh
        mesh_alpha : float, default=0.3
            Transparency for mesh
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

        # Plot elements
        if show_elements:
            self._plot_elements(
                ax,
                nodes,
                show_deformed,
                deformation_scale,
                displacements,
                element_subdivisions,
                element_color_undef,
                element_color_def,
                element_linewidth,
                mesh_color,
                mesh_alpha,
                show_element_labels
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
        self._format_axes(ax, title, len(self.list_fes), len(nodes),
                          show_deformed, deformation_scale)

        return fig

    def _plot_elements(self, ax, nodes, show_deformed, scale, displacements,
                       subdivisions, color_undef, color_def, linewidth,
                       mesh_color, mesh_alpha, show_labels):
        """Helper to plot FE elements"""
        for elem_idx, elem in enumerate(self.list_fes):
            if not hasattr(elem, 'connect'):
                continue

            # Detect element type
            is_beam = hasattr(elem, 'L') or 'Timoshenko' in elem.__class__.__name__

            if is_beam and len(elem.connect) >= 2:
                self._plot_beam_element(
                    ax, elem, nodes, show_deformed, scale, displacements,
                    subdivisions, color_undef, color_def, linewidth, elem_idx
                )
            else:
                self._plot_2d_element(
                    ax, elem, nodes, show_deformed, scale, displacements,
                    mesh_color, mesh_alpha, color_def, elem_idx
                )

            if show_labels:
                self._add_element_label(ax, elem, nodes, elem_idx)

    def _plot_beam_element(self, ax, elem, nodes, show_deformed, scale,
                           displacements, subdivisions, color_undef,
                           color_def, linewidth, idx):
        """Helper to plot beam elements"""
        i1, i2 = elem.connect[0], elem.connect[1]
        n1, n2 = nodes[i1], nodes[i2]

        # Undeformed
        if not show_deformed:
            ax.plot([n1[0], n2[0]], [n1[1], n2[1]],
                    color=color_undef, linewidth=linewidth, zorder=3)

        # Deformed with shape functions
        if show_deformed and displacements is not None and hasattr(elem, 'dofs'):
            if len(elem.dofs) >= 6:
                u_elem = displacements[elem.dofs] * scale

                # Beam geometry
                L = np.sqrt((n2[0] - n1[0]) ** 2 + (n2[1] - n1[1]) ** 2)
                cos_a = (n2[0] - n1[0]) / L
                sin_a = (n2[1] - n1[1]) / L

                # Interpolate deformed shape
                xi = np.linspace(0, 1, subdivisions)
                x_def = np.zeros((subdivisions, 2))

                for k, xi_k in enumerate(xi):
                    # Hermite shape functions
                    N1 = 1 - 3 * xi_k ** 2 + 2 * xi_k ** 3
                    N2 = xi_k - 2 * xi_k ** 2 + xi_k ** 3
                    N3 = 3 * xi_k ** 2 - 2 * xi_k ** 3
                    N4 = -xi_k ** 2 + xi_k ** 3

                    u_loc = N1 * u_elem[0] + N3 * u_elem[3]
                    v_loc = N1 * u_elem[1] + N2 * L * u_elem[2] + \
                            N3 * u_elem[4] + N4 * L * u_elem[5]

                    x_def[k, 0] = n1[0] + xi_k * L * cos_a + u_loc * cos_a - v_loc * sin_a
                    x_def[k, 1] = n1[1] + xi_k * L * sin_a + u_loc * sin_a + v_loc * cos_a

                ax.plot(x_def[:, 0], x_def[:, 1], color=color_def,
                        linewidth=linewidth, zorder=4,
                        label='Deformed' if idx == 0 else '')
                ax.plot([n1[0], n2[0]], [n1[1], n2[1]],
                        color=color_undef, linewidth=linewidth * 0.5,
                        linestyle='--', alpha=0.3, zorder=2,
                        label='Undeformed' if idx == 0 else '')

    def _plot_2d_element(self, ax, elem, nodes, show_deformed, scale,
                         displacements, mesh_color, mesh_alpha, color_def, idx):
        """Helper to plot 2D solid elements"""
        elem_nodes = np.array([nodes[i] for i in elem.connect])
        n_nodes = len(elem_nodes)

        for i in range(n_nodes):
            j = (i + 1) % n_nodes

            if not show_deformed:
                ax.plot([elem_nodes[i, 0], elem_nodes[j, 0]],
                        [elem_nodes[i, 1], elem_nodes[j, 1]],
                        color=mesh_color, linewidth=1, alpha=mesh_alpha, zorder=2)

            if show_deformed and displacements is not None and hasattr(elem, 'dofs'):
                if len(elem.dofs) >= 2 * n_nodes:
                    u_nodes = np.zeros((n_nodes, 2))
                    for k in range(n_nodes):
                        u_nodes[k, 0] = displacements[elem.dofs[2 * k]] * scale
                        u_nodes[k, 1] = displacements[elem.dofs[2 * k + 1]] * scale

                    def_nodes = elem_nodes + u_nodes
                    ax.plot([def_nodes[i, 0], def_nodes[j, 0]],
                            [def_nodes[i, 1], def_nodes[j, 1]],
                            color=color_def, linewidth=1.5, alpha=0.8, zorder=3)

    def _add_element_label(self, ax, elem, nodes, idx):
        """Helper to add element labels"""
        elem_nodes = np.array([nodes[i] for i in elem.connect])
        centroid = np.mean(elem_nodes, axis=0)
        ax.text(centroid[0], centroid[1], f'E{idx}',
                ha='center', va='center', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', alpha=0.6))

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
                   edgecolors='darkblue', linewidths=1.5,
                   zorder=5, label='Nodes')

        if show_labels:
            for i, coord in enumerate(node_coords):
                ax.annotate(str(i), xy=coord, xytext=(5, 5),
                            textcoords='offset points', fontsize=7,
                            color='darkblue', fontweight='bold')

    def _format_axes(self, ax, title, n_elem, n_nodes, show_deformed, scale):
        """Helper for axis formatting"""
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlabel('x [m]', fontsize=11)
        ax.set_ylabel('y [m]', fontsize=11)

        if title is None:
            title = f'Structure_FEM: {n_elem} elements, {n_nodes} nodes'
            if show_deformed:
                title += f' (deformed, scale={scale})'

        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        plt.tight_layout()
