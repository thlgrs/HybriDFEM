# Standart imports
import warnings
from typing import Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np

from Theo.Structures.Structure_2D import Structure_2D
from Theo.Structures.Structure_FEM import Structure_FEM
from Theo.Structures.Structure_block import Structure_block


class Hybrid(Structure_block, Structure_FEM, Structure_2D):
    def __init__(self):
        super().__init__()
        self.list_hybrid_cfs = []
        self.hybrid_tolerance = 1e-6
        self.hybrid_n_integration_points = 2

        # Coupling infrastructure (SimpleCoupling integration)
        self.coupling_enabled = False
        self.coupled_fem_nodes = {}  # {fem_node_id: block_id}
        self.coupling_T = None  # Transformation matrix: u_full = T * u_reduced
        self.coupling_dof_map = None  # Maps reduced DOFs to full DOFs
        self.nb_dofs_reduced = None  # Number of reduced DOFs after coupling

    def make_nodes(self):
        self.list_nodes = []

        # Call each method separately
        self._make_nodes_block()
        self._make_nodes_fem()

        # Initialize DOFs (use flexible DOF calculation)
        self.nb_dofs = self.compute_nb_dofs()
        self.U = np.zeros(self.nb_dofs, dtype=float)
        self.P = np.zeros(self.nb_dofs, dtype=float)
        self.P_fixed = np.zeros(self.nb_dofs, dtype=float)
        self.dof_fix = np.array([], dtype=int)
        self.dof_free = np.arange(self.nb_dofs, dtype=int)
        self.nb_dof_fix = 0
        self.nb_dof_free = len(self.dof_free)

    def detect_coupled_fem_nodes(self, tolerance: float = 1e-9, verbose: bool = True) -> Dict[int, int]:
        """
        Detect which FEM nodes are coupled to rigid blocks.

        Uses nodal coincidence: if a FEM node is at the same location as a block
        reference point, they are coupled.

        Parameters
        ----------
        tolerance : float
            Distance tolerance for node matching
        verbose : bool
            Print detection information

        Returns
        -------
        coupled_nodes : dict
            {fem_node_id: block_id} mapping

        Notes
        -----
        This implements the constraint approach from SimpleCoupling.py:
        - FEM nodes that coincide with block nodes follow block rigid body motion
        - Constraint: u_fem_node = C * [u_block, v_block, theta_block]
        """
        coupled_nodes = {}

        # Block nodes are the first len(self.list_blocks) nodes
        n_blocks = len(self.list_blocks)

        for block_idx in range(n_blocks):
            block = self.list_blocks[block_idx]
            block_node_id = block.connect  # Global node ID for this block

            # Check all FEM nodes for proximity to this block
            for fem_node_id in range(n_blocks, len(self.list_nodes)):
                fem_node_pos = self.list_nodes[fem_node_id]
                block_node_pos = self.list_nodes[block_node_id]

                dist = np.linalg.norm(fem_node_pos - block_node_pos)

                if dist < tolerance:
                    coupled_nodes[fem_node_id] = block_idx
                    if verbose:
                        print(f"  Coupled FEM node {fem_node_id} to Block {block_idx} (dist={dist:.2e})")

        if verbose and coupled_nodes:
            print(f"\n{'=' * 70}")
            print(f"COUPLING DETECTION: Found {len(coupled_nodes)} coupled FEM nodes")
            print(f"{'=' * 70}\n")
        elif verbose:
            print("\nWarning: No coupled FEM nodes detected!")

        return coupled_nodes

    def build_coupling_transformation(self, verbose: bool = True):
        """
        Build transformation matrix T for block-FEM coupling.

        The transformation enforces kinematic constraints:
            u_full = T * u_reduced

        Where:
        - u_full: All DOFs (blocks + all FEM nodes)
        - u_reduced: Independent DOFs (blocks + free FEM nodes)

        For coupled FEM nodes: u_fem = C * q_block (rigid body constraint)
        For free FEM nodes: u_fem = u_fem (identity)

        This implements the approach from SimpleCoupling.py.

        Parameters
        ----------
        verbose : bool
            Print transformation information

        Notes
        -----
        The reduced system is:
            K_reduced = T^T * K_full * T
            F_reduced = T^T * F_full
            u_full = T * u_reduced
        """
        if not self.coupled_fem_nodes:
            if verbose:
                print("No coupled nodes - coupling transformation not built")
            return

        n_dofs_full = self.nb_dofs
        n_blocks = len(self.list_blocks)

        # Count free FEM nodes (not coupled)
        all_fem_node_ids = set(range(n_blocks, len(self.list_nodes)))
        coupled_fem_node_ids = set(self.coupled_fem_nodes.keys())
        free_fem_node_ids = sorted(all_fem_node_ids - coupled_fem_node_ids)

        # Reduced DOFs = block DOFs + free FEM node DOFs
        # Use actual DOF counts instead of hardcoded 3
        n_dofs_reduced = sum(self.node_dof_counts[self.list_blocks[i].connect] for i in range(n_blocks))
        n_dofs_reduced += sum(self.node_dof_counts[node_id] for node_id in free_fem_node_ids)
        self.nb_dofs_reduced = n_dofs_reduced

        # Build transformation matrix T (n_dofs_full x n_dofs_reduced)
        T = np.zeros((n_dofs_full, n_dofs_reduced))

        # Map block DOFs (identity - blocks are always independent)
        reduced_dof_counter = 0
        for block_idx in range(n_blocks):
            block_node_id = self.list_blocks[block_idx].connect
            node_dof_count = self.node_dof_counts[block_node_id]
            base_global_dof = self.node_dof_offsets[block_node_id]

            for dof_idx in range(node_dof_count):  # Use actual DOF count
                global_dof = base_global_dof + dof_idx
                reduced_dof = reduced_dof_counter + dof_idx
                T[global_dof, reduced_dof] = 1.0

            reduced_dof_counter += node_dof_count

        # Map coupled FEM nodes (constraint matrix)
        for fem_node_id, block_idx in self.coupled_fem_nodes.items():
            block = self.list_blocks[block_idx]
            block_node_id = block.connect
            fem_node_pos = self.list_nodes[fem_node_id]

            # Get constraint matrix: u_fem = C * q_block
            C = block.constraint_matrix_for_node(fem_node_pos)

            # Get actual DOF counts for FEM node and block
            fem_node_dof_count = self.node_dof_counts[fem_node_id]
            block_dof_count = self.node_dof_counts[block_node_id]
            base_fem_global_dof = self.node_dof_offsets[fem_node_id]

            # Compute reduced DOF offset for this block
            block_reduced_dof_offset = sum(self.node_dof_counts[self.list_blocks[i].connect] for i in range(block_idx))

            # Map the FEM DOFs through constraint (typically 2 DOFs: u, v)
            for local_dof in range(min(fem_node_dof_count, 2)):  # Only u, v for FEM nodes
                global_dof = base_fem_global_dof + local_dof

                # Express this DOF in terms of block's reduced DOFs
                for block_dof_idx in range(block_dof_count):  # Block has u, v, theta
                    reduced_dof = block_reduced_dof_offset + block_dof_idx
                    T[global_dof, reduced_dof] = C[local_dof, block_dof_idx]

        # Map free FEM nodes (identity)
        for i, fem_node_id in enumerate(free_fem_node_ids):
            fem_node_dof_count = self.node_dof_counts[fem_node_id]
            base_fem_global_dof = self.node_dof_offsets[fem_node_id]

            for dof_idx in range(fem_node_dof_count):  # Use actual DOF count
                global_dof = base_fem_global_dof + dof_idx
                reduced_dof = reduced_dof_counter + dof_idx
                T[global_dof, reduced_dof] = 1.0

            reduced_dof_counter += fem_node_dof_count

        self.coupling_T = T
        self.coupling_enabled = True

        # Update DOF count for solver (store full count for later expansion)
        self.nb_dofs_full = n_dofs_full
        self.nb_dofs = n_dofs_reduced

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"COUPLING TRANSFORMATION BUILT")
            print(f"{'=' * 70}")
            print(f"  Full DOFs: {n_dofs_full}")
            print(f"  Reduced DOFs: {n_dofs_reduced}")
            print(f"  DOF reduction: {n_dofs_full - n_dofs_reduced}")
            print(f"  Coupled FEM nodes: {len(self.coupled_fem_nodes)}")
            print(f"  Free FEM nodes: {len(free_fem_node_ids)}")
            print(f"{'=' * 70}\n")

    def enable_block_fem_coupling(self, tolerance: float = 1e-9, verbose: bool = True):
        """
        Enable constraint-based coupling between blocks and FEM elements.

        This is the main method to activate coupling in a Hybrid structure.

        Algorithm:
        ----------
        1. Detect which FEM nodes coincide with block nodes
        2. Build transformation matrix T enforcing kinematic constraints
        3. Enable coupling flag (affects stiffness/mass assembly)

        Parameters
        ----------
        tolerance : float
            Distance tolerance for detecting coupled nodes
        verbose : bool
            Print coupling information

        Usage
        -----
        >>> St = Hybrid()
        >>> St.add_block(...)
        >>> St.add_fe(...)
        >>> St.make_nodes()
        >>> St.enable_block_fem_coupling()  # Activate coupling
        >>> St.get_K_str0()  # Assemble with coupling

        Notes
        -----
        Based on SimpleCoupling.py constraint transformation approach.
        Must be called after make_nodes() and before matrix assembly.
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"ENABLING BLOCK-FEM COUPLING")
            print(f"{'=' * 70}\n")

        # Step 1: Detect coupled nodes
        self.coupled_fem_nodes = self.detect_coupled_fem_nodes(tolerance=tolerance, verbose=verbose)

        if not self.coupled_fem_nodes:
            if verbose:
                print("Warning: No coupled nodes detected - coupling not enabled")
                print("Check that FEM nodes coincide with block nodes\n")
            self.coupling_enabled = False
            return

        # Step 2: Build transformation matrix
        self.build_coupling_transformation(verbose=verbose)

        if verbose:
            print(f"[SUCCESS] Block-FEM coupling enabled!")
            print(f"  Use get_K_str0() to assemble coupled stiffness\n")

    def expand_displacement(self, u_reduced: np.ndarray) -> np.ndarray:
        """
        Expand displacement from reduced DOFs to full DOFs.

        This method applies the transformation: u_full = T * u_reduced

        After solving the reduced coupled system, use this to get displacements
        for all nodes (including coupled FEM nodes).

        Parameters
        ----------
        u_reduced : np.ndarray
            Displacement vector in reduced DOF space (size: nb_dofs_reduced)

        Returns
        -------
        u_full : np.ndarray
            Displacement vector in full DOF space (size: nb_dofs_full)

        Usage
        -----
        >>> St = Hybrid()
        >>> St.enable_block_fem_coupling()
        >>> St_solved = Static.solve_linear(St)  # Solves reduced system
        >>> u_full = St_solved.expand_displacement(St_solved.U)  # Expand to full
        """
        if not self.coupling_enabled:
            # No coupling - just return as-is
            return u_reduced

        if not hasattr(self, 'coupling_T'):
            raise RuntimeError("Coupling transformation matrix not built")

        # Apply transformation: u_full = T * u_reduced
        u_full = self.coupling_T @ u_reduced

        return u_full

    def _get_P_r_hybrid(self):
        """
        Add hybrid coupling forces to residual.
        """
        if not hasattr(self, 'list_hybrid_cfs'):
            return

        if not hasattr(self, 'P_r'):
            raise RuntimeError("Residual P_r not initialized")

        if not hasattr(self, 'U'):
            warnings.warn("Displacement U not found - cannot compute coupling forces")
            return

        for cf in self.list_hybrid_cfs:
            f_cf, dof_indices = cf.get_pf_glob(self.U)

            if f_cf is not None:
                dof_array = np.array(dof_indices)
                self.P_r[dof_array] += f_cf

    def get_P_r(self):
        self.dofs_defined()

        # Assemble full force vector
        if self.coupling_enabled:
            # Use full DOF count for assembly
            nb_dofs_full = len(self.list_nodes) * 3
            self.P_r = np.zeros(nb_dofs_full, dtype=float)
        else:
            self.P_r = np.zeros(self.nb_dofs, dtype=float)

        self._get_P_r_block()
        self._get_P_r_fem()
        self._get_P_r_hybrid()

        # Apply coupling transformation if enabled
        if self.coupling_enabled:
            P_full = self.P_r
            self.P_r = self.coupling_T.T @ P_full

        return self.P_r

    def get_M_str(self, no_inertia: bool = False):
        self.dofs_defined()

        # Assemble full mass matrix
        if self.coupling_enabled:
            nb_dofs_full = len(self.list_nodes) * 3
            self.M = np.zeros((nb_dofs_full, nb_dofs_full), dtype=float)
        else:
            self.M = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)

        # Compose contributions
        self._mass_block(no_inertia=no_inertia)
        self._mass_fem(no_inertia=no_inertia)

        # Apply coupling transformation if enabled
        if self.coupling_enabled:
            M_full = self.M
            self.M = self.coupling_T.T @ M_full @ self.coupling_T

        return self.M

    def _stiffness_hybrid(self):
        """
        Add hybrid coupling stiffness to global matrix.
        """
        if not hasattr(self, 'list_hybrid_cfs'):
            return

        if not hasattr(self, 'K'):
            raise RuntimeError("Global stiffness K not initialized")

        for cf in self.list_hybrid_cfs:
            K_cf, dof_indices = cf.get_kf_glob(getattr(self, 'U', None))

            if K_cf is not None:
                dof_array = np.array(dof_indices)
                self.K[np.ix_(dof_array, dof_array)] += K_cf

    def get_K_str(self):
        self.dofs_defined()

        # Assemble full stiffness matrix
        if self.coupling_enabled:
            nb_dofs_full = len(self.list_nodes) * 3
            self.K = np.zeros((nb_dofs_full, nb_dofs_full), dtype=float)
        else:
            self.K = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)

        self._stiffness_block()
        self._stiffness_fem()
        self._stiffness_hybrid()

        # Apply coupling transformation if enabled
        if self.coupling_enabled:
            K_full = self.K
            self.K = self.coupling_T.T @ K_full @ self.coupling_T

        return self.K

    def get_K_str0(self):
        self.dofs_defined()

        # Assemble full initial stiffness matrix
        if self.coupling_enabled:
            nb_dofs_full = len(self.list_nodes) * 3
            self.K0 = np.zeros((nb_dofs_full, nb_dofs_full), dtype=float)
        else:
            self.K0 = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)

        self._stiffness0_block()
        self._stiffness0_fem()

        # Apply coupling transformation if enabled
        if self.coupling_enabled:
            K0_full = self.K0
            self.K0 = self.coupling_T.T @ K0_full @ self.coupling_T

        return self.K0

    def get_K_str_LG(self):
        self.dofs_defined()

        # Assemble full large geometry stiffness matrix
        if self.coupling_enabled:
            nb_dofs_full = len(self.list_nodes) * 3
            self.K_LG = np.zeros((nb_dofs_full, nb_dofs_full), dtype=float)
        else:
            self.K_LG = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)

        self._stiffness_LG_block()
        self._stiffness_LG_fem()

        # Apply coupling transformation if enabled
        if self.coupling_enabled:
            K_LG_full = self.K_LG
            self.K_LG = self.coupling_T.T @ K_LG_full @ self.coupling_T

        return self.K_LG

    def set_lin_geom(self, lin_geom=True):
        for cf in self.list_cfs:
            cf.set_lin_geom(lin_geom)

        for fe in self.list_fes:
            fe.lin_geom = lin_geom

    def plot(
            self,
            ax: Optional[plt.Axes] = None,
            # Node options
            show_nodes: bool = True,
            show_node_labels: bool = False,
            node_size: float = 70,
            # Block options
            show_blocks: bool = True,
            show_block_labels: bool = False,
            show_contact_faces: bool = True,
            block_facecolor: str = 'lightblue',
            block_edgecolor: str = 'black',
            block_alpha: float = 0.6,
            contact_color: str = 'green',
            contact_linewidth: float = 2,
            # FEM options
            show_elements: bool = True,
            show_element_labels: bool = False,
            element_subdivisions: int = 10,
            element_color_undef: str = 'black',
            element_color_def: str = 'red',
            element_linewidth: float = 2,
            mesh_color: str = 'gray',
            mesh_alpha: float = 0.3,
            # Deformation options
            show_deformed: bool = False,
            deformation_scale: float = 1.0,
            # General options
            title: Optional[str] = None,
            figsize: Tuple[float, float] = (16, 12),
            **kwargs
    ) -> plt.Figure:
        """
        Plot Hybrid structure combining blocks and FEM elements.

        This method intelligently combines plotting from both parent classes:
        - Uses Structure_block methods for blocks and contact faces
        - Uses Structure_FEM methods for finite elements
        - Provides unified visualization of the complete hybrid structure

        The method reuses parent class helper methods to avoid code duplication
        while providing a unified interface for the hybrid structure.

        Parameters
        ----------
        All parameters from both Structure_block.plot() and Structure_FEM.plot()
        are supported. See parent class documentation for details.

        Key Hybrid-Specific Behavior
        -----------------------------
        - Blocks use red/pink colors for distinction
        - FEM elements use blue/gray colors for distinction
        - Contact faces shown in green
        - Automatic title shows counts of both blocks and elements
        - Both block and FEM deformations handled consistently

        Returns
        -------
        fig : matplotlib.figure.Figure
            The combined figure

        Notes
        -----
        This method demonstrates intelligent reuse of parent class methods:
        1. Delegates block plotting to Structure_block._plot_blocks()
        2. Delegates contact plotting to Structure_block._plot_contact_faces()
        3. Delegates element plotting to Structure_FEM._plot_elements()
        4. Combines all on single axes for unified visualization
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

        # =====================================================================
        # INTELLIGENT DELEGATION TO PARENT METHODS
        # =====================================================================

        # 1. Plot blocks using Structure_block's method
        if show_blocks and self.list_blocks:
            Structure_block._plot_blocks(
                self,  # Pass self to access parent method
                ax,
                show_deformed,
                deformation_scale,
                displacements,
                block_facecolor,
                block_edgecolor,
                block_alpha,
                show_block_labels
            )

        # 2. Plot contact faces using Structure_block's method
        if show_contact_faces and self.list_cfs:
            Structure_block._plot_contact_faces(
                self,  # Pass self to access parent method
                ax,
                show_deformed,
                deformation_scale,
                displacements,
                contact_color,
                contact_linewidth
            )

        # 3. Plot FEM elements using Structure_FEM's method
        if show_elements and self.list_fes:
            Structure_FEM._plot_elements(
                self,  # Pass self to access parent method
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

        # 4. Plot nodes (common to both, use either parent's method)
        # We'll use Structure_block's version with adjusted color
        if show_nodes:
            # Use purple color to distinguish hybrid nodes
            hybrid_node_color = kwargs.get('node_color', 'purple')
            Structure_block._plot_nodes(
                self,
                ax,
                nodes,
                show_deformed,
                deformation_scale,
                displacements,
                node_size,
                hybrid_node_color,
                show_node_labels
            )

        # =====================================================================
        # CUSTOM FORMATTING FOR HYBRID
        # =====================================================================

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlabel('x [m]', fontsize=11)
        ax.set_ylabel('y [m]', fontsize=11)

        # Build comprehensive title
        if title is None:
            n_blocks = len(self.list_blocks) if hasattr(self, 'list_blocks') else 0
            n_elems = len(self.list_fes) if hasattr(self, 'list_fes') else 0
            n_nodes = len(self.list_nodes)
            n_cfs = len(self.list_cfs) if hasattr(self, 'list_cfs') else 0

            title = f'Hybrid Structure: {n_blocks} blocks'
            if n_elems > 0:
                title += f' + {n_elems} FEM elements'
            title += f', {n_nodes} nodes'
            if n_cfs > 0:
                title += f', {n_cfs} contact faces'
            if show_deformed:
                title += f' (deformed, scale={deformation_scale})'

        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

        # Custom legend for hybrid
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Remove duplicates
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(),
                      loc='best', fontsize=10, framealpha=0.9)

        plt.tight_layout()

        return fig
