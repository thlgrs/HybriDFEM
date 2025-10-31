"""
SimpleCoupling.py - Rigid Block and FEM Triangle Coupling

This module demonstrates coupling between rigid blocks and FEM triangle elements
using constraint transformation matrices. The coupling enforces kinematic compatibility
at the interface between discrete (rigid) and continuum (FEM) domains.

Classes:
    Block_2D: 2D rigid block with 3 DOFs (u, v, θ) from Theo.Objects.Block
    CoupledSystem: Couples rigid block with FEM element using constraint transformation

Key Features:
    - Rigid body kinematics for block motion
    - Constraint matrix formulation for interface nodes
    - Reduced-order system assembly via transformation T
    - Support for partial constraints (fixed DOFs)
    - Multiple example scenarios demonstrating different coupling configurations
    - Uses existing Block_2D class with new coupling methods

Mathematical Formulation:
    The coupling uses a transformation matrix T such that:
        u_full = T * u_reduced

    where u_full contains all FEM element DOFs (6 DOFs for triangle)
    and u_reduced contains only the independent DOFs (block DOFs + free node DOFs)

    The reduced stiffness is: K_reduced = T^T * K_full * T

Usage:
    See example_one() through example_four() for different coupling scenarios.

Updated: 2025-10-30
    - Updated to use new Triangle(nodes, mat, geom) interface
    - Triangle now inherits properly from Element2D
    - Uses get_k_glob() instead of direct K attribute access
    - Replaced RigidBlock2D with Block_2D from Theo.Objects.Block
    - Block_2D now has coupling methods: set_fixed(), get_free_dofs(),
      is_fully_fixed(), displacement_at_point(), constraint_matrix_for_node()
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix

from Theo.Objects.DFEM.Block import Block_2D
# ============================================================================
# IMPORTS
# ============================================================================
from Theo.Objects.FEM.FE import Triangle, Geometry2D
from Theo.Objects.Material.Material import PlaneStress


# ============================================================================
# COUPLED SYSTEM
# ============================================================================
class CoupledSystem:
    """Couple rigid block with FEM element"""

    def __init__(self, rigid_block, fem_element, interface_nodes, verbose=True):
        self.block = rigid_block
        self.element = fem_element
        self.interface_nodes = interface_nodes
        self.verbose = verbose

        self.n_dofs_full = 6

        all_nodes = {0, 1, 2}
        free_nodes = all_nodes - set(interface_nodes)
        self.free_nodes = sorted(list(free_nodes))

        if verbose:
            print(f"Interface nodes: {interface_nodes}")
            print(f"Free nodes: {self.free_nodes}")

        self.build_constraint_matrix()

    def build_constraint_matrix(self):
        """Build transformation matrix T: u_full = T * u_reduced"""
        if self.block.is_fully_fixed():
            n_free = len(self.free_nodes)
            self.n_dofs_reduced = 2 * n_free

            self.T = np.zeros((6, self.n_dofs_reduced))

            for i, node_idx in enumerate(self.free_nodes):
                self.T[2 * node_idx, 2 * i] = 1.0
                self.T[2 * node_idx + 1, 2 * i + 1] = 1.0

            if self.verbose:
                print(f"Block fully fixed")
                print(f"Reduced DOFs: {self.n_dofs_reduced}")

        else:
            block_free_dofs = self.block.get_free_dofs()
            n_block_free = len(block_free_dofs)
            n_free_nodes = len(self.free_nodes)

            self.n_dofs_reduced = n_block_free + 2 * n_free_nodes

            self.T = np.zeros((6, self.n_dofs_reduced))

            for node_idx in self.interface_nodes:
                node_pos = self.element.nodes[node_idx]
                C = self.block.constraint_matrix_for_node(node_pos)

                col = 0
                for block_dof in block_free_dofs:
                    self.T[2 * node_idx:2 * node_idx + 2, col] = C[:, block_dof]
                    col += 1

            for i, node_idx in enumerate(self.free_nodes):
                col_offset = n_block_free
                self.T[2 * node_idx, col_offset + 2 * i] = 1.0
                self.T[2 * node_idx + 1, col_offset + 2 * i + 1] = 1.0

            if self.verbose:
                print(f"Block free DOFs: {block_free_dofs}")
                print(f"Reduced DOFs: {self.n_dofs_reduced}")

    def reduce_stiffness(self):
        """Reduce stiffness matrix: K_red = T^T * K_full * T"""
        K_full = self.element.get_k_glob()
        self.K_reduced = self.T.T @ K_full @ self.T
        return self.K_reduced

    def reduce_force(self, F_full):
        """Reduce force vector: F_red = T^T * F_full"""
        return self.T.T @ F_full

    def expand_displacement(self, u_reduced):
        """Expand reduced displacement to full: u_full = T * u_reduced"""
        return self.T @ u_reduced

    def apply_displacement(self, u_reduced):
        """
        Apply prescribed displacement to the free DOFs.

        This is an alias for expand_displacement() for compatibility.

        Parameters
        ----------
        u_reduced : np.ndarray
            Reduced displacement vector (only free DOFs)

        Returns
        -------
        u_full : np.ndarray
            Full displacement vector (all element DOFs)
        """
        return self.expand_displacement(u_reduced)

    def compute_reactions(self, u_full, F_external):
        """
        Compute reaction forces using the constitutive law

        The constitutive chain:
        1. Displacement u → Strain ε = B*u
        2. Strain ε → Stress σ = D*ε
        3. Stress σ → Internal forces F_int = K*u = ∫B^T*σ dV
        4. Equilibrium: F_int = F_ext + F_reaction
        5. Therefore: F_reaction = F_int - F_ext = K*u - F_ext

        Parameters:
        -----------
        u_full : array (6,)
            Full displacement vector
        F_external : array (6,)
            External applied forces

        Returns:
        --------
        F_reaction : array (6,)
            Reaction forces at all DOFs
        """
        # Internal forces from constitutive law via stiffness
        F_internal = self.element.compute_nodal_forces(u_full)

        # Reaction forces maintain equilibrium
        # F_internal = F_external + F_reaction
        # F_reaction = F_internal - F_external
        F_reaction = F_internal - F_external

        return F_reaction, F_internal

    def compute_block_reactions(self, u_full, F_external):
        """
        Compute resultant force and moment on the rigid block

        This shows how forces from the FEM element are transferred
        to the rigid block through the interface nodes.

        Returns:
        --------
        F_block : array (2,)
            Resultant force on block [Fx, Fy]
        M_block : float
            Resultant moment on block
        nodal_reactions : array (n_interface, 2)
            Reaction forces at each interface node
        F_internal : array (6,)
            Internal forces at all nodes
        """
        # Get reaction forces using constitutive law
        F_reaction, F_internal = self.compute_reactions(u_full, F_external)

        # Extract reaction forces at interface nodes
        nodal_reactions = []
        interface_positions = []

        for node_idx in self.interface_nodes:
            Fx = F_reaction[2 * node_idx]
            Fy = F_reaction[2 * node_idx + 1]
            nodal_reactions.append([Fx, Fy])
            interface_positions.append(self.element.nodes[node_idx])

        nodal_reactions = np.array(nodal_reactions)
        interface_positions = np.array(interface_positions)

        # Compute resultant on block
        F_block, M_block = self.block.compute_resultant_force_moment(
            nodal_reactions, interface_positions
        )

        return F_block, M_block, nodal_reactions, F_internal

    def solve_displacement(self, F_full):
        """Solve the system with applied force"""
        K_red = self.reduce_stiffness()
        F_red = self.reduce_force(F_full)

        if self.verbose:
            print(f"\nReduced stiffness matrix ({K_red.shape[0]}×{K_red.shape[1]}):")
            print(K_red)
            print(f"\nReduced force vector:")
            print(F_red)

        cond_number = np.linalg.cond(K_red)
        if self.verbose:
            print(f"\nCondition number: {cond_number:.2e}")

        if cond_number > 1e10:
            print("WARNING: System is singular!")
            return None, None

        u_reduced = np.linalg.solve(K_red, F_red)

        if self.verbose:
            print(f"\nReduced displacement:")
            print(u_reduced)

        u_full = self.expand_displacement(u_reduced)

        return u_full, u_reduced

    def plot_solution(self, triangle_nodes, u_full, F_external, nodal_reactions,
                      F_block, M_block, block_constraint='all', title='',
                      ax=None, show_info=True):
        """Plot solution with forces"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 9))

        if block_constraint == 'all':
            block_color = 'gray'
            block_label = 'Rigid block (fixed)'
        elif block_constraint == 'horizontal':
            block_color = 'lightblue'
            block_label = 'Rigid block (u,θ fixed)'
        else:
            block_color = 'lightgreen'
            block_label = 'Rigid block (partially free)'

        block_vertices = np.array([[0, 0], [2, 0], [2, 1], [0, 1]])
        ax.fill(*block_vertices.T, color=block_color, alpha=0.3,
                edgecolor='black', linewidth=1.5, label=block_label)

        tri_undeformed = np.vstack([triangle_nodes, triangle_nodes[0]])
        ax.plot(*tri_undeformed.T, 'b--', linewidth=2, marker='o',
                markersize=8, label='Undeformed', markerfacecolor='blue')

        deformed_nodes = triangle_nodes + u_full.reshape(3, 2)
        tri_deformed = np.vstack([deformed_nodes, deformed_nodes[0]])
        ax.plot(*tri_deformed.T, 'r-', linewidth=2, marker='s',
                markersize=8, label='Deformed', markerfacecolor='red')

        # Plot forces
        force_scale = 0.05

        # External forces (green)
        for i in range(3):
            Fx = F_external[2 * i]
            Fy = F_external[2 * i + 1]
            if abs(Fx) > 1e-10 or abs(Fy) > 1e-10:
                node = triangle_nodes[i]
                ax.arrow(node[0], node[1], Fx * force_scale, Fy * force_scale,
                         head_width=0.08, head_length=0.06,
                         fc='green', ec='green', linewidth=2.5, alpha=0.8,
                         label='External' if i == 0 else '')
                ax.text(node[0] + Fx * force_scale * 1.3,
                        node[1] + Fy * force_scale * 1.3,
                        f'F=({Fx:.1f},{Fy:.1f})',
                        fontsize=9, color='green', weight='bold')

        # Reaction forces (purple)
        interface_nodes = [0, 1]
        for i, node_idx in enumerate(interface_nodes):
            Rx = nodal_reactions[i, 0]
            Ry = nodal_reactions[i, 1]
            if abs(Rx) > 1e-10 or abs(Ry) > 1e-10:
                node = triangle_nodes[node_idx]
                ax.arrow(node[0], node[1], Rx * force_scale, Ry * force_scale,
                         head_width=0.08, head_length=0.06,
                         fc='purple', ec='purple', linewidth=2.5, alpha=0.8,
                         linestyle='--',
                         label='Reaction' if i == 0 else '')
                ax.text(node[0] + Rx * force_scale * 1.3,
                        node[1] + Ry * force_scale * 1.3,
                        f'R=({Rx:.1f},{Ry:.1f})',
                        fontsize=9, color='purple', weight='bold')

        # Resultant on block (red)
        ref_point = np.array([1.0, 0.5])
        if abs(F_block[0]) > 1e-10 or abs(F_block[1]) > 1e-10:
            ax.arrow(ref_point[0], ref_point[1],
                     F_block[0] * force_scale, F_block[1] * force_scale,
                     head_width=0.12, head_length=0.08,
                     fc='red', ec='darkred', linewidth=3, alpha=0.9,
                     label='Resultant on block')
            ax.text(ref_point[0] + F_block[0] * force_scale * 1.5,
                    ref_point[1] + F_block[1] * force_scale * 1.5,
                    f'F_block=({F_block[0]:.1f},{F_block[1]:.1f})',
                    fontsize=10, color='darkred', weight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        ax.plot(*ref_point, 'k*', markersize=15, label='Block ref.')

        for i in range(3):
            node = triangle_nodes[i]
            ax.text(node[0], node[1] - 0.15, f'N{i}',
                    fontsize=10, ha='center', weight='bold')

        if show_info:
            info_text = "Displacements:\n"
            for i in range(3):
                u = u_full[2 * i]
                v = u_full[2 * i + 1]
                info_text += f"N{i}: u={u:.4f}, v={v:.4f}\n"

            info_text += f"\nResultant on Block:\n"
            info_text += f"Fx = {F_block[0]:.4f}\n"
            info_text += f"Fy = {F_block[1]:.4f}\n"
            info_text += f"M  = {M_block:.4f}\n"

            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                    family='monospace')

        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(title, fontsize=12, weight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        return ax


# ============================================================================
# GENERALIZED COUPLED SYSTEM
# ============================================================================
class GeneralizedCoupledSystem:
    """
    Generalized system that can handle:
    - Multiple FEM elements connected to each other
    - Multiple rigid blocks
    - Connections between FEM nodes and rigid blocks
    """

    def __init__(self, verbose=True):
        """Initialize empty system"""
        self.verbose = verbose

        # Storage for components
        self.fem_elements = []  # List of TriangleElement objects
        self.rigid_blocks = []  # List of RigidBlock2D objects

        # Global node management
        self.node_positions = {}  # {node_id: [x, y]}
        self.next_node_id = 0

        # Connectivity
        self.element_connectivity = []  # List of [elem_id, [node_id1, node_id2, node_id3]]
        self.rigid_connections = []  # List of [block_id, node_id]

        # DOF management
        self.node_to_dofs = {}  # {node_id: [dof_u, dof_v]}
        self.block_to_dofs = {}  # {block_id: [dof_u, dof_v, dof_theta]}
        self.total_dofs = 0

        # System matrices
        self.K_global = None
        self.F_global = None

        # Constraint information
        self.constrained_nodes = set()  # Nodes connected to rigid blocks

    def add_node(self, position, node_id=None):
        """
        Add a node to the system

        Parameters:
        -----------
        position : array-like
            [x, y] coordinates
        node_id : int, optional
            Specific node ID (if None, auto-assigned)

        Returns:
        --------
        node_id : int
        """
        if node_id is None:
            node_id = self.next_node_id
            self.next_node_id += 1
        else:
            self.next_node_id = max(self.next_node_id, node_id + 1)

        self.node_positions[node_id] = np.array(position, dtype=float)
        return node_id

    def add_fem_element(self, node_ids, E=1000.0, nu=0.3, thickness=1.0, rho=0.0):
        """
        Add a FEM element

        Parameters:
        -----------
        node_ids : list
            List of 3 node IDs that form the triangle
        E : float
            Young's modulus
        nu : float
            Poisson's ratio
        thickness : float
            Element thickness
        rho : float
            Material density (optional)

        Returns:
        --------
        elem_id : int
        """
        # Get node positions
        nodes = [self.node_positions[nid] for nid in node_ids]

        # Create material and geometry
        mat = PlaneStress(E=E, nu=nu, rho=rho)
        geom = Geometry2D(t=thickness)

        # Create element using HybriDFEM Triangle
        elem = Triangle(nodes=nodes, mat=mat, geom=geom)
        elem_id = len(self.fem_elements)
        self.fem_elements.append(elem)

        # Store connectivity
        self.element_connectivity.append([elem_id, node_ids])

        if self.verbose:
            print(f"Added FEM element {elem_id} with nodes {node_ids}")

        return elem_id

    def add_rigid_block(self, reference_point, vertices=None, rho=2400.0, b=1.0, name=None):
        """
        Add a rigid block

        Parameters:
        -----------
        reference_point : array-like
            [x, y] reference point
        vertices : array-like, optional
            Block vertices (if None, creates a unit square around reference point)
        rho : float
            Material density (kg/m^3)
        b : float
            Thickness (out-of-plane dimension)
        name : str, optional
            Block name (not used in Block_2D, kept for compatibility)

        Returns:
        --------
        block_id : int
        """
        # If no vertices provided, create a small square around reference point
        if vertices is None:
            ref = np.array(reference_point)
            size = 0.1
            vertices = np.array([
                ref + [-size, -size],
                ref + [size, -size],
                ref + [size, size],
                ref + [-size, size]
            ])

        # Create Block_2D from HybriDFEM
        block = Block_2D(vertices=vertices, rho=rho, b=b, ref_point=np.array(reference_point))
        block_id = len(self.rigid_blocks)
        self.rigid_blocks.append(block)

        if self.verbose:
            print(f"Added rigid block {block_id} at {reference_point}")

        return block_id

    def connect_node_to_block(self, node_id, block_id):
        """
        Connect a FEM node to a rigid block

        Parameters:
        -----------
        node_id : int
            Node ID
        block_id : int
            Block ID
        """
        self.rigid_connections.append([block_id, node_id])
        self.constrained_nodes.add(node_id)

        if self.verbose:
            print(f"Connected node {node_id} to rigid block {block_id}")

    def build_dof_map(self):
        """
        Build global DOF mapping

        For each free node: 2 DOFs (u, v)
        For each rigid block: 3 DOFs (u, v, θ)
        Constrained nodes don't get their own DOFs (controlled by block DOFs)
        """
        dof_counter = 0

        # Assign DOFs to free nodes
        all_nodes = set(self.node_positions.keys())
        free_nodes = all_nodes - self.constrained_nodes

        for node_id in sorted(free_nodes):
            self.node_to_dofs[node_id] = [dof_counter, dof_counter + 1]
            dof_counter += 2

        # Assign DOFs to rigid blocks
        for block_id in range(len(self.rigid_blocks)):
            self.block_to_dofs[block_id] = [dof_counter, dof_counter + 1, dof_counter + 2]
            dof_counter += 3

        self.total_dofs = dof_counter

        if self.verbose:
            print(f"\nDOF mapping built:")
            print(f"  Free nodes: {len(free_nodes)} ({len(free_nodes) * 2} DOFs)")
            print(f"  Constrained nodes: {len(self.constrained_nodes)}")
            print(f"  Rigid blocks: {len(self.rigid_blocks)} ({len(self.rigid_blocks) * 3} DOFs)")
            print(f"  Total DOFs: {self.total_dofs}")

    def assemble_global_stiffness(self):
        """
        Assemble global stiffness matrix

        Strategy:
        1. Assemble FEM element contributions
        2. Apply rigid body constraints
        """
        self.K_global = lil_matrix((self.total_dofs, self.total_dofs))

        # Build node-to-block mapping for quick lookup
        node_to_block = {}
        for block_id, node_id in self.rigid_connections:
            node_to_block[node_id] = block_id

        # Process each FEM element
        for elem_id, node_ids in self.element_connectivity:
            elem = self.fem_elements[elem_id]
            K_elem = elem.get_k_glob()  # 6x6 element stiffness

            # Map element DOFs to global DOFs
            global_dofs = []
            constraint_matrices = []  # Store constraint matrices for constrained nodes

            for local_node_idx, node_id in enumerate(node_ids):
                if node_id in self.constrained_nodes:
                    # Node is connected to a rigid block
                    block_id = node_to_block[node_id]
                    block_dofs = self.block_to_dofs[block_id]
                    node_pos = self.node_positions[node_id]

                    # Get constraint matrix C: u_node = C * q_block
                    C = self.rigid_blocks[block_id].constraint_matrix_for_node(node_pos)

                    global_dofs.append((block_dofs, C))
                    constraint_matrices.append(C)
                else:
                    # Free node
                    node_dofs = self.node_to_dofs[node_id]
                    global_dofs.append((node_dofs, None))
                    constraint_matrices.append(None)

            # Assemble element stiffness into global matrix
            for i in range(3):  # Loop over element nodes
                for j in range(3):
                    # Extract 2x2 block from element stiffness
                    K_ij = K_elem[2 * i:2 * i + 2, 2 * j:2 * j + 2]

                    dofs_i, C_i = global_dofs[i]
                    dofs_j, C_j = global_dofs[j]

                    if C_i is None and C_j is None:
                        # Both nodes are free - direct assembly
                        self.K_global[np.ix_(dofs_i, dofs_j)] += K_ij

                    elif C_i is None and C_j is not None:
                        # i is free, j is constrained
                        # K_ij_transformed = K_ij @ C_j
                        K_transformed = K_ij @ C_j
                        self.K_global[np.ix_(dofs_i, dofs_j)] += K_transformed

                    elif C_i is not None and C_j is None:
                        # i is constrained, j is free
                        # K_ij_transformed = C_i.T @ K_ij
                        K_transformed = C_i.T @ K_ij
                        self.K_global[np.ix_(dofs_i, dofs_j)] += K_transformed

                    else:
                        # Both nodes are constrained
                        # K_ij_transformed = C_i.T @ K_ij @ C_j
                        K_transformed = C_i.T @ K_ij @ C_j
                        self.K_global[np.ix_(dofs_i, dofs_j)] += K_transformed

        # Convert to CSR format for efficient solving
        self.K_global = self.K_global.tocsr()

        if self.verbose:
            print(f"\nGlobal stiffness matrix assembled: {self.K_global.shape}")
            print(f"  Non-zero entries: {self.K_global.nnz}")

    def apply_boundary_conditions(self):
        """
        Apply boundary conditions (fixed DOFs of rigid blocks)

        Returns:
        --------
        free_dofs : list
            List of free DOF indices
        """
        fixed_dofs = set()

        # Collect fixed DOFs from rigid blocks
        for block_id, block in enumerate(self.rigid_blocks):
            block_dofs = self.block_to_dofs[block_id]
            for local_dof in block.fixed_dofs:
                global_dof = block_dofs[local_dof]
                fixed_dofs.add(global_dof)

        # Get free DOFs
        all_dofs = set(range(self.total_dofs))
        free_dofs = sorted(list(all_dofs - fixed_dofs))

        if self.verbose:
            print(f"\nBoundary conditions:")
            print(f"  Fixed DOFs: {len(fixed_dofs)}")
            print(f"  Free DOFs: {len(free_dofs)}")

        return free_dofs

    def solve(self, external_forces):
        """
        Solve the coupled system

        Parameters:
        -----------
        external_forces : dict
            {node_id: [Fx, Fy]} for forces on free nodes

        Returns:
        --------
        solution : dict
            {
                'node_displacements': {node_id: [u, v]},
                'block_displacements': {block_id: [u, v, theta]},
                'reactions': reaction information
            }
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("SOLVING COUPLED SYSTEM")
            print("=" * 70)

        # Build DOF map
        self.build_dof_map()

        # Assemble global stiffness
        self.assemble_global_stiffness()

        # Build force vector
        F_global = np.zeros(self.total_dofs)
        for node_id, force in external_forces.items():
            if node_id in self.node_to_dofs:  # Only apply to free nodes
                dofs = self.node_to_dofs[node_id]
                F_global[dofs[0]] = force[0]
                F_global[dofs[1]] = force[1]

        # Apply boundary conditions
        free_dofs = self.apply_boundary_conditions()

        # Reduce system
        K_reduced = self.K_global[np.ix_(free_dofs, free_dofs)].toarray()
        F_reduced = F_global[free_dofs]

        if self.verbose:
            print(f"\nReduced system size: {len(free_dofs)} x {len(free_dofs)}")
            print(f"Condition number: {np.linalg.cond(K_reduced):.2e}")

        # Solve
        u_reduced = np.linalg.solve(K_reduced, F_reduced)

        # Expand to full solution
        u_global = np.zeros(self.total_dofs)
        u_global[free_dofs] = u_reduced

        # Extract solution
        solution = self._extract_solution(u_global)

        # Compute reactions
        solution['reactions'] = self._compute_reactions(u_global, F_global)

        return solution

    def _extract_solution(self, u_global):
        """Extract solution from global displacement vector"""
        solution = {
            'node_displacements': {},
            'block_displacements': {},
            'u_global': u_global
        }

        # Free node displacements
        for node_id, dofs in self.node_to_dofs.items():
            solution['node_displacements'][node_id] = u_global[dofs]

        # Constrained node displacements (compute from rigid body motion)
        node_to_block = {node_id: block_id for block_id, node_id in self.rigid_connections}
        for node_id in self.constrained_nodes:
            block_id = node_to_block[node_id]
            block_dofs = self.block_to_dofs[block_id]
            q_block = u_global[block_dofs]

            # Compute node displacement from rigid body motion
            C = self.rigid_blocks[block_id].constraint_matrix_for_node(
                self.node_positions[node_id]
            )
            u_node = C @ q_block
            solution['node_displacements'][node_id] = u_node

        # Block displacements
        for block_id, dofs in self.block_to_dofs.items():
            solution['block_displacements'][block_id] = u_global[dofs]

        if self.verbose:
            print("\n" + "=" * 70)
            print("SOLUTION")
            print("=" * 70)
            for node_id in sorted(solution['node_displacements'].keys()):
                u = solution['node_displacements'][node_id]
                constrained = " (constrained)" if node_id in self.constrained_nodes else ""
                print(f"Node {node_id}{constrained}: u={u[0]:8.4f}, v={u[1]:8.4f}")

            for block_id in sorted(solution['block_displacements'].keys()):
                q = solution['block_displacements'][block_id]
                print(f"Block {block_id}: u={q[0]:8.4f}, v={q[1]:8.4f}, θ={q[2]:8.4f}")

        return solution

    def _compute_reactions(self, u_global, F_global):
        """Compute reaction forces at constrained DOFs"""
        # Internal forces
        F_internal = self.K_global @ u_global

        # Reactions = Internal - External
        reactions = F_internal - F_global

        # Extract reactions at rigid blocks
        block_reactions = {}
        for block_id, dofs in self.block_to_dofs.items():
            block_reactions[block_id] = reactions[dofs]

        if self.verbose:
            print("\n" + "=" * 70)
            print("REACTIONS")
            print("=" * 70)
            for block_id, R in block_reactions.items():
                if self.rigid_blocks[block_id].is_fully_fixed():
                    print(f"Block {block_id} (fixed): Fx={R[0]:8.4f}, Fy={R[1]:8.4f}, M={R[2]:8.4f}")

        return {
            'block_reactions': block_reactions,
            'F_internal': F_internal,
            'reactions_global': reactions
        }

    def compute_element_stresses(self, solution):
        """
        Compute stresses in all FEM elements

        Parameters:
        -----------
        solution : dict
            Solution dictionary from solve()

        Returns:
        --------
        stresses : dict
            {elem_id: {'stress': [sigma_xx, sigma_yy, tau_xy], 'strain': [epsilon_xx, epsilon_yy, gamma_xy]}}
        """
        stresses = {}

        for elem_id, node_ids in self.element_connectivity:
            elem = self.fem_elements[elem_id]

            # Get element displacement vector
            u_elem = np.zeros(6)
            for i, node_id in enumerate(node_ids):
                u_node = solution['node_displacements'][node_id]
                u_elem[2 * i:2 * i + 2] = u_node

            # Compute stress
            stress, strain = elem.compute_stress(u_elem)
            stresses[elem_id] = {'stress': stress, 'strain': strain}

            if self.verbose:
                print(f"\nElement {elem_id}:")
                print(f"  Stress: sigma_xx={stress[0]:.4f}, sigma_yy={stress[1]:.4f}, tau_xy={stress[2]:.4f}")
                print(f"  Strain: epsilon_xx={strain[0]:.6f}, epsilon_yy={strain[1]:.6f}, gamma_xy={strain[2]:.6f}")

        return stresses


def plot_mesh_and_solution(system, solution, external_forces=None, scale=10.0,
                           show_undeformed=True, show_stress=True,
                           show_forces=True, force_scale=0.01, figsize=(14, 10)):
    """
    Visualize the mesh and solution

    Parameters:
    -----------
    system : GeneralizedCoupledSystem
    solution : dict from system.solve()
    external_forces : dict, optional
        {node_id: [Fx, Fy]} - external forces to display
    scale : float
        Displacement magnification factor
    show_undeformed : bool
        Show undeformed mesh
    show_stress : bool
        Color elements by stress
    show_forces : bool
        Show force arrows
    force_scale : float
        Force arrow scaling factor
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Compute stresses if needed
    if show_stress:
        stresses = system.compute_element_stresses(solution)
        stress_values = [np.linalg.norm(stresses[i]['stress']) for i in range(len(system.fem_elements))]
        vmin, vmax = min(stress_values), max(stress_values)

    # Plot FEM elements
    for elem_id, node_ids in system.element_connectivity:
        # Undeformed
        if show_undeformed:
            nodes = [system.node_positions[nid] for nid in node_ids]
            tri = np.vstack([nodes, nodes[0]])
            ax.plot(*tri.T, 'b--', linewidth=1, alpha=0.5)

        # Deformed
        nodes_def = []
        for nid in node_ids:
            pos = system.node_positions[nid]
            u = solution['node_displacements'][nid]
            nodes_def.append(pos + scale * u)

        tri_def = np.array(nodes_def)

        if show_stress:
            color = plt.cm.jet((stress_values[elem_id] - vmin) / (vmax - vmin + 1e-10))
            ax.fill(*tri_def.T, color=color, alpha=0.6, edgecolor='black', linewidth=1.5)
        else:
            ax.fill(*tri_def.T, color='lightblue', alpha=0.6, edgecolor='black', linewidth=1.5)

    # Plot nodes
    for node_id in system.node_positions.keys():
        pos = system.node_positions[node_id]
        u = solution['node_displacements'][node_id]

        # Undeformed
        if show_undeformed:
            ax.plot(*pos, 'bo', markersize=6, alpha=0.5)

        # Deformed
        pos_def = pos + scale * u
        marker = 's' if node_id in system.constrained_nodes else 'o'
        color = 'red' if node_id in system.constrained_nodes else 'blue'
        ax.plot(*pos_def, marker=marker, color=color, markersize=8)
        ax.text(pos_def[0], pos_def[1] + 0.1, f'N{node_id}', ha='center', fontsize=8)

    # Plot rigid blocks
    for block_id, block in enumerate(system.rigid_blocks):
        ref = block.ref_point
        q = solution['block_displacements'][block_id]
        ref_def = ref + scale * q[:2]

        ax.plot(*ref, 'k*', markersize=15, alpha=0.5)
        ax.plot(*ref_def, 'r*', markersize=15)
        ax.text(ref_def[0], ref_def[1] - 0.15, f'B{block_id}', ha='center', fontsize=9,
                weight='bold', color='red')

    # Plot forces
    if show_forces:
        # External forces (green arrows)
        if external_forces is not None:
            for node_id, force in external_forces.items():
                if np.linalg.norm(force) > 1e-10:
                    pos = system.node_positions[node_id]
                    u = solution['node_displacements'][node_id]
                    pos_def = pos + scale * u

                    Fx, Fy = force
                    ax.arrow(pos_def[0], pos_def[1],
                             Fx * force_scale, Fy * force_scale,
                             head_width=0.1, head_length=0.08,
                             fc='green', ec='darkgreen', linewidth=2.5,
                             alpha=0.8, label='External force' if node_id == list(external_forces.keys())[0] else '')
                    ax.text(pos_def[0] + Fx * force_scale * 1.2,
                            pos_def[1] + Fy * force_scale * 1.2,
                            f'F=({Fx:.1f},{Fy:.1f})',
                            fontsize=8, color='darkgreen', weight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

        # Reaction forces at constrained nodes (purple arrows)
        reactions = solution['reactions']
        F_internal = reactions['F_internal']

        for node_id in system.constrained_nodes:
            node_dofs_global = []
            # Get global DOFs for this constrained node
            # Find which block controls this node
            for block_id, nid in system.rigid_connections:
                if nid == node_id:
                    # Get the internal force through block DOFs
                    block_dofs = system.block_to_dofs[block_id]
                    # Transform block forces to node forces
                    C = system.rigid_blocks[block_id].constraint_matrix_for_node(
                        system.node_positions[node_id]
                    )
                    q_block = solution['u_global'][block_dofs]

                    # Get element forces at this node
                    node_reaction = np.zeros(2)
                    for elem_id, node_ids in system.element_connectivity:
                        if node_id in node_ids:
                            local_idx = node_ids.index(node_id)
                            elem = system.fem_elements[elem_id]

                            # Get element displacement
                            u_elem = np.zeros(6)
                            for i, nid in enumerate(node_ids):
                                u_node = solution['node_displacements'][nid]
                                u_elem[2 * i:2 * i + 2] = u_node

                            # Compute internal force
                            F_elem_internal = elem.Ke @ u_elem
                            node_reaction += F_elem_internal[2 * local_idx:2 * local_idx + 2]

                    # Subtract external force if any
                    if external_forces and node_id in external_forces:
                        node_reaction -= external_forces[node_id]

                    # Plot reaction force
                    if np.linalg.norm(node_reaction) > 1e-10:
                        pos = system.node_positions[node_id]
                        u = solution['node_displacements'][node_id]
                        pos_def = pos + scale * u

                        Rx, Ry = node_reaction
                        ax.arrow(pos_def[0], pos_def[1],
                                 Rx * force_scale, Ry * force_scale,
                                 head_width=0.1, head_length=0.08,
                                 fc='purple', ec='purple', linewidth=2,
                                 alpha=0.7, linestyle='--',
                                 label='Reaction' if node_id == list(system.constrained_nodes)[0] else '')
                        ax.text(pos_def[0] + Rx * force_scale * 1.2,
                                pos_def[1] + Ry * force_scale * 1.2,
                                f'R=({Rx:.1f},{Ry:.1f})',
                                fontsize=7, color='purple', weight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.7))
                    break

        # Block reactions and moments (red)
        block_reactions = reactions['block_reactions']
        for block_id, R_block in block_reactions.items():
            if system.rigid_blocks[block_id].is_fully_fixed() or np.linalg.norm(R_block[:2]) > 1e-10:
                ref = system.rigid_blocks[block_id].ref_point
                q = solution['block_displacements'][block_id]
                ref_def = ref + scale * q[:2]

                Fx, Fy, M = R_block

                # Resultant force
                if np.linalg.norm([Fx, Fy]) > 1e-10:
                    ax.arrow(ref_def[0], ref_def[1],
                             Fx * force_scale, Fy * force_scale,
                             head_width=0.12, head_length=0.1,
                             fc='darkred', ec='darkred', linewidth=3,
                             alpha=0.9, label='Block resultant' if block_id == 0 else '')

                # Moment (arc)
                if abs(M) > 1e-10:
                    moment_radius = 0.1
                    theta_start = -90 if M > 0 else 60
                    theta_end = 120 if M > 0 else 270
                    arc = Arc(xy=ref_def, width=2 * moment_radius, height=2 * moment_radius,
                              angle=0, theta1=theta_start, theta2=theta_end,
                              linewidth=2.5, linestyle='-', edgecolor='darkred', alpha=0.8)
                    ax.add_patch(arc)

                    # Arrow head for moment direction
                    angle = np.deg2rad(theta_end) if M > 0 else np.deg2rad(theta_start)
                    arrow_start = ref_def + moment_radius * np.array([np.cos(angle), np.sin(angle)])
                    arrow_dir = moment_radius * 0.3 * np.array([-np.sin(angle), np.cos(angle)]) * np.sign(M)
                    ax.arrow(arrow_start[0], arrow_start[1], arrow_dir[0], arrow_dir[1],
                             head_width=0.08, head_length=0.06, fc='darkred', ec='darkred', linewidth=2)

                    # keep the circle from looking like an ellipse
                    ax.set_aspect('equal', adjustable='datalim')

                # Text annotation
                info_text = f'Block {block_id}\n'
                if np.linalg.norm([Fx, Fy]) > 1e-10:
                    info_text += f'F=({Fx:.1f},{Fy:.1f})\n'
                if abs(M) > 1e-10:
                    info_text += f'M={M:.2f}'

                ax.text(ref_def[0] - 0.4, ref_def[1] + 0.3, info_text,
                        fontsize=9, color='black', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.5))

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    title = f'FEM-Rigid Coupled System\n(disp. scale={scale}x'
    if show_forces:
        title += f', force scale={force_scale}x'
    title += ')'
    ax.set_title(title, fontsize=12, weight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label='FEM elements'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Free nodes'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=8, label='Constrained nodes'),
        plt.Line2D([0], [0], marker='*', color='red', markersize=12, linestyle='None', label='Rigid block ref.')
    ]
    ax.legend(handles=legend_elements, loc='best')

    return fig, ax


def example_simple_case():
    """Reproduce the original simple case: single triangle on rigid block"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: SINGLE TRIANGLE ON RIGID BLOCK (Original Problem)")
    print("=" * 70)

    system = GeneralizedCoupledSystem(verbose=True)

    # Add nodes
    n0 = system.add_node([0.0, 0.0])  # Interface
    n1 = system.add_node([1.0, 0.0])  # Interface
    n2 = system.add_node([0, 1.0])  # Free

    # Add FEM element
    system.add_fem_element([n0, n1, n2], E=1000.0, nu=0.3)

    # Add rigid block
    b0 = system.add_rigid_block([0.5, -0.5])
    system.rigid_blocks[b0].set_fixed('all')

    # Connect nodes to block
    system.connect_node_to_block(n0, b0)
    system.connect_node_to_block(n1, b0)

    # Apply forces
    forces = {
        n2: [-100.0, 0]  # Downward force on free node
    }

    # Solve
    solution = system.solve(forces)

    # Visualize
    fig, ax = plot_mesh_and_solution(system, solution, external_forces=forces,
                                     scale=.5, force_scale=0.0025)
    plt.show()

    return system, solution


if __name__ == "__main__":
    print("\nRunning Example 1: Single Triangle (Original)")
    system1, sol1 = example_simple_case()
