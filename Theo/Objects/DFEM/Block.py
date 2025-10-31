# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:11:26 2024

@author: ibouckaert
"""

import os
import warnings
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

from Theo.Objects.Material.Material import Material


def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    file_short_name = filename.replace(os.path.dirname(filename), "")
    return f"Warning! In file {file_short_name}, line {lineno}: {message}\n"


warnings.formatwarning = custom_warning_format


class Block_2D:
    DOFS_PER_NODE = 3  # Rigid blocks: [ux, uy, rotation_z]

    def __init__(self, vertices, b=1, material: Material = None, ref_point=None):
        # Initializing attributes of block
        self.connect = None
        self.dofs = np.zeros(3)
        self.disps = np.zeros(3)
        self.center = np.zeros(2)
        self.A = 0
        self.I = 0

        self.v = vertices.copy()
        self.nb_vertices = len(vertices)
        self.rho = material.rho

        self.b = b
        self.cfs = []
        # Check if we use material, contact or surface law
        # if not material: warn("Warning: Block was defined without material model")

        self.material = material

        # Coupling support: track which DOFs are fixed (for FEM-Block coupling)
        self.fixed_dofs = []  # List of fixed DOF indices (0=u, 1=v, 2=θ)

        # Computing center of gravity, area, mass and rotational inertia w.r.t ref point
        self.get_area()
        self.get_center()

        if ref_point is None:
            self.ref_point = self.center.copy()
        else:
            self.ref_point = ref_point.copy()

        self.get_rot_inertia()

        if not self.is_valid_polygon():
            warn("Careful, the block is not a valid polygon", UserWarning)
        self.get_min_circle()

    def make_connect(self, index, structure=None):
        """
        Set connection to global node index and compute DOF mapping.

        This method now supports variable DOFs per node:
        - If structure provided: uses structure.node_dof_offsets (flexible DOF system)
        - If structure=None: falls back to 3*index (backward compatibility)

        Args:
            index: Global node index in Structure_2D
            structure: Structure_2D instance (optional, for flexible DOF support)
        """
        self.connect = index

        # Compute base DOF index for this node
        if structure is not None and hasattr(structure, 'node_dof_offsets') and len(structure.node_dof_offsets) > index:
            # Variable DOF mode: use node_dof_offsets
            base_dof = structure.node_dof_offsets[index]
        else:
            # Fallback: assume 3 DOFs per node
            base_dof = 3 * index

        # Map block DOFs (3: ux, uy, rz) to global structure DOFs
        self.dofs = np.arange(3, dtype=int) + base_dof

    def get_area(self):
        for i in range(self.nb_vertices - 1):
            self.A += self.v[i, 0] * self.v[i + 1, 1] - self.v[i + 1, 0] * self.v[i, 1]

        self.A += self.v[-1, 0] * self.v[0, 1] - self.v[0, 0] * self.v[-1, 1]
        self.A /= 2

        self.m = self.rho * self.A * self.b

    def get_center(self):
        for i in range(self.nb_vertices - 1):
            self.center[0] += (self.v[i, 0] + self.v[i + 1, 0]) * (
                    self.v[i, 0] * self.v[i + 1, 1] - self.v[i + 1, 0] * self.v[i, 1]
            )
            self.center[1] += (self.v[i, 1] + self.v[i + 1, 1]) * (
                    self.v[i, 0] * self.v[i + 1, 1] - self.v[i + 1, 0] * self.v[i, 1]
            )

        self.center[0] += (self.v[-1, 0] + self.v[0, 0]) * (
                self.v[-1, 0] * self.v[0, 1] - self.v[0, 0] * self.v[-1, 1]
        )
        self.center[1] += (self.v[-1, 1] + self.v[0, 1]) * (
                self.v[-1, 0] * self.v[0, 1] - self.v[0, 0] * self.v[-1, 1]
        )

        self.center /= 6 * self.A

    def get_rot_inertia(self):
        v = self.v - np.tile(self.center, (self.nb_vertices, 1))
        # Rotational inertia around centroid
        for i in range(self.nb_vertices - 1):
            self.I += (
                    self.m
                    * (v[i, 0] * v[i + 1, 1] - v[i + 1, 0] * v[i, 1])
                    * (
                            v[i, 0] ** 2
                            + v[i, 0] * v[i + 1, 0]
                            + v[i + 1, 0] ** 2
                            + v[i, 1] ** 2
                            + v[i, 1] * v[i + 1, 1]
                            + v[i + 1, 1] ** 2
                    )
            )

        self.I += (
                self.m
                * (v[-1, 0] * v[0, 1] - v[0, 0] * v[-1, 1])
                * (
                        v[-1, 0] ** 2
                        + v[-1, 0] * v[0, 0]
                        + v[0, 0] ** 2
                        + v[-1, 1] ** 2
                        + v[-1, 1] * v[0, 1]
                        + v[0, 1] ** 2
                )
        )

        self.I /= 12 * self.A

        # print(self.I)
        # ♦ Rotational inertia around reference point
        d = self.center - self.ref_point

        self.I += self.m * (d[0] ** 2 + d[1] ** 2)

    def is_valid_polygon(self):
        # Check if the block that is created corresponds to a real shape.
        def on_segment(a, b, c):
            # Given colinear points a,b,c, check if c lies on segment ab
            return (
                    c[0] <= max(a[0], b[0])
                    and c[0] >= min(a[0], b[0])
                    and c[1] <= max(a[1], b[1])
                    and c[1] >= min(a[1], b[1])
            )

        def orientation(a, b, c):
            # Check if points a b c are colinear :
            value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])

            if abs(value) <= 1e-8:  # Collinear
                return 0
            elif value > 0:  # Clockwise
                return 1
            else:  # Counterclockwise
                return 2

        def intersect(a1, b1, a2, b2):
            # Check intersection between segment a1b1 and segment a2b2:
            o1 = orientation(a1, b1, a2)
            o2 = orientation(a1, b1, b2)
            o3 = orientation(a2, b2, a1)
            o4 = orientation(a2, b2, b1)

            if o1 != o2 and o3 != o4:
                return True  # General case
            if o1 == 0 and on_segment(a1, b1, a2):
                return True
            if o2 == 0 and on_segment(a1, b1, b2):
                return True
            if o3 == 0 and on_segment(a2, b2, a1):
                return True
            if o4 == 0 and on_segment(a2, b2, b1):
                return True

            return False

        if self.nb_vertices < 4:
            return True

        for i in range(self.nb_vertices):
            for j in range(i + 2, self.nb_vertices):
                if i == 0 and j == self.nb_vertices - 1:
                    continue
                if intersect(
                        self.v[i],
                        self.v[(i + 1) % self.nb_vertices],
                        self.v[j],
                        self.v[(j + 1) % self.nb_vertices],
                ):
                    return False
        return True

    def get_min_circle(self):
        # Distance between two points:
        def distance(a, b):
            return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        def is_inside(circle, p):
            return distance(circle["c"], p) <= circle["r"]

        def are_inside(circle, P):
            for p in P:
                if not is_inside(circle, p):
                    return False

            return True

        def make_circle(c, r):
            circle = {}
            circle["c"] = c
            circle["r"] = r
            return circle

        def circle_2points(a, b):
            return make_circle((a + b) / 2, distance(a, b) / 2)

        def intermediate_circle_center(a, b):
            B = a[0] ** 2 + a[1] ** 2
            C = b[0] ** 2 + b[1] ** 2
            D = a[0] * b[1] - a[1] * b[0]

            return np.array([b[1] * B - a[1] * C, a[0] * C - b[0] * B]) / (2 * D)

        def circle_3points(a, b, c):
            i = intermediate_circle_center(
                np.array([b[0] - a[0], b[1] - a[1]]),
                np.array([c[0] - a[0], c[1] - a[1]]),
            )
            return make_circle(i + a, distance(i + a, a))

        def min_circle_3points(P):
            assert len(P) <= 3

            if len(P) == 0:
                return make_circle(np.zeros(2), 0)
            if len(P) == 1:
                return make_circle(P[0], 0)
            if len(P) == 2:
                return circle_2points(P[0], P[1])

            for i in range(3):
                for j in range(i + 1, 3):
                    if are_inside(circle_2points(P[i], P[j]), P):
                        return circle_2points(P[i], P[j])

            return circle_3points(P[0], P[1], P[2])

        def welzl_helper(P, R, n):
            if n == 0 or len(R) == 3:
                return min_circle_3points(R)

            trial_circle = welzl_helper(P[1:], R.copy(), n - 1)

            if is_inside(trial_circle, P[0]):
                return trial_circle

            R.append(P[0])
            # print(R)
            return welzl_helper(P[1:], R.copy(), n - 1)

        def welzl(P):
            return welzl_helper(P, [], len(P))

        # circle = welzl(self.v)
        circle = welzl(self.v)
        self.circle_center = circle["c"]
        self.circle_radius = circle["r"]

    def get_mass(self, no_inertia=False):
        if no_inertia:
            self.mass = np.array([[self.m, 0, 0], [0, self.m, 0], [0, 0, 0]])
        else:
            self.mass = np.diag(np.array([self.m, self.m, self.I]))

        return self.mass

    def compute_triplets(self, eps: float = 1e-12):
        """
        Build per-edge descriptors for the closed polygon defined by self.v.

        Returns
        -------
        list[dict]
            Each dict contains:
              - "ABC": np.array([nx, ny, C]) with ||(nx,ny)|| = 1 and canonical sign
              - "Vertices": np.array([[x1,y1],[x2,y2]]) endpoints in original order
              - "tangent": unit tangent vector (x2-x1)/||x2-x1||
              - "normal": unit left-hand normal (-t_y, t_x)
              - "length": segment length
              - "bbox": np.array([xmin, ymin, xmax, ymax])
        Notes
        -----
        - The "ABC" representation is normalized and sign-fixed so identical lines
          (colinear segments) compare robustly with np.allclose.
        """
        V = np.asarray(self.v, dtype=float)
        n = getattr(self, "nb_vertices", len(V))
        if n < 2:
            return []

        # consecutive edges including closing edge
        P = V
        Q = np.roll(V, -1, axis=0)  # next vertex, wraps last->first

        # vectors, lengths
        D = Q - P
        L = np.linalg.norm(D, axis=1)

        # mask out degenerate edges
        mask = L > eps
        if not np.any(mask):
            return []

        P = P[mask]
        Q = Q[mask]
        D = D[mask]
        L = L[mask]

        # unit tangent and left-hand unit normal
        T = D / L[:, None]
        N = np.stack([-T[:, 1], T[:, 0]], axis=1)  # rotate +90°

        # normal-form C = -n · a  (use first endpoint P)
        C = -np.einsum("ij,ij->i", N, P)

        # enforce canonical sign for (n, C): n_x > 0 or (n_x==0 and n_y > 0)
        flip = (N[:, 0] < 0.0) | ((np.abs(N[:, 0]) <= eps) & (N[:, 1] < 0.0))
        N[flip] *= -1.0
        C[flip] *= -1.0

        # pack results
        triplets = []
        for i in range(P.shape[0]):
            a = P[i]
            b = Q[i]
            nvec = N[i]
            cval = C[i]
            length = L[i]
            bbox = np.array([min(a[0], b[0]), min(a[1], b[1]),
                             max(a[0], b[0]), max(a[1], b[1])], dtype=float)

            triplets.append({
                "ABC": np.array([nvec[0], nvec[1], cval], dtype=float),
                "Vertices": np.vstack([a, b]),
                "tangent": T[i].copy(),
                "normal": nvec.copy(),
                "length": float(length),
                "bbox": bbox,
            })
        return triplets

    def plot_block(self, scale=0, lighter=False):
        if scale > 1:  # and abs(self.disps[2]) < np.pi/2:
            angle_scaled = np.arctan(
                scale * np.tan(self.disps[2])
            )  # Scaling the rotation angle
        elif scale == 0:
            angle_scaled = 0
        else:
            angle_scaled = scale * self.disps[2]

        T = np.array(
            [
                [np.cos(angle_scaled), -np.sin(angle_scaled)],
                [np.sin(angle_scaled), np.cos(angle_scaled)],
            ]
        )  # Rotation matrix

        # Initializing new positions of vertices
        x_vertices = []
        y_vertices = []

        for v in self.v:
            ref_point_to_vertex = v - self.ref_point
            rotation = T @ ref_point_to_vertex - ref_point_to_vertex

            x_vertices.append(
                self.ref_point[0]
                + ref_point_to_vertex[0]
                + scale * self.disps[0]
                + rotation[0]
            )
            y_vertices.append(
                self.ref_point[1]
                + ref_point_to_vertex[1]
                + scale * self.disps[1]
                + rotation[1]
            )

        ref_point_to_center = self.center - self.ref_point
        rotation = T @ ref_point_to_center - ref_point_to_center

        x_center = (
                self.ref_point[0]
                + ref_point_to_center[0]
                + scale * self.disps[0]
                + rotation[0]
        )
        y_center = (
                self.ref_point[1]
                + ref_point_to_center[1]
                + scale * self.disps[1]
                + rotation[1]
        )

        x_ref = self.ref_point[0] + scale * self.disps[0]
        y_ref = self.ref_point[1] + scale * self.disps[1]

        # Closing the shape
        x_vertices.append(x_vertices[0])
        y_vertices.append(y_vertices[0])

        if lighter:
            color = "gray"
            linewidth = 0.15
        else:
            color = "black"
            linewidth = 0.3
        # plt.plot(x_vertices, y_vertices, marker='o', markersize=0., linestyle='-', color=color, linewidth=linewidth) #Plotting edges of polygon
        try:
            if self.material.tag == "STC":
                plt.fill(
                    x_vertices, y_vertices, color="#fbb040", linewidth=0
                )  # Filling polygon
                # plt.plot(x_vertices, y_vertices, marker='o', markersize=0., linestyle='-', color='black', linewidth=.1)
            elif self.material.tag == "CTC":
                plt.fill(x_vertices, y_vertices, color="silver", linewidth=0)
                # plt.plot(x_vertices, y_vertices, marker='o', markersize=0., linestyle='-', color='black', linewidth=.1)
            else:
                plt.plot(
                    x_vertices,
                    y_vertices,
                    marker="o",
                    markersize=0.0,
                    linestyle="-",
                    color=color,
                    linewidth=linewidth,
                )
        except Exception:
            plt.plot(
                x_vertices,
                y_vertices,
                marker="o",
                markersize=0.0,
                linestyle="-",
                color=color,
                linewidth=linewidth,
            )

        # plt.plot(x_center, y_center, marker='o', color='red', markersize=3) #Plotting center of gravity of polygon
        # plt.plot(x_ref, y_ref, marker='x', color='black', markersize=2) #Plotting reference point of polygon

        if scale == 0:
            theta = np.linspace(0, 2 * np.pi, 200)
            x_circle = np.ones(200) * self.circle_center[
                0
            ] + self.circle_radius * np.cos(theta)
            y_circle = np.ones(200) * self.circle_center[
                1
            ] + self.circle_radius * np.sin(theta)

            # plt.plot(x_circle, y_circle, color='blue', linestyle='dashed', linewidth=.2)

        plt.axis("equal")
        plt.axis("off")

    # ========================================================================
    # COUPLING METHODS (for FEM-Block Hybrid Coupling)
    # ========================================================================

    def set_fixed(self, dofs='all'):
        """
        Fix specific DOFs of the rigid block for coupling purposes.

        This method is used when coupling blocks with FEM elements to specify
        which degrees of freedom should be constrained.

        Parameters
        ----------
        dofs : str or list
            Specification of which DOFs to fix:
            - 'all': Fix all DOFs (u, v, θ)
            - 'horizontal': Fix u and θ only
            - 'vertical': Fix v only
            - 'rotation': Fix θ only
            - list: List of DOF indices [0=u, 1=v, 2=θ] to fix

        Examples
        --------
        >>> block.set_fixed('all')  # Fix completely
        >>> block.set_fixed([0, 1])  # Fix translations only
        >>> block.set_fixed('vertical')  # Allow horizontal sliding and rotation
        """
        if dofs == 'all':
            self.fixed_dofs = [0, 1, 2]
        elif dofs == 'horizontal':
            self.fixed_dofs = [0, 2]  # Fix u and θ
        elif dofs == 'vertical':
            self.fixed_dofs = [1]  # Fix v only
        elif dofs == 'rotation':
            self.fixed_dofs = [2]  # Fix θ only
        elif isinstance(dofs, list):
            self.fixed_dofs = dofs
        else:
            self.fixed_dofs = []

    def get_free_dofs(self):
        """
        Get list of free (unconstrained) DOF indices.

        Returns
        -------
        list
            Sorted list of free DOF indices from {0, 1, 2}

        Examples
        --------
        >>> block.set_fixed([0, 2])
        >>> block.get_free_dofs()
        [1]
        """
        all_dofs = {0, 1, 2}
        return sorted(list(all_dofs - set(self.fixed_dofs)))

    def is_fully_fixed(self):
        """
        Check if all DOFs are fixed.

        Returns
        -------
        bool
            True if all three DOFs are constrained

        Examples
        --------
        >>> block.set_fixed('all')
        >>> block.is_fully_fixed()
        True
        """
        return len(self.fixed_dofs) == 3

    def displacement_at_point(self, point, displacements=None):
        """
        Compute displacement at any point using rigid body kinematics.

        Uses small angle approximation for rotation:
        u = u_block - (y - y_ref) * θ
        v = v_block + (x - x_ref) * θ

        Parameters
        ----------
        point : array-like
            [x, y] coordinates of the point in global frame
        displacements : array-like, optional
            [u, v, θ] block displacements. If None, uses self.disps

        Returns
        -------
        np.ndarray
            [u, v] displacement at the point

        Examples
        --------
        >>> block.disps = np.array([0.1, 0.2, 0.01])  # u, v, θ
        >>> u_point = block.displacement_at_point([1.0, 2.0])
        """
        if displacements is None:
            u_b, v_b, theta = self.disps
        else:
            u_b, v_b, theta = displacements

        x, y = point
        x_c, y_c = self.ref_point

        # Small angle approximation (linear kinematics)
        u = u_b - (y - y_c) * theta
        v = v_b + (x - x_c) * theta

        return np.array([u, v])

    def constraint_matrix_for_node(self, node_position):
        """
        Get constraint matrix for coupling a node to this rigid block.

        Returns the 2×3 matrix C such that:
            [u_node, v_node]^T = C * [u_block, v_block, θ_block]^T

        This enforces kinematic compatibility between the rigid block motion
        and a node on a deformable FEM element.

        Parameters
        ----------
        node_position : array-like
            [x, y] coordinates of the node to couple

        Returns
        -------
        np.ndarray
            2×3 constraint matrix

        Examples
        --------
        >>> block = Block_2D(vertices)
        >>> C = block.constraint_matrix_for_node([1.0, 2.0])
        >>> u_node = C @ np.array([u_block, v_block, theta_block])

        Notes
        -----
        The constraint matrix is derived from rigid body kinematics:
            C = [[1,  0,  -(y-y_ref)],
                 [0,  1,   (x-x_ref)]]
        """
        x, y = node_position
        x_c, y_c = self.ref_point

        # Rigid body kinematics constraint matrix
        # u_node = u_b - (y - y_c) * θ
        # v_node = v_b + (x - x_c) * θ

        C = np.array([
            [1, 0, -(y - y_c)],
            [0, 1, (x - x_c)]
        ])

        return C

    def compute_resultant_force_moment(self, nodal_forces, nodal_positions):
        """
        Compute resultant force and moment on the rigid block from nodal forces.

        This method is used in FEM-Block coupling to compute how forces from
        FEM elements are transferred to the rigid block through interface nodes.

        Parameters
        ----------
        nodal_forces : np.ndarray
            Array of shape (n_nodes, 2) containing [Fx, Fy] at each interface node
        nodal_positions : np.ndarray
            Array of shape (n_nodes, 2) containing [x, y] coordinates of each interface node

        Returns
        -------
        F_resultant : np.ndarray
            Resultant force vector [Fx, Fy] on the block
        M_resultant : float
            Resultant moment about the reference point

        Examples
        --------
        >>> nodal_forces = np.array([[10.0, 20.0], [5.0, -10.0]])
        >>> nodal_positions = np.array([[1.0, 0.0], [2.0, 0.0]])
        >>> F, M = block.compute_resultant_force_moment(nodal_forces, nodal_positions)

        Notes
        -----
        Moment is computed about the block's reference point using:
            M = Σ [(x_i - x_ref) * F_y,i - (y_i - y_ref) * F_x,i]
        """
        # Sum all forces for resultant force
        F_resultant = np.sum(nodal_forces, axis=0)

        # Compute moment about reference point
        M_resultant = 0.0
        for i in range(len(nodal_forces)):
            r = nodal_positions[i] - self.ref_point  # Lever arm
            # M = r × F (2D cross product: r_x * F_y - r_y * F_x)
            M_resultant += r[0] * nodal_forces[i, 1] - r[1] * nodal_forces[i, 0]

        return F_resultant, M_resultant
