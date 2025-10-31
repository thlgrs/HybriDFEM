from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Union

import numpy as np

from Theo.Objects.FEM.FE import FE
from Theo.Objects.Material.Material import PlaneStress, PlaneStrain


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


@dataclass
class QuadRule:
    # tensor-product rule on [-1,1]x[-1,1]
    xi: np.ndarray
    eta: np.ndarray
    w: np.ndarray


class Element2D(FE):
    """
    Isoparametric 2D element shell: subclasses provide N, dN/dxi, dN/deta,
    natural coordinates of nodes, and a quadrature rule.
    """
    DOFS_PER_NODE = 2  # 2D shell elements: [ux, uy] only (no rotation)

    def __init__(self, nodes: List[Tuple[float, float]], mat: Union[PlaneStrain, PlaneStress], geom: Geometry2D):
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
    def N_dN(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (N, dN_dxi, dN_deta) at (xi,eta)
        N: (nd,), dN_dxi: (nd,), dN_deta: (nd,)
        """
        pass

    @abstractmethod
    def quad_rule(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    def _xy_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.array([n[0] for n in self.nodes])
        y = np.array([n[1] for n in self.nodes])
        return x, y

    def jacobian(self, dN_dxi: np.ndarray, dN_deta: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
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

    def B_matrix(self, dN_dx: np.ndarray, dN_dy: np.ndarray) -> np.ndarray:
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

    def Ke(self) -> np.ndarray:
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

    def Me_consistent(self) -> np.ndarray:
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

    def make_connect(self, connect: int, node_number: int, structure=None) -> None:
        """
        Map local element node to global structure node and DOFs.

        This method now supports variable DOFs per node:
        - If structure provided: uses structure.node_dof_offsets (flexible DOF system)
        - If structure=None: falls back to 3*connect (backward compatibility)

        Args:
            connect: Global node index in Structure_2D.list_nodes
            node_number: Local node index in this element (0 to nd-1)
            structure: Structure_2D instance (optional, for flexible DOF support)
        """
        # Store global node index
        self.connect[node_number] = connect

        # Compute base DOF index for this node
        if structure is not None and hasattr(structure, 'node_dof_offsets') and len(
                structure.node_dof_offsets) > connect:
            # Variable DOF mode: use node_dof_offsets
            base_dof = structure.node_dof_offsets[connect]
        else:
            # Fallback: assume 3 DOFs per node
            base_dof = 3 * connect

        # Map element DOFs (2 per node: u, v) to global structure DOFs
        self.dofs[2 * node_number] = base_dof  # u component
        self.dofs[2 * node_number + 1] = base_dof + 1  # v component

        # Note: No longer tracking rotation_dofs - variable DOF system handles this automatically

    def get_mass(self):
        return self.Me_consistent()

    def get_k_glob(self):
        return self.Ke()

    def get_k_glob0(self):
        # For linear elements, K0 = K (no geometric nonlinearity)
        return self.Ke()

    def get_k_glob_LG(self):
        # For linear elements, geometric stiffness not implemented
        return np.zeros((self.edof, self.edof))

    def get_p_glob(self, q_glob):
        # Internal force vector: F = K * u
        K = self.get_k_glob()
        return K @ q_glob


class Triangle(Element2D):
    """
    Linear triangular element (CST - Constant Strain Triangle) for 2D elasticity.

    Properly inherits from Element2D and implements required abstract methods.
    Uses natural coordinates: (ξ, η) ∈ [0,1] with ζ = 1-ξ-η for the third coordinate.
    """

    def __init__(self, nodes: List[Tuple[float, float]], mat: Union[PlaneStress, PlaneStrain], geom: Geometry2D):
        """
        Initialize triangle element using parent Element2D constructor.

        Args:
            nodes: List of 3 node coordinates [(x1,y1), (x2,y2), (x3,y3)]
            mat: Material object (PlaneStress or PlaneStrain)
            geom: Geometry2D object containing thickness
        """
        # Call parent constructor - this sets up nodes, mat, t, nd, dpn, edof, dofs, connect, rotation_dofs
        super().__init__(nodes, mat, geom)

        # Verify we have exactly 3 nodes for a triangle
        if self.nd != 3:
            raise ValueError(f"Triangle element requires exactly 3 nodes, got {self.nd}")

        # Additional triangle-specific attributes (optional, for convenience)
        self.area = self._compute_area()

    def _compute_area(self) -> float:
        """
        Compute triangle area using cross product formula.
        Uses nodes from parent class.
        """
        x, y = self._xy_arrays()
        return 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))

    def N_dN(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Shape functions and derivatives for linear triangle in natural coordinates.

        Natural coordinates: (ξ, η) ∈ [0,1] with ζ = 1-ξ-η
        N1 = ζ = 1-ξ-η  (at node 1)
        N2 = ξ          (at node 2)
        N3 = η          (at node 3)

        Args:
            xi: First natural coordinate (0 to 1)
            eta: Second natural coordinate (0 to 1)

        Returns:
            N: Shape functions at (ξ,η) - array of shape (3,)
            dN_dxi: ∂N/∂ξ - array of shape (3,)
            dN_deta: ∂N/∂η - array of shape (3,)
        """
        zeta = 1.0 - xi - eta

        # Shape functions
        N = np.array([zeta, xi, eta])

        # Derivatives with respect to natural coordinates
        # ∂N/∂ξ = [∂N1/∂ξ, ∂N2/∂ξ, ∂N3/∂ξ] = [-1, 1, 0]
        dN_dxi = np.array([-1.0, 1.0, 0.0])

        # ∂N/∂η = [∂N1/∂η, ∂N2/∂η, ∂N3/∂η] = [-1, 0, 1]
        dN_deta = np.array([-1.0, 0.0, 1.0])

        return N, dN_dxi, dN_deta

    def quad_rule(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Quadrature rule for triangle in natural coordinates.

        Uses 1-point Gauss rule (centroid) for linear triangle.
        For linear elements with constant strain, 1-point rule is exact.

        Returns:
            XI: Array of ξ coordinates for integration points
            ETA: Array of η coordinates for integration points
            W: Array of weights (note: area of reference triangle is 0.5)
        """
        # 1-point rule at centroid (exact for linear triangle)
        XI = np.array([1.0 / 3.0])
        ETA = np.array([1.0 / 3.0])
        W = np.array([0.5])  # Weight for reference triangle with area 0.5

        return XI, ETA, W

    # Keep legacy B_matrix method for backward compatibility and stress computation
    def B_matrix_legacy(self) -> np.ndarray:
        """
        Alternative B-matrix computation using direct formula (constant for linear triangle).
        This is the original implementation - kept for backward compatibility.
        Gives identical results to parent's B_matrix() method.
        """
        x, y = self._xy_arrays()

        # Shape function derivatives (constant throughout element)
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])

        # B matrix (3x6)
        B = np.zeros((3, 6))
        for i in range(3):
            B[0, 2 * i] = b[i]
            B[1, 2 * i + 1] = c[i]
            B[2, 2 * i] = c[i]
            B[2, 2 * i + 1] = b[i]

        B = B / (2 * self.area)
        return B

    def compute_stress(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute stress and strain from displacement vector.

        Args:
            u: Displacement vector [u1, v1, u2, v2, u3, v3]

        Returns:
            stress: Stress vector [σxx, σyy, τxy]
            strain: Strain vector [εxx, εyy, γxy]
        """
        # Use parent's B_matrix method (or legacy version - they're identical for triangles)
        N, dN_dxi, dN_deta = self.N_dN(1.0 / 3.0, 1.0 / 3.0)  # At centroid
        J, detJ, Jinv = self.jacobian(dN_dxi, dN_deta)
        grads_nat = np.vstack((dN_dxi, dN_deta))
        grads_xy = Jinv @ grads_nat
        dN_dx, dN_dy = grads_xy[0], grads_xy[1]
        B = self.B_matrix(dN_dx, dN_dy)

        # Compute strain and stress
        strain = B @ u
        stress = self.mat.D @ strain

        return stress, strain

    def compute_nodal_forces(self, u: np.ndarray) -> np.ndarray:
        """
        Compute internal nodal forces from displacement vector.

        This is the constitutive chain:
        u → ε = B*u → σ = D*ε → F_int = K*u

        Args:
            u: Displacement vector [u1, v1, u2, v2, u3, v3]

        Returns:
            F_internal: Internal force vector [F1x, F1y, F2x, F2y, F3x, F3y]
        """
        K = self.get_k_glob()
        F_internal = K @ u
        return F_internal
