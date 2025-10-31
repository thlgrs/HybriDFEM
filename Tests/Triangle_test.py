"""
Test script to verify Triangle class correctly inherits from Element2D.

This test verifies:
1. Triangle can be instantiated with proper Material and Geometry2D objects
2. Abstract methods N_dN() and quad_rule() are implemented
3. Parent class methods (Ke, Me_consistent) work correctly
4. FE abstract methods are available through parent class
"""
import numpy as np

from Theo.Objects.FEM.FE import Triangle, Geometry2D
from Theo.Objects.Material.Material import PlaneStress


def main():
    """Test that Triangle can be instantiated with proper parent class constructor."""
    print("\n=== Test 1: Triangle Instantiation ===")

    # Define triangle nodes
    nodes = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]

    # Create material (PlaneStress)
    mat = PlaneStress(E=200e9, nu=0.3, rho=7850.0)

    # Create geometry
    geom = Geometry2D(t=0.01)  # 10mm thickness

    # Create triangle element
    tri = Triangle(nodes, mat, geom)

    """Test that abstract methods N_dN() and quad_rule() are implemented."""
    print("\n=== Test 2: Abstract Methods Implementation ===")

    # Test N_dN at centroid
    xi, eta = 1.0 / 3.0, 1.0 / 3.0
    N, dN_dxi, dN_deta = tri.N_dN(xi, eta)

    print(f"✓ N_dN() method works")
    print(f"  Shape functions at centroid: {N}")
    print(f"  Sum of shape functions: {np.sum(N):.6f} (should be 1.0)")
    print(f"  dN/dξ: {dN_dxi}")
    print(f"  dN/dη: {dN_deta}")

    # Test quad_rule
    XI, ETA, W = tri.quad_rule()

    print(f"\n✓ quad_rule() method works")
    print(f"  Number of integration points: {len(XI)}")
    print(f"  Integration point: (ξ={XI[0]:.4f}, η={ETA[0]:.4f})")
    print(f"  Weight: {W[0]:.4f}")

    """Test that parent class methods work correctly."""
    print("\n=== Test 3: Parent Class Methods ===")

    # Test Jacobian computation
    N, dN_dxi, dN_deta = tri.N_dN(1.0 / 3.0, 1.0 / 3.0)
    J, detJ, Jinv = tri.jacobian(dN_dxi, dN_deta)

    print(f"✓ jacobian() method works")
    print(f"  Jacobian determinant: {detJ:.6f}")
    print(f"  Expected (2×area): {2 * tri.area:.6f}")

    # Test stiffness matrix
    Ke = tri.Ke()

    print(f"\n✓ Ke() method works (inherited from Element2D)")
    print(f"  Stiffness matrix shape: {Ke.shape}")
    print(f"  Matrix is symmetric: {np.allclose(Ke, Ke.T)}")
    print(f"  Matrix norm: {np.linalg.norm(Ke):.3e}")

    # Test mass matrix
    Me = tri.Me_consistent()

    print(f"\n✓ Me_consistent() method works (inherited from Element2D)")
    print(f"  Mass matrix shape: {Me.shape}")
    print(f"  Matrix is symmetric: {np.allclose(Me, Me.T)}")
    print(f"  Total mass: {np.sum(Me):.6e} kg")

    """Test that FE abstract methods are available."""
    print("\n=== Test 4: FE Interface Methods ===")

    # Test get_mass
    M = tri.get_mass()
    print(f"✓ get_mass() method works")
    print(f"  Shape: {M.shape}")

    # Test get_k_glob
    K = tri.get_k_glob()
    print(f"\n✓ get_k_glob() method works")
    print(f"  Shape: {K.shape}")

    # Test make_connect
    tri.make_connect(connect=5, node_number=0)
    print(f"\n✓ make_connect() method works")
    print(f"  Global node index for local node 0: {tri.connect[0]}")
    print(f"  Global DOFs for local node 0: {tri.dofs[0:2]}")
    print(f"  Rotation DOF tracked: {tri.rotation_dofs}")

    """Test stress computation with a simple displacement field."""
    print("\n=== Test 5: Stress Computation ===")

    # Apply a simple displacement field (pure extension in x-direction)
    u = np.array([0.0, 0.0,  # Node 1: no displacement
                  0.001, 0.0,  # Node 2: 1mm in x
                  0.0, 0.0])  # Node 3: no displacement

    stress, strain = tri.compute_stress(u)

    print(f"✓ compute_stress() method works")
    print(f"  Strain [εxx, εyy, γxy]: {strain}")
    print(f"  Stress [σxx, σyy, τxy]: {stress / 1e6} MPa")


if __name__ == "__main__":
    main()
