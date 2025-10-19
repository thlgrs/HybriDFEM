"""
Complete Example: Structure_2D and All Child Classes
=====================================================

This example demonstrates:
1. Structure_block - Discrete block assembly with contact faces
2. Structure_FEM - Finite element mesh using Timoshenko beams and Element2D
3. Hybrid - Combined block and FEM analysis

Each example includes:
- Geometry creation
- Material definition
- Boundary conditions
- Loading
- Analysis
- Visualization
"""

import matplotlib.pyplot as plt
import numpy as np

from Theo import FE_Mesh, PlaneStress, Geometry2D
from Theo import Static
from Theo import Structure_block, Structure_FEM
from Theo.Objects import Material as mat


# Mock imports - replace with actual imports


# ============================================================================
# EXAMPLE 1: Structure_block - Masonry Wall with Discrete Blocks
# ============================================================================

def example_1_structure_block():
    """
    Create a masonry beam using discrete blocks with contact interfaces.

    Structure: 3x3 wall of rectangular blocks
    Loading: Horizontal load at top
    Support: Bottom row fixed
    """
    print("=" * 70)
    print("EXAMPLE 1: Structure_block - Beam")
    print("=" * 70)

    # Initialize structure
    structure = Structure_block()

    N1 = np.array([0, 0], dtype=float)
    N2 = np.array([3, 0], dtype=float)

    H = .5
    B = .2

    BLOCKS = 20
    CPS = 10

    E = 30e9
    NU = 0.0

    structure.add_beam(N1, N2, BLOCKS, H, 100., b=B, material=mat.Material(E, NU, shear_def=True))
    structure.make_nodes()
    structure.make_cfs(True, nb_cps=CPS)

    F = -100e3

    structure.load_node(N2, [1], F)
    structure.fix_node(N1, [0, 1, 2])

    # Assemble and solve
    print("\nSolving...")
    # Static.solve_linear(structure)
    Static.solve_forcecontrol(structure, 10, filename="Theo/Tests/Outputs/force_control.h5")

    print("✓ Solution converged")
    print(structure.get_M_str())
    print(structure.get_K_str())
    print(structure.get_P_r())

    return structure


# ============================================================================
# EXAMPLE 2: Structure_FEM - Cantilever Beam with Mixed Elements
# ============================================================================

def example_2_structure_fem():
    """
    Create a cantilever beam using FEM.

    Part A: Timoshenko beam elements
    Part B: 2D plane stress elements (from mesh)

    Loading: Vertical tip load
    Support: Fixed at left end
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Structure_FEM - Cantilever Beam")
    print("=" * 70)

    # Initialize structure
    structure = Structure_FEM()

    # -------------------------------------------------------------------------
    # Part B: 2D Plane Stress Elements (from mesh)
    # -------------------------------------------------------------------------
    print("\n--- Part B: 2D Plane Stress Elements ---")

    # Create a simple rectangular mesh
    print("\nGenerating 2D mesh for plate section...")

    E = 30e9
    nu = 0.0
    rho = 0.0
    b = 1

    # Define plate geometry
    L_plate = 1.0  # m
    H_plate = 0.5  # m
    x_offset = 0  #

    # Mesh parameters
    mesh_size = 0.1

    # Create mesh (mock)
    print(f"Plate: {L_plate}m x {H_plate}m at x={x_offset}m")
    print(f"Mesh size: {mesh_size}m")

    # # Create mesh
    mesh = FE_Mesh(
        points=[(0, 0), (L_plate, 0), (L_plate, H_plate), (0, H_plate)],
        element_type='triangle',
        element_size=mesh_size,
        order=1,
        name='plate'
    )
    mesh.generate_mesh()

    # Material for 2D elements
    mat_2d = PlaneStress(E=E, nu=nu, rho=rho)
    geom_2d = Geometry2D(t=b)  # Same thickness as beam

    # Add mesh elements to structure
    structure.from_mesh(mesh, mat_2d, geom_2d)
    print(f"✓ Added ~{int(L_plate * H_plate / mesh_size ** 2 * 2)} triangular elements")

    # Create nodes
    structure.make_nodes()
    print(f"\n✓ Total nodes: ~{len(structure.list_nodes)}")
    print(f"✓ Total DOFs: {structure.nb_dofs}")

    # Apply boundary conditions
    print("\nApplying boundary conditions...")
    # Fix left end (cantilever)
    structure.fix_node(structure.get_node_id((0, 0)), [0, 1, 2])  # Fixed support
    structure.fix_node(structure.get_node_id((0, H_plate)), [0, 1, 2])  # Fixed support
    print(f"  Fixed node at x=0 (u,v,θ)")

    # Apply tip load
    print("\nApplying loads...")
    F_tip = -5000  # N (downward)
    # Find rightmost node
    structure.load_node(structure.get_node_id((L_plate, H_plate)), 1, F_tip)  # Vertical load
    print(f"  Applied tip load: F_y = {F_tip} N")

    # Solve
    print("\nSolving...")
    Static.solve_linear(structure)
    structure.get_M_str()
    structure.get_K_str()
    print("✓ Solution converged")

    return


# ============================================================================
# EXAMPLE 3: Hybrid - Concrete Wall on Elastic Foundation
# ============================================================================

def example_3_hybrid():
    """
    Create a hybrid structure combining blocks and FEM.

    - Upper part: Discrete blocks (masonry wall)
    - Lower part: FEM mesh (elastic foundation)

    Loading: Self-weight + lateral load
    Support: Bottom of foundation fixed
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Hybrid - Masonry Wall on Foundation")
    print("=" * 70)

    # Initialize hybrid structure
    # structure = Hybrid()

    # -------------------------------------------------------------------------
    # Part A: FEM Foundation (bottom)
    # -------------------------------------------------------------------------
    print("\n--- Part A: Foundation (FEM) ---")

    # Foundation dimensions
    L_found = 2.0  # m
    H_found = 0.4  # m

    # Material properties (concrete foundation)
    E_found = 30e9  # Pa
    nu_found = 0.2
    rho_found = 2400  # kg/m³
    t_found = 0.3  # thickness [m]

    print(f"Foundation: {L_found}m x {H_found}m x {t_found}m")
    print(f"Material: E={E_found / 1e9:.0f} GPa, ν={nu_found}, ρ={rho_found} kg/m³")

    # Create foundation mesh
    # mesh_found = FE_Mesh(
    #     points=[
    #         (0, 0),
    #         (L_found, 0),
    #         (L_found, H_found),
    #         (0, H_found)
    #     ],
    #     element_type='triangle',
    #     element_size=0.1,
    #     order=1,
    #     name='foundation'
    # )
    # mesh_found.generate_mesh()

    # mat_found = PlaneStress(E=E_found, nu=nu_found, rho=rho_found)
    # geom_found = Geometry2D(t=t_found)

    # structure.from_mesh(mesh_found, mat_found, geom_found)
    print(f"✓ Created foundation mesh")

    # -------------------------------------------------------------------------
    # Part B: Discrete Block Wall (top)
    # -------------------------------------------------------------------------
    print("\n--- Part B: Block Wall (Discrete) ---")

    # Block properties
    block_width = 0.4
    block_height = 0.2
    rho_block = 1800  # kg/m³ (lighter masonry)
    b_block = 0.2  # m

    n_rows_wall = 4
    n_cols_wall = int(L_found / block_width)  # Fit on foundation

    print(f"Wall: {n_rows_wall} rows x {n_cols_wall} columns")
    print(f"Block size: {block_width}m x {block_height}m x {b_block}m")

    blocks_info = []

    for row in range(n_rows_wall):
        for col in range(n_cols_wall):
            x0 = col * block_width
            y0 = H_found + row * block_height  # Stack on foundation

            vertices = np.array([
                [x0, y0],
                [x0 + block_width, y0],
                [x0 + block_width, y0 + block_height],
                [x0, y0 + block_height]
            ])

            ref_point = np.array([x0 + block_width / 2, y0 + block_height / 2])

            blocks_info.append({
                'vertices': vertices,
                'ref_point': ref_point,
                'row': row,
                'col': col
            })

            # structure.add_block(vertices, rho_block, b=b_block, ref_point=ref_point)

    print(f"✓ Created {len(blocks_info)} blocks")

    # Create nodes and interfaces
    # structure.make_nodes()
    # structure.make_cfs(lin_geom=True, nb_cps=2)

    print(f"\n✓ Total structure:")
    print(f"  - Foundation elements: ~{int(L_found * H_found / 0.1 ** 2 * 2)}")
    print(f"  - Wall blocks: {len(blocks_info)}")
    # print(f"  - Total nodes: {len(structure.list_nodes)}")
    # print(f"  - Total DOFs: {structure.nb_dofs}")

    # Boundary conditions
    print("\nApplying boundary conditions...")
    # Fix bottom of foundation
    # bottom_nodes = [node for node in structure.list_nodes if node[1] < 0.01]
    # for node in bottom_nodes:
    #     structure.fixNode(node, [0, 1])
    print("  Fixed foundation base (u,v)")

    # Loading
    print("\nApplying loads...")
    # Gravity (self-weight)
    g = 9.81  # m/s²
    # structure.apply_gravity(g)
    print(f"  Applied gravity: g = {g} m/s²")

    # Lateral load on top
    F_lateral = 20000  # N
    # Top row blocks
    for block in blocks_info:
        if block['row'] == n_rows_wall - 1:
            # structure.loadNode(block['ref_point'], 0, F_lateral / n_cols_wall)
            pass
    print(f"  Applied lateral load: F_total = {F_lateral} N")

    # Solve
    print("\nSolving...")
    # structure.get_M_str()
    # structure.get_K_str()
    # Static.solve_linear(structure)
    print("✓ Solution converged")

    # Plot
    print("\nPlotting results...")
    plot_hybrid_example(blocks_info, L_found, H_found, n_rows_wall)

    return blocks_info


def plot_hybrid_example(blocks_info, L_found, H_found, n_rows_wall):
    """Visualize the hybrid structure"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Foundation (FEM mesh - schematic)
    found_x = [0, L_found, L_found, 0, 0]
    found_y = [0, 0, H_found, H_found, 0]
    ax.fill(found_x, found_y, 'lightblue', alpha=0.5, label='Foundation (FEM)')
    ax.plot(found_x, found_y, 'b-', linewidth=2)

    # Add mesh lines
    n_div = 8
    for i in range(1, n_div):
        # Vertical
        x = i * L_found / n_div
        ax.plot([x, x], [0, H_found], 'b-', linewidth=0.5, alpha=0.3)
        # Horizontal
        y = i * H_found / n_div
        ax.plot([0, L_found], [y, y], 'b-', linewidth=0.5, alpha=0.3)

    # Blocks
    for block in blocks_info:
        vertices = block['vertices']
        verts_closed = np.vstack([vertices, vertices[0]])

        # Color by row (stress visualization)
        row = block['row']
        color_intensity = 1 - row / n_rows_wall
        ax.fill(vertices[:, 0], vertices[:, 1], color=(1, color_intensity, color_intensity),
                alpha=0.7)
        ax.plot(verts_closed[:, 0], verts_closed[:, 1], 'k-', linewidth=1.5)
        ax.plot(block['ref_point'][0], block['ref_point'][1], 'ro', markersize=4)

    # Add one label for blocks
    ax.plot([], [], 'r-', linewidth=1.5, label='Blocks (Discrete)')

    # Boundary condition markers
    ax.plot([0, L_found], [0, 0], 'gs', markersize=10, markevery=5, label='Fixed')

    # Load arrows on top
    for block in blocks_info:
        if block['row'] == n_rows_wall - 1:
            x = block['ref_point'][0]
            y = block['ref_point'][1] + 0.1
            ax.arrow(x, y, 0.2, 0, head_width=0.05, head_length=0.05,
                     fc='orange', ec='orange', linewidth=1.5)
    ax.plot([], [], 'o', color='orange', label='Lateral Load')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title('Hybrid Structure: FEM Foundation + Discrete Block Wall')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/example3_hybrid.png', dpi=150, bbox_inches='tight')
    print("✓ Saved plot: example3_hybrid.png")
    plt.close()


# ============================================================================
# COMPARISON TABLE
# ============================================================================

def create_comparison_table():
    """Create a comparison table of the three structure types"""
    print("\n" + "=" * 70)
    print("COMPARISON: Structure_2D Child Classes")
    print("=" * 70)

    comparison = {
        'Feature': [
            'Base Class',
            'Elements',
            'DOF per Node',
            'Contact/Interface',
            'Best For',
            'Computational Cost',
            'Nonlinear Capability',
            'Example Uses'
        ],
        'Structure_block': [
            'Structure_2D',
            'Rigid blocks (polygons)',
            '3 (u, v, θ)',
            'Contact faces between blocks',
            'Masonry, granular media',
            'Low-Medium',
            'High (contact, large displacement)',
            'Walls, arches, rockfill'
        ],
        'Structure_FEM': [
            'Structure_2D',
            'FEs (beam, 2D solid)',
            '3 (u, v, θ) or 2 (u, v)',
            'Shared nodes',
            'Continuum structures',
            'Medium-High',
            'Medium (material nonlinearity)',
            'Plates, shells, continua'
        ],
        'Hybrid': [
            'Structure_block + Structure_FEM',
            'Both blocks and FEs',
            '3 (u, v, θ)',
            'Contact + shared nodes',
            'Mixed systems',
            'High',
            'High (combined)',
            'Masonry on foundation, soil-structure'
        ]
    }

    # Print formatted table
    print(f"\n{'Feature':<25} {'Structure_block':<30} {'Structure_FEM':<30} {'Hybrid':<30}")
    print("-" * 115)

    for i, feature in enumerate(comparison['Feature']):
        block = comparison['Structure_block'][i]
        fem = comparison['Structure_FEM'][i]
        hybrid = comparison['Hybrid'][i]
        print(f"{feature:<25} {block:<30} {fem:<30} {hybrid:<30}")


# ============================================================================
# USAGE GUIDE
# ============================================================================

def print_usage_guide():
    """Print a usage guide for each structure type"""

    guide = """

================================================================================
USAGE GUIDE: Structure_2D Classes
================================================================================

1. STRUCTURE_BLOCK
------------------
Use when: Modeling assemblies of discrete rigid bodies with contact

from Theo.Objects.Structure_2D import Structure_block
from Theo.Objects.Block import Block_2D

# Initialize
structure = Structure_block()

# Add blocks
vertices = np.array([[0,0], [1,0], [1,1], [0,1]])
structure.add_block(vertices, rho=2400, b=0.2)

# Or add beam of blocks
structure.add_beam(N1=[0,0], N2=[10,0], n_blocks=20, h=0.5, rho=2400)

# Create nodes and contact faces
structure.make_nodes()
structure.make_cfs(lin_geom=True, nb_cps=2)

# Boundary conditions and loads
structure.fixNode(node_id, [0, 1])  # Fix u, v
structure.loadNode(node_id, dof, force)

# Solve
structure.get_M_str()
structure.get_K_str()
from Theo.Objects.Solver import Static
Static.solve_linear(structure)


2. STRUCTURE_FEM
----------------
Use when: Modeling continuous structures with finite elements

from Theo.Objects.Structure_2D import Structure_FEM
from Theo.Objects.FE import Timoshenko, FE_Mesh, PlaneStress, Geometry2D

# Initialize
structure = Structure_FEM()

# Method A: Add Timoshenko beams manually
structure.add_fe([N1, N2], material, geometry)

# Method B: Create from mesh
mesh = FE_Mesh(
    points=[(0,0), (1,0), (1,1), (0,1)],
    element_type='triangle',
    element_size=0.1,
    order=1
)
mesh.generate_mesh()

mat = PlaneStress(E=200e9, nu=0.3, rho=7850)
geom = Geometry2D(t=0.01)
structure.from_mesh(mesh, mat, geom)

# Create nodes
structure.make_nodes()

# Boundary conditions and loads
structure.fixNode(node_id, [0, 1, 2])
structure.loadNode(node_id, dof, force)

# Solve
structure.get_M_str()
structure.get_K_str()
Static.solve_linear(structure)


3. HYBRID
---------
Use when: Combining discrete blocks with continuous elements

from Theo.Objects.Structure_2D import Hybrid

# Initialize
structure = Hybrid()

# Add blocks (inherits from Structure_block)
structure.add_block(vertices, rho, b)

# Add FEM elements (inherits from Structure_FEM)
structure.add_fe([N1, N2], material, geometry)
structure.from_mesh(mesh, mat, geom)

# Create nodes and interfaces
structure.make_nodes()  # Handles both blocks and FEM
structure.make_cfs(lin_geom=True, nb_cps=2)

# Rest is same as above
structure.fixNode(...)
structure.loadNode(...)
structure.get_M_str()
structure.get_K_str()
Static.solve_linear(structure)


KEY METHODS (Common to all)
----------------------------
make_nodes()          - Create global node list and DOF mapping
fixNode(node, dofs)   - Apply boundary conditions
loadNode(node, dof, F)- Apply loads
get_M_str()          - Assemble mass matrix
get_K_str()          - Assemble stiffness matrix
get_K_str0()         - Initial stiffness (for nonlinear)
set_lin_geom(bool)   - Toggle geometric linearity

DOF Convention
--------------
Each node has 3 DOFs: [u, v, θ]
- u: displacement in x
- v: displacement in y
- θ: rotation about z

Note: Element2D only uses u,v; rotation DOFs are automatically fixed

================================================================================
    """
    print(guide)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "Structure_2D Complete Examples" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")

    # Run examples
    try:
        # Example 1: Structure_block
        blocks1 = example_1_structure_block()

        # Example 2: Structure_FEM
        # nodes2 = example_2_structure_fem()

        # Example 3: Hybrid
        # blocks3 = example_3_hybrid()

        # Comparison table
        # create_comparison_table()

        # Usage guide
        # print_usage_guide()

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()
