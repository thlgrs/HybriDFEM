"""
Example demonstrating the usage of the FEM package for solving different 2D elastic problems.
This example includes:
1. Cantilever beam with end load
2. Material property variations
3. Results visualization
"""
import numpy as np

from fem import MeshHandler, Material2D, BoundaryConditions, LinearElasticSolver


def solve_cantilever(mesh_path, material_props=None, force_magnitude=1e6):
    """
    Solve a cantilever beam problem
    
    Parameters:
    -----------
    mesh_path : str
        Path to the mesh file
    material_props : dict, optional
        Material properties (E, nu)
    force_magnitude : float
        Magnitude of the applied force in N/m²
    """
    # 1. Setup mesh
    mesh_handler = MeshHandler(mesh_path)
    mesh_handler.load_mesh()
    mesh_handler.setup_domain()
    
    # 2. Create boundary regions
    bounds = mesh_handler.get_bounds()
    min_x, max_x = bounds[:, 0]
    eps = 1e-8 * (max_x - min_x)
    
    # Fixed (left) end
    fixed_region = mesh_handler.create_region(
        'Gamma_Left',
        f"vertices in x < {min_x + eps}",
        'facet'
    )
    
    # Loaded (right) end
    load_region = mesh_handler.create_region(
        'Gamma_Right',
        f"vertices in x > {max_x - eps}",
        'facet'
    )
    
    # 3. Setup material
    if material_props is None:
        material_props = {'E': 210e9, 'nu': 0.3}  # Default steel
    material = Material2D(
        young_modulus=material_props['E'],
        poisson_ratio=material_props['nu']
    )
    
    # 4. Setup boundary conditions
    bc_handler = BoundaryConditions(mesh_handler.domain, mesh_handler.regions['omega'])
    fixed_bc = bc_handler.create_fixed_bc('fix_left', fixed_region)
    
    # 5. Create and run solver
    solver = LinearElasticSolver(mesh_handler, material, bc_handler)
    
    # Apply vertical load
    force_vector = [0.0, -force_magnitude]
    displacements, stiffness = solver.solve(
        dirichlet_bcs=[fixed_bc],
        load_region=load_region,
        force_vector=force_vector
    )
    
    return {
        'displacements': displacements,
        'stiffness': stiffness,
        'mesh': mesh_handler
    }

def print_results(results):
    """Print analysis results"""
    displacements = results['displacements']
    
    # Calculate max displacement
    max_displacement = np.abs(displacements).max()
    max_vert_displacement = np.abs(displacements[:, 1]).max()
    
    print("\nResults Summary:")
    print("-" * 50)
    print(f"Number of nodes: {len(displacements)}")
    print(f"Maximum total displacement: {max_displacement:.6e} m")
    print(f"Maximum vertical displacement: {max_vert_displacement:.6e} m")
    
def main():
    # Mesh file path
    mesh_path = "Theo/output/beam_tri0.5.msh"
    
    # 1. Solve with default steel properties
    print("\nSolving cantilever beam with steel properties...")
    results_steel = solve_cantilever(mesh_path)
    print("\nSteel Cantilever Results:")
    print_results(results_steel)
    
    # 2. Solve with aluminum properties
    aluminum_props = {'E': 69e9, 'nu': 0.33}  # Aluminum properties
    print("\nSolving cantilever beam with aluminum properties...")
    results_aluminum = solve_cantilever(mesh_path, material_props=aluminum_props)
    print("\nAluminum Cantilever Results:")
    print_results(results_aluminum)
    
    # 3. Solve with increased load
    print("\nSolving steel cantilever with 2x load...")
    results_heavy = solve_cantilever(mesh_path, force_magnitude=2e6)
    print("\nHeavy Load Results:")
    print_results(results_heavy)

if __name__ == '__main__':
    main()
