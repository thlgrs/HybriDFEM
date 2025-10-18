"""
Example usage of the reorganized FEM package
"""
from fem import MeshHandler, Material2D, BoundaryConditions, LinearElasticSolver

def main():
    # Initialize mesh handler
    mesh_handler = MeshHandler("Theo/output/beam_tri0.5.msh")
    mesh_handler.load_mesh()
    mesh_handler.setup_domain()
    
    # Get mesh bounds for boundary conditions
    bounds = mesh_handler.get_bounds()
    min_x, max_x = bounds[:, 0]
    eps = 1e-8 * (max_x - min_x)
    
    # Create regions for boundary conditions
    fixed_region = mesh_handler.create_region(
        'Gamma_Left',
        f"vertices in x < {min_x + eps}",
        'facet'
    )
    load_region = mesh_handler.create_region(
        'Gamma_Right',
        f"vertices in x > {max_x - eps}",
        'facet'
    )
    
    # Setup material
    material = Material2D()  # Using default steel properties
    
    # Setup boundary conditions
    bc_handler = BoundaryConditions(mesh_handler.domain, mesh_handler.regions['omega'])
    fixed_bc = bc_handler.create_fixed_bc('fix_left', fixed_region)
    
    # Create solver
    solver = LinearElasticSolver(mesh_handler, material, bc_handler)
    
    # Solve the problem
    displacements, stiffness = solver.solve(
        dirichlet_bcs=[fixed_bc],
        load_region=load_region,
        force_vector=[0.0, -1e6]  # Vertical load of 1 MPa
    )
    
    print("Displacement shape:", displacements.shape)
    print("Maximum displacement:", np.abs(displacements).max())
    print("Stiffness matrix shape:", stiffness.shape)

if __name__ == '__main__':
    import numpy as np
    main()
