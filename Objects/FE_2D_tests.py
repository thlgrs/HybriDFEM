#!/usr/bin/env python3
"""
Test file for the FE_2D module.
This test simulates a square domain (1 m x 1 m) under uniaxial load:
- The left edge is fixed (clamped).
- A force is applied in the x direction on the right edge.
"""

import numpy as np
from FE_2D import Mesh, FE_2D, Material  # assuming the provided code is in fe_module.py
import gmsh


def main():
    # ------------------------------
    # 1. Generate a rectangle mesh using gmsh
    # ------------------------------
    # Define a square (counter-clockwise order) with coordinates in meters
    length = 10
    height = 1
    points = [(0.0, 0.0), (length, 0.0), (length, height), (0.0, height)]
    element_size = .5

    # Create and generate the mesh (using quad)
    mesh = Mesh(points, "quad", element_size, name="cantilever")
    mesh.generate_mesh()
    print("Mesh generated and saved as", mesh.file)

    # Finalize gmsh if not using context manager
    gmsh.finalize()

    # ------------------------------
    # 2. Define material properties (e.g., steel)
    # ------------------------------
    E = 210e9  # Young's modulus in Pa
    nu = 0.3  # Poisson's ratio (dimensionless)
    rho = 7850
    material = Material(E, nu, rho)

    # Example usage
    fe_model = FE_2D(mesh, material)
    fe_model.define_support(1, (True, True, True, True, False, False))  # Pinned support at node 1
    fe_model.define_support(4, (True, True, True, True, False, False))  # Pinned support at node 4
    fe_model.add_nodal_load(3, (0, -100000, 0, 0, 0, 0))  # 10 kN downward force at node 2
    fe_model.analyze()
    displacement = fe_model.get_node_displacement(2)
    print(f"Displacement at node 2: {displacement}")
    fe_model.visualize()  # Visualize the results

if __name__ == '__main__':
    main()
