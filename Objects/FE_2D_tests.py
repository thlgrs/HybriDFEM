import gmsh
import numpy as np
import matplotlib.pyplot as plt
from FE_2D import Mesh  # Assuming the Mesh class is in a file named Mesh.py

def main():
    # Define a square domain
    square_points = [(0, 0), (1, 0), (1, 1), (0, 1)]
    element_type = "triangle"
    element_size = 0.1
    filename = "test_mesh.msh"

    # Initialize the Mesh class
    mesh = Mesh(points=square_points, element_type=element_type, element_size=element_size, filename=filename)

    # Generate the mesh
    print("Generating mesh...")
    mesh.generate_mesh()

    # Plot the generated mesh
    print("Plotting mesh...")
    mesh.plot()

    # Add a point to the mesh and regenerate
    print("Adding a point and regenerating the mesh...")
    mesh.add_point(0.5, 0.5, regen=True)
    mesh.plot()

    # Change the element size and update the mesh
    print("Changing element size to 0.05 and updating the mesh...")
    mesh.change_size(new_size=0.05, regen=True)
    mesh.plot()

    # Change the element type to "quad" and update the mesh
    print("Changing element type to 'quad' and updating the mesh...")
    mesh.change_type(new_type="quad", regen=True)
    mesh.plot()

    # Find specific points in the mesh
    print("Finding points near (0.5, 0.5)...")
    points_found = mesh.find_points((0.5, 0.5), tolerance=1e-4)
    print(f"Points found: {points_found}")

    # Assign a physical group to some points
    print("Assigning physical group 'CenterPoint' to point (0.5, 0.5)...")
    mesh.assign_physical_points([(0.5, 0.5)], "CenterPoint")

    # Find elements containing a specific point
    print("Finding elements containing point (0.5, 0.5)...")
    elements_found = mesh.find_elements((0.5, 0.5), tolerance=1e-6)
    print(f"Elements found: {elements_found}")

    # Assign a physical group to the elements containing (0.5, 0.5)
    print("Assigning physical group 'CenterElement' to elements containing (0.5, 0.5)...")
    mesh.assign_physical_elements((0.5, 0.5), "CenterElement")

    print("Test script completed.")


if __name__ == "__main__":
    gmsh.initialize()
    main()
    gmsh.finalize()
