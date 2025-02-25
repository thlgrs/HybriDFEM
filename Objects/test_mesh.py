from FE_2D import Mesh

def usage():
    points = [(0, 0), (1, 0), (1, 1), (0, 1)]
    with Mesh(points, "triangle", 0.1, name="test_mesh") as mesh:
        mesh.generate_mesh()
        mesh.plot()

def test_mesh():
    # Data : square domain (preprocessing GUI)
    square_points = [(0, 0), (1, 0), (1, 1), (0, 1)]
    element_type = "triangle"
    element_size = 0.1
    name = "test_mesh"

    # Initialize the Mesh class
    mesh = Mesh(points=square_points, element_type=element_type, element_size=element_size, name=name)

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
    points_found = mesh.find_points((0.5, 0.5), tolerance=1e-5)
    print(f"Points found: {points_found}")


def cantilever(height, length):
    points = [(0, 0), (length, 0), (length, height), (0, height)]
    mesh = Mesh(points, "triangle", 0.1, name="cantilever")
    mesh.generate_mesh()
    mesh.plot()

if __name__ == "__main__":
    test_mesh()
    #cantilever(0.3, 3)