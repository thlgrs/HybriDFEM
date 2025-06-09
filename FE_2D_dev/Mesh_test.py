from Objects.FE_2D import Mesh


def usage():
    points = [(0, 0), (1, 0), (1, 1), (0, 1)]
    with Mesh(points, "triangle", 0.1, name="test_mesh") as mesh:
        mesh.generate_mesh()
        mesh.plot()


def test_methods():
    # Data
    square_points = [(0, 0), (1, 0), (1, 1), (0, 1)]
    element_type = "triangle"
    element_size = 0.1
    name = "test_mesh"

    # Initialize the Mesh class
    mesh = Mesh(
        points=square_points,
        element_type=element_type,
        element_size=element_size,
        name=name,
    )

    # Generate the mesh
    print("Generating mesh...")
    mesh.generate_mesh()

    # Plot the generated mesh
    print("Plotting mesh...")
    mesh.plot(save_path="mesh_0.png")

    # Add a point to the mesh and regenerate
    print("Adding a point and regenerating the mesh...")
    mesh.add_point(0.5, 0.5, regen=True)
    mesh.plot(save_path="mesh_1.png")

    # Change the element size and update the mesh
    print("Changing element size to 0.05 and updating the mesh...")
    mesh.change_size(new_size=0.05, regen=True)
    mesh.plot(save_path="mesh_2.png")

    # Change the element type to "quad" and update the mesh
    print("Changing element type to 'quad' and updating the mesh...")
    mesh.change_type(new_type="quad", regen=True)
    mesh.plot(save_path="mesh_3.png")

    # Find specific points in the mesh
    print("Finding points near (0.5, 0.5)...")
    points_found = mesh.find_points((0.5, 0.5), tolerance=1e-5)
    print(f"Points found: {points_found}")


def cantilever(height, length, sort, size=0.05, plot=False):
    points = [(0, 0), (length, 0), (length, height), (0, height)]
    mesh = Mesh(points, sort, size, name="output/beam_{}{}".format(sort, size))
    mesh.generate_mesh()
    if plot:
        mesh.plot()


def gen_sizes(sizes=None):
    if sizes is None:
        sizes = [
            1,
            3 / 4,
            1 / 2,
            1 / 3,
            1 / 4,
            1 / 6,
            1 / 8,
            1 / 12,
            1 / 16,
            1 / 24,
            1 / 32,
            1 / 48,
            1 / 64,
            1 / 128,
        ]
    for size in sizes:
        cantilever(1, 10, sort="tri", size=size)
        cantilever(1, 10, sort="quad", size=size)


if __name__ == "__main__":
    gen_sizes()
