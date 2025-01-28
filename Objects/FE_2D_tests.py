import unittest
import os
from FE_2D import Mesh

class TestMesh(unittest.TestCase):
    def setUp(self):
        """
        Set up the Mesh instance and generate a test mesh for all methods.
        """
        # Define a square domain
        self.points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.element_type = "triangle"
        self.element_size = 0.1
        self.file = "test_mesh.msh"

        # Create a Mesh instance
        self.mesh = Mesh(
            points=self.points,
            element_type=self.element_type,
            element_size=self.element_size,
            filename=self.file,
        )
        self.mesh.generate_mesh()

    def test_generate_mesh_triangle(self):
        """
        Test mesh generation with triangular elements.
        """
        mesh = Mesh(self.points, "triangle", self.element_size, self.file)
        mesh.generate_mesh()
        self.assertTrue(os.path.exists(self.file), "Mesh file was not created.")
        generated_mesh = mesh.read_mesh()
        self.assertIn("triangle", generated_mesh.cells_dict, "No triangles found in the mesh.")

    def test_generate_mesh_quad(self):
        """
        Test mesh generation with quadrilateral elements.
        """
        mesh = Mesh(self.points, "quad", self.element_size, self.file)
        mesh.generate_mesh()
        self.assertTrue(os.path.exists(self.file), "Mesh file was not created.")
        generated_mesh = mesh.read_mesh()
        self.assertIn("quad", generated_mesh.cells_dict, "No quads found in the mesh.")


    def test_lines(self):
        """
        Test if the `lines` method correctly extracts line elements.
        """
        mesh = Mesh(self.points, "triangle", self.element_size, self.file)
        mesh.generate_mesh()
        lines = mesh.lines()
        self.assertGreater(len(lines), 0, "No line elements found.")
        self.assertEqual(len(lines[0]), 2, "Line element should have 2 nodes.")

    def test_elements(self):
        """
        Test if the `elements` method correctly extracts triangular or quadrilateral elements.
        """
        # Test for triangles
        mesh = Mesh(self.points, "triangle", self.element_size, self.file)
        mesh.generate_mesh()
        elements = mesh.elements()
        self.assertGreater(len(elements), 0, "No triangular elements found.")
        self.assertEqual(len(elements[0]), 3, "Triangle element should have 3 nodes.")

        # Test for quads
        mesh = Mesh(self.points, "quad", self.element_size, self.file)
        mesh.generate_mesh()
        elements = mesh.elements()
        self.assertGreater(len(elements), 0, "No quadrilateral elements found.")
        self.assertEqual(len(elements[0]), 4, "Quad element should have 4 nodes.")

    def test_plot(self):
        """
        Test the `plot` method for visualization (ensure no errors occur during plotting).
        """
        mesh = Mesh(self.points, "triangle", self.element_size, self.file)
        mesh.generate_mesh()
        try:
            mesh.plot()
        except Exception as e:
            self.fail(f"Plotting failed with error: {e}")



    def test_find_points(self):
        """
        Test the `find_points` method.
        """
        # Test with a point in the mesh
        target_point = (0, 0)
        matching_points = self.mesh.find_points(target_point)
        self.assertGreater(len(matching_points), 0, "No matching points found.")
        print(f"Matching points for {target_point}: {matching_points}")

        # Test with a point not in the mesh
        target_point = (2, 2)
        matching_points = self.mesh.find_points(target_point)
        self.assertEqual(len(matching_points), 0, "Points found for a non-existent coordinate.")
        print(f"Matching points for {target_point}: {matching_points}")

    def test_find_lines_between_coordinates(self):
        """
        Test the `find_path` method.
        """
        coord1 = (0, 0)
        coord2 = (1, 0)
        matching_lines = self.mesh.find_path(coord1, coord2)
        self.assertGreater(len(matching_lines), 0, "No matching lines found.")
        print(f"Lines connecting {coord1} and {coord2}: {matching_lines}")

    def test_find_elements_from_coordinate(self):
        """
        Test the `find_elements_from_coordinate` method.
        """
        # Test with a coordinate inside an element
        target_coord = (0.5, 0.5)  # Center of the square
        matching_elements = self.mesh.find_elements_from_coordinate(target_coord)
        self.assertGreater(len(matching_elements), 0, "No matching elements found.")
        print(f"Elements containing {target_coord}: {matching_elements}")

        # Test with a coordinate outside the domain
        target_coord = (2, 2)
        matching_elements = self.mesh.find_elements_from_coordinate(target_coord)
        self.assertEqual(len(matching_elements), 0, "Elements found for an out-of-domain coordinate.")
        print(f"Elements containing {target_coord}: {matching_elements}")


    def tearDown(self):
        """
        Clean up the test mesh file after tests are complete.
        """
        if os.path.exists(self.file):
            os.remove(self.file)

if __name__ == "__main__":
    unittest.main()

