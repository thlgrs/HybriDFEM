from collections import deque

import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
import skfem
from matplotlib.collections import LineCollection
from scipy.spatial import KDTree
from skfem.models.poisson import laplace, unit_load
from skfem.helpers import dot, grad  # helpers make forms look nice

from Material import Material


class Mesh:
    def __init__(self, points, element_type, element_size, name="myMesh"):
        self.points_list = points
        self.type = element_type
        self.element_size = element_size
        self.file = name + '.msh'
        self.name = name
        self.groups = 0

    ### Basic methods: getter, setter, reader, generate
    def add_point(self, x, y, regen=False):
        if self.groups != 0: self.clear_mesh()  # if Mesh already generated
        self.points_list.append((x, y))
        if regen: self.generate_mesh()  # Only 2 physical groups Boundary and Domain

    def change_size(self, new_size, regen=False):
        if self.groups != 0: self.clear_mesh()  # if Mesh already generated
        self.element_size = new_size
        if regen: self.generate_mesh()  # Only 2 physical groups Boundary and Domain

    def change_type(self, new_type, regen=False):
        if self.groups != 0: self.clear_mesh()  # if Mesh already generated
        self.type = new_type
        if regen: self.generate_mesh()  # Only 2 physical groups Boundary and Domain

    def clear_mesh(self):
        gmsh.open(self.file)
        gmsh.model.remove()  # Clear everything geometry, physical groups and mesh

    def generate_mesh(self):

        gmsh.model.add(self.name)

        point_ids = []
        for i, (x, y) in enumerate(self.points_list, start=1):
            point_ids.append(gmsh.model.geo.addPoint(x, y, 0, self.element_size, i))

        line_ids = []
        for i in range(len(point_ids)):
            start = point_ids[i]
            end = point_ids[(i + 1) % len(point_ids)]  # Wrap around to form a closed loop
            line_ids.append(gmsh.model.geo.addLine(start, end))

        curve_loop = gmsh.model.geo.addCurveLoop(line_ids)
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])

        # -------------------------------------------------------------------------------------------
        # /!\ Possibly here where we set up domain connected to other elements like blocks or CF /!\
        # --------------------------------------------------------------------------------------------
        gmsh.model.addPhysicalGroup(1, line_ids, 1)  # Physical group for boundary lines
        gmsh.model.addPhysicalGroup(2, [surface], 2)  # Physical group for the domain surface
        gmsh.model.setPhysicalName(1, 1, "Boundary")  # Edge
        gmsh.model.setPhysicalName(2, 2, "Domain")  # Domain
        self.groups = 2  # 2 groups created Boundary and Domain

        gmsh.model.geo.synchronize()
        if self.type == "quad":
            # Set recombination for quads
            gmsh.model.mesh.setRecombine(2, surface)
        gmsh.model.mesh.generate(2)  # Generate a 2D mesh

        gmsh.write(self.file)

    ### Extraction methods ###
    def read_mesh(self):
        return meshio.read(self.file)

    def nodes(self):
        mesh = self.read_mesh()
        nodes = mesh.points[:, :2]  # {i + 1: (coord[0], coord[1]) for i, coord in enumerate(mesh.points)}
        return nodes

    def lines(self):
        mesh = self.read_mesh()
        lines = mesh.cells_dict.get("line", [])
        return lines

    def elements(self):
        mesh = self.read_mesh()
        if self.type == "quad":
            elements = mesh.cells_dict.get("quad", [])  # 4-node quads
        else:
            elements = mesh.cells_dict.get("triangle", [])  # 3-node triangles
        return elements

    def plot(self):
        mesh = self.read_mesh()

        # Extract nodes and elements
        points = mesh.points  # Node coordinates
        lines = []
        for cell_type, elements in mesh.cells_dict.items():
            if cell_type in ["line", "triangle", "quad"]:  # Process only relevant elements
                for element in elements:
                    # Add edges of the element as line segments
                    for i in range(len(element)):
                        start = element[i]
                        end = element[(i + 1) % len(element)]  # Wrap around for closed loops
                        lines.append([points[start][:2], points[end][:2]])

        # Convert to a format compatible with Matplotlib
        line_segments = LineCollection(lines, linewidths=0.5, colors="black")

        # Plot the mesh
        fig, ax = plt.subplots()
        ax.add_collection(line_segments)
        ax.autoscale()
        ax.set_aspect("equal")
        plt.title("2D Mesh Visualization")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    ### Searching methods ###
    def find_points(self, target_coords, tolerance=1e-6):
        """
        Finds point tags in the Gmsh model based on their coordinates.

        :param target_coords: Coordinates (x, y) or [(x1, y1), (x2, y2), ...] to find.
        :param tolerance: Tolerance for floating-point comparison.
        :return: List of point tags that match the coordinates.
        """

        gmsh.open(self.file)

        # Get all points and their coordinates
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)[:, :2]  # Only X, Y

        # Find matching points
        if isinstance(target_coords[0], (float, int)):  # Single coordinate
            target_coords = [target_coords]

        matching_points = []
        for target in target_coords:
            for tag, coord in zip(node_tags, node_coords):
                if np.allclose(coord, target, atol=tolerance):
                    matching_points.append(tag)

        return matching_points

    def find_path(self, coord1, coord2, tolerance=1e-6):
        """
        Finds the shortest path of line elements connecting two coordinates in the mesh.

        :param coord1: First coordinate (x, y).
        :param coord2: Second coordinate (x, y).
        :param tolerance: Tolerance for floating-point comparison.
        :return: List of line element tags that form the path.
        """

        gmsh.open(self.file)

        # Step 1: Get all mesh nodes and their coordinates
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)[:, :2]  # Only X, Y

        # Step 2: Find the nearest mesh nodes to coord1 and coord2
        kdtree = KDTree(node_coords)
        start_node = kdtree.query(coord1, k=1)[1] + 1  # Convert to 1-based indexing
        end_node = kdtree.query(coord2, k=1)[1] + 1

        # Step 3: Get all line elements and their connectivity
        _, _, line_connectivity = gmsh.model.mesh.getElements(1)
        line_connectivity = np.array(line_connectivity[0]).reshape(-1, 2)  # Start, end nodes for each line

        # Step 4: Build a graph from the line connectivity
        graph = {}
        for line_id, (node1, node2) in enumerate(line_connectivity, start=1):
            if node1 not in graph:
                graph[node1] = []
            if node2 not in graph:
                graph[node2] = []
            graph[node1].append((node2, line_id))  # (neighbor, line element ID)
            graph[node2].append((node1, line_id))  # Undirected graph

        # Step 5: Perform BFS to find the shortest path from start_node to end_node
        def bfs_path(graph, start, end):
            queue = deque([(start, [])])
            visited = set()

            while queue:
                current, path = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)

                # Check if we've reached the end
                if current == end:
                    return path

                # Add neighbors to the queue
                for neighbor, line_id in graph.get(current, []):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [line_id]))

            return None  # No path found

        # Find the path
        path = bfs_path(graph, start_node, end_node)

        return path

    def find_elements(self, coord, tolerance=1e-6):
        """ NOT EFFECTIVE check for every element -> change to NN, elements formed by NN and if inside one element
        :param coord: Coordinate (x, y) to check.
        :param tolerance: Tolerance for floating-point comparisons.
        :return: List of element tags that contain the coordinate.
        """

        gmsh.open(self.file)

        # Get all nodes and their coordinates
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)[:, :2]  # Only X, Y

        # Get all elements and their connectivity
        surface_dim = 2  # Dimension for surface elements
        _, _, elem_nodes = gmsh.model.mesh.getElements(surface_dim)

        # Separate triangle and quad elements
        triangle_nodes = np.array(elem_nodes[0]).reshape(-1, 3) if len(elem_nodes) > 0 else []
        quad_nodes = np.array(elem_nodes[1]).reshape(-1, 4) if len(elem_nodes) > 1 else []

        # Function to check if a point is inside a triangle
        def is_point_in_triangle(p, a, b, c):
            # Compute vectors
            v0 = c - a
            v1 = b - a
            v2 = p - a

            # Compute dot products
            dot00 = np.dot(v0, v0)
            dot01 = np.dot(v0, v1)
            dot02 = np.dot(v0, v2)
            dot11 = np.dot(v1, v1)
            dot12 = np.dot(v1, v2)

            # Compute barycentric coordinates
            denom = dot00 * dot11 - dot01 * dot01
            if abs(denom) < tolerance:
                return False  # Degenerate triangle

            inv_denom = 1 / denom
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom

            # Check if point is inside the triangle
            return (u >= -tolerance) and (v >= -tolerance) and (u + v <= 1 + tolerance)

        # Function to check if a point is inside a quadrilateral
        def is_point_in_quad(p, a, b, c, d):
            # Split the quad into two triangles: (a, b, d) and (b, c, d)
            return is_point_in_triangle(p, a, b, d) or is_point_in_triangle(p, b, c, d)

        # Check each element to see if it contains the coordinate
        matching_elements = []

        # Check triangles
        for i, nodes in enumerate(triangle_nodes):
            a, b, c = node_coords[nodes - 1]  # Convert 1-based indexing to 0-based
            if is_point_in_triangle(np.array(coord), a, b, c):
                matching_elements.append(i + 1)  # Element tags are 1-based

        # Check quads
        offset = len(triangle_nodes)  # Quads come after triangles
        for i, nodes in enumerate(quad_nodes):
            a, b, c, d = node_coords[nodes - 1]  # Convert 1-based indexing to 0-based
            if is_point_in_quad(np.array(coord), a, b, c, d):
                matching_elements.append(offset + i + 1)  # Element tags are 1-based

        return matching_elements

    ### Physical domain methods ###
    def physical_exists(self, name, dimension):

        gmsh.open(self.file)
        physical_groups = gmsh.model.getPhysicalGroups(dimension)
        for dim, tag in physical_groups:
            assigned_name = gmsh.model.getPhysicalName(dim, tag)
            if dim == dimension and assigned_name == name:
                return True
        return False

    def add_group(self, name, dimension):
        if self.physical_exists(name, dimension):
            print("Physical group name already used")
            return False
        else:
            self.groups += 1
            return True

    def assign_physical_points(self, point_coords, name, tolerance=1e-6):
        """
        Assign a physical group for specific points in the mesh.

        :param point_coords: List of coordinates [(x1, y1), (x2, y2), ...] to group as physical points.
        :param name: Name of the physical group.
        :param tolerance: Tolerance for floating-point comparison.
        """
        if not self.add_group(name, 0): return
        group_tag = self.groups

        gmsh.open(self.file)

        # Find matching point tags
        point_tags = self.find_points(point_coords, tolerance)
        if point_tags:
            gmsh.model.addPhysicalGroup(0, point_tags, group_tag)  # Dimension 0 = points
            gmsh.model.setPhysicalName(0, group_tag, name)
        gmsh.write(self.file)

    def assign_physical_lines(self, start_coord, end_coord, name, tolerance=1e-6):
        """
        Assign a physical group for lines connecting two coordinates in the mesh.

        :param start_coord: Starting coordinate (x1, y1).
        :param end_coord: Ending coordinate (x2, y2).
        :param name: Name of the physical group.
        :param tolerance: Tolerance for floating-point comparison.
        """

        gmsh.open(self.file)
        if not self.add_group(name, 1): return
        group_tag = self.groups

        # Find matching line tags
        line_tags = self.find_path(start_coord, end_coord, tolerance)
        if line_tags:
            gmsh.model.addPhysicalGroup(1, line_tags, group_tag)  # Dimension 1 = lines
            gmsh.model.setPhysicalName(1, group_tag, name)
        gmsh.write(self.file)

    def assign_physical_elements(self, element_coord, name, tolerance=1e-6):
        """
        Assign a physical group for elements containing a specific coordinate in the mesh.

        :param element_coord: Coordinate (x, y) inside the target elements.
        :param name: Name of the physical group.
        :param tolerance: Tolerance for floating-point comparison.
        """

        gmsh.open(self.file)
        if not self.add_group(name, 2): return
        group_tag = self.groups

        # Find matching element tags
        element_tags = self.find_elements(element_coord, tolerance)
        if element_tags:
            gmsh.model.addPhysicalGroup(2, element_tags, group_tag)  # Dimension 2 = surfaces
            gmsh.model.setPhysicalName(2, group_tag, name)
        gmsh.write(self.file)

    def get_physical_group_tag(self, name, dimension):
        """
        Retrieves the tag of a physical group by its name and dimension.

        :param name: The name of the physical group.
        :param dimension: The dimension of the physical group (0 = points, 1 = lines, 2 = surfaces, etc.).
        :return: The tag of the physical group if found, or None if not found.
        """

        gmsh.open(self.file)

        # Get all physical groups in the specified dimension
        physical_groups = gmsh.model.getPhysicalGroups(dimension)
        for dim, tag in physical_groups:
            if dim == dimension:
                # Get the name of the physical group
                physical_name = gmsh.model.getPhysicalName(dim, tag)
                if physical_name == name:
                    return tag  # Return the matching tag

        return None  # Return None if no match is found


class FE_2D:
    def __init__(self, mesh: Mesh, material: Material):
        self.mesh = mesh
        self.material = material  # recycle from Material class

    def assembly(self, order=1):
        mesh_data = skfem.Mesh.load(self.mesh.file)
        print(mesh_data)
        """points = np.array(self.mesh.nodes())
        elements = np.array(self.mesh.elements())
        match self.mesh.type:
            case 'triangle':
                sk_mesh = skfem.MeshTri(points.T, elements.T)
            case 'quad':
                sk_mesh = skfem.MeshQuad(points.T, elements.T)
            case _:
                pass"""

        Vh = skfem.Basis(mesh_data, skfem.ElementTriP1())  # Define the basis with linear elements

        @skfem.BilinearForm
        def a(u, v, _):
            return dot(grad(u), grad(v))
        @skfem.LinearForm
        def l(v, w):
            x, y = w.x  # global coordinates
            f = np.sin(np.pi * x) * np.sin(np.pi * y)
            return f * v
        A = a.assemble(Vh)  # Stiffness matrix
        b = l.assemble(Vh)  # Load vector (constant)

        """
        D = basis.get_dofs()  # Get Dirichlet boundary DOFs (nodes on the boundary)
        A, b = basis.condense(A, b, D=D)  # Apply Dirichlet conditions (u = 0 on boundary)
        """
        return A, b

    def solve(self):
        ...
