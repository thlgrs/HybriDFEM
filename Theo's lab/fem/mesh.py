from sfepy.discrete.fem import Mesh


class MeshHandler:
    """Handles mesh operations and domain setup for FEM analysis"""

    def __init__(self, mesh_path):
        """
        Initialize mesh handler

        Parameters:
        -----------
        mesh_path : str
            Path to the input mesh file (.msh format)
        """
        self.mesh_path = mesh_path
        self.mesh = None
        self.domain = None
        self.regions = {}

    def load_mesh(self):
        """Load and preprocess the mesh"""
        temp_mesh = self.mesh_path.replace(".msh", "_temp.vtk")
        self.mesh = Mesh.from_file(self.mesh_path)
        self.mesh.coors[:, 2] = 0.0  # ensure 2D
        self.mesh.write(temp_mesh, io="auto")
        self.mesh = Mesh.from_file(temp_mesh)
        return self.mesh

    def get_bounds(self):
        """Get mesh bounding box"""
        if self.domain is None:
            raise RuntimeError("Domain not initialized. Call setup_domain() first.")
        return self.domain.get_mesh_bounding_box()

    def setup_domain(self, domain_name="domain"):
        """Setup FEM domain"""
        from sfepy.discrete.fem import FEDomain

        self.domain = FEDomain(domain_name, self.mesh)
        self.setup_base_region()
        return self.domain

    def setup_base_region(self):
        """Setup the main domain region (Omega)"""
        self.regions["omega"] = self.domain.create_region("Omega", "all")
        return self.regions["omega"]

    def create_region(self, name, selector, kind="facet"):
        """
        Create a new region in the domain

        Parameters:
        -----------
        name : str
            Name of the region
        selector : str
            Selection expression (e.g., "vertices in x < 0.001")
        kind : str
            Type of region ('facet', 'cell', etc.)
        """
        if self.domain is None:
            raise RuntimeError("Domain not initialized. Call setup_domain() first.")

        self.regions[name] = self.domain.create_region(name, selector, kind)
        return self.regions[name]

    def get_region(self, name):
        """Get a region by name"""
        return self.regions.get(name)
