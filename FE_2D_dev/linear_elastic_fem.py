import numpy as np
from sfepy.discrete.fem import Mesh, FEDomain
from sfepy.discrete import Field, FieldVariable
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.discrete import Material
from sfepy.discrete.conditions import EssentialBC, Conditions
from sfepy.discrete import Integral, Equation, Equations
from sfepy.terms import Term
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.discrete import Problem


class LinearElasticFEM:
    def __init__(self, input_mesh_path, young_modulus=210e9, poisson_ratio=0.3):
        """
        Initialize the Linear Elastic FEM solver

        Parameters:
        -----------
        input_mesh_path : str
            Path to the input mesh file (.msh format)
        young_modulus : float
            Young's modulus in Pa (default: 210 GPa for steel)
        poisson_ratio : float
            Poisson's ratio (default: 0.3 for steel)
        """
        self.input_mesh_path = input_mesh_path
        self.E = young_modulus
        self.nu = poisson_ratio

        # Initialize solver components
        self.mesh = None
        self.domain = None
        self.regions = {}
        self.field = None
        self.u = None  # displacement variable
        self.v = None  # test variable

        # Load and setup mesh
        self._setup_mesh()

    def _setup_mesh(self):
        """Set up the mesh and domain"""
        temp_mesh = self.input_mesh_path.replace(".msh", "_temp.vtk")
        self.mesh = Mesh.from_file(self.input_mesh_path)
        self.mesh.coors[:, 2] = 0.0  # ensure 2D
        self.mesh.write(temp_mesh, io="auto")
        self.mesh = Mesh.from_file(temp_mesh)

        # Create domain
        self.domain = FEDomain("domain", self.mesh)

        # Create main region (Omega)
        self.regions["omega"] = self.domain.create_region("Omega", "all")

    def setup_boundaries(self, fixed_region_expr, load_region_expr):
        """
        Set up boundary regions for fixed and loaded areas

        Parameters:
        -----------
        fixed_region_expr : str
            Expression for selecting fixed nodes (e.g., "vertices in x < 0.001")
        load_region_expr : str
            Expression for selecting loaded nodes
        """
        self.regions["fixed"] = self.domain.create_region(
            "Gamma_Fixed", fixed_region_expr, kind="facet"
        )
        self.regions["loaded"] = self.domain.create_region(
            "Gamma_Loaded", load_region_expr, kind="facet"
        )

        # Setup field after regions are defined
        self._setup_field()

    def _setup_field(self):
        """Set up the displacement field and variables"""
        # Create displacement field
        self.field = Field.from_args(
            "displacement", np.float64, "vector", self.regions["omega"], approx_order=1
        )

        # Create variables
        self.u = FieldVariable("u", "unknown", self.field)
        self.v = FieldVariable("v", "test", self.field, primary_var_name="u")

    def solve(self, force_vector=[0.0, -1e6]):
        """
        Solve the linear elastic problem

        Parameters:
        -----------
        force_vector : list
            Force vector [Fx, Fy] in N/m²

        Returns:
        --------
        tuple
            (displacement_field, stiffness_matrix)
        """
        # Material definition
        D_matrix = stiffness_from_youngpoisson(2, self.E, self.nu, plane="stress")
        material = Material("m", D=D_matrix)

        # Load definition
        load = Material("load", val=[force_vector])

        # Boundary conditions
        fix_dofs = EssentialBC("fix", self.regions["fixed"], {"u.all": 0.0})

        # Setup integral
        integral = Integral("i", order=2)

        # Define terms
        t_internal = Term.new(
            "dw_lin_elastic(m.D, v, u)",
            integral,
            self.regions["omega"],
            m=material,
            v=self.v,
            u=self.u,
        )

        t_traction = Term.new(
            "dw_surface_ltr(load.val, v)",
            integral,
            self.regions["loaded"],
            load=load,
            v=self.v,
        )

        # Setup equation
        equation = Equation("balance", t_internal - t_traction)
        equations = Equations([equation])

        # Create and solve problem
        problem = Problem("elasticity", equations=equations, domain=self.domain)
        problem.set_bcs(ebcs=Conditions([fix_dofs]))

        # Setup solvers
        ls = ScipyDirect({})
        nls = Newton({"i_max": 1}, lin_solver=ls)
        problem.set_solver(nls)

        # Solve
        solution = problem.solve()

        # Get results
        u_vals = self.u()
        u_vals.shape = (self.domain.shape.n_nod, 2)
        K_matrix = problem.mtx_a.toarray()

        return u_vals, K_matrix

    def get_mesh_bounds(self):
        """Get the mesh bounding box"""
        return self.domain.get_mesh_bounding_box()


# Example usage:
if __name__ == "__main__":
    # Create solver instance
    fem_solver = LinearElasticFEM("FE_2D_dev/output/beam_tri0.5.msh")

    # Get mesh bounds to help define regions
    bounds = fem_solver.get_mesh_bounds()
    min_x, max_x = bounds[:, 0]
    eps = 1e-8 * (max_x - min_x)

    # Setup boundary regions
    fixed_expr = f"vertices in x < {min_x + eps}"
    loaded_expr = f"vertices in x > {max_x - eps}"
    fem_solver.setup_boundaries(fixed_expr, loaded_expr)

    # Solve with default material properties and loading
    displacements, stiffness = fem_solver.solve()
    print("Displacements shape:", displacements.shape)
    print("Stiffness matrix shape:", stiffness.shape)
