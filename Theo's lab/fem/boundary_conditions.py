import numpy as np
from sfepy.discrete import Field, FieldVariable
from sfepy.discrete.conditions import EssentialBC, Conditions


class BoundaryConditions:
    """Handles boundary conditions and field variables for FEM analysis"""

    def __init__(self, domain, omega_region):
        """
        Initialize boundary conditions handler

        Parameters:
        -----------
        domain : FEDomain
            The FEM domain
        omega_region : Region
            The main domain region
        """
        self.domain = domain
        self.omega = omega_region
        self.field = None
        self.u = None  # displacement variable
        self.v = None  # test variable
        self._setup_field()

    def _setup_field(self):
        """Setup displacement field and variables"""
        self.field = Field.from_args(
            "displacement", np.float64, "vector", self.omega, approx_order=1
        )

        # Create variables
        self.u = FieldVariable("u", "unknown", self.field)
        self.v = FieldVariable("v", "test", self.field, primary_var_name="u")

    def create_dirichlet_bc(self, name, region, dofs):
        """
        Create Dirichlet (essential) boundary condition

        Parameters:
        -----------
        name : str
            Name of the boundary condition
        region : Region
            Region where BC is applied
        dofs : dict
            Degrees of freedom specification
        """
        return EssentialBC(name, region, dofs)

    def create_fixed_bc(self, name, region):
        """
        Create fixed boundary condition (all DOFs = 0)

        Parameters:
        -----------
        name : str
            Name of the boundary condition
        region : Region
            Region to fix
        """
        return self.create_dirichlet_bc(name, region, {"u.all": 0.0})

    def create_bc_conditions(self, bcs):
        """
        Create boundary conditions container

        Parameters:
        -----------
        bcs : list
            List of boundary conditions
        """
        return Conditions(bcs)
