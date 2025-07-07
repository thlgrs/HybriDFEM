import numpy as np
from sfepy.discrete import Integral, Equation, Equations, Problem
from sfepy.terms import Term
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton


class LinearElasticSolver:
    """Main solver class for linear elastic FEM problems"""

    def __init__(self, mesh_handler, material, bc_handler):
        """
        Initialize solver

        Parameters:
        -----------
        mesh_handler : MeshHandler
            Handler for mesh operations
        material : Material2D
            Material properties handler
        bc_handler : BoundaryConditions
            Boundary conditions handler
        """
        self.mesh = mesh_handler
        self.material = material
        self.bc = bc_handler
        self.integral = None
        self._setup_integral()

    def _setup_integral(self):
        """Setup integration order for quadrature"""
        self.integral = Integral("i", order=2)

    def create_elasticity_term(self):
        """Create internal elasticity term"""
        return Term.new(
            "dw_lin_elastic(m.D, v, u)",
            self.integral,
            self.mesh.regions["omega"],
            m=self.material.create_material(),
            v=self.bc.v,
            u=self.bc.u,
        )

    def create_traction_term(self, load_region, force_vector):
        """
        Create traction term for Neumann boundary condition

        Parameters:
        -----------
        load_region : Region
            Region where load is applied
        force_vector : list
            Force components [Fx, Fy]
        """
        load_material = self.material.create_load_material(force_vector)
        return Term.new(
            "dw_surface_ltr(load.val, v)",
            self.integral,
            load_region,
            load=load_material,
            v=self.bc.v,
        )

    def solve(self, dirichlet_bcs, load_region, force_vector=[0.0, -1e6]):
        """
        Solve the linear elastic problem

        Parameters:
        -----------
        dirichlet_bcs : list
            List of Dirichlet boundary conditions
        load_region : Region
            Region where load is applied
        force_vector : list
            Force components [Fx, Fy]

        Returns:
        --------
        tuple
            (displacements, stiffness_matrix)
        """
        # Create terms
        t_internal = self.create_elasticity_term()
        t_traction = self.create_traction_term(load_region, force_vector)

        # Setup equation
        equation = Equation("balance", t_internal - t_traction)
        equations = Equations([equation])

        # Create and setup problem
        problem = Problem("elasticity", equations=equations, domain=self.mesh.domain)
        problem.set_bcs(ebcs=self.bc.create_bc_conditions(dirichlet_bcs))

        # Setup solvers
        ls = ScipyDirect({})
        nls = Newton({"i_max": 1}, lin_solver=ls)
        problem.set_solver(nls)

        # Solve
        solution = problem.solve()

        # Get results
        u_vals = self.bc.u()
        u_vals.shape = (self.mesh.domain.shape.n_nod, 2)
        K_matrix = problem.mtx_a.toarray()

        return u_vals, K_matrix
