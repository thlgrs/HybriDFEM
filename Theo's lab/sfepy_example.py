def linear_elasticity():
    from sfepy.discrete.fem import Mesh, FEDomain
    import numpy as np

    # Load a 2D mesh of a rectangle (triangular elements)
    mesh = Mesh.from_file("cantilever.msh")
    mesh.coors[:, 2] = 0.0  # remove z-component
    mesh.write("2D_mesh.vtk", io="auto")
    mesh = Mesh.from_file("2D_mesh.vtk")

    domain = FEDomain("domain", mesh)

    # Define regions: entire domain Omega, left and right boundaries (Gamma_Left, Gamma_Right)
    omega = domain.create_region("Omega", "all")  # all elements/cells
    min_x, max_x = domain.get_mesh_bounding_box()[:, 0]  # domain extent in x-direction
    eps = 1e-8 * (max_x - min_x)
    gamma_left = domain.create_region(
        "Gamma_Left", "vertices in x < %.10f" % (min_x + eps), "facet"
    )
    gamma_right = domain.create_region(
        "Gamma_Right", "vertices in x > %.10f" % (max_x - eps), "facet"
    )

    from sfepy.discrete import Field, FieldVariable

    # Define a finite element field for displacement (2 DOF per node for 2D vector)
    field = Field.from_args(
        "displacement",
        np.float64,
        "vector",
        domain.create_region("Omega", "all"),
        approx_order=1,
    )

    # Define field variables: 'u' as unknown (trial) and 'v' as test
    u = FieldVariable("u", "unknown", field)
    v = FieldVariable("v", "test", field, primary_var_name="u")

    from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
    from sfepy.discrete import Material

    # Material properties for steel
    E = 210e9  # Young's modulus in Pascals (210 GPa)
    nu = 0.3  # Poisson's ratio
    # Compute stiffness matrix for plane stress (dim=2)
    D_matrix = stiffness_from_youngpoisson(2, E, nu, plane="stress")
    m = Material("m", D=D_matrix)

    # Define traction load magnitude (Neumann BC) as a material for convenience
    traction_vector = [[0.0], [-1e6]]  # column vector [tx; ty]
    load = Material("load", val=traction_vector)

    from sfepy.discrete.conditions import EssentialBC

    # Essential (Dirichlet) BC: Fix left boundary (all displacement components = 0)
    fix_left = EssentialBC("fix_left", gamma_left, {"u.all": 0.0})
    # (No EssentialBC for right boundary because we'll apply a Neumann traction via the weak form)

    from sfepy.discrete import Integral, Equation, Equations
    from sfepy.terms import Term

    # Define integration order for quadrature
    integral = Integral("i", order=2)

    # Define the elasticity term over the volume Omega
    t_internal = Term.new("dw_lin_elastic(m.D, v, u)", integral, omega, m=m, v=v, u=u)

    # Define the traction term over the boundary Gamma_Right
    t_traction = Term.new(
        "dw_surface_ltr(load.val, v)", integral, gamma_right, load=load, v=v
    )

    # Combine terms to form the equilibrium equation (internal - external = 0)
    equation = Equation("balance", t_internal - t_traction)
    equations = Equations([equation])

    from sfepy.solvers.ls import ScipyDirect
    from sfepy.solvers.nls import Newton
    from sfepy.discrete import Problem
    from sfepy.discrete.conditions import Conditions

    # Set up the problem with equations and boundary conditions
    pb = Problem("elasticity_2D", equations=equations, domain=domain)
    pb.set_bcs(ebcs=Conditions([fix_left]))

    # Set up solvers: direct linear solver and Newton nonlinear solver
    ls = ScipyDirect({})  # empty options = default (direct LU solver)
    nls = Newton({"i_max": 1}, lin_solver=ls)  # i_max=1 is enough for linear problem

    pb.set_solver(nls)

    # Solve the problem
    solution = pb.solve()
    print("Solution:", solution)

    u_vals = u()  # This returns the current solution for u.
    n_nodes = domain.shape.n_nod  # number of nodes in the mesh
    u_vals.shape = (n_nodes, 2)  # reshape if there are 2 DOFs per node
    print(u_vals)

    K_sparse = pb.mtx_a  # the assembled stiffness matrix (sparse)
    K_dense = K_sparse.toarray()  # convert to a dense NumPy array
    print(K_dense)


def linear_elasticity2():
    import numpy as np
    from sfepy.discrete.fem import Mesh, FEDomain, Field
    from sfepy.discrete import (
        Material,
        FieldVariable,
        Integral,
        Equation,
        Equations,
        Problem,
        Conditions,
    )
    from sfepy.discrete.conditions import EssentialBC
    from sfepy.solvers.ls import ScipyDirect
    from sfepy.solvers.nls import Newton
    from sfepy.terms import Term
    from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson

    # ------------------------------------------------------------------
    # 1. Mesh and domain
    # ------------------------------------------------------------------
    mesh = Mesh.from_file("cantilever.msh")
    mesh.coors[:, 2] = 0.0  # remove z-component
    mesh.write("cantilever.vtk", io="auto")
    mesh = Mesh.from_file("cantilever.vtk")

    domain = FEDomain("domain", mesh)

    omega = domain.create_region("Omega", "all")
    min_x, max_x = domain.get_mesh_bounding_box()[:, 0]
    tol = 1e-8 * (max_x - min_x)

    gamma_left = domain.create_region(
        "Gamma_Left", f"vertices in x < {min_x + tol:.10e}", kind="facet"
    )
    gamma_right = domain.create_region(
        "Gamma_Right", f"vertices in x > {max_x - tol:.10e}", kind="facet"
    )

    # ------------------------------------------------------------------
    # 2. Field and variables
    # ------------------------------------------------------------------
    field = Field.from_args("disp", np.float64, "vector", omega, approx_order=1)
    u = FieldVariable("u", "unknown", field)
    v = FieldVariable("v", "test", field, primary_var_name="u")

    # ------------------------------------------------------------------
    # 3. Material data (Eurocode 3 §3.2.6)
    # ------------------------------------------------------------------
    E, nu = 210e9, 0.30  # Pa, –
    D = stiffness_from_youngpoisson(2, E, nu, plane="stress")
    m = Material("m", D=D)

    traction = Material("load", val=[[0.0], [-1e6]])  # −1 MPa on Γ_R

    # ------------------------------------------------------------------
    # 4. Boundary conditions
    # ------------------------------------------------------------------
    ebcs = Conditions([EssentialBC("fix_left", gamma_left, {"u.all": 0.0})])

    # ------------------------------------------------------------------
    # 5. Weak form
    # ------------------------------------------------------------------
    integral = Integral("i", order=2)
    t_int = Term.new("dw_lin_elastic(m.D, v, u)", integral, omega, m=m, v=v, u=u)
    t_ext = Term.new(
        "dw_surface_ltr(load.val, v)", integral, gamma_right, load=traction, v=v
    )

    eq = Equation("balance", t_int - t_ext)
    pb = Problem("cantilever_2D", equations=Equations([eq]), domain=domain)
    pb.set_bcs(ebcs=ebcs)

    # ------------------------------------------------------------------
    # 6. Solvers and run
    # ------------------------------------------------------------------
    ls = ScipyDirect({})
    nls = Newton({"i_max": 1}, lin_solver=ls)
    pb.set_solver(nls)

    state = pb.solve()
    pb.save_state("linear_elasticity.vtk", state)  # for ParaView

    # K_sparse = pb.mtx_a  # the assembled stiffness matrix (sparse)
    # print(K_sparse.toarray())


if __name__ == "__main__":
    linear_elasticity2()
