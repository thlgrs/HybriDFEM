import numpy as np
from sfepy.discrete.fem import Mesh, FEDomain
from sfepy.discrete import Field, FieldVariable
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.discrete import Material
from sfepy.discrete.conditions import EssentialBC
from sfepy.discrete import Integral, Equation, Equations
from sfepy.terms import Term
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.discrete import Problem
from sfepy.discrete.conditions import Conditions

# Load a 2D mesh of a rectangle (triangular elements)
input_mesh = "FE_2D_dev/output/beam_tri0.5.msh"
temp_mesh = "FE_2D_dev/output/temp_beam_tri0.5.vtk"
mesh = Mesh.from_file(input_mesh)
mesh.coors[:, 2] = 0.0  # remove z-component
mesh.write(temp_mesh, io="auto")
mesh = Mesh.from_file(temp_mesh)

domain_data = {"name": "domain", "mesh": mesh}
domain = FEDomain(domain_data["name"], domain_data["mesh"])

# Define regions: entire domain Omega, left and right boundaries (Gamma_Left, Gamma_Right)
region_data = {"name": "Omega", "selector": "all"}  # all elements/cells

omega = domain.create_region(
    region_data["name"], region_data["selector"]
)  # all elements/cells
min_x, max_x = domain.get_mesh_bounding_box()[:, 0]  # domain extent in x-direction
eps = 1e-8 * (max_x - min_x)

left_edge = {
    "name": "Gamma_Left",
    "selector": "vertices in x < %.10f" % (min_x + eps),
    "kind": "facet",
}

right_edge = {
    "name": "Gamma_Right",
    "selector": "vertices in x > %.10f" % (max_x - eps),
    "kind": "facet",
}

gamma_left = domain.create_region(
    left_edge["name"], left_edge["selector"], left_edge["kind"]
)
gamma_right = domain.create_region(
    right_edge["name"], right_edge["selector"], right_edge["kind"]
)

# Define a finite element field for displacement (2 DOF per node for 2D vector)
field_data = {
    "name": "displacement",
    "dtype": np.float64,
    "kind": "vector",
    "region": omega,
    "approx_order": 1,
}

field = Field.from_args(
    field_data["name"],
    field_data["dtype"],
    field_data["kind"],
    field_data["region"],
    approx_order=field_data["approx_order"],
)

# Define field variables: 'u' as unknown (trial) and 'v' as test
field_variable_data = [
    {
        "name": "u",
        "kind": "unknown",
        "field": field,
        "primary_var_name": None,  # 'u' is the primary variable
    },
    {
        "name": "v",
        "kind": "test",
        "field": field,
        "primary_var_name": "u",
    },
]
u = FieldVariable(
    field_variable_data[0]["name"],
    field_variable_data[0]["kind"],
    field_variable_data[0]["field"],
    primary_var_name=field_variable_data[0]["primary_var_name"],
)
v = FieldVariable(
    field_variable_data[1]["name"],
    field_variable_data[1]["kind"],
    field_variable_data[1]["field"],
    primary_var_name=field_variable_data[1]["primary_var_name"],
)

# Material properties for steel
E = 210e9  # Young's modulus in Pascals (210 GPa)
nu = 0.3  # Poisson's ratio
# Compute stiffness matrix for plane stress (dim=2)
D_matrix = stiffness_from_youngpoisson(2, E, nu, plane="stress")
m = Material("m", D=D_matrix)

# Define traction load magnitude (Neumann BC) as a material for convenience
force_vector = {"name": "load", "val": [[0.0], [-1e6]]}
load = Material(force_vector["name"], val=force_vector["val"])  # column vector [tx; ty]

# Essential (Dirichlet) BC: Fix left boundary (all displacement components = 0)
boundary_conditions = {
    "name": "fix_left",
    "region": gamma_left,
    "dofs": {"u.all": 0.0},  # all components of u are fixed
}
fix_left = EssentialBC(
    boundary_conditions["name"],
    boundary_conditions["region"],
    boundary_conditions["dofs"],
)
# (No EssentialBC for right boundary because we'll apply a Neumann traction via the weak form)

# Define integration order for quadrature
integral_data = {"name": "i", "order": 2}  # order 2 is sufficient for linear elasticity
integral = Integral(integral_data["name"], order=integral_data["order"])

# Define the elasticity term over the volume Omega
internal_term_data = {
    "name": "dw_lin_elastic(m.D, v, u)",
    "integral": integral,
    "region": omega,
    "m": m,
    "v": v,
    "u": u,
}
t_internal = Term.new(
    internal_term_data["name"],
    internal_term_data["integral"],
    internal_term_data["region"],
    m=internal_term_data["m"],
    v=internal_term_data["v"],
    u=internal_term_data["u"],
)

# Define the traction term over the boundary Gamma_Right
traction_term_data = {
    "name": "dw_surface_ltr(load.val, v)",
    "integral": integral,
    "region": gamma_right,
    "load": load,
    "v": v,
}
t_traction = Term.new(
    traction_term_data["name"],
    traction_term_data["integral"],
    traction_term_data["region"],
    load=traction_term_data["load"],
    v=traction_term_data["v"],
)

# Combine terms to form the equilibrium equation (internal - external = 0)
equation = Equation("balance", t_internal - t_traction)
equations = Equations([equation])

# Set up the problem with equations and boundary conditions
pb = Problem("elasticity_2D", equations=equations, domain=domain)
pb.set_bcs(ebcs=Conditions([fix_left]))


# Set up solvers: direct linear solver and Newton nonlinear solver
ls = ScipyDirect({})  # empty options = default (direct LU solver)
nls = Newton({"i_max": 1}, lin_solver=ls)  # i_max=1 is enough for linear problem

pb.set_solver(nls)

# Solve the problem
solution = pb.solve()

u_vals = u()  # This returns the current solution for u.
n_nodes = domain.shape.n_nod  # number of nodes in the mesh
u_vals.shape = (n_nodes, 2)  # reshape if there are 2 DOFs per node
print(u_vals)

K_sparse = pb.mtx_a  # the assembled stiffness matrix (sparse)
K_dense = K_sparse.toarray()  # convert to a dense NumPy array
print(K_dense)
