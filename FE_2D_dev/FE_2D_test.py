import numpy as np
from Objects.FE_2D import FE_Mesh, FE_Material, FE  # adapt the import path

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Geometry and groups -----------------------------------------------------
    L = 1.0  # beam length  [m]
    h = 0.1  # beam height  [m]
    points = [(0, 0), (L, 0), (L, h), (0, h)]  # counter-clockwise
    edge_groups = {
        "clamped": [3],  # line 3 → left edge (3-0)
        "tip": [1],  # line 1 → right edge (1-2)
    }

    with FE_Mesh(points=points,
                 element_type='triangle',
                 element_size=0.05,  # ~20 elements along the length
                 order=2,  # quadratic T6 elements
                 name="cantilever",
                 edge_groups=edge_groups) as mesh:
        mesh.generate_mesh()  # writes cantilever.msh
        # mesh.plot()                 # optional preview

    # -------------------------------------------------------------------------
    # Material (structural steel S355, plane stress) --------------------------
    E, nu = 210e9, 0.3  # [Pa], –
    mat = FE_Material(E, nu, rho=7850, plane="stress")

    # -------------------------------------------------------------------------
    # FE model ----------------------------------------------------------------
    fe = FE(mesh, mat, thickness=0.01)  # plate thickness 10 mm
    fe.assemble_stiffness_matrix()

    # Boundary conditions -----------------------------------------------------
    fe.dirichlet_on_group("clamped", ux=0.0, uy=0.0)  # built-in
    q = -1e4  # uniform downward traction [N/m] on the tip edge
    fe.neumann_on_group("tip", tx=0.0, ty=q)

    # Solve -------------------------------------------------------------------
    u = fe.solve()
    print(f"Max |u| = {np.linalg.norm(u.reshape(-1, 2), axis=1).max():.4e} m")

    # Export to VTK for ParaView ----------------------------------------------
    fe.export_to_vtk("cantilever.vtk")
