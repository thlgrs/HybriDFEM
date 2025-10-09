from Legacy.Objects import FE, Material

if __name__ == "__main__":
    mesh_file = "triMesh.msh"

    material = Material(E=210e9, nu=0.3, rho=7800, plane="stress")
    fe = FE(mesh_file, material)
    fe.assemble()
    print(fe.K)

    # 5. Apply boundary conditions
    #    Fix left edge nodes (0 & 3) in both u,v
    bc = {0: 0.0, 1: 0.0, 6: 0.0, 7: 0.0}
    fe.apply_dirichlet_bc(bc)
    #    Uniform downward traction q=1e5 Pa on top edge between nodes 3→2
    q = 1e5
    fe.neumann_bc([((3, 2), (0.0, -q))])

    # 6. Solve and display
    d = fe.solve()
    disp = d.reshape(-1, 2)
    print("Nodal displacements [u, v] (m):")
    for i, (ux, uy) in enumerate(disp):
        print(f" Node {i}: ux={ux:.6e}, uy={uy:.6e}")

