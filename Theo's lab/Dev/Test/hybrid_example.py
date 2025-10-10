"""Minimal demonstration of a hybrid structure mixing rigid blocks and a planar FE patch."""

from pathlib import Path

import numpy as np

from Theo.Dev.Objects.FE import FE_2D
from Theo.Dev.Objects.Structure_2D import Hybrid
from Theo.Dev.Objects.Surface_FE import FE_Material, FE_Mesh


def build_hybrid_structure() -> Hybrid:
    """Create a tiny hybrid model coupling a masonry block with a 2D continuum patch."""
    # Create planar continuum mesh and material
    mesh_path = Path(__file__).with_name("triMesh.msh")
    fe_mesh = FE_Mesh(mesh_file=str(mesh_path), element_type="triangle", order=1)
    fe_material = FE_Material(E=3.0e9, nu=0.2, rho=1800.0, plane="stress")
    fe_patch = FE_2D(fe_mesh, fe_material, thickness=0.25)

    # Instantiate hybrid structure and add a rigid block sharing the origin node
    structure = Hybrid()
    block_vertices = np.array(
        [
            [0.0, -0.3],
            [0.0, 0.3],
            [-0.5, 0.3],
            [-0.5, -0.3],
        ],
        dtype=float,
    )
    structure.add_block(block_vertices, rho=2200.0, ref_point=np.array([0.0, 0.0]))
    structure.list_fes.append(fe_patch)

    # Generate nodes and assemble linear stiffness for later use
    structure.make_nodes()
    structure.set_lin_geom(True)

    # Clamp the shared base node and pin one transversal DOF on the free end
    base_node = structure.get_node_id(np.array([0.0, 0.0]))
    if base_node is not None:
        structure.fix_node(base_node, [0, 1, 2])

    tip_node = structure.get_node_id(np.array([10.0, 0.0]))
    if tip_node is not None:
        structure.fix_node(tip_node, 1)

    structure.get_K_str0()
    return structure


def main() -> None:
    structure = build_hybrid_structure()
    print(f"Total nodes      : {len(structure.list_nodes)}")
    print(f"Free DOFs        : {structure.nb_dof_free}")
    print(f"Fixed DOFs       : {structure.nb_dof_fix}")
    print("Leading stiffness block (5x5):")
    print(np.round(structure.K0[:5, :5], 3))


if __name__ == "__main__":
    main()
