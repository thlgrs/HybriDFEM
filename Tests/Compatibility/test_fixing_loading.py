# tests/test_loading_fixing.py
import numpy as np
import pytest


def _prepare_structure(st, n_nodes=3, dof_per_node=3):
    st.list_nodes = [np.array([0.0, 0.0]),
                     np.array([1.0, 0.0]),
                     np.array([2.0, 0.0])]
    st.nb_dofs = dof_per_node * len(st.list_nodes)
    st.P = np.zeros(st.nb_dofs)
    st.P_fixed = np.zeros(st.nb_dofs)
    st.dof_fix = np.array([], dtype=int)
    st.dof_free = np.arange(st.nb_dofs, dtype=int)
    st.nb_dof_fix = 0
    st.nb_dof_free = st.nb_dofs


def _assert_same_load_effect(st_old, st_new):
    assert np.allclose(st_old.P, st_new.P)
    assert np.allclose(st_old.P_fixed, st_new.P_fixed)


def _assert_same_fix_sets(st_old, st_new):
    # compare sets (order may differ)
    assert set(st_old.dof_fix.tolist()) == set(st_new.dof_fix.tolist())
    assert set(st_old.dof_free.tolist()) == set(st_new.dof_free.tolist())
    assert st_old.nb_dof_fix == st_new.nb_dof_fix
    assert st_old.nb_dof_free == st_new.nb_dof_free


@pytest.mark.parametrize("fixed", [False, True])
def test_load_node_equivalence(impls, fixed):
    # Create parallel structures and apply same loads via load_node
    new_mod, old_mod = impls["new"], impls["tests"]
    S = getattr(new_mod, "Structure", None) or getattr(new_mod, "Structure_2D", None)
    SO = getattr(old_mod, "Structure", None) or getattr(old_mod, "Structure_2D", None)
    st_new, st_old = S(), SO()
    _prepare_structure(st_new)
    _prepare_structure(st_old)

    # Apply to single node/dof, list of dofs, list of nodes
    if hasattr(st_new, "load_node") and hasattr(st_old, "load_node"):
        st_new.load_node(1, 0, 10.0, fixed=fixed)
        st_old.load_node(1, 0, 10.0, fixed=fixed)

        st_new.load_node(2, [0, 2], 5.0, fixed=fixed)
        st_old.load_node(2, [0, 2], 5.0, fixed=fixed)

        st_new.load_node([0, 2], 1, 7.0, fixed=fixed)
        st_old.load_node([0, 2], 1, 7.0, fixed=fixed)

        # by coordinates near node 1
        st_new.load_node(np.array([1.0, 0.0]) + 1e-12, 2, 3.0, fixed=fixed)
        st_old.load_node(np.array([1.0, 0.0]) + 1e-12, 2, 3.0, fixed=fixed)

        _assert_same_load_effect(st_old, st_new)
    else:
        pytest.skip("load_node not found on one of the implementations")


def test_fix_node_equivalence(impls):
    new_mod, old_mod = impls["new"], impls["tests"]
    S = getattr(new_mod, "Structure", None) or getattr(new_mod, "Structure_2D", None)
    SO = getattr(old_mod, "Structure", None) or getattr(old_mod, "Structure_2D", None)
    st_new, st_old = S(), SO()
    _prepare_structure(st_new)
    _prepare_structure(st_old)

    if hasattr(st_new, "fix_node") and hasattr(st_old, "fix_node"):
        st_new.fix_node(0, [0, 1])
        st_old.fix_node(0, [0, 1])

        st_new.fix_node([1, 2], 2)
        st_old.fix_node([1, 2], 2)

        st_new.fix_node(np.array([1.0, 0.0]) + 1e-12, 0)
        st_old.fix_node(np.array([1.0, 0.0]) + 1e-12, 0)

        _assert_same_fix_sets(st_old, st_new)
    else:
        pytest.skip("fix_node not found on one of the implementations")
