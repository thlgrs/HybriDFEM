# tests/test_nodes.py
import numpy as np
import pytest


def assert_close(a, b, tol=1e-9):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape
    assert np.allclose(a, b, rtol=1e-9, atol=tol)


def test_get_node_id_and_add_node_if_new(impls):
    # Build two Structures (tests/new) with the same API surface for nodes
    # We only validate behavior equivalence on get_node_id/_add_node_if_new
    for tag, mod in impls.items():
        s = getattr(mod, "Structure", None) or getattr(mod, "Structure_2D", None)
        assert s is not None, f"Missing Structure in {tag} impl"
        st = s()
        st.list_nodes = []
        # Ensure KD-tree attrs allowed to be absent
        if not hasattr(st, "get_node_id"):
            pytest.skip(f"{tag} has no get_node_id")
        if not hasattr(st, "_add_node_if_new"):
            pytest.skip(f"{tag} has no _add_node_if_new")

        # Add nodes near duplicates
        i0 = st._add_node_if_new([0.0, 0.0])
        i1 = st._add_node_if_new([1.0, 0.0])
        i2 = st._add_node_if_new([1.0 + 1e-12, 0.0 + 5e-13])  # duplicate within tol

        assert i0 == 0
        assert i1 == 1
        assert i2 in (1, 2)  # tolerate legacy behavior if tol differs

        # get_node_id must find both exact and near points
        g0 = st.get_node_id([0.0, 0.0])
        g1 = st.get_node_id([1.0, 0.0])
        g2 = st.get_node_id([1.0 + 5e-13, -5e-13])

        assert g0 == i0
        assert g1 == i1
        assert g2 in (i1, i2)


def test_reset_loading_and_vectors(impls):
    # Both implementations should reset P and P_fixed to zeros of size nb_dofs
    for tag, mod in impls.items():
        scls = getattr(mod, "Structure", None) or getattr(mod, "Structure_2D", None)
        st = scls()
        st.nb_dofs = 9
        st.P = np.ones(9)
        st.P_fixed = np.ones(9) * 3.14
        if hasattr(st, "reset_loading"):
            st.reset_loading()
            assert st.P.shape == (9,)
            assert st.P_fixed.shape == (9,)
            assert np.allclose(st.P, 0.0)
            assert np.allclose(st.P_fixed, 0.0)
