# tests/test_interfaces.py
import numpy as np
import pytest


def _iface_signature(iface, tol=1e-8):
    """Return a hashable summary ignoring endpoint order but keeping geometry."""
    a = np.asarray(iface["x_e1"], float)
    b = np.asarray(iface["x_e2"], float)
    # order-insensitive: sort by x then y
    p, q = (a, b) if (a[0], a[1]) <= (b[0], b[1]) else (b, a)
    # round for hash
    rp = tuple(np.round(p, 8))
    rq = tuple(np.round(q, 8))
    # include block identities in a stable way (by connect if present)
    A = iface.get("Block A");
    B = iface.get("Block B")
    keyA = getattr(A, "connect", None)
    keyB = getattr(B, "connect", None)
    return (rp, rq, keyA, keyB)


def _collect_signatures(struct, tol=1e-8):
    if not hasattr(struct, "detect_interfaces"):
        pytest.skip("detect_interfaces not available")
    interfaces = struct.detect_interfaces()
    return set(_iface_signature(ifc, tol) for ifc in interfaces)


def test_detect_interfaces_equivalence(toy_scene):
    data, _, _ = toy_scene
    new_s = data["new"];
    old_s = data["old"]

    sig_new = _collect_signatures(new_s)
    sig_old = _collect_signatures(old_s)

    # Same number and geometry (within rounding)
    assert sig_new == sig_old


def test_make_cfs_equivalence(toy_scene):
    data, _, _ = toy_scene
    new_s = data["new"];
    old_s = data["old"]

    if not hasattr(new_s, "make_cfs") or not hasattr(old_s, "make_cfs"):
        pytest.skip("make_cfs not available")

    new_s.make_cfs(lin_geom=True, nb_cps=2)
    old_s.make_cfs(lin_geom=True, nb_cps=2)

    # length and block linkage equivalence
    assert len(getattr(new_s, "list_cfs", [])) == len(getattr(old_s, "list_cfs", []))

    # Optional: verify each CF references its blocks similarly by 'connect'
    def cf_signature(cf):
        A = getattr(cf, "bl_A", None) or cf["Block A"]
        B = getattr(cf, "bl_B", None) or cf["Block B"]
        if hasattr(cf, "x_e1"):
            p, q = np.asarray(cf.x_e1), np.asarray(cf.x_e2)
        else:
            p, q = np.asarray(cf["x_e1"]), np.asarray(cf["x_e2"])
        if (p[0], p[1]) > (q[0], q[1]):
            p, q = q, p
        return (tuple(np.round(p, 8)), tuple(np.round(q, 8)),
                getattr(A, "connect", None), getattr(B, "connect", None))

    new_sigs = {cf_signature(cf) for cf in getattr(new_s, "list_cfs", [])}
    old_sigs = {cf_signature(cf) for cf in getattr(old_s, "list_cfs", [])}
    assert new_sigs == old_sigs
