"""Part meshes: load any CAD mesh (STL, PLY, OBJ, ... - anything trimesh reads), physical properties, resting poses.

Part frame = the mesh file's own frame (after the unit scale), so a pose estimator that reports poses of the same
mesh can be used directly in phase 2. Use a closed (watertight) mesh for mass / centre of mass / resting poses.
"""
import numpy as np
import trimesh

UNITS = {"m": 1.0, "mm": 0.001, "cm": 0.01, "in": 0.0254}


def load_part(path, units="mm", offset=(0, 0, 0)):
    """Part mesh in metres. offset (in the file's units) is added before scaling - e.g. to move a bounding-box-centred
    export back to its CAD frame."""
    m = trimesh.load(path, force="mesh")
    m.apply_translation(offset)
    m.apply_scale(UNITS[units])
    m.merge_vertices()
    m.fix_normals()
    return m


def properties(m, density=7850.0, mass=None, thickness=None):
    """(mass kg, centre of mass m). A closed mesh uses its volume x density (steel by default); an open shell (e.g. a
    sheet modelled as one surface) uses its area x `thickness` x density and its area-weighted centroid. A given
    `mass` overrides both. NaN if an open shell has neither (the hold check is then skipped, see build_db)."""
    com = np.asarray(m.center_mass) if m.is_watertight else (m.triangles_center * m.area_faces[:, None]).sum(0) / m.area
    if mass is None:
        if m.is_watertight:
            mass = float(m.volume * density)
        elif thickness is not None:
            mass = float(m.area * thickness * density)
        else:
            mass = float("nan")
    return float(mass), com


def stable_poses(m, com, min_prob=0.01):
    """[(T_rest_part 4x4: part frame -> resting on the plane z = 0, probability)] with probability > min_prob."""
    _, sv, vt = np.linalg.svd(m.vertices - m.vertices.mean(0), full_matrices=False)
    if sv[2] < 1e-9 * max(sv[0], 1e-12):
        # a completely flat part (zero-volume hull - trimesh's search does not end on it): it lies on either face
        out = []
        for down in (vt[2], -vt[2]):
            T = np.eye(4)
            T[:3, :3] = trimesh.geometry.align_vectors(down, [0, 0, -1])[:3, :3]
            c = T[:3, :3] @ com
            T[:3, 3] = [-c[0], -c[1], -(T[:3, :3] @ m.vertices.T)[2].min()]
            out.append((T, 0.5))
        return out
    tf, prob = trimesh.poses.compute_stable_poses(m, center_mass=com, n_samples=1, threshold=0.0)
    return [(t, float(q)) for t, q in zip(tf, prob) if q > min_prob]
