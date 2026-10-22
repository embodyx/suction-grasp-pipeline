"""Part meshes: load in the CAD frame (metres), physical properties, convex decomposition for PyBullet.

Source meshes (assets/parts_src/, copied from the pose-estimation service's models/): obj_XXXXXX.ply in mm,
centred on their bounding box, plus parts.json {id: {part, cad_center_mm, extents_mm}}. The service reports part
poses as T_base_cad (CAD frame), so the CAD frame = mesh + cad_center_mm is the pipeline's PART FRAME.
Each small part is a closed 2 mm steel sheet with 3 weld nuts (separate bodies in the mesh, kept).
"""
import json
import os

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "assets", "parts_src")
OUT = os.path.join(ROOT, "assets", "parts")
DENSITY = 7850.0           # kg/m^3, steel
PIECES = 600               # convex pieces in the collision model (~10 mm each, as in g2_pybullet_sim)

_META = json.load(open(os.path.join(SRC, "parts.json")))
PART_IDS = {d["part"]: int(k) for k, d in _META.items()}       # part name -> model id
SMALL = ["A5917204100", "A5917204200", "A5917204300", "A5917204400", "A5917304100", "A5917304200"]


def load_part(name):
    """Part mesh in the CAD frame, metres. Vertices merged, normals outward (closed solid)."""
    i = PART_IDS[name]
    m = trimesh.load(os.path.join(SRC, f"obj_{i:06d}.ply"), force="mesh")
    m.apply_translation(_META[str(i)]["cad_center_mm"])        # bounding-box-centred -> CAD frame
    m.apply_scale(0.001)                                       # mm -> m
    m.merge_vertices()
    m.fix_normals()
    return m


def properties(m):
    """Mass (kg) and centre of mass (CAD frame, m) of the solid steel part."""
    return float(m.volume * DENSITY), np.asarray(m.center_mass)


def collision_obj(name, m=None):
    """OBJ of PIECES convex pieces (k-means on face centres, each piece = convex hull of its triangles), in the
    CAD frame. PyBullet loads a multi-object OBJ as a compound of convex shapes. Built once, cached."""
    path = os.path.join(OUT, name + "_collision.obj")
    if os.path.exists(path):
        return path
    from sklearn.cluster import MiniBatchKMeans
    m = load_part(name) if m is None else m
    label = MiniBatchKMeans(PIECES, random_state=0, n_init=3, batch_size=4096).fit_predict(m.triangles_center)
    lines, base = [], 1
    for k in range(PIECES):
        v = m.triangles[label == k].reshape(-1, 3)
        if len(v) < 4:
            continue
        try:
            h = trimesh.convex.convex_hull(v)
        except Exception:                      # degenerate (flat) piece - its neighbours cover it
            continue
        lines.append(f"o piece_{k}")
        lines += [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in h.vertices]
        lines += [f"f {i + base} {j + base} {l + base}" for i, j, l in h.faces]
        base += len(h.vertices)
    os.makedirs(OUT, exist_ok=True)
    open(path, "w").write("\n".join(lines) + "\n")
    return path
