"""Checks for mesh_io.py on the real parts: units, CAD frame, closed solid, collision model.
Also times the seal and collision checks on a real part. Run: .venv/bin/python tests/test_mesh_io.py"""
import json
import os
import sys
import time

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import collide as C
import gripper as G
import mesh_io as M
import seal as S

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def main():
    meta = json.load(open(os.path.join(M.SRC, "parts.json")))
    cid = p.connect(p.DIRECT)
    for name in M.SMALL:
        d = meta[str(M.PART_IDS[name])]
        m = M.load_part(name)
        ext = (m.bounds[1] - m.bounds[0]) * 1000
        ctr = m.bounds.mean(0) * 1000
        mass, _ = M.properties(m)
        # collision model: a static PyBullet body at the CAD origin; its bounds should match the mesh (+ margin)
        shape = p.createCollisionShape(p.GEOM_MESH, fileName=M.collision_obj(name, m), physicsClientId=cid)
        body = p.createMultiBody(0, shape, physicsClientId=cid)
        lo, hi = p.getAABB(body, physicsClientId=cid)
        aabb_err = max(np.abs(np.array(lo) - m.bounds[0]).max(), np.abs(np.array(hi) - m.bounds[1]).max()) * 1000
        check(f"{name}: extents = parts.json, centre = cad_center (mm), closed, 0.5-1.5 kg, collision bounds",
              np.allclose(ext, d["extents_mm"], atol=0.02) and np.allclose(ctr, d["cad_center_mm"], atol=0.02)
              and m.is_watertight and 0.5 < mass < 1.5 and aabb_err < 3.0,
              f"mass {mass:.3f} kg, collision bounds within {aabb_err:.1f} mm")

    # timing on one real part: random points on its surface, approach along -normal
    g = G.load()
    name = M.SMALL[0]
    m = M.load_part(name)
    pts, fi = m.sample(1000, return_index=True, seed=0)
    T = np.array([G.tool_from_point(pt, -m.face_normals[f], a) for pt, f, a in
                  zip(pts, fi, np.random.default_rng(0).uniform(0, np.pi / 2, len(pts)))])
    sc = S.SealChecker(m, g)
    t0 = time.time()
    r = sc.check(T)
    t_seal = (time.time() - t0) / len(T) * 1000
    hc = C.HeadCollider(M.collision_obj(name), g)
    valid = np.flatnonzero(r["valid"])[:100]
    t0 = time.time()
    clear = [hc.check(T[i])[0] for i in valid]
    t_col = (time.time() - t0) / max(1, len(valid)) * 1000
    print(f"INFO  {name}: {r['valid'].mean() * 100:.0f}% of 1000 random surface grasps seal "
          f"({t_seal:.2f} ms each); {np.mean(clear) * 100:.0f}% of {len(valid)} sealed ones are collision-free "
          f"({t_col:.1f} ms each)")
    check("seal check < 5 ms per grasp, collision check < 50 ms per grasp", t_seal < 5 and t_col < 50)

    print(f"\n{sum(results)}/{len(results)} passed")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
