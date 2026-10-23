"""End-to-end test on a synthetic part (tests/shapes.ribbed_plate) and the synthetic test gripper: phase 1 builds a
small database, check_db re-verifies it, phase 2 selects grasps for the part resting on a plane, with and without
the reach / order plug-ins and the local-search fallback; plus a single-cup head. Run: python tests/test_pipeline.py
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import shapes as SH
from suction_grasps import part as PT
from suction_grasps import poses as G
from suction_grasps.build_db import build
from suction_grasps.check_db import check_db
from suction_grasps.gripper import Gripper
from suction_grasps.select import Selector

GRIPPER = os.path.join(HERE, "test_gripper.json")
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def main():
    with tempfile.TemporaryDirectory(prefix="suction_grasps_test_") as tmp:   # removed again at the end
        return run(tmp)


def run(tmp):
    stl = os.path.join(tmp, "ribbed_plate.stl")
    SH.as_mesh(SH.ribbed_plate()).export(stl)

    # ---- phase 1
    name, st = build(stl, GRIPPER, tmp, units="m", points=400, turns=8, picture=False)
    db = os.path.join(tmp, name + ".npz")
    check("phase 1 builds a database", st["kept"] > 50, str({k: st[k] for k in ("candidates", "sealed", "collision_free", "kept")}))
    mesh, g = PT.load_part(stl, "m"), Gripper.load(GRIPPER)
    for cname, (ok, detail) in check_db(db, mesh, g, min_per_pose=1).items():
        if not cname.startswith(">="):                       # coverage of the synthetic part's side poses is not tested
            check(f"check_db: {cname}", ok, detail)

    # ---- phase 2: the part resting in its most likely pose on a plane at z = 0.5, turned 40 deg, shifted
    d = np.load(db)
    T_rest = d["T_rest"][int(np.argmax(d["pose_prob"]))]
    T_world_part = G.pose([0.3, -0.2, 0.5]) @ G.rot_z(np.radians(40)) @ T_rest
    support = dict(point=[0, 0, 0.5], normal=[0, 0, 1])
    sel = Selector(db, g, mesh)
    out, info = sel.select(T_world_part, support, k=5)
    tilt_ok = all(r["tilt_deg"] <= 60 for r in out)
    frame_ok = all(np.allclose(r["T_tool"], T_world_part @ d["T"][r["id"]]) for r in out if r["source"] == "database")
    check("phase 2: 5 grasps, resting pose found, tilts <= 60 deg, T_tool = T_world_part @ T_part_tool",
          len(out) == 5 and info["resting_pose"] is not None and tilt_ok and frame_ok,
          f"{len(out)} grasps, pose {info['resting_pose']}, {info['after_filters']} after filters")
    check("phase 2: ranked by score", all(a["score"] >= b["score"] for a, b in zip(out, out[1:])))

    # reach plug-in: accept only grasps whose tool x axis heads within 45 deg of world +x (like a wrist limit)
    def reach(T):
        h = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
        return dict(ok=abs(h) <= 45, reason="wrist limit", cost=0.01, heading=h)
    out_r, info_r = sel.select(T_world_part, support, reach=reach, k=3)
    check("reach plug-in: only accepted grasps returned, with its extra fields",
          len(out_r) > 0 and all(abs(r["heading"]) <= 45 for r in out_r), f"{len(out_r)} grasps, {info_r['reasons']}")

    # order plug-in: the reach callback records the order grasps are tried in
    tried = []
    def record(T):
        tried.append(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
        return dict(ok=True, reason="")
    key = lambda Ts: np.abs(np.degrees(np.arctan2(Ts[:, 1, 0], Ts[:, 0, 0])))      # smallest |heading| first
    sel.select(T_world_part, support, reach=record, order=key, k=10)
    check("order plug-in: grasps tried in key order", all(abs(a) <= abs(b) + 1e-9 for a, b in zip(tried, tried[1:])),
          f"first headings {np.round(tried[:4], 1).tolist()}")

    # nothing reachable from the database -> local search runs (with the mesh) and its grasps pass the same reach
    calls = {"n": 0}
    def strict(T):
        calls["n"] += 1
        return dict(ok=calls["n"] > 40, reason="first 40 refused")          # refuses every database candidate tried
    # (a tiny budget for the database pass, so it gives up after its first refused candidate)
    out_s, info_s = sel.select(T_world_part, support, reach=strict, k=2, budget=1e-4, fallback_budget=5.0)
    check("local search runs when the database pass finds nothing and returns grasps",
          "local_search" in info_s and len(out_s) > 0 and all(r["source"] == "local search" for r in out_s),
          f"{info_s.get('local_search', {})}")

    # obstacle box right over the part: everything is blocked
    blocked, info_b = sel.select(T_world_part, support, boxes=[dict(center=[0.3, -0.2, 0.56], size=[1, 1, 0.02])],
                                 k=3, fallback=False)
    check("an obstacle over the part blocks every grasp", len(blocked) == 0, str(info_b["reasons"]))

    # ---- an open (single-surface) mesh: no volume -> mass from --thickness, or the hold check is skipped
    plate_top = os.path.join(tmp, "sheet.stl")
    import trimesh
    sheet = trimesh.creation.box(extents=[0.2, 0.15, 0.003])
    sheet.update_faces(sheet.face_normals[:, 2] > 0.9)                    # keep only the top face: an open surface
    sheet.remove_unreferenced_vertices()
    sheet.export(plate_top)
    _, st_t = build(plate_top, GRIPPER, os.path.join(tmp, "sheet"), units="m", points=150, turns=4, thickness=0.003,
                    picture=False)
    _, st_n = build(plate_top, GRIPPER, os.path.join(tmp, "sheet2"), units="m", points=150, turns=4, picture=False)
    check("open mesh: grasps with --thickness, and without it (hold check skipped, warning recorded)",
          st_t["kept"] > 0 and st_n["kept"] > 0 and "warning" in st_n and "warning" not in st_t,
          f"kept {st_t['kept']} / {st_n['kept']}")

    # ---- a single-cup head
    one = os.path.join(tmp, "one_cup.json")
    import json
    json.dump(dict(name="one_cup", cups=[dict(pos=[0, 0, 0], radius=0.015)], cup_height=0.03,
                   head=[dict(type="cylinder", center=[0, 0, -0.07], radius=0.03, height=0.08)],
                   hold=dict(vacuum_kpa=60, safety_g=1.5)), open(one, "w"))
    n1, st1 = build(stl, one, os.path.join(tmp, "one"), units="m", points=200, turns=4, picture=False)
    check("single-cup head: phase 1 works", st1["kept"] > 10, str({k: st1[k] for k in ("sealed", "collision_free", "kept")}))

    print(f"\n{sum(results)}/{len(results)} passed")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
