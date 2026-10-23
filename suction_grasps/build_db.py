"""Phase 1: suction grasp database for a part (Dex-Net 3.0-style sampling) - from the CAD mesh and the gripper
description only (no robot) -> <out>/<name>.npz, <name>.json, <name>_grasps.png

Steps:
  1. load the part (mesh frame, metres); mass and centre of mass
  2. stable resting poses (trimesh), probability > min_prob
  3. sample `points` surface points (area-weighted), approach = -surface normal, `turns` turns of the head over the
     gripper's turn_range (360 deg unless the head looks the same after a smaller turn)
  4. seal rule (seal.py)               5. head collision at the grasp + straight approach (collide.py)
  6. hold margin (seal.hold_margin)    7. robustness = fraction of PERTURB perturbed copies that pass 4 + 5
     quality = robustness x hold margin
  8. resting-pose tags: in each stable pose the approach points down within max_down_deg and the head and its
     approach clear the support plane
  9. non-maximum suppression, keep the best `keep` with at least MIN_PER_POSE per resting pose; picture

Usage:
  python -m suction_grasps.build_db --gripper my_head.json --parts part1.stl part2.stl --units mm --out db
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import trimesh

from . import part as PT
from . import poses as G
from . import seal as S
from .collide import HeadCollider
from .gripper import Gripper

PERTURB = 10
PERTURB_POS = 0.002                  # m, uniform +- in x, y, z of the tool
PERTURB_TILT = np.radians(3)         # max tilt of the approach axis
NMS_POS, NMS_AXIS, NMS_TURN = 0.005, np.radians(10), np.radians(25)
MIN_PER_POSE = 20


def perturbations(rng):
    """PERTURB fixed small tool-frame offsets (4x4): shift +-PERTURB_POS, tilt up to PERTURB_TILT."""
    out = []
    for _ in range(PERTURB):
        a = rng.uniform(0, 2 * np.pi)
        tilt = trimesh.transformations.rotation_matrix(rng.uniform(0, PERTURB_TILT), [np.cos(a), np.sin(a), 0])
        tilt[:3, 3] = rng.uniform(-PERTURB_POS, PERTURB_POS, 3)
        out.append(tilt)
    return out


def nms(T, quality):
    """Greedy non-maximum suppression, best first. Two grasps are duplicates if their positions are within
    NMS_POS, their approach axes within NMS_AXIS and their heads turned within NMS_TURN."""
    keep = []
    for i in np.argsort(-quality):
        if keep:
            K = T[keep]
            near = np.linalg.norm(K[:, :3, 3] - T[i, :3, 3], axis=1) < NMS_POS
            near &= np.einsum("kj,j->k", K[:, :3, 2], T[i, :3, 2]) > np.cos(NMS_AXIS)
            near &= np.einsum("kj,j->k", K[:, :3, 0], T[i, :3, 0]) > np.cos(NMS_TURN)
            if near.any():
                continue
        keep.append(i)
    return np.array(keep, int)


def build(part_path, gripper_path, out, name=None, units="mm", offset=(0, 0, 0), points=10000, turns=24,
          density=7850.0, mass=None, thickness=None, max_down_deg=60.0, keep_max=1000, min_prob=0.01, seed=0,
          picture=True):
    t0 = time.time()
    g = Gripper.load(gripper_path)
    name = name or os.path.splitext(os.path.basename(part_path))[0]
    rng = np.random.default_rng(seed)
    stats = {}
    # 1-2. part, mass, resting poses
    m = PT.load_part(part_path, units, offset)
    mass, com = PT.properties(m, density, mass, thickness)
    if not np.isfinite(mass):                     # a mesh that is not watertight has no volume to weigh
        stats["warning"] = ("mesh is not watertight and neither --mass nor --thickness is given: mass unknown, hold "
                            "check skipped (pass --mass for a solid part, --thickness for a part modelled as one surface)")
        print(f"{name}: {stats['warning']}", flush=True)
    poses = PT.stable_poses(m, com, min_prob)

    # 3. candidates: surface points x turns, approach into the surface
    pts, fi = m.sample(points, return_index=True, seed=seed)
    turn_list = np.arange(turns) * g.turn_range / turns
    T = np.array([G.tool_from_point(pt, -m.face_normals[f], a) for pt, f in zip(pts, fi) for a in turn_list])
    stats["candidates"] = len(T)

    # 4. seal
    sc = S.SealChecker(m, g)
    r = sc.check(T)
    stats["sealed"] = int(r["valid"].sum())

    # 5. collision (only for sealed grasps)
    hc = HeadCollider(m, g)
    idx = np.flatnonzero(r["valid"])
    idx = idx[np.array([hc.check(T[i])[0] for i in idx], bool)] if len(idx) else idx
    stats["collision_free"] = len(idx)
    T, sealed, resid = T[idx], r["sealed"][idx], r["resid"][idx]

    # 6. hold margin
    hold = np.array([S.hold_margin(t, s, com, mass, g) for t, s in zip(T, sealed)])

    # 7. robustness: perturbed copies through seal + collision
    passed = np.zeros(len(T))
    for D in perturbations(rng):
        Tp = T @ D
        for i in np.flatnonzero(sc.check(Tp)["valid"]):
            passed[i] += hc.check(Tp[i])[0]
    robust = passed / PERTURB
    quality = robust * hold
    good = quality > 0
    T, sealed, resid, hold, robust, quality = T[good], sealed[good], resid[good], hold[good], robust[good], quality[good]
    stats["quality_above_0"] = len(T)

    # 8. resting-pose tags: approach points down, head clears the support plane (z = 0 of the resting frame)
    tags = np.zeros((len(T), len(poses)), bool)
    for k, (T_rest, _) in enumerate(poses):
        down = (T_rest[:3, :3] @ T[:, :3, 2].T).T @ [0, 0, -1] if len(T) else np.zeros(0)
        hc.set_scene(G.inv(T_rest) @ G.pose([*(T_rest @ np.r_[com, 1])[:2], 0]))   # plane z = 0, under the part
        for i in np.flatnonzero(down >= np.cos(np.radians(max_down_deg))):
            tags[i, k] = hc.check(T[i])[0]
        stats[f"usable_in_pose_{k}"] = int(tags[:, k].sum())
    hc.set_scene(None)
    usable = tags.any(1)                                             # a grasp no resting pose can use is dropped
    T, sealed, resid, hold, robust, quality, tags = (a[usable] for a in (T, sealed, resid, hold, robust, quality, tags))

    # 9. prune: NMS, then the best MIN_PER_POSE per pose, filled up to keep_max by quality
    keep = nms(T, quality)
    sel = []
    for k in range(len(poses)):
        sel += [i for i in keep if tags[i, k]][:MIN_PER_POSE]
    chosen = set(sel)
    sel += [i for i in keep if i not in chosen]
    sel = np.array(list(dict.fromkeys(sel))[:keep_max], int)
    sel = sel[np.argsort(-quality[sel])] if len(sel) else sel
    stats.update(after_nms=len(keep), kept=len(sel), kept_per_pose=[int(tags[sel, k].sum()) for k in range(len(poses))],
                 seconds=round(time.time() - t0, 1))

    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, name + ".npz"), T=T[sel], quality=quality[sel], robustness=robust[sel],
                        hold=hold[sel], sealed=sealed[sel], resid=resid[sel], pose_ok=tags[sel],
                        T_rest=np.array([t for t, _ in poses]).reshape(-1, 4, 4), pose_prob=np.array([q for _, q in poses]),
                        mass=mass, com=com)
    info = dict(part=name, mesh=os.path.basename(part_path), units=units, offset=list(offset),
                frame="mesh frame (after the unit scale), metres", mass_kg=mass, com=com.tolist(),
                stable_poses=[dict(down_in_part=(t[:3, :3].T @ [0, 0, -1]).round(4).tolist(), prob=q) for t, q in poses],
                gripper=g.spec, settings=dict(points=points, turns=turns, perturb=PERTURB, perturb_pos_m=PERTURB_POS,
                                              perturb_tilt_deg=float(np.degrees(PERTURB_TILT)), max_down_deg=max_down_deg,
                                              seed=seed), stats=stats, quality_top10=quality[sel][:10].round(3).tolist())
    json.dump(info, open(os.path.join(out, name + ".json"), "w"), indent=1)
    if picture:
        from . import render
        render.db_picture(os.path.join(out, name + ".npz"), m, g)
    return name, stats


def _run(kw):
    return build(**kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gripper", required=True, help="gripper description JSON (see suction_grasps/gripper.py)")
    ap.add_argument("--parts", nargs="+", required=True, help="part mesh files (STL, PLY, OBJ, ...)")
    ap.add_argument("--units", default="mm", choices=sorted(PT.UNITS))
    ap.add_argument("--out", default="db")
    ap.add_argument("--points", type=int, default=10000)
    ap.add_argument("--turns", type=int, default=24)
    ap.add_argument("--offset", type=float, nargs=3, default=(0, 0, 0),
                    help="added to the mesh before scaling, in its units (e.g. a bounding-box-centred export)")
    ap.add_argument("--density", type=float, default=7850.0, help="kg/m^3 (steel by default)")
    ap.add_argument("--mass", type=float, help="kg; overrides volume x density")
    ap.add_argument("--thickness", type=float, help="m; sheet thickness for an open (single-surface) mesh")
    ap.add_argument("--max-down-deg", type=float, default=60.0)
    ap.add_argument("--keep", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    jobs = [dict(part_path=p_, gripper_path=args.gripper, out=args.out, units=args.units, offset=tuple(args.offset),
                 points=args.points, turns=args.turns, density=args.density, mass=args.mass, thickness=args.thickness,
                 max_down_deg=args.max_down_deg, keep_max=args.keep)
            for p_ in args.parts]
    with Pool(args.workers) as pool:
        for name, stats in pool.imap_unordered(_run, jobs):
            print(name, json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
