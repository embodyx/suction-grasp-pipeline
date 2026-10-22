"""Phase 1: suction grasp database for a part (Dex-Net 3.0-style sampling) -> db/<part>.npz, .json, _grasps.png

Steps (PLAN.md, phase 1):
  1. load the part in the CAD frame; mass and centre of mass (closed steel solid)
  2. stable resting poses (trimesh, same method and result as the pose-estimation service), prob > 1%
  3. sample N_POINTS surface points (area-weighted), approach = -surface normal, N_TURNS turns of the head over
     360 deg (the cup square repeats every 90 deg, but the head does not: it has a lobe on one side)
  4. seal rule (seal.py)               5. head collision at the grasp + straight approach (collide.py)
  6. hold margin (seal.hold_margin)    7. robustness = fraction of N_PERTURB perturbed copies that pass 4 + 5
     quality = robustness x hold margin
  8. resting-pose tags: in each stable pose the approach points down within MAX_DOWN_DEG and the head and its
     approach clear the support plane
  9. non-maximum suppression, keep the best MAX_KEEP with at least MIN_PER_POSE per resting pose; picture

Usage: .venv/bin/python build_grasp_db.py [--parts A5917204100 ...] [--workers 3] [--points 20000]
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import trimesh

import gripper as G
import mesh_io as M
import seal as S

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(ROOT, "db")
N_POINTS = 10000                     # ~3.5 mm apart on these parts (finer than NMS_POS)
N_TURNS = 24                         # 0..345 deg in 15 deg steps
MIN_POSE_PROB = 0.01
N_PERTURB = 10
PERTURB_POS = 0.002                  # m, uniform +- in x, y, z of the tool
PERTURB_TILT = np.radians(3)         # max tilt of the approach axis
MAX_DOWN_DEG = 60                    # approach at most this far from straight down (resting-pose tags)
NMS_POS, NMS_AXIS, NMS_TURN = 0.005, np.radians(10), np.radians(25)
MAX_KEEP, MIN_PER_POSE = 1000, 20     # 1000: a top-300 crowds around the centre of mass


def perturbations(rng):
    """N_PERTURB fixed small tool-frame offsets (4x4): shift +-PERTURB_POS, tilt up to PERTURB_TILT."""
    out = []
    for _ in range(N_PERTURB):
        a = rng.uniform(0, 2 * np.pi)
        tilt = trimesh.transformations.rotation_matrix(rng.uniform(0, PERTURB_TILT), [np.cos(a), np.sin(a), 0])
        tilt[:3, 3] = rng.uniform(-PERTURB_POS, PERTURB_POS, 3)
        out.append(tilt)
    return out


def stable_poses(m, com):
    """[(T_rest_part 4x4: part frame -> resting on z = 0, probability)] with probability > MIN_POSE_PROB."""
    tf, prob = trimesh.poses.compute_stable_poses(m, center_mass=com, n_samples=1, threshold=0.0)
    return [(t, float(q)) for t, q in zip(tf, prob) if q > MIN_POSE_PROB]


def nms(T, quality):
    """Greedy non-maximum suppression, best first. Two grasps are duplicates if their positions are within
    NMS_POS, their approach axes within NMS_AXIS and their heads turned within NMS_TURN (full circle: the head
    is not symmetric, so a 90 deg turn is a different grasp)."""
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


def build(name, n_points=N_POINTS, seed=0):
    import collide as C
    t0 = time.time()
    g = G.load()
    rng = np.random.default_rng(seed)
    stats = {}
    # 1-2. part, mass, resting poses
    m = M.load_part(name)
    mass, com = M.properties(m)
    poses = stable_poses(m, com)

    # 3. candidates: surface points x turns, approach into the surface
    pts, fi = m.sample(n_points, return_index=True, seed=seed)
    turns = np.radians(np.arange(N_TURNS) * 360 / N_TURNS)
    T = np.array([G.tool_from_point(pt, -m.face_normals[f], a) for pt, f in zip(pts, fi) for a in turns])
    stats["candidates"] = len(T)

    # 4. seal
    sc = S.SealChecker(m, g)
    r = sc.check(T)
    ok = r["valid"]
    stats["sealed"] = int(ok.sum())

    # 5. collision (only for sealed grasps)
    hc = C.HeadCollider(M.collision_obj(name, m), g)
    idx = np.flatnonzero(ok)
    clear = np.array([hc.check(T[i])[0] for i in idx], bool)
    idx = idx[clear]
    stats["collision_free"] = len(idx)
    T, sealed, resid = T[idx], r["sealed"][idx], r["resid"][idx]

    # 6. hold margin
    hold = np.array([S.hold_margin(t, s, com, mass, g) for t, s in zip(T, sealed)])

    # 7. robustness: perturbed copies through seal + collision
    passed = np.zeros(len(T))
    for D in perturbations(rng):
        Tp = T @ D
        rp = sc.check(Tp)["valid"]
        for i in np.flatnonzero(rp):
            passed[i] += hc.check(Tp[i])[0]
    robust = passed / N_PERTURB
    quality = robust * hold
    good = quality > 0
    T, sealed, resid, hold, robust, quality = T[good], sealed[good], resid[good], hold[good], robust[good], quality[good]
    stats["quality_above_0"] = len(T)

    # 8. resting-pose tags
    tags = np.zeros((len(T), len(poses)), bool)
    for k, (T_rest, _) in enumerate(poses):
        down = (T_rest[:3, :3] @ T[:, :3, 2].T).T @ [0, 0, -1]      # approach . world down
        hc.set_support(G.inv(T_rest))                               # plane z = 0 of the resting frame
        for i in np.flatnonzero(down >= np.cos(np.radians(MAX_DOWN_DEG))):
            tags[i, k] = hc.check(T[i])[0]
        hc.set_support(None)
        stats[f"usable_in_pose_{k}"] = int(tags[:, k].sum())
    usable = tags.any(1)                                             # a grasp no resting pose can use is dropped
    T, sealed, resid, hold, robust, quality, tags = (a[usable] for a in (T, sealed, resid, hold, robust, quality, tags))

    # 9. prune: NMS, then the best MIN_PER_POSE per pose, filled up to MAX_KEEP by quality
    keep = nms(T, quality)
    sel = []
    for k in range(len(poses)):
        sel += [i for i in keep if tags[i, k]][:MIN_PER_POSE]
    sel += [i for i in keep if i not in set(sel)]
    sel = np.array(list(dict.fromkeys(sel))[:MAX_KEEP], int)
    sel = sel[np.argsort(-quality[sel])]
    stats["after_nms"] = len(keep)
    stats["kept"] = len(sel)
    stats["kept_per_pose"] = [int(tags[sel, k].sum()) for k in range(len(poses))]
    stats["seconds"] = round(time.time() - t0, 1)

    os.makedirs(DB, exist_ok=True)
    np.savez_compressed(os.path.join(DB, name + ".npz"), T=T[sel], quality=quality[sel], robustness=robust[sel],
                        hold=hold[sel], sealed=sealed[sel], resid=resid[sel], pose_ok=tags[sel],
                        T_rest=np.array([t for t, _ in poses]), pose_prob=np.array([q for _, q in poses]),
                        mass=mass, com=com)
    info = dict(part=name, frame="CAD frame (m), = pose-estimation T_base_cad", mass_kg=mass, com=com.tolist(),
                stable_poses=[dict(down_in_part=(t[:3, :3].T @ [0, 0, -1]).round(4).tolist(), prob=q) for t, q in poses],
                gripper={k: v for k, v in json.load(open(os.path.join(ROOT, "gripper.json"))).items() if not k.startswith("_")},
                settings=dict(points=n_points, turns=N_TURNS, perturb=N_PERTURB, perturb_pos_m=PERTURB_POS,
                              perturb_tilt_deg=float(np.degrees(PERTURB_TILT)), max_down_deg=MAX_DOWN_DEG, seed=seed),
                stats=stats, quality_top10=quality[sel][:10].round(3).tolist())
    json.dump(info, open(os.path.join(DB, name + ".json"), "w"), indent=1)
    import render
    render.db_picture(name, m, sc)
    return name, stats


def _run(args):
    return build(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=M.SMALL)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--points", type=int, default=N_POINTS)
    args = ap.parse_args()
    with Pool(args.workers) as pool:
        for name, stats in pool.imap_unordered(_run, [(n, args.points) for n in args.parts]):
            print(name, stats, flush=True)


if __name__ == "__main__":
    main()
