"""Re-verify a phase-1 database against its part mesh and gripper:
  - every saved grasp passes the seal rule and the collision check again (independent re-run)
  - the resting-pose tags reproduce (support plane + approach direction)
  - the saved poses are rigid transforms, sorted by quality, quality = robustness x hold
  - coverage: at least min_per_pose grasps for every resting pose more likely than 5% (reported)

Usage: python -m suction_grasps.check_db --db db/part.npz --mesh part.stl --units mm --gripper my_head.json
"""
import argparse

import numpy as np

from . import part as PT
from . import poses as G
from .collide import HeadCollider
from .gripper import Gripper
from .seal import SealChecker


def check_db(db_path, mesh, gripper, max_down_deg=60.0, min_per_pose=20):
    """{check name: (ok, detail)}."""
    d = np.load(db_path)
    T, q = d["T"], d["quality"]
    out = {}
    R = T[:, :3, :3]
    rigid = len(T) == 0 or (np.abs(np.einsum("nij,nkj->nik", R, R) - np.eye(3)).max() < 1e-9
                            and np.allclose(np.linalg.det(R), 1))
    out["rigid, sorted, quality = robustness x hold"] = (
        bool(rigid and (np.diff(q) <= 1e-12).all() and np.allclose(q, d["robustness"] * d["hold"])), f"{len(T)} grasps")
    r = SealChecker(mesh, gripper).check(T)
    hc = HeadCollider(mesh, gripper)
    clear = np.array([hc.check(t)[0] for t in T], bool)
    out["every grasp seals and is collision-free again"] = (
        bool(r["valid"].all() and clear.all() and (r["sealed"] == d["sealed"]).all()),
        f"seal {int(r['valid'].sum())}/{len(T)}, clear {int(clear.sum())}/{len(T)}")
    tags = np.zeros_like(d["pose_ok"])
    for k, T_rest in enumerate(d["T_rest"]):
        down = (T_rest[:3, :3] @ T[:, :3, 2].T).T @ [0, 0, -1] if len(T) else np.zeros(0)
        com = d["com"]
        hc.set_scene(G.inv(T_rest) @ G.pose([*(T_rest @ np.r_[com, 1])[:2], 0]))   # as build_db: plane under the part
        for i in np.flatnonzero(down >= np.cos(np.radians(max_down_deg))):
            tags[i, k] = hc.check(T[i])[0]
    out["resting-pose tags reproduce"] = (bool((tags == d["pose_ok"]).all()), f"{int((tags != d['pose_ok']).sum())} differ")
    cover = [int(d["pose_ok"][:, k].sum()) for k in range(len(d["pose_prob"]))]
    need = [k for k, pr in enumerate(d["pose_prob"]) if pr > 0.05]
    out[f">= {min_per_pose} grasps per resting pose (prob > 5%)"] = (all(cover[k] >= min_per_pose for k in need),
                                                                      f"per pose {cover}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--units", default="mm")
    ap.add_argument("--offset", type=float, nargs=3, default=(0, 0, 0), help="as given to build_db")
    ap.add_argument("--gripper", required=True)
    args = ap.parse_args()
    res = check_db(args.db, PT.load_part(args.mesh, args.units, args.offset), Gripper.load(args.gripper))
    for name, (ok, detail) in res.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    raise SystemExit(0 if all(ok for ok, _ in res.values()) else 1)


if __name__ == "__main__":
    main()
