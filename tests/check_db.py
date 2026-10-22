"""Database checks (PLAN.md test plan) on the built db/<part>.npz files:
  - coverage: at least MIN_PER_POSE grasps for every resting pose with probability > 5%
  - the saved grasps are proper rigid transforms, sorted by quality, quality = robustness x hold
  - every saved grasp passes the seal rule and the collision check again (independent re-run)
  - the resting-pose tags reproduce (support plane + approach direction) on every saved grasp
Run: .venv/bin/python tests/check_db.py [parts...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_grasp_db as B
import collide as C
import gripper as G
import mesh_io as M
import seal as S

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def main(parts):
    g = G.load()
    for name in parts:
        d = np.load(os.path.join(B.DB, name + ".npz"))
        T, q = d["T"], d["quality"]
        cover = [int(d["pose_ok"][:, k].sum()) for k in range(len(d["pose_prob"]))]
        need = [k for k, pr in enumerate(d["pose_prob"]) if pr > 0.05]
        check(f"{name}: >= {B.MIN_PER_POSE} grasps per resting pose (prob > 5%)",
              all(cover[k] >= B.MIN_PER_POSE for k in need), f"{len(T)} grasps, per pose {cover}")
        R = T[:, :3, :3]
        rigid = np.abs(np.einsum("nij,nkj->nik", R, R) - np.eye(3)).max() < 1e-9 and np.allclose(np.linalg.det(R), 1)
        check(f"{name}: rigid poses, sorted by quality, quality = robustness x hold",
              rigid and (np.diff(q) <= 1e-12).all() and np.allclose(q, d["robustness"] * d["hold"]))
        m = M.load_part(name)
        r = S.SealChecker(m, g).check(T)
        hc = C.HeadCollider(M.collision_obj(name, m), g)
        clear = np.array([hc.check(t)[0] for t in T])
        check(f"{name}: all grasps seal and are collision-free on a re-run",
              r["valid"].all() and clear.all() and (r["sealed"] == d["sealed"]).all(),
              f"seal {r['valid'].sum()}/{len(T)}, clear {clear.sum()}/{len(T)}")
        tags = np.zeros_like(d["pose_ok"])
        for k, T_rest in enumerate(d["T_rest"]):
            down = (T_rest[:3, :3] @ T[:, :3, 2].T).T @ [0, 0, -1]
            hc.set_support(G.inv(T_rest))
            for i in np.flatnonzero(down >= np.cos(np.radians(B.MAX_DOWN_DEG))):
                tags[i, k] = hc.check(T[i])[0]
        hc.set_support(None)
        check(f"{name}: resting-pose tags reproduce", (tags == d["pose_ok"]).all(),
              f"{(tags != d['pose_ok']).sum()} differ")
    print(f"\n{sum(results)}/{len(results)} passed")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:] or M.SMALL) else 1)
