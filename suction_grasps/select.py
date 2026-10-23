"""Phase 2: choose suction grasps at execution time, guided by the phase-1 database. Nothing here moves a robot.

Input: the part's database, its pose T_world_part (4x4, metres; the pose of the same mesh frame the database was
built in, e.g. from a pose estimator), optionally the support plane the part rests on and obstacle boxes (same
world frame). Optional plug-ins for a robot:
  reach(T_world_tool) -> dict(ok=bool, reason=str, cost=float (optional), ...)   reachability (e.g. IK); extra
                         keys are returned with the grasp
  order(T_world_tools (n, 4, 4)) -> (n,) keys, lower first                        e.g. try wrist-friendly turns first
Steps:
  1. database grasps -> world frame: T_world_tool = T_world_part @ T_part_tool
  2. resting pose: the stable pose whose "down" direction matches the support normal (within POSE_TOL); only
     grasps tagged for it are used. No support given or no match (part tilted, leaning) -> all grasps.
  3. fast filters: approach within max_down_deg of straight down (support normal, else world -z); the head outline
     clears the support plane by SUPPORT_CLEAR at the grasp and at the pre-grasp
  4. in order (order() keys, then quality): the head and its approach clear the support and obstacle boxes (exact
     check), then reach() if given; stops after k grasps or `budget` seconds
  5. guided local search, only when nothing survives and the part mesh is given: variants of the best database
     grasps of the resting pose (turns about the approach axis, small shifts in the cup plane) checked online
     (seal rule, hold margin, head collision with part + support, robustness), then step 4. Own time budget.
  6. rank: score = quality - W_TILT x tilt / max_down_deg - reach()'s cost

Command line (dry run): python -m suction_grasps.select --db db/part.npz --gripper my_head.json --request req.json
  req.json: {"T_world_part": 4x4 (m), "support": {"point": [x, y, z], "normal": [0, 0, 1]},
             "boxes": [{"center": [..], "size": [..], "quat_xyzw": [..]}]}
"""
import json
import time

import numpy as np

from . import poses as G
from .collide import APPROACH_DIST, HeadCollider
from .gripper import Gripper

POSE_TOL = np.radians(15)          # resting-pose match tolerance
SUPPORT_CLEAR = 0.002              # m, head outline above the support plane
PREGRASP = APPROACH_DIST           # m, the approach distance checked in phase 1
W_TILT = 0.10                      # ranking cost for the approach tilt (quality is 0..1)
# step 5 (guided local search)
N_SEEDS, SEED_SPACING = 30, 0.010  # best database grasps of the resting pose, at least this far apart
TURN_STEP_DEG = 10                 # turns about the approach axis
N_SHIFTS, SHIFT = 2, 0.005         # random shifts in the cup plane per seed (+ the unshifted one), m
MAX_CHECKED = 60                   # variants that pass collision + hold + robustness before step 4


def plane_frame(support, centre):
    """Support plane as a frame: origin = `centre` (world) projected onto the plane, +z = the plane normal. The
    collision check stands in a finite slab for the plane, so it is centred under the part."""
    n = np.asarray(support["normal"], float) / np.linalg.norm(support["normal"])
    c = np.asarray(centre, float)
    return G.tool_from_point(c - n * ((c - np.asarray(support["point"], float)) @ n), n)


def box_list(boxes):
    """[{"center", "size", "quat_xyzw"}] -> [(T 4x4, size)]."""
    return [(G.pose(b["center"], b.get("quat_xyzw", [0, 0, 0, 1])), np.asarray(b["size"], float)) for b in boxes]


class Selector:
    def __init__(self, db_path, gripper, mesh=None):
        """db_path: phase-1 database (.npz); gripper: Gripper or path to its JSON; mesh: the part mesh in the
        database's frame (trimesh, metres) - only needed for the local search (step 5)."""
        d = np.load(db_path)
        self.db = {k: d[k] for k in d.files}
        self.g = gripper if isinstance(gripper, Gripper) else Gripper.load(gripper)
        self.mesh = mesh
        pts = self.g.outline_points()
        self.outline = np.c_[pts, np.ones(len(pts))]       # head outline points, tool frame, homogeneous
        self.scene = HeadCollider(None, self.g)            # head vs support + boxes, world frame

    # ------------------------------------------------------------------ steps 2-3
    def resting_pose(self, T_world_part, normal):
        """Index of the stable pose the part is in (its down direction along -normal), or None."""
        best, best_ang = None, POSE_TOL
        for k, T_rest in enumerate(self.db["T_rest"]):
            down = T_world_part[:3, :3] @ (T_rest[:3, :3].T @ [0, 0, -1])
            ang = np.arccos(np.clip(down @ -normal, -1, 1))
            if ang < best_ang:
                best, best_ang = k, ang
        return best

    def fast_filter(self, T, support, max_down_deg, pregrasp):
        """Approach tilt and head-outline-above-the-support filters on world tool poses T (n, 4, 4).
        Returns (indices passing, tilt in deg)."""
        n = np.asarray(support["normal"], float) / np.linalg.norm(support["normal"]) if support else np.array([0, 0, 1.0])
        tilt = np.degrees(np.arccos(np.clip(T[:, :3, 2] @ -n, -1, 1)))
        ok = tilt <= max_down_deg
        if support:
            h0 = np.asarray(support["point"], float) @ n
            for back in (0.0, pregrasp):
                pts = np.einsum("nij,cj->nci", T @ G.pose([0, 0, -back]), self.outline)[..., :3]
                ok &= (pts @ n - h0).min(1) >= SUPPORT_CLEAR
        return np.flatnonzero(ok), tilt

    # ------------------------------------------------------------------ step 4 + 6
    def try_grasps(self, T, quality, tilt, ids, source, reach, k, deadline, out, info, max_down_deg, pregrasp):
        """Check the given world tool poses in order (scene clearance, then reach); append accepted ones to out."""
        for j in range(len(T)):
            if len(out) >= k or time.time() > deadline:
                break
            info["tried"] += 1
            clear, why = self.scene.check(T[j], approach=pregrasp)
            r = dict(ok=True, reason="no reachability check")
            if clear and reach is not None:
                r = reach(T[j])
            if not clear or not r["ok"]:
                key = (why if not clear else r.get("reason", "unreachable")).split(" (")[0]
                info["reasons"][key] = info["reasons"].get(key, 0) + 1
                continue
            extra = {kk: v for kk, v in r.items() if kk not in ("ok", "reason", "cost")}
            score = quality[j] - W_TILT * tilt[j] / max_down_deg - r.get("cost", 0.0)
            out.append(dict(id=ids[j], source=source, score=float(score), quality=float(quality[j]),
                            tilt_deg=float(tilt[j]), T_tool=T[j], T_pregrasp=T[j] @ G.pose([0, 0, -pregrasp]), **extra))

    # ------------------------------------------------------------------ steps 1-6
    def select(self, T_world_part, support=None, boxes=(), reach=None, order=None, k=5, budget=2.0, fallback=True,
               fallback_budget=3.0, max_down_deg=60.0, pregrasp=PREGRASP, use_database=True):
        """Ranked grasps (best first) and an info dict (counts, rejection reasons, time). See the module doc."""
        t0 = time.time()
        T_world_part = np.asarray(T_world_part, float)
        centre = (T_world_part @ np.r_[self.db["com"], 1])[:3]                            # the part's centre
        T_plane = plane_frame(support, centre) if support else None
        self.scene.set_scene(T_plane, box_list(boxes))
        T = T_world_part @ self.db["T"]                                                     # step 1
        pose = self.resting_pose(T_world_part, np.asarray(support["normal"], float) /
                                 np.linalg.norm(support["normal"])) if support else None      # step 2
        idx, tilt = self.fast_filter(T, support, max_down_deg, pregrasp)                   # step 3
        if pose is not None:
            idx = idx[self.db["pose_ok"][idx, pose]]
        info = dict(resting_pose=None if pose is None else int(pose), database=len(T), after_filters=len(idx),
                    tried=0, reasons={})
        keys = np.asarray(order(T[idx]), float) if order is not None and len(idx) else np.zeros(len(idx))
        idx = idx[np.lexsort((-self.db["quality"][idx], keys))] if use_database else idx[:0]
        out = []
        self.try_grasps(T[idx], self.db["quality"][idx], tilt[idx], [int(i) for i in idx], "database", reach, k,
                        t0 + budget, out, info, max_down_deg, pregrasp)                    # step 4
        if not out and fallback and self.mesh is not None:                                 # step 5
            info["local_search"] = self.local_search(T_world_part, support, boxes, pose, reach, order, k,
                                                     time.time() + fallback_budget, out, max_down_deg, pregrasp)
        out.sort(key=lambda r: -r["score"])                                                # step 6
        info["seconds"] = round(time.time() - t0, 3)
        return out, info

    # ------------------------------------------------------------------ step 5
    def local_search(self, T_world_part, support, boxes, pose, reach, order, k, deadline, out, max_down_deg, pregrasp):
        """Variants of the best database grasps for this resting pose (turns about the approach axis, small shifts in
        the cup plane), checked online on the part mesh, then step 4. Appends accepted grasps to out; returns info."""
        from . import seal as S
        from .build_db import perturbations
        if not hasattr(self, "_seal"):                     # online checkers on the part mesh (part frame), built once
            self._seal = S.SealChecker(self.mesh, self.g)
            self._head = HeadCollider(self.mesh, self.g)
            self.mass, self.com = float(self.db["mass"]), self.db["com"]
        sc, hc = self._seal, self._head
        info = dict(seeds=0, variants=0, sealed=0, clear=0, tried=0, reasons={})
        tagged = np.flatnonzero(self.db["pose_ok"][:, pose]) if pose is not None else np.arange(len(self.db["T"]))
        seeds = []                                         # best database grasps, at least SEED_SPACING apart
        for i in tagged[np.argsort(-self.db["quality"][tagged])]:
            if all(np.linalg.norm(self.db["T"][i][:3, 3] - self.db["T"][j][:3, 3]) >= SEED_SPACING for j in seeds):
                seeds.append(i)
            if len(seeds) >= N_SEEDS:
                break
        info["seeds"] = len(seeds)
        rng = np.random.default_rng(0)
        turns = np.radians(np.arange(0, 360, TURN_STEP_DEG))
        V, origin = [], []
        for i in seeds:                                    # variants in the part frame: shift, then turn
            for shift in [(0.0, 0.0)] + [tuple(rng.uniform(-SHIFT, SHIFT, 2)) for _ in range(N_SHIFTS)]:
                Ts = self.db["T"][i] @ G.pose([shift[0], shift[1], 0])
                V += [Ts @ G.rot_z(a) for a in turns]
                origin += [int(i)] * len(turns)
        V, origin = np.array(V).reshape(-1, 4, 4), np.array(origin, int)
        info["variants"] = len(V)
        r = sc.check(V)                                    # seal rule on the exact mesh + fast world-frame filters
        idx, tilt = self.fast_filter(T_world_part @ V, support, max_down_deg, pregrasp)
        keep = np.intersect1d(np.flatnonzero(r["valid"]), idx)
        info["sealed"] = int(r["valid"].sum())
        # head collision against the part and the support (part frame), best seed quality first
        Ti = G.inv(T_world_part)
        hc.set_scene(Ti @ plane_frame(support, (T_world_part @ np.r_[self.com, 1])[:3]) if support else None,
                     [(Ti @ T_b, s) for T_b, s in box_list(boxes)])
        keep = keep[np.argsort(-self.db["quality"][origin[keep]], kind="stable")]
        good, qual = [], []
        for j in keep:
            if time.time() > deadline or len(good) >= MAX_CHECKED:
                break
            if not hc.check(V[j], approach=pregrasp)[0]:
                continue
            hold = S.hold_margin(V[j], r["sealed"][j], self.com, self.mass, self.g)
            Tp = np.array([V[j] @ D for D in perturbations(np.random.default_rng(j))])   # as phase 1: seal + collision
            sealed_p = sc.check(Tp)["valid"]
            robust = np.mean([bool(s) and hc.check(t, approach=pregrasp)[0] for s, t in zip(sealed_p, Tp)])
            if hold * robust > 0:
                good.append(j)
                qual.append(hold * robust)
        info["clear"] = len(good)
        good, qual = np.array(good, int), np.array(qual)
        Tw = T_world_part @ V[good] if len(good) else np.zeros((0, 4, 4))
        keys = np.asarray(order(Tw), float) if order is not None and len(good) else np.zeros(len(good))
        o = np.lexsort((-qual, keys)) if len(good) else np.zeros(0, int)
        self.try_grasps(Tw[o], qual[o], tilt[good[o]], [f"search from {origin[j]}" for j in good[o]], "local search",
                        reach, k, deadline, out, info, max_down_deg, pregrasp)
        return info


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--gripper", required=True)
    ap.add_argument("--request", required=True)
    ap.add_argument("--mesh", help="part mesh (enables the local search fallback)")
    ap.add_argument("--units", default="mm")
    ap.add_argument("--offset", type=float, nargs=3, default=(0, 0, 0), help="as given to build_db")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    req = json.load(open(args.request))
    mesh = None
    if args.mesh:
        from . import part as PT
        mesh = PT.load_part(args.mesh, args.units, args.offset)
    sel = Selector(args.db, args.gripper, mesh)
    out, info = sel.select(np.array(req["T_world_part"], float), req.get("support"), req.get("boxes", ()), k=args.k)
    print(json.dumps(info))
    for r in out:
        pos, quat = G.pos_quat(r["T_tool"])
        print(json.dumps(dict(id=r["id"], source=r["source"], score=round(r["score"], 3), quality=round(r["quality"], 3),
                              tilt_deg=round(r["tilt_deg"], 1), tool_pos=pos.round(4).tolist(),
                              tool_quat_xyzw=quat.round(4).tolist())))


if __name__ == "__main__":
    main()
