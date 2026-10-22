"""Phase 2: choose suction grasps at execution time, guided by the phase-1 database. Nothing here moves a robot.

Input: part name, part pose T_base_part (4x4, metres, CAD frame in base_link: the pose-estimation service's
T_base_cad_mm with mm -> m), the scene (support plane + obstacle boxes, base frame), optionally the current joints.
Steps (PLAN.md, phase 2):
  1. database grasps -> base frame:  T_base_tool = T_base_part @ T_part_tool  (no 90 deg expansion: each stored
     turn was collision-checked on its own)
  2. resting pose: the stable pose whose "down" direction matches the support normal (within POSE_TOL);
     only grasps tagged for it are used. No match (part tilted, leaning) -> all grasps, checked directly.
  3. fast filters: approach within MAX_DOWN_DEG of straight down; the EVS08 body box clears the support plane
     by SUPPORT_CLEAR at the grasp and at the pre-grasp (PREGRASP back along the approach)
  4. reachability: collision-free IK (right arm + body lift/waist joints, as in the sim) at the pre-grasp, at the
     grasp (seeded from the pre-grasp) and at APPROACH_STEPS points in between; joints at least LIMIT_MARGIN_DEG
     inside their limits. The part (cups may touch it at the grasp), the scene boxes and the floor are obstacles;
     the idle left arm is not checked. Stops after k reachable grasps or `budget` seconds.
     Order: grasps whose tool x axis heads within HEADING_TOL of HEADING (base frame) first, each group by
     quality. The wrist (joint 7 +-88 deg, joint 6 +-58 deg) cannot reach most other turns (measured 2026-10-22:
     heading 60..180 deg 78/100 reachable, -90..0 deg 1/75); nothing is dropped, only tried later.
  5. guided local search, only when no database grasp is reachable (local_search): variants of the best database
     grasps of the resting pose - turns about the approach axis in TURN_STEP_DEG steps that land in the preferred
     wrist heading, plus small shifts in the cup plane - checked online on the part mesh (seal rule, hold margin,
     head collision incl. the support plane, seal robustness), then the same IK as step 4. Own time budget.
  6. rank: score = quality - small costs for approach tilt, wrist turn from q_now and joint-limit closeness.

Usage (dry run on a JSON request): .venv/bin/python select_grasp.py request.json [--render out.png]
  request.json: {"part": "A5917204100", "T_base_part": 4x4 (m) or "T_base_cad_mm": 4x4 (mm),
                 "support": {"point": [x, y, z], "normal": [0, 0, 1]},
                 "boxes": [{"center": [..], "size": [..], "quat_xyzw": [..]}], "q_now": {joint: angle}}
"""
import json
import os
import time

import numpy as np
import pybullet as p

import gripper as G
import ik
import mesh_io as M

ROOT = os.path.dirname(os.path.abspath(__file__))
POSE_TOL = np.radians(15)          # resting-pose match tolerance
MAX_DOWN_DEG = 60                  # approach at most this far from straight down
SUPPORT_CLEAR = 0.002              # m, body box above the support plane
PREGRASP = 0.10                    # m, = the approach distance checked offline (collide.APPROACH_DIST)
APPROACH_STEPS = 4                 # IK checks between pre-grasp and grasp
LIMIT_MARGIN_DEG = 5.0
PRE_RESTARTS, GRASP_RESTARTS = 4, 2
BODY = True                        # IK may move the body joints (lift, waist) too
HEADING, HEADING_TOL = np.radians(120), np.radians(75)   # preferred tool x heading (base frame), see step 4
W_TILT, W_TURN, W_LIMIT = 0.10, 0.10, 0.10   # ranking costs (quality is 0..1)
# step 5 (guided local search)
N_SEEDS, SEED_SPACING = 30, 0.010  # best database grasps of the resting pose, at least this far apart
TURN_STEP_DEG = 10                 # turns about the approach axis (only those in the preferred heading band)
N_SHIFTS, SHIFT = 2, 0.005         # random shifts in the cup plane per seed (+ the unshifted one), m
MAX_CHECKED = 60                   # variants that pass collision + hold + robustness before the IK


class Selector:
    def __init__(self, part):
        self.part = part
        self.g = G.load()
        d = np.load(os.path.join(ROOT, "db", part + ".npz"))
        self.db = {k: d[k] for k in d.files}
        self.T_flange_tool_inv = G.inv(self.g["T_flange_tool"])
        self.kin = ik.G2Kin()
        c = self.kin.cid
        self.kin.ignore = {l for n, l in self.kin.links.items() if "_l_" in n}
        self.cups = [self.kin.links[n] for n in self.g["cup_links"]]
        obj = M.collision_obj(part)
        self.part_body = p.createMultiBody(0, p.createCollisionShape(p.GEOM_MESH, fileName=obj, physicsClientId=c),
                                           p.createVisualShape(p.GEOM_MESH, fileName=obj, rgbaColor=[0.9, 0.55, 0.2, 1],
                                                               physicsClientId=c),
                                           basePosition=[0, 0, -5], physicsClientId=c)
        self.boxes = []
        # head outline points in the tool frame (for the support-plane filter): EVS08 body box corners and the
        # side-hardware ring as a cylinder (12 points around at its bottom and top)
        hx, hy = self.g["body_box"]["half_xy"]
        box = [[sx * hx, sy * hy, z, 1] for sx in (-1, 1) for sy in (-1, 1) for z in self.g["body_box"]["z"]]
        r, (z0, z1) = self.g["ring"]["radius"], self.g["ring"]["z"]
        ring = [[r * np.cos(a), r * np.sin(a), z, 1] for a in np.arange(12) * np.pi / 6 for z in (z0, z1)]
        self.corners = np.array(box + ring)

    def close(self):
        """Disconnect this selector's PyBullet worlds (they are not freed by garbage collection)."""
        for cid in [self.kin.cid] + ([self._seal.cid, self._head.cid] if hasattr(self, "_seal") else []):
            p.disconnect(physicsClientId=cid)

    # ------------------------------------------------------------------ scene
    def set_scene(self, T_base_part, boxes=()):
        c = self.kin.cid
        pos, quat = G.pos_quat(T_base_part)
        p.resetBasePositionAndOrientation(self.part_body, pos.tolist(), quat.tolist(), physicsClientId=c)
        for b in self.boxes:
            p.removeBody(b, physicsClientId=c)
        self.boxes = []
        for b in boxes:
            half = (np.asarray(b["size"]) / 2).tolist()
            self.boxes.append(p.createMultiBody(
                0, p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=c),
                p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.55, 0.6, 0.65, 1], physicsClientId=c),
                basePosition=list(b["center"]), baseOrientation=list(b.get("quat_xyzw", [0, 0, 0, 1])),
                physicsClientId=c))
        self.kin.obstacles = [self.part_body] + self.boxes

    # ------------------------------------------------------------------ steps 2-3
    def resting_pose(self, T_base_part, normal):
        """Index of the stable pose the part is in (its down direction along -normal), or None."""
        best, best_ang = None, POSE_TOL
        for k, T_rest in enumerate(self.db["T_rest"]):
            down = T_base_part[:3, :3] @ (T_rest[:3, :3].T @ [0, 0, -1])
            ang = np.arccos(np.clip(down @ -normal, -1, 1))
            if ang < best_ang:
                best, best_ang = k, ang
        return best

    def candidates_for(self, T, support):
        """Fast filters (approach tilt, body box above the support) on base-frame tool poses T (n, 4, 4).
        Returns (indices passing, T, tilt deg, None)."""
        n = np.asarray(support["normal"], float)
        n /= np.linalg.norm(n)
        tilt = np.degrees(np.arccos(np.clip(T[:, :3, 2] @ -n, -1, 1)))
        ok = tilt <= MAX_DOWN_DEG
        h0 = np.asarray(support["point"], float) @ n
        for back in (0.0, PREGRASP):
            pts = np.einsum("nij,cj->nci", T @ G.pose([0, 0, -back]), self.corners)[..., :3]
            ok &= (pts @ n - h0).min(1) >= SUPPORT_CLEAR
        return np.flatnonzero(ok), T, tilt, None

    def candidates(self, T_base_part, support):
        """Steps 1-3: indices of database grasps passing the pose tag and fast filters, their base-frame tool
        poses and approach tilts (deg), plus the resting pose index."""
        n = np.asarray(support["normal"], float)
        T = T_base_part @ self.db["T"]
        k = self.resting_pose(T_base_part, n / np.linalg.norm(n))
        idx, _, tilt, _ = self.candidates_for(T, support)
        if k is not None:
            idx = idx[self.db["pose_ok"][idx, k]]
        return idx, T, tilt, k

    # ------------------------------------------------------------------ step 4
    def reach(self, T_tool, q_seed=None):
        """IK at the pre-grasp, the approach points and the grasp. Returns (ok, reason, q_pre, q_grasp, margin)."""
        kin = self.kin
        targets = [T_tool @ G.pose([0, 0, -PREGRASP * (1 - i / (APPROACH_STEPS + 1))]) for i in range(APPROACH_STEPS + 2)]
        q_pre = q = None
        margin = 180.0
        for i, Tt in enumerate(targets):
            last = i == len(targets) - 1
            kin.obstacle_ok = {self.part_body: self.cups} if last else {}   # cups touch the part only at the grasp
            pos, quat = G.pos_quat(Tt @ self.T_flange_tool_inv)
            r = kin.solve("r", pos, quat=quat, frame="flange", body=BODY,
                          restarts=PRE_RESTARTS if i == 0 else GRASP_RESTARTS,
                          seed=q if q is not None else q_seed)
            if not r["ok"] or r["near_limit"]:
                where = "pre-grasp" if i == 0 else "grasp" if last else f"approach {i}/{APPROACH_STEPS}"
                why = "joint near limit" if r["ok"] else r["reason"]
                if r["collisions"]:
                    why += f" ({r['collisions'][0][0]} - {r['collisions'][0][1]})"
                return False, f"{where}: {why}", None, None, None
            q = r["q"]
            margin = min(margin, r["limit_margin_deg"])
            if i == 0:
                q_pre = q
        return True, "reachable", q_pre, q, margin

    # ------------------------------------------------------------------ steps 4 + 6 (shared by both passes)
    def try_grasps(self, T, quality, tilt, ids, source, q_now, k, deadline, out, info):
        """IK-check the given base-frame tool poses in order; append the reachable ones (with ranking costs) to out."""
        for j in range(len(T)):
            if len(out) >= k or time.time() > deadline:
                break
            info["tried"] += 1
            ok, why, q_pre, q_grasp, margin = self.reach(T[j], q_now)
            if not ok:
                key = why.split(" (")[0]
                info["reasons"][key] = info["reasons"].get(key, 0) + 1
                continue
            turn = abs(q_grasp["idx67_arm_r_joint7"] - q_now["idx67_arm_r_joint7"]) / np.pi if q_now else 0.0
            cost = dict(tilt=W_TILT * tilt[j] / MAX_DOWN_DEG, turn=W_TURN * turn, limit=W_LIMIT * max(0.0, 1 - margin / 20.0))
            out.append(dict(id=ids[j], source=source, score=float(quality[j] - sum(cost.values())),
                            quality=float(quality[j]), costs=cost, tilt_deg=float(tilt[j]),
                            T_tool=T[j], T_flange=T[j] @ self.T_flange_tool_inv,
                            T_pre_flange=T[j] @ G.pose([0, 0, -PREGRASP]) @ self.T_flange_tool_inv,
                            q_pre=q_pre, q_grasp=q_grasp, limit_margin_deg=float(margin)))

    def preferred(self, T):
        """True where the tool x axis heads within HEADING_TOL of HEADING (base frame)."""
        heading = np.arctan2(T[:, 1, 0], T[:, 0, 0])
        return np.abs((heading - HEADING + np.pi) % (2 * np.pi) - np.pi) <= HEADING_TOL

    # ------------------------------------------------------------------ steps 1-6
    def select(self, T_base_part, support, boxes=(), q_now=None, k=5, budget=2.0, fallback=True,
               fallback_budget=3.0, use_database=True):
        """Ranked reachable grasps (best first) and an info dict (counts, reasons, time).
        If no database grasp is reachable, the guided local search (step 5) runs with its own budget.
        use_database=False skips the database pass (to test the local search on its own)."""
        t0 = time.time()
        if q_now is not None:                             # joints not given are taken from the home pose
            q_now = {**self.kin.home, **q_now}
        self.set_scene(T_base_part, boxes)
        idx, T, tilt, pose = self.candidates(T_base_part, support)
        info = dict(resting_pose=None if pose is None else int(pose), database=len(T), after_filters=len(idx),
                    tried=0, reasons={})
        out = []
        order = idx[np.lexsort((-self.db["quality"][idx], ~self.preferred(T[idx])))]  # preferred first, then quality
        order = order if use_database else order[:0]
        self.try_grasps(T[order], self.db["quality"][order], tilt[order], [int(i) for i in order], "database",
                        q_now, k, t0 + budget, out, info)
        if not out and fallback:
            info["local_search"] = self.local_search(T_base_part, support, pose, q_now, k, time.time() + fallback_budget, out)
        out.sort(key=lambda r: -r["score"])
        info["seconds"] = round(time.time() - t0, 3)
        return out, info

    # ------------------------------------------------------------------ step 5
    def checkers(self):
        """Seal checker and head collider on the part mesh (part frame), loaded on first use."""
        if not hasattr(self, "_seal"):
            import collide as C
            import seal as S
            m = M.load_part(self.part)
            self.mass, self.com = M.properties(m)
            self._seal = S.SealChecker(m, self.g)
            self._head = C.HeadCollider(M.collision_obj(self.part, m), self.g)
        return self._seal, self._head

    def local_search(self, T_base_part, support, pose, q_now, k, deadline, out):
        """Step 5: variants of the best database grasps for this resting pose (turns about the approach axis that
        land in the preferred wrist heading, small shifts in the cup plane), checked online: seal rule, hold
        margin, head collision with the part and the support plane, seal robustness; then the usual IK.
        Appends reachable grasps to out; returns an info dict."""
        import build_grasp_db as B
        import seal as S
        sc, hc = self.checkers()
        info = dict(seeds=0, variants=0, sealed=0, clear=0, tried=0, reasons={})
        # seeds: best database grasps tagged for this pose, at least SEED_SPACING apart
        tagged = np.flatnonzero(self.db["pose_ok"][:, pose]) if pose is not None else np.arange(len(self.db["T"]))
        seeds = []
        for i in tagged[np.argsort(-self.db["quality"][tagged])]:
            if all(np.linalg.norm(self.db["T"][i][:3, 3] - self.db["T"][j][:3, 3]) >= SEED_SPACING for j in seeds):
                seeds.append(i)
            if len(seeds) >= N_SEEDS:
                break
        info["seeds"] = len(seeds)
        # variants in the part frame: shift in the cup plane, then turn about the approach axis
        rng = np.random.default_rng(0)
        turns = np.radians(np.arange(0, 360, TURN_STEP_DEG))
        V, origin = [], []
        for i in seeds:
            for shift in [(0.0, 0.0)] + [tuple(rng.uniform(-SHIFT, SHIFT, 2)) for _ in range(N_SHIFTS)]:
                Ts = self.db["T"][i] @ G.pose([shift[0], shift[1], 0])
                cand = np.array([Ts @ G.rot_z(a) for a in turns])
                cand = cand[self.preferred(T_base_part @ cand)]          # only turns the wrist can reach
                V += list(cand)
                origin += [int(i)] * len(cand)
        V, origin = np.array(V).reshape(-1, 4, 4), np.array(origin, int)
        info["variants"] = len(V)
        # seal rule (online, exact part mesh) and the fast base-frame filters
        r = sc.check(V)
        idx, Tb, tilt, _ = self.candidates_for(T_base_part @ V, support)
        keep = np.intersect1d(np.flatnonzero(r["valid"]), idx)
        info["sealed"] = int(r["valid"].sum())
        # head collision against the part and the support plane, best seed quality first
        n = np.asarray(support["normal"], float) / np.linalg.norm(support["normal"])
        hc.set_support(G.inv(T_base_part) @ G.tool_from_point(support["point"], n))
        keep = keep[np.argsort(-self.db["quality"][origin[keep]], kind="stable")]
        good, qual = [], []
        for j in keep:
            if time.time() > deadline or len(good) >= MAX_CHECKED:
                break
            if not hc.check(V[j])[0]:
                continue
            hold = S.hold_margin(V[j], r["sealed"][j], self.com, self.mass, self.g)
            robust = np.mean(sc.check([V[j] @ D for D in B.perturbations(np.random.default_rng(j))])["valid"])
            if hold * robust > 0:
                good.append(j)
                qual.append(hold * robust)
        hc.set_support(None)
        info["clear"] = len(good)
        good, qual = np.array(good, int), np.array(qual)
        order = np.argsort(-qual)
        self.try_grasps(Tb[good[order]], qual[order], tilt[good[order]],
                        [f"search from {origin[j]}" for j in good[order]], "local search", q_now, k, deadline, out, info)
        return info

    # ------------------------------------------------------------------ dry-run picture
    def render(self, result, path, title=""):
        """Picture of the robot in the chosen grasp posture with the part and the scene (two views)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        kin, c = self.kin, self.kin.cid
        kin.set_q(result["q_grasp"])
        target = result["T_tool"][:3, 3]
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
        for ax, (eye, name) in zip(axes, ((target + [0.9, -0.7, 0.55], "overview"), (target + [0.35, -0.30, 0.20], "close-up"))):
            V = p.computeViewMatrix(eye.tolist(), target.tolist(), [0, 0, 1])
            P = p.computeProjectionMatrixFOV(45, 1.2, 0.01, 5.0)
            img = p.getCameraImage(600, 500, V, P, renderer=p.ER_TINY_RENDERER, physicsClientId=c)[2]
            ax.imshow(np.reshape(img, (500, 600, 4))[:, :, :3])
            ax.set_title(name, loc="left", fontsize=10)
            ax.axis("off")
        fig.suptitle(title or f"{self.part}: grasp {result['id']} (quality {result['quality']:.2f}, "
                     f"score {result['score']:.2f}, tilt {result['tilt_deg']:.0f} deg)", x=0.02, ha="left", fontsize=11)
        fig.savefig(path, dpi=80, bbox_inches="tight")
        plt.close(fig)
        return path


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("request")
    ap.add_argument("--render")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    req = json.load(open(args.request))
    if "T_base_part" in req:                              # metres
        T = np.array(req["T_base_part"], float)
    else:                                                 # the pose-estimation service's T_base_cad_mm
        T = np.array(req["T_base_cad_mm"], float)
        T[:3, 3] /= 1000.0
    sel = Selector(req["part"])
    out, info = sel.select(T, req["support"], req.get("boxes", ()), req.get("q_now"), k=args.k)
    print(json.dumps(info))
    for r in out:
        pos, quat = G.pos_quat(r["T_flange"])
        print(json.dumps(dict(id=r["id"], score=round(r["score"], 3), quality=round(r["quality"], 3),
                              flange_pos=pos.round(4).tolist(), flange_quat_xyzw=quat.round(4).tolist(),
                              q_grasp={k: round(v, 4) for k, v in r["q_grasp"].items()
                                       if k in ik.ARM_JOINTS["r"] + ik.BODY_JOINTS})))   # arm + body joints IK moved
    if args.render and out:
        print("wrote", sel.render(out[0], args.render))


if __name__ == "__main__":
    main()
