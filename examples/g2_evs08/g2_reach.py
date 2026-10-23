"""Reachability plug-in for suction_grasps.select on the Agibot G2 (right arm, EVS08 head), using PyBullet IK.

  reach = G2Reach(mesh, gripper_json, name)      # robot model from assets/robot (copy_robot.py)
  reach.set_scene(T_base_part, boxes)            # part + obstacle boxes as IK obstacles (call per scene)
  sel.select(T_base_part, support, boxes, reach=reach, order=reach.order)

reach(T_tool) checks collision-free IK (right arm + body lift/waist joints) at the pre-grasp (PREGRASP back), at
APPROACH_STEPS points on the way and at the grasp (each seeded from the previous), joints at least 5 deg
inside their limits (ik.LIMIT_MARGIN); the part (the cups may touch it at the grasp), the obstacle boxes and the floor are obstacles,
the idle left arm is not checked. It returns ok, reason, cost (joint-limit closeness) and the joint solutions.
order(T_tools): grasps whose tool x axis heads within HEADING_TOL of HEADING (base frame) first - the wrist
(joint 7 +-88 deg, joint 6 +-58 deg) cannot reach most other turns (measured: heading 60..180 deg 78/100 reachable,
-90..0 deg 1/75).
"""
import json
import os
import sys

import numpy as np
import pybullet as p

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ik                                    # noqa: E402  (G2 IK, this folder)
from suction_grasps import poses as G        # noqa: E402
from suction_grasps.collide import APPROACH_DIST as PREGRASP   # noqa: E402  (the approach checked in phase 1)

APPROACH_STEPS = 4                 # IK checks between pre-grasp and grasp
PRE_RESTARTS, GRASP_RESTARTS = 4, 2
BODY = True                        # IK may move the body joints (lift, waist) too
HEADING, HEADING_TOL = np.radians(120), np.radians(75)
W_LIMIT = 0.10                     # ranking cost for joint-limit closeness (quality is 0..1)


PIECES = 600                       # convex pieces of the part for the IK obstacle check (~10 mm each)


def convex_pieces(mesh, name):
    """OBJ of PIECES convex pieces of the part (k-means on face centres, each piece = convex hull of its triangles),
    in the part frame; PyBullet loads a multi-object OBJ as a compound of convex shapes. Cached in build/."""
    key = f"{len(mesh.vertices)}_{len(mesh.faces)}_{abs(hash(tuple(mesh.bounds.round(6).ravel()))) % 10 ** 8}"
    path = os.path.join(HERE, "build", f"{name}_{key}_pieces.obj")      # a changed mesh gets a new file
    if os.path.exists(path):
        return path
    import trimesh
    from sklearn.cluster import MiniBatchKMeans
    label = MiniBatchKMeans(PIECES, random_state=0, n_init=3, batch_size=4096).fit_predict(mesh.triangles_center)
    lines, base = [], 1
    for k in range(PIECES):
        v = mesh.triangles[label == k].reshape(-1, 3)
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(lines) + "\n")
    return path


class G2Reach:
    def __init__(self, mesh, gripper_json, part_name="part"):
        spec = json.load(open(gripper_json))
        t = spec["T_flange_tool"]
        self.T_tool_flange = G.inv(G.pose(t["pos"], t["quat_xyzw"]))
        self.kin = ik.G2Kin()
        c = self.kin.cid
        self.kin.ignore = {l for n, l in self.kin.links.items() if "_l_" in n}       # idle left arm
        self.cups = [self.kin.links[f"gripper_r_cup{i}"] for i in range(1, 5)]
        # the part as a compound of small convex pieces (fast distance queries for the arm; the head vs the part
        # was already checked exactly in phase 1), parked until set_scene
        obj = convex_pieces(mesh, part_name)
        shape = p.createCollisionShape(p.GEOM_MESH, fileName=obj, physicsClientId=c)
        vis = p.createVisualShape(p.GEOM_MESH, fileName=obj, rgbaColor=[0.9, 0.55, 0.2, 1], physicsClientId=c)
        self.part = p.createMultiBody(0, shape, vis, basePosition=[0, 0, -5], physicsClientId=c)
        self.boxes = []
        self.q_seed = None

    def close(self):
        """Disconnect the PyBullet world (not freed by garbage collection)."""
        p.disconnect(physicsClientId=self.kin.cid)

    def set_scene(self, T_base_part, boxes=()):
        """Place the part and the obstacle boxes ({"center", "size", "quat_xyzw"}, base frame) as IK obstacles."""
        c = self.kin.cid
        pos, quat = G.pos_quat(T_base_part)
        p.resetBasePositionAndOrientation(self.part, pos.tolist(), quat.tolist(), physicsClientId=c)
        for b in self.boxes:
            p.removeBody(b, physicsClientId=c)
        self.boxes = []
        for b in boxes:
            half = (np.asarray(b["size"]) / 2).tolist()
            self.boxes.append(p.createMultiBody(
                0, p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=c),
                p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.55, 0.6, 0.65, 1], physicsClientId=c),
                basePosition=list(b["center"]), baseOrientation=list(b.get("quat_xyzw", [0, 0, 0, 1])), physicsClientId=c))
        self.kin.obstacles = [self.part] + self.boxes

    def order(self, T_tools):
        """0 for grasps in the preferred wrist heading band, 1 for the rest (tried later, not dropped)."""
        heading = np.arctan2(T_tools[:, 1, 0], T_tools[:, 0, 0])
        return (np.abs((heading - HEADING + np.pi) % (2 * np.pi) - np.pi) > HEADING_TOL).astype(float)

    def __call__(self, T_tool):
        kin = self.kin
        targets = [T_tool @ G.pose([0, 0, -PREGRASP * (1 - i / (APPROACH_STEPS + 1))]) for i in range(APPROACH_STEPS + 2)]
        q, q_pre, margin = None, None, 180.0
        for i, Tt in enumerate(targets):
            last = i == len(targets) - 1
            kin.obstacle_ok = {self.part: self.cups} if last else {}   # cups touch the part only at the grasp
            pos, quat = G.pos_quat(Tt @ self.T_tool_flange)
            r = kin.solve("r", pos, quat=quat, frame="flange", body=BODY, restarts=PRE_RESTARTS if i == 0 else GRASP_RESTARTS,
                          seed=q if q is not None else self.q_seed)
            if not r["ok"] or r["near_limit"]:
                where = "pre-grasp" if i == 0 else "grasp" if last else f"approach {i}/{APPROACH_STEPS}"
                why = "joint near limit" if r["ok"] else r["reason"]
                return dict(ok=False, reason=f"{where}: {why}")
            q = r["q"]
            margin = min(margin, r["limit_margin_deg"])
            q_pre = q if i == 0 else q_pre
        return dict(ok=True, reason="reachable", cost=W_LIMIT * max(0.0, 1 - margin / 20.0), q_pre=q_pre, q_grasp=q,
                    T_flange=T_tool @ self.T_tool_flange, limit_margin_deg=float(margin))

    def render(self, result, path, title=""):
        """Dry-run picture: the robot in the grasp's IK posture with the part and the scene (two views)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        c = self.kin.cid
        self.kin.set_q(result["q_grasp"])
        target = result["T_tool"][:3, 3]
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
        for ax, (eye, name) in zip(axes, ((target + [0.9, -0.7, 0.55], "overview"), (target + [0.35, -0.30, 0.20], "close-up"))):
            V = p.computeViewMatrix(eye.tolist(), target.tolist(), [0, 0, 1])
            P = p.computeProjectionMatrixFOV(45, 1.2, 0.01, 5.0)
            img = p.getCameraImage(600, 500, V, P, renderer=p.ER_TINY_RENDERER, physicsClientId=c)[2]
            ax.imshow(np.reshape(img, (500, 600, 4))[:, :, :3])
            ax.set_title(name, loc="left", fontsize=10)
            ax.axis("off")
        fig.suptitle(title or f"grasp {result['id']} (quality {result['quality']:.2f}, score {result['score']:.2f}, "
                     f"tilt {result['tilt_deg']:.0f} deg)", x=0.02, ha="left", fontsize=11)
        fig.savefig(path, dpi=80, bbox_inches="tight")
        plt.close(fig)
        return path
