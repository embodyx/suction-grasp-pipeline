"""Inverse kinematics + reachability/collision checking for the G2 with the EVS08 suction head (robot.py).
Copied from g2_pybullet_sim/ik.py (2026-10-22). Changes: robot loading (robot.py instead of g2sim); solve() takes a
`seed` posture (tried first); `ignore` = link indices left out of the floor/obstacle checks (e.g. the idle left arm).

All poses are in the robot's base_link frame (the frame the robot reports its end-effector poses in).
The "approach axis" is the direction the suction cups point = +Z of arm_*_end_link (the flange).
This pipeline targets the flange (frame="flange"); the tool point is 128 mm further along +Z (gripper_evs08.json).

Example:
    kin = G2Kin()
    r = kin.solve("r", pos=[0.55, -0.2, 0.90], axis=[0, 0, -1], frame="flange", body=True)   # cups straight down
    print(r["ok"], r["pos_err_mm"], r["collisions"])
"""
import re

import numpy as np
import pybullet as p
import pybullet_data

from robot import SRDF, home_pose, load_robot
POS_TOL = 0.001                 # 1 mm
ANG_TOL = np.radians(0.5)       # 0.5 deg
FLOOR_OK = ("base_link", "chassis")   # links allowed to touch the floor (name prefixes)
LIMIT_MARGIN = np.radians(5)    # keep searching if a solution is closer than this to a joint limit
BODY_WEIGHT = 4.0               # body (lift/waist) joints cost this much more than arm joints in IK
# Per-joint IK cost overrides: turning the waist (joint5, +-174 deg) is cheap so the torso turns toward
# targets on the robot's left instead of the elbow folding into the torso (as the real robot does);
# joint4 (+-25 deg) is expensive so it doesn't saturate. Tuned (in the sim) on 60 random containers: 46/60 -> 60/60.
JOINT_WEIGHT = {"idx05_body_joint5": 0.5, "idx04_body_joint4": 8.0}
LIMIT_SOFTNESS = 0.05           # rad; a joint this close to a limit costs 2x, 10x closer costs 11x

ARM_JOINTS = {s: [f"idx{a}{i}_arm_{s}_joint{i}" for i in range(1, 8)] for s, a in (("l", 2), ("r", 6))}
BODY_JOINTS = [f"idx0{i}_body_joint{i}" for i in range(1, 6)]
TCP = {"suction": "gripper_{s}_suction_center", "flange": "arm_{s}_end_link"}


class G2Kin:
    def __init__(self, gui=False):
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.cid)
        self.robot, self.links, _ = load_robot(self.cid)
        self.links["base_link"] = -1
        self.names = {v: k for k, v in self.links.items()}

        # Movable joints (PyBullet's Jacobian has one column per movable joint, in index order)
        self.jidx, self.lim, self.movable = {}, {}, []
        for j in range(p.getNumJoints(self.robot, physicsClientId=self.cid)):
            info = p.getJointInfo(self.robot, j, physicsClientId=self.cid)
            self.jidx[info[1].decode()] = j
            if info[2] != p.JOINT_FIXED:
                self.movable.append(j)
                self.lim[info[1].decode()] = (info[8], info[9])

        self.home = home_pose()
        self.set_q(self.home)
        self._build_collision_pairs()
        self.obstacles = []      # extra static bodies (e.g. a conveyor) the robot must not touch; checked by collisions()
        self.obstacle_ok = {}    # {obstacle body: link indices allowed to touch it} (e.g. suction cups on the part)
        self.ignore = set()      # link indices not checked against the floor and obstacles

    # ---------------------------------------------------------------- state / FK
    def set_q(self, q):
        for n, v in q.items():
            p.resetJointState(self.robot, self.jidx[n], float(v), physicsClientId=self.cid)

    def get_q(self, names):
        return {n: p.getJointState(self.robot, self.jidx[n], physicsClientId=self.cid)[0] for n in names}

    def fk(self, link):
        """World (= base_link) pose of a link frame: (pos, quat xyzw)."""
        s = p.getLinkState(self.robot, self.links[link], computeForwardKinematics=True, physicsClientId=self.cid)
        return np.array(s[4]), np.array(s[5])

    def approach_axis(self, side):
        _, q = self.fk(f"arm_{side}_end_link")
        return np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)[:, 2]

    # ---------------------------------------------------------------- collisions
    def _build_collision_pairs(self):
        """All link pairs to check = every pair with collision geometry, minus SRDF/gripper exclusions."""
        skip = set()
        for a, b in re.findall(r'<disable_collisions link1="([^"]+)" link2="([^"]+)"', open(SRDF).read()):
            skip.add(frozenset((a, b)))
        # Gripper parts may touch each other and their own wrist (not covered by the factory SRDF)
        for s in "lr":
            group = [n for n in self.links if n.startswith(f"gripper_{s}_")] + [f"arm_{s}_end_link", f"arm_{s}_link7"]
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    skip.add(frozenset((a, b)))
            # EVS08 side hardware (gripper_s_empty_link: hand camera + wrist adapter, measured boxes) is mounted at the
            # wrist; its adapter box overlaps the arm's own link5 in 326 of 900 frames recorded on the real robot (the
            # scan likely caught part of the wrist) while the real hardware does not hit it -> not checked against
            # the wrist links 5 and 6
            for w in (f"arm_{s}_link5", f"arm_{s}_link6"):
                skip.add(frozenset((f"gripper_{s}_empty_link", w)))
        self.skip = skip

        has_geom = [l for l in self.links.values()
                    if p.getCollisionShapeData(self.robot, l, physicsClientId=self.cid)]
        self.pairs = [(a, b) for i, a in enumerate(has_geom) for b in has_geom[i + 1:]
                      if frozenset((self.names[a], self.names[b])) not in skip]
        self.floor_links = [l for l in has_geom if not self.names[l].startswith(FLOOR_OK)]

    def collisions(self, margin=0.0):
        """Return (list of colliding link-name pairs, min clearance in m among checked pairs within 5 cm)."""
        hits, clearance = [], 0.05
        for a, b in self.pairs:
            for c in p.getClosestPoints(self.robot, self.robot, 0.05, a, b, physicsClientId=self.cid):
                clearance = min(clearance, c[8])
                if c[8] < margin:
                    hits.append((self.names[a], self.names[b], round(c[8] * 1000, 1)))
                    break
        for l in self.floor_links:
            if l in self.ignore:
                continue
            for c in p.getClosestPoints(self.robot, self.plane, 0.05, l, -1, physicsClientId=self.cid):
                clearance = min(clearance, c[8])
                if c[8] < margin:
                    hits.append((self.names[l], "floor", round(c[8] * 1000, 1)))
                    break
        for body in self.obstacles:
            allowed = self.obstacle_ok.get(body, ())
            for c in p.getClosestPoints(self.robot, body, 0.05, physicsClientId=self.cid):
                if c[3] in allowed or c[3] in self.ignore:
                    continue
                clearance = min(clearance, c[8])
                if c[8] < margin:
                    hits.append((self.names[c[3]], f"obstacle {body}", round(c[8] * 1000, 1)))
                    break
        return hits, clearance

    # ---------------------------------------------------------------- IK
    def _ik_from(self, side, link, pos, axis, quat, active, seed, pref, iters=300, damping=0.02, null_gain=0.1,
                 stop_early=True):
        """Damped least squares from one seed. Returns (q dict, pos err m, ang err rad)."""
        self.set_q(seed)
        cols = [self.movable.index(self.jidx[n]) for n in active]
        lo = np.array([self.lim[n][0] for n in active])
        hi = np.array([self.lim[n][1] for n in active])
        q = np.array([seed[n] for n in active])
        q_pref = np.array([pref[n] for n in active])     # preferred posture (home unless given)
        base_w = np.array([JOINT_WEIGHT.get(n, BODY_WEIGHT if "body" in n else 1.0) for n in active])
        for _ in range(iters):
            cur_p, cur_q = self.fk(link)
            e_p = pos - cur_p
            if quat is not None:
                # Rotation error as axis-angle of target * current^-1
                _, inv = p.invertTransform([0, 0, 0], cur_q)
                dq = p.getAxisAngleFromQuaternion(p.multiplyTransforms([0, 0, 0], quat, [0, 0, 0], inv)[1])
                ang = dq[1] if dq[1] <= np.pi else dq[1] - 2 * np.pi
                e_w = np.array(dq[0]) * ang
            else:
                a = self.approach_axis(side)
                e_w = np.cross(a, axis)                      # rotate a toward axis
                ang = np.arccos(np.clip(np.dot(a, axis), -1, 1))
            if stop_early and np.linalg.norm(e_p) < POS_TOL and abs(ang) < ANG_TOL:
                break      # (tracking keeps iterating so the null-space pull toward `pref` still acts)
            allq = [p.getJointState(self.robot, j, physicsClientId=self.cid)[0] for j in self.movable]
            jl, ja = p.calculateJacobian(self.robot, self.links[link], [0, 0, 0], allq,
                                         [0.0] * len(allq), [0.0] * len(allq), physicsClientId=self.cid)
            J = np.vstack([np.array(jl)[:, cols], np.array(ja)[:, cols]])
            e = np.concatenate([e_p, e_w])
            if quat is None:
                # Axis-only target: ignore rotation about the approach axis (free spin of the cup square)
                P = np.eye(3) - np.outer(a, a)
                J[3:] = P @ J[3:]
            # Weighted DLS: expensive joints (body, or any joint near a limit) move less, so the arm
            # does most of the work and no joint saturates while others still have range
            d_lim = np.minimum(q - lo, hi - q)
            w = base_w * (1 + LIMIT_SOFTNESS / np.maximum(d_lim, 1e-3))
            S = np.diag(1 / np.sqrt(w))
            Jw = J @ S
            Jpinv = S @ Jw.T @ np.linalg.inv(Jw @ Jw.T + damping ** 2 * np.eye(6))
            step = Jpinv @ e
            # Null-space pull toward the seed posture: uses the arm's redundancy to stay close to
            # the robot's usual configuration (and away from joint limits) without affecting the target
            step += (np.eye(len(q)) - Jpinv @ J) @ (null_gain * (q_pref - q))
            step *= min(1.0, 0.2 / (np.abs(step).max() + 1e-12))   # max 0.2 rad per iteration
            q = np.clip(q + step, lo, hi)
            self.set_q(dict(zip(active, q)))
        cur_p, _ = self.fk(link)
        return dict(zip(active, q)), float(np.linalg.norm(pos - cur_p)), float(abs(ang))

    def track(self, side, pos, quat, active, iters=5, frame="suction"):
        """Fast incremental IK for control loops: a few iterations from the CURRENT joints, no restarts
        or collision checks. Leaves the robot at the result and returns {joint: angle} for `active`."""
        current = self.get_q(list(self.home))
        q, _, _ = self._ik_from(side, TCP[frame].format(s=side), np.asarray(pos, float), None,
                                np.asarray(quat, float), active, current, self.home, iters=iters, stop_early=False)
        return q

    def solve(self, side, pos, axis=None, quat=None, frame="suction", body=False,
              restarts=30, margin=0.0, rng_seed=0, seed=None):
        """IK for one arm. Give `axis` (cup pointing direction) or `quat` (full orientation of the TCP frame).

        Returns a dict: ok, reason, q (all 22 joints), pos_err_mm, ang_err_deg, collisions,
        clearance_mm, limit_margin_deg (closest any active joint gets to a limit).
        """
        link = TCP[frame].format(s=side)
        pos = np.asarray(pos, float)
        axis = None if axis is None else np.asarray(axis, float) / np.linalg.norm(axis)
        active = ARM_JOINTS[side] + (BODY_JOINTS if body else [])
        rng = np.random.default_rng(rng_seed)

        best = None
        for k in range(restarts + 1):
            start = dict(self.home)
            if k == 0 and seed is not None:   # given posture first (e.g. the pre-grasp solution)
                start.update({n: seed[n] for n in active})
            elif k > 0:   # random restarts within joint limits
                start.update({n: rng.uniform(*self.lim[n]) for n in active})
            q, pe, ae = self._ik_from(side, link, pos, axis, quat, active, start, self.home)
            converged = pe < POS_TOL and ae < ANG_TOL
            full = dict(self.home); full.update(q)
            self.set_q(full)
            hits, clear = self.collisions(margin) if converged else ([], None)
            margin_deg = float(np.degrees(min(min(q[n] - self.lim[n][0], self.lim[n][1] - q[n]) for n in active)))
            cand = dict(ok=converged and not hits, q=full, pos_err_mm=pe * 1000, ang_err_deg=np.degrees(ae),
                        collisions=hits, clearance_mm=None if clear is None else clear * 1000,
                        limit_margin_deg=margin_deg, converged=converged)
            # Prefer: collision-free solution > converged-but-colliding > closest miss; then farthest from limits
            score = (cand["ok"], converged, -pe, margin_deg)
            if best is None or score > best[0]:
                best = (score, cand)
            if cand["ok"] and margin_deg >= np.degrees(LIMIT_MARGIN):
                break      # first good solution wins (home seed is tried first -> natural posture)
        r = best[1]
        r["near_limit"] = r["ok"] and r["limit_margin_deg"] < np.degrees(LIMIT_MARGIN)
        r["reason"] = ("reachable" + (" (joint near limit)" if r["near_limit"] else "") if r["ok"] else
                       "in collision" if r["converged"] else
                       f"out of reach (closest {r['pos_err_mm']:.0f} mm / {r['ang_err_deg']:.1f} deg)")
        self.set_q(r["q"])
        return r
