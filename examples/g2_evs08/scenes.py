"""Random conveyor scenes in the robot base frame, mirroring g2_pybullet_sim/conveyor.py (real cell, 2026-10-22;
numbers copied). The robot stands at the near end of the conveyor, facing along the belt. The part rests on the
belt in one of its stable poses (drawn by probability).

Conveyor frame C: origin on the belt surface at the near end, on the centreline; x along the belt (away from the
robot), y across (left +), z up. Robot base at (-dist, side) in C, heading = small jitter.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import g2_parts as GP                     # noqa: E402
from suction_grasps import poses as G     # noqa: E402

# measured: belt top, width, side frame height, robot placement; placeholders: length, bed depth, frame thickness
BELT_TOP, BELT_WIDTH, BELT_LEN, BED_DEPTH = 0.783, 0.88, 3.00, 0.12
RAIL_HEIGHT, RAIL_THICK = 0.012, 0.03
ROBOT_DIST, ROBOT_SIDE, ROBOT_JITTER, ROBOT_YAW_JITTER = 0.557, 0.09, 0.02, np.radians(2)
PICK_ZONE, RAIL_MARGIN = (0.10, 0.60), 0.01

_FOOT = {}


def footprints(name, T_rest):
    """Per stable pose: (xy centre offset, footprint xy points) of the resting part (cached)."""
    if name not in _FOOT:
        v = GP.load(name).vertices
        _FOOT[name] = []
        for T in T_rest:
            w = (T[:3, :3] @ v.T).T + T[:3, 3]
            xy = w[:, :2]
            ctr = (xy.min(0) + xy.max(0)) / 2
            _FOOT[name].append((ctr, xy[::20] - ctr))
    return _FOOT[name]


def rot_z(a):
    return G.rot_z(a)


def sample(rng, name, T_rest, prob):
    """One scene: dict(T_base_part, support, boxes, pose_index, params). Retries if the part doesn't fit."""
    j = lambda a: rng.uniform(-a, a)
    while True:
        k = rng.choice(len(prob), p=np.asarray(prob) / np.sum(prob))
        dist, side, heading = ROBOT_DIST + j(ROBOT_JITTER), ROBOT_SIDE + j(ROBOT_JITTER), j(ROBOT_YAW_JITTER)
        yaw, across, along = rng.uniform(-np.pi, np.pi), rng.uniform(0, 1), rng.uniform(0, 1)
        ctr, xy = footprints(name, T_rest)[k]
        xy = xy @ rot_z(yaw)[:2, :2].T
        lo, hi = -BELT_WIDTH / 2 + RAIL_MARGIN - xy[:, 1].min(), BELT_WIDTH / 2 - RAIL_MARGIN - xy[:, 1].max()
        if hi >= lo:
            break
    x = max(PICK_ZONE[0] + along * (PICK_ZONE[1] - PICK_ZONE[0]), RAIL_MARGIN - xy[:, 0].min())
    # C -> base: p_base = Rz(-heading) (p_C - b), b = robot base in C
    b = np.array([-dist, side, -BELT_TOP])
    T_base_C = rot_z(-heading) @ G.pose(-b)
    T_C_part = G.pose([x, lo + across * (hi - lo), 0]) @ rot_z(yaw) @ G.pose([-ctr[0], -ctr[1], 0]) @ T_rest[k]
    rail_h = BED_DEPTH + RAIL_HEIGHT
    boxes_C = [((BELT_LEN / 2, 0, -BED_DEPTH / 2), (BELT_LEN, BELT_WIDTH, BED_DEPTH))]
    boxes_C += [((BELT_LEN / 2, sy * (BELT_WIDTH / 2 + RAIL_THICK / 2), RAIL_HEIGHT - rail_h / 2), (BELT_LEN, RAIL_THICK, rail_h))
                for sy in (-1, 1)]
    quat = G.pos_quat(T_base_C)[1].tolist()
    boxes = [dict(center=(T_base_C @ np.r_[c, 1])[:3].tolist(), size=list(s_), quat_xyzw=quat) for c, s_ in boxes_C]
    return dict(T_base_part=T_base_C @ T_C_part, support=dict(point=T_base_C[:3, 3].tolist(), normal=[0, 0, 1]),
                boxes=boxes, pose_index=int(k),
                params=dict(dist=dist, side=side, heading=heading, yaw=yaw, across=across, along=along))


def noisy(T, centre, rng, pos=0.002, ang=np.radians(2)):
    """T (base <- part) with a pose-estimation-like error: shifted uniform +-pos per base axis and turned up to
    ang about a random axis through the part's centre (`centre` in the part frame). The CAD origin is 1-2 m away
    from the part, so the turn must not be about the part-frame origin."""
    import trimesh
    axis = rng.normal(size=3)
    c = (T @ np.r_[centre, 1])[:3]                                  # part centre in the base frame
    D = trimesh.transformations.rotation_matrix(rng.uniform(0, ang), axis / np.linalg.norm(axis), point=c)
    D[:3, 3] += rng.uniform(-pos, pos, 3)
    return D @ T
