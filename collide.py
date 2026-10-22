"""Gripper-head collision check (PyBullet DIRECT): head vs part at the grasp and along the straight approach.

The robot stands at home and never moves; the PART is placed so that the requested tool pose (part frame)
lands on the robot's tool frame. Only the links rigidly fixed to the tool (gripper.json head_links) are checked:
the rest of the arm depends on the IK solution and is checked at execution time.
The part is a static compound of convex pieces (mesh_io.collision_obj), body frame = CAD frame.
At the grasp itself the cups may touch the part (their lips are what seals); during the approach nothing may.
Optionally a support plane (the belt/table the part rests on) moves with the part; nothing of the head,
cups included, may touch it at the grasp or on the approach.
"""
import numpy as np
import pybullet as p

import gripper as G
import robot

APPROACH_DIST = 0.10       # m, straight approach along the tool axis before the grasp
APPROACH_STEP = 0.010      # m, checked every this far


class HeadCollider:
    def __init__(self, collision_obj, g):
        self.cid = p.connect(p.DIRECT)
        self.robot, links, joints = robot.load_robot(self.cid)
        robot.set_joints(self.robot, joints, robot.home_pose(), self.cid)
        self.cups = [links[n] for n in g["cup_links"]]
        self.head = [links[n] for n in g["head_links"]]
        fp, fq = p.getLinkState(self.robot, links[g["flange_link"]], computeForwardKinematics=True,
                                physicsClientId=self.cid)[4:6]
        self.T_world_tool = G.pose(fp, fq) @ g["T_flange_tool"]     # the robot's tool frame at home
        shape = p.createCollisionShape(p.GEOM_MESH, fileName=collision_obj, physicsClientId=self.cid)
        self.part = p.createMultiBody(0, shape, basePosition=[0, 0, -5], physicsClientId=self.cid)
        # support plane: a 4 x 4 m box, 20 mm thick, top face = the plane; parked far away until set
        box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2, 2, 0.01], physicsClientId=self.cid)
        self.plane = p.createMultiBody(0, box, basePosition=[0, 0, -50], physicsClientId=self.cid)
        self.T_part_plane = None

    def set_support(self, T_part_plane):
        """Support plane as a frame in the part frame (origin on the plane, +z = up, away from the support),
        or None for no plane."""
        self.T_part_plane = T_part_plane
        if T_part_plane is None:
            p.resetBasePositionAndOrientation(self.plane, [0, 0, -50], [0, 0, 0, 1], physicsClientId=self.cid)

    def place(self, T_part_tool, back=0.0):
        """Put the part so the robot's tool sits at T_part_tool, pulled back `back` m along -z (approach)."""
        T_tool_part = G.inv(T_part_tool @ G.pose([0, 0, -back]))
        T_world_part = self.T_world_tool @ T_tool_part
        pos, quat = G.pos_quat(T_world_part)
        p.resetBasePositionAndOrientation(self.part, pos.tolist(), quat.tolist(), physicsClientId=self.cid)
        if self.T_part_plane is not None:                  # box centre 10 mm below the plane
            pos, quat = G.pos_quat(T_world_part @ self.T_part_plane @ G.pose([0, 0, -0.01]))
            p.resetBasePositionAndOrientation(self.plane, pos.tolist(), quat.tolist(), physicsClientId=self.cid)

    def hits(self, links, body=None, margin=0.0):
        """True if any of these robot links is closer than margin to the body (default: the part)."""
        body = self.part if body is None else body
        return any(p.getClosestPoints(self.robot, body, margin, linkIndexA=l, physicsClientId=self.cid)
                   for l in links)

    def check(self, T_part_tool, approach=APPROACH_DIST, step=APPROACH_STEP):
        """(clear: bool, reason). Grasp pose first (head only), then the approach back to front (head + cups)."""
        plane = self.T_part_plane is not None
        self.place(T_part_tool)
        if self.hits(self.head):
            return False, "head hits the part at the grasp"
        if plane and self.hits(self.head + self.cups, self.plane):
            return False, "head hits the support at the grasp"
        for back in np.arange(approach, step / 2, -step):
            self.place(T_part_tool, back)
            if self.hits(self.head + self.cups) or (plane and self.hits(self.head + self.cups, self.plane)):
                return False, f"approach blocked {back * 1000:.0f} mm before the grasp"
        return True, "clear"

    def tool_in_part(self):
        """Current tool pose in the part frame (for tests: place() round trip)."""
        pos, quat = p.getBasePositionAndOrientation(self.part, physicsClientId=self.cid)
        return G.inv(G.pose(pos, quat)) @ self.T_world_tool
