"""Head collision check: the head shapes (gripper "head") and the cups vs the part, at the grasp and along the
straight approach - no robot model, no physics engine (python-fcl against the exact part mesh).

At the grasp itself the cups may touch the part (their lips are what seals); the rest of the head may not. During
the approach (APPROACH_DIST back to APPROACH_STEP before the grasp) nothing may touch the part. Optionally a
support plane (the belt/table the part rests on) and obstacle boxes are checked too: nothing of the head, cups
included, may touch them at the grasp or on the approach (the support with a SUPPORT_TOL contact tolerance).

The fcl collision objects are built once (meshes as BVH models) and only moved per check.
"""
import fcl
import numpy as np
import trimesh

from .poses import pose

APPROACH_DIST = 0.10       # m, straight approach along the tool axis before the grasp
APPROACH_STEP = 0.010      # m, checked every this far
PLANE_SIZE = 4.0           # m, side of the box that stands in for a support plane
SUPPORT_TOL = 0.001        # m, the head and cups may touch or reach this far into the support plane (contact
                           #   tolerance: a part modelled as one surface lies exactly in the plane, and so do the cup
                           #   lips on it)


def fcl_object(mesh, T=np.eye(4)):
    """fcl collision object for a trimesh mesh (exact triangles, BVH), placed at T."""
    bvh = fcl.BVHModel()
    bvh.beginModel(len(mesh.vertices), len(mesh.faces))
    bvh.addSubModel(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int32))
    bvh.endModel()
    return fcl.CollisionObject(bvh, fcl.Transform(T[:3, :3], T[:3, 3]))


class HeadCollider:
    def __init__(self, mesh, gripper):
        """mesh: the part in its own frame, or None to check the head against a scene only (phase 2: the poses
        are then given in the scene's frame)."""
        self.g = gripper
        self.part = [fcl_object(mesh)] if mesh is not None else []
        self.scene = []                        # support slab + obstacle boxes (fcl objects)
        self.head = [fcl_object(m) for m in gripper.head]
        self.cups = [fcl_object(m) for m in gripper.cup_shapes]
        self.request = fcl.CollisionRequest()

    def set_scene(self, T_plane=None, boxes=()):
        """Support plane as a frame (origin on the plane, +z = up, away from the support) and obstacle boxes
        [(T_box 4x4 centre pose, size xyz)], both in the frame of the poses given to check(); or nothing."""
        self.scene = []
        if T_plane is not None:                # a thick slab whose top face is SUPPORT_TOL below the plane
            self.scene.append(fcl_object(trimesh.creation.box(extents=[PLANE_SIZE, PLANE_SIZE, 0.02]),
                                         T_plane @ pose([0, 0, -0.01 - SUPPORT_TOL])))
        for T_box, size in boxes:
            self.scene.append(fcl_object(trimesh.creation.box(extents=size), T_box))

    def _hits(self, others, shapes, T):
        """True if any of the head objects `shapes`, moved to T, touches any of `others`."""
        if not others:
            return False
        tf = fcl.Transform(T[:3, :3], T[:3, 3])
        for s in shapes:
            s.setTransform(tf)
            for o in others:
                if fcl.collide(s, o, self.request, fcl.CollisionResult()):
                    return True
        return False

    def check(self, T_tool, approach=APPROACH_DIST, step=APPROACH_STEP):
        """(clear: bool, reason). Grasp pose first (head without the cups vs the part; everything vs the scene),
        then the approach from far to near (head + cups vs part and scene)."""
        if self._hits(self.part, self.head, T_tool):
            return False, "head hits the part at the grasp"
        if self._hits(self.scene, self.head + self.cups, T_tool):
            return False, "head hits the support/obstacles at the grasp"
        for back in np.arange(approach, step / 2, -step):
            T = T_tool @ pose([0, 0, -back])
            if self._hits(self.part + self.scene, self.head + self.cups, T):
                return False, f"approach blocked {back * 1000:.0f} mm before the grasp"
        return True, "clear"
