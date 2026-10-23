"""Unit tests for collide.py on synthetic shapes. Run: python tests/test_collide.py

Test gripper (tests/test_gripper.json), tool frame: body box x +-50, y +-40 mm, 30..100 mm behind the lips; side
box ("camera") x +-20, y 40..90 mm, 50..90 mm behind the lips; cups r 10 mm at +-25 mm, 30 mm long (lips reach
35 mm from the centre). With the approach straight down onto a part whose top is z = 0: tool x = part x,
tool y = part -y, height above the lips = part z.
"""
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import shapes as SH
from suction_grasps import poses as G
from suction_grasps.collide import HeadCollider
from suction_grasps.gripper import Gripper

g = Gripper.load(os.path.join(HERE, "test_gripper.json"))
DOWN = [0, 0, -1]
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def collider(boxes):
    return HeadCollider(SH.as_mesh(boxes), g)


def main():
    T = G.tool_from_point([0, 0, 0], DOWN)       # grasp on the floor top at the origin

    # 1. flat plate, straight approach: clear (cups touch only at the grasp)
    ok, why = collider(SH.plate()).check(T)
    check("flat plate: clear", ok, why)

    # 2. channel 76 mm wide (cups reach +-35 mm): 20 mm walls (below the body at 30 mm) clear; 40 mm walls hit the
    #    body (y +-40 mm)
    for h, want in ((0.020, True), (0.040, False)):
        ok, why = collider(SH.channel(0.076, h)).check(T)
        check(f"channel 76 mm, walls {h * 1000:.0f} mm: {'clear' if want else 'rejected'}", ok == want, why)

    # 3. an 80 mm wall 45 mm to the side (part -y = tool +y): the side box (tool y 40..90) hits it; turned 180 deg
    #    the side box points away and the body (y +-40 mm) clears the wall
    wall = collider(SH.plate(0.25) + [SH.box([0, -0.050, 0.040], [0.5, 0.010, 0.080])])
    for turn, want in ((0, False), (np.pi, True)):
        ok, why = wall.check(G.tool_from_point([0, 0, 0], DOWN, turn))
        check(f"wall 45 mm beside, turned {np.degrees(turn):.0f} deg: {'clear' if want else 'rejected (side box)'}",
              ok == want, why)

    # 4. a bar above the side box's top (tool y 60..70, 95..105 mm up): clear at the grasp, but the side box passes
    #    through it on the way down -> blocked during the approach
    ok, why = collider(SH.plate(0.25) + [SH.box([0, -0.065, 0.100], [0.5, 0.010, 0.010])]).check(T)
    check("bar above the side box: approach blocked, grasp itself clear", not ok and why.startswith("approach"), why)
    # a bar right above the grasp is inside the head at the grasp
    ok, why = collider(SH.overhang(0.050)).check(T)
    check("bar 50 mm above the grasp: rejected at the grasp", not ok and why.endswith("at the grasp"), why)

    # 5. a nut beside the cups, under the body: the body starts 30 mm up -> clear
    ok, why = collider(SH.with_nut([0.0, 0.0])).check(G.tool_from_point([0.06, 0, 0], DOWN))
    check("nut 60 mm beside the grasp, 7 mm tall: clear", ok, why)

    # 6. support plane: a 30 x 30 x 5 mm plate tilted 60 deg, grasped along its normal - a lower body corner reaches
    #    below the plate's lowest edge, into the support plane under it
    a = np.radians(60)
    n = np.array([0, -np.sin(a), np.cos(a)])
    R = trimesh.transformations.rotation_matrix(a, [1, 0, 0])[:3, :3]
    tilted = [SH.box(-n * 0.0025, [0.03, 0.03, 0.005], R)]
    low = SH.as_mesh(tilted).vertices[:, 2].min()
    Tt = G.tool_from_point([0, 0, 0], -n)
    hc = collider(tilted)
    ok0, why0 = hc.check(Tt)
    hc.set_scene(G.pose([0, 0, low]))
    ok1, why1 = hc.check(Tt)
    check("tilted plate: clear without the support plane, rejected with it",
          ok0 and not ok1 and "support" in why1, f"{why0}; {why1}")

    # 7. scene only (phase 2: no part): an obstacle box 20 mm from the body's side is clear, one overlapping it is not
    sc = HeadCollider(None, g)
    for y, want in ((-0.07, True), (-0.035, False)):         # box centre at part y (tool +y side), 20 mm wide
        sc.set_scene(None, [(G.pose([0.0, y - 0.01 if want else y, 0.06]), [0.1, 0.02, 0.04])])
        ok, why = sc.check(G.tool_from_point([0, 0, 0], DOWN, np.pi))   # turned: side box away from the obstacle
        check(f"scene only, obstacle box {'clear of' if want else 'overlapping'} the body: {'clear' if want else 'rejected'}",
              ok == want, why)

    # 8. tool origin inside material (25 mm into a 30 mm block): the cups pass through the block on the approach
    ok, why = collider([SH.box([0, 0, -0.015], [0.2, 0.2, 0.030])]).check(G.tool_from_point([0, 0, -0.025], DOWN))
    check("tool origin inside material: rejected", not ok, why)

    print(f"\n{sum(results)}/{len(results)} passed")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
