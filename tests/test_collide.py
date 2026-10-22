"""Unit tests for collide.py on synthetic shapes. Run: .venv/bin/python tests/test_collide.py

Head in the tool frame (EVS08, g2_pybullet_sim build_evs08.py): body box x +-57.5, y +-50 mm, 29..114 mm above the
cup lips; cups (24 mm, 1 mm convex margin) at +-24.75 mm, 0..29 mm up; side hardware = 6 capsules (r 65 mm,
axes 90 mm out at 0, 60, .. 300 deg from tool x, lowest point 23 mm above the lips, reaching 155 mm out between
~88 and ~148 mm up). (With approach straight down onto a part whose top is z = 0: tool x = part x, tool y = part -y,
height above the lips = part z.)
"""
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import collide as C
import gripper as G
import shapes as SH

g = G.load()
TMP = os.path.join(os.path.dirname(HERE), "build", "test_shapes")
DOWN = [0, 0, -1]
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def collider(name, boxes):
    return C.HeadCollider(SH.as_obj(boxes, os.path.join(TMP, name + ".obj")), g)


def main():
    T = G.tool_from_point([0, 0, 0], DOWN)       # grasp on the floor top at the origin
    hc = collider("plate", SH.plate())

    # 1. place() round trip: the robot's tool lands exactly on the requested pose, for several poses
    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(5):
        Tq = G.tool_from_point(rng.uniform(-0.05, 0.05, 3), rng.normal(size=3), rng.uniform(0, np.pi))
        hc.place(Tq)
        err = max(err, np.abs(hc.tool_in_part() - Tq).max())
    check("place() round trip", err < 1e-6, f"max error {err:.1e}")

    # 2. flat plate, straight approach: clear (cups touch only at the grasp)
    ok, why = hc.check(T)
    check("flat plate: clear", ok, why)

    # 3. channel 80 mm wide (cups reach +-37 mm): low walls (20 mm: below the body at 29 mm and the side hardware,
    #    ~35 mm up at the walls) clear; 40 mm walls hit the body (y +-50 mm)
    for h, want in ((0.020, True), (0.040, False)):
        ok, why = collider(f"channel_{h}", SH.channel(0.080, h)).check(T)
        check(f"channel 80 mm, walls {h * 1000:.0f} mm: {'clear' if want else 'rejected'}", ok == want, why)
    # one 60 mm wall to the side: 120 mm away the side hardware (reaches 155 mm) hits it at any turn; 170 mm away clear
    for dist, want in ((0.120, False), (0.170, True)):
        hc_w = collider(f"wall_{dist}", SH.plate(0.25) + [SH.box([0, -(dist + 0.005), 0.030], [0.5, 0.010, 0.060])])
        for turn in (0, np.pi / 2):
            ok, why = hc_w.check(G.tool_from_point([0, 0, 0], DOWN, turn))
            check(f"wall {dist * 1000:.0f} mm beside, 60 mm tall, turned {np.degrees(turn):.0f} deg: {'clear' if want else 'rejected'}",
                  ok == want, why)

    # 4. a bar 145-152 mm out (tool x) and 225-235 mm up: clear at the grasp (the ring's top is ~177 mm up there),
    #    but the ring passes through it on the way down -> blocked during the approach
    ok, why = collider("bar", SH.plate(0.25) + [SH.box([0.1485, 0, 0.230], [0.007, 0.5, 0.010])]).check(T)
    check("bar beside the side hardware: approach blocked, grasp itself clear", not ok and why.startswith("approach"), why)
    # a bar right above the grasp is inside the head at the grasp
    ok, why = collider("overhang", SH.overhang(0.050)).check(T)
    check("bar 50 mm above the grasp: rejected at the grasp", not ok and why.endswith("at the grasp"), why)

    # 5. a nut next to the cups (inside the body's footprint, below the body): cups pass it, body clears 7 mm
    ok, why = collider("nut_side", SH.with_nut([0.0, 0.0])).check(G.tool_from_point([0.06, 0, 0], DOWN))
    check("nut 60 mm beside the grasp, 7 mm tall: clear (body starts 29 mm up)", ok, why)

    # 5b. support plane: a small 30 x 30 x 5 mm plate tilted 60 deg, grasped along its normal. The cups sit
    #     square on it; the lower body corner (50 mm out, 29 mm up) reaches ~29 mm below the grasp point,
    #     the plate's lowest edge only ~15 mm -> the corner is ~13 mm into the support plane under the plate
    a = np.radians(60)
    n = np.array([0, -np.sin(a), np.cos(a)])                       # plate top normal
    R = trimesh.transformations.rotation_matrix(a, [1, 0, 0])[:3, :3]
    tilted = [SH.box(-n * 0.0025, [0.03, 0.03, 0.005], R)]
    low = SH.as_mesh(tilted).vertices[:, 2].min()
    Tt = G.tool_from_point([0, 0, 0], -n)
    hc_s = collider("tilted_plate", tilted)
    ok0, why0 = hc_s.check(Tt)
    hc_s.set_support(G.pose([0, 0, low]))
    ok1, why1 = hc_s.check(Tt)
    check("tilted plate: clear without the support plane, rejected with it",
          ok0 and not ok1 and "support" in why1, f"{why0}; {why1} (plate lowest point {low * 1000:.1f} mm)")

    # 6. seal + collision together: a grasp whose ray origins start inside material. Thick block (top at
    #    z = 0, 30 mm deep) with the tool origin 15 mm below its top: the rays (from 20 mm behind the tips)
    #    start 5 mm inside the block and their first hit is the bottom face 15 mm on -> the seal rule alone
    #    could be fooled; the collision check must reject it (cups pass through the block on the approach)
    import seal as S
    block = [SH.box([0, 0, -0.015], [0.2, 0.2, 0.030])]
    Tin = G.tool_from_point([0, 0, -0.015], DOWN)
    ok, why = collider("block", block).check(Tin)
    sealed = S.SealChecker(SH.as_mesh(block), g).check([Tin])["valid"][0]
    check("tool origin inside material: collision check rejects it", not ok, f"{why}; seal rule alone says valid={sealed}")

    print(f"\n{sum(results)}/{len(results)} passed")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
