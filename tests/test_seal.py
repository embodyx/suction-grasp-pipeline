"""Unit tests for seal.py on synthetic shapes. Run: python tests/test_seal.py

Test gripper (tests/test_gripper.json): 4 cups, lip radius 10 mm, in a 50 mm square (lips reach 35 mm from the
centre), seal_tol 1.5 mm, reach 5 mm, 30 deg, 3 of 4 cups.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import shapes as SH
from suction_grasps import poses as G
from suction_grasps import seal as S
from suction_grasps.gripper import Gripper

g = Gripper.load(os.path.join(HERE, "test_gripper.json"))
DOWN = [0, 0, -1]          # approach straight down onto a surface whose top is at z = 0
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def one_cup(radius=0.01, reach=0.02, **seal):
    """Single-cup gripper (cup at the tool origin) with a generous reach, so other rules can be tested alone."""
    return Gripper(dict(cups=[dict(pos=[0, 0, 0], radius=radius)], seal=dict(reach=reach, min_cups=1, **seal)))


def cups_in_part(T):
    """Cup centres in the part frame for tool pose T."""
    return (T[:3, :3] @ g.cups.T).T + T[:3, 3]


def main():
    # 1. flat plate: every turn seals all 4 cups, residual ~0
    sc = S.SealChecker(SH.as_mesh(SH.plate()), g)
    r = sc.check([G.tool_from_point([0, 0, 0], DOWN, a) for a in np.radians(np.arange(0, 360, 30))])
    check("flat plate, 12 turns: 4 cups sealed", (r["n_sealed"] == 4).all(), f"n_sealed {r['n_sealed']}")
    check("flat plate: residual < 0.01 mm", r["resid"].max() < 1e-5, f"max {r['resid'].max() * 1000:.4f} mm")

    # 2. holes: under one cup -> 3 cups -> valid; under two cups -> invalid
    T = G.tool_from_point([0, 0, 0], DOWN)
    c = cups_in_part(T)
    r1 = S.SealChecker(SH.as_mesh(SH.plate_with_holes([c[0][:2]])), g).check([T])
    check("hole under cup 1: cups 2-4 seal, valid", r1["sealed"][0].tolist() == [False, True, True, True] and r1["valid"][0],
          f"sealed {r1['sealed'][0].tolist()}")
    r2 = S.SealChecker(SH.as_mesh(SH.plate_with_holes([c[0][:2], c[1][:2]])), g).check([T])
    check("holes under cups 1+2: invalid", r2["n_sealed"][0] == 2 and not r2["valid"][0], f"n_sealed {r2['n_sealed'][0]}")

    # 3. plate edge: the outer cups' lips reach x = 25 + 10 = 35 mm
    r_out = S.SealChecker(SH.as_mesh(SH.plate_edge(0.030)), g).check([T])
    r_in = S.SealChecker(SH.as_mesh(SH.plate_edge(0.040)), g).check([T])
    check("edge at 30 mm: lips over the edge, invalid", r_out["n_sealed"][0] == 2, f"n_sealed {r_out['n_sealed'][0]}")
    check("edge at 40 mm: all lips on the plate, 4 cups", r_in["n_sealed"][0] == 4, f"n_sealed {r_in['n_sealed'][0]}")

    # 4. a nut under one cup stands in front of it (hit at -7 mm, beyond the 5 mm reach)
    r = S.SealChecker(SH.as_mesh(SH.with_nut(c[2][:2])), g).check([T])
    check("nut under cup 3: that cup fails, 3 cups valid", r["sealed"][0].tolist() == [True, True, False, True] and r["valid"][0],
          f"sealed {r['sealed'][0].tolist()}, cup 3 centre depth {r['depth'][0, 2, 0] * 1000:.1f} mm")

    # 5. normal rule alone (one cup, generous reach): plate tilted 25 deg seals, 35 deg fails
    g1 = one_cup()
    for deg, want in ((25, True), (35, False)):
        tilt = np.eye(4)
        tilt[:3, :3] = G.pose([0, 0, 0], [np.sin(np.radians(deg) / 2), 0, 0, np.cos(np.radians(deg) / 2)])[:3, :3]
        r = S.SealChecker(SH.as_mesh(SH.plate(), tilt), g1).check([G.tool_from_point([0, 0, 0], DOWN)])
        check(f"surface tilted {deg} deg to the cup axis: {'seals' if want else 'fails'}", bool(r["valid"][0]) == want)

    # 6. residual rule alone (one cup on a cylinder top): the decision follows the analytic plane-fit residual
    outcomes = []
    for R in (0.08, 0.02):
        xy = S.cup_pattern(0.01, g1.seal["rim_points"])
        z = np.sqrt(R ** 2 - xy[:, 1] ** 2) - R
        A = np.c_[xy, np.ones(len(xy))]
        want_resid = np.abs(A @ np.linalg.lstsq(A, z, rcond=None)[0] - z).max()
        r = S.SealChecker(SH.cylinder(R), g1).check([G.tool_from_point([0, 0, 0], DOWN)])
        want = want_resid <= g1.seal["seal_tol"]
        outcomes.append(want)
        check(f"one cup on a cylinder R={R * 1000:.0f} mm: {'seals' if want else 'fails on the residual'}",
              bool(r["valid"][0]) == want, f"residual analytic {want_resid * 1000:.2f} / measured {r['resid'][0, 0] * 1000:.2f} mm")
    check("the cylinder cases cover both outcomes", outcomes == [True, False])

    # 7. back-face rule: tool origin 25 mm below the top of a 30 mm block, so the rays (from 20 mm behind the lips)
    #    start inside it and first hit its bottom face from inside, 5 mm beyond the lips (within the reach)
    block = SH.as_mesh([SH.box([0, 0, -0.015], [0.2, 0.2, 0.030])])
    r = S.SealChecker(block, one_cup(reach=0.03)).check([G.tool_from_point([0, 0, -0.025], DOWN)])
    check("rays starting inside material: no seal (first hits are back faces)", not r["valid"][0])

    # 8. cups of different sizes on a 20 mm wide strip: the 10 mm cup fits, the 30 mm cup's lip overhangs it
    g2 = Gripper(dict(cups=[dict(pos=[0.03, 0, 0], radius=0.015), dict(pos=[-0.03, 0, 0], radius=0.005)],
                      seal=dict(min_cups=1)))
    strip = SH.as_mesh([SH.box([0, 0, -0.001], [0.2, 0.020, 0.002])])
    r = S.SealChecker(strip, g2).check([G.tool_from_point([0, 0, 0], DOWN)])
    check("per-cup radius on a 20 mm strip: big cup (30 mm) fails, small cup (10 mm) seals",
          r["sealed"][0].tolist() == [False, True], f"sealed {r['sealed'][0].tolist()}")

    # 9. hold check
    cap = g.cup_capacity()[0]
    W = 0.2 * 9.81 * g.hold["safety_g"]                      # light part: every share below one cup's capacity
    T0 = G.tool_from_point([0, 0, 0], DOWN)
    m4 = S.hold_margin(T0, np.ones(4, bool), np.array([0, 0, -0.001]), 0.2, g)
    check("hold, 0.2 kg at the centre, 4 cups: each pulls W/4", abs(m4 - (1 - W / 4 / cap)) < 1e-6, f"margin {m4:.3f}")
    m3 = S.hold_margin(T0, np.array([True, True, True, False]), np.array([0, 0, -0.001]), 0.2, g)
    check("hold, 3 cups: the corner cup carries W/2", abs(m3 - (1 - W / 2 / cap)) < 1e-6, f"margin {m3:.3f}")
    mf = S.hold_margin(T0, np.ones(4, bool), np.array([0.10, 0, -0.001]), 3.0, g)
    check("hold, 3 kg 100 mm off-centre: peels off (margin 0)", mf == 0.0, f"margin {mf:.3f}")
    gs = one_cup(radius=0.02)
    ms0 = S.hold_margin(T0, np.ones(1, bool), np.array([0, 0, -0.001]), 0.2, gs)
    ms1 = S.hold_margin(T0, np.ones(1, bool), np.array([0.05, 0, -0.001]), 0.2, gs)
    check("hold, single cup: centred holds, 50 mm off-centre peels more (peel model)", ms0 > ms1 >= 0,
          f"margin {ms0:.3f} -> {ms1:.3f}")

    # 10. no grasps at all: empty arrays, no error
    r0 = S.SealChecker(SH.as_mesh(SH.plate()), g).check(np.zeros((0, 4, 4)))
    check("empty input gives empty results", r0["valid"].shape == (0,) and r0["sealed"].shape == (0, 4))

    print(f"\n{sum(results)}/{len(results)} passed")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
