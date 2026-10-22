"""Unit tests for seal.py on synthetic shapes. Run: .venv/bin/python tests/test_seal.py"""
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gripper as G
import seal as S
import shapes as SH

g = G.load()
DOWN = [0, 0, -1]          # approach straight down onto a surface whose top is at z = 0
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def cups_in_part(T):
    """Cup centres in the part frame for tool pose T."""
    return (T[:3, :3] @ g["cups"].T).T + T[:3, 3]


def main():
    # 1. flat plate: every turn of the cup square seals all 4 cups, residual ~0
    sc = S.SealChecker(SH.as_mesh(SH.plate()), g)
    turns = np.radians(np.arange(0, 90, 15))
    r = sc.check([G.tool_from_point([0, 0, 0], DOWN, a) for a in turns])
    check("flat plate, 6 turns: 4 cups sealed", (r["n_sealed"] == 4).all(), f"n_sealed {r['n_sealed']}")
    check("flat plate: residual < 0.01 mm", r["resid"].max() < 1e-5, f"max {r['resid'].max() * 1000:.4f} mm")

    # 2. holes: under one cup -> 3 cups -> valid; under two cups -> invalid
    T = G.tool_from_point([0, 0, 0], DOWN)
    c = cups_in_part(T)
    r1 = S.SealChecker(SH.as_mesh(SH.plate_with_holes([c[0][:2]])), g).check([T])
    check("hole under cup 1: cups 2-4 seal, valid", r1["sealed"][0].tolist() == [False, True, True, True] and r1["valid"][0],
          f"sealed {r1['sealed'][0].tolist()}")
    r2 = S.SealChecker(SH.as_mesh(SH.plate_with_holes([c[0][:2], c[1][:2]])), g).check([T])
    check("holes under cups 1+2: invalid", r2["n_sealed"][0] == 2 and not r2["valid"][0], f"n_sealed {r2['n_sealed'][0]}")

    # 3. plate edge: the outer cups' lips reach x = 25 + 12 = 37 mm
    r_out = S.SealChecker(SH.as_mesh(SH.plate_edge(0.033)), g).check([T])
    r_in = S.SealChecker(SH.as_mesh(SH.plate_edge(0.040)), g).check([T])
    check("edge at 33 mm: lips over the edge, invalid", r_out["n_sealed"][0] == 2, f"n_sealed {r_out['n_sealed'][0]}")
    check("edge at 40 mm: all lips on the plate, 4 cups", r_in["n_sealed"][0] == 4, f"n_sealed {r_in['n_sealed'][0]}")

    # 4. a nut under one cup stands in front of it (hit at -7 mm, beyond cup_reach)
    r = S.SealChecker(SH.as_mesh(SH.with_nut(c[2][:2])), g).check([T])
    check("nut under cup 3: that cup fails, 3 cups valid", r["sealed"][0].tolist() == [True, True, False, True] and r["valid"][0],
          f"sealed {r['sealed'][0].tolist()}, cup 3 centre depth {r['depth'][0, 2, 0] * 1000:.1f} mm")

    # 5. normal rule alone: one cup at the tool origin, generous reach; plate tilted 25 deg passes, 35 deg fails
    g1 = copy.deepcopy(g)
    g1.update(cups=np.zeros((1, 3)), min_cups=1, cup_reach=0.02)
    for deg, want in ((25, True), (35, False)):
        tilt = np.eye(4)
        tilt[:3, :3] = G.pose([0, 0, 0], [np.sin(np.radians(deg) / 2), 0, 0, np.cos(np.radians(deg) / 2)])[:3, :3]
        r = S.SealChecker(SH.as_mesh(SH.plate(), tilt), g1).check([G.tool_from_point([0, 0, 0], DOWN)])
        check(f"surface tilted {deg} deg to the cup axis: {'seals' if want else 'fails'}", bool(r["valid"][0]) == want)

    # 6. cylinder: measured plane-fit residual matches the analytic surface; seal decision follows seal_tol
    pat = S.cup_pattern(g)
    for R in (0.30, 0.10, 0.06, 0.04):
        sc = S.SealChecker(SH.cylinder(R), g)
        r = sc.check([T])
        ok, detail = True, []
        for j, cup in enumerate(c):
            xy = cup[:2] + (T[:2, :2] @ pat.T).T                 # ray points in the part frame (xy)
            if np.abs(xy[:, 1]).max() >= R:
                continue
            z = np.sqrt(R ** 2 - xy[:, 1] ** 2) - R
            A = np.c_[xy, np.ones(len(xy))]
            coef, *_ = np.linalg.lstsq(A, z, rcond=None)
            want_resid = np.abs(A @ coef - z).max()
            got = r["resid"][0, j]
            if np.isfinite(got):
                ok &= abs(got - want_resid) < 1e-4
                ok &= bool(r["sealed"][0, j]) == (want_resid <= g["seal_tol"] and np.abs(z).max() <= g["cup_reach"]
                                                  and 1 / np.linalg.norm([coef[0], coef[1], 1]) >= np.cos(np.radians(g["max_normal_deg"])))
            else:                         # no residual: some ray must be out of reach (the rim drops below the tips)
                ok &= np.abs(z).max() > g["cup_reach"] and not r["sealed"][0, j]
            detail.append(f"{want_resid * 1000:.2f}/{got * 1000:.2f} (rim {np.abs(z).max() * 1000:.1f} mm down)")
        check(f"cylinder R={R * 1000:.0f} mm: residual analytic/measured (mm) and decision agree", ok,
              f"{detail[0]}  n_sealed {r['n_sealed'][0]}")
    # single cup on the top line, generous reach: only the residual rule decides (seal_tol 1.5 mm)
    #    (the expected decision comes from the analytic plane-fit residual of the cup's ray points)
    outcomes = []
    for R in (0.08, 0.02):
        xy = S.cup_pattern(g1)
        z = np.sqrt(R ** 2 - xy[:, 1] ** 2) - R
        A = np.c_[xy, np.ones(len(xy))]
        want_resid = np.abs(A @ np.linalg.lstsq(A, z, rcond=None)[0] - z).max()
        r = S.SealChecker(SH.cylinder(R), g1).check([G.tool_from_point([0, 0, 0], DOWN)])
        want = want_resid <= g["seal_tol"]
        outcomes.append(want)
        check(f"one cup on a cylinder R={R * 1000:.0f} mm: {'seals' if want else 'fails on the residual'}",
              bool(r["valid"][0]) == want, f"residual analytic {want_resid * 1000:.2f} / measured {r['resid'][0, 0] * 1000:.2f} mm")
    check("the single-cup cylinder cases cover both outcomes", outcomes == [True, False])

    # 7. hold check
    cap = g["vacuum_kpa"] * 1000 * np.pi * g["cup_radius"] ** 2
    W = 0.2 * 9.81 * g["safety_g"]                  # light part: every share below one cup's capacity
    T0 = G.tool_from_point([0, 0, 0], DOWN)
    m4 = S.hold_margin(T0, np.ones(4, bool), np.array([0, 0, -0.001]), 0.2, g)
    check("hold, 0.2 kg at the centre, 4 cups: each pulls W/4", abs(m4 - (1 - W / 4 / cap)) < 1e-6, f"margin {m4:.3f}")
    m3 = S.hold_margin(T0, np.array([True, True, True, False]), np.array([0, 0, -0.001]), 0.2, g)
    check("hold, 3 cups: the corner cup carries W/2", abs(m3 - (1 - W / 2 / cap)) < 1e-6, f"margin {m3:.3f}")
    mf = S.hold_margin(T0, np.ones(4, bool), np.array([0.10, 0, -0.001]), 3.0, g)
    check("hold, 3 kg 100 mm off-centre: peels off (margin 0)", mf == 0.0, f"margin {mf:.3f}")

    # 8. no grasps at all (e.g. the local search with no seeds): empty arrays, no error
    r0 = S.SealChecker(SH.as_mesh(SH.plate()), g).check(np.zeros((0, 4, 4)))
    check("empty input gives empty results", r0["valid"].shape == (0,) and r0["sealed"].shape == (0, 4))

    print(f"\n{sum(results)}/{len(results)} passed")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
