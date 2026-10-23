"""Phase 2 checks for the G2 example on random real-cell conveyor scenes (scenes.py), dry run (nothing moves):
  - share of scenes with at least one reachable grasp (G2 reachability plug-in), time per select() call, reasons
  - pose noise: select with a noisy part pose (+-2 mm, up to 2 deg about the part centre), then check the chosen
    grasp against the TRUE pose with the seal rule and the head collision check; same for the noise-free choice
  - which pass produced the chosen grasp (database or the local search)
--search-only: skip the database pass, so every scene tests the local search on its own.
Writes logs/eval_select[_search_only].json and db/dry_run_<part>.png (first scene with a grasp).
Run (from the repository root): python examples/g2_evs08/eval_select.py [--per-part 34] [--workers 3]
"""
import argparse
import collections
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
EVAL_SEED = 90_000


def run_part(args):
    name, n, use_db = args
    import g2_parts as GP
    import scenes as SC
    from g2_reach import G2Reach
    from suction_grasps import part as PT
    from suction_grasps import poses as G
    from suction_grasps.collide import HeadCollider
    from suction_grasps.seal import SealChecker
    from suction_grasps.select import Selector
    m = GP.load(name)
    sel = Selector(GP.db_path(name), GP.GRIPPER, m)
    reach = G2Reach(m, GP.GRIPPER, name)
    _, com = PT.properties(m)
    sc_check, hc = SealChecker(m, sel.g), HeadCollider(m, sel.g)

    def holds(T_true, T_tool):
        """The chosen tool pose against the true part: seals and is collision-free."""
        Tp = G.inv(T_true) @ T_tool
        return bool(sc_check.check([Tp])["valid"][0]) and hc.check(Tp)[0]

    def select(T, sc, k=5):
        reach.set_scene(T, sc["boxes"])                   # the robot plans with the pose it was given
        return sel.select(T, sc["support"], sc["boxes"], reach=reach, order=reach.order, k=k, use_database=use_db)

    rng = np.random.default_rng([EVAL_SEED, GP.part_id(name)])
    res = dict(found=0, scenes=n, times=[], reasons=collections.Counter(), poses=collections.Counter(),
               found_per_pose=collections.Counter(), clean_ok=0, noisy_found=0, noisy_ok=0, pose_matched=0,
               source=collections.Counter())
    drawn = False
    for _ in range(n):
        sc = SC.sample(rng, name, sel.db["T_rest"], sel.db["pose_prob"])
        out, info = select(sc["T_base_part"], sc)
        res["times"].append(info["seconds"])
        res["reasons"].update(info["reasons"])
        res["reasons"].update({"search: " + k_: v for k_, v in info.get("local_search", {}).get("reasons", {}).items()})
        res["search_ran"] = res.get("search_ran", 0) + ("local_search" in info)
        res["poses"][sc["pose_index"]] += 1
        res["pose_matched"] += info["resting_pose"] == sc["pose_index"]
        if not out:
            continue
        res["found"] += 1
        res["source"][out[0]["source"]] += 1
        res["found_per_pose"][sc["pose_index"]] += 1
        res["clean_ok"] += holds(sc["T_base_part"], out[0]["T_tool"])
        if not drawn and use_db:
            reach.render(out[0], os.path.join(GP.DB, f"dry_run_{name}.png"))
            drawn = True
        T_noisy = SC.noisy(sc["T_base_part"], com, rng)
        out_n, _ = select(T_noisy, sc, k=1)
        if out_n:
            res["noisy_found"] += 1
            res["noisy_ok"] += holds(sc["T_base_part"], out_n[0]["T_tool"])
    reach.close()
    return name, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-part", type=int, default=34)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--parts", nargs="+")
    ap.add_argument("--search-only", action="store_true")
    args = ap.parse_args()
    import g2_parts as GP
    parts = args.parts or GP.SMALL
    summary, all_t = {}, []
    with Pool(args.workers) as pool:
        for name, r in pool.imap_unordered(run_part, [(p_, args.per_part, not args.search_only) for p_ in parts]):
            t = np.array(r["times"])
            all_t += r["times"]
            summary[name] = dict(found=f"{r['found']}/{r['scenes']}", pose_matched=f"{r['pose_matched']}/{r['scenes']}",
                                 scenes_per_pose=dict(r["poses"]), found_per_pose=dict(r["found_per_pose"]),
                                 clean_choice_holds=f"{r['clean_ok']}/{r['found']}",
                                 noisy_choice_holds=f"{r['noisy_ok']}/{r['noisy_found']}",
                                 chosen_from=dict(r["source"]), search_ran=r.get("search_ran", 0),
                                 seconds_median=round(float(np.median(t)), 3), seconds_p95=round(float(np.percentile(t, 95)), 3),
                                 reasons=dict(r["reasons"].most_common(6)))
            print(name, json.dumps(summary[name]), flush=True)
    all_t = np.array(all_t)
    summary["all"] = dict(seconds_median=round(float(np.median(all_t)), 3), seconds_p95=round(float(np.percentile(all_t, 95)), 3))
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    out = "eval_select_search_only.json" if args.search_only else "eval_select.json"
    json.dump(summary, open(os.path.join(HERE, "logs", out), "w"), indent=1)
    print("all:", summary["all"])


if __name__ == "__main__":
    main()
