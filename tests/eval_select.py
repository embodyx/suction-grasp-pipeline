"""Selection checks (PLAN.md test plan, phase 2) on random conveyor scenes (tests/scenes.py), no robot:
  - share of scenes with at least one reachable grasp, time per select() call, rejection reasons
  - pose noise: select with a noisy part pose (+-2 mm, up to 2 deg about the part centre), then check the chosen
    grasp against the TRUE pose with the seal rule and the head collision check; same for the noise-free choice
  - which pass produced the chosen grasp (database or the step-5 local search)
--search-only: skip the database pass, so every scene tests the local search on its own.
Writes logs/eval_select[_search_only].json and db/dry_run_<part>.png (first scene with a grasp).
Run: .venv/bin/python tests/eval_select.py [--per-part 34] [--workers 3] [--search-only]
"""
import argparse
import collections
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
EVAL_SEED = 90_000


def run_part(args):
    name, n, use_db = args
    import collide as C
    import gripper as G
    import mesh_io as M
    import scenes as SC
    import seal as S
    import select_grasp as SG
    sel = SG.Selector(name)
    m = M.load_part(name)
    _, com = M.properties(m)
    sc_check, hc = S.SealChecker(m, sel.g), C.HeadCollider(M.collision_obj(name, m), sel.g)

    def holds(T_true, T_tool):
        """The chosen tool pose against the true part: seals and is collision-free."""
        Tp = G.inv(T_true) @ T_tool
        return bool(sc_check.check([Tp])["valid"][0]) and hc.check(Tp)[0]

    rng = np.random.default_rng([EVAL_SEED, M.PART_IDS[name]])
    res = dict(found=0, scenes=n, times=[], reasons=collections.Counter(), poses=collections.Counter(),
               found_per_pose=collections.Counter(), clean_ok=0, noisy_found=0, noisy_ok=0, pose_matched=0,
               source=collections.Counter())
    drawn = False
    for _ in range(n):
        sc = SC.sample(rng, name, sel.db["T_rest"], sel.db["pose_prob"])
        out, info = sel.select(sc["T_base_part"], sc["support"], sc["boxes"], use_database=use_db)
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
            sel.render(out[0], os.path.join(ROOT, "db", f"dry_run_{name}.png"))
            drawn = True
        T_noisy = SC.noisy(sc["T_base_part"], com, rng)
        out_n, _ = sel.select(T_noisy, sc["support"], sc["boxes"], k=1, use_database=use_db)
        if out_n:
            res["noisy_found"] += 1
            res["noisy_ok"] += holds(sc["T_base_part"], out_n[0]["T_tool"])
    return name, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-part", type=int, default=34)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--parts", nargs="+")
    ap.add_argument("--search-only", action="store_true")
    args = ap.parse_args()
    import mesh_io as M
    parts = args.parts or M.SMALL
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
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    out = "eval_select_search_only.json" if args.search_only else "eval_select.json"
    json.dump(summary, open(os.path.join(ROOT, "logs", out), "w"), indent=1)
    print("all:", summary["all"])


if __name__ == "__main__":
    main()
