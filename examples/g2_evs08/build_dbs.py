"""Phase 1 for the example's parts with the EVS08 head -> db/<part>.npz/.json/_grasps.png (generic build_db with
each part's CAD offset). Usage (from the repository root):
  python examples/g2_evs08/build_dbs.py [--parts A5917204100 ...] [--workers 3]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import g2_parts as GP                          # noqa: E402
from suction_grasps.build_db import build      # noqa: E402


def run(args):
    name, points = args
    path, offset = GP.meta()[name]
    return build(path, GP.GRIPPER, GP.DB, name=name, units="mm", offset=offset, points=points)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=GP.SMALL)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--points", type=int, default=10000)
    args = ap.parse_args()
    with Pool(args.workers) as pool:
        for name, stats in pool.imap_unordered(run, [(n, args.points) for n in args.parts]):
            print(name, json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
