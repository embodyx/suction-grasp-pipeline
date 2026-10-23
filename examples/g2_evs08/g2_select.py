"""Phase 2 on the G2 (dry run, nothing moves): grasps for a request in the pose-estimation service's format, with the
G2 reachability plug-in. Usage (from the repository root):
  python examples/g2_evs08/g2_select.py examples/g2_evs08/request_A5917204300.json [--render dry_run.png]
  request: {"part": name, "T_base_cad_mm": 4x4 (mm) or "T_base_part": 4x4 (m), "support": {"point", "normal"},
            "boxes": [{"center", "size", "quat_xyzw"}]}
Prints, best first: flange pose to command, IK joints (right arm + body), score.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import g2_parts as GP                           # noqa: E402
import ik                                       # noqa: E402
from g2_reach import G2Reach                    # noqa: E402
from suction_grasps import poses as G           # noqa: E402
from suction_grasps.select import Selector      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request")
    ap.add_argument("--render")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    req = json.load(open(args.request))
    if "T_base_part" in req:                              # metres
        T = np.array(req["T_base_part"], float)
    else:                                                 # the pose-estimation service's T_base_cad_mm
        T = np.array(req["T_base_cad_mm"], float)
        T[:3, 3] /= 1000.0
    mesh = GP.load(req["part"])
    sel = Selector(GP.db_path(req["part"]), GP.GRIPPER, mesh)
    reach = G2Reach(mesh, GP.GRIPPER, req["part"])
    reach.set_scene(T, req.get("boxes", ()))
    out, info = sel.select(T, req.get("support"), req.get("boxes", ()), reach=reach, order=reach.order, k=args.k)
    print(json.dumps(info))
    for r in out:
        pos, quat = G.pos_quat(r["T_flange"])
        print(json.dumps(dict(id=r["id"], source=r["source"], score=round(r["score"], 3), quality=round(r["quality"], 3),
                              flange_pos=pos.round(4).tolist(), flange_quat_xyzw=quat.round(4).tolist(),
                              q_grasp={k: round(v, 4) for k, v in r["q_grasp"].items()
                                       if k in ik.ARM_JOINTS["r"] + ik.BODY_JOINTS})))
    if args.render and out:
        print("wrote", reach.render(out[0], args.render))
    reach.close()


if __name__ == "__main__":
    main()
