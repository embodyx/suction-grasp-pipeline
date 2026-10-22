"""Copy the G2 model with the default suction head (EVS08 on both arms) from g2_pybullet_sim into assets/robot/.

The head is built in the sim by build_evs08.py (measured dimensions: flange -> cup lip 128 mm, 49.5 mm cup square,
24 mm cups, EVS08 body box, side hardware as lfd-pick's capsule ring) - one source for both projects. This copies
that URDF, only the mesh files it references (same relative paths), the SRDF (collision exclusions) and the home
pose. After this the pipeline needs nothing from g2_pybullet_sim. Rerun after the sim's head changes.

Usage: .venv/bin/python tools/copy_robot.py [--sim ../g2_pybullet_sim]   (default: a g2_pybullet_sim folder next to this repo)
"""
import argparse
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = "urdf/G2_t2_crsB/G2_t2_crsB_evs08.urdf"
SRDF = "srdf/G2_t2_crsB/G2_t2_crsB.srdf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=os.path.join(os.path.dirname(ROOT), "g2_pybullet_sim"))
    args = ap.parse_args()
    src = os.path.join(args.sim, "assets", "genie_robot_description")
    dst = os.path.join(ROOT, "assets", "robot")
    text = open(os.path.join(src, URDF)).read()
    meshes = sorted(set(re.findall(r'filename="package://genie_robot_description/([^"]+)"', text)))
    for rel in [URDF, SRDF] + meshes:
        os.makedirs(os.path.dirname(os.path.join(dst, rel)), exist_ok=True)
        shutil.copy2(os.path.join(src, rel), os.path.join(dst, rel))
    shutil.copy2(os.path.join(args.sim, "home_pose.json"), os.path.join(dst, "home_pose.json"))
    print(f"copied {URDF}, SRDF, home pose and {len(meshes)} meshes -> {dst}")


if __name__ == "__main__":
    main()
