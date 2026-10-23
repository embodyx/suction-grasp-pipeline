"""Load the G2 with the default suction head (EVS08 on both arms; assets/robot, copied by copy_robot.py) into PyBullet."""
import json
import os
import re

import pybullet as p

ROOT = os.path.dirname(os.path.abspath(__file__))
DESC_DIR = os.path.join(ROOT, "assets", "robot")
URDF = os.path.join(DESC_DIR, "urdf/G2_t2_crsB/G2_t2_crsB_evs08.urdf")   # EVS08 on both arms (g2_pybullet_sim build_evs08.py)
SRDF = os.path.join(DESC_DIR, "srdf/G2_t2_crsB/G2_t2_crsB.srdf")
HOME = os.path.join(DESC_DIR, "home_pose.json")


def resolved_urdf():
    """PyBullet doesn't resolve ROS 'package://' URIs, so write a copy with absolute mesh paths."""
    text = re.sub(r"package://genie_robot_description/", DESC_DIR + "/", open(URDF).read())
    out = os.path.join(ROOT, "build", "evs08.urdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = f"{out}.{os.getpid()}.tmp"          # atomic replace: parallel workers never read a half-written file
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, out)
    return out


def load_robot(client):
    """Load the G2 with its base fixed to the world. Returns (body id, {link name: index}, {joint name: index})."""
    # Fixed links are kept (no merging) so tool frames like gripper_r_suction_center can be looked up
    body = p.loadURDF(resolved_urdf(), useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=client)
    links, joints = {}, {}
    for j in range(p.getNumJoints(body, physicsClientId=client)):
        info = p.getJointInfo(body, j, physicsClientId=client)
        links[info[12].decode()] = j
        joints[info[1].decode()] = j
    return body, links, joints


def home_pose():
    """{joint name: angle} of the robot's home posture."""
    return json.load(open(HOME))["joints"]


def set_joints(body, joints, q, client):
    for n, v in q.items():
        p.resetJointState(body, joints[n], float(v), physicsClientId=client)
