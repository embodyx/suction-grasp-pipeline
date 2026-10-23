"""The example's parts: meshes from a pose-estimation service's models folder, copied to assets/parts_src/
(obj_XXXXXX.ply in mm, centred on their bounding box, plus parts.json {id: {"part", "cad_center_mm", ...}}).
The part frame is the CAD frame (mesh + cad_center_mm), the frame the service reports poses in (T_base_cad)."""
import json
import os

from suction_grasps import part as PT

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "assets", "parts_src")
DB = os.path.join(HERE, "db")
SMALL = ["A5917204100", "A5917204200", "A5917204300", "A5917204400", "A5917304100", "A5917304200"]
GRIPPER = os.path.join(HERE, "gripper_evs08.json")


def meta():
    """{part name: (mesh path, cad_center_mm)} from parts.json."""
    m = json.load(open(os.path.join(SRC, "parts.json")))
    return {d["part"]: (os.path.join(SRC, f"obj_{int(i):06d}.ply"), d["cad_center_mm"]) for i, d in m.items()}


def load(name):
    """Part mesh in the CAD frame, metres."""
    path, offset = meta()[name]
    return PT.load_part(path, "mm", offset)


def db_path(name):
    return os.path.join(DB, name + ".npz")


def part_id(name):
    """The model id of a part in parts.json (used to seed the example's random scenes)."""
    return int(next(i for i, d in json.load(open(os.path.join(SRC, "parts.json"))).items() if d["part"] == name))
