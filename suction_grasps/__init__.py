"""Suction grasp pipeline: an offline grasp database per part from its CAD mesh and a gripper description (phase 1,
build_db), and grasp selection at execution time from the part's pose (phase 2, select)."""
from .gripper import Gripper
from .part import load_part, properties, stable_poses


def __getattr__(name):
    """Selector is imported on first use, so `python -m suction_grasps.select` does not import the module twice."""
    if name == "Selector":
        from .select import Selector
        return Selector
    raise AttributeError(name)
