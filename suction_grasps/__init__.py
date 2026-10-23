"""Suction grasp pipeline: an offline grasp database per part from its CAD mesh and a gripper description (phase 1,
build_db), and grasp selection at execution time from the part's pose (phase 2, select)."""
from .gripper import Gripper
from .part import load_part, properties, stable_poses
from .select import Selector
