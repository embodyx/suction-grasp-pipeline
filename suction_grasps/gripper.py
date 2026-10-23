"""Suction gripper description (a JSON file) for any head with round cups.

Tool frame: origin at the centre of the cups' lip plane, +z = approach direction (into the part), x/y in the lip
plane. Everything in metres, in the tool frame:

  {
    "name": "my_head",
    "cups": [{"pos": [x, y, 0], "radius": r}, ...],      any number of round cups (lips in the plane z = pos[2])
    "cup_height": 0.03,                                    cup + fitting length behind the lip (approach check)
    "seal": {"rim_points": 16, "seal_tol": 0.0015, "reach": 0.005, "max_normal_deg": 30, "min_cups": 3},
    "hold": {"vacuum_kpa": 60, "safety_g": 1.5},
    "head": [                                              the rest of the head (NOT the cups), as shapes:
      {"type": "box", "lo": [x, y, z], "hi": [x, y, z]},
      {"type": "cylinder", "center": [x, y, z], "radius": r, "height": h},   axis along tool z
      {"type": "capsule", "center": [x, y, z], "radius": r, "height": h},    axis along tool z, h = between caps
      {"type": "mesh", "file": "head.stl", "scale": 0.001, "transform": 4x4}  path relative to the JSON file;
                                                                              optional transform into the tool frame
    ],
    "turn_range_deg": 360                                  turns sampled over this range (90 for a head that looks
  }                                                        the same every 90 deg, e.g. a bare square of cups)

Seal and hold rules: see seal.py. Shapes behind the lip plane have negative z.
"""
import json
import os

import numpy as np
import trimesh

DEFAULT_SEAL = dict(rim_points=16, seal_tol=0.0015, reach=0.005, max_normal_deg=30.0, min_cups=None)
DEFAULT_HOLD = dict(vacuum_kpa=60.0, safety_g=1.5)


class Gripper:
    def __init__(self, spec, base_dir="."):
        self.spec = spec
        self.name = spec.get("name", "gripper")
        self.cups = np.array([c["pos"] for c in spec["cups"]], float)            # (n, 3) lip centres
        self.cup_radii = np.array([c["radius"] for c in spec["cups"]], float)    # (n,)
        self.cup_height = float(spec.get("cup_height", 0.03))
        self.seal = {**DEFAULT_SEAL, **spec.get("seal", {})}
        if self.seal["min_cups"] is None:                    # default: all cups of a 1-2 cup head, else all but one
            self.seal["min_cups"] = len(self.cups) if len(self.cups) <= 2 else len(self.cups) - 1
        self.hold = {**DEFAULT_HOLD, **spec.get("hold", {})}
        self.turn_range = np.radians(spec.get("turn_range_deg", 360.0))
        self.head = [self._shape(s, base_dir) for s in spec.get("head", [])]    # trimesh objects, tool frame
        # cup bodies for the approach check: cylinders from the lip back by cup_height
        self.cup_shapes = [trimesh.creation.cylinder(r, self.cup_height,
                                                     transform=trimesh.transformations.translation_matrix(
                                                         [c[0], c[1], c[2] - self.cup_height / 2]))
                           for c, r in zip(self.cups, self.cup_radii)]

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path)), os.path.dirname(os.path.abspath(path)))

    @staticmethod
    def _shape(s, base_dir):
        """One head shape as a trimesh object in the tool frame."""
        t = s["type"]
        if t == "box":
            lo, hi = np.array(s["lo"], float), np.array(s["hi"], float)
            return trimesh.creation.box(extents=hi - lo, transform=trimesh.transformations.translation_matrix((lo + hi) / 2))
        tr = trimesh.transformations.translation_matrix(s.get("center", [0, 0, 0]))
        if t == "cylinder":
            return trimesh.creation.cylinder(s["radius"], s["height"], transform=tr)
        if t == "capsule":
            return trimesh.creation.capsule(height=s["height"], radius=s["radius"], transform=tr)
        if t == "mesh":
            m = trimesh.load(os.path.join(base_dir, s["file"]), force="mesh")
            m.apply_scale(s.get("scale", 1.0))
            if "transform" in s:
                m.apply_transform(np.array(s["transform"]))
            return m
        raise ValueError(f"unknown head shape type {t!r}")

    def outline_points(self):
        """Vertices of all head shapes and cups (tool frame, (n, 3)): for fast plane-clearance checks."""
        return np.vstack([m.convex_hull.vertices for m in self.head + self.cup_shapes])

    def cup_capacity(self):
        """Pull force one cup can hold (N): vacuum x lip area, per cup."""
        return self.hold["vacuum_kpa"] * 1000 * np.pi * self.cup_radii ** 2
