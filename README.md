# Suction grasp pipeline

Suction grasps for rigid parts from their CAD meshes, for any suction head with round cups. It works in two phases:

| Phase | When | Input | Output |
|---|---|---|---|
| 1. Grasp database (`suction_grasps.build_db`) | Offline, once per part and gripper | Part mesh + gripper description | Ranked, validated grasps in the part's mesh frame, tagged with the resting poses they can be used in |
| 2. Grasp selection (`suction_grasps.select`) | At execution time | Database + the part's pose in the world (e.g. from a pose estimator), optionally the support plane and obstacle boxes | Ranked grasps (tool poses) in the world frame; optionally checked by your own robot's reachability |

Phase 1 needs no robot model and no physics engine: it checks the gripper alone against the part. Robot-specific
checks (IK reachability, preferred wrist turns) plug into phase 2. A worked example for an Agibot G2 with a JODELL
EVS08 head is in [examples/g2_evs08](examples/g2_evs08).

---

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/pip install embreex
```

The dependencies are numpy, scipy, trimesh, rtree, networkx, python-fcl and matplotlib. `embreex` (second command)
is optional but recommended: with it, trimesh casts rays about 100 times faster.

## Quick start

```bash
.venv/bin/python -m suction_grasps.build_db --gripper my_head.json --parts part.stl --units mm --out db
```

```bash
.venv/bin/python -m suction_grasps.check_db --db db/part.npz --mesh part.stl --units mm --gripper my_head.json
```

```bash
.venv/bin/python -m suction_grasps.select --db db/part.npz --gripper my_head.json --request request.json --mesh part.stl --units mm
```

1. **Build the database** for a part. It writes `db/part.npz` (grasps), `db/part.json` (settings and statistics) and `db/part_grasps.png` (a picture to review). Useful options:
   - `--offset x y z`: moves the mesh, in its file units, before scaling;
   - `--mass` (kg): overrides volume × density;
   - `--thickness` (m): sheet thickness, for a mesh that is a single open surface;
   - `--density`;
   - `--points`, `--turns`, `--workers`.

   `check_db` and `select` take the same `--units` and `--offset`.
2. **Re-check it** (optional): every saved grasp seals and is collision-free again, and the tags reproduce.
3. **Select grasps** for a part pose. The request JSON holds:
   - `T_world_part`: a 4 × 4 pose in metres;
   - `support`: `{"point": [...], "normal": [...]}`;
   - `boxes`: `[{"center", "size", "quat_xyzw"}]`.

   Passing `--mesh` enables the local-search fallback.

From Python:

```python
from suction_grasps import Selector, load_part
sel = Selector("db/part.npz", "my_head.json", mesh=load_part("part.stl", "mm"))
grasps, info = sel.select(T_world_part, support={"point": [0, 0, 0.78], "normal": [0, 0, 1]},
                          boxes=[], reach=my_reach, order=my_order, k=5)
grasps[0]["T_tool"], grasps[0]["T_pregrasp"], grasps[0]["score"]
```

`reach` and `order` are optional plug-ins:
- **`reach(T_world_tool) -> dict(ok, reason, cost=0, ...)`:** e.g. collision-free IK for your robot. Extra keys, such as joint angles, are returned with the grasp.
- **`order(T_world_tools) -> keys`:** lower keys are tried first, e.g. turns your wrist can reach. Nothing is dropped.

## Frames

- **Part frame:** the mesh file's own frame, after the unit scale (`--units`). A pose estimator that reports poses of the same mesh can feed phase 2 directly. A mesh exported centred on its bounding box can be moved back with `--offset`.
- **Tool frame:** origin at the centre of the cups' lip plane, +z = approach (into the part), x and y in the lip plane. A grasp is a tool pose in the part frame. Phase 2 returns it in the world frame; turning it into your robot's flange pose is your robot's mounting transform.

## Gripper description

A JSON file. All values are in metres, in the tool frame.

```json
{
  "name": "my_head",
  "cups": [{"pos": [0.025, 0.025, 0], "radius": 0.01}, {"pos": [-0.025, 0.025, 0], "radius": 0.01}],
  "cup_height": 0.03,
  "seal": {"rim_points": 16, "seal_tol": 0.0015, "reach": 0.005, "max_normal_deg": 30, "min_cups": 2},
  "hold": {"vacuum_kpa": 60, "safety_g": 1.5},
  "head": [{"type": "box", "lo": [-0.05, -0.04, -0.10], "hi": [0.05, 0.04, -0.03]},
           {"type": "cylinder", "center": [0, 0, -0.15], "radius": 0.04, "height": 0.06},
           {"type": "capsule", "center": [0.06, 0, -0.08], "radius": 0.01, "height": 0.04},
           {"type": "mesh", "file": "head.stl", "scale": 0.001}],
  "turn_range_deg": 360
}
```

| Key | Meaning |
|---|---|
| `cups` | Any number of round cups, each with its lip centre (`pos`) and lip radius. Cup axes are along tool z. |
| `cup_height` | How far the cup and its fitting reach behind the lip; used for the approach check. |
| `seal.seal_tol` | Bellows compliance: how far the lip contact may be from a plane and still seal. |
| `seal.reach` | Bellows travel: how far from the lip plane the surface may be. |
| `seal.max_normal_deg` | Steepest surface tilt under a cup. |
| `seal.min_cups` | Cups that must seal. Default: all of a 1–2 cup head, otherwise all but one. |
| `hold.vacuum_kpa`, `hold.safety_g` | Each cup pulls at most vacuum × lip area; the part's weight × `safety_g` must be held. |
| `head` | Everything except the cups, as boxes, cylinders and capsules (axes along tool z) or meshes. Include whatever moves rigidly with the tool, such as brackets, cameras, the adapter and the wrist. |
| `turn_range_deg` | Head turns about the approach axis are sampled over this range. Use 360 unless the whole head looks the same after a smaller turn. |

`tests/test_gripper.json` is a small complete example, and `examples/g2_evs08/gripper_evs08.json` a real one.

---

## How it works

### Phase 1

1. **Load the part.** Mass and centre of mass come from a closed mesh (volume × density, steel by default). An open single-surface mesh needs `--thickness` or `--mass`; without either, the hold check is skipped and a warning is recorded.
2. **Stable resting poses** (trimesh), with probability above 1%.
3. **Candidates.** Area-weighted surface points (default 10,000) × head turns (default 24 over the turn range). The approach is along −surface normal.
4. **Seal rule, per cup.** Rays along the approach at the cup centre, half the lip radius and the lip, against the exact mesh. A cup seals if:
   - every ray hits within `reach` of the lip plane, on a face looking back at the cup;
   - the hits fit a plane within `seal_tol`;
   - that plane is within `max_normal_deg` of the cup axis.

   At least `min_cups` cups must seal.
5. **Collision** (python-fcl, exact mesh). The head (not the cups) must not touch the part at the grasp, and nothing may touch it along a straight 100 mm approach.
6. **Hold** (simplified Dex-Net 3.0 wrench resistance).
   - The gripper points down, and the weight × `safety_g` acts at the centre of mass. The sealed cups share it in a minimum-norm force split that balances the force and both tipping moments.
   - Margin = 1 − largest pull / cup capacity.
   - One cup, or cups in a line, can't balance a moment this way. Their lips resist peeling up to capacity × radius / 2.
7. **Robustness.** The share of 10 perturbed copies (±2 mm, up to 3°) that still pass steps 4–5. Quality = robustness × hold margin.
8. **Resting-pose tags.** In each stable pose, the approach must be within 60° of down and the head must clear the support plane.
9. **Pruning.** Non-maximum suppression (5 mm, 10°, 25° turn), keeping up to 1000 grasps, with at least 20 per resting pose where they exist.

### Phase 2

1. Transform the database grasps into the world frame.
2. **Resting pose.** Find the stable pose whose down direction matches the support normal (within 15°) and keep its grasps. With no support, or no match, all grasps are used.
3. **Fast filters.** The approach must be within 60° of down, and the head outline must stay above the support at the grasp and the pre-grasp.
4. **Check in order** (`order` keys, then quality):
   - the head and its approach clear the support and the obstacle boxes (exact);
   - then `reach`, if given.

   It stops after `k` grasps or `budget` seconds.
5. **Local search.** Only when nothing survives and the mesh is given: turns and small shifts of the best database grasps, checked online (seal, hold, collision, robustness) before step 4.
6. **Rank** by score = quality − 0.1 × tilt / 60° − `reach`'s cost.

## Tests

```bash
.venv/bin/python tests/test_seal.py
```

```bash
.venv/bin/python tests/test_collide.py
```

```bash
.venv/bin/python tests/test_pipeline.py
```

| Test | What it checks | Status |
|---|---|---|
| `test_seal.py` | Holes, edges, nuts, tilts, cylinders, back faces, mixed cup sizes, hold and peel | 19/19 |
| `test_collide.py` | Channels, a one-sided head, approach-only obstacles, support plane, scene-only checks | 12/12 |
| `test_pipeline.py` | Phase 1 + `check_db` + phase 2 on a synthetic part: plug-ins, local search, obstacles, an open mesh, a single-cup head | 12/12 |

## Limits

- **Cups:** round cups whose axes are along the tool's approach axis only; no oval or rectangular pads.
- **Head:** the head is rigid. Hoses, cables and bellows sideways are not modelled.
- **Hold check:** it assumes the gripper points down while carrying, and has no dynamic loads beyond `safety_g`.
- **Seal model:** geometric only (contact planarity and coverage), not airflow. `seal_tol` and `reach` should be measured for your cups.
- **Phase 2 checks:** the part is only checked against the gripper and the given boxes. Neighbouring parts must be passed as boxes, and arm or robot collisions belong in `reach`.
- **Support plane:** the exact collision check stands in a 4 m square slab, centred under the part, for the support plane.
