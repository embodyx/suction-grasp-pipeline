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

From the repository root:

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
```

```bash
.venv/bin/pip install embreex
```

- **Requirements:** Python 3.10 or newer (tested 3.10–3.14). The dependencies are numpy, scipy, trimesh, rtree, networkx, python-fcl and matplotlib; `requirements.txt` lists the same.
- **`embreex`** (second command) is optional but recommended: with it, trimesh casts rays about 100 times faster.
- **Installing:** `pip install -e .` makes `suction_grasps` importable from anywhere. Without installing, run from the repository root.
- **CPU threads:** numpy may use all CPU cores. Set `OMP_NUM_THREADS=1` to keep a run on one core per process.

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
   - `--points`, `--turns`;
   - `--workers`: parts built in parallel; one part always uses one process.

   `--mass`, `--thickness`, `--offset` and `--units` apply to every part in one call, so parts that differ need separate calls. `check_db` and `select` take the same `--units` and `--offset`.

   A mesh that is not watertight has no volume to weigh. Pass `--mass` for a solid part, or `--thickness` for a part modelled as one surface. Otherwise the hold check is skipped, and `<part>.json` records a warning.
2. **Re-check it** (optional): every saved grasp seals and is collision-free again, and the tags reproduce.
3. **Select grasps** for a part pose. The request JSON holds:
   - `T_world_part`: a 4 × 4 pose in metres;
   - `support`: `{"point": [...], "normal": [...]}`;
   - `boxes`: `[{"center", "size", "quat_xyzw"}]`, where `size` is the full edge length in metres.

   `--k` sets the number of grasps (default 5); `--mesh` enables the local-search fallback. It prints one info line (counts and rejection reasons) and then one JSON line per grasp, best first: `id`, `source`, `score`, `quality`, `tilt_deg`, `tool_pos`, `tool_quat_xyzw`.

   For a part lying on another part, put the support plane at the part's lowest point, and pass the part below as boxes if needed.

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

Each grasp carries:
- `T_tool`, and `T_pregrasp`, which is 100 mm back along the approach (`pregrasp=`);
- `score`, `quality` and `tilt_deg`;
- `source`: `database` or `local search`.

Other `select` arguments:
- `budget`: seconds for the database pass (default 2), which matters when `reach` is slow;
- `fallback` and `fallback_budget` (default 3 s) for the local search;
- `max_down_deg` (default 60).

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
  "channels": [[0, 1]],
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
| `seal.rim_points` | Rays on each cup's lip (default 16), plus half as many at half the lip radius and one at the centre. |
| `seal.seal_tol` | Bellows compliance: how far the lip contact may be from a plane and still seal. |
| `seal.reach` | Bellows travel: how far from the lip plane the surface may be. |
| `seal.max_normal_deg` | Steepest surface tilt under a cup. |
| `seal.min_cups` | Cups that must seal. Default: all of a 1–2 cup head, otherwise all but one. If all cups share one vacuum source, an open cup leaks: use `channels`, or set `min_cups` to the number of cups. |
| `channels` | Optional: cups that share a vacuum source, as lists of indices into `cups`. A channel holds only if all its cups seal; its cups don't count otherwise. A cup not listed has its own source (the default for every cup). |
| `hold.vacuum_kpa`, `hold.safety_g` | Each cup pulls at most vacuum × lip area; the part's weight × `safety_g` must be held. |
| `head` | Everything except the cups, as boxes, cylinders and capsules (axes along tool z) or meshes. Include whatever moves rigidly with the tool, such as brackets, cameras, the adapter and the wrist. Shape details below. |
| `turn_range_deg` | Head turns about the approach axis are sampled over this range. Use 360 unless the whole head looks the same after a smaller turn. |

Head shapes:
- **Box:** `lo` and `hi` are opposite corners.
- **Cylinder:** `height` is its full length.
- **Capsule:** `height` is the distance between the centres of its two end spheres, so the total length is `height + 2 radius`.
- **Mesh:** `file` is relative to the gripper JSON. It also takes `scale`, applied first, then `transform`: a 4 × 4 matrix into the tool frame, with its translation in metres.

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

   Cups on a shared vacuum channel hold only if all of them seal. At least `min_cups` cups must hold.
5. **Collision** (python-fcl, exact mesh). The head (not the cups) must not touch the part at the grasp, and nothing may touch it along a straight 100 mm approach.
6. **Hold** (simplified Dex-Net 3.0 wrench resistance [[1]](#references)).
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
| `test_seal.py` | Holes, edges, nuts, tilts, cylinders, back faces, mixed cup sizes, vacuum channels, hold and peel | 23/23 |
| `test_collide.py` | Channels, a one-sided head, approach-only obstacles, support plane, scene-only checks | 12/12 |
| `test_pipeline.py` | Phase 1 + `check_db` + phase 2 on a synthetic part: plug-ins, local search, obstacles, an open mesh, a single-cup head | 12/12 |

## Limits

- **Cups:** round cups whose axes are along the tool's approach axis only; no oval or rectangular pads.
- **Head:** the head is rigid. Hoses, cables and bellows sideways are not modelled.
- **Hold check:** it assumes the gripper points down while carrying, and has no dynamic loads beyond `safety_g`.
- **Seal model:** geometric only (contact planarity and coverage), not airflow. `seal_tol` and `reach` should be measured for your cups. Rays sample each cup's face on two rings (the lip and half its radius) plus the centre, 25 per cup by default. A hole smaller than their spacing can be missed: `rim_points` makes the rings denser, but a hole between them can still be missed.
- **Phase 2 checks:** the part is only checked against the gripper and the given boxes. Neighbouring parts must be passed as boxes, and arm or robot collisions belong in `reach`.
- **Support plane:** the exact collision check stands in a 4 m square slab, centred under the part, for the support plane.

## References

### Papers

1. J. Mahler, M. Matl, X. Liu, A. Li, D. Gealy, K. Goldberg. *Dex-Net 3.0: Computing Robust Vacuum Suction Grasp Targets in Point Clouds using a New Analytic Model and Deep Learning.* ICRA 2018. [arXiv:1709.06670](https://arxiv.org/abs/1709.06670)
2. J. Mahler, F. T. Pokorny, B. Hou, M. Roderick, M. Laskey, M. Aubry, K. Kohlhoff, T. Kröger, J. Kuffner, K. Goldberg. *Dex-Net 1.0: A Cloud-Based Network of 3D Objects for Robust Grasp Planning Using a Multi-Armed Bandit Model with Correlated Rewards.* ICRA 2016. [doi:10.1109/ICRA.2016.7487342](https://doi.org/10.1109/ICRA.2016.7487342)
3. J. Pan, S. Chitta, D. Manocha. *FCL: A General Purpose Library for Collision and Proximity Queries.* ICRA 2012. [doi:10.1109/ICRA.2012.6225337](https://doi.org/10.1109/ICRA.2012.6225337)
4. I. Wald, S. Woop, C. Benthin, G. S. Johnson, M. Ernst. *Embree: A Kernel Framework for Efficient CPU Ray Tracing.* ACM Transactions on Graphics 33(4), 2014. [doi:10.1145/2601097.2601199](https://doi.org/10.1145/2601097.2601199)

### Software

| Package | Reference |
|---|---|
| trimesh | M. Dawson-Haggerty et al., [trimesh.org](https://trimesh.org) |
| python-fcl | Python bindings for FCL [3], [github.com/BerkeleyAutomation/python-fcl](https://github.com/BerkeleyAutomation/python-fcl) |
| embreex (optional) | Python bindings for Embree [4], [github.com/trimesh/embreex](https://github.com/trimesh/embreex) |
| NumPy | C. R. Harris et al., *Array programming with NumPy*, Nature 585, 2020 |
| SciPy | P. Virtanen et al., *SciPy 1.0*, Nature Methods 17, 2020 |
| Rtree, NetworkX | [Rtree](https://github.com/Toblerity/rtree) (libspatialindex); A. Hagberg, D. Schult, P. Swart, *NetworkX*, SciPy 2008 |
| Matplotlib | J. D. Hunter, *Matplotlib: A 2D Graphics Environment*, Computing in Science & Engineering 9(3), 2007 |
| PyBullet (example only) | E. Coumans, Y. Bai, *PyBullet*, [pybullet.org](https://pybullet.org) |
| scikit-learn (example only) | F. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, 2011 |

## Where each reference is used

### Ideas taken from the papers

| Source | Where | What was taken, and how this pipeline differs |
|---|---|---|
| Dex-Net 3.0 [1] | Phase 1, step 3 (`build_db.py`) | **Candidates:** points sampled on the object's surface, approaching along the surface normal. Here the points are area-weighted on the CAD mesh, and each is tried with several head turns, since a multi-cup head is not round. |
| Dex-Net 3.0 [1] | Phase 1, step 4 (`seal.py`) | **Seal check per contact:** a grasp is only considered if the cup can seal on the local surface. Dex-Net 3.0 models the cup as springs that deform onto the surface. This pipeline uses a simpler geometric rule: rays from the cup face must all hit the part, the hits must fit a plane within `seal_tol`, and the surface must face the cup. |
| Dex-Net 3.0 [1] | Phase 1, step 6 (`seal.py`, `hold_margin`) | **Wrench resistance:** can the sealed cups hold the part's weight. Simplified here to a static check with the gripper pointing down: a minimum-norm split of the weight over the sealed cups, against each cup's vacuum force. |
| Dex-Net 3.0 [1] | Phase 1, step 7 (`build_db.py`) | **Robustness:** grasp quality as the share of perturbed grasps that still succeed. Here only the tool pose is perturbed: 10 copies, ±2 mm, up to 3°. |
| Dex-Net 1.0 [2] | Phases 1 and 2 | **Grasp database:** grasps computed offline once per object from its 3D model and looked up at run time from the object's pose. Dex-Net 1.0 does this for a two-finger gripper over many objects; here it is one database per part and suction head, tagged by resting pose. |

The rest has no paper behind it and was developed for this pipeline:
- several cups of different sizes, `min_cups` and vacuum channels;
- the peel limit for one cup or cups in a line;
- resting-pose tags and the support-plane check;
- non-maximum suppression settings;
- the phase 2 plug-ins (`reach`, `order`) and the local-search fallback.

### Software

| Package | Where it is used |
|---|---|
| trimesh | Loading parts (`part.py`), resting poses (`trimesh.poses.compute_stable_poses`), head shapes (`gripper.py`), surface sampling (`build_db.py`), ray casting for the seal check (`seal.py`), the top views in the database pictures (`render.py`). |
| python-fcl (FCL [3]) | All collision checks against the exact part mesh, the support plane and obstacle boxes (`collide.py`). |
| embreex (Embree [4]) | Optional. When installed, trimesh casts the seal-check rays with Embree, about 100 times faster. |
| NumPy | Everywhere. |
| SciPy | Rotations and quaternions (`poses.py`). |
| Rtree, NetworkX | Not imported directly. trimesh uses Rtree to cast rays without Embree, and NetworkX for mesh repairs such as fixing face normals. |
| Matplotlib | Database pictures (`render.py`) and the example's pictures. |
| PyBullet | Example only (`examples/g2_evs08`): the robot model, IK for the `reach` plug-in, and the simulation validation. |
| scikit-learn | Example only: splits the part into convex pieces (`MiniBatchKMeans`) for the fast IK collision check in `g2_reach.py`. |
