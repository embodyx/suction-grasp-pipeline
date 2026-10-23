"""Validate the pipeline in g2_pybullet_sim (the G2 simulation; only this example uses it). Sim only, no robot.

The sim and this example both have an ik.py, so the stages run in separate processes (main() runs them in order):
  frames  (pipeline) CAD -> sim part frame per part (exact: both meshes come from the same conversion, same vertex
                     order -> rigid fit on the vertices) and which stable pose is the sim's "flat" pose
  scenes  (sim)      conveyor scenes (real cell: robot at the conveyor's near end, conveyor.py): FLAT (the sim's own
                     sample_scene, part lying as the sim always places it) and OTHER (a different stable pose of the
                     part, placed on the belt like scenes.py)
  select  (pipeline) suction_grasps Selector + the G2 reachability plug-in for every scene -> top K grasps
  run     (sim)      (1) database cross-check: every database grasp usable in the flat pose against the sim's own cup
                         rule (grasp_eval.SuctionEvaluator), and the sim's single model grasp vs the database
                     (2) the whole pick task (reach, grab, carry home) with the planned expert (motion_plan) on the
                         conveyor: the pipeline's grasps (top K, first that plans wins) and, on FLAT scenes, the sim's
                         model grasp (4 turns of the cup square) as the baseline
The sim's default suction head is the EVS08 (g2_pybullet_sim build_evs08.py, the URDF this project copies); in the
sim process parts use the sim's fine collision model (patched at runtime, no sim file edited).
Note: the sim's grab rule counts a cup only on a surface facing up within 45 deg (pick_env TOP_NORMAL_Z); the
pipeline allows approaches up to 60 deg from down.

Run (from the repository root): python examples/g2_evs08/validate_in_sim.py [--per-part 10] [--workers 3]
The sim checkout (with its own .venv) is G2_SIM, default: a g2_pybullet_sim folder next to the repository.
Writes logs/validate_in_sim.json, logs/validate_in_sim.png (and logs/sim_*.json between the stages).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.abspath(__file__))                       # this example folder
REPO = os.path.dirname(os.path.dirname(ROOT))
SIM = os.path.expanduser(os.environ.get("G2_SIM", os.path.join(os.path.dirname(REPO), "g2_pybullet_sim")))  # sim checkout
PIPE_PY = sys.executable                                                 # the pipeline's Python (this one)
SIM_PY = os.path.join(SIM, ".venv", "bin", "python")
LOGS = os.path.join(ROOT, "logs")
# the example's parts (= g2_parts.SMALL; not imported: the sim-side stages run in the sim's Python, which does not
# have the pipeline's packages)
PARTS = ["A5917204100", "A5917204200", "A5917204300", "A5917204400", "A5917304100", "A5917304200"]
K = 5                       # pipeline grasps tried per scene (first that plans wins)
EVAL_SEED = 90_000


def jpath(name):
    return os.path.join(LOGS, name)


# ====================================================================== pipeline-side stages
def stage_frames():
    import trimesh
    import g2_parts as GP
    out = {}
    for n in PARTS:
        cad = GP.load(n)
        sim = trimesh.load(os.path.join(SIM, "assets", "parts", n + "_visual.obj"), force="mesh", process=False)
        sim.merge_vertices()
        A, B = cad.vertices, sim.vertices
        ca, cb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, cb - R @ ca                       # sim = T @ cad
        err = np.linalg.norm(A @ R.T + T[:3, 3] - B, axis=1).max()
        d = np.load(os.path.join(ROOT, "db", n + ".npz"))
        up_cad = R.T @ [0, 0, 1]                                     # the sim part frame's up axis, in CAD
        downs = [Tr[:3, :3].T @ [0, 0, -1] for Tr in d["T_rest"]]
        flat = int(np.argmax([dn @ -up_cad for dn in downs]))
        out[n] = dict(T_sim_cad=T.tolist(), fit_error_mm=float(err * 1000), flat_pose=flat,
                      flat_match_deg=float(np.degrees(np.arccos(np.clip(downs[flat] @ -up_cad, -1, 1)))),
                      T_rest=d["T_rest"].tolist(), pose_prob=d["pose_prob"].tolist())
        print(n, f"fit {err * 1000:.4f} mm, sim flat = resting pose {flat + 1} ({out[n]['flat_match_deg']:.1f} deg)")
    json.dump(out, open(jpath("sim_frames.json"), "w"))


def stage_select():
    import g2_parts as GP
    from g2_reach import G2Reach
    from suction_grasps.select import Selector
    scenes = json.load(open(jpath("sim_scenes.json")))
    out = []
    part, sel, reach = None, None, None
    for i, sc in enumerate(scenes):
        if sc["part"] != part:                             # one part at a time (memory); scenes are grouped by part
            if reach is not None:
                reach.close()
            part = sc["part"]
            mesh = GP.load(part)
            sel, reach = Selector(GP.db_path(part), GP.GRIPPER, mesh), G2Reach(mesh, GP.GRIPPER, part)
        T = np.array(sc["T_base_cad"])
        reach.set_scene(T, sc["boxes"])
        res, info = sel.select(T, sc["support"], sc["boxes"], reach=reach, order=reach.order, k=K)
        out.append(dict(grasps=[dict(T=r["T_tool"].tolist(), source=r["source"], quality=r["quality"],
                                     tilt_deg=r["tilt_deg"]) for r in res], info=info))
        print(f"[{i + 1}/{len(scenes)}] {sc['part']} {sc['kind']}: {len(res)} grasps, {info['seconds']} s", flush=True)
    json.dump(out, open(jpath("sim_grasps.json"), "w"))


# ====================================================================== sim-side setup
def sim_imports():
    """Import the sim with fine part collisions (runtime only, no sim file edited). The sim's default suction head
    ("suck4") is the EVS08 since 2026-10-22, the same URDF this project copies (copy_robot.py)."""
    sys.path.insert(0, SIM)
    import parts as P
    lp = P.load_part
    P.load_part = lambda cid, name, color=(0.9, 0.5, 0.1, 1), visual=True, fine=False: lp(cid, name, color, visual, True)


def pose(pos, R):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, pos
    return T


def conveyor_boxes(C, pos, yaw):
    """The sim conveyor's obstacle boxes (conveyor.obstacle_boxes, frame C) in the robot frame, for the pipeline."""
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    quat = [0, 0, np.sin(yaw / 2), np.cos(yaw / 2)]
    return [dict(center=(np.asarray(pos) + Rz @ np.array(cc)).tolist(), size=list(sz), quat_xyzw=quat)
            for sz, cc in C.obstacle_boxes()]


def stage_scenes(per_part):
    sim_imports()
    import trimesh
    import conveyor as C
    from pick_env import G2PickEnv
    frames = json.load(open(jpath("sim_frames.json")))
    env = G2PickEnv(use_pool=False, normalize_obs=False, gripper="suction", surface="conveyor")
    out = []
    for n in PARTS:
        f = frames[n]
        T_sim_cad = np.array(f["T_sim_cad"])
        T_rest, prob = np.array(f["T_rest"]), np.array(f["pose_prob"])
        v_cad = (trimesh.load(os.path.join(SIM, "assets", "parts", n + "_visual.obj"), force="mesh").vertices
                 - T_sim_cad[:3, 3]) @ T_sim_cad[:3, :3]               # sim -> CAD
        rng = np.random.default_rng([EVAL_SEED, PARTS.index(n)])
        for j in range(2 * per_part):
            flat = j < per_part
            sc = env.sample_scene(rng, n)
            poses = env.scene_poses(sc)
            conv_pos, conv_yaw = poses["table"]
            if flat:                                                   # the sim's own placement
                T_world_sim = pose(*poses["part"])
                k = f["flat_pose"]
            else:                                                      # another stable pose, placed like scenes.py
                others = [i for i in range(len(prob)) if i != f["flat_pose"]]
                k = int(rng.choice(others, p=prob[others] / prob[others].sum()))
                while True:                                            # frame C: x along the belt, y across
                    yaw, across, along = rng.uniform(-np.pi, np.pi), rng.uniform(0, 1), rng.uniform(0, 1)
                    w = (T_rest[k][:3, :3] @ v_cad.T).T + T_rest[k][:3, 3]
                    ctr = (w[:, :2].min(0) + w[:, :2].max(0)) / 2
                    c, s = np.cos(yaw), np.sin(yaw)
                    xy = (w[::20, :2] - ctr) @ np.array([[c, -s], [s, c]]).T
                    lo = -C.BELT_WIDTH / 2 + C.RAIL_MARGIN - xy[:, 1].min()
                    hi = C.BELT_WIDTH / 2 - C.RAIL_MARGIN - xy[:, 1].max()
                    if hi >= lo:
                        break
                x = max(C.PICK_ZONE[0] + along * (C.PICK_ZONE[1] - C.PICK_ZONE[0]), C.RAIL_MARGIN - xy[:, 0].min())
                cy, sy = np.cos(conv_yaw), np.sin(conv_yaw)
                T_world_C = pose(conv_pos, np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]))
                Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                T_C_cad = pose([x, lo + across * (hi - lo), 0], Rz) @ pose([-ctr[0], -ctr[1], 0], np.eye(3)) @ T_rest[k]
                T_world_sim = T_world_C @ T_C_cad @ np.linalg.inv(T_sim_cad)
            out.append(dict(part=n, kind="flat" if flat else "other", pose_index=int(k), scene=sc,
                            part_sim=dict(pos=T_world_sim[:3, 3].tolist(), R=T_world_sim[:3, :3].tolist()),
                            T_base_cad=(T_world_sim @ T_sim_cad).tolist(),
                            support=dict(point=list(map(float, conv_pos)), normal=[0, 0, 1]),
                            boxes=conveyor_boxes(C, conv_pos, conv_yaw)))
    json.dump(out, open(jpath("sim_scenes.json"), "w"))
    print(f"{len(out)} scenes")


# ====================================================================== sim-side run
def run_part(part):
    sim_imports()
    import grasp_eval as GE
    import parts as P
    from motion_plan import PlannedExpert
    from pick_env import G2PickEnv
    frames = json.load(open(jpath("sim_frames.json")))[part]
    T_sim_cad = np.array(frames["T_sim_cad"])
    res = dict(part=part)

    # (1) database cross-check against the sim's cup rule, flat pose only (the evaluator puts the belt under the
    #     sim part frame); and the sim's model grasp vs the database
    d = np.load(os.path.join(ROOT, "db", part + ".npz"))
    flat = np.flatnonzero(d["pose_ok"][:, frames["flat_pose"]])
    ev = GE.SuctionEvaluator(part)
    ok, tilt = [], []
    for i in flat:
        Ts = T_sim_cad @ d["T"][i]
        ok.append(ev.try_grasp((np.zeros(3), np.eye(3)), (Ts[:3, 3], P.matrix_to_quat(Ts[:3, :3])))[0])
        tilt.append(np.degrees(np.arccos(np.clip(Ts[2, 2] * -1, -1, 1))))
    ok, tilt = np.array(ok), np.array(tilt)
    res["db_vs_sim_rule"] = dict(checked=len(ok), pass_all=int(ok.sum()), tilt_le_45=f"{int(ok[tilt <= 45].sum())}/{int((tilt <= 45).sum())}",
                                 tilt_gt_45=f"{int(ok[tilt > 45].sum())}/{int((tilt > 45).sum())}")
    g, n = np.array(P.META[part]["grasp"]), np.array(P.META[part]["grasp_normal"])
    T_cad_sim = np.linalg.inv(T_sim_cad)
    g_cad, n_cad = T_cad_sim[:3, :3] @ g + T_cad_sim[:3, 3], T_cad_sim[:3, :3] @ n
    near = np.argmin(np.linalg.norm(d["T"][:, :3, 3] - g_cad, axis=1))
    res["model_grasp_vs_db"] = dict(nearest_db_grasp_mm=float(np.linalg.norm(d["T"][near][:3, 3] - g_cad) * 1000),
                                    nearest_axis_deg=float(np.degrees(np.arccos(np.clip(d["T"][near][:3, 2] @ -n_cad / np.linalg.norm(n_cad), -1, 1)))),
                                    db_best_quality=float(d["quality"][0]))

    # (2) the whole task
    scenes = json.load(open(jpath("sim_scenes.json")))
    grasps = json.load(open(jpath("sim_grasps.json")))
    env = G2PickEnv(use_pool=False, normalize_obs=False, gripper="suction", surface="conveyor")
    orig_poses = env.scene_poses
    override = {}
    env.scene_poses = lambda sc: dict(orig_poses(sc), **override)    # part pose override for OTHER scenes

    def attempt(sc, part_pose, grasp):
        override.clear()
        if part_pose is not None:
            override["part"] = part_pose
        opts = dict(scene=sc) if grasp is None else dict(scene=sc, grasp=grasp)
        env.reset(options=opts)
        ex = PlannedExpert(env)
        return ex.ok, ("" if ex.ok else ex.stats.get("fail", "?").split(" (")[0])

    rows = []
    for sc, gr in zip(scenes, grasps):
        if sc["part"] != part:
            continue
        part_pose = None if sc["kind"] == "flat" else (np.array(sc["part_sim"]["pos"]), np.array(sc["part_sim"]["R"]))
        row = dict(kind=sc["kind"], pose_index=sc["pose_index"], pipeline_found=len(gr["grasps"]))
        t0 = time.time()
        row["pipeline"], row["pipeline_fail"], row["pipeline_try"] = False, "no grasp from the selector", 0
        for j, gg in enumerate(gr["grasps"]):
            T = np.array(gg["T"])
            ok_, why = attempt(sc["scene"], part_pose, (T[:3, 3], P.matrix_to_quat(T[:3, :3])))
            row["pipeline_try"] = j + 1
            if ok_:
                row["pipeline"], row["pipeline_fail"], row["pipeline_tilt"] = True, "", gg["tilt_deg"]
                break
            row["pipeline_fail"] = why
        row["pipeline_s"] = round(time.time() - t0, 1)
        if sc["kind"] == "flat":                                       # baseline: the sim's model grasp, 4 turns
            row["model"], row["model_fail"] = False, ""
            for spin in range(4):
                ok_, why = attempt(dict(sc["scene"], spin=spin), None, None)
                if ok_:
                    row["model"] = True
                    break
                row["model_fail"] = why
        rows.append(row)
        print(part, sc["kind"], {k_: v for k_, v in row.items() if k_ != "kind"}, flush=True)
    res["rows"] = rows
    return res


def stage_run(workers):
    from multiprocessing import Pool
    with Pool(workers) as pool:
        results = pool.map(run_part, PARTS)
    json.dump(results, open(jpath("validate_in_sim_raw.json"), "w"), indent=1)
    summary = {}
    for r in results:
        rows = r["rows"]
        fl, ot = [x for x in rows if x["kind"] == "flat"], [x for x in rows if x["kind"] == "other"]
        fails = {}
        for x in rows:
            if not x["pipeline"]:
                fails[x["pipeline_fail"]] = fails.get(x["pipeline_fail"], 0) + 1
        summary[r["part"]] = dict(db_vs_sim_rule=r["db_vs_sim_rule"], model_grasp_vs_db=r["model_grasp_vs_db"],
                                  flat_model=f"{sum(x['model'] for x in fl)}/{len(fl)}",
                                  flat_pipeline=f"{sum(x['pipeline'] for x in fl)}/{len(fl)}",
                                  other_pipeline=f"{sum(x['pipeline'] for x in ot)}/{len(ot)}",
                                  pipeline_failures=fails,
                                  mean_tries=float(np.mean([x["pipeline_try"] for x in rows if x["pipeline"]] or [0])))
        print(r["part"], json.dumps(summary[r["part"]]))
    json.dump(summary, open(jpath("validate_in_sim.json"), "w"), indent=1)
    chart(summary)


def chart(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frac = lambda s: 100 * int(s.split("/")[0]) / max(1, int(s.split("/")[1]))
    rows = [("Flat, sim model grasp (4 turns)", "flat_model", "#2a78d6"),
            ("Flat, pipeline grasps (top 5)", "flat_pipeline", "#eb6834"),
            ("Other resting poses, pipeline (no baseline)", "other_pipeline", "#1baf7a")]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(PARTS))
    h = 0.8 / len(rows)
    for i, (label, key, col) in enumerate(rows):
        vals = np.array([frac(summary[p_][key]) for p_ in PARTS])
        tot = sum(int(summary[p_][key].split("/")[0]) for p_ in PARTS)
        n = sum(int(summary[p_][key].split("/")[1]) for p_ in PARTS)
        bars = ax.barh(y + (i - 1) * h, vals, height=h * 0.9, color=col, label=f"{label}  ({tot}/{n})")
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{v:.0f}", va="center", fontsize=7, color="#3d3d3a")
    ax.set_yticks(y, PARTS)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("scenes completed in the sim (%): reach, grab, carry home with the planned expert, EVS08 head")
    ax.grid(axis="x", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False, fontsize=8)
    ax.set_title("Suction grasp pipeline vs the sim's model grasp, random conveyor scenes", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(jpath("validate_in_sim.png"), dpi=90)


# ====================================================================== driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all", choices=["all", "frames", "scenes", "select", "run", "chart"])
    ap.add_argument("--per-part", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    if args.stage == "all":
        me = os.path.abspath(__file__)
        for stage, py in (("frames", PIPE_PY), ("scenes", SIM_PY), ("select", PIPE_PY), ("run", SIM_PY)):
            print(f"=== {stage}", flush=True)
            subprocess.run([py, me, stage, "--per-part", str(args.per_part), "--workers", str(args.workers)],
                           check=True, cwd=ROOT)
    elif args.stage == "frames":
        stage_frames()
    elif args.stage == "scenes":
        stage_scenes(args.per_part)
    elif args.stage == "select":
        stage_select()
    elif args.stage == "run":
        stage_run(args.workers)
    elif args.stage == "chart":
        chart(json.load(open(jpath("validate_in_sim.json"))))


if __name__ == "__main__":
    main()
