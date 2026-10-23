"""Pictures for review. db_picture: the part seen from above in each resting pose (ray-cast shaded view) with
every database grasp usable in that pose as a dot at its tool centre, coloured by quality, and the best SHOW spots
as numbered cup outlines (filled circle = cup seals, open = cup off the part)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import poses as G

RES = 0.001                # m per pixel of the top view
SHOW = 5                   # grasps drawn with their cups per resting pose (best first, spread apart)


def top_view(m, T_rest):
    """Shaded top view of the part resting with T_rest (part -> rest frame).
    Returns (image (h, w) 0..1 with NaN = background, extent [x0, x1, y0, y1] in mm)."""
    v = (T_rest[:3, :3] @ m.vertices.T).T + T_rest[:3, 3]
    lo, hi = v.min(0) - 0.01, v.max(0) + 0.01
    xs, ys = np.arange(lo[0], hi[0], RES), np.arange(lo[1], hi[1], RES)
    X, Y = np.meshgrid(xs, ys)
    top = np.c_[X.ravel(), Y.ravel(), np.full(X.size, hi[2] + 0.05)]
    Ti = G.inv(T_rest)
    o = (Ti[:3, :3] @ top.T).T + Ti[:3, 3]                       # ray origins and direction in the part frame
    d = np.broadcast_to(Ti[:3, :3] @ [0, 0, -1.0], o.shape)
    shade = np.full(len(o), np.nan)
    _, ir, tri = m.ray.intersects_location(o, d, multiple_hits=False)
    n_up = (T_rest[:3, :3] @ m.face_normals[tri].T)[2]             # surface normal in the rest frame
    shade[ir] = 0.45 + 0.5 * np.abs(n_up)
    return shade.reshape(X.shape), [xs[0] * 1000, xs[-1] * 1000, ys[0] * 1000, ys[-1] * 1000]


def db_picture(npz_path, m, gripper, out=None):
    d = np.load(npz_path)
    name = os.path.splitext(os.path.basename(npz_path))[0]
    poses = d["T_rest"]
    spread = 2 * (np.linalg.norm(gripper.cups[:, :2], axis=1) + gripper.cup_radii).max()   # drawn spots this far apart
    fig, axes = plt.subplots(1, max(1, len(poses)), figsize=(6.2 * max(1, len(poses)), 5.2), squeeze=False)
    cmap = plt.get_cmap("Blues")
    q_all = d["quality"]
    qmax = max(q_all.max(), 1e-9) if len(q_all) else 1.0
    colour = lambda q: cmap(0.35 + 0.65 * q / qmax)
    for k, ax in enumerate(axes[0][:len(poses)]):
        img, ext = top_view(m, poses[k])
        ax.imshow(img, extent=ext, origin="lower", cmap="gray", vmin=0, vmax=1.15, interpolation="nearest")
        usable = np.flatnonzero(d["pose_ok"][:, k])                   # database is sorted best first
        centres = np.array([(poses[k] @ d["T"][i])[:3, 3] for i in usable]).reshape(-1, 3) * 1000
        if not len(usable):
            ax.text(0.5, 0.5, "no usable grasp", transform=ax.transAxes, ha="center", fontsize=12, color="#1d1d1b")
        ax.scatter(centres[::-1, 0], centres[::-1, 1], s=10, c=[colour(d["quality"][i]) for i in usable[::-1]],
                   edgecolors="white", linewidths=0.4, zorder=3)
        shown = []                                                     # best grasps, spread apart
        for i in usable:
            if len(shown) < SHOW and all(np.linalg.norm(d["T"][i][:3, 3] - d["T"][j][:3, 3]) >= spread for j in shown):
                shown.append(i)
        for rank, i in reversed(list(enumerate(shown))):                # draw the best last (on top)
            T = poses[k] @ d["T"][i]
            cups = (T[:3, :3] @ gripper.cups.T).T + T[:3, 3]
            col = colour(d["quality"][i])
            if len(cups) > 2:                                          # outline of the cup group
                order = np.argsort(np.arctan2(*(cups[:, :2] - cups[:, :2].mean(0)).T[::-1]))
                ring = np.vstack([cups[order], cups[order][:1]]) * 1000
                ax.plot(ring[:, 0], ring[:, 1], color="#1d1d1b", lw=1.0, zorder=4)
            for c, r, s in zip(cups, gripper.cup_radii, d["sealed"][i]):
                ax.add_patch(plt.Circle(c[:2] * 1000, r * 1000, fill=bool(s), fc=col, ec="#1d1d1b", lw=0.8, zorder=4))
            ax.text(*(T[:2, 3] * 1000), str(rank + 1), ha="center", va="center", fontsize=8, zorder=5,
                    color="#1d1d1b", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        ax.set_title(f"resting pose {k + 1} (probability {d['pose_prob'][k] * 100:.0f}%)\n"
                     f"{len(usable)} usable grasps (dots); best {len(shown)} spots numbered", fontsize=10, loc="left")
        ax.set_xlabel("mm (resting frame, seen from above)")
        ax.set_aspect("equal")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "q", [cmap(0.35), cmap(1.0)]), norm=plt.Normalize(0, qmax))
    fig.colorbar(sm, ax=axes[0].tolist(), shrink=0.7, label="quality = robustness x hold margin")
    fig.suptitle(f"{name}: suction grasp database (filled circle = cup seals, open = cup off)", x=0.02, y=0.98,
                 ha="left", fontsize=12)
    out = out or os.path.splitext(npz_path)[0] + "_grasps.png"
    fig.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return out
