"""Suction seal rule (per cup, ray casts on the exact part mesh) and the hold check, for any set of round cups.

Seal rule for one cup. Rays are cast along the approach (+z of the tool) at points of the cup face: the centre,
rim_points / 2 at half the lip radius and `rim_points` on the lip. They start RAY_BACK behind the lip plane.
The cup seals if:
  1. every ray hits the part within `reach` of the lip plane. A miss is a hole or an edge under the cup; a hit
     closer than -reach is material standing in front of the cup (a nut, a flange);
  2. every hit is on a face looking back at the cup (not the inside of material);
  3. the hits fit a plane within `seal_tol` (max residual): the bellows can follow small unevenness only;
  4. that plane's normal is within `max_normal_deg` of the cup axis.
A grasp is valid if at least `min_cups` cups seal (collisions are checked separately, collide.py).

Hold check (simplified Dex-Net 3.0 wrench resistance): the gripper points straight down, so the part's weight
x safety_g pulls along +z of the tool at the centre of mass. The sealed cups share that load (minimum-norm force
split that balances the force and both tipping moments); each cup can pull at most vacuum x its lip area. A
negative share is the cup pressing on the part, which the rim can take. Margin = 1 - largest pull / capacity.

Rays use trimesh's ray engine on the exact mesh - Embree when the `embreex` package is installed (~0.1 ms per grasp
of 100 rays), else trimesh's pure-Python engine (much slower; grasps are processed in chunks to bound memory).
"""
import numpy as np

RAY_BACK = 0.020           # m, rays start this far behind the lip plane
CHUNK = 5000               # grasps per ray-casting call (memory)


def cup_pattern(radius, rim_points):
    """(k, 2) xy offsets of the ray points of one cup face, relative to the cup centre."""
    k = rim_points // 2
    inner = [(0.5 * radius * np.cos(a), 0.5 * radius * np.sin(a)) for a in (np.arange(k) + 0.5) * 2 * np.pi / k]
    rim = [(radius * np.cos(a), radius * np.sin(a)) for a in np.arange(rim_points) * 2 * np.pi / rim_points]
    return np.array([(0.0, 0.0)] + inner + rim)


class SealChecker:
    def __init__(self, mesh, gripper):
        self.mesh, self.g = mesh, gripper
        s = gripper.seal
        # ray origins in the tool frame for all cups: (n_cups, k, 3), RAY_BACK behind each cup's lip
        self.local = np.array([[[c[0] + dx, c[1] + dy, c[2] - RAY_BACK] for dx, dy in cup_pattern(r, s["rim_points"])]
                               for c, r in zip(gripper.cups, gripper.cup_radii)])
        # plane fit z = a x + b y + c over one cup's ray points: the same matrix for every grasp -> precompute
        self.fit = [np.linalg.pinv(np.c_[l[:, :2], np.ones(len(l))]) for l in self.local]
        self.max_normal = np.cos(np.radians(s["max_normal_deg"]))

    def check(self, T_list):
        """T_list: (n, 4, 4) tool poses in the part frame. Returns dict of arrays:
        sealed (n, cups) bool, n_sealed (n,), valid (n,) bool, resid (n, cups) m, depth (n, cups, k) m."""
        T = np.asarray(T_list, float).reshape(-1, 4, 4)
        parts = [self._check(T[s:s + CHUNK]) for s in range(0, len(T), CHUNK)] or [self._check(T)]
        return {key: np.concatenate([q[key] for q in parts]) for key in parts[0]}

    def _check(self, T):
        n, nc, k = len(T), *self.local.shape[:2]
        s = self.g.seal
        R, t = T[:, :3, :3], T[:, :3, 3]
        o = (np.einsum("nij,ckj->ncki", R, self.local) + t[:, None, None]).reshape(-1, 3)   # origins, part frame
        d = np.broadcast_to(R[:, None, None, :, 2], (n, nc, k, 3)).reshape(-1, 3)          # all rays along tool +z
        depth = np.full(len(o), np.nan)          # hit position along the ray relative to the lip plane (+ = beyond)
        facing = np.zeros(len(o), bool)
        if len(o):
            loc, ir, tri = self.mesh.ray.intersects_location(o, d, multiple_hits=False)
            if len(ir):
                depth[ir] = np.einsum("ij,ij->i", loc - o[ir], d[ir]) - RAY_BACK
                facing[ir] = np.einsum("ij,ij->i", self.mesh.face_normals[tri], d[ir]) < 0
        depth, facing = depth.reshape(n, nc, k), facing.reshape(n, nc, k)
        all_hit = ((np.abs(depth) <= s["reach"]) & facing).all(-1)                     # rules 1 + 2 (NaN -> False)
        resid = np.full((n, nc), np.inf)
        normal_ok = np.zeros((n, nc), bool)
        for j in range(nc):
            # plane z = a x + b y + c through the hits, in the tool frame (rules 3 + 4)
            i = np.flatnonzero(all_hit[:, j])
            coef = depth[i, j] @ self.fit[j].T                                        # (m, 3)
            A = np.c_[self.local[j, :, :2], np.ones(k)]
            resid[i, j] = np.abs(coef @ A.T - depth[i, j]).max(-1) if len(i) else resid[i, j]
            normal_ok[i, j] = 1 / np.sqrt(1 + coef[:, 0] ** 2 + coef[:, 1] ** 2) >= self.max_normal
        sealed = all_hit & (resid <= s["seal_tol"]) & normal_ok
        n_sealed = sealed.sum(-1)
        return dict(sealed=sealed, n_sealed=n_sealed, valid=n_sealed >= s["min_cups"], resid=resid, depth=depth)


def hold_margin(T, sealed, com, mass, gripper):
    """Margin 0..1 that the sealed cups hold the part with the gripper pointing down (see module doc).
    T: tool pose in the part frame, sealed: (cups,) bool, com: centre of mass in the part frame (m), mass: kg."""
    if sealed.sum() < 1:
        return 0.0
    if not np.isfinite(mass):
        return 1.0                                          # mass unknown: hold check skipped (build_db warns)
    r = (np.linalg.inv(T) @ np.r_[com, 1.0])[:3]           # centre of mass in the tool frame
    W = mass * 9.81 * gripper.hold["safety_g"]
    c = gripper.cups[sealed]
    cap = gripper.cup_capacity()[sealed]
    # sum f = W ; sum x_i f_i = r_x W ; sum y_i f_i = r_y W   (cup pulls f_i along -z at the cup centres)
    A = np.vstack([np.ones(len(c)), c[:, 0], c[:, 1]])
    b = W * np.array([1.0, r[0], r[1]])
    f = np.linalg.pinv(A) @ b
    if np.allclose(A @ f, b, atol=1e-6):
        return float(np.clip(1 - (f / cap).max(), 0, 1))
    # one cup (or cups in a line): the cup forces alone cannot balance the tipping moment; the lips resist peeling
    # with up to capacity x radius / 2 each (simple peel model) - check the total pull and that moment
    radii = gripper.cup_radii[sealed]
    arm = np.hypot(r[0] - c[:, 0].mean(), r[1] - c[:, 1].mean())     # centre of mass off the cups' centroid
    return float(np.clip(1 - max(W / cap.sum(), W * arm / (cap * radii / 2).sum()), 0, 1))
