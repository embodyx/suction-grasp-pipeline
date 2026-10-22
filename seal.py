"""Suction seal rule (per cup, ray casts on the exact part mesh) and the hold check.

Seal rule for one cup. Rays are cast along the approach (+z of the tool) at points of the cup face:
the centre, rim_points / 2 at half the radius and `rim_points` on the lip. They start RAY_BACK behind the tip
plane. The cup seals if:
  1. every ray hits the part within `cup_reach` of the tip plane. A miss is a hole or an edge under the cup;
     a hit closer than -cup_reach is material standing in front of the cup (a nut, a flange);
  2. the hits fit a plane within `seal_tol` (max residual): the bellows can follow small unevenness only;
  3. that plane's normal is within `max_normal_deg` of the cup axis.
A grasp is valid if at least `min_cups` cups seal AND the collision check passes (collide.py). That check also
catches a ray that starts inside material (whose first hit is then a back face): during the approach the cups
(12 mm lip radius) pass through every ray origin, so such a grasp is rejected there.

Rays use PyBullet's batched ray test against the exact part mesh (static triangle mesh with a BVH): about
0.2 ms per grasp. (trimesh's ray caster was ~16 ms per grasp and its memory grew with the batch.)

Hold check (simplified Dex-Net 3.0 wrench resistance): the gripper points straight down, so the part's weight
x safety_g pulls along +z of the tool at the centre of mass. The sealed cups share that load (minimum-norm
force split that balances the force and both tipping moments); each cup can pull at most vacuum x cup area.
A negative share is the cup pressing on the part, which the rim can take. Margin = 1 - largest pull / capacity.
"""
import numpy as np
import pybullet as p

RAY_BACK = 0.020           # m, rays start this far behind the tip plane
BATCH = 16000              # rays per rayTestBatch call (PyBullet's limit is 16384)
CHUNK = 5000               # grasps per check() chunk (memory)


def cup_pattern(g):
    """(k, 2) xy offsets of the ray points of one cup face, relative to the cup centre."""
    r = g["cup_radius"]
    k = g["rim_points"] // 2
    inner = [(0.5 * r * np.cos(a), 0.5 * r * np.sin(a)) for a in (np.arange(k) + 0.5) * 2 * np.pi / k]
    rim = [(r * np.cos(a), r * np.sin(a)) for a in np.arange(g["rim_points"]) * 2 * np.pi / g["rim_points"]]
    return np.array([(0.0, 0.0)] + inner + rim)


class SealChecker:
    def __init__(self, mesh, g):
        self.g = g
        # exact part mesh as a static triangle mesh in its own PyBullet world, body frame = part frame
        self.cid = p.connect(p.DIRECT)
        shape = p.createCollisionShape(p.GEOM_MESH, vertices=mesh.vertices.tolist(), indices=mesh.faces.ravel().tolist(),
                                       flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self.cid)
        p.createMultiBody(0, shape, physicsClientId=self.cid)
        pat = cup_pattern(g)
        # ray origins in the tool frame for all cups: (n_cups, k, 3)
        self.local = np.array([[[c[0] + dx, c[1] + dy, -RAY_BACK] for dx, dy in pat] for c in g["cups"]])
        # plane fit z = a x + b y + c over one cup's ray points: the same matrix for every grasp -> precompute
        self.fit = [np.linalg.pinv(np.c_[l[:, :2], np.ones(len(l))]) for l in self.local]
        self.max_normal = np.cos(np.radians(g["max_normal_deg"]))
        self.length = RAY_BACK + g["cup_reach"] + 0.001      # rays end just past the reach

    def cast(self, o, d):
        """Hit distance along each ray (NaN = no hit within self.length)."""
        dist = np.full(len(o), np.nan)
        end = o + d * self.length
        for s in range(0, len(o), BATCH):
            hits = p.rayTestBatch(o[s:s + BATCH].tolist(), end[s:s + BATCH].tolist(), physicsClientId=self.cid)
            f = np.array([h[2] if h[0] >= 0 else np.nan for h in hits])
            dist[s:s + BATCH] = f * self.length
        return dist

    def check(self, T_list):
        """T_list: (n, 4, 4) tool poses in the part frame. Returns dict of arrays:
        sealed (n, cups) bool, n_sealed (n,), valid (n,) bool, resid (n, cups) m, depth (n, cups, k) m.
        Works through CHUNK grasps at a time so the ray arrays stay small (~100 rays per grasp)."""
        T = np.asarray(T_list).reshape(-1, 4, 4)
        parts = [self._check(T[s:s + CHUNK]) for s in range(0, len(T), CHUNK)] or [self._check(T)]   # empty input too
        return {key: np.concatenate([q[key] for q in parts]) for key in parts[0]}

    def _check(self, T):
        n, nc, k = len(T), *self.local.shape[:2]
        R, t = T[:, :3, :3], T[:, :3, 3]
        o = np.einsum("nij,ckj->ncki", R, self.local) + t[:, None, None]     # origins in the part frame
        d = np.broadcast_to(R[:, None, None, :, 2], o.shape)                 # all rays along tool +z
        o, d = o.reshape(-1, 3), d.reshape(-1, 3)
        # hit position along the ray relative to the tip plane (+ = beyond the tips)
        depth = (self.cast(o, d) - RAY_BACK).reshape(n, nc, k)
        all_hit = (np.abs(depth) <= self.g["cup_reach"]).all(-1)            # rule 1 (NaN -> False)
        resid = np.full((n, nc), np.inf)
        normal_ok = np.zeros((n, nc), bool)
        for j in range(nc):
            # plane z = a x + b y + c through the hits, in the tool frame (rules 2 + 3)
            i = np.flatnonzero(all_hit[:, j])
            coef = depth[i, j] @ self.fit[j].T                               # (m, 3)
            A = np.c_[self.local[j, :, :2], np.ones(k)]
            resid[i, j] = np.abs(coef @ A.T - depth[i, j]).max(-1)
            normal_ok[i, j] = 1 / np.sqrt(1 + coef[:, 0] ** 2 + coef[:, 1] ** 2) >= self.max_normal
        sealed = all_hit & (resid <= self.g["seal_tol"]) & normal_ok
        n_sealed = sealed.sum(-1)
        return dict(sealed=sealed, n_sealed=n_sealed, valid=n_sealed >= self.g["min_cups"], resid=resid, depth=depth)


def hold_margin(T, sealed, com, mass, g):
    """Margin 0..1 that the sealed cups hold the part with the gripper pointing down (see module doc).
    T: tool pose in the part frame, sealed: (cups,) bool, com: centre of mass in the part frame (m), mass: kg."""
    if sealed.sum() < 1:
        return 0.0
    r = (np.linalg.inv(T) @ np.r_[com, 1.0])[:3]           # centre of mass in the tool frame
    W = mass * 9.81 * g["safety_g"]
    c = g["cups"][sealed]
    # sum f = W ; sum x_i f_i = r_x W ; sum y_i f_i = r_y W   (cup pulls f_i along -z at the cup centres)
    A = np.vstack([np.ones(len(c)), c[:, 0], c[:, 1]])
    f = np.linalg.pinv(A) @ (W * np.array([1.0, r[0], r[1]]))
    if not np.allclose(A @ f, W * np.array([1.0, r[0], r[1]]), atol=1e-6):
        return 0.0                                          # cups in a line cannot balance the tipping moment
    capacity = g["vacuum_kpa"] * 1000 * np.pi * g["cup_radius"] ** 2
    return float(np.clip(1 - f.max() / capacity, 0, 1))
