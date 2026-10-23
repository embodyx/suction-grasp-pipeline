"""Synthetic test parts built from boxes (metres). Each shape is a list of boxes (pose, size); as_mesh() joins them
into one trimesh object (used by both the seal and the collision checks)."""
import numpy as np
import trimesh

PLATE_T = 0.002            # m, plate thickness


def box(centre, size, R=None):
    """Box with its centre and size (m), optionally rotated by the 3x3 matrix R about its centre."""
    T = trimesh.transformations.translation_matrix(centre)
    if R is not None:
        T[:3, :3] = R
    return (T, np.asarray(size, float))


def plate(half=0.10, top=0.0):
    """Square plate, top face at z = top."""
    return [box([0, 0, top - PLATE_T / 2], [2 * half, 2 * half, PLATE_T])]


def plate_with_holes(holes, hole=0.016, half=0.10):
    """Plate with square holes of side `hole` centred at the given xy points: made of strips between the holes
    (the plate cut along x into strips at the hole x positions, the hole strips cut along y)."""
    xs = sorted({round(h[0] - hole / 2, 6) for h in holes} | {round(h[0] + hole / 2, 6) for h in holes} | {-half, half})
    out = []
    for x0, x1 in zip(xs[:-1], xs[1:]):
        in_hole = [h for h in holes if h[0] - hole / 2 <= x0 + 1e-9 and x1 <= h[0] + hole / 2 + 1e-9]
        ys = sorted({-half, half} | {round(h[1] + s * hole / 2, 6) for h in in_hole for s in (-1, 1)})
        for y0, y1 in zip(ys[:-1], ys[1:]):
            if any(h[1] - hole / 2 <= y0 + 1e-9 and y1 <= h[1] + hole / 2 + 1e-9 for h in in_hole):
                continue                              # this cell is a hole
            out.append(box([(x0 + x1) / 2, (y0 + y1) / 2, -PLATE_T / 2], [x1 - x0, y1 - y0, PLATE_T]))
    return out


def plate_edge(edge_x, half=0.10):
    """Plate from x = -half to x = edge_x."""
    return [box([(edge_x - half) / 2, 0, -PLATE_T / 2], [edge_x + half, 2 * half, PLATE_T])]


def with_nut(xy, size=0.021, height=0.007):
    """Plate with a nut-sized block standing on it."""
    return plate() + [box([xy[0], xy[1], height / 2], [size, size, height])]


def channel(gap, wall_h, half=0.10, wall_t=0.010):
    """Floor plate with two walls along x, inner faces at y = +-gap/2, height wall_h above the floor top."""
    return plate(half) + [box([0, s * (gap / 2 + wall_t / 2), wall_h / 2], [2 * half, wall_t, wall_h]) for s in (-1, 1)]


def overhang(height, half=0.10):
    """Plate with a bar across it (along y) at `height` above the floor, held up at the plate's ends."""
    return plate(half) + [box([0, 0, height + 0.005], [0.020, 2 * half, 0.010])]


def as_mesh(boxes, transform=None):
    m = trimesh.util.concatenate([trimesh.creation.box(extents=s, transform=T) for T, s in boxes])
    if transform is not None:
        m.apply_transform(transform)
    return m


def cylinder(radius, length=0.2):
    """Solid cylinder along x with its top line at z = 0 (720 facets: chord error < 1e-5 x radius)."""
    m = trimesh.creation.cylinder(radius=radius, height=length, sections=720)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    m.apply_translation([0, 0, -radius])
    return m


def ribbed_plate():
    """A small closed test part (metres): 200 x 150 x 3 mm plate with a 20 mm tall, 10 mm wide rib across it
    and a 30 mm square boss - flat areas to grasp, a rib and a boss the head must avoid."""
    return [box([0, 0, 0.0015], [0.20, 0.15, 0.003]), box([0.04, 0, 0.013], [0.010, 0.15, 0.020]),
            box([-0.06, 0.04, 0.008], [0.030, 0.030, 0.010])]
