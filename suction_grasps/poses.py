"""Pose helpers. Poses are 4x4 homogeneous matrices in metres; quaternions are x, y, z, w."""
import numpy as np
from scipy.spatial.transform import Rotation


def pose(pos=(0, 0, 0), quat_xyzw=(0, 0, 0, 1)):
    """4x4 matrix from a position and an xyzw quaternion."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = pos
    return T


def pos_quat(T):
    """(position, xyzw quaternion) of a 4x4 matrix."""
    return T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3]).as_quat()


def inv(T):
    """Inverse of a rigid transform."""
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def rot_z(angle):
    """4x4 rotation about z (a turn of the head about the approach axis)."""
    c, s = np.cos(angle), np.sin(angle)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def tool_from_point(point, approach, turn=0.0):
    """Tool pose with its origin at `point`, +z along `approach`, turned by `turn` rad about z.
    The tool x axis starts as the part's x (or y if the approach is near x) projected onto the cup plane."""
    z = np.asarray(approach, float) / np.linalg.norm(approach)
    ref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = ref - z * (ref @ z)
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = point
    return T @ rot_z(turn)
