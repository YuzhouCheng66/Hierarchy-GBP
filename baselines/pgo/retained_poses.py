"""Retained AMM matrix conversion and first-pose gauge alignment."""
import numpy as np
from scipy.spatial.transform import Rotation


def amm_poses(path, initial, se2):
    n = len(initial)
    d = 2 if se2 else 3
    matrix = np.loadtxt(path)
    if matrix.shape != ((d+1)*n, d):
        raise ValueError('Unexpected AMM saved matrix shape')
    rotations = matrix[n:].reshape(n, d, d).transpose(0, 2, 1)
    if se2:
        return np.column_stack((matrix[:n], np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])))
    return np.column_stack((matrix[:n], Rotation.from_matrix(rotations).as_quat()))


def anchor_align(poses, target, se2):
    result = poses.copy()
    if se2:
        theta = target[0, 2] - poses[0, 2]
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c, -s], [s, c]])
        result[:, :2] = (poses[:, :2] - poses[0, :2]) @ rotation.T + target[0, :2]
        result[:, 2] = np.arctan2(np.sin(poses[:, 2] + theta), np.cos(poses[:, 2] + theta))
    else:
        rotation = Rotation.from_quat(target[0, 3:]) * Rotation.from_quat(poses[0, 3:]).inv()
        result[:, :3] = rotation.apply(poses[:, :3] - poses[0, :3]) + target[0, :3]
        result[:, 3:] = (rotation * Rotation.from_quat(poses[:, 3:])).as_quat()
    return result
