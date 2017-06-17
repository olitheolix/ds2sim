import numpy as np


def compileCameraMatrix(right, up, pos):
    """Return serialised camera matrix, or None if `cmat` is invalid.

    """
    # Sanity check `cmat` and construct the forward vector from the right/up
    # vectors.
    try:
        # Unpack the right/up/pos vectors.
        cmat = np.vstack([right, up, pos]).astype(np.float32)
        assert cmat.shape == (3, 3)
        assert not any(np.isnan(cmat.flatten()))
        right, up, pos = cmat[0], cmat[1], cmat[2]

        # Ensure righ/up are unit vectors.
        assert (np.linalg.norm(right) - 1) < 1E-5, 'RIGHT is not a unit vector'
        assert (np.linalg.norm(up) - 1) < 1E-5, 'UP is not a unit vector'

        # Ensure right/up vectors are orthogonal.
        eps = np.amax(np.abs(right @ up))
        assert eps < 1E-5, 'Camera vectors not orthogonal'
    except (AssertionError, ValueError):
        return None

    # Compute forward vector and assemble the rotation matrix.
    forward = np.cross(right, up)
    rot = np.vstack([right, up, forward])

    ret = np.eye(4)
    ret[:3, :3] = rot
    ret[3, :3] = pos
    ret = ret.astype(np.float32)
    return ret.flatten('C').tobytes()


class FreeFlight:
    """ Free flight camera.

    Args:
        init_pos (Vec3): Initial camera position
    """
    def __init__(self, init_pos):
        self.init_pos = np.array(init_pos, np.float32)
        assert self.init_pos.shape == (3, )
        self.pos = self.init_pos
        self.reset()

    def reset(self):
        self.pos = self.init_pos
        self.Q = np.array([1, 0, 0, 0], np.float64)

        # Camera vector: camera points in -z direction initially.
        self.c_r = np.array([1, 0, 0], np.float64)
        self.c_u = np.array([0, 1, 0], np.float64)
        self.c_f = np.array([0, 0, 1], np.float64)

    def prodQuatVec(self, q, v):
        """Return vector that corresponds to `v` rotated by `q`"""
        t = 2 * np.cross(q[1:], v)
        return v + q[0] * t + np.cross(q[1:], t)

    def prodQuatQuat(self, q0, q1):
        """Return product of two Quaternions (q0 * q1)"""
        w0, x0, y0, z0 = q0
        w1, x1, y1, z1 = q1
        m = [
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            +x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            +x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
        ]
        return np.array(m, dtype=np.float64)

    def update(self, phi: float, theta: float, dx, dy, dz) -> None:
        """Update camera vectors."""
        # Compute the Quaternions to rotate around x (theta) and y (phi).a
        sp, cp = np.sin(phi / 2), np.cos(phi / 2)
        st, ct = np.sin(theta / 2), np.cos(theta / 2)
        q_phi = np.array([ct, st, 0, 0], np.float64)
        q_theta = np.array([cp, 0, sp, 0], np.float64)

        # Combine the phi/theta update, then update our camera Quaternion.
        q_rot = self.prodQuatQuat(q_phi, q_theta)
        self.Q = self.prodQuatQuat(q_rot, self.Q)

        # Compute the new camera vectors.
        self.c_r = self.prodQuatVec(self.Q, np.array([1, 0, 0], np.float64))
        self.c_u = self.prodQuatVec(self.Q, np.array([0, 1, 0], np.float64))
        self.c_f = self.prodQuatVec(self.Q, np.array([0, 0, 1], np.float64))

        # Compute new position based on latest camera vectors.
        self.pos += -dz * self.c_f + dx * self.c_r

    def getCameraVectors(self):
        """Return the right, up, forward, and position vector"""
        return self.c_r, self.c_u, self.c_f, self.pos
