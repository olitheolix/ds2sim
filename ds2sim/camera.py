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
