
import numpy as np
import numpy.linalg as la

def qmul(q1, q2):
    """Quaternion multiplication
    """
    
    # unpack quaternion components
    x1, y1, z1, w1 = [q1[...,p] for p in range(4)]
    x2, y2, z2, w2 = [q2[...,p] for p in range(4)]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    maxdim = max(q1.ndim, q2.ndim)

    if maxdim == 1:
        return np.array([x, y, z, w], dtype=np.float32)

    else:
        return np.concatenate([p[...,np.newaxis] for p in [x, y, z, w]], axis=maxdim-1)
    

def qconjugate(q):
    """Quaternion conjugate
    """
    return np.array([-q[0], -q[1], -q[2], q[3]])


def qnormalize(q):
    """Return unit quaternion
    """
    return q / la.norm(q)


def qreciprocal(q):
    """Return reciprocal quaternion
    """
    norm = la.norm(q)
    return qconjugate(q) / (norm*norm)


def qrotate(x, q):
    """Rotate x by quaternion q
    
    x is an imaginary quaternion
    """
    
    xr = qmul(q, qmul(x, qreciprocal(q)))
    return xr


def q2axisAngle(q):
    """Quaternion to axis angle
    """
    
    theta = 2.0*np.arccos(q[3])
    scale = 1.0/np.sin(0.5*theta)
    axis = scale*q[:3]
    axis /= la.norm(axis)
    return theta, axis


def axisAngle2q(axis, theta):
    """Axis angle to quaternion
    """
    
    q = np.zeros(4, dtype=np.float64)
    q[:3] = axis*np.sin(0.5*theta)
    q[3] = np.cos(0.5*theta)
    return q


def q2Euler(q):
    """Quaternion to Yaw Pitch Roll
    
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles


    Returns
    -------
    v : ndarray.
        3-vector [roll, pitch, yaw].
    """
    
    x, y, z, w = q
    
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(2*(w*y - z*x))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    
    return np.array([roll, pitch, yaw], dtype=np.float64)


# def q2Euler2(q):
#     """Quaternion to Yaw Pitch Roll
    
#     http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/
#     """
    
#     x, y, z, w = q

#     roll = np.arctan2(2*(y*w - x*z), 1 - 2*(y*y - z*z))
#     pitch = np.arcsin(2*(x*y + z*w))
#     yaw = np.arctan2(2*(x*w - y*z), 1 - 2*(x*x + z*z))
    
#     return np.array([roll, pitch, yaw], dtype=np.float64)


def qdifference(q1, q2):
    """Computes finite difference between quaternions
    """

    # tangent space projection matrix
    qc = np.copy(q1[...,np.newaxis])
    T1 = np.eye(4) - np.dot(qc, qc.T)

    # difference between q2 and q1 projected on tangent space
    qdiff = np.dot(T1, q2 - q1)

    # angular velocity quaternion
    w = 2.0*qmul(qdiff, qconjugate(q1))

    return w
