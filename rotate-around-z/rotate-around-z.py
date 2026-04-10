import numpy as np

def rotate_around_z(points, theta):
    """
    Rotate 3D point(s) around the Z-axis by angle theta (radians).
    """
    # Your code here
    points = np.asarray(points, dtype=float)
    ori_dim = points.ndim
    points = points.reshape(-1,3)
    rotate_mat = np.asarray([[np.cos(theta), -np.sin(theta),0], [np.sin(theta), np.cos(theta),0],[0,0,1]])
    res = points @ rotate_mat.T
    if ori_dim==1:
        return res.reshape(3)
    else:
        return res