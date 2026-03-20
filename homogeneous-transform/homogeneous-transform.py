import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    points = np.asarray(points, dtype=float)
    # 把 (3,) 也变成(1, 3) 的(N, 3)格式
    single = points.ndim
    if points.ndim==1:
        points = points[np.newaxis, :]

    N= points.shape[0]
    dim4 = np.ones((N,1))
    points_4 = np.hstack((points, dim4))
    result = points_4 @ np.asarray(T).T
    result =result[:,:3]
    return result[0] if single==1 else result 