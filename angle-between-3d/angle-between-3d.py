import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # Your code here
    normv1 =np.linalg.norm(v)
    normv2 = np.linalg.norm(w)
    if normv1<10**-10 or normv2 < 10**-10 :
        return np.nan
    v1_u = v / normv1
    v2_u = w / normv2
    
    # Compute the dot product
    dot_product = np.dot(v1_u, v2_u)
    
    # Clip the value to the valid range [-1.0, 1.0] due to floating point errors
    clipped_dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Calculate the angle in radians using arccos
    angle_radians = np.arccos(clipped_dot_product)
    return angle_radians