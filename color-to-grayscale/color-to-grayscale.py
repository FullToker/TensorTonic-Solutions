def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    # Write code here
    rows, cols  = len(image), len(image[0])
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = 0.299*image[i][j][0]+0.587*image[i][j][1]+0.114*image[i][j][2]
    return result