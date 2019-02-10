from math import sqrt
import numpy as np


def generate_twin_spiral_data(n_data=500, base_r=5,
    d_theta=2 * np.pi / 200, x_shift=2, dr=0.2):

    """
    Usage
    -----
        >>> X, y = generate_twin_spiral_data()
    """

    get_r = lambda i: base_r + i * dr

    X = np.zeros((2 * n_data, 2))
    y = np.zeros(2 * n_data, dtype=np.int)
    y[n_data:] = 1

    # class 0
    for i in range(n_data):
        idx = n_data - 1 - i
        r = get_r(i)
        theta = d_theta * i
        X[idx,0] = r * np.cos(theta)
        X[idx,1] = r * np.sin(theta) - x_shift

    # class 1
    for i in range(n_data):
        r = get_r(i)
        theta = d_theta * i + np.pi
        idx = n_data + i
        X[idx,0] = r * np.cos(theta)
        X[idx,1] = r * np.sin(theta) + x_shift

    return X, y