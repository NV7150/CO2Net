import numpy as np


def softmax(x):
    x_c = x - np.max(x)
    mask = x_c > -89
    f_x = np.zeros(x.shape)
    if len(f_x[mask]) > 1:
        f_x[mask] = np.exp(x_c[mask]) / np.sum(np.exp(x_c[mask]))
    else:
        f_x[mask] = 1

    return f_x
