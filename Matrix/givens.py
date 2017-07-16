__author__ = 'kingofspace0wzz'

import numpy as np
from scipy import linalg as la

# Givens rotation of rank 2
def givens(x, i=0, j=1):

    if x[j] == 0:
        c, s = 1, 0
    else:
        if abs(x[j]) > abs(x[i]):
            t = -x[i]/x[j]
            s = 1/np.sqrt(1 + t**2)
            c = s * t
        else:
            t = -x[j]/x[i]
            c = 1/np.sqrt(1 + t**2)
            s = c * t

    t1 = x[i]
    t2 = x[j]
    x[i] = c*t1 - s*t2
    x[j] = s*t1 + c*t2

    return c, s, x

# fast givens transformation
def fast_givens(x, d):

    if x[1] != 0:
        alpha = -x[0]/x[1]
        beta = -alpha * d[1]/d[0]
        r = -alpha * beta
        if r <= 1:
            ty = 1
            tal = d[0]
            d[0] = (1+r) * d[1]
            d[1] = (1+r) * tal
        else:
            ty = 2
            alpha, beta, r = 1/alpha, 1/beta, 1/r
            d[0] = (1+r) * d[0]
            d[1] = (1+r) * d[1]
    else:
        ty = 2
        alpha, beta = 0, 0

    return alpha, beta, ty



# apply givens rotation to matrix A
def givens_rotation(A, c, s, i, k):

    for j in range(A.shape[0]):
        t1 = A[j, i]
        t2 = A[j, k]
        A[j, i] = c*t1 - s*t2
        A[j, k] = s*t1 + c*t2


def givens_matrix(c, s):

    return np.array([[c, -s],
                     [s, c]])

def test():

    x = np.array([1, 2, 3, 4])
    print(givens(x, 1, 3)[2])

if __name__ == '__main__':
    test()
