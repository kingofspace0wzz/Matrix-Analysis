__author__ = 'kingofspace0wzz'

import numpy as np
from scipy import linalg as la
from Decomposition import qr
from numpy.linalg import matrix_rank as rank
from Matrix import givens as gi

'''
    Least square problems with numerical methods


'''

# least square using QR (A must be full column rank)
def ls_qr(A, b):

    m = A.shape[0]
    n = A.shape[1]
    if rank(A) < n:
        raise Exception('Rank deficient')

    A = qr.qr_householder(A)
    for j in range(n):
        v = np.hstack((1, A[j+1:, j]))
        A[j+1:, j] = 0
        b[j:] = (np.eye(m - j + 1) - 2 * np.outer(v, v) / la.norm(v, 2)).dot(b[j:])

    x_ls = la.solve(A, b)

    return x_ls0


def ls_fast_givens(A, b):

    m = A.shape[0]
    n = A.shape[1]
    if rank(A) < n:
        raise Exception('Rank deficient')

    S = qr.qr_fast_givens(A)
    M^T = np.dot(S, la.inv(A))
    b = M^T.dot(b)
    x_ls = la.solve(S[:n, :n], b[:n])

    return x_ls
