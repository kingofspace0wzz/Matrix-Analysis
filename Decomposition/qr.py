__author__ = 'kingofspace0wzz'

import numpy as np
from scipy import linalg as la
from Matrix import householder
import Matrix.givens
from numpy.linalg import matrix_rank as rank

# QR factorization via householder method
def qr_householder(A):
    '''
    QR factorization via householder method
    '''
    m = A.shape[0]
    n = A.shape[1]
    for i in range(n):
        v, beta = householder.householder_vector(A[i:, i].reshape((m-i+1, 1)))
        A[i:, i:] = householder.householder_beta(v, beta).dot(A[i:, i:])
        if i < m:
            A[i+1:, i] = v[1:]
    return A

# QR factorization via givens transformation
def qr_givens(A):
    '''
    QR factorization via givens transformation
    '''
    m = A.shape[0]
    n = A.shape[1]
    for j in range(n):
        for i in range(m-1, j, -1):
            c, s = givens.givens(np.array([A[i-1, j], A[i, j]]))[0:2]
            G = given.givens_matrix(c,s)
            A[i-1:i+1, j:n+1] = G.T.dot(A[i-1:i+1, j:n+1])

    return A

# QR factorization via hessenberg method with the help of givens transformation
def qr_hessenberg_givens(A):
    '''
    QR factorization via hessenberg method with the help of givens transformation
    '''
    m = A.shape[0]
    n = A.shape[1]
    for j in range(n-1):
        c, s = givens.givens(np.array([A[j, j], A[j+1, j]]))[0:2]
        G = givens.givens_matrix(c,s)
        A[j:j+2, j:n+1] = G.T.dot(A[j:j+2, j:n+1])

    return A

# QR factorization via fast givens transformation
def qr_fast_givens(A):
    '''
    QR factorization via fast givens transformation
    '''
    m = A.shape[0]
    n = A.shape[1]
    d = np.ones(m)
    for j in range(n):
        for i in range(m-1, j, -1):
            alpha, beta, ty = givens.fast_givens(A[i-1:i+1, j], d[i-1:i+1])
            if ty == 1:
                A[i-1:i+1, j:n+1] = np.array([[beta, 1],
                                              [1, alpha]]).dot(A[i-1:i+1, j:n+1])
            else:
                A[i-1:i+1, j:n+1] = np.array([[1, alpha],
                                              [beta, 1]]).dot(A[i-1:i+1, j:n+1])

    return A

# least square using QR (A must be full column rank)
def qr_ls(A, b):
    '''
    least square using QR (A must be full column rank)
    '''
    m = A.shape[0]
    n = A.shape[1]
    if rank(A) < n:
        raise Exception('Rank deficient')

    A = qr_householder(A)
    for j in range(n):
        v = np.hstack((1, A[j+1:, j]))
        A[j+1:, j] = 0
        b[j:] = (np.eye(m - j + 1) - 2 * np.outer(v, v) / la.norm(v, 2)).dot(b[j:])

    x_ls = la.solve(A[:n, :n], b[:n])

    return x_ls
