__author__ = 'kingofspace0wzz'

import numpy as np
from scipy import linalg as la
import sys
sys.path.append('G:\\Github\\Matrix-Analysis')
from Matrix import householder
import Matrix.givens
# from numpy.linalg import matrix_rank as rank

# QR factorization via householder method
def qr_householder(A):

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

    m = A.shape[0]
    n = A.shape[1]
    for j in range(n-1):
        c, s = givens.givens(np.array([A[j, j], A[j+1, j]]))[0:2]
        G = givens.givens_matrix(c,s)
        A[j:j+2, j:n+1] = G.T.dot(A[j:j+2, j:n+1])

    return A

def qr_householder_block(A):

    m = A.shape[0]
    n = A.shape[1]

    return
