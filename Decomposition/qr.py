__author__ = 'kingofspace0wzz'

import numpy as np
from scipy import linalg as la
import sys
sys.path.append('g:/Github/Matrix-Analysis')
import Matrix.householder.householder_vector as house
import Matrix.householder.householder_beta as householder
# from numpy.linalg import matrix_rank as rank

# QR factorization via householder method
def qr_householder(A):

    m = A.shape[0]
    n = A.shape[1]
    for i in range(n):
        v, beta = house(A[i:, i].reshape((m-i+1, 1)))
        A[i:, i:] = householder(v, beta).dot(A[i:, i:])
        if i < m:
            A[i+1:, i] = v[1:]
    return A
