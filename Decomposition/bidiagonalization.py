__author__ = 'kingofspace0wzz'

import numpy as np
from scipy import linalg as la
from Matrix import householder


def bidiagonalization_householder(A):

    m = A.shape[0]
    n = A.shape[1]
    if m < n:
        raise Exception("Rows are less then columns")

    for j in range(n):
        v, beta = householder.householder_vector(A[j:, j])
        A[j:, j:] = householder.householder(v).dot(A[j:, j:])
        A[j+1:, j] = v[1:m-j+2]
        if j <= n-2
            v, beta = householder.householder_vector(A[j, j+1:])
            A[j:, j+1:] = A[j:, j+1:].dot(householder.householder(v))
            A[j, j+2:] = v[1:n-j+1]

    return A
