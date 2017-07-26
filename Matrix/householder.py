__author__ = 'kingofspace0wzz'

import numpy as np
from scipy import linalg as la





''' Constrct a householder matrix/transformation
    v is the direction vector, not neccessary to be unit
    the function householder(v) would turn non-unit v into a unit vector
'''

'''Properties of Householder
    1, is hermitian
    2, is Unitary
    3, is involutory: H^2 = I
    4, has eigenvalues +1 and -1
    5, its determinant is -1 (product of eigenvalues)

'''
def householder(v):

    # check if vector v's 2-norm is 1
    # If not, make it unit
    if la.norm(v, 2) != 1:
        v = v/la.norm(v, 2)

    return np.eye(np.outer(v,v).shape[0]) - 2 * np.outer(v, v)

def householder_beta(v, beta):

    return np.eye(np.outer(v, v).shape[0]) - beta * np.outer(v, v)

# Given a vector x, return a unit vector v and a scalar beta that form a householder transformation which projects x onto basis e1
def householder_vector(x):

    sigma = x[1:].conjugate().T.dot(x[1:])
    if x.shape[0] == 1:
        v = np.vstack((1.0, x[1:]))
    else:
        v = np.hstack((1.0, x[1:]))

    if sigma == 0:
        beta = 0
        return v, beta
    else:
        miu = np.sqrt(x[0]**2 + sigma)

        if x[0] <= 0:
            v[0] = x[0] - miu

        else:

            v[0] = -sigma/(x[0]+miu)


        beta = 2 * v[0]**2 / (sigma + v[0]**2)
        v = v / v[0]

        return v, beta

# blocked form householder
def householder_block(v, beta):

    Y = v[:, 0]
    W = - beta[0] * v[:, 0]
    for i in range(1, v.shape[1]):
        z = - beta[i] * (np.eye(v.shape[0]) +np.dot(W.T, Y)).dot(v[:, i])
        W = np.hstack((W, z))
        Y = np.hstack((Y, v[:, i]))

    return np.eye(v.shape[0]) + np.dot(W.T, Y), W.T, Y.T


def test():

    x = np.array([[1],
                  [1],
                  [1]])
    v, beta = householder_vector(x)
    print(v, beta, '\n')
    H = householder_beta(v, beta)
    re = H.dot(x)
    print(re)
if __name__ == '__main__':
    test()
