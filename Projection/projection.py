__author__ = 'kingofspace0wzz'

import numpy as np
from scipy import linalg as la
from numpy.linalg import matrix_rank as rank

'''
    This file contains functions and methods that are able to compute projection matrix

    A 'Projection matrix' is a transformation that projects the original space onto its subspaces

    Consider a n-dimensional space Z, and its subspaces X and Y (z = x + y, 'must be Direct Sum'), then Pz = P(x + y) = Px is projects Z into its subspace X

'''


def projection(X, Y):

    rankX = rank(X)
    rankY = rank(Y)

    # rank, or dimension, or the original space
    rankO = rankX + rankY

    # check if two subspaces have the same shapes
    if X.shape != Y.shape:
        raise Exception('The two subspaces do not have the same shapes')

    # check if O is singular
    if la.det(np.hstack((X, Y))) == 0:
        raise Exception('X + Y is not the direct sum of the original space')


    if rankX < min(X.shape):
        raise Exception('subspace X is not of full rank')

    elif rankY < min(Y.shape):
        raise Exception('subspace Y is not of full rank')
    # X and Y are of full column rank
    elif rankX == X.shape[1] & rankY == Y.shape[1]:
        return np.hstack((X, np.zeros((X.shape[0], rankO - rankX)))).dot(la.inv(np.hstack((X, Y))))
    # X and Y are of full row rank
    elif rankX == X.shape[0] & rankY == Y.shape[0]:
        return np.vstack((X, np.zeros((rankO - rankX, X.shape[1])))).dot(la.inv(np.vstack(X, Y)))

# orthogonal projection matrix
def orthoProjection(X, n):

    # check if X is a subspace of the original space
    if rank(X) < n:
        P = X.dot(la.inv(X.conjugate().T.dot(X))).dot(X.conjugate().T)

        # return: orthogonal projection from O to X, orthogonal projection from O to Y(a subspace that is orthogonal to X)
        return P, np.eye(P.shape[0]) - P
    else:
        raise Exception('not a subspace')

def project(X, Y, a):

    P = projection(X, Y)
    return P.dot(a)

def orthoProject(X, a, n, subspace = ''):

    if subspace != '':
        P = orthoProjection(X,n)[1]
        return P.dot(a)
    else:
        P = orthoProjection(X, n)[0]
        return P.dot(a)

def test():

    X = np.array([[1],
                  [0]])
    Y = np.array([[1],
                  [1]])

    a = np.array([[3],
                  [1]])

    print('Projection: ', project(X, Y, a), '\n')

def orthoTest():

    X = np.array([[1],
                  [1]])

    a = np.array([[0],
                  [1]])
    print('Orthogoal projection: ', orthoProject(X, a, 2))

if __name__ == '__main__':
    test()
    orthoTest()
