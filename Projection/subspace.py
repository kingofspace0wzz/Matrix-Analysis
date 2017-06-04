__author__ = 'kingofspace0wzz'

import projection as pro
import numpy as np
from scipy import linalg as la
from numpy.linalg import matrix_rank as rank


'''
    This file gives methods that analyze subspaces' relationships, those of which include:

    Distance between subspaces by different definitions

    Angles between subspaces (Minimal angle, Maximal angle, and Principal angles)

    Rotation measure: how close a matrix can be to another matrix by rotation

'''

# compute the distance/gap between subspaces
# @param1: X, subspace
# @param2: Y, subspace
# @param3: n, rank/dimension of the original space
def subspace_distance(X, Y, n, method = 'definition'):

    # ask if two subspaces have the same dimension/rank
    if rank(X) != rank(Y):
        return 1 # the gap/distance between any pair of subspaces with different dimensions is one
    # compute distance by its definition
    if method == 'definition':

        P1 = pro.orthoProjection(X, n)[0]
        P2 = pro.orthoProjection(Y, n)[0]
        # distance = ||P1 - P2||_2
        return la.norm(P1 - P2, 2)
    # compute distance by use of completement subspace
    else:

        # orthogonal projection onto Y's completement
        P2 = pro.orthoProjection(Y, n)[1]
        # find the completement of Y using orthogonal projection
        completement = P2.dot(np.eye(n))

        return la.norm(X.conjugate().T.dot(completement), 2)


# function rotationMeasure measures how close a matrix can be to another matrix by rotation
# @param1: Matrix A
# @param2: Matrix B
# find the orthogonal matrix Q that minimizes ||A - BQ||_2
def rotation_measure(A, B):

    # ask if the two matrics have the same shape
    if A.shape != B.shape:
        raise Exception('Two matrics do not have the same shape')

    # C=B^T * A
    C = B.conjugate().T.dot(A)

    U = la.svd(C)[0]
    Vh = la.svd(C)[2]

    # Q = UVh
    return U.dot(Vh)


# compute the minimal angle between subspaces
def min_angle(X, Y, n):

    P1 = pro.orthoProjection(X, n)[0]
    P2 = pro.orthoProjection(Y, n)[0]

    return np.arccos(la.norm(P1.dot(P2), 2))


# compute the maximal angle between subspaces with equal dimension
def max_angle(X, Y, n):

    return np.arcsin(subspace_distance(X, Y, n))


# compute the principal angles between two subspaces
# return: np.array of principal angles, orthogonal matrics U and V
def principal_angles(X, Y, n):

    QX, RX = la.qr(X)
    QY, RY = la.qr(Y)

    if X.shape[1] >= Y.shape[1]:

        C = QX.conjugate().T.dot(QY)
        M, cos, Nh = la.svd(C)

        U = QX.dot(M)
        V = QY.dot(Nh.conjugate().T)

        angles = np.arccos(cos)

        return angles, U, V

    else:

        C = QY.conjugate().T.dot(QX)
        M, cos, Nh = la.svd(C)

        U = QX.dot(M)
        V = QY.dot(Nh.conjugate().T)

        angles = np.arccos(cos)

        return angles, U, V

# Similarity between subspaces by Yamaguchi's definition
def similarity_Yama(X, Y, n):

    angles = principal_angles(X, Y, n)[0]

    return np.min(angles)

# Similarity between subspaces by Wolf & Shashua's definition
def similarity_Wolf(X, Y, n):

    cos = np.cos(principal_angles(X, Y, n)[0])

    similarity = 1
    for c in cos:
        similarity = similarity * np.square(c)

    return c


# distace between subspaces by Hausdorff's definition
def hausdorff_distance(X, Y, n):

    if rank(X) != X.shape[1] & rank(Y) != Y.shape[1]:
        raise Exception('Please provide subspaces with full COLUMN rank')

    inner = 0

    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            inner = inter + np.square(X[:, i].conjugate().T.dot(Y[:, j]))

    distance = np.sqrt(np.max(rank(X), rank(Y)) - inner)

    return distance

# distance with inner-product
def kernel_distance(X, Y, n):

    if rank(X) != X.shape[1] & rank(Y) != Y.shape[1]:
        raise Exception('Please provide subspaces with full COLUMN rank')

    inner = 0

    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            inter = inter + np.square(X[:, i].conjugate().T.dot(Y[:, j]))

    distance = np.sqrt(inner)

def test():

    A = np.array([[1,2],
                  [3,4],
                  [5,6],
                  [7,8]])

    B = np.array([[1.2,2.1],
                  [2.9,4.3],
                  [5.2,6.1],
                  [6.8,8.1]])

    print('Matrix Q is: ', rotation_measure(A, B), '\n')

if __name__ == '__main__':
    test()
