__author__ = 'kingofspace0wzz'

import numpy as np
from scipy import linalg as la

# Several special matries

# Nomral Matrix

# Symmetric Matrix(in real space)/ Hermite Matrix(in complex space)

# Orthgonal Matrix/ Unitary Matrix

# Hadamard Matrix

# Simple Matrix

# Householder Matrix


def isNormal(A, method = 'definition'):

    # use Schur inequality to determine whether it's normal
    if method == 'Schur':
        # initialize eigenValue
        eigenValue = la.eig(A)[0]

        if abs(np.sum(eigenValue**2) - la.norm(A, 'fro')**2) < 0.00001:
            return True
        else:
            return False
    # use definition
    else:
        if abs((A.conjugate().T.dot(A) - A.dot(A.conjugate().T)).all()) < 0.00001:
            return True
        else:
            return False


def isSymmetric(A):

    if abs((A.T - A).any()) > 0.00001:
        return False
    else:
        return True

def isHermite(A):

    if abs((A.conjugate().T - A).any()) > 0.00001:
        return False
    else:
        return True

def isOrthogonal(A):

    if abs((A.T.dot(A) - np.eye(A.shape[0])).all()) < 0.00001:
        return True
    else:
        return False

def isUnitary(A):

    if abs((A.conjugate().T.dot(A) - np.eye(A.shape[0])).all()) < 0.00001:
        return True
    else:
        return False

def isSimple(A):

    # check if A is a squre matrix
    if A.shape[1] != A.shape[0]:
        return False

    eigenValues, eigenVectors = la.eig(A)


    while (eigenValues.shape[0] != 0):

        #dictValues.update({eigenValues[0]: 1})

        index = np.argwhere(abs(eigenValues - eigenValues[0]) < 0.00001)
        algebraicMulti = len(index)

        geometricMulti = eigenVectors[:, index].shape[1]

        if algebraicMulti != geometricMulti:
            return False

        #dictValues.update({eigenValues[0]: len})
        eigenValues = np.delete(eigenValues, index)

    # stack another spaces of eigenvalue and eigenvector

    return True


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

    return np.eye(np.outer(v,v.conjugate().T).shape[0]) - 2 * np.outer(v, v.T.conjugate().T)


# Given a vector x, return a unit vector v and a scalar beta that form a householder transformation which projects x onto basis e1
def householder_vector(x):

    dimensionX  = len(x)
    sigma = x[1:].conjugate().T.dot(x[1:])
    v = np.vstack((1, x[1:]))

    if sigma == 0:
        beta = 0
        return v, beta
    else:
        miu = np.sqrt(x[0]**2 / sigma)
        if x[0] <= 0:
            v[0] = x[0] - miu
        else:
            v[0] = - sigma / (x[0] + miu)
        beta = 2 * v[0]**2 / (sigma + v[0]**2)
        v = v / la.norm(v, 2)

        return v, beta

# a test fuction that asks whether a particular matrix is one of the special matries above
def testIS():

    # A is a symmetric matrix, and thereby is normal
    A = np.array([[1,4,6],
                  [4,5,7],
                  [6,7,9]])

    # B is a complex matrix
    B = np.array([[1,4+3j,6-2j],
                  [4-3j,5,7-1j],
                  [6+2j,7+1j,9]])

    # C is an orthogonal matrix
    C = np.array([[np.cos(30.),-np.sin(30.)],
                  [np.sin(30.),np.cos(30.)]])

    print('Matrix A: ', A)
    print('Is A a normal Matrix: ', isNormal(A))
    print('Use Schur inequality to determine its normality: ', isNormal(A, 'Schur'))
    print('Is A a symmetric Matrix: ', isSymmetric(A),'\n' )
    print('Matrix B: ', B)
    print('Is B a Hermite Matrix: ', isHermite(B),'\n')
    print('Matrix C: ', C)
    print('Is C an orthogonal Matrix: ', isOrthogonal(C), '\n')


def test():

    v = np.array([0.5, 1, 4, 0.8])
    H = householder(v)
    print('vector v: ', v)
    print('Householder based on v: ', H)
    print('Test the properties of Householder: ')
    print('is it Unitary: ', isUnitary(H))
    print('is it hermitian: ', isHermite(H))

    E = np.array([[4,6,0],
                  [-3,-5,0],
                  [-3,-6,1]])


if __name__ == '__main__':
    testIS()
    test()
