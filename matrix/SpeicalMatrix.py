__author__ = 'kingofspacewzz'

import numpy as np

from scipy import linalg as la

# Several special matries

# Nomral Matrix

# Symmetric Matrix(in real space)/ Hermite Matrix(in complex space)

# Orthgonal Matrix/ Unitary Matrix

# Hadamard Matrix

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

if __name__ == '__main__':
    testIS()
    test()
